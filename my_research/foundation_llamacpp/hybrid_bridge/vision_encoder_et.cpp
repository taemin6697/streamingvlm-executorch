#include "vision_encoder_et.hpp"

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/utils.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/assert.h>

#include <cinttypes>
#include <fstream>
#include <sstream>

namespace streamingvlm::hybrid_bridge {

namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

std::vector<int64_t> tensor_shape(const Tensor& tensor) {
  std::vector<int64_t> shape;
  for (auto dim : tensor.sizes()) {
    shape.push_back(static_cast<int64_t>(dim));
  }
  return shape;
}

size_t product(const std::vector<int64_t>& shape) {
  size_t n = 1;
  for (int64_t dim : shape) {
    ET_CHECK_MSG(dim > 0, "Invalid tensor shape dimension: %" PRId64, dim);
    n *= static_cast<size_t>(dim);
  }
  return n;
}

} // namespace

std::vector<std::string> split_csv_paths(const std::string& value) {
  std::vector<std::string> paths;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      paths.push_back(item);
    }
  }
  return paths;
}

std::vector<size_t> split_csv_sizes(const std::string& value) {
  std::vector<size_t> sizes;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      sizes.push_back(static_cast<size_t>(std::stoul(item)));
    }
  }
  return sizes;
}

void validate_group_sizes(const std::vector<size_t>& group_sizes, size_t n_inputs) {
  if (group_sizes.empty()) {
    return;
  }
  size_t grouped_inputs = 0;
  for (size_t count : group_sizes) {
    ET_CHECK_MSG(count > 0, "Invalid zero entry in --group_sizes.");
    grouped_inputs += count;
  }
  ET_CHECK_MSG(
      grouped_inputs == n_inputs,
      "--group_sizes total (%zu) must match --image_paths count (%zu).",
      grouped_inputs,
      n_inputs);
}

VisionEncodeResult encode_images_with_executorch(
    const std::string& encoder_path,
    const std::vector<std::string>& image_paths,
    const std::string& warmup_image_path) {
  VisionEncodeResult result;
  result.load_start_ms = executorch::extension::llm::time_in_ms();
  Module encoder_module(
      encoder_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  example::EncoderRunner encoder(&encoder_module);
  ET_CHECK_MSG(
      encoder.load() == executorch::runtime::Error::Ok,
      "Failed to load encoder module.");
  result.load_end_ms = executorch::extension::llm::time_in_ms();

  Result<MethodMeta> method_meta = encoder_module.method_meta("forward");
  ET_CHECK_MSG(method_meta.ok(), "Failed to read encoder method metadata.");
  auto input_meta = method_meta->input_tensor_meta(0);
  ET_CHECK_MSG(input_meta.ok(), "Failed to read encoder input metadata.");
  std::vector<int32_t> expected_size(
      input_meta->sizes().begin(), input_meta->sizes().end());
  result.input_shape.assign(input_meta->sizes().begin(), input_meta->sizes().end());
  const ScalarType expected_dtype = input_meta->scalar_type();

  result.image_load_start_ms = executorch::extension::llm::time_in_ms();
  std::vector<executorch::extension::llm::Image> images(image_paths.size());
  for (size_t i = 0; i < image_paths.size(); ++i) {
    example::load_image(image_paths[i], images[i], expected_size, expected_dtype);
  }
  executorch::extension::llm::Image warmup_image;
  bool has_warmup_image = !warmup_image_path.empty();
  if (has_warmup_image) {
    example::load_image(warmup_image_path, warmup_image, expected_size, expected_dtype);
  }
  result.image_load_end_ms = executorch::extension::llm::time_in_ms();

  if (has_warmup_image) {
    auto warmup_tensor_res = warmup_image.toTensor(/*with_batch=*/true);
    auto warmup_res = encoder.encode(warmup_tensor_res.get());
    ET_CHECK_MSG(warmup_res.ok(), "Encoder warmup execution failed for %s.", warmup_image_path.c_str());
    Tensor warmup_output = warmup_res.get();
    ET_CHECK_MSG(
        warmup_output.scalar_type() == ScalarType::Float,
        "Hybrid bridge expects float32 warmup encoder output.");
    result.warmup_output_shape = tensor_shape(warmup_output);
    const size_t warmup_values = product(result.warmup_output_shape);
    const float* warmup_output_data = warmup_output.const_data_ptr<float>();
    result.warmup_values.insert(
        result.warmup_values.end(),
        warmup_output_data,
        warmup_output_data + warmup_values);
  }

  std::vector<int64_t> per_input_shape;
  int64_t feature_dim = 0;
  int64_t tokens_per_input = 0;
  result.encode_ranges.reserve(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    auto image_tensor_res = images[i].toTensor(/*with_batch=*/true);
    auto image_tensor_ptr = image_tensor_res.get();
    const long start_ms = executorch::extension::llm::time_in_ms();
    auto encode_res = encoder.encode(image_tensor_ptr);
    ET_CHECK_MSG(encode_res.ok(), "Encoder execution failed for input %zu.", i);
    const long end_ms = executorch::extension::llm::time_in_ms();
    result.encode_total_ms += end_ms - start_ms;
    result.encode_ranges.emplace_back(start_ms, end_ms);

    Tensor output = encode_res.get();
    ET_CHECK_MSG(
        output.scalar_type() == ScalarType::Float,
        "Hybrid bridge expects float32 encoder output.");
    const std::vector<int64_t> output_shape = tensor_shape(output);
    ET_CHECK_MSG(
        output_shape.size() >= 2,
        "Hybrid bridge expects encoder output with token and feature dimensions.");
    const int64_t n_tokens = output_shape[output_shape.size() - 2];
    const int64_t n_feature = output_shape.back();
    ET_CHECK_MSG(n_tokens > 0 && n_feature > 0, "Invalid encoder output shape.");
    if (i == 0) {
      per_input_shape = output_shape;
      feature_dim = n_feature;
      tokens_per_input = n_tokens;
    } else {
      ET_CHECK_MSG(
          n_feature == feature_dim,
          "All encoder outputs must have the same feature dimension.");
      ET_CHECK_MSG(
          n_tokens == tokens_per_input,
          "All encoder outputs must have the same token count for the merged embedding file.");
    }
    const size_t n_values = product(output_shape);
    ET_CHECK_MSG(
        n_values == static_cast<size_t>(n_tokens * n_feature),
        "Hybrid bridge expected a single image embedding tensor shaped [..., tokens, features].");
    const float* output_data = output.const_data_ptr<float>();
    result.values.insert(result.values.end(), output_data, output_data + n_values);
  }

  if (image_paths.size() == 1) {
    result.output_shape = per_input_shape;
  } else {
    result.output_shape = {static_cast<int64_t>(image_paths.size()), tokens_per_input, feature_dim};
  }
  return result;
}

void write_vision_stats(
    const std::string& path,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    long encode_ms,
    size_t n_inputs,
    size_t n_values) {
  std::ofstream out(path);
  ET_CHECK_MSG(out.is_open(), "Failed to open stats CSV: %s", path.c_str());
  out << "metric,value\n";
  out << "input_dims,";
  for (size_t i = 0; i < input_shape.size(); ++i) {
    out << (i == 0 ? "" : "x") << input_shape[i];
  }
  out << "\noutput_dims,";
  for (size_t i = 0; i < output_shape.size(); ++i) {
    out << (i == 0 ? "" : "x") << output_shape[i];
  }
  out << "\noutput_values," << n_values << "\n";
  out << "input_count," << n_inputs << "\n";
  out << "encode_ms," << encode_ms << "\n";
}

} // namespace streamingvlm::hybrid_bridge

