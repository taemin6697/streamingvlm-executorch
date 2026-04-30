#include "hybrid_embedding_file.h"

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/utils.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <cinttypes>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

DEFINE_string(encoder_path, "encoder.pte", "ExecuTorch QNN vision encoder PTE.");
DEFINE_string(image_path, "frame_0000.bin", "Preprocessed CHW float32 image bin.");
DEFINE_string(output_path, "vision_embedding.svlmemb", "Output bridge embedding file.");
DEFINE_string(stats_path, "vision_output_stats.csv", "Output stats CSV path.");

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

void write_stats(
    const std::string& path,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    long encode_ms,
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
  out << "encode_ms," << encode_ms << "\n";
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  Module encoder_module(
      FLAGS_encoder_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  example::EncoderRunner encoder(&encoder_module);
  ET_CHECK_MSG(
      encoder.load() == executorch::runtime::Error::Ok,
      "Failed to load encoder module.");

  Result<MethodMeta> method_meta = encoder_module.method_meta("forward");
  ET_CHECK_MSG(method_meta.ok(), "Failed to read encoder method metadata.");
  auto input_meta = method_meta->input_tensor_meta(0);
  ET_CHECK_MSG(input_meta.ok(), "Failed to read encoder input metadata.");
  std::vector<int32_t> expected_size(
      input_meta->sizes().begin(), input_meta->sizes().end());
  std::vector<int64_t> input_shape(
      input_meta->sizes().begin(), input_meta->sizes().end());
  const ScalarType expected_dtype = input_meta->scalar_type();

  executorch::extension::llm::Image image;
  example::load_image(FLAGS_image_path, image, expected_size, expected_dtype);
  auto image_tensor_res = image.toTensor(/*with_batch=*/true);
  auto image_tensor_ptr = image_tensor_res.get();

  const long start_ms = executorch::extension::llm::time_in_ms();
  auto encode_res = encoder.encode(image_tensor_ptr);
  ET_CHECK_MSG(encode_res.ok(), "Encoder execution failed.");
  const long end_ms = executorch::extension::llm::time_in_ms();

  Tensor output = encode_res.get();
  ET_CHECK_MSG(
      output.scalar_type() == ScalarType::Float,
      "Hybrid bridge expects float32 encoder output.");
  const std::vector<int64_t> output_shape = tensor_shape(output);
  const size_t n_values = product(output_shape);

  streamingvlm::hybrid_bridge::write_embedding_file(
      FLAGS_output_path,
      output_shape,
      output.const_data_ptr<float>(),
      n_values);
  write_stats(
      FLAGS_stats_path,
      input_shape,
      output_shape,
      end_ms - start_ms,
      n_values);
  ET_LOG(
      Info,
      "Wrote hybrid vision embedding: %s (%zu float32 values)",
      FLAGS_output_path.c_str(),
      n_values);
  return 0;
}
