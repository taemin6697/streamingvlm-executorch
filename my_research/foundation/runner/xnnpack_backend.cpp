/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "backend.h"
#include "internal_memory_sampler.h"

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace executorch::examples::foundation {

namespace {

using ::executorch::extension::llm::load_tokenizer;
using ::executorch::extension::llm::logits_to_token;
using ::executorch::extension::llm::populate_start_pos_or_cache_position;
using ::executorch::extension::clone_tensor_ptr;
using ::executorch::extension::from_blob;
using ::executorch::extension::make_tensor_ptr;
using ::executorch::extension::Module;

constexpr int32_t kInternVL3ImageSeqLen = 256;
constexpr const char* kInternVL3ImageToken = "<IMG_CONTEXT>";
constexpr const char* kFoundationProcCsv = "foundation_proc.csv";
constexpr const char* kAndroidMemoryTimelineCsv = "android_memory_timeline.csv";

long rss_kb() {
  const size_t bytes = executorch::extension::llm::get_rss_bytes();
  return bytes > 0 ? static_cast<long>(bytes / 1024) : 0;
}

double elapsed_s(long timestamp_ms, long start_ms) {
  return (timestamp_ms - start_ms) / 1000.0;
}

void write_proc_header(std::ofstream& fproc) {
  fproc << "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
           "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
           "kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx\n";
  fproc << "# L: Loading  V_Encode: vision encoder  EmbeddingAndMerging: prompt/"
           "embedding merge  T_Prefill: decoder prefill  Decode: total decode  "
           "D: token decode\n";
}

void write_proc_row(
    std::ofstream* fproc,
    const char* row_type,
    long start_ms,
    long end_ms,
    long run_start_ms,
    long rss_start,
    long rss_end,
    long kv_pos = 0,
    long token_idx = 0) {
  if (fproc == nullptr || !fproc->is_open()) {
    return;
  }
  const long total_ms = end_ms - start_ms;
  *fproc << row_type << "," << elapsed_s(start_ms, run_start_ms) << ","
         << elapsed_s(end_ms, run_start_ms) << "," << rss_start << ","
         << rss_end << "," << total_ms << ",," << total_ms << "," << kv_pos
         << ",,,,,,"
         << token_idx << "\n";
  fproc->flush();
}

std::string frame_path(const std::string& dir, int idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "frame_%04d.bin", idx);
  return (std::filesystem::path(dir) / buf).string();
}

::executorch::extension::TensorPtr load_preprocessed_frame(
    const std::string& path,
    int32_t image_size,
    int32_t channels = 3) {
  std::ifstream input(path, std::ios::binary);
  ET_CHECK_MSG(input.is_open(), "Failed to open frame bin: %s", path.c_str());

  input.seekg(0, std::ios::end);
  const auto num_bytes = input.tellg();
  input.seekg(0, std::ios::beg);

  const size_t expected_floats =
      static_cast<size_t>(channels) * image_size * image_size;
  const size_t expected_bytes = expected_floats * sizeof(float);
  ET_CHECK_MSG(
      static_cast<size_t>(num_bytes) == expected_bytes,
      "Unexpected frame size for %s (expected %zu bytes, got %lld)",
      path.c_str(),
      expected_bytes,
      static_cast<long long>(num_bytes));

  std::vector<float> data(expected_floats);
  input.read(reinterpret_cast<char*>(data.data()), expected_bytes);
  ET_CHECK_MSG(input.good(), "Failed to read frame data: %s", path.c_str());
  return make_tensor_ptr(
      std::vector<executorch::aten::SizesType>{1, channels, image_size, image_size},
      std::move(data));
}

std::vector<std::string> split_questions(const std::string& questions) {
  std::vector<std::string> out;
  std::stringstream ss(questions);
  std::string token;
  while (std::getline(ss, token, ';')) {
    if (!token.empty()) {
      out.push_back(token);
    }
  }
  if (out.empty()) {
    out.push_back("Describe this image.");
  }
  return out;
}

std::string build_full_prompt_text(int32_t frame_count, const std::string& question) {
  std::string s = "<|im_start|>user:\n";
  for (int32_t idx = 0; idx < frame_count; ++idx) {
    const int32_t frame_num = idx + 1;
    s += "Frame" + std::to_string(frame_num) + ": <img>";
    for (int32_t i = 0; i < kInternVL3ImageSeqLen; ++i) {
      s += kInternVL3ImageToken;
    }
    s += "</img>\n";
  }
  s += question + "<|im_end|>\n<|im_start|>assistant\n";
  return s;
}

uint64_t placeholder_token_id(tokenizers::Tokenizer* tokenizer) {
  auto tk = tokenizer->encode(kInternVL3ImageToken, 0, 0);
  ET_CHECK_MSG(tk.ok(), "Failed to encode placeholder token");
  ET_CHECK_MSG(!tk->empty(), "Placeholder token encoding is empty");
  return tk->at(0);
}

std::unordered_set<uint64_t> stop_token_ids(tokenizers::Tokenizer* tokenizer) {
  std::unordered_set<uint64_t> ids;
  const auto eos = tokenizer->eos_tok();
  if (eos >= 0) {
    ids.insert(static_cast<uint64_t>(eos));
  }
  for (const char* token : {"<|im_end|>", "<|end_of_text|>"}) {
    auto encoded = tokenizer->encode(token, 0, 0);
    if (!encoded.ok()) {
      continue;
    }
    for (auto id : *encoded) {
      ids.insert(id);
    }
  }
  return ids;
}

bool contains_stop_marker(const std::string& piece) {
  return piece.find("<|im_end|>") != std::string::npos ||
      piece.find("<|end_of_text|>") != std::string::npos;
}

template <typename T>
void merge_image_features_typed(
    std::vector<T>& merged,
    const executorch::aten::Tensor& text_embeddings,
    const std::vector<executorch::aten::Tensor>& image_tensors,
    const std::vector<uint64_t>& input_ids,
    uint64_t image_placeholder_id) {
  const auto* text_ptr = text_embeddings.const_data_ptr<T>();
  const int64_t num_tokens = text_embeddings.size(1);
  const int64_t hidden_dim = text_embeddings.size(2);
  merged.assign(
      text_ptr,
      text_ptr + static_cast<size_t>(num_tokens * hidden_dim));

  std::vector<size_t> placeholder_positions;
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] == image_placeholder_id) {
      placeholder_positions.push_back(i);
    }
  }

  size_t image_token_offset = 0;
  for (const auto& image_tensor : image_tensors) {
    ET_CHECK_MSG(
        image_tensor.scalar_type() == text_embeddings.scalar_type(),
        "Image hidden state dtype must match text embedding dtype");
    const auto* image_ptr = image_tensor.const_data_ptr<T>();
    const int64_t image_seq_len = image_tensor.size(1);
    for (int64_t i = 0; i < image_seq_len; ++i) {
      const size_t pos = placeholder_positions.at(image_token_offset + i);
      std::memcpy(
          merged.data() + pos * hidden_dim,
          image_ptr + i * hidden_dim,
          static_cast<size_t>(hidden_dim) * sizeof(T));
    }
    image_token_offset += static_cast<size_t>(image_seq_len);
  }
}

::executorch::extension::TensorPtr build_merged_embeddings(
    const executorch::aten::Tensor& text_embeddings,
    const std::vector<executorch::aten::Tensor>& image_tensors,
    const std::vector<uint64_t>& input_ids,
    uint64_t image_placeholder_id) {
  const auto sizes = std::vector<executorch::aten::SizesType>{
      1,
      static_cast<executorch::aten::SizesType>(text_embeddings.size(1)),
      static_cast<executorch::aten::SizesType>(text_embeddings.size(2))};
  switch (text_embeddings.scalar_type()) {
    case executorch::aten::ScalarType::Float: {
      std::vector<float> merged;
      merge_image_features_typed<float>(
          merged, text_embeddings, image_tensors, input_ids, image_placeholder_id);
      return make_tensor_ptr(std::move(sizes), std::move(merged));
    }
    case executorch::aten::ScalarType::Half: {
      std::vector<executorch::aten::Half> merged;
      merge_image_features_typed<executorch::aten::Half>(
          merged, text_embeddings, image_tensors, input_ids, image_placeholder_id);
      return make_tensor_ptr(
          std::move(sizes),
          std::move(merged),
          {},
          {},
          executorch::aten::ScalarType::Half);
    }
    case executorch::aten::ScalarType::BFloat16: {
      std::vector<executorch::aten::BFloat16> merged;
      merge_image_features_typed<executorch::aten::BFloat16>(
          merged, text_embeddings, image_tensors, input_ids, image_placeholder_id);
      return make_tensor_ptr(
          std::move(sizes),
          std::move(merged),
          {},
          {},
          executorch::aten::ScalarType::BFloat16);
    }
    default:
      ET_CHECK_MSG(false, "Unsupported embedding dtype for XNNPACK split backend");
  }
}

executorch::runtime::Result<executorch::aten::Tensor> run_token_embedding(
    Module& embedding_module,
    const std::vector<uint64_t>& tokens) {
  auto token_tensor = from_blob(
      const_cast<uint64_t*>(tokens.data()),
      {1, static_cast<executorch::aten::SizesType>(tokens.size())},
      executorch::aten::ScalarType::Long);
  auto outputs = embedding_module.execute("forward", token_tensor);
  if (!outputs.ok()) {
    return outputs.error();
  }
  return outputs->at(0).toTensor();
}

executorch::runtime::Result<executorch::aten::Tensor> run_decoder_forward(
    Module& decoder_module,
    const executorch::runtime::EValue& embeddings,
    int64_t& start_pos,
    int seq_len) {
  std::vector<int64_t> cache_positions;
  auto cache_position_tensor =
      populate_start_pos_or_cache_position(&decoder_module, start_pos, cache_positions, seq_len);
  if (!cache_position_tensor.ok()) {
    return cache_position_tensor.error();
  }
  auto outputs =
      decoder_module.execute("forward", {embeddings, *cache_position_tensor.get()});
  if (!outputs.ok()) {
    return outputs.error();
  }
  return outputs->at(0).toTensor();
}

class XnnpackBackendRunner final : public BackendRunner {
 public:
  explicit XnnpackBackendRunner(ManifestData manifest)
      : manifest_(std::move(manifest)) {}

  executorch::runtime::Error validate() override {
    if (manifest_.paths.tokenizer_path.empty()) {
      ET_LOG(Error, "Missing tokenizer_path in manifest");
      return executorch::runtime::Error::InvalidArgument;
    }
    if (manifest_.paths.vision_encoder_pte.empty() ||
        manifest_.paths.text_embedding_pte.empty() ||
        manifest_.paths.text_decoder_pte.empty()) {
      ET_LOG(Error, "XNNPACK manifest requires split-PTE paths.");
      return executorch::runtime::Error::NotSupported;
    }
    return executorch::runtime::Error::Ok;
  }

  executorch::runtime::Error run(const UnifiedRunConfig& config) override {
    ET_CHECK_MSG(!config.frame_dir.empty(), "--frame_dir is required.");
    ET_CHECK_MSG(config.frame_count > 0, "--frame_count must be > 0.");

    const long t_run_start = executorch::extension::llm::time_in_ms();
    std::unique_ptr<std::ofstream> fproc;
    std::unique_ptr<InternalMemorySampler> memory_sampler;
    if (config.save_log) {
      fproc = std::make_unique<std::ofstream>(kFoundationProcCsv);
      ET_CHECK_MSG(
          fproc->is_open(),
          "Failed to open proc csv file: %s",
          kFoundationProcCsv);
      write_proc_header(*fproc);
      memory_sampler = std::make_unique<InternalMemorySampler>(
          kAndroidMemoryTimelineCsv, []() { return BackendMemoryMetrics{}; });
      memory_sampler->start();
    }

    const long t_load_start = executorch::extension::llm::time_in_ms();
    const long rss_load_start = rss_kb();
    auto tokenizer = load_tokenizer(manifest_.paths.tokenizer_path);
    ET_CHECK_MSG(
        tokenizer != nullptr,
        "Failed to load tokenizer: %s",
        manifest_.paths.tokenizer_path.c_str());
    const auto eos_token_id = static_cast<uint64_t>(tokenizer->eos_tok());
    const auto image_placeholder_id = placeholder_token_id(tokenizer.get());
    const auto stop_ids = stop_token_ids(tokenizer.get());

    Module vision_module(
        manifest_.paths.vision_encoder_pte,
        Module::LoadMode::MmapUseMlockIgnoreErrors);
    Module embedding_module(
        manifest_.paths.text_embedding_pte,
        Module::LoadMode::MmapUseMlockIgnoreErrors);
    ET_CHECK_OK_OR_RETURN_ERROR(vision_module.load_method("forward"));
    ET_CHECK_OK_OR_RETURN_ERROR(embedding_module.load_method("forward"));
    const long t_load_end = executorch::extension::llm::time_in_ms();
    const long rss_load_end = rss_kb();
    write_proc_row(
        fproc.get(),
        "L",
        t_load_start,
        t_load_end,
        t_run_start,
        rss_load_start,
        rss_load_end);

    std::vector<executorch::aten::Tensor> image_tensors;
    image_tensors.reserve(config.frame_count);
    for (int idx = 0; idx < config.frame_count; ++idx) {
      const long t_vision_start = executorch::extension::llm::time_in_ms();
      const long rss_vision_start = rss_kb();
      auto frame_tensor = load_preprocessed_frame(frame_path(config.frame_dir, idx), 448);
      // fp16 vision encoder expects Half; frame .bin is always float32
      auto frame_fp16 =
          clone_tensor_ptr(*frame_tensor, executorch::aten::ScalarType::Half);
      auto outputs = vision_module.execute("forward", frame_fp16);
      ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());
      image_tensors.push_back(outputs->at(0).toTensor());
      const long t_vision_end = executorch::extension::llm::time_in_ms();
      const long rss_vision_end = rss_kb();
      write_proc_row(
          fproc.get(),
          "V_Encode",
          t_vision_start,
          t_vision_end,
          t_run_start,
          rss_vision_start,
          rss_vision_end,
          /*kv_pos=*/0,
          /*token_idx=*/idx);
    }

    std::ostringstream final_output;
    for (const auto& question : split_questions(config.questions)) {
      std::string full_prompt = build_full_prompt_text(config.frame_count, question);
      auto encoded = tokenizer->encode(full_prompt, 0, 0);
      ET_CHECK_MSG(encoded.ok(), "Failed to encode prompt for question");
      std::vector<uint64_t> prompt_tokens = std::move(*encoded);
      final_output << full_prompt;

      const long t_merge_start = executorch::extension::llm::time_in_ms();
      const long rss_merge_start = rss_kb();
      auto text_embeddings_res = run_token_embedding(embedding_module, prompt_tokens);
      ET_CHECK_OK_OR_RETURN_ERROR(text_embeddings_res.error());
      auto text_embeddings = text_embeddings_res.get();

      auto merged_embeddings = build_merged_embeddings(
          text_embeddings, image_tensors, prompt_tokens, image_placeholder_id);
      int64_t start_pos = 0;
      const long t_merge_end = executorch::extension::llm::time_in_ms();
      const long rss_merge_end = rss_kb();
      write_proc_row(
          fproc.get(),
          "EmbeddingAndMerging",
          t_merge_start,
          t_merge_end,
          t_run_start,
          rss_merge_start,
          rss_merge_end,
          static_cast<long>(prompt_tokens.size()));

      Module decoder_module(
          manifest_.paths.text_decoder_pte,
          Module::LoadMode::MmapUseMlockIgnoreErrors);
      ET_CHECK_OK_OR_RETURN_ERROR(decoder_module.load_method("forward"));

      const long t_prefill_start = executorch::extension::llm::time_in_ms();
      const long rss_prefill_start = rss_kb();
      auto logits_res = run_decoder_forward(
          decoder_module,
          *merged_embeddings,
          start_pos,
          static_cast<int>(prompt_tokens.size()));
      ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
      auto logits = logits_res.get();
      start_pos += static_cast<int64_t>(prompt_tokens.size());
      const long t_prefill_end = executorch::extension::llm::time_in_ms();
      const long rss_prefill_end = rss_kb();
      write_proc_row(
          fproc.get(),
          "T_Prefill",
          t_prefill_start,
          t_prefill_end,
          t_run_start,
          rss_prefill_start,
          rss_prefill_end,
          static_cast<long>(prompt_tokens.size()));
      uint64_t cur_token =
          static_cast<uint64_t>(logits_to_token(logits, static_cast<float>(config.temperature)));
      uint64_t prev_token = cur_token;
      int64_t kv_pos = static_cast<int64_t>(prompt_tokens.size());
      std::ostringstream answer;
      const long t_decode_start = executorch::extension::llm::time_in_ms();
      const long rss_decode_start = rss_kb();

      for (int i = 0; i < config.seq_len; ++i) {
        auto decode_piece = tokenizer->decode(prev_token, cur_token);
        if (decode_piece.ok()) {
          answer << *decode_piece;
          if (contains_stop_marker(*decode_piece)) {
            break;
          }
        }
        if (cur_token == eos_token_id || stop_ids.count(cur_token) > 0) {
          break;
        }

        std::vector<uint64_t> next_token{cur_token};
        auto next_emb_res = run_token_embedding(embedding_module, next_token);
        ET_CHECK_OK_OR_RETURN_ERROR(next_emb_res.error());
        auto next_emb = next_emb_res.get();

        const long t_token_start = executorch::extension::llm::time_in_ms();
        const long rss_token_start = rss_kb();
        start_pos = kv_pos;
        auto next_logits_res =
            run_decoder_forward(decoder_module, next_emb, start_pos, 1);
        ET_CHECK_OK_OR_RETURN_ERROR(next_logits_res.error());
        auto next_logits = next_logits_res.get();
        const long t_token_end = executorch::extension::llm::time_in_ms();
        const long rss_token_end = rss_kb();
        prev_token = cur_token;
        cur_token = static_cast<uint64_t>(
            logits_to_token(next_logits, static_cast<float>(config.temperature)));
        kv_pos += 1;
        write_proc_row(
            fproc.get(),
            "D",
            t_token_start,
            t_token_end,
            t_run_start,
            rss_token_start,
            rss_token_end,
            static_cast<long>(kv_pos),
            i);
      }
      const long t_decode_end = executorch::extension::llm::time_in_ms();
      const long rss_decode_end = rss_kb();
      write_proc_row(
          fproc.get(),
          "Decode",
          t_decode_start,
          t_decode_end,
          t_run_start,
          rss_decode_start,
          rss_decode_end,
          static_cast<long>(kv_pos));

      final_output << answer.str();
      if (!question.empty()) {
        final_output << "\n";
      }
    }

    if (!config.output_path.empty()) {
      std::ofstream out(config.output_path);
      ET_CHECK_MSG(out.is_open(), "Failed to open output file: %s", config.output_path.c_str());
      out << final_output.str();
    } else {
      std::fwrite(final_output.str().data(), 1, final_output.str().size(), stdout);
    }
    if (memory_sampler) {
      memory_sampler->stop();
    }
    return executorch::runtime::Error::Ok;
  }

 private:
  ManifestData manifest_;
};

} // namespace

std::unique_ptr<BackendRunner> create_xnnpack_backend_runner(
    const ManifestData& manifest) {
  return std::make_unique<XnnpackBackendRunner>(manifest);
}

} // namespace executorch::examples::foundation
