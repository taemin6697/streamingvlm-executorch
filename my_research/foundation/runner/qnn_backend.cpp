/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "backend.h"
#include "foundation_qnn_multimodal_runner.h"
#include "internal_memory_sampler.h"

#ifdef FOUNDATION_ENABLE_QNN
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/chat_template.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/utils.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/module/module.h>
#endif
#include <executorch/runtime/platform/log.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace executorch::examples::foundation {

namespace {

#ifdef FOUNDATION_ENABLE_QNN

constexpr const char* kFoundationProcCsv = "foundation_proc.csv";
constexpr const char* kAndroidMemoryTimelineCsv = "android_memory_timeline.csv";
constexpr int32_t kInternVL3ImageSeqLen = 256;

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
  fproc << "# L: Loading  V_Encode: QNN vision encoder  "
           "EmbeddingAndMerging: text embedding + multimodal merge  "
           "T_Prefill: QNN prompt prefill  Decode: QNN token generation  "
           "D: token callback timing\n";
}

void write_proc_row(
    std::ofstream* fproc,
    const char* row_type,
    long start_ms,
    long end_ms,
    long run_start_ms,
    long rss_start,
    long rss_end,
    long token_idx = 0) {
  if (fproc == nullptr || !fproc->is_open()) {
    return;
  }
  const long total_ms = end_ms - start_ms;
  *fproc << row_type << "," << elapsed_s(start_ms, run_start_ms) << ","
         << elapsed_s(end_ms, run_start_ms) << "," << rss_start << ","
         << rss_end << "," << total_ms << ",," << total_ms << ",,,,,,,"
         << token_idx << "\n";
  fproc->flush();
}

void write_phase_row(
    std::ofstream* fproc,
    const char* row_type,
    const QnnProfilePhaseTiming& timing,
    long run_start_ms,
    int64_t kv_pos = 0,
    int64_t kv_total = 0,
    int64_t kv_total_kb = 0,
    int64_t token_idx = 0) {
  if (fproc == nullptr || !fproc->is_open() || timing.start_ms <= 0 ||
      timing.end_ms < timing.start_ms) {
    return;
  }
  const long total_ms = timing.end_ms - timing.start_ms;
  const double kv_pct = kv_total > 0 ? 100.0 * kv_pos / kv_total : 0.0;
  const int64_t kv_used_kb =
      (kv_total > 0 && kv_total_kb > 0) ? (kv_pos * kv_total_kb) / kv_total : 0;
  *fproc << row_type << "," << elapsed_s(timing.start_ms, run_start_ms) << ","
         << elapsed_s(timing.end_ms, run_start_ms) << ","
         << timing.rss_kb_start << "," << timing.rss_kb_end << ","
         << total_ms << ",," << total_ms << ","
         << kv_pos << "," << kv_total << "," << kv_pct << ","
         << kv_used_kb << "," << kv_total_kb << ",,"
         << token_idx << "\n";
  fproc->flush();
}

void write_token_row(
    std::ofstream* fproc,
    const QnnProfileTokenTiming& timing,
    long run_start_ms,
    int64_t kv_total,
    int64_t kv_total_kb) {
  if (fproc == nullptr || !fproc->is_open()) {
    return;
  }
  const long total_ms = timing.end_ms - timing.start_ms;
  const double kv_pct =
      kv_total > 0 ? 100.0 * timing.kv_pos / kv_total : 0.0;
  const int64_t kv_used_kb = (kv_total > 0 && kv_total_kb > 0)
      ? (timing.kv_pos * kv_total_kb) / kv_total
      : 0;
  *fproc << "D," << elapsed_s(timing.start_ms, run_start_ms) << ","
         << elapsed_s(timing.end_ms, run_start_ms) << ","
         << timing.rss_kb << "," << timing.rss_kb << ",,"
         << total_ms << "," << total_ms << ","
         << timing.kv_pos << "," << kv_total << "," << kv_pct << ","
         << kv_used_kb << "," << kv_total_kb << ",,"
         << timing.token_idx << "\n";
}

std::string frame_path(const std::string& dir, int idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "frame_%04d.bin", idx);
  return (std::filesystem::path(dir) / buf).string();
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

std::string format_streaming_query(
    const std::string& question,
    const std::string& decoder_model_version) {
  if (decoder_model_version == "internvl3") {
    return question + "<|im_end|>\n<|im_start|>assistant\n";
  }
  return question;
}

std::string build_batch_prompt(
    const std::string& decoder_model_version,
    int num_frames,
    int64_t img_seq_len,
    const std::string& question) {
  if (decoder_model_version != "internvl3") {
    return question;
  }
  std::string s = "<|im_start|>user:\n";
  for (int f = 0; f < num_frames; ++f) {
    s += "Frame" + std::to_string(f + 1) + ": <img>";
    for (int64_t i = 0; i < img_seq_len; ++i) {
      s += "<IMG_CONTEXT>";
    }
    s += "</img>\n";
  }
  s += format_streaming_query(question, decoder_model_version);
  return s;
}

template <typename T>
executorch::runtime::Error run_batch_qnn(
    const ManifestData& manifest,
    const UnifiedRunConfig& config) {
  using ::executorch::extension::Module;
  using ::executorch::extension::llm::Image;
  using ::executorch::extension::llm::make_image_input;
  using ::executorch::extension::llm::make_text_input;
  using ::executorch::extension::llm::MultimodalInput;
  using ::executorch::runtime::MethodMeta;
  using ::executorch::runtime::Result;

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
  auto encoder_module = std::make_unique<Module>(
      manifest.paths.vision_encoder_pte,
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  auto embedding_module = std::make_unique<Module>(
      manifest.paths.text_embedding_pte,
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  auto decoder_module = std::make_unique<Module>(
      manifest.paths.text_decoder_pte,
      Module::LoadMode::MmapUseMlockIgnoreErrors);

  ProfiledQNNMultimodalRunner<T> runner(
      std::move(encoder_module),
      std::move(embedding_module),
      std::move(decoder_module),
      "internvl3",
      manifest.paths.tokenizer_path,
      /*performance_output_path=*/"",
      /*dump_logits_path=*/"",
      static_cast<float>(config.temperature),
      config.eval_mode,
      /*shared_buffer=*/false,
      /*ngram=*/0,
      /*window=*/0,
      /*gcap=*/0);

  ET_CHECK_OK_OR_RETURN_ERROR(runner.load());
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
  auto model_version = runner.get_model_version();
  ET_CHECK_OK_OR_RETURN_ERROR(model_version.error());
  Result<MethodMeta> method_meta = runner.get_encoder_method_meta();
  ET_CHECK_OK_OR_RETURN_ERROR(method_meta.error());
  auto input_meta_result = method_meta->input_tensor_meta(0);
  ET_CHECK_OK_OR_RETURN_ERROR(input_meta_result.error());
  std::vector<int32_t> expected_size(
      input_meta_result->sizes().begin(), input_meta_result->sizes().end());
  auto expected_dtype = input_meta_result->scalar_type();

  const bool force_generate = config.force_generate_token > 0;
  const int generation_len =
      force_generate ? config.force_generate_token : config.seq_len;
  executorch::extension::llm::GenerationConfig gen_config{
      /*echo=*/false,
      /*ignore_eos=*/force_generate,
      /*max_new_tokens=*/-1,
      /*warming=*/false,
      /*seq_len=*/generation_len,
      /*temperature=*/static_cast<float>(config.temperature),
      /*num_bos=*/0,
      /*num_eos=*/0};

  std::ofstream fout(config.output_path);
  ET_CHECK_MSG(
      fout.is_open(),
      "Failed to open output file: %s",
      config.output_path.c_str());

  std::vector<std::string> image_paths;
  image_paths.reserve(config.frame_count);
  for (int i = 0; i < config.frame_count; ++i) {
    image_paths.push_back(frame_path(config.frame_dir, i));
  }
  std::vector<std::string> audio_paths;
  auto prompts = split_questions(config.questions);
  const std::vector<std::string> output_prompts = prompts;
  auto messages = prepare_messages(prompts, image_paths, audio_paths);
  int token_idx = 0;
  for (size_t message_idx = 0; message_idx < messages.size(); ++message_idx) {
    const auto& message = messages[message_idx];
    std::vector<char> out_buf;
    auto cb = [&](const std::string& piece) {
      for (char c : piece) {
        out_buf.push_back(c);
      }
    };
    int64_t num_prompt_tokens = 0;
    int64_t num_generated_tokens = 0;
    auto stats_cb = [&](const executorch::extension::llm::Stats& stats) {
      num_prompt_tokens = stats.num_prompt_tokens;
      num_generated_tokens = stats.num_generated_tokens;
    };

    std::vector<MultimodalInput> inputs;
    for (const auto& file_path : message.files_path) {
      Image image;
      example::load_image(file_path, image, expected_size, expected_dtype);
      inputs.emplace_back(make_image_input(image));
    }
    std::string formatted_prompt =
        apply_chat_template(message.text, /*system_prompt=*/"", model_version.get());
    inputs.emplace_back(make_text_input(formatted_prompt));
    inputs = dispatch_inputs(inputs, formatted_prompt);
    const long t_generate_start = executorch::extension::llm::time_in_ms();
    const long rss_generate_start = rss_kb();
    ET_CHECK_OK_OR_RETURN_ERROR(runner.generate(inputs, gen_config, cb, stats_cb));
    const long t_generate_end = executorch::extension::llm::time_in_ms();
    const long rss_generate_end = rss_kb();
    const auto& timings = runner.last_generate_timings();
    const int64_t kv_total = runner.context_len();
    const int64_t kv_total_kb =
        static_cast<int64_t>(runner.kv_cache_total_bytes() / 1024);
    const int64_t kv_prefill = num_prompt_tokens;
    const int64_t kv_after = runner.cur_pos();

    write_phase_row(
        fproc.get(),
        "V_Encode",
        timings.vision_encode,
        t_run_start,
        /*kv_pos=*/0,
        kv_total,
        kv_total_kb,
        token_idx);
    QnnProfilePhaseTiming adjusted_embedding = timings.embedding_and_merging;
    if (timings.vision_encode.start_ms > 0 &&
        timings.vision_encode.end_ms > timings.vision_encode.start_ms &&
        adjusted_embedding.start_ms > 0 &&
        adjusted_embedding.end_ms > adjusted_embedding.start_ms) {
      const long embedding_ms =
          adjusted_embedding.end_ms - adjusted_embedding.start_ms;
      const long vision_ms =
          timings.vision_encode.end_ms - timings.vision_encode.start_ms;
      const long adjusted_ms = std::max<long>(0, embedding_ms - vision_ms);
      adjusted_embedding.start_ms = timings.vision_encode.end_ms;
      adjusted_embedding.end_ms = adjusted_embedding.start_ms + adjusted_ms;
      if (timings.prefill.start_ms > 0 &&
          adjusted_embedding.end_ms > timings.prefill.start_ms) {
        adjusted_embedding.end_ms = timings.prefill.start_ms;
      }
    }
    write_phase_row(
        fproc.get(),
        "EmbeddingAndMerging",
        adjusted_embedding,
        t_run_start,
        /*kv_pos=*/0,
        kv_total,
        kv_total_kb,
        token_idx);
    write_phase_row(
        fproc.get(),
        "T_Prefill",
        timings.prefill,
        t_run_start,
        kv_prefill,
        kv_total,
        kv_total_kb,
        token_idx);
    write_phase_row(
        fproc.get(),
        "Decode",
        timings.decode,
        t_run_start,
        kv_after,
        kv_total,
        kv_total_kb,
        token_idx);
    for (const auto& token_timing : timings.token_timings) {
      write_token_row(fproc.get(), token_timing, t_run_start, kv_total, kv_total_kb);
    }
    if (timings.decode.start_ms <= 0) {
      write_proc_row(
          fproc.get(),
          "Decode",
          t_generate_start,
          t_generate_end,
          t_run_start,
          rss_generate_start,
          rss_generate_end,
          token_idx);
    }
    (void)num_generated_tokens;
    token_idx++;
    const std::string output_prompt = build_batch_prompt(
        "internvl3",
        config.frame_count,
        kInternVL3ImageSeqLen,
        message_idx < output_prompts.size() ? output_prompts[message_idx]
                                            : config.questions);
    fout << output_prompt << std::string(out_buf.begin(), out_buf.end()) << "\n";
  }
  if (memory_sampler) {
    memory_sampler->stop();
  }
  return executorch::runtime::Error::Ok;
}

class QnnBackendRunner final : public BackendRunner {
 public:
  explicit QnnBackendRunner(ManifestData manifest) : manifest_(std::move(manifest)) {}

  executorch::runtime::Error validate() override {
    if (manifest_.paths.vision_encoder_pte.empty() ||
        manifest_.paths.text_embedding_pte.empty() ||
        manifest_.paths.text_decoder_pte.empty() ||
        manifest_.paths.tokenizer_path.empty()) {
      ET_LOG(Error, "QNN manifest is missing required split-PTE paths.");
      return executorch::runtime::Error::InvalidArgument;
    }
    return executorch::runtime::Error::Ok;
  }

  executorch::runtime::Error run(const UnifiedRunConfig& config) override {
    auto decoder_module = std::make_unique<executorch::extension::Module>(
        manifest_.paths.text_decoder_pte,
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    auto method_names = decoder_module->method_names();
    ET_CHECK_OK_OR_RETURN_ERROR(method_names.error());
    example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
    if (method_names->count("get_kv_io_bit_width") > 0) {
      auto kv_width_value = decoder_module->get("get_kv_io_bit_width");
      ET_CHECK_OK_OR_RETURN_ERROR(kv_width_value.error());
      kv_bitwidth = static_cast<example::KvBitWidth>(
          kv_width_value->toScalar().to<int64_t>());
    }

    if (kv_bitwidth == example::KvBitWidth::kWidth8) {
      return run_batch_qnn<uint8_t>(manifest_, config);
    }
    if (kv_bitwidth == example::KvBitWidth::kWidth16) {
      return run_batch_qnn<uint16_t>(manifest_, config);
    }
    ET_LOG(Error, "Unsupported QNN kv bit width");
    return executorch::runtime::Error::NotSupported;
  }

 private:
  ManifestData manifest_;
};

#else

class QnnBackendRunner final : public BackendRunner {
 public:
  explicit QnnBackendRunner(ManifestData manifest)
      : manifest_(std::move(manifest)) {}

  executorch::runtime::Error validate() override {
    ET_LOG(
        Error,
        "QNN backend support is not available in this xnnpack_qnn_runner build.");
    return executorch::runtime::Error::NotSupported;
  }

  executorch::runtime::Error run(const UnifiedRunConfig&) override {
    ET_LOG(
        Error,
        "QNN backend support is not available in this xnnpack_qnn_runner build.");
    return executorch::runtime::Error::NotSupported;
  }

 private:
  ManifestData manifest_;
};

#endif

} // namespace

std::unique_ptr<BackendRunner> create_qnn_backend_runner(
    const ManifestData& manifest) {
  return std::make_unique<QnnBackendRunner>(manifest);
}

} // namespace executorch::examples::foundation
