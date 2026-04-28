/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file xnnpack_qnn_runner.cpp
 *
 * Thin CLI entry point for the project-local foundation runner.
 * Backend-specific logic lives in xnnpack_backend.cpp and qnn_backend.cpp so
 * ExecuTorch/Qualcomm sources can stay as clean upstream dependencies.
 */

#include "backend.h"

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

DEFINE_string(backend, "qnn", "Backend: xnnpack, vulkan, or qnn.");

DEFINE_string(embedding_path, "embedding.pte", "Path to embedding model.");
DEFINE_string(encoder_path, "encoder.pte", "Path to vision encoder model.");
DEFINE_string(decoder_path, "decoder.pte", "Path to decoder model.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path.");

DEFINE_string(output_path, "outputs.txt", "Output file path.");
DEFINE_string(performance_output_path, "inference_speed.txt", "Inference speed log.");
DEFINE_string(dump_logits_path, "", "Dump logits path (KV mode only).");

DEFINE_string(decoder_model_version, "internvl3", "Decoder model version.");
DEFINE_string(prompt, "Describe this image:", "Text prompt.");
DEFINE_string(tokenized_prompt, "", "Alternative: tokenized prompt file.");
DEFINE_string(image_path, "", "Path to preprocessed image .bin or frame directory.");
DEFINE_string(
    decoder_input_mode,
    "embeddings",
    "Decoder input mode: embeddings or token_ids.");
DEFINE_string(system_prompt, "", "System prompt.");

DEFINE_double(temperature, 0.0f, "Sampling temperature.");
DEFINE_int32(seq_len, 128, "Max tokens to generate.");
DEFINE_int32(eval_mode, 1, "0=KV, 1=Hybrid, 2=Lookahead.");
DEFINE_bool(shared_buffer, false, "Use shared buffers where supported.");

DEFINE_int32(ngram, 0, "Lookahead ngram size.");
DEFINE_int32(window, 0, "Lookahead window.");
DEFINE_int32(gcap, 0, "Lookahead gcap.");
DEFINE_int32(num_iters, 1, "Number of iterations.");
DEFINE_bool(save_log, false, "Reserved for launcher compatibility.");
DEFINE_bool(vision_only, false, "Run only the vision encoder and dump its output.");

namespace {

std::vector<std::string> collect_prompts(int argc, char** argv) {
  std::vector<std::string> prompts;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
      prompts.push_back(argv[i + 1]);
      i++;
    }
  }
  return prompts;
}

std::string join_prompts(const std::vector<std::string>& prompts) {
  if (prompts.empty()) {
    return FLAGS_prompt;
  }
  std::string joined = prompts[0];
  for (size_t i = 1; i < prompts.size(); ++i) {
    joined += ";" + prompts[i];
  }
  return joined;
}

std::string normalize_frame_dir(const std::string& image_path) {
  namespace fs = std::filesystem;
  fs::path p(image_path);
  if (fs::is_directory(p)) {
    return p.string();
  }

  ET_CHECK_MSG(fs::exists(p), "Image path does not exist: %s", image_path.c_str());
  fs::path frame_dir = p.parent_path();
  fs::path expected = frame_dir / "frame_0000.bin";
  if (p == expected) {
    return frame_dir.empty() ? "." : frame_dir.string();
  }

  fs::path tmp = fs::temp_directory_path() / "xnnpack_qnn_runner_frame";
  fs::create_directories(tmp);
  fs::copy_file(
      p,
      tmp / "frame_0000.bin",
      fs::copy_options::overwrite_existing);
  return tmp.string();
}

int count_frames(const std::string& frame_dir) {
  namespace fs = std::filesystem;
  int count = 0;
  while (true) {
    char filename[32];
    std::snprintf(filename, sizeof(filename), "frame_%04d.bin", count);
    if (!fs::exists(fs::path(frame_dir) / filename)) {
      break;
    }
    count++;
  }
  return count > 0 ? count : 1;
}

} // namespace

int main(int argc, char** argv) {
  std::vector<std::string> prompts = collect_prompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!gflags::GetCommandLineFlagInfoOrDie("prompt").is_default &&
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default) {
    ET_CHECK_MSG(false, "Provide prompt or tokenized_prompt, not both.");
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("dump_logits_path").is_default &&
      FLAGS_eval_mode != 0) {
    ET_CHECK_MSG(false, "dump_logits only supported in KV mode.");
  }
  std::string frame_dir =
      FLAGS_image_path.empty() ? "" : normalize_frame_dir(FLAGS_image_path);

  executorch::examples::foundation::ManifestData manifest;
  manifest.backend = FLAGS_backend;
  manifest.model_family = "internvl3";
  manifest.variant = FLAGS_decoder_model_version;
  manifest.runner_type = "multimodal_split";
  manifest.decoder_input_mode = FLAGS_decoder_input_mode;
  manifest.paths.vision_encoder_pte = FLAGS_encoder_path;
  manifest.paths.text_embedding_pte = FLAGS_embedding_path;
  manifest.paths.text_decoder_pte = FLAGS_decoder_path;
  manifest.paths.tokenizer_path = FLAGS_tokenizer_path;

  executorch::examples::foundation::UnifiedRunConfig config;
  config.frame_dir = frame_dir;
  config.frame_count = frame_dir.empty() ? 0 : count_frames(frame_dir);
  config.questions = join_prompts(prompts);
  config.seq_len = FLAGS_seq_len;
  config.temperature = FLAGS_temperature;
  config.eval_mode = FLAGS_eval_mode;
  config.save_log = FLAGS_save_log;
  config.vision_only = FLAGS_vision_only;
  config.output_path = FLAGS_output_path;

  std::unique_ptr<executorch::examples::foundation::BackendRunner> runner;
  if (FLAGS_backend == "xnnpack" || FLAGS_backend == "vulkan") {
    runner = executorch::examples::foundation::create_xnnpack_backend_runner(manifest);
  } else if (FLAGS_backend == "qnn") {
    runner = executorch::examples::foundation::create_qnn_backend_runner(manifest);
  } else {
    ET_LOG(Error, "Unsupported backend: %s", FLAGS_backend.c_str());
    return 1;
  }

  ET_CHECK_MSG(runner != nullptr, "Failed to create backend runner.");
  auto err = runner->validate();
  ET_CHECK_MSG(err == executorch::runtime::Error::Ok, "Backend validation failed.");

  err = runner->run(config);
  ET_CHECK_MSG(
      err == executorch::runtime::Error::Ok,
      "Foundation run failed: %d",
      static_cast<int>(err));
  return 0;
}
