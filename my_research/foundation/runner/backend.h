/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>

#include <memory>
#include <string>

namespace executorch::examples::foundation {

struct ManifestPaths {
  std::string artifact_root;
  std::string vision_encoder_pte;
  std::string text_embedding_pte;
  std::string text_decoder_pte;
  std::string tokenizer_path;
  std::string combined_pte;
};

struct ManifestData {
  int schema_version{1};
  std::string backend;
  std::string model_family;
  std::string variant;
  std::string runner_type;
  ManifestPaths paths;
};

struct UnifiedRunConfig {
  std::string frame_dir;
  int frame_count{0};
  double fps{1.0};
  std::string questions;
  std::string query_timestamps;
  int seq_len{128};
  double temperature{0.0};
  int eval_mode{0};
  bool stream{false};
  bool save_log{false};
  std::string output_path;
};

class BackendRunner {
 public:
  virtual ~BackendRunner() = default;
  virtual executorch::runtime::Error validate() = 0;
  virtual executorch::runtime::Error run(const UnifiedRunConfig& config) = 0;
};

std::unique_ptr<BackendRunner> create_backend_runner(const ManifestData& manifest);
std::unique_ptr<BackendRunner> create_xnnpack_backend_runner(
    const ManifestData& manifest);
std::unique_ptr<BackendRunner> create_qnn_backend_runner(
    const ManifestData& manifest);

} // namespace executorch::examples::foundation
