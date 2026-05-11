#include "file_sync.hpp"
#include "hybrid_embedding_file.h"
#include "phase_trace.hpp"
#include "vision_encoder_et.hpp"

#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <string>
#include <vector>

DEFINE_string(encoder_path, "encoder.pte", "ExecuTorch QNN vision encoder PTE.");
DEFINE_string(image_path, "frame_0000.bin", "Preprocessed CHW float32 image bin.");
DEFINE_string(image_paths, "", "Comma-separated preprocessed CHW float32 image bins, encoded in order.");
DEFINE_string(warmup_image_path, "", "Optional preprocessed CHW float32 image bin used only for QNN encoder warmup.");
DEFINE_string(group_sizes, "", "Comma-separated frame patch counts for metadata; inputs are still written one chunk per image path.");
DEFINE_string(output_path, "vision_embedding.svlmemb", "Output bridge embedding file.");
DEFINE_string(warmup_output_path, "", "Optional output embedding file for the fixed warmup image.");
DEFINE_string(stats_path, "vision_output_stats.csv", "Output stats CSV path.");
DEFINE_string(phase_stats_path, "vision_phase_stats.csv", "Output phase timing CSV path.");
DEFINE_string(ready_path, "", "Optional file to create after load/input preparation is complete.");
DEFINE_string(wait_path, "", "Optional file to wait for before starting QNN vision encode.");
DEFINE_int32(wait_timeout_ms, 120000, "Timeout while waiting for --wait_path.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const long origin_ms = executorch::extension::llm::time_in_ms();

  std::vector<std::string> image_paths = streamingvlm::hybrid_bridge::split_csv_paths(FLAGS_image_paths);
  if (image_paths.empty()) {
    image_paths.push_back(FLAGS_image_path);
  }
  const std::vector<size_t> group_sizes = streamingvlm::hybrid_bridge::split_csv_sizes(FLAGS_group_sizes);
  streamingvlm::hybrid_bridge::validate_group_sizes(group_sizes, image_paths.size());

  const auto result = streamingvlm::hybrid_bridge::encode_images_with_executorch(
      FLAGS_encoder_path,
      image_paths,
      FLAGS_warmup_image_path);

  streamingvlm::hybrid_bridge::write_text_file(FLAGS_ready_path, "ready\n");
  streamingvlm::hybrid_bridge::wait_for_file(
      FLAGS_wait_path,
      FLAGS_wait_timeout_ms,
      executorch::extension::llm::time_in_ms);

  const long write_start_ms = executorch::extension::llm::time_in_ms();
  if (!FLAGS_warmup_output_path.empty() && !result.warmup_values.empty()) {
    streamingvlm::hybrid_bridge::write_embedding_file(
        FLAGS_warmup_output_path,
        result.warmup_output_shape,
        result.warmup_values.data(),
        result.warmup_values.size());
  }
  streamingvlm::hybrid_bridge::write_embedding_file(
      FLAGS_output_path,
      result.output_shape,
      result.values.data(),
      result.values.size());
  const long write_end_ms = executorch::extension::llm::time_in_ms();
  streamingvlm::hybrid_bridge::write_vision_stats(
      FLAGS_stats_path,
      result.input_shape,
      result.output_shape,
      result.encode_total_ms,
      image_paths.size(),
      result.values.size());
  if (!FLAGS_phase_stats_path.empty()) {
    streamingvlm::hybrid_bridge::phase_recorder phases(
        FLAGS_phase_stats_path,
        origin_ms,
        streamingvlm::hybrid_bridge::vision_phase_description());
    phases.row("L_VisionLoad", result.load_start_ms, result.load_end_ms);
    phases.row("ImageLoad", result.image_load_start_ms, result.image_load_end_ms);
    for (const auto& range : result.encode_ranges) {
      phases.row("V_Encode", range.first, range.second);
    }
    phases.row("EmbeddingFileWrite", write_start_ms, write_end_ms);
  }
  ET_LOG(
      Info,
      "Wrote hybrid vision embedding: %s (%zu float32 values)",
      FLAGS_output_path.c_str(),
      result.values.size());
  return 0;
}
