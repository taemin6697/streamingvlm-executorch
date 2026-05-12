#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace streamingvlm::hybrid_bridge {

struct VisionEncodeResult {
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  std::vector<float> values;
  std::vector<int64_t> warmup_output_shape;
  std::vector<float> warmup_values;
  long load_start_ms = 0;
  long load_end_ms = 0;
  long image_load_start_ms = 0;
  long image_load_end_ms = 0;
  long encode_total_ms = 0;
  std::vector<std::pair<long, long>> encode_ranges;
};

std::vector<std::string> split_csv_paths(const std::string& value);
std::vector<size_t> split_csv_sizes(const std::string& value);

void validate_group_sizes(const std::vector<size_t>& group_sizes, size_t n_inputs);

class VisionEncoderSession {
 public:
  explicit VisionEncoderSession(const std::string& encoder_path);
  ~VisionEncoderSession();

  VisionEncoderSession(const VisionEncoderSession&) = delete;
  VisionEncoderSession& operator=(const VisionEncoderSession&) = delete;

  long load_start_ms() const;
  long load_end_ms() const;
  VisionEncodeResult encode(const std::vector<std::string>& image_paths);
  VisionEncodeResult encode_with_optional_warmup(
      const std::vector<std::string>& image_paths,
      const std::string& warmup_image_path = "");

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

VisionEncodeResult encode_images_with_executorch(
    const std::string& encoder_path,
    const std::vector<std::string>& image_paths,
    const std::string& warmup_image_path = "");

void write_vision_stats(
    const std::string& path,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    long encode_ms,
    size_t n_inputs,
    size_t n_values);

} // namespace streamingvlm::hybrid_bridge

