#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace streamingvlm::hybrid_bridge {

constexpr char kEmbeddingMagic[16] = {
    'S', 'V', 'L', 'M', '_', 'E', 'M', 'B', 'D', '_', 'F', '3', '2', '\0', '\0', '\0'};
constexpr uint32_t kEmbeddingVersion = 1;

struct EmbeddingHeader {
  char magic[16];
  uint32_t version;
  uint32_t dtype; // 1 == float32
  uint64_t n_dims;
  uint64_t n_values;
};

struct EmbeddingFile {
  std::vector<int64_t> shape;
  std::vector<float> values;
};

void write_embedding_file(
    const std::string& path,
    const std::vector<int64_t>& shape,
    const float* values,
    size_t n_values);

EmbeddingFile read_embedding_file(const std::string& path);

} // namespace streamingvlm::hybrid_bridge
