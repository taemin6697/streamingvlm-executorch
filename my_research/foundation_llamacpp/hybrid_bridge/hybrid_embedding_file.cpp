#include "hybrid_embedding_file.h"

#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace streamingvlm::hybrid_bridge {

void write_embedding_file(
    const std::string& path,
    const std::vector<int64_t>& shape,
    const float* values,
    size_t n_values) {
  EmbeddingHeader header{};
  std::memcpy(header.magic, kEmbeddingMagic, sizeof(header.magic));
  header.version = kEmbeddingVersion;
  header.dtype = 1;
  header.n_dims = shape.size();
  header.n_values = n_values;

  const std::string tmp_path = path + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open embedding output: " + tmp_path);
  }
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  out.write(
      reinterpret_cast<const char*>(shape.data()),
      static_cast<std::streamsize>(shape.size() * sizeof(int64_t)));
  out.write(
      reinterpret_cast<const char*>(values),
      static_cast<std::streamsize>(n_values * sizeof(float)));
  if (!out.good()) {
    throw std::runtime_error("failed to write embedding output: " + path);
  }
  out.close();
  if (!out.good()) {
    throw std::runtime_error("failed to close embedding output: " + tmp_path);
  }
  if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
    throw std::runtime_error("failed to publish embedding output: " + path);
  }
}

EmbeddingFile read_embedding_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open embedding file: " + path);
  }

  EmbeddingHeader header{};
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!in.good() ||
      std::memcmp(header.magic, kEmbeddingMagic, sizeof(header.magic)) != 0 ||
      header.version != kEmbeddingVersion || header.dtype != 1) {
    throw std::runtime_error("invalid embedding file header: " + path);
  }

  EmbeddingFile file;
  file.shape.resize(static_cast<size_t>(header.n_dims));
  in.read(
      reinterpret_cast<char*>(file.shape.data()),
      static_cast<std::streamsize>(file.shape.size() * sizeof(int64_t)));
  file.values.resize(static_cast<size_t>(header.n_values));
  in.read(
      reinterpret_cast<char*>(file.values.data()),
      static_cast<std::streamsize>(file.values.size() * sizeof(float)));
  if (!in.good()) {
    throw std::runtime_error("truncated embedding file: " + path);
  }
  return file;
}

} // namespace streamingvlm::hybrid_bridge
