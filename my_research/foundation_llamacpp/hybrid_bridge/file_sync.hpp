#pragma once

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace streamingvlm::hybrid_bridge {

inline void write_text_file(const std::string& path, const std::string& value) {
  if (path.empty()) {
    return;
  }
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to write file: " + path);
  }
  out << value;
}

template <typename ClockFn>
inline void wait_for_file(const std::string& path, int timeout_ms, ClockFn time_ms) {
  if (path.empty()) {
    return;
  }
  const long start_ms = time_ms();
  while (true) {
    std::ifstream in(path);
    if (in.good()) {
      return;
    }
    if (time_ms() - start_ms > timeout_ms) {
      throw std::runtime_error("timed out waiting for file: " + path);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace streamingvlm::hybrid_bridge

