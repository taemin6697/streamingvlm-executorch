#pragma once

#include <executorch/extension/llm/runner/util.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace executorch::examples::foundation {

struct BackendMemoryMetrics {
  long kv_physical_committed_kb{0};
  long kv_total_kb{0};
};

class InternalMemorySampler {
 public:
  using Probe = std::function<BackendMemoryMetrics()>;

  InternalMemorySampler(std::string output_path, Probe probe, int interval_ms = 20)
      : output_path_(std::move(output_path)),
        probe_(std::move(probe)),
        interval_ms_(interval_ms) {}

  ~InternalMemorySampler() {
    stop();
  }

  void start() {
    if (running_.exchange(true)) {
      return;
    }
    out_.open(output_path_);
    if (!out_.is_open()) {
      running_.store(false);
      return;
    }
    write_header();
    started_at_ = std::chrono::steady_clock::now();
    append_sample_locked("runner_start", true);
    worker_ = std::thread([this]() {
      while (running_.load()) {
        append_sample("running", true);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
      }
    });
  }

  void stop() {
    if (!running_.exchange(false)) {
      if (worker_.joinable()) {
        worker_.join();
      }
      if (out_.is_open()) {
        out_.close();
      }
      return;
    }
    if (worker_.joinable()) {
      worker_.join();
    }
    append_sample("postrun", false);
    if (out_.is_open()) {
      out_.close();
    }
  }

  void append_sample(const char* phase, bool pid_alive) {
    std::lock_guard<std::mutex> lock(mu_);
    append_sample_locked(phase, pid_alive);
  }

 private:
  static std::string slurp_file(const char* path) {
    std::ifstream input(path);
    if (!input.is_open()) {
      return "";
    }
    std::ostringstream oss;
    oss << input.rdbuf();
    return oss.str();
  }

  static long extract_named_kb_value(
      const std::string& text,
      const std::string& name) {
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
      if (line.rfind(name + ":", 0) != 0) {
        continue;
      }
      std::istringstream line_stream(line.substr(name.size() + 1));
      long value = 0;
      line_stream >> value;
      return value;
    }
    return 0;
  }

  void write_header() {
    out_ << "elapsed_s,phase,pid_alive,dumpsys_total_pss_kb,dumpsys_total_rss_kb,"
            "dumpsys_total_swap_kb,smaps_rss_kb,smaps_pss_kb,smaps_private_dirty_kb,"
            "smaps_shared_clean_kb,mem_available_kb,cached_kb,dma_heap_pool_kb,"
            "gpu_total_kb,kgsl_shmem_usage_kb,self_rss_kb,kv_physical_committed_kb,"
            "kv_total_kb\n";
  }

  void append_sample_locked(const char* phase, bool pid_alive) {
    if (!out_.is_open()) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(now - started_at_)
            .count() /
        1000000.0;
    const std::string smaps_rollup = slurp_file("/proc/self/smaps_rollup");
    const std::string meminfo = slurp_file("/proc/meminfo");
    const size_t rss_bytes = executorch::extension::llm::get_rss_bytes();
    const long self_rss_kb =
        rss_bytes > 0 ? static_cast<long>(rss_bytes / 1024) : 0;
    const BackendMemoryMetrics metrics =
        probe_ ? probe_() : BackendMemoryMetrics{};
    out_ << elapsed << "," << phase << "," << (pid_alive ? "1" : "0")
         << ",,,," << extract_named_kb_value(smaps_rollup, "Rss") << ","
         << extract_named_kb_value(smaps_rollup, "Pss") << ","
         << extract_named_kb_value(smaps_rollup, "Private_Dirty") << ","
         << extract_named_kb_value(smaps_rollup, "Shared_Clean") << ","
         << extract_named_kb_value(meminfo, "MemAvailable") << ","
         << extract_named_kb_value(meminfo, "Cached") << ","
         << extract_named_kb_value(meminfo, "DmaHeapPool") << ","
         << extract_named_kb_value(meminfo, "GpuTotal") << ","
         << extract_named_kb_value(meminfo, "KgslShmemUsage") << ","
         << self_rss_kb << "," << metrics.kv_physical_committed_kb << ","
         << metrics.kv_total_kb << "\n";
    out_.flush();
  }

  std::string output_path_;
  Probe probe_;
  int interval_ms_;
  std::atomic<bool> running_{false};
  std::chrono::steady_clock::time_point started_at_{};
  std::ofstream out_;
  std::mutex mu_;
  std::thread worker_;
};

} // namespace executorch::examples::foundation
