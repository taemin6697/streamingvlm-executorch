/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

DEFINE_string(
    model_path,
    "internvl3_xnnpack_multimodal.pte",
    "Combined multimodal .pte path.");
DEFINE_string(tokenizer_path, "tokenizer.json", "Tokenizer json path.");
DEFINE_string(frame_dir, "", "Directory containing frame_NNNN.bin files.");
DEFINE_int32(frame_count, 1, "Number of frames to read from frame_dir.");
DEFINE_string(question, "Describe this image.", "Question for the model.");
DEFINE_string(output_path, "", "Optional file path to save the generated text.");
DEFINE_int32(
    image_size,
    448,
    "Expected square image size for preprocessed CHW float frames.");
DEFINE_int32(
    max_new_tokens,
    128,
    "Maximum number of new tokens to generate for the answer.");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy).");
DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads. -1 lets ExecuTorch choose performant cores.");
DEFINE_bool(echo, true, "Echo prompt and response to stdout.");
DEFINE_string(
    dump_input_path,
    "",
    "Dump actual input prompt (text + token_ids) to file before inference.");
DEFINE_bool(
    save_log,
    false,
    "Save output + proc.csv (timing) + mem.csv (RSS) + tokens.csv. Requires --output_path.");

using ::executorch::extension::llm::GenerationConfig;
using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::MultimodalInput;
using ::executorch::extension::llm::create_multimodal_runner;
using ::executorch::extension::llm::load_tokenizer;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using ::executorch::runtime::Error;

namespace {

// Read RSS from /proc/self/status (Linux/Android). Returns KB or -1.
static long rss_kb() {
#if defined(__linux__) || defined(__ANDROID__)
  std::ifstream f("/proc/self/status");
  if (!f.is_open())
    return -1;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      long kb = 0;
      if (std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb) == 1)
        return kb;
    }
  }
#endif
  return -1;
}

struct MemInfo {
  long mem_total_kb = -1;
  long mem_available_kb = -1;
};
static MemInfo meminfo_kb() {
  MemInfo out;
#if defined(__linux__) || defined(__ANDROID__)
  std::ifstream f("/proc/meminfo");
  if (!f.is_open())
    return out;
  std::string line;
  while (std::getline(f, line)) {
    long kb = 0;
    if (line.rfind("MemTotal:", 0) == 0 &&
        std::sscanf(line.c_str(), "MemTotal: %ld kB", &kb) == 1) {
      out.mem_total_kb = kb;
    } else if (line.rfind("MemAvailable:", 0) == 0 &&
               std::sscanf(line.c_str(), "MemAvailable: %ld kB", &kb) == 1) {
      out.mem_available_kb = kb;
    }
    if (out.mem_total_kb >= 0 && out.mem_available_kb >= 0)
      break;
  }
#endif
  return out;
}

constexpr int32_t kInternVL3ImageSeqLen = 256;
constexpr const char* kInternVL3ImageToken = "<IMG_CONTEXT>";

std::string frame_path(const std::string& dir, int idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/frame_%04d.bin", idx);
  return dir + buf;
}

Image load_preprocessed_frame(
    const std::string& path,
    int32_t image_size,
    int32_t channels = 3) {
  std::ifstream input(path, std::ios::binary);
  ET_CHECK_MSG(input.is_open(), "Failed to open frame bin: %s", path.c_str());

  input.seekg(0, std::ios::end);
  const auto num_bytes = input.tellg();
  input.seekg(0, std::ios::beg);

  const size_t expected_floats = static_cast<size_t>(channels) * image_size *
      image_size;
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
  return Image(std::move(data), image_size, image_size, channels);
}

std::string build_full_prompt_text(
    int32_t frame_count,
    const std::string& question) {
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

std::vector<MultimodalInput> build_inputs(
    const std::string& frame_dir,
    int32_t frame_count,
    int32_t image_size,
    const std::string& question) {
  std::vector<MultimodalInput> inputs;
  inputs.emplace_back(
      make_text_input(std::string("<|im_start|>user:\n")));

  for (int32_t idx = 0; idx < frame_count; ++idx) {
    const int32_t frame_num = idx + 1;
    std::string frame_prompt = "Frame" + std::to_string(frame_num) + ": <img>";
    for (int32_t i = 0; i < kInternVL3ImageSeqLen; ++i) {
      frame_prompt += kInternVL3ImageToken;
    }
    frame_prompt += "</img>\n";
    inputs.emplace_back(make_text_input(frame_prompt));
    inputs.emplace_back(
        make_image_input(load_preprocessed_frame(
            frame_path(frame_dir, idx), image_size)));
  }

  inputs.emplace_back(make_text_input(
      question + "<|im_end|>\n<|im_start|>assistant\n"));
  return inputs;
}

void maybe_configure_threadpool(int32_t cpu_threads) {
#if defined(ET_USE_THREADPOOL)
  const uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#else
  (void)cpu_threads;
#endif
}

} // namespace

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ET_CHECK_MSG(!FLAGS_frame_dir.empty(), "--frame_dir is required.");
  ET_CHECK_MSG(FLAGS_frame_count > 0, "--frame_count must be > 0.");

  maybe_configure_threadpool(FLAGS_cpu_threads);

  auto tokenizer = load_tokenizer(FLAGS_tokenizer_path);
  ET_CHECK_MSG(
      tokenizer != nullptr,
      "Failed to load tokenizer: %s",
      FLAGS_tokenizer_path.c_str());

  if (!FLAGS_dump_input_path.empty()) {
    std::string full_prompt =
        build_full_prompt_text(FLAGS_frame_count, FLAGS_question);
    auto encode_res = tokenizer->encode(full_prompt, 0, 0);
    ET_CHECK_MSG(
        encode_res.ok(),
        "Failed to encode prompt for dump: %s",
        FLAGS_dump_input_path.c_str());
    std::vector<uint64_t> input_ids = encode_res.get();

    std::ofstream dump(FLAGS_dump_input_path);
    ET_CHECK_MSG(
        dump.is_open(),
        "Failed to open dump file: %s",
        FLAGS_dump_input_path.c_str());
    dump << "# C++ runner 실제 입력 (토크나이저 기준)\n";
    dump << "# question=" << FLAGS_question << ", frame_count="
         << FLAGS_frame_count << "\n\n";
    dump << "=== full prompt (text) ===\n" << full_prompt << "\n\n";
    dump << "=== input_ids (" << input_ids.size() << " tokens) ===\n[";
    for (size_t i = 0; i < input_ids.size(); ++i) {
      if (i > 0)
        dump << ",";
      dump << input_ids[i];
    }
    dump << "]\n";
    dump.close();
    ET_LOG(Info, "Input dumped to %s", FLAGS_dump_input_path.c_str());
  }

  auto runner =
      create_multimodal_runner(FLAGS_model_path, std::move(tokenizer));
  ET_CHECK_MSG(
      runner != nullptr,
      "Failed to create multimodal runner from %s",
      FLAGS_model_path.c_str());

  // save_log: mem sampler, t_run_start (로딩 전 시작)
  std::atomic<bool> sampler_running{false};
  std::thread rss_sampler;
  std::mutex fmem_mutex;
  std::ofstream fmem;
  std::ofstream fproc_append;
  std::ofstream ftokens;
  int64_t t_run_start_ms = 0;
  long rss_load_start = -1;

  if (FLAGS_save_log) {
    ET_CHECK_MSG(
        !FLAGS_output_path.empty(),
        "--save_log requires --output_path");
    t_run_start_ms = executorch::extension::llm::time_in_ms();
    rss_load_start = rss_kb();
    fmem.open(FLAGS_output_path + ".mem.csv");
    if (fmem.is_open()) {
      fmem << "elapsed_s,rss_kb,rss_mb,delta_rss_kb,mem_total_kb,mem_available_kb\n";
      sampler_running.store(true);
      rss_sampler = std::thread([&]() {
        long last_rss = rss_kb();
        double t_run_start = static_cast<double>(t_run_start_ms) / 1000.0;
        while (sampler_running.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          if (!sampler_running.load())
            break;
          double elapsed_s =
              (executorch::extension::llm::time_in_ms() / 1000.0) - t_run_start;
          long cur = rss_kb();
          long delta = (last_rss >= 0 && cur >= 0) ? cur - last_rss : 0;
          last_rss = cur;
          MemInfo mi = meminfo_kb();
          std::lock_guard<std::mutex> lk(fmem_mutex);
          fmem << elapsed_s << "," << cur << "," << cur / 1024.0 << ","
               << delta << "," << mi.mem_total_kb << "," << mi.mem_available_kb
               << "\n";
          fmem.flush();
        }
      });
    }
  }

  const Error load_error = runner->load();
  ET_CHECK_MSG(load_error == Error::Ok, "Failed to load multimodal runner");

  if (FLAGS_save_log) {
    long rss_load_end = rss_kb();
    double elapsed_load_end =
        (executorch::extension::llm::time_in_ms() - t_run_start_ms) / 1000.0;
    fproc_append.open(FLAGS_output_path + ".proc.csv");
    if (fproc_append.is_open()) {
      fproc_append << "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
                   "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
                   "kv_used_kb,kv_total_kb,token_idx\n";
      fproc_append << "# L: Loading  T_Prefill: col_a=prefill_ms  Decode: col_b=token_gen_ms\n";
      fproc_append << "# D: 토큰별 decode\n";
      fproc_append << "L,0," << elapsed_load_end << "," << rss_load_start << ","
                   << rss_load_end << ",,,,,,,,\n";
      fproc_append.flush();
    }
    std::lock_guard<std::mutex> lk(fmem_mutex);
    if (fmem.is_open()) {
      MemInfo mi = meminfo_kb();
      fmem << elapsed_load_end << "," << rss_load_end << ","
           << rss_load_end / 1024.0 << ",0," << mi.mem_total_kb << ","
           << mi.mem_available_kb << "\n";
      fmem.flush();
    }
  }

  auto inputs = build_inputs(
      FLAGS_frame_dir, FLAGS_frame_count, FLAGS_image_size, FLAGS_question);

  GenerationConfig config;
  config.temperature = static_cast<float>(FLAGS_temperature);
  config.echo = FLAGS_echo;
  config.max_new_tokens = FLAGS_max_new_tokens;

  std::ostringstream output;
  struct TokenTiming {
    int64_t token_idx;
    int64_t kv_pos;
    long start_ms;
    long end_ms;
  };
  std::vector<TokenTiming> token_timing_events;
  std::atomic<int64_t> token_count{0};
  std::atomic<long> last_token_end_ms{-1};

  auto token_callback = [&](const std::string& piece) {
    output << piece;
    if (FLAGS_echo) {
      std::fwrite(piece.data(), 1, piece.size(), stdout);
      std::fflush(stdout);
    }
    if (FLAGS_save_log) {
      int64_t idx = token_count.fetch_add(1);
      long now_ms = executorch::extension::llm::time_in_ms();
      long start_ms = (idx == 0) ? t_run_start_ms : last_token_end_ms.load();
      last_token_end_ms.store(now_ms);
      token_timing_events.push_back(
          {idx, -1, start_ms, now_ms});  // kv_pos filled in stats_cb
    }
  };

  if (FLAGS_save_log) {
    ftokens.open(FLAGS_output_path + ".tokens.csv");
    if (ftokens.is_open()) {
      ftokens << "token_idx,kv_pos,elapsed_s_start,elapsed_s_end,ms\n";
    }
  }

  std::function<void(const executorch::extension::llm::Stats&)> stats_cb;
  if (FLAGS_save_log && fproc_append.is_open()) {
    const int64_t kv_total = 2048;  // fallback, 모델 메타에서 가져올 수 있으면 개선
    stats_cb = [&](const executorch::extension::llm::Stats& s) {
      double elapsed_prefill_start =
          static_cast<double>(s.inference_start_ms - t_run_start_ms) / 1000.0;
      double elapsed_prefill_end =
          static_cast<double>(s.first_token_ms - t_run_start_ms) / 1000.0;
      double prefill_ms = s.first_token_ms - s.inference_start_ms;
      double token_gen_ms = s.inference_end_ms - s.first_token_ms;
      int64_t kv_prefill = s.num_prompt_tokens;
      int64_t kv_pos_end = s.num_prompt_tokens + s.num_generated_tokens;
      double kv_pct = (kv_total > 0) ? 100.0 * kv_pos_end / kv_total : 0;
      int64_t kv_used_kb = (kv_total > 0) ? (kv_pos_end * 24) / 1024 : 0;
      int64_t kv_total_kb = (kv_total * 24) / 1024;

      long rss_first = rss_kb();
      long rss_end = rss_kb();

      fproc_append << "T_Prefill," << elapsed_prefill_start << ","
                   << elapsed_prefill_end << ","
                   << rss_load_start << "," << rss_first << ","
                   << prefill_ms << ",," << prefill_ms << ","
                   << kv_prefill << "," << kv_total << "," << kv_pct << ","
                   << kv_used_kb << "," << kv_total_kb << ",\n";
      fproc_append << "Decode,"
                   << static_cast<double>(s.first_token_ms - t_run_start_ms) /
                          1000.0
                   << ","
                   << static_cast<double>(s.inference_end_ms - t_run_start_ms) /
                          1000.0
                   << "," << rss_first << "," << rss_end << ","
                   << "," << token_gen_ms << "," << token_gen_ms << ","
                   << kv_pos_end << "," << kv_total << "," << kv_pct << ","
                   << kv_used_kb << "," << kv_total_kb << ",\n";

      for (size_t i = 0; i < token_timing_events.size(); ++i) {
        const auto& te = token_timing_events[i];
        int64_t kv_pos = s.num_prompt_tokens + static_cast<int64_t>(i);
        double start_s =
            static_cast<double>(te.start_ms - t_run_start_ms) / 1000.0;
        double end_s =
            static_cast<double>(te.end_ms - t_run_start_ms) / 1000.0;
        long ms = te.end_ms - te.start_ms;
        fproc_append << "D," << start_s << "," << end_s << ","
                    << rss_first << "," << rss_first << ","
                    << "," << ms << "," << ms << ","
                    << kv_pos << "," << kv_total << ","
                    << (kv_total > 0 ? 100.0 * kv_pos / kv_total : 0) << ","
                    << (kv_total > 0 ? (kv_pos * 24) / 1024 : 0) << ","
                    << kv_total_kb << "," << te.token_idx << "\n";

        if (ftokens.is_open()) {
          ftokens << te.token_idx << "," << kv_pos << "," << start_s << ","
                  << end_s << "," << ms << "\n";
          ftokens.flush();
        }
      }
      fproc_append.flush();
    };
  }

  const Error error = runner->generate(
      inputs, config, token_callback, FLAGS_save_log ? stats_cb : nullptr);
  ET_CHECK_MSG(error == Error::Ok, "Failed to run multimodal generation");

  if (!FLAGS_output_path.empty()) {
    std::ofstream output_file(FLAGS_output_path);
    ET_CHECK_MSG(
        output_file.is_open(),
        "Failed to open output file: %s",
        FLAGS_output_path.c_str());
    output_file << output.str();
  }

  if (FLAGS_save_log) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    sampler_running.store(false);
    if (rss_sampler.joinable())
      rss_sampler.join();
    if (fproc_append.is_open())
      fproc_append.close();
    if (ftokens.is_open())
      ftokens.close();
    if (fmem.is_open())
      fmem.close();
    ET_LOG(
        Info,
        "save_log: %s  %s.proc.csv  %s.tokens.csv  %s.mem.csv",
        FLAGS_output_path.c_str(),
        FLAGS_output_path.c_str(),
        FLAGS_output_path.c_str(),
        FLAGS_output_path.c_str());
  }

  if (FLAGS_echo) {
    std::fwrite("\n", 1, 1, stdout);
  }
  return 0;
}
