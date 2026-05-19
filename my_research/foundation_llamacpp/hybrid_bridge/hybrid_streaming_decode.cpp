#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
#include "vision_encoder_et.hpp"
#include "hybrid_decode.cpp"
#else
#include "opencl_phase_mtmd.cpp"
#endif

#include "kv_reposition.hpp"

namespace {

struct TileRecord {
  std::string bin;
  std::string layout_image;
};

struct FrameRecord {
  int index = 0;
  double timestamp_s = 0.0;
  std::vector<TileRecord> tiles;
};

struct PromptEvent {
  double timestamp_s = 0.0;
  std::string prompt;
};

struct Manifest {
  std::string source_kind;
  double sampling_fps = 0.0;
  std::string stream_mode;
  double window_sec = -1.0;
  int window_max_frames = 0;
  std::string prompt;
  std::vector<FrameRecord> frames;
  std::vector<PromptEvent> prompts;
};

struct PhaseTiming {
  std::string name;
  long start_ms = 0;
  long end_ms = 0;
};

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "failed to open manifest: %s\n", path.c_str());
    std::exit(1);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string unescape_json_string(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] != '\\' || i + 1 >= value.size()) {
      out.push_back(value[i]);
      continue;
    }
    const char c = value[++i];
    switch (c) {
      case 'n':
        out.push_back('\n');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case '"':
      case '\\':
      case '/':
        out.push_back(c);
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

bool find_number_after(const std::string& text, size_t pos, const std::string& key, double& out) {
  const size_t key_pos = text.find("\"" + key + "\"", pos);
  if (key_pos == std::string::npos) {
    return false;
  }
  const size_t colon = text.find(':', key_pos);
  if (colon == std::string::npos) {
    return false;
  }
  char* end = nullptr;
  out = std::strtod(text.c_str() + colon + 1, &end);
  return end != text.c_str() + colon + 1;
}

bool find_string_after(const std::string& text, size_t pos, const std::string& key, std::string& out) {
  const size_t key_pos = text.find("\"" + key + "\"", pos);
  if (key_pos == std::string::npos) {
    return false;
  }
  const size_t colon = text.find(':', key_pos);
  const size_t quote = text.find('"', colon + 1);
  if (colon == std::string::npos || quote == std::string::npos) {
    return false;
  }
  std::string raw;
  bool escaped = false;
  for (size_t i = quote + 1; i < text.size(); ++i) {
    const char c = text[i];
    if (!escaped && c == '"') {
      out = unescape_json_string(raw);
      return true;
    }
    if (!escaped && c == '\\') {
      escaped = true;
      raw.push_back(c);
      continue;
    }
    escaped = false;
    raw.push_back(c);
  }
  return false;
}

size_t find_matching(const std::string& text, size_t open_pos, char open_c, char close_c) {
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t i = open_pos; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == open_c) {
      ++depth;
    } else if (c == close_c) {
      --depth;
      if (depth == 0) {
        return i;
      }
    }
  }
  return std::string::npos;
}

std::vector<std::string> object_blocks_in_array(const std::string& text, const std::string& key) {
  std::vector<std::string> out;
  const size_t key_pos = text.find("\"" + key + "\"");
  if (key_pos == std::string::npos) {
    return out;
  }
  const size_t array_start = text.find('[', key_pos);
  const size_t array_end = array_start == std::string::npos ? std::string::npos : find_matching(text, array_start, '[', ']');
  if (array_start == std::string::npos || array_end == std::string::npos) {
    return out;
  }
  size_t cursor = array_start + 1;
  while (cursor < array_end) {
    const size_t obj_start = text.find('{', cursor);
    if (obj_start == std::string::npos || obj_start >= array_end) {
      break;
    }
    const size_t obj_end = find_matching(text, obj_start, '{', '}');
    if (obj_end == std::string::npos || obj_end > array_end) {
      break;
    }
    out.push_back(text.substr(obj_start, obj_end - obj_start + 1));
    cursor = obj_end + 1;
  }
  return out;
}

Manifest parse_manifest(const std::string& path) {
  const std::string text = read_file(path);
  Manifest manifest;
  find_string_after(text, 0, "source_kind", manifest.source_kind);
  find_number_after(text, 0, "sampling_fps", manifest.sampling_fps);
  find_string_after(text, 0, "stream_mode", manifest.stream_mode);
  find_number_after(text, 0, "window_sec", manifest.window_sec);
  find_string_after(text, 0, "prompt", manifest.prompt);
  double window_max_frames = 0.0;
  if (find_number_after(text, 0, "window_max_frames", window_max_frames)) {
    manifest.window_max_frames = static_cast<int>(window_max_frames);
  }

  for (const std::string& block : object_blocks_in_array(text, "frames")) {
    FrameRecord frame;
    double frame_index = 0.0;
    if (!find_number_after(block, 0, "stream_frame", frame_index)) {
      if (find_number_after(block, 0, "frame", frame_index)) {
        frame_index -= 1.0;
      }
    }
    frame.index = static_cast<int>(frame_index);
    find_number_after(block, 0, "timestamp_s", frame.timestamp_s);
    for (const std::string& tile_block : object_blocks_in_array(block, "tiles")) {
      TileRecord tile;
      find_string_after(tile_block, 0, "bin", tile.bin);
      find_string_after(tile_block, 0, "layout_image", tile.layout_image);
      if (!tile.bin.empty() || !tile.layout_image.empty()) {
        frame.tiles.push_back(tile);
      }
    }
    if (!frame.tiles.empty()) {
      manifest.frames.push_back(frame);
    }
  }

  for (const std::string& block : object_blocks_in_array(text, "prompt_events")) {
    PromptEvent event;
    find_number_after(block, 0, "time", event.timestamp_s);
    if (!find_string_after(block, 0, "prompt", event.prompt)) {
      find_string_after(block, 0, "text", event.prompt);
    }
    manifest.prompts.push_back(event);
  }
  if (manifest.prompts.empty() && !manifest.prompt.empty()) {
    manifest.prompts.push_back(PromptEvent{0.0, manifest.prompt});
  }
  std::sort(manifest.prompts.begin(), manifest.prompts.end(), [](const auto& a, const auto& b) {
    return a.timestamp_s < b.timestamp_s;
  });
  return manifest;
}

class EventWriter {
 public:
  explicit EventWriter(const std::string& path) : out_(path) {
    out_ << "event,frame_idx,prompt_idx,video_time_s,elapsed_s_start,elapsed_s_end,detail\n";
  }

  void row(
      const std::string& event,
      int frame_idx,
      int prompt_idx,
      double video_time_s,
      long origin_ms,
      long start_ms,
      long end_ms,
      const std::string& detail) {
    std::lock_guard<std::mutex> lock(mu_);
    out_ << event << ',' << frame_idx << ',' << prompt_idx << ',' << video_time_s << ','
         << (start_ms - origin_ms) / 1000.0 << ',' << (end_ms - origin_ms) / 1000.0 << ','
         << '"' << detail << '"' << '\n';
    out_.flush();
  }

 private:
  std::mutex mu_;
  std::ofstream out_;
};

struct Args {
  std::string manifest = "media_manifest.json";
  std::string encoder_path;
  std::string warmup_image_path;
  std::string runner = "./opencl_phase_mtmd";
  std::string media_mode;
  std::string stream_mode;
  std::string prompt_format = "internvl3";
  std::string model;
  std::string mmproj;
  std::string stream_events_path = "stream_events.csv";
  std::string phase_stats_path = "streaming_phase_stats.csv";
  std::string output_path = "foundation_output.txt";
  std::string token_io_path = "foundation_token_io.txt";
  std::string device;
  std::string cache_type_k;
  std::string cache_type_v;
  std::string fit;
  std::string flash_attn;
  std::string rope_suffix;
  int n_predict = 32;
  int ctx_size = 4096;
  int kv_init_size = 0;
  int kv_grow_step = 0;
  int batch_size = 2048;
  int ubatch_size = 512;
  int gpu_layers = 99;
  int threads = 4;
  int kv_reposition_keep_latest_frames = 0;
  double temperature = 0.0;
  double window_sec = -1.0;
  double play_speed = 1.0;
  int window_max_frames = 8;
  bool realtime = true;
  bool force_generation = false;
  bool single_buffer = false;
  bool online_buffer = false;
  bool latest_frame_only = false;
  bool partial_vision_kv = false;
  bool dynamic_kv_cache = false;
  bool no_kv_offload = false;
  bool mmproj_offload = true;
  bool no_warmup = false;
};

bool consume_value(int argc, char** argv, int& i, std::string& out) {
  if (i + 1 >= argc) {
    return false;
  }
  out = argv[++i];
  return true;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto eq = a.find('=');
    auto read_string = [&](const std::string& key, std::string& target) {
      if (a == key) {
        return consume_value(argc, argv, i, target);
      }
      if (eq != std::string::npos && a.substr(0, eq) == key) {
        target = a.substr(eq + 1);
        return true;
      }
      return false;
    };
    std::string tmp;
    if (read_string("--stream-manifest", args.manifest) || read_string("--stream_manifest", args.manifest)) {
      continue;
    }
    if (read_string("--encoder-path", args.encoder_path) || read_string("--encoder_path", args.encoder_path) ||
        read_string("--warmup-image-path", args.warmup_image_path) || read_string("--warmup_image_path", args.warmup_image_path) ||
        read_string("--media-mode", args.media_mode) || read_string("--media_mode", args.media_mode) ||
        read_string("--stream-mode", args.stream_mode) || read_string("--stream_mode", args.stream_mode) ||
        read_string("--prompt-format", args.prompt_format) || read_string("--prompt_format", args.prompt_format) ||
        read_string("--runner", args.runner) || read_string("-m", args.model) || read_string("--model", args.model) ||
        read_string("--mmproj", args.mmproj) || read_string("--stream-events-path", args.stream_events_path) ||
        read_string("--stream_events_path", args.stream_events_path) || read_string("--phase-stats-path", args.phase_stats_path) ||
        read_string("--phase_stats_path", args.phase_stats_path) || read_string("--output", args.output_path) ||
        read_string("--token-io-path", args.token_io_path) || read_string("--device", args.device) ||
        read_string("--cache-type-k", args.cache_type_k) || read_string("--cache-type-v", args.cache_type_v) ||
        read_string("--fit", args.fit) || read_string("--flash-attn", args.flash_attn) ||
        read_string("--rope-suffix", args.rope_suffix)) {
      continue;
    }
    if (read_string("-n", tmp) || read_string("--n-predict", tmp)) {
      args.n_predict = std::atoi(tmp.c_str());
    } else if (read_string("-c", tmp) || read_string("--ctx-size", tmp)) {
      args.ctx_size = std::atoi(tmp.c_str());
    } else if (read_string("--kv-init-size", tmp)) {
      args.kv_init_size = std::atoi(tmp.c_str());
    } else if (read_string("--kv-grow-step", tmp)) {
      args.kv_grow_step = std::atoi(tmp.c_str());
    } else if (read_string("-b", tmp) || read_string("--batch-size", tmp)) {
      args.batch_size = std::atoi(tmp.c_str());
    } else if (read_string("-ub", tmp) || read_string("--ubatch-size", tmp)) {
      args.ubatch_size = std::atoi(tmp.c_str());
    } else if (read_string("-ngl", tmp) || read_string("--gpu-layers", tmp)) {
      args.gpu_layers = std::atoi(tmp.c_str());
    } else if (read_string("-t", tmp) || read_string("--threads", tmp)) {
      args.threads = std::atoi(tmp.c_str());
    } else if (read_string("--kv-reposition-keep-latest-frames", tmp) ||
               read_string("--kv_reposition_keep_latest_frames", tmp)) {
      args.kv_reposition_keep_latest_frames = std::atoi(tmp.c_str());
    } else if (read_string("--temp", tmp) || read_string("--temperature", tmp)) {
      args.temperature = std::atof(tmp.c_str());
    } else if (read_string("--window-sec", tmp) || read_string("--window_sec", tmp)) {
      args.window_sec = std::atof(tmp.c_str());
    } else if (read_string("--window-max-frames", tmp) || read_string("--window_max_frames", tmp)) {
      args.window_max_frames = std::atoi(tmp.c_str());
    } else if (read_string("--play-speed", tmp) || read_string("--play_speed", tmp)) {
      args.play_speed = std::atof(tmp.c_str());
    } else if (a == "--single-buffer" || a == "--single_buffer") {
      args.single_buffer = true;
    } else if (a == "--online-buffer" || a == "--online_buffer") {
      args.online_buffer = true;
    } else if (a == "--latest-frame-only" || a == "--latest_frame_only") {
      args.latest_frame_only = true;
    } else if (a == "--partial-vision-kv" || a == "--partial_vision_kv") {
      args.partial_vision_kv = true;
    } else if (a == "--dynamic-kv-cache") {
      args.dynamic_kv_cache = true;
    } else if (a == "--force-generation") {
      args.force_generation = true;
    } else if (a == "--no-kv-offload") {
      args.no_kv_offload = true;
    } else if (a == "--no-mmproj-offload") {
      args.mmproj_offload = false;
    } else if (a == "--mmproj-offload") {
      args.mmproj_offload = true;
    } else if (a == "--no-warmup") {
      args.no_warmup = true;
    } else if (a == "--no-realtime") {
      args.realtime = false;
    }
  }
  return args;
}

long now_ms() {
  return ggml_time_ms();
}

std::string normalize_stream_mode(std::string mode, bool single_buffer) {
  if (single_buffer) {
    return "on_demand";
  }
  std::replace(mode.begin(), mode.end(), '-', '_');
  if (mode.empty()) {
    return "on_demand";
  }
  if (mode == "single_buffer") {
    return "on_demand";
  }
  if (mode == "on_demand" || mode == "sliding_window" || mode == "vision_prefill") {
    return mode;
  }
  std::fprintf(stderr, "unsupported --stream-mode: %s\n", mode.c_str());
  std::exit(2);
}

bool is_singleton_video_mode(const Args& args) {
  (void)args;
  return false;
}

void reset_decode_context_for_singleton(decode_context& ctx) {
  llama_memory_clear(llama_get_memory(ctx.lctx), true);
  llama_synchronize(ctx.lctx);
  llama_perf_context_reset(ctx.lctx);
  common_sampler_reset(ctx.smpl);
  ctx.chat_history.clear();
  ctx.n_past = 0;
}

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out += "'";
  return out;
}

void append_phase_row(std::ofstream& out, const std::string& name, long start_ms, long end_ms, long origin_ms) {
  out << name << "," << (start_ms - origin_ms) / 1000.0 << "," << (end_ms - origin_ms) / 1000.0
      << ",,," << (end_ms - start_ms) << ",," << (end_ms - start_ms) << ",,,,,,,0\n";
  out.flush();
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

std::string prompt_phase_path(int prompt_idx) {
  return "stream_prompt_phase_" + std::to_string(prompt_idx) + ".csv";
}

void write_stream_text_file(const std::string& path, const std::string& value) {
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
  streamingvlm::hybrid_bridge::write_text_file(path, value);
#else
  write_text_file(path, value);
#endif
}

void append_phase_file(std::ofstream& out, const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line.rfind("row_type,", 0) == 0 || line[0] == '#') {
      continue;
    }
    std::vector<std::string> cols = split_csv_line(line);
    if (cols.size() < 15) {
      continue;
    }
    for (size_t i = 0; i < cols.size(); ++i) {
      if (i) {
        out << ',';
      }
      out << cols[i];
    }
    out << '\n';
  }
  out.flush();
}

std::vector<std::string> build_llama_args(const Args& args, const std::string& image, const std::string& prompt) {
  std::vector<std::string> out = {
      "hybrid_streaming_decode",
      "-m",
      args.model,
      "--mmproj",
      args.mmproj,
      "--image",
      image,
      "-p",
      prompt,
      "-n",
      std::to_string(args.n_predict),
      "-t",
      std::to_string(args.threads),
      "--n-gpu-layers",
      std::to_string(args.gpu_layers),
      "--batch-size",
      std::to_string(args.batch_size),
      "--ubatch-size",
      std::to_string(args.ubatch_size),
      "--temp",
      std::to_string(args.temperature),
  };
  if (!args.dynamic_kv_cache) {
    out.push_back("--ctx-size");
    out.push_back(std::to_string(args.ctx_size));
  } else {
    out.push_back("--dynamic-kv-cache");
    if (args.kv_init_size > 0) {
      out.push_back("--kv-init-size");
      out.push_back(std::to_string(args.kv_init_size));
    }
    if (args.kv_grow_step > 0) {
      out.push_back("--kv-grow-step");
      out.push_back(std::to_string(args.kv_grow_step));
    }
  }
  if (!args.device.empty()) {
    out.push_back("--device");
    out.push_back(args.device);
  }
  if (!args.cache_type_k.empty()) {
    out.push_back("--cache-type-k");
    out.push_back(args.cache_type_k);
  }
  if (!args.cache_type_v.empty()) {
    out.push_back("--cache-type-v");
    out.push_back(args.cache_type_v);
  }
  if (!args.fit.empty()) {
    out.push_back("--fit");
    out.push_back(args.fit);
  }
  if (!args.flash_attn.empty()) {
    out.push_back("--flash-attn");
    out.push_back(args.flash_attn);
  }
  if (args.no_kv_offload) {
    out.push_back("--no-kv-offload");
  }
  if (!args.mmproj_offload) {
    out.push_back("--no-mmproj-offload");
  }
  if (args.no_warmup) {
    out.push_back("--no-warmup");
  }
  if (!args.rope_suffix.empty()) {
    out.push_back(args.rope_suffix);
  }
  return out;
}

std::vector<char*> mutable_argv(std::vector<std::string>& args) {
  std::vector<char*> out;
  out.reserve(args.size());
  for (std::string& arg : args) {
    out.push_back(arg.data());
  }
  return out;
}

std::vector<FrameRecord> evenly_limit_frames(const std::vector<FrameRecord>& frames, int limit) {
  if (limit <= 0) {
    std::fprintf(stderr, "--window-max-frames must be positive\n");
    std::exit(2);
  }
  if (static_cast<int>(frames.size()) <= limit) {
    return frames;
  }
  if (limit == 1) {
    return {frames.back()};
  }
  std::vector<FrameRecord> out;
  out.reserve(static_cast<size_t>(limit));
  const int last = static_cast<int>(frames.size()) - 1;
  for (int i = 0; i < limit; ++i) {
    const int idx = static_cast<int>(std::llround(static_cast<double>(i) * last / (limit - 1)));
    out.push_back(frames[static_cast<size_t>(idx)]);
  }
  return out;
}

std::vector<FrameRecord> select_prompt_frames(
    const Args& args,
    const std::vector<FrameRecord>& available_frames,
    const FrameRecord& current_frame,
    const PromptEvent& prompt) {
  if (args.stream_mode == "on_demand") {
    return {current_frame};
  }

  if (args.stream_mode == "vision_prefill") {
    std::vector<FrameRecord> selected;
    for (const FrameRecord& frame : available_frames) {
      if (frame.timestamp_s <= prompt.timestamp_s) {
        selected.push_back(frame);
      }
    }
    if (selected.empty()) {
      selected.push_back(current_frame);
    }
    return selected;
  }

  const double start_s = args.window_sec > 0.0 ? prompt.timestamp_s - args.window_sec : -1.0e30;
  std::vector<FrameRecord> selected;
  for (const FrameRecord& frame : available_frames) {
    if (frame.timestamp_s >= start_s && frame.timestamp_s <= prompt.timestamp_s) {
      selected.push_back(frame);
    }
  }
  if (selected.empty()) {
    for (auto it = available_frames.rbegin(); it != available_frames.rend(); ++it) {
      if (it->timestamp_s <= prompt.timestamp_s) {
        selected.push_back(*it);
        break;
      }
    }
  }
  if (selected.empty()) {
    selected.push_back(current_frame);
  }
  return evenly_limit_frames(selected, args.window_max_frames);
}

constexpr const char* SVLM_QUESTION_SENTINEL = "<SVLM_QUESTION_SENTINEL>";

std::string build_video_prompt_prefix(const std::vector<FrameRecord>& frames) {
  std::string out;
  for (size_t frame_i = 0; frame_i < frames.size(); ++frame_i) {
    out += "Frame" + std::to_string(frame_i + 1) + ": ";
    const size_t n_tiles = std::max<size_t>(1, frames[frame_i].tiles.size());
    for (size_t tile_i = 0; tile_i < n_tiles; ++tile_i) {
      out += mtmd_default_marker();
    }
    out += "\n";
  }
  return out;
}

std::string build_stream_frame_prompt_line(const FrameRecord& frame) {
  std::string out = "Frame" + std::to_string(frame.index + 1) + ": ";
  const size_t n_tiles = std::max<size_t>(1, frame.tiles.size());
  for (size_t tile_i = 0; tile_i < n_tiles; ++tile_i) {
    out += mtmd_default_marker();
  }
  out += "\n";
  return out;
}

std::string build_stream_video_prompt_prefix(const std::vector<FrameRecord>& frames) {
  std::string out;
  for (const FrameRecord& frame : frames) {
    out += build_stream_frame_prompt_line(frame);
  }
  return out;
}

size_t next_non_frame_prefix_offset(const std::string& content) {
  size_t pos = 0;
  const std::string marker = mtmd_default_marker();
  while (pos < content.size()) {
    const size_t line_start = pos;
    if (content.compare(pos, 5, "Frame") != 0) {
      break;
    }
    pos += 5;
    const size_t digits_begin = pos;
    while (pos < content.size() && std::isdigit(static_cast<unsigned char>(content[pos]))) {
      ++pos;
    }
    if (pos == digits_begin || pos + 2 > content.size() || content[pos] != ':' || content[pos + 1] != ' ') {
      pos = line_start;
      break;
    }
    pos += 2;
    bool saw_image_marker = false;
    while (content.compare(pos, marker.size(), marker) == 0) {
      saw_image_marker = true;
      pos += marker.size();
    }
    if (!saw_image_marker || pos >= content.size() || content[pos] != '\n') {
      pos = line_start;
      break;
    }
    ++pos;
  }
  return pos;
}

std::string strip_stream_video_prompt_prefix(const std::string& content) {
  return content.substr(next_non_frame_prefix_offset(content));
}

bool update_first_video_user_message(
    std::vector<common_chat_msg>& chat_history,
    const std::vector<FrameRecord>& frames) {
  for (common_chat_msg& msg : chat_history) {
    if (msg.role == "user") {
      msg.content = build_stream_video_prompt_prefix(frames) + strip_stream_video_prompt_prefix(msg.content);
      return true;
    }
  }
  return false;
}

std::string build_video_prompt(const std::vector<FrameRecord>& frames, const std::string& raw_prompt) {
  std::string out = build_video_prompt_prefix(frames);
  out += raw_prompt;
  return out;
}

std::vector<int> frame_indices_for(const std::vector<FrameRecord>& frames) {
  std::vector<int> out;
  out.reserve(frames.size());
  for (const FrameRecord& frame : frames) {
    out.push_back(frame.index);
  }
  return out;
}

std::vector<std::string> layout_images_for_frames(const std::vector<FrameRecord>& frames) {
  std::vector<std::string> out;
  for (const FrameRecord& frame : frames) {
    for (const TileRecord& tile : frame.tiles) {
      if (!tile.layout_image.empty()) {
        out.push_back(tile.layout_image);
      }
    }
  }
  return out;
}

std::vector<int> image_frame_indices_for_frames(const std::vector<FrameRecord>& frames) {
  std::vector<int> out;
  for (const FrameRecord& frame : frames) {
    const size_t n_tiles = std::max<size_t>(1, frame.tiles.size());
    for (size_t tile_i = 0; tile_i < n_tiles; ++tile_i) {
      out.push_back(frame.index);
    }
  }
  return out;
}

std::vector<FrameRecord> filter_frames_by_index_set(
    const std::vector<FrameRecord>& frames,
    const std::set<int>& keep) {
  std::vector<FrameRecord> out;
  for (const FrameRecord& frame : frames) {
    if (keep.find(frame.index) != keep.end()) {
      out.push_back(frame);
    }
  }
  return out;
}

std::vector<FrameRecord> filter_frames_by_index_list(
    const std::vector<FrameRecord>& frames,
    const std::vector<int>& indices) {
  std::vector<FrameRecord> out;
  for (int index : indices) {
    auto it = std::find_if(
        frames.begin(),
        frames.end(),
        [&](const FrameRecord& frame) { return frame.index == index; });
    if (it != frames.end()) {
      out.push_back(*it);
    }
  }
  return out;
}

std::vector<std::string> bins_for_frames(const std::vector<FrameRecord>& frames) {
  std::vector<std::string> out;
  for (const FrameRecord& frame : frames) {
    for (const TileRecord& tile : frame.tiles) {
      if (!tile.bin.empty()) {
        out.push_back(tile.bin);
      }
    }
  }
  return out;
}

int last_frame_index(const std::vector<FrameRecord>& frames) {
  return frames.empty() ? -1 : frames.back().index;
}

std::string join_strings(const std::vector<std::string>& values, const char* sep) {
  std::string out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) {
      out += sep;
    }
    out += values[i];
  }
  return out;
}

std::unique_ptr<decode_context> load_single_buffer_decoder_context(
    const Args& args,
    const Manifest& manifest,
    std::vector<PhaseTiming>& setup_phases) {
  std::string warm_image;
  for (const auto& frame : manifest.frames) {
    if (!frame.tiles.empty() && !frame.tiles.front().layout_image.empty()) {
      warm_image = frame.tiles.front().layout_image;
      break;
    }
  }
  if (warm_image.empty()) {
    std::fprintf(stderr, "stream manifest has no layout image for warm load\n");
    std::exit(2);
  }

  std::vector<std::string> argv_storage = build_llama_args(args, warm_image, "<image>");
  std::vector<char*> argv_ptrs = mutable_argv(argv_storage);
  common_params params;
  const long parse_start_ms = now_ms();
  if (!common_params_parse(
          static_cast<int>(argv_ptrs.size()),
          argv_ptrs.data(),
          params,
          LLAMA_EXAMPLE_MTMD,
          show_usage)) {
    std::exit(2);
  }
  setup_phases.push_back({"L_DecoderRuntimeInit", parse_start_ms, now_ms()});
  const long load_start_ms = now_ms();
  auto ctx = std::make_unique<decode_context>(params);
  setup_phases.push_back({"L_DecoderLoad", load_start_ms, now_ms()});
  return ctx;
}

#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
std::unique_ptr<streamingvlm::hybrid_bridge::VisionEncoderSession> load_single_buffer_encoder_context(
    const Args& args,
    std::vector<PhaseTiming>& setup_phases) {
  if (args.encoder_path.empty()) {
    std::fprintf(stderr, "--encoder-path is required for QNN on-demand streaming\n");
    std::exit(2);
  }
  const long load_start_ms = now_ms();
  auto encoder = std::make_unique<streamingvlm::hybrid_bridge::VisionEncoderSession>(args.encoder_path);
  setup_phases.push_back({"L_VisionLoad", load_start_ms, now_ms()});
  if (!args.warmup_image_path.empty()) {
    (void)encoder->encode_with_optional_warmup({}, args.warmup_image_path);
  }
  return encoder;
}
#endif

#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
struct VisionKvSpan {
  int frame_index = -1;
  llama_pos begin = 0;
  llama_pos end = 0;

  llama_pos length() const {
    return end - begin;
  }
};

struct VisionPrefillCache {
  bool valid = false;
  std::vector<FrameRecord> frames;
  std::vector<int> frame_indices;
  std::vector<VisionKvSpan> frame_kv_spans;
  std::vector<int> open_user_frame_indices;
  std::vector<std::string> images;
  std::vector<common_chat_msg> chat_history;
  std::string open_user_content;
  bool open_user_prefix = false;
  std::string prefill_trace_body;
  std::string prefill_trace_flat;
  std::string prefill_trace_video_body;
  std::string prefill_trace_video_flat;
  std::string prefill_trace_tail_body;
  std::string prefill_trace_tail_flat;
  std::size_t prefill_trace_next_chunk_idx = 0;
  std::size_t prefill_trace_next_image_idx = 0;
  std::size_t prefill_trace_video_next_chunk_idx = 0;
  std::size_t prefill_trace_video_next_image_idx = 0;
  std::vector<uint8_t> state;
  std::vector<uint8_t> host_state;
  std::vector<uint8_t> video_prefix_state;
  llama_state_seq_flags state_flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
  llama_state_seq_flags host_state_flags = static_cast<llama_state_seq_flags>(0);
  llama_state_seq_flags video_prefix_state_flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
  llama_pos n_past = 0;
  llama_pos video_prefix_n_past = 0;
  llama_pos video_prefix_insert_pos = 0;
  bool video_prefix_insert_pos_valid = false;
  bool video_prefix_state_valid = false;
  int last_kv_reposition_compactions = 0;
  int last_kv_reposition_removed_frames = 0;
  llama_pos last_kv_reposition_removed_tokens = 0;
};

bool vision_prefill_cache_matches(const VisionPrefillCache& cache, const std::vector<FrameRecord>& frames) {
  return cache.valid && cache.frame_indices == frame_indices_for(frames);
}

size_t vision_prefill_cache_prefix_size(
    const VisionPrefillCache& cache,
    const std::vector<FrameRecord>& frames) {
  if (!cache.valid) {
    return 0;
  }
  const std::vector<int> target = frame_indices_for(frames);
  if (target.size() <= cache.frame_indices.size()) {
    return 0;
  }
  if (!std::equal(cache.frame_indices.begin(), cache.frame_indices.end(), target.begin())) {
    return 0;
  }
  return cache.frame_indices.size();
}

enum class VisionPrefillCacheBuildStatus {
  Ok,
  Failed,
  Preempted,
  Partial,
};

bool cache_preempt_requested(const std::atomic<int>* pending_prompt_jobs) {
  return pending_prompt_jobs != nullptr && pending_prompt_jobs->load(std::memory_order_acquire) > 0;
}

void record_cache_preempt(streamingvlm::hybrid_bridge::phase_recorder& phases) {
  const long t = now_ms();
  phases.row("VisionPrefillCachePreempt", t, t);
}

struct CachePreemptDecodeCallback {
  const std::atomic<int>* pending_prompt_jobs = nullptr;
  streamingvlm::hybrid_bridge::phase_recorder* phases = nullptr;
  bool* preempted = nullptr;
  const int32_t* completed_image_batches = nullptr;
  bool require_completed_batch_before_abort = false;
};

bool cache_preempt_decode_callback(void* user_data) {
  auto* callback = static_cast<CachePreemptDecodeCallback*>(user_data);
  if (callback == nullptr || !cache_preempt_requested(callback->pending_prompt_jobs)) {
    return false;
  }
  if (callback->require_completed_batch_before_abort &&
      callback->completed_image_batches != nullptr &&
      *callback->completed_image_batches <= 0) {
    return false;
  }
  if (callback->preempted != nullptr) {
    *callback->preempted = true;
  }
  if (callback->phases != nullptr) {
    record_cache_preempt(*callback->phases);
  }
  return true;
}

struct ImagePrefillBatchProgress {
  streamingvlm::hybrid_bridge::phase_recorder* phases = nullptr;
  int32_t* completed_image_batches = nullptr;
};

void image_prefill_batch_progress_callback(
    int32_t /*batch_idx*/,
    int32_t /*n_batches*/,
    int32_t /*n_tokens_batch*/,
    int64_t start_ms,
    int64_t end_ms,
    void* user_data) {
  auto* progress = static_cast<ImagePrefillBatchProgress*>(user_data);
  if (progress == nullptr) {
    return;
  }
  if (progress->completed_image_batches != nullptr) {
    ++*progress->completed_image_batches;
  }
  if (progress->phases != nullptr && end_ms > start_ms) {
    progress->phases->row("VisionPrefillImagePrefillBatch", start_ms, end_ms);
  }
}

const char* cache_build_status_detail(VisionPrefillCacheBuildStatus status) {
  switch (status) {
    case VisionPrefillCacheBuildStatus::Ok:
      return "ok";
    case VisionPrefillCacheBuildStatus::Preempted:
      return "preempted";
    case VisionPrefillCacheBuildStatus::Partial:
      return "partial";
    case VisionPrefillCacheBuildStatus::Failed:
    default:
      return "miss";
  }
}

std::string format_user_message_for_current_history(decode_context& ctx, const std::string& content) {
  common_chat_msg msg;
  msg.role = "user";
  msg.content = content;
  return common_chat_format_single(ctx.tmpls.get(), ctx.chat_history, msg, true, ctx.use_jinja);
}

void append_user_message_to_history(decode_context& ctx, const std::string& content) {
  common_chat_msg msg;
  msg.role = "user";
  msg.content = content;
  ctx.chat_history.push_back(std::move(msg));
}

void append_assistant_message_to_history(decode_context& ctx, const std::string& content) {
  common_chat_msg msg;
  msg.role = "assistant";
  msg.content = content;
  ctx.chat_history.push_back(std::move(msg));
}

bool split_formatted_at_question_sentinel(
    const std::string& formatted,
    std::string* prefix,
    std::string* suffix) {
  const size_t pos = formatted.find(SVLM_QUESTION_SENTINEL);
  if (pos == std::string::npos) {
    return false;
  }
  if (prefix != nullptr) {
    *prefix = formatted.substr(0, pos);
  }
  if (suffix != nullptr) {
    *suffix = formatted.substr(pos + std::strlen(SVLM_QUESTION_SENTINEL));
  }
  return true;
}

bool build_formatted_vision_cache_prefix(
    decode_context& ctx,
    const std::string& open_user_content,
    std::string& out) {
  const std::string content = open_user_content + SVLM_QUESTION_SENTINEL;
  return split_formatted_at_question_sentinel(format_user_message_for_current_history(ctx, content), &out, nullptr);
}

bool build_formatted_incremental_vision_cache_append(
    const std::vector<FrameRecord>& frames,
    std::string& out) {
  if (frames.empty()) {
    return false;
  }
  out.clear();
  for (const FrameRecord& frame : frames) {
    out += build_stream_frame_prompt_line(frame);
  }
  return true;
}

bool build_formatted_question_suffix(
    decode_context& ctx,
    const std::string& open_user_content,
    const std::string& raw_prompt,
    std::string& out) {
  const std::string content = open_user_content + SVLM_QUESTION_SENTINEL + raw_prompt;
  return split_formatted_at_question_sentinel(format_user_message_for_current_history(ctx, content), nullptr, &out);
}

struct RenderedPrefillTrace {
  std::string body;
  std::string flat;
  std::size_t next_chunk_idx = 0;
  std::size_t next_image_idx = 0;
};

RenderedPrefillTrace render_prefill_trace_for_chunks(
    decode_context& ctx,
    mtmd::input_chunks& chunks,
    std::size_t chunk_idx_offset = 0,
    std::size_t image_idx_offset = 0,
    const std::vector<std::size_t>* visible_image_tokens = nullptr,
    std::size_t max_committed_chunks = std::numeric_limits<std::size_t>::max()) {
  RenderedPrefillTrace rendered;
  rendered.next_chunk_idx = chunk_idx_offset;
  rendered.next_image_idx = image_idx_offset;
  const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
  const size_t n_visible_chunks = std::min(n_chunks, max_committed_chunks);
  std::size_t image_visible_idx = 0;

  for (size_t ci = 0; ci < n_visible_chunks; ++ci) {
    const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks.ptr.get(), ci);
    const auto ctype = mtmd_input_chunk_get_type(ch);
    if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t nt = 0;
      const llama_token* toks = mtmd_input_chunk_get_tokens_text(ch, &nt);
      rendered.body += "## CHUNK " + std::to_string(rendered.next_chunk_idx) +
                       " TEXT n_tokens=" + std::to_string(nt) + "\n";
      for (size_t ti = 0; ti < nt; ++ti) {
        const std::string piece = common_token_to_piece(ctx.lctx, toks[ti], true);
        const std::string esc = streamingvlm::hybrid_bridge::inference_trace_collector::escape_piece_str(piece);
        const std::string line = std::to_string(static_cast<long long>(toks[ti])) + "\t" + esc + "\n";
        rendered.body += line;
        rendered.flat += line;
      }
      ++rendered.next_chunk_idx;
    } else if (ctype == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      const std::size_t nominal_tokens = mtmd_input_chunk_get_n_tokens(ch);
      std::size_t visible_tokens = nominal_tokens;
      if (visible_image_tokens != nullptr) {
        if (image_visible_idx >= visible_image_tokens->size()) {
          break;
        }
        visible_tokens = std::min<std::size_t>((*visible_image_tokens)[image_visible_idx], nominal_tokens);
      }
      ++image_visible_idx;
      if (visible_tokens == 0) {
        break;
      }
      rendered.body += "## CHUNK " + std::to_string(rendered.next_chunk_idx) +
                       " IMAGE image_index=" + std::to_string(rendered.next_image_idx + 1) +
                       " n_placeholder_tokens=" + std::to_string(visible_tokens);
      if (visible_tokens != nominal_tokens) {
        rendered.body += " nominal_placeholder_tokens=" + std::to_string(nominal_tokens);
      }
      const char* cid = mtmd_input_chunk_get_id(ch);
      if (cid != nullptr && cid[0]) {
        rendered.body += std::string(" mtmd_chunk_id=") + cid;
      }
      rendered.body += "\n";
      for (std::size_t slot_i = 0; slot_i < visible_tokens; ++slot_i) {
        const std::string piece =
            streamingvlm::hybrid_bridge::inference_trace_collector::vision_slot_piece(slot_i + 1);
        const std::string line = std::string("-1\t") + piece + "\n";
        rendered.body += line;
        rendered.flat += line;
      }
      rendered.body += "# (each slot: projected vision embedding into decoder KV; not a BPE vocab id)\n";
      ++rendered.next_chunk_idx;
      ++rendered.next_image_idx;
    }
  }
  return rendered;
}

void append_rendered_trace_to_cache(VisionPrefillCache& cache, const RenderedPrefillTrace& trace) {
  cache.prefill_trace_body += trace.body;
  cache.prefill_trace_flat += trace.flat;
  cache.prefill_trace_next_chunk_idx = trace.next_chunk_idx;
  cache.prefill_trace_next_image_idx = trace.next_image_idx;
}

std::size_t count_trace_chunks(const std::string& body) {
  std::size_t count = 0;
  std::istringstream in(body);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("## CHUNK ", 0) == 0) {
      ++count;
    }
  }
  return count;
}

std::size_t count_trace_images(const std::string& body) {
  std::size_t count = 0;
  std::istringstream in(body);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("## CHUNK ", 0) == 0 && line.find(" IMAGE image_index=") != std::string::npos) {
      ++count;
    }
  }
  return count;
}

std::string replace_trace_number_after(std::string line, const std::string& marker, std::size_t value) {
  const size_t marker_pos = line.find(marker);
  if (marker_pos == std::string::npos) {
    return line;
  }
  const size_t value_begin = marker_pos + marker.size();
  size_t value_end = value_begin;
  while (value_end < line.size() && std::isdigit(static_cast<unsigned char>(line[value_end]))) {
    ++value_end;
  }
  line.replace(value_begin, value_end - value_begin, std::to_string(value));
  return line;
}

std::string renumber_trace_body(
    const std::string& body,
    std::size_t first_chunk_idx,
    std::size_t first_image_idx) {
  std::ostringstream out;
  std::istringstream in(body);
  std::string line;
  std::size_t next_chunk = first_chunk_idx;
  std::size_t next_image = first_image_idx;
  while (std::getline(in, line)) {
    if (line.rfind("## CHUNK ", 0) == 0) {
      line = replace_trace_number_after(std::move(line), "## CHUNK ", next_chunk++);
      if (line.find(" IMAGE image_index=") != std::string::npos) {
        line = replace_trace_number_after(std::move(line), " IMAGE image_index=", next_image + 1);
        ++next_image;
      }
    }
    out << line << "\n";
  }
  return out.str();
}

void rebuild_prefill_trace_from_video_and_tail(VisionPrefillCache& cache) {
  const std::string tail_body = renumber_trace_body(
      cache.prefill_trace_tail_body,
      cache.prefill_trace_video_next_chunk_idx,
      cache.prefill_trace_video_next_image_idx);
  cache.prefill_trace_body = cache.prefill_trace_video_body + tail_body;
  cache.prefill_trace_flat = cache.prefill_trace_video_flat + cache.prefill_trace_tail_flat;
  cache.prefill_trace_next_chunk_idx =
      cache.prefill_trace_video_next_chunk_idx + count_trace_chunks(cache.prefill_trace_tail_body);
  cache.prefill_trace_next_image_idx =
      cache.prefill_trace_video_next_image_idx + count_trace_images(cache.prefill_trace_tail_body);
}

void append_rendered_video_trace_to_cache(VisionPrefillCache& cache, const RenderedPrefillTrace& trace) {
  cache.prefill_trace_video_body += trace.body;
  cache.prefill_trace_video_flat += trace.flat;
  cache.prefill_trace_video_next_chunk_idx = trace.next_chunk_idx;
  cache.prefill_trace_video_next_image_idx = trace.next_image_idx;
  rebuild_prefill_trace_from_video_and_tail(cache);
}

void set_open_video_trace_from_combined(VisionPrefillCache& cache) {
  cache.prefill_trace_video_body = cache.prefill_trace_body;
  cache.prefill_trace_video_flat = cache.prefill_trace_flat;
  cache.prefill_trace_video_next_chunk_idx = cache.prefill_trace_next_chunk_idx;
  cache.prefill_trace_video_next_image_idx = cache.prefill_trace_next_image_idx;
  cache.prefill_trace_tail_body.clear();
  cache.prefill_trace_tail_flat.clear();
}

llama_pos count_chunk_positions(mtmd::input_chunks& chunks) {
  llama_pos total = 0;
  const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n_tokens = 0;
      (void)mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
      total += static_cast<llama_pos>(n_tokens);
    } else if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      total += static_cast<llama_pos>(mtmd_input_chunk_get_n_tokens(chunk));
    }
  }
  return total;
}

bool eval_streaming_chunks_with_external_embedding(
    decode_context& ctx,
    mtmd::input_chunks& chunks,
    streamingvlm::hybrid_bridge::EmbeddingFile* embedding,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    llama_seq_id seq_id,
    bool logits_last,
    const char* text_phase_name,
    const char* image_phase_name,
    const char* mmproj_phase_name,
    bool require_image) {
  const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
  bool used_image = false;
  const int32_t decoder_embedding_size = llama_model_n_embd_inp(ctx.model);
  std::optional<embedding_cursor> embeddings;
  if (embedding != nullptr) {
    embeddings.emplace(*embedding);
  }

  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const bool chunk_logits_last = logits_last && i == n_chunks - 1;
    llama_pos new_n_past = ctx.n_past;
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      if (!embeddings.has_value()) {
        LOG_ERR("cached suffix unexpectedly contains an image chunk\n");
        return false;
      }
      const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
      const int32_t n_embd = decoder_embedding_size;
      const embedding_cursor::embedding_slice slice = embeddings->next_slice_for_chunk(n_tokens, n_embd);
      float* image_slice = slice.data;
      float* image_embedding = image_slice;
      if (slice.feature_tokens != n_tokens || slice.feature_dim != n_embd) {
        const long mmproj_start_ms = now_ms();
        if (mtmd_project_features(
                ctx.ctx_vision.get(),
                image_slice,
                static_cast<int32_t>(slice.feature_tokens),
                slice.feature_dim) != 0) {
          LOG_ERR("failed to project cached vision features with mmproj\n");
          return false;
        }
        phases.row(mmproj_phase_name, mmproj_start_ms, now_ms());
        image_embedding = mtmd_get_output_embd(ctx.ctx_vision.get());
      }
      if (image_embedding == nullptr) {
        LOG_ERR("mmproj output is null for cached vision chunk\n");
        return false;
      }
      std::vector<float> image_embedding_copy(
          image_embedding,
          image_embedding + static_cast<size_t>(n_tokens) * decoder_embedding_size);
      const long image_prefill_start_ms = now_ms();
      if (mtmd_helper_decode_image_chunk(
              ctx.ctx_vision.get(),
              ctx.lctx,
              chunk,
              image_embedding_copy.data(),
              ctx.n_past,
              seq_id,
              ctx.n_batch,
              &new_n_past) != 0) {
        LOG_ERR("failed to decode cached external image embedding\n");
        return false;
      }
      llama_synchronize(ctx.lctx);
      phases.row(image_phase_name, image_prefill_start_ms, now_ms());
      used_image = true;
    } else {
      const long text_prefill_start_ms = now_ms();
      if (mtmd_helper_eval_chunk_single(
              ctx.ctx_vision.get(),
              ctx.lctx,
              chunk,
              ctx.n_past,
              seq_id,
              ctx.n_batch,
              chunk_logits_last,
              &new_n_past) != 0) {
        LOG_ERR("failed to eval cached text chunk\n");
        return false;
      }
      llama_synchronize(ctx.lctx);
      phases.row(text_phase_name, text_prefill_start_ms, now_ms());
    }
    ctx.n_past = new_n_past;
  }

  if (embeddings.has_value()) {
    embeddings->finish();
  }
  if (require_image && !used_image) {
    LOG_ERR("vision prefill cache prefix did not produce an image chunk\n");
    return false;
  }
  return true;
}

bool eval_streaming_chunks_with_on_demand_vision(
    decode_context& ctx,
    mtmd::input_chunks& chunks,
    streamingvlm::hybrid_bridge::VisionEncoderSession& encoder,
    const std::vector<std::string>& bins,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    llama_seq_id seq_id,
    bool logits_last,
    const char* text_phase_name,
    const char* image_load_phase_name,
    const char* vision_phase_name,
    const char* image_phase_name,
    const char* mmproj_phase_name,
    bool require_image,
    int32_t image_prefill_batch_size,
    bool allow_partial_image_commit = false,
    bool* partial_image_committed = nullptr,
    const std::atomic<int>* pending_prompt_jobs = nullptr,
    bool* preempted = nullptr,
    std::vector<std::size_t>* committed_image_tokens = nullptr,
    std::vector<streamingvlm::hybrid_bridge::KvTokenRange>* committed_image_kv_ranges = nullptr,
    std::size_t* committed_chunk_count = nullptr) {
  const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
  bool used_image = false;
  size_t image_chunk_idx = 0;
  bool llm_state_mutated = false;
  const int32_t decoder_embedding_size = llama_model_n_embd_inp(ctx.model);
  if (partial_image_committed != nullptr) {
    *partial_image_committed = false;
  }
  if (committed_image_tokens != nullptr) {
    committed_image_tokens->clear();
  }
  if (committed_image_kv_ranges != nullptr) {
    committed_image_kv_ranges->clear();
  }
  if (committed_chunk_count != nullptr) {
    *committed_chunk_count = 0;
  }
  auto mark_committed_chunk = [&](std::size_t count) {
    if (committed_chunk_count != nullptr) {
      *committed_chunk_count = std::max(*committed_chunk_count, count);
    }
  };
  auto record_committed_image_tokens = [&](llama_pos before, llama_pos after) {
    if (after <= before) {
      return;
    }
    if (committed_image_tokens != nullptr) {
      committed_image_tokens->push_back(static_cast<std::size_t>(after - before));
    }
    if (committed_image_kv_ranges != nullptr) {
      committed_image_kv_ranges->push_back(
          streamingvlm::hybrid_bridge::KvTokenRange{before, after});
    }
  };
  auto preempt = [&]() {
    if (!cache_preempt_requested(pending_prompt_jobs)) {
      return false;
    }
    if (preempted != nullptr) {
      *preempted = true;
    }
    record_cache_preempt(phases);
    return true;
  };
  auto commit_partial_cache_preempt = [&]() {
    if (!allow_partial_image_commit || !llm_state_mutated || !cache_preempt_requested(pending_prompt_jobs)) {
      return false;
    }
    if (preempted != nullptr) {
      *preempted = true;
    }
    if (partial_image_committed != nullptr) {
      *partial_image_committed = true;
    }
    record_cache_preempt(phases);
    return true;
  };
  auto next_chunk_is_image = [&](size_t chunk_idx) {
    if (chunk_idx + 1 >= n_chunks) {
      return false;
    }
    const mtmd_input_chunk* next_chunk = mtmd_input_chunks_get(chunks.ptr.get(), chunk_idx + 1);
    return mtmd_input_chunk_get_type(next_chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE;
  };
  auto eval_text_chunk = [&](const mtmd_input_chunk* text_chunk, bool chunk_logits_last) {
    llama_pos text_new_n_past = ctx.n_past;
    const long text_prefill_start_ms = now_ms();
    if (mtmd_helper_eval_chunk_single(
            ctx.ctx_vision.get(),
            ctx.lctx,
            text_chunk,
            ctx.n_past,
            seq_id,
            ctx.n_batch,
            chunk_logits_last,
            &text_new_n_past) != 0) {
      LOG_ERR("failed to eval cached text chunk\n");
      return false;
    }
    llama_synchronize(ctx.lctx);
    phases.row(text_phase_name, text_prefill_start_ms, now_ms());
    ctx.n_past = text_new_n_past;
    llm_state_mutated = true;
    return true;
  };
  auto drain_text_chunks_after_partial_image = [&](size_t start_idx) {
    for (size_t drain_idx = start_idx; drain_idx < n_chunks; ++drain_idx) {
      const mtmd_input_chunk* drain_chunk = mtmd_input_chunks_get(chunks.ptr.get(), drain_idx);
      if (mtmd_input_chunk_get_type(drain_chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        break;
      }
      if (!eval_text_chunk(drain_chunk, logits_last && drain_idx == n_chunks - 1)) {
        return false;
      }
      mark_committed_chunk(drain_idx + 1);
    }
    return true;
  };

  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const bool is_image_chunk = mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE;
    if (!(allow_partial_image_commit && is_image_chunk)) {
      if (commit_partial_cache_preempt()) {
        return false;
      }
      if (preempt()) {
        return false;
      }
    }
    const bool chunk_logits_last = logits_last && i == n_chunks - 1;
    llama_pos new_n_past = ctx.n_past;
    if (is_image_chunk) {
      if (image_chunk_idx >= bins.size()) {
        LOG_ERR("tokenized prefix has more image chunks than cached vision bins\n");
        return false;
      }

      const long vision_start_ms = now_ms();
      auto vision = encoder.encode({bins[image_chunk_idx]});
      long cursor_ms = vision_start_ms;
      const long image_load_ms = vision.image_load_end_ms - vision.image_load_start_ms;
      if (image_load_ms > 0) {
        phases.row(image_load_phase_name, cursor_ms, cursor_ms + image_load_ms);
        cursor_ms += image_load_ms;
      }
      for (const auto& range : vision.encode_ranges) {
        const long encode_ms = range.second - range.first;
        if (encode_ms > 0) {
          phases.row(vision_phase_name, cursor_ms, cursor_ms + encode_ms);
          cursor_ms += encode_ms;
        }
      }
      if (!allow_partial_image_commit && preempt()) {
        return false;
      }

      streamingvlm::hybrid_bridge::EmbeddingFile embedding;
      embedding.shape = vision.output_shape;
      embedding.values = std::move(vision.values);
      embedding_cursor embeddings(embedding);

      const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
      const int32_t n_embd = decoder_embedding_size;
      const embedding_cursor::embedding_slice slice = embeddings.next_slice_for_chunk(n_tokens, n_embd);
      float* image_slice = slice.data;
      float* image_embedding = image_slice;
      if (slice.feature_tokens != n_tokens || slice.feature_dim != n_embd) {
        if (!allow_partial_image_commit && preempt()) {
          return false;
        }
        const long mmproj_start_ms = now_ms();
        if (mtmd_project_features(
                ctx.ctx_vision.get(),
                image_slice,
                static_cast<int32_t>(slice.feature_tokens),
                slice.feature_dim) != 0) {
          LOG_ERR("failed to project on-demand vision features with mmproj\n");
          return false;
        }
        phases.row(mmproj_phase_name, mmproj_start_ms, now_ms());
        image_embedding = mtmd_get_output_embd(ctx.ctx_vision.get());
      }
      if (image_embedding == nullptr) {
        LOG_ERR("mmproj output is null for on-demand vision chunk\n");
        return false;
      }
      std::vector<float> image_embedding_copy(
          image_embedding,
          image_embedding + static_cast<size_t>(n_tokens) * decoder_embedding_size);
      if (!allow_partial_image_commit && preempt()) {
        return false;
      }
      const long image_prefill_start_ms = now_ms();
      const llama_pos image_n_past_before = ctx.n_past;
      const int32_t preemptible_image_batch =
          std::min<int32_t>(ctx.n_batch, image_prefill_batch_size);
      int32_t completed_image_batches = 0;
      CachePreemptDecodeCallback decode_preempt{
          pending_prompt_jobs,
          &phases,
          preempted,
          &completed_image_batches,
          allow_partial_image_commit,
      };
      ImagePrefillBatchProgress image_prefill_progress{
          &phases,
          &completed_image_batches,
      };
      const int32_t image_decode_ret = mtmd_helper_decode_image_chunk_with_abort_and_progress(
              ctx.ctx_vision.get(),
              ctx.lctx,
              chunk,
              image_embedding_copy.data(),
              ctx.n_past,
              seq_id,
              preemptible_image_batch,
              &new_n_past,
              cache_preempt_decode_callback,
              &decode_preempt,
              image_prefill_batch_progress_callback,
              &image_prefill_progress);
      llama_synchronize(ctx.lctx);
      (void) image_prefill_start_ms;
      (void) image_phase_name;
      if (image_decode_ret == 2) {
        if (preempted != nullptr) {
          *preempted = true;
        }
        if (allow_partial_image_commit && new_n_past > image_n_past_before) {
          ctx.n_past = new_n_past;
          llm_state_mutated = true;
          record_committed_image_tokens(image_n_past_before, new_n_past);
          mark_committed_chunk(i + 1);
          if (!drain_text_chunks_after_partial_image(i + 1)) {
            return false;
          }
          if (partial_image_committed != nullptr) {
            *partial_image_committed = true;
          }
        }
        return false;
      }
      if (image_decode_ret != 0) {
        LOG_ERR("failed to decode on-demand vision image embedding\n");
        return false;
      }
      ctx.n_past = new_n_past;
      llm_state_mutated = true;
      record_committed_image_tokens(image_n_past_before, new_n_past);
      mark_committed_chunk(i + 1);
      embeddings.finish();
      used_image = true;
      ++image_chunk_idx;
      if (commit_partial_cache_preempt()) {
        return false;
      }
      if (!allow_partial_image_commit && preempt()) {
        return false;
      }
    } else {
      if (commit_partial_cache_preempt()) {
        return false;
      }
      if (preempt()) {
        return false;
      }
      if (!eval_text_chunk(chunk, chunk_logits_last)) {
        return false;
      }
      new_n_past = ctx.n_past;
      mark_committed_chunk(i + 1);
      if (allow_partial_image_commit) {
        if (!next_chunk_is_image(i) && commit_partial_cache_preempt()) {
          return false;
        }
      } else if (preempt()) {
          return false;
      }
    }
    ctx.n_past = new_n_past;
  }

  if (image_chunk_idx != bins.size()) {
    LOG_ERR(
        "tokenized prefix used %zu image chunks but cache build has %zu vision bins\n",
        image_chunk_idx,
        bins.size());
    return false;
  }
  if (require_image && !used_image) {
    LOG_ERR("vision prefill cache prefix did not produce an image chunk\n");
    return false;
  }
  return true;
}

bool tokenize_formatted_text(
    decode_context& ctx,
    const std::string& formatted,
    mtmd::bitmaps& bitmaps,
    bool add_special,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    const char* phase_name,
    mtmd::input_chunks& chunks) {
  mtmd_input_text text{formatted.c_str(), add_special, true};
  auto bitmaps_c_ptr = bitmaps.c_ptr();
  const long tokenize_start_ms = now_ms();
  const int32_t tokenize_res = mtmd_tokenize(
      ctx.ctx_vision.get(),
      chunks.ptr.get(),
      &text,
      bitmaps_c_ptr.data(),
      bitmaps_c_ptr.size());
  phases.row(phase_name, tokenize_start_ms, now_ms());
  if (tokenize_res != 0) {
    LOG_ERR("mtmd_tokenize failed for vision prefill cache path: %d\n", tokenize_res);
    return false;
  }
  return true;
}

bool load_layout_bitmaps(
    decode_context& ctx,
    const std::vector<std::string>& images,
    mtmd::bitmaps& bitmaps) {
  for (const auto& image : images) {
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), image.c_str()));
    if (!bmp.ptr) {
      LOG_ERR("failed to load image for vision prefill cache layout: %s\n", image.c_str());
      return false;
    }
    bitmaps.entries.push_back(std::move(bmp));
  }
  return true;
}

bool eval_cached_text_segment(
    decode_context& ctx,
    const std::string& formatted,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    const char* tokenize_phase,
    const char* eval_phase,
    bool add_special = false,
    bool logits_last = false) {
  if (formatted.empty()) {
    return true;
  }
  mtmd::bitmaps empty_bitmaps;
  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  if (!tokenize_formatted_text(ctx, formatted, empty_bitmaps, add_special, phases, tokenize_phase, chunks)) {
    return false;
  }
  return eval_streaming_chunks_with_external_embedding(
      ctx,
      chunks,
      nullptr,
      phases,
      0,
      logits_last,
      eval_phase,
      "ImagePrefill",
      "Mmproj",
      false);
}

bool replay_cached_conversation_tail_after_video_prefix(
    decode_context& ctx,
    const std::vector<common_chat_msg>& updated_history,
    const std::vector<FrameRecord>& target_frames,
    streamingvlm::hybrid_bridge::phase_recorder& phases) {
  if (updated_history.empty()) {
    return true;
  }
  if (updated_history.front().role != "user") {
    LOG_ERR("cannot replay cached tail: first cached message is not a user message\n");
    return false;
  }

  ctx.chat_history.clear();
  const std::string open_user_content = build_stream_video_prompt_prefix(target_frames);
  const std::string first_question = strip_stream_video_prompt_prefix(updated_history.front().content);
  std::string first_suffix;
  if (!build_formatted_question_suffix(ctx, open_user_content, first_question, first_suffix)) {
    LOG_ERR("failed to format first cached user suffix while replaying closed video-prefix tail\n");
    return false;
  }
  if (!eval_cached_text_segment(
          ctx,
          first_suffix,
          phases,
          "VisionPrefillTailReplayTokenize",
          "VisionPrefillTailReplayT_Prefill")) {
    return false;
  }
  append_user_message_to_history(ctx, updated_history.front().content);

  for (size_t i = 1; i < updated_history.size(); ++i) {
    const common_chat_msg& msg = updated_history[i];
    if (msg.role == "assistant") {
      if (!eval_cached_text_segment(
              ctx,
              msg.content,
              phases,
              "VisionPrefillTailReplayTokenize",
              "VisionPrefillTailReplayT_Prefill")) {
        return false;
      }
      append_assistant_message_to_history(ctx, msg.content);
    } else if (msg.role == "user") {
      const std::string formatted = format_user_message_for_current_history(ctx, msg.content);
      if (!eval_cached_text_segment(
              ctx,
              formatted,
              phases,
              "VisionPrefillTailReplayTokenize",
              "VisionPrefillTailReplayT_Prefill")) {
        return false;
      }
      append_user_message_to_history(ctx, msg.content);
    } else {
      const std::string formatted =
          common_chat_format_single(ctx.tmpls.get(), ctx.chat_history, msg, false, ctx.use_jinja);
      if (!eval_cached_text_segment(
              ctx,
              formatted,
              phases,
              "VisionPrefillTailReplayTokenize",
              "VisionPrefillTailReplayT_Prefill")) {
        return false;
      }
      ctx.chat_history.push_back(msg);
    }
  }
  return true;
}

bool save_vision_prefill_cache_state_blob(
    decode_context& ctx,
    std::vector<uint8_t>& state,
    llama_state_seq_flags flags,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    const char* phase_name) {
  const long save_start_ms = now_ms();
  const size_t state_size = llama_state_seq_get_size_ext(ctx.lctx, 0, flags);
  if (state_size == 0) {
    LOG_ERR("failed to determine vision prefill cache state size\n");
    return false;
  }
  state.assign(state_size, 0);
  const size_t copied = llama_state_seq_get_data_ext(
      ctx.lctx,
      state.data(),
      state.size(),
      0,
      flags);
  phases.row(phase_name, save_start_ms, now_ms());
  if (copied != state.size()) {
    LOG_ERR("failed to save vision prefill cache state: copied %zu of %zu bytes\n", copied, state.size());
    state.clear();
    return false;
  }
  return true;
}

bool save_vision_prefill_cache_state(
    decode_context& ctx,
    VisionPrefillCache& cache,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    bool live_only = false) {
  if (live_only) {
    const long t = now_ms();
    phases.row("VisionPrefillCacheSave", t, t);
    cache.state.clear();
    cache.host_state.clear();
    cache.state_flags = static_cast<llama_state_seq_flags>(0);
    cache.host_state_flags = static_cast<llama_state_seq_flags>(0);
    cache.n_past = ctx.n_past;
    cache.chat_history = ctx.chat_history;
    return true;
  }

  cache.state_flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
  if (!save_vision_prefill_cache_state_blob(
          ctx,
          cache.state,
          cache.state_flags,
          phases,
          "VisionPrefillCacheSave")) {
    cache.state_flags = static_cast<llama_state_seq_flags>(0);
    if (!save_vision_prefill_cache_state_blob(
            ctx,
            cache.state,
            cache.state_flags,
            phases,
            "VisionPrefillCacheSave")) {
      return false;
    }
  }
  cache.host_state_flags = static_cast<llama_state_seq_flags>(0);
  if (cache.state_flags == cache.host_state_flags) {
    cache.host_state = cache.state;
  } else if (!save_vision_prefill_cache_state_blob(
                 ctx,
                 cache.host_state,
                 cache.host_state_flags,
                 phases,
                 "VisionPrefillCacheHostSave")) {
    return false;
  }
  cache.n_past = ctx.n_past;
  cache.chat_history = ctx.chat_history;
  return true;
}

bool save_vision_prefill_video_prefix_state(
    decode_context& ctx,
    VisionPrefillCache& cache,
    streamingvlm::hybrid_bridge::phase_recorder& phases) {
  cache.video_prefix_state_flags = static_cast<llama_state_seq_flags>(0);
  if (!save_vision_prefill_cache_state_blob(
          ctx,
          cache.video_prefix_state,
          cache.video_prefix_state_flags,
          phases,
          "VisionPrefillVideoPrefixSave")) {
    cache.video_prefix_state.clear();
    cache.video_prefix_state_valid = false;
    return false;
  }
  cache.video_prefix_n_past = ctx.n_past;
  cache.video_prefix_insert_pos = ctx.n_past;
  cache.video_prefix_insert_pos_valid = true;
  cache.video_prefix_state_valid = true;
  return true;
}

bool restore_vision_prefill_video_prefix_state(
    decode_context& ctx,
    const VisionPrefillCache& cache,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    const char* phase_name = "VisionPrefillVideoPrefixRestore") {
  if (!cache.video_prefix_state_valid || cache.video_prefix_state.empty()) {
    return false;
  }
  reset_decode_context_for_singleton(ctx);
  const long restore_start_ms = now_ms();
  const size_t restored = llama_state_seq_set_data_ext(
      ctx.lctx,
      cache.video_prefix_state.data(),
      cache.video_prefix_state.size(),
      0,
      cache.video_prefix_state_flags);
  llama_synchronize(ctx.lctx);
  phases.row(phase_name, restore_start_ms, now_ms());
  if (restored != cache.video_prefix_state.size()) {
    LOG_ERR(
        "failed to restore vision video-prefix state: restored %zu of %zu bytes\n",
        restored,
        cache.video_prefix_state.size());
    return false;
  }
  ctx.n_past = cache.video_prefix_n_past;
  ctx.chat_history.clear();
  return true;
}

bool restore_vision_prefill_cache_state(
    decode_context& ctx,
    const VisionPrefillCache& cache,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    const char* phase_name = "VisionPrefillCacheRestore",
    bool prefer_host_state = false) {
  if (!cache.valid || (cache.state.empty() && cache.host_state.empty())) {
    if (!cache.valid) {
      return false;
    }
    const long t = now_ms();
    phases.row(phase_name, t, t);
    ctx.n_past = cache.n_past;
    ctx.chat_history = cache.chat_history;
    return true;
  }
  reset_decode_context_for_singleton(ctx);
  const long restore_start_ms = now_ms();
  const std::vector<uint8_t>& state =
      prefer_host_state && !cache.host_state.empty() ? cache.host_state : cache.state;
  const llama_state_seq_flags state_flags =
      prefer_host_state && !cache.host_state.empty() ? cache.host_state_flags : cache.state_flags;
  const size_t restored = llama_state_seq_set_data_ext(
      ctx.lctx,
      state.data(),
      state.size(),
      0,
      state_flags);
  llama_synchronize(ctx.lctx);
  phases.row(phase_name, restore_start_ms, now_ms());
  if (restored != state.size()) {
    LOG_ERR("failed to restore vision prefill cache state: restored %zu of %zu bytes\n", restored, state.size());
    return false;
  }
  ctx.n_past = cache.n_past;
  ctx.chat_history = cache.chat_history;
  return true;
}

bool rollback_vision_prefill_cache_build(
    decode_context& ctx,
    const VisionPrefillCache& cache,
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    bool record_preempt_phase) {
  if (record_preempt_phase) {
    record_cache_preempt(phases);
  }
  if (!cache.valid) {
    reset_decode_context_for_singleton(ctx);
    return true;
  }
  return restore_vision_prefill_cache_state(ctx, cache, phases, "VisionPrefillCacheRollback", true);
}

void refresh_vision_prefill_cache_frame_views(VisionPrefillCache& cache) {
  std::set<int> resident_frame_indices;
  for (const VisionKvSpan& span : cache.frame_kv_spans) {
    if (span.end > span.begin) {
      resident_frame_indices.insert(span.frame_index);
    }
  }
  if (!resident_frame_indices.empty()) {
    cache.frames = filter_frames_by_index_set(cache.frames, resident_frame_indices);
  } else {
    cache.frames.clear();
  }
  cache.frame_indices = frame_indices_for(cache.frames);
  cache.images = layout_images_for_frames(cache.frames);

  std::vector<int> open_indices;
  for (int index : cache.open_user_frame_indices) {
    if (resident_frame_indices.find(index) != resident_frame_indices.end()) {
      open_indices.push_back(index);
    }
  }
  cache.open_user_frame_indices = std::move(open_indices);
  if (cache.open_user_prefix) {
    cache.open_user_content =
        build_stream_video_prompt_prefix(filter_frames_by_index_list(cache.frames, cache.open_user_frame_indices));
  }
}

bool compact_vision_prefill_cache_frames(
    const Args& args,
    decode_context& ctx,
    VisionPrefillCache& cache,
    streamingvlm::hybrid_bridge::phase_recorder& phases) {
  cache.last_kv_reposition_compactions = 0;
  cache.last_kv_reposition_removed_frames = 0;
  cache.last_kv_reposition_removed_tokens = 0;
  if (args.kv_reposition_keep_latest_frames <= 0 || cache.frame_kv_spans.empty()) {
    return true;
  }

  std::set<int> keep_frames;
  for (auto it = cache.frame_kv_spans.rbegin(); it != cache.frame_kv_spans.rend(); ++it) {
    if (it->end <= it->begin) {
      continue;
    }
    keep_frames.insert(it->frame_index);
    if (static_cast<int>(keep_frames.size()) >= args.kv_reposition_keep_latest_frames) {
      break;
    }
  }

  std::vector<size_t> remove_indices;
  std::set<int> removed_frames;
  for (size_t i = 0; i < cache.frame_kv_spans.size(); ++i) {
    const VisionKvSpan& span = cache.frame_kv_spans[i];
    if (span.end > span.begin && keep_frames.find(span.frame_index) == keep_frames.end()) {
      remove_indices.push_back(i);
      removed_frames.insert(span.frame_index);
    }
  }
  if (remove_indices.empty()) {
    return true;
  }

  llama_pos sequence_end = ctx.n_past;
  llama_pos removed_tokens = 0;
  int compactions = 0;
  for (auto rit = remove_indices.rbegin(); rit != remove_indices.rend(); ++rit) {
    const size_t remove_idx = *rit;
    if (remove_idx >= cache.frame_kv_spans.size()) {
      continue;
    }
    const VisionKvSpan span = cache.frame_kv_spans[remove_idx];
    if (span.end <= span.begin) {
      continue;
    }
    streamingvlm::hybrid_bridge::KvTailCompactionPlan plan;
    std::string error;
    if (!streamingvlm::hybrid_bridge::build_tail_compaction_plan(
            streamingvlm::hybrid_bridge::KvTokenRange{span.begin, span.end},
            sequence_end,
            &plan,
            &error)) {
      LOG_ERR("failed to build KV reposition compaction plan: %s\n", error.c_str());
      return false;
    }
    const long compact_start_ms = now_ms();
    if (!streamingvlm::hybrid_bridge::apply_tail_compaction_plan(
            llama_get_memory(ctx.lctx),
            0,
            plan,
            &error)) {
      LOG_ERR("failed to apply KV reposition compaction plan: %s\n", error.c_str());
      return false;
    }
    llama_synchronize(ctx.lctx);
    phases.row("KVRepositionCompact", compact_start_ms, now_ms());

	    const llama_pos delta = plan.removed.length();
	    removed_tokens += delta;
	    sequence_end = plan.compacted_sequence_end;
	    ctx.n_past = sequence_end;
	    cache.n_past = sequence_end;
	    if (cache.video_prefix_insert_pos_valid) {
	      cache.video_prefix_insert_pos =
	          streamingvlm::hybrid_bridge::compacted_position_after(
	              streamingvlm::hybrid_bridge::KvTokenRange{span.begin, span.end},
	              cache.video_prefix_insert_pos);
	    }
	    ++compactions;

    for (size_t j = remove_idx + 1; j < cache.frame_kv_spans.size(); ++j) {
      cache.frame_kv_spans[j].begin -= delta;
      cache.frame_kv_spans[j].end -= delta;
    }
  }

  std::vector<bool> remove_mask(cache.frame_kv_spans.size(), false);
  for (size_t idx : remove_indices) {
    if (idx < remove_mask.size()) {
      remove_mask[idx] = true;
    }
  }
  std::vector<VisionKvSpan> kept_spans;
  kept_spans.reserve(cache.frame_kv_spans.size() - remove_indices.size());
  for (size_t i = 0; i < cache.frame_kv_spans.size(); ++i) {
    if (!remove_mask[i]) {
      kept_spans.push_back(cache.frame_kv_spans[i]);
    }
	  }
	  cache.frame_kv_spans = std::move(kept_spans);
	  refresh_vision_prefill_cache_frame_views(cache);
	  if (!cache.open_user_prefix && !cache.chat_history.empty()) {
	    (void)update_first_video_user_message(cache.chat_history, cache.frames);
	    ctx.chat_history = cache.chat_history;
	  }
	  cache.last_kv_reposition_compactions = compactions;
  cache.last_kv_reposition_removed_frames = static_cast<int>(removed_frames.size());
  cache.last_kv_reposition_removed_tokens = removed_tokens;
	  cache.prefill_trace_body +=
	      "## KV_REPOSITION_COMPACT removed_frames=" +
      std::to_string(cache.last_kv_reposition_removed_frames) +
      " removed_vision_tokens=" +
      std::to_string(static_cast<long long>(cache.last_kv_reposition_removed_tokens)) +
      " keep_latest_frames=" +
      std::to_string(args.kv_reposition_keep_latest_frames) + "\n";
	  return true;
	}

bool compact_unused_insert_gap(
    decode_context& ctx,
    llama_pos gap_begin,
    llama_pos gap_end,
    llama_pos expanded_sequence_end,
    streamingvlm::hybrid_bridge::phase_recorder& phases) {
  if (gap_end <= gap_begin) {
    return true;
  }
  streamingvlm::hybrid_bridge::KvTailCompactionPlan plan;
  std::string error;
  if (!streamingvlm::hybrid_bridge::build_tail_compaction_plan(
          streamingvlm::hybrid_bridge::KvTokenRange{gap_begin, gap_end},
          expanded_sequence_end,
          &plan,
          &error)) {
    LOG_ERR("failed to build unused insert-gap compaction plan: %s\n", error.c_str());
    return false;
  }
  const long compact_start_ms = now_ms();
  if (!streamingvlm::hybrid_bridge::apply_tail_compaction_plan(
          llama_get_memory(ctx.lctx),
          0,
          plan,
          &error)) {
    LOG_ERR("failed to compact unused insert gap: %s\n", error.c_str());
    return false;
  }
  llama_synchronize(ctx.lctx);
  phases.row("KVRepositionCompact", compact_start_ms, now_ms());
  return true;
}

VisionPrefillCacheBuildStatus build_vision_prefill_cache(
    const Args& args,
    decode_context& ctx,
    streamingvlm::hybrid_bridge::VisionEncoderSession& encoder,
    const std::vector<FrameRecord>& frames,
    int frame_idx,
    long origin_ms,
    VisionPrefillCache& cache,
    const std::atomic<int>* pending_prompt_jobs) {
  std::vector<FrameRecord> target_frames = frames;
  if (args.online_buffer && cache.valid && frames.size() == 1) {
    const int latest_frame_index = frames.back().index;
    if (std::find(cache.frame_indices.begin(), cache.frame_indices.end(), latest_frame_index) != cache.frame_indices.end()) {
      return VisionPrefillCacheBuildStatus::Ok;
    }
    target_frames = cache.frames;
    target_frames.push_back(frames.back());
  }

  const size_t cached_prefix_size = vision_prefill_cache_prefix_size(cache, target_frames);
  if (cached_prefix_size > 0 && target_frames.size() > cached_prefix_size + 1) {
    target_frames.resize(cached_prefix_size + 1);
  }

  VisionPrefillCache next_cache;
  next_cache.frames = target_frames;
  next_cache.frame_indices = frame_indices_for(target_frames);
  next_cache.images = layout_images_for_frames(frames);
  next_cache.last_kv_reposition_compactions = 0;
  next_cache.last_kv_reposition_removed_frames = 0;
  next_cache.last_kv_reposition_removed_tokens = 0;
  if (target_frames.empty() || next_cache.images.empty()) {
    LOG_ERR("cannot build vision prefill cache for frame %d: missing bins or layout images\n", frame_idx);
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Failed;
  }

  const std::string phase_path = "stream_vision_prefill_cache_" + std::to_string(frame_idx) + ".csv";
  streamingvlm::hybrid_bridge::phase_recorder cache_phases(
      phase_path,
      origin_ms,
      streamingvlm::hybrid_bridge::hybrid_decode_phase_description());
  const long build_start_ms = now_ms();

  next_cache.images = layout_images_for_frames(target_frames);

  if (vision_prefill_cache_matches(cache, target_frames)) {
    return VisionPrefillCacheBuildStatus::Ok;
  }
  if (cache_preempt_requested(pending_prompt_jobs)) {
    if (!rollback_vision_prefill_cache_build(ctx, cache, cache_phases, true)) {
      return VisionPrefillCacheBuildStatus::Failed;
    }
    return VisionPrefillCacheBuildStatus::Preempted;
  }

  const bool can_append_incrementally = cached_prefix_size > 0 && target_frames.size() > cached_prefix_size;
  bool use_incremental_append = false;
  std::string formatted_prefix;
  bool formatted_prefix_add_special = false;
  std::vector<std::string> bins;
  std::vector<std::string> images_to_load;
  std::vector<FrameRecord> frames_loaded_for_kv_span;
  bool insert_frame_into_closed_video_prefix = false;
  bool shift_tail_for_closed_video_prefix = false;
  bool closed_video_prefix_tail_shifted = false;
  streamingvlm::hybrid_bridge::KvTailInsertionPlan closed_video_prefix_insert_plan;
  llama_pos closed_video_prefix_original_sequence_end = 0;

  const bool can_insert_into_closed_video_prefix =
      can_append_incrementally && !cache.open_user_prefix && cache.video_prefix_insert_pos_valid;
  bool restored_for_incremental_append = false;
  if (can_insert_into_closed_video_prefix) {
    restored_for_incremental_append =
        restore_vision_prefill_cache_state(ctx, cache, cache_phases, "VisionPrefillCacheAppendRestore");
    if (restored_for_incremental_append) {
      closed_video_prefix_original_sequence_end = cache.n_past;
    }
  } else if (can_append_incrementally) {
    restored_for_incremental_append =
        restore_vision_prefill_cache_state(ctx, cache, cache_phases, "VisionPrefillCacheAppendRestore");
  }

  if (can_append_incrementally && restored_for_incremental_append) {
    auto append_begin = target_frames.begin() + static_cast<std::ptrdiff_t>(cached_prefix_size);
    std::vector<FrameRecord> append_frames(append_begin, append_begin + 1);
    frames_loaded_for_kv_span = append_frames;
    bins = bins_for_frames(append_frames);
    images_to_load = layout_images_for_frames(append_frames);
    next_cache.frame_kv_spans = cache.frame_kv_spans;
    next_cache.open_user_frame_indices = cache.open_user_frame_indices;
    if (!build_formatted_incremental_vision_cache_append(append_frames, formatted_prefix)) {
      LOG_ERR("failed to build incremental vision prefill cache append\n");
      cache = std::move(next_cache);
      return VisionPrefillCacheBuildStatus::Failed;
    }
    next_cache.open_user_content = cache.open_user_content + formatted_prefix;
    next_cache.open_user_prefix = true;
    next_cache.prefill_trace_body = cache.prefill_trace_body;
    next_cache.prefill_trace_flat = cache.prefill_trace_flat;
    next_cache.prefill_trace_video_body = cache.prefill_trace_video_body;
    next_cache.prefill_trace_video_flat = cache.prefill_trace_video_flat;
    next_cache.prefill_trace_tail_body = cache.prefill_trace_tail_body;
    next_cache.prefill_trace_tail_flat = cache.prefill_trace_tail_flat;
    next_cache.prefill_trace_next_chunk_idx = cache.prefill_trace_next_chunk_idx;
    next_cache.prefill_trace_next_image_idx = cache.prefill_trace_next_image_idx;
    next_cache.prefill_trace_video_next_chunk_idx = cache.prefill_trace_video_next_chunk_idx;
    next_cache.prefill_trace_video_next_image_idx = cache.prefill_trace_video_next_image_idx;
    next_cache.chat_history = cache.chat_history;
    next_cache.video_prefix_insert_pos = cache.video_prefix_insert_pos;
    next_cache.video_prefix_insert_pos_valid = cache.video_prefix_insert_pos_valid;
    next_cache.video_prefix_state = cache.video_prefix_state;
    next_cache.video_prefix_state_flags = cache.video_prefix_state_flags;
    next_cache.video_prefix_n_past = cache.video_prefix_n_past;
    next_cache.video_prefix_state_valid = cache.video_prefix_state_valid;
    for (const FrameRecord& frame : append_frames) {
      next_cache.open_user_frame_indices.push_back(frame.index);
    }
    if (!cache.open_user_prefix) {
      insert_frame_into_closed_video_prefix = true;
      shift_tail_for_closed_video_prefix = true;
      next_cache.open_user_content = formatted_prefix;
      next_cache.open_user_content.clear();
      next_cache.open_user_prefix = false;
      next_cache.open_user_frame_indices.clear();
      formatted_prefix_add_special = false;
    }
    use_incremental_append = true;
  } else {
    reset_decode_context_for_singleton(ctx);
    frames_loaded_for_kv_span = target_frames;
    bins = bins_for_frames(target_frames);
    images_to_load = next_cache.images;
    next_cache.open_user_content = build_stream_video_prompt_prefix(target_frames);
    next_cache.open_user_prefix = true;
    next_cache.open_user_frame_indices = frame_indices_for(target_frames);
    next_cache.video_prefix_insert_pos_valid = false;
    next_cache.video_prefix_insert_pos = 0;
    if (!build_formatted_vision_cache_prefix(ctx, next_cache.open_user_content, formatted_prefix)) {
      LOG_ERR("failed to split formatted vision prefill cache prefix\n");
      cache = std::move(next_cache);
      return VisionPrefillCacheBuildStatus::Failed;
    }
    formatted_prefix_add_special = ctx.chat_history.empty();
  }

  if (bins.empty() || images_to_load.empty()) {
    LOG_ERR(
        "cannot build vision prefill cache for frame %d: missing %s bins or layout images\n",
        frame_idx,
        use_incremental_append ? "incremental" : "full-history");
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Failed;
  }

  mtmd::bitmaps bitmaps;
  if (!load_layout_bitmaps(ctx, images_to_load, bitmaps)) {
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Failed;
  }
  if (cache_preempt_requested(pending_prompt_jobs)) {
    if (!rollback_vision_prefill_cache_build(ctx, cache, cache_phases, true)) {
      return VisionPrefillCacheBuildStatus::Failed;
    }
    return VisionPrefillCacheBuildStatus::Preempted;
  }

  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  if (!tokenize_formatted_text(
          ctx,
          formatted_prefix,
          bitmaps,
          formatted_prefix_add_special,
          cache_phases,
          "VisionPrefillLayoutTokenize",
          chunks)) {
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Failed;
  }
  if (cache_preempt_requested(pending_prompt_jobs)) {
    if (!rollback_vision_prefill_cache_build(ctx, cache, cache_phases, true)) {
      return VisionPrefillCacheBuildStatus::Failed;
    }
    return VisionPrefillCacheBuildStatus::Preempted;
  }

  if (shift_tail_for_closed_video_prefix) {
    llama_pos reserved_insert_len = 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    for (size_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
      const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), chunk_idx);
      reserved_insert_len += static_cast<llama_pos>(mtmd_input_chunk_get_n_tokens(chunk));
    }
    std::string error;
    if (!streamingvlm::hybrid_bridge::build_tail_insertion_plan(
            cache.video_prefix_insert_pos,
            reserved_insert_len,
            closed_video_prefix_original_sequence_end,
            &closed_video_prefix_insert_plan,
            &error)) {
      LOG_ERR("failed to build KV tail insertion plan: %s\n", error.c_str());
      cache = std::move(next_cache);
      return VisionPrefillCacheBuildStatus::Failed;
    }
    const long shift_start_ms = now_ms();
    if (!streamingvlm::hybrid_bridge::apply_tail_insertion_plan(
            llama_get_memory(ctx.lctx),
            0,
            closed_video_prefix_insert_plan,
            &error)) {
      LOG_ERR("failed to shift KV tail before video-prefix insert: %s\n", error.c_str());
      cache = std::move(next_cache);
      return VisionPrefillCacheBuildStatus::Failed;
    }
    llama_synchronize(ctx.lctx);
    cache_phases.row("KVRepositionTailShift", shift_start_ms, now_ms());
    ctx.n_past = closed_video_prefix_insert_plan.insert_pos;
    closed_video_prefix_tail_shifted = true;
  }

  bool preempted = false;
  bool partial_image_committed = false;
  std::vector<std::size_t> committed_image_tokens;
  std::vector<streamingvlm::hybrid_bridge::KvTokenRange> committed_image_kv_ranges;
  std::size_t committed_chunk_count = 0;
  const int32_t image_prefill_batch_size = std::max(1, args.ubatch_size);
  auto finish_closed_video_prefix_insert = [&]() {
    if (!insert_frame_into_closed_video_prefix) {
      return true;
    }
    if (!update_first_video_user_message(next_cache.chat_history, target_frames)) {
      next_cache.chat_history = cache.chat_history;
      if (!update_first_video_user_message(next_cache.chat_history, target_frames)) {
        LOG_ERR("failed to update cached first video user message before tail replay\n");
        return false;
      }
    }
    if (shift_tail_for_closed_video_prefix) {
      if (!closed_video_prefix_tail_shifted) {
        LOG_ERR("cannot finish closed video-prefix insert: KV tail was not shifted\n");
        return false;
      }
      const llama_pos inserted_len = ctx.n_past - closed_video_prefix_insert_plan.insert_pos;
      if (inserted_len < 0 || inserted_len > closed_video_prefix_insert_plan.insert_len) {
        LOG_ERR("invalid closed video-prefix insert length: %lld\n", static_cast<long long>(inserted_len));
        return false;
      }
      if (inserted_len < closed_video_prefix_insert_plan.insert_len) {
        const llama_pos gap_begin = closed_video_prefix_insert_plan.insert_pos + inserted_len;
        const llama_pos gap_end = closed_video_prefix_insert_plan.insert_pos + closed_video_prefix_insert_plan.insert_len;
        if (!compact_unused_insert_gap(
                ctx,
                gap_begin,
                gap_end,
                closed_video_prefix_insert_plan.expanded_sequence_end,
                cache_phases)) {
          return false;
        }
      }
      ctx.n_past = closed_video_prefix_original_sequence_end + inserted_len;
      next_cache.n_past = ctx.n_past;
      next_cache.video_prefix_insert_pos = closed_video_prefix_insert_plan.insert_pos + inserted_len;
      next_cache.video_prefix_insert_pos_valid = true;
      next_cache.video_prefix_state.clear();
      next_cache.video_prefix_state_valid = false;
    }
    if (!compact_vision_prefill_cache_frames(args, ctx, next_cache, cache_phases)) {
      return false;
    }
    next_cache.chat_history = ctx.chat_history;
    next_cache.n_past = ctx.n_past;
    return true;
  };
  if (!eval_streaming_chunks_with_on_demand_vision(
          ctx,
          chunks,
          encoder,
          bins,
          cache_phases,
          0,
          false,
          "VisionPrefillT_Prefill",
          "VisionPrefillImageLoad",
          "VisionPrefillV_Encode",
          "VisionPrefillImagePrefill",
          "VisionPrefillMmproj",
          true,
          image_prefill_batch_size,
          args.partial_vision_kv,
          &partial_image_committed,
          pending_prompt_jobs,
          &preempted,
          &committed_image_tokens,
          &committed_image_kv_ranges,
          &committed_chunk_count)) {
    if (preempted) {
      if (partial_image_committed) {
        const std::vector<int> image_frame_indices = image_frame_indices_for_frames(frames_loaded_for_kv_span);
        for (size_t span_i = 0; span_i < committed_image_kv_ranges.size(); ++span_i) {
          const int frame_index = span_i < image_frame_indices.size() ? image_frame_indices[span_i] : frame_idx;
          next_cache.frame_kv_spans.push_back(VisionKvSpan{
              frame_index,
              committed_image_kv_ranges[span_i].begin,
              committed_image_kv_ranges[span_i].end});
        }
        const RenderedPrefillTrace rendered = render_prefill_trace_for_chunks(
            ctx,
            chunks,
            insert_frame_into_closed_video_prefix
                ? next_cache.prefill_trace_video_next_chunk_idx
                : next_cache.prefill_trace_next_chunk_idx,
            insert_frame_into_closed_video_prefix
                ? next_cache.prefill_trace_video_next_image_idx
                : next_cache.prefill_trace_next_image_idx,
            &committed_image_tokens,
            committed_chunk_count);
        if (insert_frame_into_closed_video_prefix) {
          append_rendered_video_trace_to_cache(next_cache, rendered);
        } else {
          append_rendered_trace_to_cache(next_cache, rendered);
          if (next_cache.open_user_prefix) {
            set_open_video_trace_from_combined(next_cache);
          }
        }
        if (!finish_closed_video_prefix_insert()) {
          cache = std::move(next_cache);
          return VisionPrefillCacheBuildStatus::Failed;
        }
        if (next_cache.open_user_prefix) {
          next_cache.video_prefix_insert_pos = ctx.n_past;
          next_cache.video_prefix_insert_pos_valid = true;
        }
        if (!save_vision_prefill_cache_state(ctx, next_cache, cache_phases, args.partial_vision_kv)) {
          cache = std::move(next_cache);
          return VisionPrefillCacheBuildStatus::Failed;
        }
        next_cache.valid = true;
        cache_phases.row("VisionPrefillCachePartialCommit", build_start_ms, now_ms());
        cache = std::move(next_cache);
        return VisionPrefillCacheBuildStatus::Partial;
      }
      if (!rollback_vision_prefill_cache_build(ctx, cache, cache_phases, false)) {
        return VisionPrefillCacheBuildStatus::Failed;
      }
      return VisionPrefillCacheBuildStatus::Preempted;
    }
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Failed;
  }
  const std::vector<int> image_frame_indices = image_frame_indices_for_frames(frames_loaded_for_kv_span);
  for (size_t span_i = 0; span_i < committed_image_kv_ranges.size(); ++span_i) {
    const int committed_frame_index = span_i < image_frame_indices.size() ? image_frame_indices[span_i] : frame_idx;
    next_cache.frame_kv_spans.push_back(VisionKvSpan{
        committed_frame_index,
        committed_image_kv_ranges[span_i].begin,
        committed_image_kv_ranges[span_i].end});
  }
	  const RenderedPrefillTrace rendered = render_prefill_trace_for_chunks(
	      ctx,
	      chunks,
	      insert_frame_into_closed_video_prefix
	          ? next_cache.prefill_trace_video_next_chunk_idx
	          : next_cache.prefill_trace_next_chunk_idx,
	      insert_frame_into_closed_video_prefix
	          ? next_cache.prefill_trace_video_next_image_idx
	          : next_cache.prefill_trace_next_image_idx,
	      &committed_image_tokens,
	      committed_chunk_count);
	  if (insert_frame_into_closed_video_prefix) {
	    append_rendered_video_trace_to_cache(next_cache, rendered);
	  } else {
	    append_rendered_trace_to_cache(next_cache, rendered);
	    if (next_cache.open_user_prefix) {
	      set_open_video_trace_from_combined(next_cache);
	    }
	  }
  if (!finish_closed_video_prefix_insert()) {
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Failed;
  }
  if (cache_preempt_requested(pending_prompt_jobs) && !args.partial_vision_kv) {
    if (!rollback_vision_prefill_cache_build(ctx, cache, cache_phases, true)) {
      return VisionPrefillCacheBuildStatus::Failed;
    }
    return VisionPrefillCacheBuildStatus::Preempted;
  } else if (cache_preempt_requested(pending_prompt_jobs)) {
    record_cache_preempt(cache_phases);
  }

  if (!insert_frame_into_closed_video_prefix &&
      !compact_vision_prefill_cache_frames(args, ctx, next_cache, cache_phases)) {
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Failed;
  }
  if (next_cache.open_user_prefix) {
    next_cache.video_prefix_insert_pos = ctx.n_past;
    next_cache.video_prefix_insert_pos_valid = true;
  }

  if (!save_vision_prefill_cache_state(ctx, next_cache, cache_phases, args.partial_vision_kv)) {
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Failed;
  }
  next_cache.valid = true;
  cache_phases.row("VisionPrefillCacheBuild", build_start_ms, now_ms());
  cache = std::move(next_cache);
  (void)args;
  return VisionPrefillCacheBuildStatus::Ok;
}

int run_vision_prefill_prompt_from_committed_cache(
    const Args& args,
    decode_context& ctx,
    const PromptEvent& prompt,
    int prompt_idx,
    long origin_ms,
    VisionPrefillCache* vision_cache) {
  const bool has_committed_cache =
      vision_cache != nullptr && vision_cache->valid && !vision_cache->frames.empty() && !vision_cache->images.empty();
  const std::vector<FrameRecord> cached_frames = has_committed_cache ? vision_cache->frames : std::vector<FrameRecord>{};
  const std::vector<std::string> images = has_committed_cache ? vision_cache->images : std::vector<std::string>{};
  const std::string token_io = "stream_token_io_" + std::to_string(prompt_idx) + ".txt";
  const std::string inference_tokens = "stream_inference_tokens_" + std::to_string(prompt_idx) + ".txt";
  const std::string phase_path = prompt_phase_path(prompt_idx);

  streamingvlm::hybrid_bridge::phase_recorder prompt_phases(
      phase_path,
      origin_ms,
      streamingvlm::hybrid_bridge::hybrid_decode_phase_description());
  if (!has_committed_cache) {
    const long miss_ms = now_ms();
    prompt_phases.row("VisionPrefillCacheMiss", miss_ms, miss_ms);
    std::fprintf(stderr, "prompt %d has no committed vision-prefill cache snapshot\n", prompt_idx);
    return 2;
  }

  std::unique_ptr<streamingvlm::hybrid_bridge::inference_trace_collector> trace_writer;
  if (!token_io.empty()) {
    trace_writer = std::make_unique<streamingvlm::hybrid_bridge::inference_trace_collector>(
        inference_tokens);
  }

  int rc = 0;
  std::string generated_text;
  const long cache_check_ms = now_ms();
  prompt_phases.row("VisionPrefillCacheHit", cache_check_ms, cache_check_ms);
  const bool prefer_host_restore = args.dynamic_kv_cache && !vision_cache->host_state.empty();
  if (restore_vision_prefill_cache_state(ctx, *vision_cache, prompt_phases, "VisionPrefillCacheRestore", prefer_host_restore)) {
    std::string suffix;
    const bool prompt_closes_open_video_prefix = vision_cache->open_user_prefix;
    const llama_pos video_prefix_insert_pos_before_suffix = ctx.n_past;
    const std::string user_content =
        vision_cache->open_user_prefix ? (vision_cache->open_user_content + prompt.prompt) : prompt.prompt;
    const bool add_special = ctx.chat_history.empty() && !vision_cache->open_user_prefix;
    const bool suffix_ready = vision_cache->open_user_prefix
        ? build_formatted_question_suffix(ctx, vision_cache->open_user_content, prompt.prompt, suffix)
        : (suffix = format_user_message_for_current_history(ctx, prompt.prompt), true);
    if (suffix_ready) {
      if (prompt_closes_open_video_prefix &&
          !save_vision_prefill_video_prefix_state(ctx, *vision_cache, prompt_phases)) {
        rc = 1;
      }
      mtmd::bitmaps empty_bitmaps;
      mtmd::input_chunks suffix_chunks(mtmd_input_chunks_init());
      const bool tokenized_suffix = rc == 0 && tokenize_formatted_text(
              ctx,
              suffix,
              empty_bitmaps,
              add_special,
              prompt_phases,
              "VisionPrefillSuffixTokenize",
              suffix_chunks);
      RenderedPrefillTrace suffix_trace;
      if (tokenized_suffix) {
        suffix_trace = render_prefill_trace_for_chunks(
            ctx,
            suffix_chunks,
            vision_cache->prefill_trace_next_chunk_idx,
            vision_cache->prefill_trace_next_image_idx);
      }
      if (tokenized_suffix && trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
        trace_writer->write_prefill_header();
        trace_writer->append_prefill_trace_body(
            vision_cache->prefill_trace_body,
            vision_cache->prefill_trace_flat);
        trace_writer->append_prefill_trace_body(suffix_trace.body, suffix_trace.flat);
      }
      if (tokenized_suffix &&
          eval_streaming_chunks_with_external_embedding(
              ctx,
              suffix_chunks,
              nullptr,
              prompt_phases,
              0,
              true,
              "T_Prefill",
              "ImagePrefill",
              "Mmproj",
              false)) {
        append_user_message_to_history(ctx, user_content);
        const int n_predict = args.n_predict < 0 ? INT32_MAX : args.n_predict;
        generated_text =
            generate_response(ctx, n_predict, args.force_generation, prompt_phases, trace_writer.get());
        std::string tail_trace_body = vision_cache->prefill_trace_tail_body + suffix_trace.body;
        std::string tail_trace_flat = vision_cache->prefill_trace_tail_flat + suffix_trace.flat;
        std::size_t tail_next_chunk_idx = suffix_trace.next_chunk_idx;
        if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
          const std::string decode_history_body = trace_writer->decode_history_body(tail_next_chunk_idx);
          if (!decode_history_body.empty()) {
            tail_trace_body += decode_history_body;
            tail_trace_flat += trace_writer->decode_flat();
            ++tail_next_chunk_idx;
          }
        }
        vision_cache->frames = cached_frames;
        vision_cache->frame_indices = frame_indices_for(cached_frames);
        vision_cache->images = images;
        vision_cache->open_user_prefix = false;
        vision_cache->open_user_content.clear();
        vision_cache->open_user_frame_indices.clear();
        if (prompt_closes_open_video_prefix) {
          vision_cache->video_prefix_insert_pos = video_prefix_insert_pos_before_suffix;
          vision_cache->video_prefix_insert_pos_valid = true;
          if (vision_cache->prefill_trace_video_body.empty()) {
            set_open_video_trace_from_combined(*vision_cache);
          }
          vision_cache->prefill_trace_tail_body = std::move(tail_trace_body);
          vision_cache->prefill_trace_tail_flat = std::move(tail_trace_flat);
        } else {
          vision_cache->prefill_trace_tail_body = std::move(tail_trace_body);
          vision_cache->prefill_trace_tail_flat = std::move(tail_trace_flat);
        }
        rebuild_prefill_trace_from_video_and_tail(*vision_cache);
        if (!save_vision_prefill_cache_state(ctx, *vision_cache, prompt_phases, args.partial_vision_kv)) {
          rc = 1;
        } else {
          vision_cache->valid = true;
        }
      } else {
        rc = 1;
      }
    } else {
      LOG_ERR("failed to split formatted vision prefill question suffix\n");
      rc = 1;
    }
  } else {
    rc = 1;
  }

  if (rc == 0) {
    write_stream_text_file("stream_response_" + std::to_string(prompt_idx) + ".txt", generated_text);
    std::string token_io_doc = std::string("User: ") + prompt.prompt + "\nAssistant: " + generated_text + "\n";
    if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
      token_io_doc += trace_writer->format_token_io_appendix();
    }
    write_stream_text_file(token_io, token_io_doc);
    trace_writer.reset();
    std::ofstream aggregate("foundation_inference_tokens.txt", std::ios::app);
    std::ifstream raw_trace(inference_tokens);
    if (aggregate && raw_trace) {
      aggregate << "\n===== stream prompt " << prompt_idx << " @ " << prompt.timestamp_s << "s =====\n";
      aggregate << "images: " << join_strings(images, ";") << "\n";
      aggregate << "user: " << prompt.prompt << "\n\n";
      aggregate << raw_trace.rdbuf();
      aggregate << "\n";
    }
  }
  (void)args;
  return rc;
}

int run_single_buffer_prompt(
    const Args& args,
    decode_context& ctx,
    streamingvlm::hybrid_bridge::VisionEncoderSession& encoder,
    const std::vector<FrameRecord>& frames,
    const PromptEvent& prompt,
    int prompt_idx,
    long origin_ms,
    VisionPrefillCache* vision_cache) {
  if (args.stream_mode == "vision_prefill") {
    return run_vision_prefill_prompt_from_committed_cache(
        args,
        ctx,
        prompt,
        prompt_idx,
        origin_ms,
        vision_cache);
  }

  const std::vector<std::string> bins = bins_for_frames(frames);
  const std::vector<std::string> images = layout_images_for_frames(frames);
  if (bins.empty()) {
    std::fprintf(stderr, "prompt %d has no QNN input bins for stream mode %s\n", prompt_idx, args.stream_mode.c_str());
    return 2;
  }
  if (images.empty()) {
    std::fprintf(stderr, "prompt %d has no layout images for stream mode %s\n", prompt_idx, args.stream_mode.c_str());
    return 2;
  }
  const std::string token_io = "stream_token_io_" + std::to_string(prompt_idx) + ".txt";
  const std::string inference_tokens = "stream_inference_tokens_" + std::to_string(prompt_idx) + ".txt";
  const std::string phase_path = prompt_phase_path(prompt_idx);

  streamingvlm::hybrid_bridge::phase_recorder prompt_phases(
      phase_path,
      origin_ms,
      streamingvlm::hybrid_bridge::hybrid_decode_phase_description());
  std::string prompt_text = prompt.prompt;
  if (args.stream_mode == "on_demand") {
    if (prompt_text.find(mtmd_default_marker()) == std::string::npos) {
      prompt_text = std::string(mtmd_default_marker()) + prompt_text;
    }
  } else {
    prompt_text = build_video_prompt(frames, prompt.prompt);
  }

  std::unique_ptr<streamingvlm::hybrid_bridge::inference_trace_collector> trace_writer;
  if (!token_io.empty()) {
    trace_writer = std::make_unique<streamingvlm::hybrid_bridge::inference_trace_collector>(
        inference_tokens);
  }

  int rc = 0;
  std::string generated_text;
  if (is_singleton_video_mode(args)) {
    reset_decode_context_for_singleton(ctx);
  }
  const long vision_start_ms = now_ms();
  auto vision = encoder.encode(bins);
  long cursor_ms = vision_start_ms;
  const long image_load_ms = vision.image_load_end_ms - vision.image_load_start_ms;
  if (image_load_ms > 0) {
    prompt_phases.row("ImageLoad", cursor_ms, cursor_ms + image_load_ms);
    cursor_ms += image_load_ms;
  }
  for (const auto& range : vision.encode_ranges) {
    const long encode_ms = range.second - range.first;
    if (encode_ms > 0) {
      prompt_phases.row("V_Encode", cursor_ms, cursor_ms + encode_ms);
      cursor_ms += encode_ms;
    }
  }
  streamingvlm::hybrid_bridge::EmbeddingFile embedding;
  embedding.shape = vision.output_shape;
  embedding.values = std::move(vision.values);

  if (eval_with_external_embedding(ctx, prompt_text, images, embedding, prompt_phases, nullptr, trace_writer.get()) != 0) {
    rc = 1;
  } else {
    const int n_predict = args.n_predict < 0 ? INT32_MAX : args.n_predict;
    generated_text =
        generate_response(ctx, n_predict, args.force_generation, prompt_phases, trace_writer.get());
  }

  if (rc == 0) {
    write_stream_text_file("stream_response_" + std::to_string(prompt_idx) + ".txt", generated_text);
    std::string token_io_doc = std::string("User: ") + prompt.prompt + "\nAssistant: " + generated_text + "\n";
    if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
      token_io_doc += trace_writer->format_token_io_appendix();
    }
    write_stream_text_file(token_io, token_io_doc);
    trace_writer.reset();
    std::ofstream aggregate("foundation_inference_tokens.txt", std::ios::app);
    std::ifstream raw_trace(inference_tokens);
    if (aggregate && raw_trace) {
      aggregate << "\n===== stream prompt " << prompt_idx << " @ " << prompt.timestamp_s << "s =====\n";
      aggregate << "images: " << join_strings(images, ";") << "\n";
      aggregate << "user: " << prompt.prompt << "\n\n";
      aggregate << raw_trace.rdbuf();
      aggregate << "\n";
    }
  }
  return rc;
}

int run_offline_media_prompt(
    const Args& args,
    decode_context& ctx,
    streamingvlm::hybrid_bridge::VisionEncoderSession& encoder,
    const Manifest& manifest,
    long origin_ms) {
  const std::vector<FrameRecord>& frames = manifest.frames;
  const std::vector<std::string> bins = bins_for_frames(frames);
  const std::vector<std::string> images = layout_images_for_frames(frames);
  const std::string prompt_text =
      !manifest.prompt.empty()
          ? manifest.prompt
          : (!manifest.prompts.empty() ? manifest.prompts.front().prompt : std::string());
  if (bins.empty() || images.empty() || prompt_text.empty()) {
    std::fprintf(stderr, "offline media manifest missing bins, images, or prompt\n");
    return 2;
  }

  streamingvlm::hybrid_bridge::phase_recorder prompt_phases(
      prompt_phase_path(0),
      origin_ms,
      streamingvlm::hybrid_bridge::hybrid_decode_phase_description());
  auto trace_writer = std::make_unique<streamingvlm::hybrid_bridge::inference_trace_collector>(
      "foundation_inference_tokens.txt");

  const long vision_start_ms = now_ms();
  auto vision = encoder.encode(bins);
  long cursor_ms = vision_start_ms;
  const long image_load_ms = vision.image_load_end_ms - vision.image_load_start_ms;
  if (image_load_ms > 0) {
    prompt_phases.row("ImageLoad", cursor_ms, cursor_ms + image_load_ms);
    cursor_ms += image_load_ms;
  }
  for (const auto& range : vision.encode_ranges) {
    const long encode_ms = range.second - range.first;
    if (encode_ms > 0) {
      prompt_phases.row("V_Encode", cursor_ms, cursor_ms + encode_ms);
      cursor_ms += encode_ms;
    }
  }

  streamingvlm::hybrid_bridge::EmbeddingFile embedding;
  embedding.shape = vision.output_shape;
  embedding.values = std::move(vision.values);
  if (eval_with_external_embedding(ctx, prompt_text, images, embedding, prompt_phases, nullptr, trace_writer.get()) != 0) {
    return 1;
  }
  const int n_predict = args.n_predict < 0 ? INT32_MAX : args.n_predict;
  const std::string generated_text =
      generate_response(ctx, n_predict, args.force_generation, prompt_phases, trace_writer.get());
  write_stream_text_file(args.output_path, generated_text);
  std::string token_io_doc = std::string("User: ") + prompt_text + "\nAssistant: " + generated_text + "\n";
  if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
    token_io_doc += trace_writer->format_token_io_appendix();
  }
  write_stream_text_file("foundation_token_io.txt", token_io_doc);
  write_stream_text_file("stream_response_0.txt", generated_text);
  return 0;
}
#else
int run_single_buffer_prompt(
    const Args& args,
    decode_context& ctx,
    const std::vector<FrameRecord>& frames,
    const PromptEvent& prompt,
    int prompt_idx,
    long origin_ms) {
  if (is_singleton_video_mode(args)) {
    reset_decode_context_for_singleton(ctx);
  }
  const std::vector<std::string> images = layout_images_for_frames(frames);
  if (images.empty()) {
    std::fprintf(stderr, "prompt %d has no layout images for OpenCL stream mode %s\n", prompt_idx, args.stream_mode.c_str());
    return 2;
  }
  const std::string token_io = "stream_token_io_" + std::to_string(prompt_idx) + ".txt";
  const std::string inference_tokens = "stream_inference_tokens_" + std::to_string(prompt_idx) + ".txt";
  const std::string phase_path = prompt_phase_path(prompt_idx);

  streamingvlm::hybrid_bridge::phase_recorder prompt_phases(
      phase_path,
      origin_ms,
      streamingvlm::hybrid_bridge::opencl_phase_description());
  common_chat_msg msg;
  std::string prompt_text = prompt.prompt;
  if (args.stream_mode == "on_demand") {
    if (prompt_text.find(mtmd_default_marker()) == std::string::npos) {
      prompt_text = std::string(mtmd_default_marker()) + prompt_text;
    }
  } else {
    prompt_text = build_video_prompt(frames, prompt.prompt);
  }
  msg.role = "user";
  msg.content = prompt_text;

  std::unique_ptr<streamingvlm::hybrid_bridge::inference_trace_collector> trace_writer;
  if (!token_io.empty()) {
    trace_writer = std::make_unique<streamingvlm::hybrid_bridge::inference_trace_collector>(
        inference_tokens);
  }

  int rc = 0;
  if (eval_message(ctx, msg, images, prompt_phases, nullptr, nullptr, trace_writer.get()) != 0) {
    rc = 1;
  } else {
    const int n_predict = args.n_predict < 0 ? INT32_MAX : args.n_predict;
    const std::string generated_text =
        generate_response(ctx, n_predict, args.force_generation, prompt_phases, trace_writer.get());
    write_stream_text_file("stream_response_" + std::to_string(prompt_idx) + ".txt", generated_text);
    std::string token_io_doc = std::string("User: ") + prompt.prompt + "\nAssistant: " + generated_text + "\n";
    if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
      token_io_doc += trace_writer->format_token_io_appendix();
    }
    write_stream_text_file(token_io, token_io_doc);
    trace_writer.reset();
    std::ofstream aggregate("foundation_inference_tokens.txt", std::ios::app);
    std::ifstream raw_trace(inference_tokens);
    if (aggregate && raw_trace) {
      aggregate << "\n===== stream prompt " << prompt_idx << " @ " << prompt.timestamp_s << "s =====\n";
      aggregate << "images: " << join_strings(images, ";") << "\n";
      aggregate << "user: " << prompt.prompt << "\n\n";
      aggregate << raw_trace.rdbuf();
      aggregate << "\n";
    }
  }
  return rc;
}

int run_offline_media_prompt(
    const Args& args,
    decode_context& ctx,
    const Manifest& manifest,
    long origin_ms) {
  const std::vector<FrameRecord>& frames = manifest.frames;
  const std::vector<std::string> images = layout_images_for_frames(frames);
  const std::string prompt_text =
      !manifest.prompt.empty()
          ? manifest.prompt
          : (!manifest.prompts.empty() ? manifest.prompts.front().prompt : std::string());
  if (images.empty() || prompt_text.empty()) {
    std::fprintf(stderr, "offline media manifest missing images or prompt\n");
    return 2;
  }
  streamingvlm::hybrid_bridge::phase_recorder prompt_phases(
      prompt_phase_path(0),
      origin_ms,
      streamingvlm::hybrid_bridge::opencl_phase_description());
  auto trace_writer = std::make_unique<streamingvlm::hybrid_bridge::inference_trace_collector>(
      "foundation_inference_tokens.txt");
  common_chat_msg msg;
  msg.role = "user";
  msg.content = prompt_text;
  if (eval_message(ctx, msg, images, prompt_phases, nullptr, nullptr, trace_writer.get()) != 0) {
    return 1;
  }
  const int n_predict = args.n_predict < 0 ? INT32_MAX : args.n_predict;
  const std::string generated_text =
      generate_response(ctx, n_predict, args.force_generation, prompt_phases, trace_writer.get());
  write_stream_text_file(args.output_path, generated_text);
  std::string token_io_doc = std::string("User: ") + prompt_text + "\nAssistant: " + generated_text + "\n";
  if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
    token_io_doc += trace_writer->format_token_io_appendix();
  }
  write_stream_text_file("foundation_token_io.txt", token_io_doc);
  write_stream_text_file("stream_response_0.txt", generated_text);
  return 0;
}
#endif

void append_file_to_output(
    const std::string& output_path,
    const std::string& input_path,
    int prompt_idx,
    double timestamp_s,
    const std::string& image,
    const std::string& prompt) {
  std::ifstream in(input_path);
  std::ofstream out(output_path, std::ios::app);
  out << "\n===== stream prompt " << prompt_idx << " @ " << timestamp_s << "s =====\n";
  out << "image: " << image << "\n";
  out << "user: " << prompt << "\n";
  out << "assistant:\n";
  if (in) {
    out << in.rdbuf();
  } else {
    out << "(missing response file: " << input_path << ")\n";
  }
  out << "\n";
}

enum class StreamJobKind {
  CacheUpdate,
  Prompt,
};

struct StreamJob {
  StreamJobKind kind = StreamJobKind::Prompt;
  std::vector<FrameRecord> frames;
  PromptEvent prompt;
  int prompt_idx = -1;
  int frame_idx = -1;
  long event_ms = 0;
};

struct StreamBufferStats {
  int input_frames = 0;
  int processed_visual_jobs = 0;
  int committed_cache_updates = 0;
  int prompt_decode_jobs = 0;
  int skipped_cache_updates = 0;
  int latest_frame_only_dropped_cache_updates = 0;
  int kv_reposition_compactions = 0;
  int kv_reposition_removed_frames = 0;
  llama_pos kv_reposition_removed_tokens = 0;
  long committed_cache_update_ms = 0;
  long prompt_decode_ms = 0;
  long first_input_ms = 0;
  long last_input_ms = 0;
  long first_process_ms = 0;
  long last_process_ms = 0;
  std::vector<double> prompt_frame_lag_s;
};

int drop_pending_cache_updates(std::deque<StreamJob>& stream_jobs) {
  const size_t before = stream_jobs.size();
  stream_jobs.erase(
      std::remove_if(
          stream_jobs.begin(),
          stream_jobs.end(),
          [](const StreamJob& job) { return job.kind == StreamJobKind::CacheUpdate; }),
      stream_jobs.end());
  return static_cast<int>(before - stream_jobs.size());
}

bool cache_update_in_queue(const std::deque<StreamJob>& stream_jobs) {
  return std::any_of(
      stream_jobs.begin(),
      stream_jobs.end(),
      [](const StreamJob& job) { return job.kind == StreamJobKind::CacheUpdate; });
}

bool should_drop_cache_update_for_latest_frame_only(
    const Args& args,
    bool cache_worker_busy,
    const std::deque<StreamJob>& stream_jobs,
    const std::atomic<int>& pending_prompt_jobs) {
  return args.latest_frame_only && args.stream_mode == "vision_prefill" &&
         (cache_worker_busy ||
          cache_update_in_queue(stream_jobs) ||
          pending_prompt_jobs.load(std::memory_order_acquire) > 0);
}

std::vector<FrameRecord> resolve_online_buffer_frames(
    const Args& args,
    const std::vector<FrameRecord>& available_frames,
    const FrameRecord& current_frame,
    const PromptEvent& prompt) {
  PromptEvent effective_prompt = prompt;
  effective_prompt.timestamp_s = current_frame.timestamp_s;
  return select_prompt_frames(args, available_frames, current_frame, effective_prompt);
}

void note_processed_visual_job(StreamBufferStats& stats, long start_ms, long end_ms) {
  stats.processed_visual_jobs += 1;
  if (stats.first_process_ms == 0 || start_ms < stats.first_process_ms) {
    stats.first_process_ms = start_ms;
  }
  if (end_ms > stats.last_process_ms) {
    stats.last_process_ms = end_ms;
  }
}

void note_committed_cache_update(StreamBufferStats& stats, long start_ms, long end_ms) {
  stats.committed_cache_updates += 1;
  stats.committed_cache_update_ms += std::max<long>(0, end_ms - start_ms);
  note_processed_visual_job(stats, start_ms, end_ms);
}

void note_prompt_decode_job(StreamBufferStats& stats, long start_ms, long end_ms) {
  stats.prompt_decode_jobs += 1;
  stats.prompt_decode_ms += std::max<long>(0, end_ms - start_ms);
  note_processed_visual_job(stats, start_ms, end_ms);
}

void write_stream_buffer_summary(const Args& args, const Manifest& manifest, const StreamBufferStats& stats) {
  std::ofstream out("stream_buffer_summary.txt");
  const double input_span_s =
      stats.first_input_ms > 0 && stats.last_input_ms > stats.first_input_ms
          ? (stats.last_input_ms - stats.first_input_ms) / 1000.0
          : 0.0;
  const double observed_input_fps = input_span_s > 0.0 ? (stats.input_frames - 1) / input_span_s : 0.0;
  const double process_span_s =
      stats.first_process_ms > 0 && stats.last_process_ms > stats.first_process_ms
          ? (stats.last_process_ms - stats.first_process_ms) / 1000.0
          : 0.0;
  const double processed_visual_fps =
      process_span_s > 0.0 ? stats.processed_visual_jobs / process_span_s : 0.0;
  const double committed_cache_fps =
      input_span_s > 0.0 ? stats.committed_cache_updates / input_span_s : 0.0;
  const double cache_worker_s = stats.committed_cache_update_ms / 1000.0;
  const double cache_worker_fps =
      cache_worker_s > 0.0 ? stats.committed_cache_updates / cache_worker_s : 0.0;
  const double prompt_decode_total_s = stats.prompt_decode_ms / 1000.0;
  double prompt_frame_lag_sum_s = 0.0;
  for (double lag : stats.prompt_frame_lag_s) {
    prompt_frame_lag_sum_s += lag;
  }
  const double avg_prompt_frame_lag_s =
      stats.prompt_frame_lag_s.empty() ? 0.0 : prompt_frame_lag_sum_s / stats.prompt_frame_lag_s.size();
  out << "online_buffer=" << (args.online_buffer ? "true" : "false") << "\n";
  out << "latest_frame_only=" << (args.latest_frame_only ? "true" : "false") << "\n";
  out << "requested_input_fps=" << manifest.sampling_fps << "\n";
  out << "observed_input_fps=" << observed_input_fps << "\n";
  out << "input_frame_count=" << stats.input_frames << "\n";
  out << "processed_visual_jobs=" << stats.processed_visual_jobs << "\n";
  out << "processed_visual_fps=" << processed_visual_fps << "\n";
  out << "committed_cache_updates=" << stats.committed_cache_updates << "\n";
  out << "committed_cache_fps=" << committed_cache_fps << "\n";
  out << "cache_worker_fps=" << cache_worker_fps << "\n";
  out << "cache_worker_total_s=" << cache_worker_s << "\n";
  out << "prompt_decode_jobs=" << stats.prompt_decode_jobs << "\n";
  out << "prompt_decode_total_s=" << prompt_decode_total_s << "\n";
  out << "skipped_cache_updates=" << stats.skipped_cache_updates << "\n";
  out << "latest_frame_only_dropped_cache_updates=" << stats.latest_frame_only_dropped_cache_updates << "\n";
  out << "kv_reposition_keep_latest_frames=" << args.kv_reposition_keep_latest_frames << "\n";
  out << "kv_reposition_compactions=" << stats.kv_reposition_compactions << "\n";
  out << "kv_reposition_removed_frames=" << stats.kv_reposition_removed_frames << "\n";
  out << "kv_reposition_removed_tokens=" << stats.kv_reposition_removed_tokens << "\n";
  out << "prompt_frame_lag_s_avg=" << avg_prompt_frame_lag_s << "\n";
  out << "prompt_frame_lag_s_count=" << stats.prompt_frame_lag_s.size() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  std::setlocale(LC_NUMERIC, "C");
  ggml_time_init();
  common_init();
  mtmd_helper_log_set(common_log_default_callback, nullptr);
  Args args = parse_args(argc, argv);
  Manifest manifest = parse_manifest(args.manifest);
  if (args.stream_mode.empty() && !manifest.stream_mode.empty()) {
    args.stream_mode = manifest.stream_mode;
  }
  if (args.window_sec <= 0.0 && manifest.window_sec > 0.0) {
    args.window_sec = manifest.window_sec;
  }
  if (args.window_max_frames <= 0 && manifest.window_max_frames > 0) {
    args.window_max_frames = manifest.window_max_frames;
  }
  args.stream_mode = normalize_stream_mode(args.stream_mode, args.single_buffer);
  args.single_buffer = args.stream_mode == "on_demand";
  if (args.kv_reposition_keep_latest_frames > 0) {
    setenv("LLAMA_ALLOW_KV_GAP_FILL", "1", 1);
  }
  if (manifest.frames.empty()) {
    std::fprintf(stderr, "stream manifest has no frames: %s\n", args.manifest.c_str());
    return 2;
  }
  if (args.play_speed <= 0.0) {
    std::fprintf(stderr, "--play-speed must be positive\n");
    return 2;
  }
  if (args.window_max_frames <= 0) {
    std::fprintf(stderr, "--window-max-frames must be positive\n");
    return 2;
  }

  const long origin_ms = now_ms();
  std::vector<PhaseTiming> setup_phases;
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
  auto encoder_ctx = load_single_buffer_encoder_context(args, setup_phases);
#endif
  auto decode_ctx = load_single_buffer_decoder_context(args, manifest, setup_phases);

  EventWriter events(args.stream_events_path);
  std::ofstream phases(args.phase_stats_path);
  phases << "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
            "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
            "kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx\n";
  phases << "# StreamFrameEnqueue: sampled frame enters buffer  OnDemandBufferUpdate: current image buffer update  "
            "StreamPromptPrefill/StreamDecode: prompt handled against current buffered image\n";
  phases << "# clock_origin_ms: " << origin_ms << "\n";
  for (const auto& setup : setup_phases) {
    append_phase_row(phases, setup.name, setup.start_ms, setup.end_ms, origin_ms);
  }

  const bool offline_media_mode =
      args.media_mode != "streaming" &&
      manifest.source_kind != "streaming_video";
  if (offline_media_mode) {
    for (const auto& frame : manifest.frames) {
      events.row(
          "StreamFrameEnqueue",
          frame.index,
          -1,
          frame.timestamp_s,
          origin_ms,
          origin_ms,
          origin_ms,
          "offline");
    }
    const long prompt_start_ms = now_ms();
    append_phase_row(phases, "StreamPromptPrefill", prompt_start_ms, prompt_start_ms, origin_ms);
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
    const int rc = run_offline_media_prompt(args, *decode_ctx, *encoder_ctx, manifest, origin_ms);
#else
    const int rc = run_offline_media_prompt(args, *decode_ctx, manifest, origin_ms);
#endif
    append_phase_file(phases, prompt_phase_path(0));
    const long prompt_end_ms = now_ms();
    events.row(
        "StreamDecode",
        last_frame_index(manifest.frames),
        0,
        manifest.prompts.empty() ? 0.0 : manifest.prompts.front().timestamp_s,
        origin_ms,
        prompt_start_ms,
        prompt_end_ms,
        "rc=" + std::to_string(rc));
    return rc;
  }

  std::deque<StreamJob> stream_jobs;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  FrameRecord latest_frame;
  bool have_latest_frame = false;
  std::vector<FrameRecord> latest_available_frames;
  StreamBufferStats buffer_stats;
  std::atomic<int> pending_prompt_jobs{0};
  bool cache_worker_busy = false;

  std::thread producer([&]() {
    double last_ts = 0.0;
    size_t prompt_cursor = 0;
    FrameRecord current_frame;
    bool have_current_frame = false;
    std::vector<FrameRecord> available_frames;
    for (const auto& frame : manifest.frames) {
      if (args.realtime) {
        const double delta_s = std::max(0.0, frame.timestamp_s - last_ts) / args.play_speed;
        if (delta_s > 0.0) {
          std::this_thread::sleep_for(std::chrono::duration<double>(delta_s));
        }
      }
      const long t = now_ms();
      current_frame = frame;
      have_current_frame = true;
      available_frames.push_back(frame);
      {
        std::lock_guard<std::mutex> lock(mu);
        latest_frame = current_frame;
        have_latest_frame = true;
        latest_available_frames = available_frames;
        buffer_stats.input_frames += 1;
        if (buffer_stats.first_input_ms == 0) {
          buffer_stats.first_input_ms = t;
        }
        buffer_stats.last_input_ms = t;
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
        if (args.stream_mode == "vision_prefill") {
          PromptEvent cache_event;
          cache_event.timestamp_s = frame.timestamp_s;
          if (should_drop_cache_update_for_latest_frame_only(
                  args,
                  cache_worker_busy,
                  stream_jobs,
                  pending_prompt_jobs)) {
            buffer_stats.skipped_cache_updates += 1;
            buffer_stats.latest_frame_only_dropped_cache_updates += 1;
            events.row(
                "LatestFrameOnlyCacheDrop",
                frame.index,
                -1,
                frame.timestamp_s,
                origin_ms,
                t,
                t,
                "cache_worker_busy_or_queued");
          } else {
            if (args.online_buffer) {
              buffer_stats.skipped_cache_updates += drop_pending_cache_updates(stream_jobs);
            }
            stream_jobs.push_back(StreamJob{
                StreamJobKind::CacheUpdate,
                args.online_buffer
                    ? std::vector<FrameRecord>{}
                    : (args.latest_frame_only
                           ? std::vector<FrameRecord>{current_frame}
                           : select_prompt_frames(args, available_frames, current_frame, cache_event)),
                cache_event,
                -1,
                frame.index,
                t,
            });
          }
        }
#endif
        while (prompt_cursor < manifest.prompts.size() && manifest.prompts[prompt_cursor].timestamp_s <= frame.timestamp_s) {
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
          if (args.stream_mode == "vision_prefill") {
            pending_prompt_jobs.fetch_add(1, std::memory_order_release);
            buffer_stats.skipped_cache_updates += drop_pending_cache_updates(stream_jobs);
          }
#endif
          stream_jobs.push_back(StreamJob{
              StreamJobKind::Prompt,
              args.online_buffer ? std::vector<FrameRecord>{} : select_prompt_frames(args, available_frames, current_frame, manifest.prompts[prompt_cursor]),
              manifest.prompts[prompt_cursor],
              static_cast<int>(prompt_cursor),
              frame.index,
              t,
          });
          ++prompt_cursor;
        }
      }
      events.row("StreamFrameEnqueue", frame.index, -1, frame.timestamp_s, origin_ms, t, t, "queued");
      append_phase_row(phases, "OnDemandBufferUpdate", t, t, origin_ms);
      events.row(
          "OnDemandBufferUpdate",
          frame.index,
          -1,
          frame.timestamp_s,
          origin_ms,
          t,
          t,
          frame.tiles.empty() ? "" : frame.tiles.front().layout_image);
      cv.notify_one();
      last_ts = frame.timestamp_s;
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      while (have_current_frame && prompt_cursor < manifest.prompts.size()) {
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
        if (args.stream_mode == "vision_prefill") {
          pending_prompt_jobs.fetch_add(1, std::memory_order_release);
          buffer_stats.skipped_cache_updates += drop_pending_cache_updates(stream_jobs);
        }
#endif
        stream_jobs.push_back(StreamJob{
            StreamJobKind::Prompt,
            args.online_buffer ? std::vector<FrameRecord>{} : select_prompt_frames(args, available_frames, current_frame, manifest.prompts[prompt_cursor]),
            manifest.prompts[prompt_cursor],
            static_cast<int>(prompt_cursor),
            current_frame.index,
            now_ms(),
        });
        ++prompt_cursor;
      }
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      done = true;
    }
    cv.notify_all();
  });

  size_t handled_prompts = 0;
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
  VisionPrefillCache vision_prefill_cache;
#endif

  while (true) {
    StreamJob job;
    {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [&]() { return done || !stream_jobs.empty(); });
      if (stream_jobs.empty()) {
        if (done) {
          break;
        }
        continue;
      }
      job = stream_jobs.front();
      stream_jobs.pop_front();
      cache_worker_busy = true;
      if (args.online_buffer && have_latest_frame) {
        const bool prompt_uses_committed_vision_cache =
            args.stream_mode == "vision_prefill" && job.kind == StreamJobKind::Prompt;
        if (prompt_uses_committed_vision_cache) {
          buffer_stats.prompt_frame_lag_s.push_back(latest_frame.timestamp_s - job.prompt.timestamp_s);
        } else if (args.stream_mode == "vision_prefill" && job.kind == StreamJobKind::CacheUpdate && args.latest_frame_only) {
          job.frames = {latest_frame};
          job.frame_idx = latest_frame.index;
          job.prompt.timestamp_s = latest_frame.timestamp_s;
        } else if (args.stream_mode == "vision_prefill" && job.kind == StreamJobKind::CacheUpdate) {
          job.frames = {latest_frame};
          job.frame_idx = latest_frame.index;
          job.prompt.timestamp_s = latest_frame.timestamp_s;
        } else {
          job.frames = resolve_online_buffer_frames(args, latest_available_frames, latest_frame, job.prompt);
          job.frame_idx = latest_frame.index;
          if (job.kind == StreamJobKind::CacheUpdate) {
            job.prompt.timestamp_s = latest_frame.timestamp_s;
          } else if (job.kind == StreamJobKind::Prompt) {
            buffer_stats.prompt_frame_lag_s.push_back(latest_frame.timestamp_s - job.prompt.timestamp_s);
          }
        }
      }
    }

    auto mark_cache_worker_idle = [&]() {
      std::lock_guard<std::mutex> lock(mu);
      cache_worker_busy = false;
    };

#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
    if (job.kind == StreamJobKind::CacheUpdate) {
      const long cache_start_ms = now_ms();
      if (cache_preempt_requested(&pending_prompt_jobs)) {
        const long preempt_ms = now_ms();
        append_phase_row(phases, "VisionPrefillCachePreempt", preempt_ms, preempt_ms, origin_ms);
        events.row(
            "VisionPrefillCacheBuild",
            last_frame_index(job.frames),
            -1,
            job.prompt.timestamp_s,
            origin_ms,
            cache_start_ms,
            preempt_ms,
            "preempted");
        {
          std::lock_guard<std::mutex> lock(mu);
          buffer_stats.skipped_cache_updates += 1;
        }
        mark_cache_worker_idle();
        continue;
      }
      const VisionPrefillCacheBuildStatus status = build_vision_prefill_cache(
          args,
          *decode_ctx,
          *encoder_ctx,
          job.frames,
          job.frame_idx,
          origin_ms,
          vision_prefill_cache,
          &pending_prompt_jobs);
      const long cache_end_ms = now_ms();
      append_phase_file(phases, "stream_vision_prefill_cache_" + std::to_string(job.frame_idx) + ".csv");
      events.row(
          "VisionPrefillCacheBuild",
          last_frame_index(job.frames),
          -1,
          job.prompt.timestamp_s,
          origin_ms,
          cache_start_ms,
          cache_end_ms,
          cache_build_status_detail(status));
      {
        std::lock_guard<std::mutex> lock(mu);
        if (status == VisionPrefillCacheBuildStatus::Preempted) {
          buffer_stats.skipped_cache_updates += 1;
        } else if (status == VisionPrefillCacheBuildStatus::Ok ||
                   status == VisionPrefillCacheBuildStatus::Partial) {
          note_committed_cache_update(buffer_stats, cache_start_ms, cache_end_ms);
          buffer_stats.kv_reposition_compactions += vision_prefill_cache.last_kv_reposition_compactions;
          buffer_stats.kv_reposition_removed_frames += vision_prefill_cache.last_kv_reposition_removed_frames;
          buffer_stats.kv_reposition_removed_tokens += vision_prefill_cache.last_kv_reposition_removed_tokens;
        } else {
          note_processed_visual_job(buffer_stats, cache_start_ms, cache_end_ms);
        }
      }
      mark_cache_worker_idle();
      continue;
    }
#endif

#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
    if (args.stream_mode == "vision_prefill" && job.kind == StreamJobKind::Prompt && vision_prefill_cache.valid) {
      job.frames = vision_prefill_cache.frames;
      job.frame_idx = last_frame_index(job.frames);
    }
    if (args.stream_mode == "vision_prefill" && job.kind == StreamJobKind::Prompt) {
      if (pending_prompt_jobs.load(std::memory_order_acquire) > 0) {
        pending_prompt_jobs.fetch_sub(1, std::memory_order_acq_rel);
      }
    }
#endif

    events.row(
        "StreamPromptPrefill",
        last_frame_index(job.frames),
        job.prompt_idx,
        job.prompt.timestamp_s,
        origin_ms,
        job.event_ms,
        job.event_ms,
        job.prompt.prompt);
    append_phase_row(phases, "StreamPromptPrefill", job.event_ms, job.event_ms, origin_ms);
    const long decode_start_ms = now_ms();
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
    const int rc = run_single_buffer_prompt(
        args,
        *decode_ctx,
        *encoder_ctx,
        job.frames,
        job.prompt,
        job.prompt_idx,
        origin_ms,
        args.stream_mode == "vision_prefill" ? &vision_prefill_cache : nullptr);
#else
    const int rc = run_single_buffer_prompt(args, *decode_ctx, job.frames, job.prompt, job.prompt_idx, origin_ms);
#endif
    const long decode_end_ms = now_ms();
    append_phase_file(phases, prompt_phase_path(job.prompt_idx));
    const std::string response_path = "stream_response_" + std::to_string(job.prompt_idx) + ".txt";
    const std::string image = join_strings(layout_images_for_frames(job.frames), ";");
    append_file_to_output(
        args.output_path,
        response_path,
        job.prompt_idx,
        job.prompt.timestamp_s,
        image,
        job.prompt.prompt);
    events.row(
        "StreamDecode",
        last_frame_index(job.frames),
        job.prompt_idx,
        job.prompt.timestamp_s,
        origin_ms,
        decode_start_ms,
        decode_end_ms,
        "rc=" + std::to_string(rc));
    {
      std::lock_guard<std::mutex> lock(mu);
      note_prompt_decode_job(buffer_stats, decode_start_ms, decode_end_ms);
    }
    mark_cache_worker_idle();
    ++handled_prompts;
  }

  producer.join();
  write_stream_buffer_summary(args, manifest, buffer_stats);
  std::fprintf(stderr, "Processed %zu streaming frames and %zu prompt events\n", manifest.frames.size(), handled_prompts);
  return 0;
}
