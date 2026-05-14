#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
#include "vision_encoder_et.hpp"
#include "hybrid_decode.cpp"
#else
#include "opencl_phase_mtmd.cpp"
#endif

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
  double sampling_fps = 0.0;
  std::string stream_mode;
  double window_sec = -1.0;
  int window_max_frames = 0;
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
  find_number_after(text, 0, "sampling_fps", manifest.sampling_fps);
  find_string_after(text, 0, "stream_mode", manifest.stream_mode);
  find_number_after(text, 0, "window_sec", manifest.window_sec);
  double window_max_frames = 0.0;
  if (find_number_after(text, 0, "window_max_frames", window_max_frames)) {
    manifest.window_max_frames = static_cast<int>(window_max_frames);
  }

  for (const std::string& block : object_blocks_in_array(text, "frames")) {
    FrameRecord frame;
    double frame_index = 0.0;
    find_number_after(block, 0, "stream_frame", frame_index);
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
  std::string stream_mode;
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
  int kv_page_size = 256;
  int batch_size = 2048;
  int ubatch_size = 512;
  int gpu_layers = 99;
  int threads = 4;
  double temperature = 0.0;
  double window_sec = -1.0;
  double play_speed = 1.0;
  int window_max_frames = 8;
  bool realtime = true;
  bool force_generation = false;
  bool single_buffer = false;
  bool dynamic_kv_cache = false;
  bool paged_kv_cache = false;
  bool no_kv_offload = false;
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
        read_string("--stream-mode", args.stream_mode) || read_string("--stream_mode", args.stream_mode) ||
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
    } else if (read_string("--kv-page-size", tmp)) {
      args.kv_page_size = std::atoi(tmp.c_str());
    } else if (read_string("-b", tmp) || read_string("--batch-size", tmp)) {
      args.batch_size = std::atoi(tmp.c_str());
    } else if (read_string("-ub", tmp) || read_string("--ubatch-size", tmp)) {
      args.ubatch_size = std::atoi(tmp.c_str());
    } else if (read_string("-ngl", tmp) || read_string("--gpu-layers", tmp)) {
      args.gpu_layers = std::atoi(tmp.c_str());
    } else if (read_string("-t", tmp) || read_string("--threads", tmp)) {
      args.threads = std::atoi(tmp.c_str());
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
    } else if (a == "--dynamic-kv-cache") {
      args.dynamic_kv_cache = true;
    } else if (a == "--paged-kv-cache") {
      args.paged_kv_cache = true;
    } else if (a == "--force-generation") {
      args.force_generation = true;
    } else if (a == "--no-kv-offload") {
      args.no_kv_offload = true;
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
    return "single_buffer";
  }
  std::replace(mode.begin(), mode.end(), '-', '_');
  if (mode.empty()) {
    return "single_buffer";
  }
  if (mode == "single_buffer" || mode == "sliding_window" || mode == "vision_prefill") {
    return mode;
  }
  std::fprintf(stderr, "unsupported --stream-mode: %s\n", mode.c_str());
  std::exit(2);
}

bool is_singleton_video_mode(const Args& args) {
  return args.stream_mode == "sliding_window" || args.stream_mode == "vision_prefill";
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
  if (args.paged_kv_cache) {
    out.push_back("--paged-kv-cache");
    out.push_back("--kv-page-size");
    out.push_back(std::to_string(args.kv_page_size));
  } else if (!args.dynamic_kv_cache) {
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
  if (args.stream_mode == "single_buffer") {
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
    out += "Frame " + std::to_string(frame_i + 1) + ": ";
    const size_t n_tiles = std::max<size_t>(1, frames[frame_i].tiles.size());
    for (size_t tile_i = 0; tile_i < n_tiles; ++tile_i) {
      out += mtmd_default_marker();
    }
    out += "\n";
  }
  return out;
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
    std::fprintf(stderr, "--encoder-path is required for QNN single-buffer streaming\n");
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
struct VisionPrefillCache {
  bool valid = false;
  std::vector<int> frame_indices;
  std::vector<std::string> images;
  std::vector<uint8_t> state;
  llama_state_seq_flags state_flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
  llama_pos n_past = 0;
};

bool vision_prefill_cache_matches(const VisionPrefillCache& cache, const std::vector<FrameRecord>& frames) {
  return cache.valid && cache.frame_indices == frame_indices_for(frames);
}

std::string format_singleton_user_message(decode_context& ctx, const std::string& content) {
  common_chat_msg msg;
  msg.role = "user";
  msg.content = content;
  return common_chat_format_single(ctx.tmpls.get(), {}, msg, true, ctx.use_jinja);
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
    const std::vector<FrameRecord>& frames,
    std::string& out) {
  const std::string content = build_video_prompt_prefix(frames) + SVLM_QUESTION_SENTINEL;
  return split_formatted_at_question_sentinel(format_singleton_user_message(ctx, content), &out, nullptr);
}

bool build_formatted_question_suffix(
    decode_context& ctx,
    const std::vector<FrameRecord>& frames,
    const std::string& raw_prompt,
    std::string& out) {
  const std::string content = build_video_prompt_prefix(frames) + SVLM_QUESTION_SENTINEL + raw_prompt;
  return split_formatted_at_question_sentinel(format_singleton_user_message(ctx, content), nullptr, &out);
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
      const int32_t n_feature_embd = embeddings->feature_dim_for_chunk(n_tokens, n_embd);
      float* image_slice = embeddings->next_slice(n_tokens, n_feature_embd);
      float* image_embedding = image_slice;
      if (n_feature_embd != n_embd) {
        const long mmproj_start_ms = now_ms();
        if (mtmd_project_features(
                ctx.ctx_vision.get(),
                image_slice,
                static_cast<int32_t>(n_tokens),
                n_feature_embd) != 0) {
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
    bool require_image) {
  const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
  bool used_image = false;
  size_t image_chunk_idx = 0;
  const int32_t decoder_embedding_size = llama_model_n_embd_inp(ctx.model);

  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const bool chunk_logits_last = logits_last && i == n_chunks - 1;
    llama_pos new_n_past = ctx.n_past;
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
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

      streamingvlm::hybrid_bridge::EmbeddingFile embedding;
      embedding.shape = vision.output_shape;
      embedding.values = std::move(vision.values);
      embedding_cursor embeddings(embedding);

      const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
      const int32_t n_embd = decoder_embedding_size;
      const int32_t n_feature_embd = embeddings.feature_dim_for_chunk(n_tokens, n_embd);
      float* image_slice = embeddings.next_slice(n_tokens, n_feature_embd);
      float* image_embedding = image_slice;
      if (n_feature_embd != n_embd) {
        const long mmproj_start_ms = now_ms();
        if (mtmd_project_features(
                ctx.ctx_vision.get(),
                image_slice,
                static_cast<int32_t>(n_tokens),
                n_feature_embd) != 0) {
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
        LOG_ERR("failed to decode on-demand vision image embedding\n");
        return false;
      }
      llama_synchronize(ctx.lctx);
      phases.row(image_phase_name, image_prefill_start_ms, now_ms());
      embeddings.finish();
      used_image = true;
      ++image_chunk_idx;
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

bool save_vision_prefill_cache_state(
    decode_context& ctx,
    VisionPrefillCache& cache,
    streamingvlm::hybrid_bridge::phase_recorder& phases) {
  const long save_start_ms = now_ms();
  cache.state_flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
  size_t state_size = llama_state_seq_get_size_ext(ctx.lctx, 0, cache.state_flags);
  if (state_size == 0) {
    cache.state_flags = static_cast<llama_state_seq_flags>(0);
    state_size = llama_state_seq_get_size_ext(ctx.lctx, 0, cache.state_flags);
  }
  if (state_size == 0) {
    LOG_ERR("failed to determine vision prefill cache state size\n");
    return false;
  }
  cache.state.assign(state_size, 0);
  const size_t copied = llama_state_seq_get_data_ext(
      ctx.lctx,
      cache.state.data(),
      cache.state.size(),
      0,
      cache.state_flags);
  phases.row("VisionPrefillCacheSave", save_start_ms, now_ms());
  if (copied != cache.state.size()) {
    LOG_ERR("failed to save vision prefill cache state: copied %zu of %zu bytes\n", copied, cache.state.size());
    cache.state.clear();
    return false;
  }
  cache.n_past = ctx.n_past;
  return true;
}

bool restore_vision_prefill_cache_state(
    decode_context& ctx,
    const VisionPrefillCache& cache,
    streamingvlm::hybrid_bridge::phase_recorder& phases) {
  if (!cache.valid || cache.state.empty()) {
    return false;
  }
  reset_decode_context_for_singleton(ctx);
  const long restore_start_ms = now_ms();
  const size_t restored = llama_state_seq_set_data_ext(
      ctx.lctx,
      cache.state.data(),
      cache.state.size(),
      0,
      cache.state_flags);
  llama_synchronize(ctx.lctx);
  phases.row("VisionPrefillCacheRestore", restore_start_ms, now_ms());
  if (restored != cache.state.size()) {
    LOG_ERR("failed to restore vision prefill cache state: restored %zu of %zu bytes\n", restored, cache.state.size());
    return false;
  }
  ctx.n_past = cache.n_past;
  return true;
}

bool build_vision_prefill_cache(
    const Args& args,
    decode_context& ctx,
    streamingvlm::hybrid_bridge::VisionEncoderSession& encoder,
    const std::vector<FrameRecord>& frames,
    int frame_idx,
    long origin_ms,
    VisionPrefillCache& cache) {
  VisionPrefillCache next_cache;
  next_cache.frame_indices = frame_indices_for(frames);
  next_cache.images = layout_images_for_frames(frames);
  const std::vector<std::string> bins = bins_for_frames(frames);
  if (bins.empty() || next_cache.images.empty()) {
    LOG_ERR("cannot build vision prefill cache for frame %d: missing bins or layout images\n", frame_idx);
    cache = std::move(next_cache);
    return false;
  }

  const std::string phase_path = "stream_vision_prefill_cache_" + std::to_string(frame_idx) + ".csv";
  streamingvlm::hybrid_bridge::phase_recorder cache_phases(
      phase_path,
      origin_ms,
      streamingvlm::hybrid_bridge::hybrid_decode_phase_description());
  const long build_start_ms = now_ms();
  reset_decode_context_for_singleton(ctx);

  std::string formatted_prefix;
  if (!build_formatted_vision_cache_prefix(ctx, frames, formatted_prefix)) {
    LOG_ERR("failed to split formatted vision prefill cache prefix\n");
    cache = std::move(next_cache);
    return false;
  }

  mtmd::bitmaps bitmaps;
  if (!load_layout_bitmaps(ctx, next_cache.images, bitmaps)) {
    cache = std::move(next_cache);
    return false;
  }

  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  if (!tokenize_formatted_text(
          ctx,
          formatted_prefix,
          bitmaps,
          true,
          cache_phases,
          "VisionPrefillLayoutTokenize",
          chunks)) {
    cache = std::move(next_cache);
    return false;
  }

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
          true)) {
    cache = std::move(next_cache);
    return false;
  }

  if (!save_vision_prefill_cache_state(ctx, next_cache, cache_phases)) {
    cache = std::move(next_cache);
    return false;
  }
  next_cache.valid = true;
  cache_phases.row("VisionPrefillCacheBuild", build_start_ms, now_ms());
  cache = std::move(next_cache);
  (void)args;
  return true;
}

int run_single_buffer_prompt(
    const Args& args,
    decode_context& ctx,
    streamingvlm::hybrid_bridge::VisionEncoderSession& encoder,
    const std::vector<FrameRecord>& frames,
    const PromptEvent& prompt,
    int prompt_idx,
    long origin_ms,
    const VisionPrefillCache* vision_cache) {
  if (args.stream_mode == "sliding_window") {
    reset_decode_context_for_singleton(ctx);
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
  if (args.stream_mode == "single_buffer") {
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
  bool handled_with_cache = false;
  bool cache_miss_recorded = false;
  std::string generated_text;

  if (args.stream_mode == "vision_prefill") {
    const long cache_check_ms = now_ms();
    if (vision_cache != nullptr && vision_prefill_cache_matches(*vision_cache, frames)) {
      prompt_phases.row("VisionPrefillCacheHit", cache_check_ms, cache_check_ms);
      if (restore_vision_prefill_cache_state(ctx, *vision_cache, prompt_phases)) {
        std::string suffix;
        if (build_formatted_question_suffix(ctx, frames, prompt.prompt, suffix)) {
          mtmd::bitmaps empty_bitmaps;
          mtmd::input_chunks suffix_chunks(mtmd_input_chunks_init());
          if (tokenize_formatted_text(
                  ctx,
                  suffix,
                  empty_bitmaps,
                  false,
                  prompt_phases,
                  "VisionPrefillSuffixTokenize",
                  suffix_chunks) &&
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
            const int n_predict = args.n_predict < 0 ? INT32_MAX : args.n_predict;
            generated_text =
                generate_response(ctx, n_predict, args.force_generation, prompt_phases, trace_writer.get());
            handled_with_cache = true;
          } else {
            rc = 1;
          }
        } else {
          LOG_ERR("failed to split formatted vision prefill question suffix\n");
          rc = 1;
        }
      }
    } else {
      prompt_phases.row("VisionPrefillCacheMiss", cache_check_ms, cache_check_ms);
      cache_miss_recorded = true;
    }
  }

  if (!handled_with_cache) {
    if (args.stream_mode == "vision_prefill" && !cache_miss_recorded) {
      const long miss_ms = now_ms();
      prompt_phases.row("VisionPrefillCacheMiss", miss_ms, miss_ms);
    }
    rc = 0;
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
  if (args.stream_mode == "single_buffer") {
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
  args.single_buffer = args.stream_mode == "single_buffer";
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
  phases << "# StreamFrameEnqueue: sampled frame enters buffer  SingleBufferUpdate: current image buffer update  "
            "StreamPromptPrefill/StreamDecode: prompt handled against current buffered image\n";
  phases << "# clock_origin_ms: " << origin_ms << "\n";
  for (const auto& setup : setup_phases) {
    append_phase_row(phases, setup.name, setup.start_ms, setup.end_ms, origin_ms);
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
  std::deque<StreamJob> stream_jobs;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;

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
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
        if (args.stream_mode == "vision_prefill") {
          PromptEvent cache_event;
          cache_event.timestamp_s = frame.timestamp_s;
          stream_jobs.push_back(StreamJob{
              StreamJobKind::CacheUpdate,
              select_prompt_frames(args, available_frames, current_frame, cache_event),
              cache_event,
              -1,
              frame.index,
              t,
          });
        }
#endif
        while (prompt_cursor < manifest.prompts.size() && manifest.prompts[prompt_cursor].timestamp_s <= frame.timestamp_s) {
          stream_jobs.push_back(StreamJob{
              StreamJobKind::Prompt,
              select_prompt_frames(args, available_frames, current_frame, manifest.prompts[prompt_cursor]),
              manifest.prompts[prompt_cursor],
              static_cast<int>(prompt_cursor),
              frame.index,
              t,
          });
          ++prompt_cursor;
        }
      }
      events.row("StreamFrameEnqueue", frame.index, -1, frame.timestamp_s, origin_ms, t, t, "queued");
      append_phase_row(phases, "SingleBufferUpdate", t, t, origin_ms);
      events.row(
          "SingleBufferUpdate",
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
        stream_jobs.push_back(StreamJob{
            StreamJobKind::Prompt,
            select_prompt_frames(args, available_frames, current_frame, manifest.prompts[prompt_cursor]),
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
    }

#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
    if (job.kind == StreamJobKind::CacheUpdate) {
      const long cache_start_ms = now_ms();
      const bool ok = build_vision_prefill_cache(
          args,
          *decode_ctx,
          *encoder_ctx,
          job.frames,
          job.frame_idx,
          origin_ms,
          vision_prefill_cache);
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
          ok ? "ok" : "miss");
      continue;
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
    ++handled_prompts;
  }

  producer.join();
  std::fprintf(stderr, "Processed %zu streaming frames and %zu prompt events\n", manifest.frames.size(), handled_prompts);
  return 0;
}
