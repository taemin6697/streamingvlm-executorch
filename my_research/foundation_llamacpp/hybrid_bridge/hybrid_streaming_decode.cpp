#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "opencl_phase_mtmd.cpp"

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
  std::string runner = "./opencl_phase_mtmd";
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
  int batch_size = 2048;
  int ubatch_size = 512;
  int gpu_layers = 99;
  int threads = 4;
  double temperature = 0.0;
  double play_speed = 1.0;
  bool realtime = true;
  bool force_generation = false;
  bool single_buffer = false;
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
    if (read_string("--runner", args.runner) || read_string("-m", args.model) || read_string("--model", args.model) ||
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
    } else if (read_string("--play-speed", tmp) || read_string("--play_speed", tmp)) {
      args.play_speed = std::atof(tmp.c_str());
    } else if (a == "--single-buffer" || a == "--single_buffer") {
      args.single_buffer = true;
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
      "--ctx-size",
      std::to_string(args.ctx_size),
      "--batch-size",
      std::to_string(args.batch_size),
      "--ubatch-size",
      std::to_string(args.ubatch_size),
      "--temp",
      std::to_string(args.temperature),
  };
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

std::unique_ptr<decode_context> load_single_buffer_context(
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
  warmup_split_encoder_with_image(*ctx, warm_image);
  return ctx;
}

int run_single_buffer_prompt(
    const Args& args,
    decode_context& ctx,
    const FrameRecord& frame,
    const PromptEvent& prompt,
    int prompt_idx,
    long origin_ms) {
  const std::string image = frame.tiles.empty() ? "" : frame.tiles.front().layout_image;
  if (image.empty()) {
    std::fprintf(stderr, "frame %d has no layout image for single-buffer mode\n", frame.index);
    return 2;
  }
  const std::string token_io = "stream_token_io_" + std::to_string(prompt_idx) + ".txt";
  const std::string phase_path = prompt_phase_path(prompt_idx);

  llama_memory_clear(llama_get_memory(ctx.lctx), true);
  common_sampler_reset(ctx.smpl);
  ctx.chat_history.clear();
  ctx.n_past = 0;

  streamingvlm::hybrid_bridge::phase_recorder prompt_phases(
      phase_path,
      origin_ms,
      streamingvlm::hybrid_bridge::opencl_phase_description());
  common_chat_msg msg;
  std::string prompt_text = prompt.prompt;
  if (prompt_text.find(mtmd_default_marker()) == std::string::npos) {
    prompt_text = std::string(mtmd_default_marker()) + prompt_text;
  }
  msg.role = "user";
  msg.content = prompt_text;

  std::unique_ptr<streamingvlm::hybrid_bridge::inference_trace_collector> trace_writer;
  if (!token_io.empty()) {
    trace_writer = std::make_unique<streamingvlm::hybrid_bridge::inference_trace_collector>(
        streamingvlm::hybrid_bridge::sibling_foundation_inference_tokens_path(token_io));
  }

  int rc = 0;
  if (eval_message(ctx, msg, {image}, prompt_phases, nullptr, nullptr, trace_writer.get()) != 0) {
    rc = 1;
  } else {
    const int n_predict = args.n_predict < 0 ? INT32_MAX : args.n_predict;
    const std::string generated_text =
        generate_response(ctx, n_predict, args.force_generation, prompt_phases, trace_writer.get());
    write_text_file("stream_response_" + std::to_string(prompt_idx) + ".txt", generated_text);
    std::string token_io_doc = std::string("User: ") + prompt.prompt + "\nAssistant: " + generated_text + "\n";
    if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
      token_io_doc += trace_writer->format_token_io_appendix();
    }
    write_text_file(token_io, token_io_doc);
  }
  return rc;
}

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
  const Args args = parse_args(argc, argv);
  Manifest manifest = parse_manifest(args.manifest);
  if (manifest.frames.empty()) {
    std::fprintf(stderr, "stream manifest has no frames: %s\n", args.manifest.c_str());
    return 2;
  }
  if (!args.single_buffer) {
    std::fprintf(stderr, "Only --single-buffer streaming mode is implemented in this runner.\n");
    return 2;
  }
  if (args.play_speed <= 0.0) {
    std::fprintf(stderr, "--play-speed must be positive\n");
    return 2;
  }

  const long origin_ms = now_ms();
  std::vector<PhaseTiming> setup_phases;
  auto decode_ctx = load_single_buffer_context(args, manifest, setup_phases);

  EventWriter events(args.stream_events_path);
  std::ofstream phases(args.phase_stats_path);
  phases << "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
            "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
            "kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx\n";
  phases << "# StreamFrameEnqueue: sampled frame enters buffer  SingleBufferUpdate: current image buffer update  "
            "StreamPromptPrefill/StreamDecode: prompt handled against current buffered image\n";
  for (const auto& setup : setup_phases) {
    append_phase_row(phases, setup.name, setup.start_ms, setup.end_ms, origin_ms);
  }
  struct PromptJob {
    FrameRecord frame;
    PromptEvent prompt;
    int prompt_idx = -1;
    long event_ms = 0;
  };
  std::deque<PromptJob> prompt_jobs;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;

  std::thread producer([&]() {
    double last_ts = 0.0;
    size_t prompt_cursor = 0;
    FrameRecord current_frame;
    bool have_current_frame = false;
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
      {
        std::lock_guard<std::mutex> lock(mu);
        while (prompt_cursor < manifest.prompts.size() && manifest.prompts[prompt_cursor].timestamp_s <= frame.timestamp_s) {
          prompt_jobs.push_back(PromptJob{
              current_frame,
              manifest.prompts[prompt_cursor],
              static_cast<int>(prompt_cursor),
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
        prompt_jobs.push_back(PromptJob{
            current_frame,
            manifest.prompts[prompt_cursor],
            static_cast<int>(prompt_cursor),
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

  while (true) {
    PromptJob job;
    {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [&]() { return done || !prompt_jobs.empty(); });
      if (prompt_jobs.empty()) {
        if (done) {
          break;
        }
        continue;
      }
      job = prompt_jobs.front();
      prompt_jobs.pop_front();
    }

    events.row(
        "StreamPromptPrefill",
        job.frame.index,
        job.prompt_idx,
        job.prompt.timestamp_s,
        origin_ms,
        job.event_ms,
        job.event_ms,
        job.prompt.prompt);
    append_phase_row(phases, "StreamPromptPrefill", job.event_ms, job.event_ms, origin_ms);
    const long decode_start_ms = now_ms();
    const int rc = run_single_buffer_prompt(args, *decode_ctx, job.frame, job.prompt, job.prompt_idx, origin_ms);
    const long decode_end_ms = now_ms();
    append_phase_file(phases, prompt_phase_path(job.prompt_idx));
    const std::string response_path = "stream_response_" + std::to_string(job.prompt_idx) + ".txt";
    const std::string image = job.frame.tiles.empty() ? "" : job.frame.tiles.front().layout_image;
    append_file_to_output(
        args.output_path,
        response_path,
        job.prompt_idx,
        job.prompt.timestamp_s,
        image,
        job.prompt.prompt);
    events.row(
        "StreamDecode",
        job.frame.index,
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
