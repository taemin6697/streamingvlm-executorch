#include "hybrid_embedding_file.h"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include "inference_trace.hpp"

#include <clocale>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct custom_args {
  std::string embedding_path;
  std::string phase_stats_path = "decoder_phase_stats.csv";
  std::string token_io_path;
  std::string ready_path;
  bool wait_for_embedding = false;
  bool force_generation = false;
  int wait_timeout_ms = 120000;
  std::vector<std::string> passthrough;
};

custom_args strip_custom_args(int argc, char** argv) {
  custom_args out;
  out.passthrough.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--external-embedding" || arg == "--embedding-file") {
      if (i + 1 >= argc) {
        die("missing value for --external-embedding");
      }
      out.embedding_path = argv[++i];
    } else if (arg.rfind("--external-embedding=", 0) == 0) {
      out.embedding_path = arg.substr(std::string("--external-embedding=").size());
    } else if (arg.rfind("--embedding-file=", 0) == 0) {
      out.embedding_path = arg.substr(std::string("--embedding-file=").size());
    } else if (arg == "--phase-stats-path") {
      if (i + 1 >= argc) {
        die("missing value for --phase-stats-path");
      }
      out.phase_stats_path = argv[++i];
    } else if (arg.rfind("--phase-stats-path=", 0) == 0) {
      out.phase_stats_path = arg.substr(std::string("--phase-stats-path=").size());
    } else if (arg == "--token-io-path") {
      if (i + 1 >= argc) {
        die("missing value for --token-io-path");
      }
      out.token_io_path = argv[++i];
    } else if (arg.rfind("--token-io-path=", 0) == 0) {
      out.token_io_path = arg.substr(std::string("--token-io-path=").size());
    } else if (arg == "--ready-path") {
      if (i + 1 >= argc) {
        die("missing value for --ready-path");
      }
      out.ready_path = argv[++i];
    } else if (arg.rfind("--ready-path=", 0) == 0) {
      out.ready_path = arg.substr(std::string("--ready-path=").size());
    } else if (arg == "--wait-for-embedding") {
      out.wait_for_embedding = true;
    } else if (arg == "--force-generation") {
      out.force_generation = true;
    } else if (arg == "--wait-timeout-ms") {
      if (i + 1 >= argc) {
        die("missing value for --wait-timeout-ms");
      }
      out.wait_timeout_ms = std::atoi(argv[++i]);
    } else if (arg.rfind("--wait-timeout-ms=", 0) == 0) {
      out.wait_timeout_ms = std::atoi(arg.substr(std::string("--wait-timeout-ms=").size()).c_str());
    } else {
      out.passthrough.push_back(std::move(arg));
    }
  }
  return out;
}

void write_text_file(const std::string& path, const std::string& value) {
  if (path.empty()) {
    return;
  }
  std::ofstream out(path);
  if (!out.is_open()) {
    die_fmt("failed to write file: %s", path.c_str());
  }
  out << value;
}

void wait_for_file(const std::string& path, int timeout_ms) {
  const long start_ms = ggml_time_ms();
  while (true) {
    std::ifstream in(path);
    if (in.good()) {
      return;
    }
    if (ggml_time_ms() - start_ms > timeout_ms) {
      die_fmt("timed out waiting for file: %s", path.c_str());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

struct phase_recorder {
  long origin_ms = 0;
  std::ofstream out;

  explicit phase_recorder(const std::string& path, long origin) : origin_ms(origin) {
    if (!path.empty()) {
      out.open(path);
      if (!out.is_open()) {
        die_fmt("failed to open phase stats CSV: %s", path.c_str());
      }
      out << "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
             "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
             "kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx\n";
      out << "# L_DecoderLoad: llama.cpp model/context/mmproj load  "
             "ExternalEmbeddingRead: .svlmemb read  LayoutTokenize: mtmd text/image layout  "
             "Mmproj: OpenCL mmproj projection for QNN pre-projector features  "
             "ImagePrefill: external image embedding decode  T_Prefill: text prompt eval  "
             "D: one generated-token decode\n";
    }
  }

  void row(const char* row_type, long start_ms, long end_ms, int token_idx = 0) {
    if (!out.is_open()) {
      return;
    }
    const double start_s = static_cast<double>(start_ms - origin_ms) / 1000.0;
    const double end_s = static_cast<double>(end_ms - origin_ms) / 1000.0;
    const long total_ms = end_ms - start_ms;
    out << row_type << "," << start_s << "," << end_s
        << ",,," << total_ms << ",," << total_ms << ",,,,,,," << token_idx << "\n";
  }
};

struct decode_context {
  mtmd::context_ptr ctx_vision;
  common_init_result_ptr llama_init;
  llama_model* model = nullptr;
  llama_context* lctx = nullptr;
  const llama_vocab* vocab = nullptr;
  common_sampler* smpl = nullptr;
  common_chat_templates_ptr tmpls;
  std::vector<common_chat_msg> chat_history;
  llama_batch batch;
  int n_batch = 0;
  llama_pos n_past = 0;
  bool use_jinja = false;

  explicit decode_context(common_params& params)
      : llama_init(common_init_from_params(params)) {
    model = llama_init->model();
    lctx = llama_init->context();
    vocab = llama_model_get_vocab(model);
    smpl = common_sampler_init(model, params.sampling);
    batch = llama_batch_init(1, 0, 1);
    n_batch = params.n_batch;
    if (!model || !lctx) {
      std::exit(1);
    }

    tmpls = common_chat_templates_init(model, params.chat_template);
    use_jinja = params.use_jinja;

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params.mmproj_use_gpu;
    mparams.print_timings = true;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn_type;
    // Vision activations come from QNN (hybrid_vision_dump); do not run CLIP ViT warmup on OpenCL/mmproj.
    // `params.warmup` still applies to the text decoder via `common_init_from_params` above.
    mparams.warmup = false;
    mparams.image_min_tokens = params.image_min_tokens;
    mparams.image_max_tokens = params.image_max_tokens;
    ctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!ctx_vision.get()) {
      die_fmt("failed to load mmproj: %s", params.mmproj.path.c_str());
    }
  }

  ~decode_context() {
    llama_batch_free(batch);
    common_sampler_free(smpl);
  }
};

std::string chat_add_and_format(decode_context& ctx, common_chat_msg& msg) {
  auto formatted = common_chat_format_single(
      ctx.tmpls.get(), ctx.chat_history, msg, msg.role == "user", ctx.use_jinja);
  ctx.chat_history.push_back(msg);
  return formatted;
}

std::string render_chunks_with_special_tokens(decode_context& ctx, mtmd::input_chunks& chunks) {
  std::string rendered;
  const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const auto chunk_type = mtmd_input_chunk_get_type(chunk);
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n_tokens = 0;
      const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
      llama_tokens token_vec(tokens, tokens + n_tokens);
      rendered += common_detokenize(ctx.vocab, token_vec, true);
    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
      for (size_t j = 0; j < n_tokens; ++j) {
        rendered += "<IMG_CONTEXT>";
      }
    }
  }
  return rendered;
}

int eval_with_external_embedding(
    decode_context& ctx,
    const std::string& prompt,
    const std::vector<std::string>& image_paths,
    streamingvlm::hybrid_bridge::EmbeddingFile& embedding,
    phase_recorder& phases,
    std::string* input_special_text,
    streamingvlm::hybrid_bridge::inference_trace_collector* trace) {
  if (prompt.empty()) {
    die("prompt is required");
  }
  if (image_paths.empty()) {
    die("at least one --image is required to create mtmd image tokens");
  }

  std::string content = prompt;
  if (content.find(mtmd_default_marker()) == std::string::npos) {
    for (size_t i = 0; i < image_paths.size(); ++i) {
      content = std::string(mtmd_default_marker()) + content;
    }
  }

  const long layout_start_ms = ggml_time_ms();
  mtmd::bitmaps bitmaps;
  for (const auto& image : image_paths) {
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), image.c_str()));
    if (!bmp.ptr) {
      die_fmt("failed to load image for token layout: %s", image.c_str());
    }
    bitmaps.entries.push_back(std::move(bmp));
  }

  common_chat_msg msg;
  msg.role = "user";
  msg.content = content;
  std::string formatted = chat_add_and_format(ctx, msg);
  mtmd_input_text text{formatted.c_str(), ctx.chat_history.size() == 1, true};
  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  auto bitmaps_c_ptr = bitmaps.c_ptr();
  int32_t tokenize_res = mtmd_tokenize(
      ctx.ctx_vision.get(),
      chunks.ptr.get(),
      &text,
      bitmaps_c_ptr.data(),
      bitmaps_c_ptr.size());
  if (tokenize_res != 0) {
    die_fmt("mtmd_tokenize failed: %d", tokenize_res);
  }
  const long layout_end_ms = ggml_time_ms();
  phases.row("LayoutTokenize", layout_start_ms, layout_end_ms);
  if (input_special_text != nullptr) {
    *input_special_text = render_chunks_with_special_tokens(ctx, chunks);
  }

  const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());

  if (trace != nullptr && static_cast<bool>(*trace)) {
    trace->write_prefill_header();
    for (size_t ci = 0; ci < n_chunks; ++ci) {
      const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks.ptr.get(), ci);
      const auto ctype = mtmd_input_chunk_get_type(ch);
      if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t nt = 0;
        const llama_token* toks = mtmd_input_chunk_get_tokens_text(ch, &nt);
        trace->chunk_text_begin(ci, nt);
        for (size_t ti = 0; ti < nt; ++ti) {
          const std::string piece = common_token_to_piece(ctx.lctx, toks[ti], true);
          trace->token_line(toks[ti], piece);
        }
      } else if (ctype == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        trace->chunk_image_begin(ci, mtmd_input_chunk_get_n_tokens(ch), mtmd_input_chunk_get_id(ch));
      }
    }
  }

  bool used_external_embedding = false;
  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const bool logits_last = i == n_chunks - 1;
    llama_pos new_n_past = ctx.n_past;
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
      const int32_t n_embd = llama_model_n_embd_inp(ctx.model);
      float* image_embedding = embedding.values.data();
      if (embedding.values.size() == n_tokens * static_cast<size_t>(n_embd)) {
        // Already projected into the decoder input embedding dimension.
        LOG_INF(
            "external embedding is already projected: tokens=%zu embd=%d floats=%zu\n",
            n_tokens,
            n_embd,
            embedding.values.size());
      } else {
        int32_t n_feature_embd = 0;
        if (embedding.shape.size() >= 3 && embedding.shape[embedding.shape.size() - 2] == static_cast<int64_t>(n_tokens)) {
          n_feature_embd = static_cast<int32_t>(embedding.shape.back());
        } else if (embedding.shape.size() == 2 && embedding.shape[0] == static_cast<int64_t>(n_tokens)) {
          n_feature_embd = static_cast<int32_t>(embedding.shape[1]);
        }
        if (n_feature_embd <= 0 || embedding.values.size() != n_tokens * static_cast<size_t>(n_feature_embd)) {
          die_fmt(
              "embedding size mismatch: file has %zu floats, image chunk expects %zu x %d projected embedding or valid pre-projector features",
              embedding.values.size(),
              n_tokens,
              n_embd);
        }
        LOG_INF(
            "external embedding is pre-projector: tokens=%zu feature_embd=%d projected_embd=%d floats=%zu\n",
            n_tokens,
            n_feature_embd,
            n_embd,
            embedding.values.size());
        const long mmproj_start_ms = ggml_time_ms();
        if (mtmd_project_features(
                ctx.ctx_vision.get(),
                embedding.values.data(),
                static_cast<int32_t>(n_tokens),
                n_feature_embd) != 0) {
          die("failed to project external vision features with mmproj");
        }
        const long mmproj_end_ms = ggml_time_ms();
        phases.row("Mmproj", mmproj_start_ms, mmproj_end_ms);
        image_embedding = mtmd_get_output_embd(ctx.ctx_vision.get());
      }
      if (image_embedding == nullptr) {
        die_fmt(
            "mmproj output is null for image chunk with %zu tokens",
            n_tokens);
      }
      const long image_prefill_start_ms = ggml_time_ms();
      if (mtmd_helper_decode_image_chunk(
              ctx.ctx_vision.get(),
              ctx.lctx,
              chunk,
              image_embedding,
              ctx.n_past,
              0,
              ctx.n_batch,
              &new_n_past) != 0) {
        die("failed to decode external image embedding");
      }
      const long image_prefill_end_ms = ggml_time_ms();
      phases.row("ImagePrefill", image_prefill_start_ms, image_prefill_end_ms);
      used_external_embedding = true;
    } else {
      const long text_prefill_start_ms = ggml_time_ms();
      if (mtmd_helper_eval_chunk_single(
              ctx.ctx_vision.get(),
              ctx.lctx,
              chunk,
              ctx.n_past,
              0,
              ctx.n_batch,
              logits_last,
              &new_n_past) != 0) {
        die("failed to eval text chunk");
      }
      const long text_prefill_end_ms = ggml_time_ms();
      phases.row("T_Prefill", text_prefill_start_ms, text_prefill_end_ms);
    }
    ctx.n_past = new_n_past;
  }
  if (!used_external_embedding) {
    die("prompt did not produce an image chunk");
  }
  return 0;
}

std::string generate_response(
    decode_context& ctx,
    int n_predict,
    bool force_generation,
    phase_recorder& phases,
    streamingvlm::hybrid_bridge::inference_trace_collector* trace) {
  std::string generated_text;
  if (trace != nullptr && static_cast<bool>(*trace)) {
    trace->decode_header();
  }
  for (int i = 0; i < n_predict; ++i) {
    llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
    common_sampler_accept(ctx.smpl, token_id, true);
    const std::string piece = common_token_to_piece(ctx.lctx, token_id, true);
    generated_text += piece;
    LOG("%s", piece.c_str());
    fflush(stdout);
    if (trace != nullptr && static_cast<bool>(*trace)) {
      trace->token_line(token_id, piece);
    }
    if (llama_vocab_is_eog(ctx.vocab, token_id) && !force_generation) {
      LOG("\n");
      break;
    }

    common_batch_clear(ctx.batch);
    common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
    const long token_decode_start_ms = ggml_time_ms();
    if (llama_decode(ctx.lctx, ctx.batch)) {
      die("failed to decode generated token");
    }
    const long token_decode_end_ms = ggml_time_ms();
    phases.row("D", token_decode_start_ms, token_decode_end_ms, i);
  }
  return generated_text;
}

void show_usage(int, char** argv) {
  LOG(
      "Usage: %s -m <model.gguf> --mmproj <mmproj.gguf> --image <layout-image> "
      "--external-embedding <vision_embedding.svlmemb> -p <prompt> [llama.cpp opts]\n",
      argv[0]);
}

} // namespace

int main(int argc, char** argv) {
  std::setlocale(LC_NUMERIC, "C");
  ggml_time_init();
  common_init();
  mtmd_helper_log_set(common_log_default_callback, nullptr);
  const long origin_ms = ggml_time_ms();

  custom_args custom = strip_custom_args(argc, argv);
  if (custom.embedding_path.empty()) {
    show_usage(argc, argv);
    die("missing --external-embedding");
  }
  std::vector<char*> passthrough_argv;
  passthrough_argv.reserve(custom.passthrough.size());
  for (auto& arg : custom.passthrough) {
    passthrough_argv.push_back(arg.data());
  }
  int passthrough_argc = static_cast<int>(passthrough_argv.size());

  const long params_parse_start_ms = ggml_time_ms();
  common_params params;
  if (!common_params_parse(
          passthrough_argc,
          passthrough_argv.data(),
          params,
          LLAMA_EXAMPLE_MTMD,
          show_usage)) {
    return 1;
  }
  if (params.mmproj.path.empty()) {
    die("missing --mmproj");
  }

  phase_recorder phases(custom.phase_stats_path, origin_ms);
  phases.row("L_DecoderRuntimeInit", params_parse_start_ms, ggml_time_ms());
  const long load_start_ms = ggml_time_ms();
  decode_context ctx(params);
  const long load_end_ms = ggml_time_ms();
  phases.row("L_DecoderLoad", load_start_ms, load_end_ms);
  write_text_file(custom.ready_path, "ready\n");
  if (custom.wait_for_embedding) {
    wait_for_file(custom.embedding_path, custom.wait_timeout_ms);
  }

  const long embedding_read_start_ms = ggml_time_ms();
  auto embedding = streamingvlm::hybrid_bridge::read_embedding_file(custom.embedding_path);
  const long embedding_read_end_ms = ggml_time_ms();
  phases.row("ExternalEmbeddingRead", embedding_read_start_ms, embedding_read_end_ms);

  std::unique_ptr<streamingvlm::hybrid_bridge::inference_trace_collector> trace_writer;
  if (!custom.token_io_path.empty()) {
    trace_writer = std::make_unique<streamingvlm::hybrid_bridge::inference_trace_collector>(
        streamingvlm::hybrid_bridge::sibling_foundation_inference_tokens_path(custom.token_io_path));
  }

  const std::string export_plain_prompt = params.prompt;
  if (
      eval_with_external_embedding(
          ctx, params.prompt, params.image, embedding, phases, nullptr, trace_writer.get()) != 0) {
    return 1;
  }
  int n_predict = params.n_predict < 0 ? INT32_MAX : params.n_predict;
  const std::string generated_text = generate_response(ctx, n_predict, custom.force_generation, phases, trace_writer.get());
  std::string token_io_doc =
      std::string("User: ") + export_plain_prompt + "\nAssistant: " + generated_text + "\n";
  if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
    token_io_doc += trace_writer->format_token_io_appendix();
  }
  write_text_file(custom.token_io_path, token_io_doc);
  LOG("\n\n");
  llama_perf_context_print(ctx.lctx);
  return 0;
}
