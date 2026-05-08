#include "arg.h"
#include "chat.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include "foundation_token_io_format.hpp"
#include "inference_trace.hpp"

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

struct custom_args {
  std::string phase_stats_path = "foundation_phase_stats.csv";
  std::string token_io_path;
  bool force_generation = false;
  std::vector<std::string> passthrough;
};

custom_args strip_custom_args(int argc, char** argv) {
  custom_args out;
  out.passthrough.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--phase-stats-path") {
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
    } else if (arg == "--force-generation") {
      out.force_generation = true;
    } else {
      out.passthrough.push_back(std::move(arg));
    }
  }
  return out;
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
      out << "# L_DecoderRuntimeInit: llama.cpp args/OpenCL runtime init  "
             "L_DecoderLoad: llama.cpp text model/context/mmproj load  "
             "ImageLoad: input image load  LayoutTokenize: mtmd text/image layout  "
             "V_Encode: llama.cpp OpenCL vision encode/projector  "
             "ImagePrefill: projected image embedding prefill  "
             "T_Prefill: text prompt prefill  D: one generated-token decode\n";
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
  bool use_jinja = false;
  llama_batch batch;
  int n_batch = 0;
  llama_pos n_past = 0;
  std::optional<common_debug_cb_user_data> mtmd_debug_graph_cb;

  explicit decode_context(common_params& params) : llama_init(common_init_from_params(params)) {
    model = llama_init->model();
    lctx = llama_init->context();
    vocab = llama_model_get_vocab(model);
    smpl = common_sampler_init(model, params.sampling);
    batch = llama_batch_init(1, 0, 1);
    n_batch = params.n_batch;
    if (!model || !lctx) {
      die("failed to initialize llama model/context");
    }
    if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
      die("model does not have chat template");
    }
    tmpls = common_chat_templates_init(model, params.chat_template);
    use_jinja = params.use_jinja;

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params.mmproj_use_gpu;
    mparams.print_timings = true;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn_type;
    mparams.warmup = params.warmup;
    mparams.image_min_tokens = params.image_min_tokens;
    mparams.image_max_tokens = params.image_max_tokens;
    if (std::getenv("MTMD_DEBUG_GRAPH") != nullptr) {
      mtmd_debug_graph_cb.emplace();
      mparams.cb_eval_user_data = &*mtmd_debug_graph_cb;
      mparams.cb_eval = common_debug_cb_eval;
    }
    ctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!ctx_vision.get()) {
      die_fmt("failed to load vision model from %s", params.mmproj.path.c_str());
    }
  }

  ~decode_context() {
    llama_batch_free(batch);
    common_sampler_free(smpl);
  }
};

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

std::string chat_add_and_format(decode_context& ctx, common_chat_msg& new_msg) {
  auto formatted = common_chat_format_single(
      ctx.tmpls.get(),
      ctx.chat_history,
      new_msg,
      new_msg.role == "user",
      ctx.use_jinja);
  ctx.chat_history.push_back(new_msg);
  return formatted;
}

int eval_message(
    decode_context& ctx,
    common_chat_msg& msg,
    const std::vector<std::string>& images,
    phase_recorder& phases,
    std::string* input_special_text,
    std::size_t* out_image_placeholder_slots,
    streamingvlm::hybrid_bridge::inference_trace_collector* trace) {
  bool add_bos = ctx.chat_history.empty();
  auto formatted_chat = chat_add_and_format(ctx, msg);

  mtmd::bitmaps bitmaps;
  for (const auto& image : images) {
    const long image_load_start_ms = ggml_time_ms();
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), image.c_str()));
    const long image_load_end_ms = ggml_time_ms();
    phases.row("ImageLoad", image_load_start_ms, image_load_end_ms);
    if (!bmp.ptr) {
      die_fmt("failed to load image: %s", image.c_str());
    }
    bitmaps.entries.push_back(std::move(bmp));
  }

  mtmd_input_text text;
  text.text = formatted_chat.c_str();
  text.add_special = add_bos;
  text.parse_special = true;

  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  auto bitmaps_c_ptr = bitmaps.c_ptr();
  const long layout_start_ms = ggml_time_ms();
  int32_t res = mtmd_tokenize(
      ctx.ctx_vision.get(),
      chunks.ptr.get(),
      &text,
      bitmaps_c_ptr.data(),
      bitmaps_c_ptr.size());
  const long layout_end_ms = ggml_time_ms();
  phases.row("LayoutTokenize", layout_start_ms, layout_end_ms);
  if (res != 0) {
    die_fmt("mtmd_tokenize failed: %d", res);
  }
  if (input_special_text != nullptr) {
    *input_special_text = render_chunks_with_special_tokens(ctx, chunks);
  }

  if (out_image_placeholder_slots != nullptr) {
    *out_image_placeholder_slots = 0;
    const size_t n_img_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    for (size_t i = 0; i < n_img_chunks; ++i) {
      const mtmd_input_chunk* pch = mtmd_input_chunks_get(chunks.ptr.get(), i);
      if (mtmd_input_chunk_get_type(pch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        *out_image_placeholder_slots += mtmd_input_chunk_get_n_tokens(pch);
      }
    }
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

  std::vector<std::vector<float>> encoded_image_embeddings(n_chunks);
  const int64_t decoder_embedding_size = llama_model_n_embd_inp(ctx.model);
  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const auto chunk_type = mtmd_input_chunk_get_type(chunk);
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      const long vision_encode_start_ms = ggml_time_ms();
      LOG_INF("encoding image slice...\n");
      if (mtmd_encode_chunk(ctx.ctx_vision.get(), chunk) != 0) {
        die("failed to encode image slice");
      }
      const long vision_encode_end_ms = ggml_time_ms();
      LOG_INF("image slice encoded in %ld ms\n", vision_encode_end_ms - vision_encode_start_ms);
      phases.row("V_Encode", vision_encode_start_ms, vision_encode_end_ms);

      float* embd = mtmd_get_output_embd(ctx.ctx_vision.get());
      const size_t n_values = static_cast<size_t>(decoder_embedding_size) * mtmd_input_chunk_get_n_tokens(chunk);
      encoded_image_embeddings[i].assign(embd, embd + n_values);
    }
  }

  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const bool logits_last = i == n_chunks - 1;
    llama_pos new_n_past = ctx.n_past;
    const auto chunk_type = mtmd_input_chunk_get_type(chunk);
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
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
    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      if (encoded_image_embeddings[i].empty()) {
        die("missing precomputed image embedding");
      }
      const long image_prefill_start_ms = ggml_time_ms();
      if (mtmd_helper_decode_image_chunk(
              ctx.ctx_vision.get(),
              ctx.lctx,
              chunk,
              encoded_image_embeddings[i].data(),
              ctx.n_past,
              0,
              ctx.n_batch,
              &new_n_past) != 0) {
        die("failed to decode image chunk");
      }
      const long image_prefill_end_ms = ggml_time_ms();
      phases.row("ImagePrefill", image_prefill_start_ms, image_prefill_end_ms);
    } else {
      die("unsupported non-image multimodal chunk");
    }
    ctx.n_past = new_n_past;
  }
  LOG("\n");
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
      "Usage: %s -m <model.gguf> --mmproj <mmproj.gguf> --image <image> -p <prompt> [llama.cpp opts]\n",
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
  if (params.mmproj.path.empty() || params.model.path.empty() || params.image.empty()) {
    show_usage(argc, argv);
    return 1;
  }

  phase_recorder phases(custom.phase_stats_path, origin_ms);
  phases.row("L_DecoderRuntimeInit", params_parse_start_ms, ggml_time_ms());
  const long load_start_ms = ggml_time_ms();
  decode_context ctx(params);
  const long load_end_ms = ggml_time_ms();
  phases.row("L_DecoderLoad", load_start_ms, load_end_ms);

  const std::string export_plain_prompt = params.prompt;

  if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
    for (size_t i = 0; i < params.image.size(); ++i) {
      params.prompt = mtmd_default_marker() + params.prompt;
    }
  }

  common_chat_msg msg;
  msg.role = "user";
  msg.content = params.prompt;

  std::unique_ptr<streamingvlm::hybrid_bridge::inference_trace_collector> trace_writer;
  if (!custom.token_io_path.empty()) {
    trace_writer = std::make_unique<streamingvlm::hybrid_bridge::inference_trace_collector>(
        streamingvlm::hybrid_bridge::sibling_foundation_inference_tokens_path(custom.token_io_path));
  }

  const std::string hf_q =
      streamingvlm::hybrid_bridge::internvl_hf_official_question_single_image(export_plain_prompt);

  if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
    trace_writer->write_hf_reference_question_literal(hf_q);
    trace_writer->write_hf_official_user_segment_reference(ctx.lctx, hf_q);
  }

  if (
      eval_message(
          ctx,
          msg,
          params.image,
          phases,
          nullptr,
          nullptr,
          trace_writer.get()) != 0) {
    return 1;
  }
  int n_predict = params.n_predict < 0 ? INT32_MAX : params.n_predict;
  const std::string generated_text = generate_response(ctx, n_predict, custom.force_generation, phases, trace_writer.get());
  std::string token_io_doc =
      streamingvlm::hybrid_bridge::build_hf_user_assistant_echo(hf_q, generated_text) + "\n";
  if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
    token_io_doc += trace_writer->format_token_io_appendix();
  }
  write_text_file(custom.token_io_path, token_io_doc);
  LOG("\n\n");
  llama_perf_context_print(ctx.lctx);
  return 0;
}
