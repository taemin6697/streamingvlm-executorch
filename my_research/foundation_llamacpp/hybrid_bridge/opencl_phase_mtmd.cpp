#include "arg.h"
#include "chat.h"
#include "common.h"
#include "debug.h"
#include "hybrid_embedding_file.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include "inference_trace.hpp"
#include "phase_trace.hpp"

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
  std::string warmup_image_path;
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
    } else if (arg == "--warmup-image") {
      if (i + 1 >= argc) {
        die("missing value for --warmup-image");
      }
      out.warmup_image_path = argv[++i];
    } else if (arg.rfind("--warmup-image=", 0) == 0) {
      out.warmup_image_path = arg.substr(std::string("--warmup-image=").size());
    } else if (arg == "--force-generation") {
      out.force_generation = true;
    } else {
      out.passthrough.push_back(std::move(arg));
    }
  }
  return out;
}

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

void warmup_split_encoder_with_image(decode_context& ctx, const std::string& image_path) {
  if (image_path.empty()) {
    return;
  }
  mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), image_path.c_str()));
  if (!bmp.ptr) {
    die_fmt("failed to load warmup image: %s", image_path.c_str());
  }
  bmp.set_id("warmup_image");
  mtmd::bitmaps bitmaps;
  bitmaps.entries.push_back(std::move(bmp));

  mtmd_input_text text;
  std::string warmup_text = mtmd_default_marker();
  text.text = warmup_text.c_str();
  text.add_special = true;
  text.parse_special = true;
  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  auto bitmaps_c_ptr = bitmaps.c_ptr();
  if (mtmd_tokenize(
          ctx.ctx_vision.get(),
          chunks.ptr.get(),
          &text,
          bitmaps_c_ptr.data(),
          bitmaps_c_ptr.size()) != 0) {
    die("failed to tokenize warmup image");
  }
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      continue;
    }
    int64_t warmup_vision_ms = 0;
    int64_t warmup_projector_ms = 0;
    if (mtmd_encode_chunk_split_timing(
            ctx.ctx_vision.get(),
            chunk,
            &warmup_vision_ms,
            &warmup_projector_ms) != 0) {
      die("failed to warm up split image encode");
    }
    LOG_INF(
        "warmed split image encode with %s: vision=%lld ms, mmproj=%lld ms\n",
        image_path.c_str(),
        static_cast<long long>(warmup_vision_ms),
        static_cast<long long>(warmup_projector_ms));
    return;
  }
  die("warmup image did not produce an image chunk");
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
    streamingvlm::hybrid_bridge::phase_recorder& phases,
    std::string* input_special_text,
    std::size_t* out_image_placeholder_slots,
    streamingvlm::hybrid_bridge::inference_trace_collector* trace) {
  bool add_bos = ctx.chat_history.empty();
  auto formatted_chat = chat_add_and_format(ctx, msg);

  mtmd::bitmaps bitmaps;
  for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    const auto& image = images[image_idx];
    const long image_load_start_ms = ggml_time_ms();
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), image.c_str()));
    const long image_load_end_ms = ggml_time_ms();
    phases.row("ImageLoad", image_load_start_ms, image_load_end_ms);
    if (!bmp.ptr) {
      die_fmt("failed to load image: %s", image.c_str());
    }
    const std::string bitmap_id = "image_" + std::to_string(image_idx + 1);
    bmp.set_id(bitmap_id.c_str());
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
    size_t image_trace_idx = 0;
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
        trace->chunk_image_begin(
            ci,
            mtmd_input_chunk_get_n_tokens(ch),
            mtmd_input_chunk_get_id(ch),
            image_trace_idx++);
      }
    }
  }

  std::vector<std::vector<float>> encoded_image_embeddings(n_chunks);
  const int64_t decoder_embedding_size = llama_model_n_embd_inp(ctx.model);
  std::vector<float> projected_embedding_dump;
  size_t n_image_chunks = 0;
  size_t image_tokens_per_chunk = 0;
  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const auto chunk_type = mtmd_input_chunk_get_type(chunk);
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      const long vision_encode_start_ms = ggml_time_ms();
      LOG_INF("encoding image slice...\n");
      int64_t vision_ms = 0;
      int64_t projector_ms = 0;
      if (mtmd_encode_chunk_split_timing(ctx.ctx_vision.get(), chunk, &vision_ms, &projector_ms) != 0) {
        die("failed to encode image slice");
      }
      const long vision_encode_end_ms = ggml_time_ms();
      LOG_INF("image slice encoded in %ld ms\n", vision_encode_end_ms - vision_encode_start_ms);
      const long projector_start_ms = vision_encode_end_ms - static_cast<long>(projector_ms);
      const long vision_end_ms = projector_start_ms;
      phases.row("V_Encode", vision_encode_start_ms, vision_end_ms);
      phases.row("Mmproj", projector_start_ms, vision_encode_end_ms);

      float* embd = mtmd_get_output_embd(ctx.ctx_vision.get());
      const size_t n_values = static_cast<size_t>(decoder_embedding_size) * mtmd_input_chunk_get_n_tokens(chunk);
      encoded_image_embeddings[i].assign(embd, embd + n_values);
      projected_embedding_dump.insert(projected_embedding_dump.end(), embd, embd + n_values);
      n_image_chunks += 1;
      image_tokens_per_chunk = mtmd_input_chunk_get_n_tokens(chunk);
    }
  }
  if (!projected_embedding_dump.empty()) {
    streamingvlm::hybrid_bridge::write_embedding_file(
        "opencl_projected_embedding.svlmemb",
        {
            static_cast<int64_t>(n_image_chunks),
            static_cast<int64_t>(image_tokens_per_chunk),
            decoder_embedding_size,
        },
        projected_embedding_dump.data(),
        projected_embedding_dump.size());
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
      llama_synchronize(ctx.lctx);
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
      llama_synchronize(ctx.lctx);
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
    streamingvlm::hybrid_bridge::phase_recorder& phases,
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
    llama_synchronize(ctx.lctx);
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

#ifndef STREAMINGVLM_OPENCL_PHASE_MTMD_NO_MAIN
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

  streamingvlm::hybrid_bridge::phase_recorder phases(
      custom.phase_stats_path,
      origin_ms,
      streamingvlm::hybrid_bridge::opencl_phase_description());
  phases.row("L_DecoderRuntimeInit", params_parse_start_ms, ggml_time_ms());
  const long load_start_ms = ggml_time_ms();
  decode_context ctx(params);
  const long load_end_ms = ggml_time_ms();
  phases.row("L_DecoderLoad", load_start_ms, load_end_ms);
  warmup_split_encoder_with_image(ctx, custom.warmup_image_path);

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
      std::string("User: ") + export_plain_prompt + "\nAssistant: " + generated_text + "\n";
  if (trace_writer != nullptr && static_cast<bool>(*trace_writer)) {
    token_io_doc += trace_writer->format_token_io_appendix();
  }
  write_text_file(custom.token_io_path, token_io_doc);
  LOG("\n\n");
  llama_perf_context_print(ctx.lctx);
  return 0;
}
#endif
