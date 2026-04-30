#include "hybrid_embedding_file.h"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct custom_args {
  std::string embedding_path;
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
    mparams.warmup = params.warmup;
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

int eval_with_external_embedding(
    decode_context& ctx,
    const std::string& prompt,
    const std::vector<std::string>& image_paths,
    streamingvlm::hybrid_bridge::EmbeddingFile& embedding) {
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

  bool used_external_embedding = false;
  const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    const bool logits_last = i == n_chunks - 1;
    llama_pos new_n_past = ctx.n_past;
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
      const int32_t n_embd = llama_model_n_embd_inp(ctx.model);
      if (embedding.values.size() != n_tokens * static_cast<size_t>(n_embd)) {
        die_fmt(
            "embedding size mismatch: file has %zu floats, image chunk expects %zu x %d",
            embedding.values.size(),
            n_tokens,
            n_embd);
      }
      if (mtmd_helper_decode_image_chunk(
              ctx.ctx_vision.get(),
              ctx.lctx,
              chunk,
              embedding.values.data(),
              ctx.n_past,
              0,
              ctx.n_batch,
              &new_n_past) != 0) {
        die("failed to decode external image embedding");
      }
      used_external_embedding = true;
    } else {
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
    }
    ctx.n_past = new_n_past;
  }
  if (!used_external_embedding) {
    die("prompt did not produce an image chunk");
  }
  return 0;
}

int generate_response(decode_context& ctx, int n_predict) {
  llama_tokens generated_tokens;
  for (int i = 0; i < n_predict; ++i) {
    llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
    generated_tokens.push_back(token_id);
    common_sampler_accept(ctx.smpl, token_id, true);
    if (llama_vocab_is_eog(ctx.vocab, token_id)) {
      LOG("\n");
      break;
    }
    LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
    fflush(stdout);

    common_batch_clear(ctx.batch);
    common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
    if (llama_decode(ctx.lctx, ctx.batch)) {
      die("failed to decode generated token");
    }
  }
  return 0;
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

  auto embedding = streamingvlm::hybrid_bridge::read_embedding_file(custom.embedding_path);
  decode_context ctx(params);
  if (eval_with_external_embedding(ctx, params.prompt, params.image, embedding) != 0) {
    return 1;
  }
  int n_predict = params.n_predict < 0 ? INT32_MAX : params.n_predict;
  generate_response(ctx, n_predict);
  LOG("\n\n");
  llama_perf_context_print(ctx.lctx);
  return 0;
}
