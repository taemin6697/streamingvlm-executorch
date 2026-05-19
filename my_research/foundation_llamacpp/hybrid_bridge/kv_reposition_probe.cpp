#include "kv_reposition.hpp"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model;
  int threads = 8;
  int ctx_size = 1024;
  int n_predict = 24;
  int top_k = 8;
  std::string prefix =
      "Context: alpha means apple. This fact is important.\n";
  std::string removed =
      "Old video filler: beta means banana. beta means banana. beta means banana.\n";
  std::string history =
      "Conversation history: The user previously asked about alpha.\n";
  std::string suffix =
      "Question: What did the user ask about earlier?\nAnswer:";
};

struct LlamaModelDeleter {
  void operator()(llama_model* model) const {
    llama_model_free(model);
  }
};

struct LlamaContextDeleter {
  void operator()(llama_context* ctx) const {
    llama_free(ctx);
  }
};

using ModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

void usage(const char* argv0) {
  std::fprintf(stderr,
      "usage: %s --model MODEL.gguf [--threads N] [--ctx-size N] [--n-predict N]\n",
      argv0);
}

bool parse_int_arg(int argc, char** argv, int& i, int* out) {
  if (i + 1 >= argc) {
    return false;
  }
  *out = std::atoi(argv[++i]);
  return true;
}

bool parse_string_arg(int argc, char** argv, int& i, std::string* out) {
  if (i + 1 >= argc) {
    return false;
  }
  *out = argv[++i];
  return true;
}

bool parse_args(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" || arg == "-m") {
      if (!parse_string_arg(argc, argv, i, &args->model)) {
        return false;
      }
    } else if (arg == "--threads") {
      if (!parse_int_arg(argc, argv, i, &args->threads)) {
        return false;
      }
    } else if (arg == "--ctx-size" || arg == "-c") {
      if (!parse_int_arg(argc, argv, i, &args->ctx_size)) {
        return false;
      }
    } else if (arg == "--n-predict" || arg == "-n") {
      if (!parse_int_arg(argc, argv, i, &args->n_predict)) {
        return false;
      }
    } else if (arg == "--top-k") {
      if (!parse_int_arg(argc, argv, i, &args->top_k)) {
        return false;
      }
    } else if (arg == "--prefix") {
      if (!parse_string_arg(argc, argv, i, &args->prefix)) {
        return false;
      }
    } else if (arg == "--removed") {
      if (!parse_string_arg(argc, argv, i, &args->removed)) {
        return false;
      }
    } else if (arg == "--history") {
      if (!parse_string_arg(argc, argv, i, &args->history)) {
        return false;
      }
    } else if (arg == "--suffix") {
      if (!parse_string_arg(argc, argv, i, &args->suffix)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return !args->model.empty() && args->threads > 0 && args->ctx_size > 0 && args->n_predict >= 0;
}

std::vector<llama_token> tokenize(
    const llama_vocab* vocab,
    const std::string& text,
    bool add_special) {
  const int n = -llama_tokenize(
      vocab,
      text.c_str(),
      static_cast<int32_t>(text.size()),
      nullptr,
      0,
      add_special,
      true);
  std::vector<llama_token> tokens(n);
  const int got = llama_tokenize(
      vocab,
      text.c_str(),
      static_cast<int32_t>(text.size()),
      tokens.data(),
      static_cast<int32_t>(tokens.size()),
      add_special,
      true);
  if (got < 0 || got != n) {
    std::fprintf(stderr, "failed to tokenize text segment\n");
    std::exit(2);
  }
  return tokens;
}

std::string piece(const llama_vocab* vocab, llama_token token) {
  char buf[256];
  const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
  if (n < 0) {
    return "<piece-error>";
  }
  return std::string(buf, n);
}

std::vector<llama_token> concat(
    const std::vector<llama_token>& a,
    const std::vector<llama_token>& b,
    const std::vector<llama_token>& c) {
  std::vector<llama_token> out;
  out.reserve(a.size() + b.size() + c.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  out.insert(out.end(), c.begin(), c.end());
  return out;
}

std::vector<llama_token> concat(
    const std::vector<llama_token>& a,
    const std::vector<llama_token>& b) {
  std::vector<llama_token> out;
  out.reserve(a.size() + b.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  return out;
}

bool eval_tokens(
    llama_context* ctx,
    const std::vector<llama_token>& tokens,
    llama_pos start_pos,
    bool logits_last) {
  if (tokens.empty()) {
    return true;
  }

  llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
  batch.n_tokens = static_cast<int32_t>(tokens.size());
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = start_pos + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = logits_last && i == batch.n_tokens - 1;
  }
  const int ret = llama_decode(ctx, batch);
  llama_batch_free(batch);
  return ret == 0;
}

ModelPtr load_model(const Args& args) {
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  return ModelPtr(llama_model_load_from_file(args.model.c_str(), model_params));
}

ContextPtr make_context(llama_model* model, const Args& args) {
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = args.ctx_size;
  ctx_params.n_batch = args.ctx_size;
  ctx_params.n_ubatch = std::min(args.ctx_size, 512);
  ctx_params.no_perf = false;
  ContextPtr ctx(llama_init_from_model(model, ctx_params));
  if (ctx) {
    llama_set_n_threads(ctx.get(), args.threads, args.threads);
  }
  return ctx;
}

std::vector<int> top_token_ids(const float* logits, int vocab_size, int top_k) {
  std::vector<int> ids(vocab_size);
  std::iota(ids.begin(), ids.end(), 0);
  const int k = std::min(top_k, vocab_size);
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
    return logits[a] > logits[b];
  });
  ids.resize(k);
  return ids;
}

double rms_logits_delta(const float* a, const float* b, int n) {
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum_sq += d * d;
  }
  return std::sqrt(sum_sq / std::max(1, n));
}

llama_token greedy_sample(llama_context* ctx) {
  const float* logits = llama_get_logits_ith(ctx, -1);
  const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
  return static_cast<llama_token>(
      std::max_element(logits, logits + vocab_size) - logits);
}

std::string generate_greedy(
    llama_context* ctx,
    const llama_vocab* vocab,
    llama_pos* n_past,
    int n_predict) {
  std::string out;
  for (int i = 0; i < n_predict; ++i) {
    const llama_token token = greedy_sample(ctx);
    if (llama_vocab_is_eog(vocab, token)) {
      break;
    }
    out += piece(vocab, token);
    if (!eval_tokens(ctx, std::vector<llama_token>{token}, *n_past, true)) {
      std::fprintf(stderr, "failed to decode generated token\n");
      std::exit(2);
    }
    ++*n_past;
  }
  return out;
}

void print_top_tokens(
    const char* label,
    const llama_vocab* vocab,
    const float* logits,
    int vocab_size,
    int top_k) {
  std::printf("%s\n", label);
  const std::vector<int> ids = top_token_ids(logits, vocab_size, top_k);
  for (int id : ids) {
    std::printf("  id=%d logit=%.6f piece=%s\n", id, logits[id], piece(vocab, id).c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, &args)) {
    usage(argv[0]);
    return 2;
  }

  llama_backend_init();
  ggml_backend_load_all();
  llama_log_set([](ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
      std::fprintf(stderr, "%s", text);
    }
  }, nullptr);

  ModelPtr model = load_model(args);
  if (!model) {
    std::fprintf(stderr, "failed to load model: %s\n", args.model.c_str());
    return 2;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  const int vocab_size = llama_vocab_n_tokens(vocab);

  const std::vector<llama_token> prefix = tokenize(vocab, args.prefix, true);
  const std::vector<llama_token> removed = tokenize(vocab, args.removed, false);
  const std::vector<llama_token> history = tokenize(vocab, args.history, false);
  const std::vector<llama_token> suffix = tokenize(vocab, args.suffix, false);

  const std::vector<llama_token> original_base = concat(prefix, removed, history);
  const std::vector<llama_token> compact_base = concat(prefix, history);

  const llama_pos removed_begin = static_cast<llama_pos>(prefix.size());
  const llama_pos removed_end = static_cast<llama_pos>(prefix.size() + removed.size());
  const llama_pos original_end = static_cast<llama_pos>(original_base.size());

  streamingvlm::hybrid_bridge::KvTailCompactionPlan plan;
  std::string error;
  if (!streamingvlm::hybrid_bridge::build_tail_compaction_plan(
          streamingvlm::hybrid_bridge::KvTokenRange{removed_begin, removed_end},
          original_end,
          &plan,
          &error)) {
    std::fprintf(stderr, "failed to build compaction plan: %s\n", error.c_str());
    return 2;
  }

  ContextPtr reference_ctx = make_context(model.get(), args);
  ContextPtr shifted_ctx = make_context(model.get(), args);
  if (!reference_ctx || !shifted_ctx) {
    std::fprintf(stderr, "failed to create llama context\n");
    return 2;
  }

  if (!eval_tokens(reference_ctx.get(), compact_base, 0, false)) {
    std::fprintf(stderr, "failed to eval compact reference base\n");
    return 2;
  }
  if (!eval_tokens(reference_ctx.get(), suffix, static_cast<llama_pos>(compact_base.size()), true)) {
    std::fprintf(stderr, "failed to eval compact reference suffix\n");
    return 2;
  }
  std::vector<float> reference_logits(
      llama_get_logits_ith(reference_ctx.get(), -1),
      llama_get_logits_ith(reference_ctx.get(), -1) + vocab_size);

  if (!eval_tokens(shifted_ctx.get(), original_base, 0, false)) {
    std::fprintf(stderr, "failed to eval original base\n");
    return 2;
  }
  if (!streamingvlm::hybrid_bridge::apply_tail_compaction_plan(
          llama_get_memory(shifted_ctx.get()),
          0,
          plan,
          &error)) {
    std::fprintf(stderr, "failed to apply compaction plan: %s\n", error.c_str());
    return 2;
  }
  if (!eval_tokens(shifted_ctx.get(), suffix, plan.compacted_sequence_end, true)) {
    std::fprintf(stderr, "failed to eval suffix after KV reposition\n");
    return 2;
  }
  std::vector<float> shifted_logits(
      llama_get_logits_ith(shifted_ctx.get(), -1),
      llama_get_logits_ith(shifted_ctx.get(), -1) + vocab_size);

  llama_pos reference_n_past = static_cast<llama_pos>(compact_base.size() + suffix.size());
  llama_pos shifted_n_past = plan.compacted_sequence_end + static_cast<llama_pos>(suffix.size());
  const std::string reference_answer =
      generate_greedy(reference_ctx.get(), vocab, &reference_n_past, args.n_predict);
  const std::string shifted_answer =
      generate_greedy(shifted_ctx.get(), vocab, &shifted_n_past, args.n_predict);

  const llama_token ref_top = static_cast<llama_token>(
      std::max_element(reference_logits.begin(), reference_logits.end()) - reference_logits.begin());
  const llama_token shifted_top = static_cast<llama_token>(
      std::max_element(shifted_logits.begin(), shifted_logits.end()) - shifted_logits.begin());

  std::printf("prefix_tokens=%zu removed_tokens=%zu history_tokens=%zu suffix_tokens=%zu\n",
      prefix.size(),
      removed.size(),
      history.size(),
      suffix.size());
  std::printf("removed_range=[%d,%d) original_end=%d compacted_end=%d shift=%d\n",
      static_cast<int>(plan.removed.begin),
      static_cast<int>(plan.removed.end),
      static_cast<int>(plan.sequence_end),
      static_cast<int>(plan.compacted_sequence_end),
      static_cast<int>(plan.shift));
  std::printf("reference_top=%d %s\n", ref_top, piece(vocab, ref_top).c_str());
  std::printf("shifted_top=%d %s\n", shifted_top, piece(vocab, shifted_top).c_str());
  std::printf("top1_match=%s\n", ref_top == shifted_top ? "true" : "false");
  std::printf("logits_rms_delta=%.9f\n",
      rms_logits_delta(reference_logits.data(), shifted_logits.data(), vocab_size));
  print_top_tokens("reference_top_tokens:", vocab, reference_logits.data(), vocab_size, args.top_k);
  print_top_tokens("shifted_top_tokens:", vocab, shifted_logits.data(), vocab_size, args.top_k);
  std::printf("reference_answer:%s\n", reference_answer.c_str());
  std::printf("shifted_answer:%s\n", shifted_answer.c_str());

  llama_backend_free();
  return 0;
}
