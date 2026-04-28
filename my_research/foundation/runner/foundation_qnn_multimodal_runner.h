/*
 * Local QNN multimodal runner overlay for foundation profiling.
 *
 * This mirrors the upstream QNNMultimodalRunner flow without modifying
 * ExecuTorch sources, and exposes phase timings for foundation_proc.csv.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_embedding_merger.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/tok_embedding_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/tok_embedding_runner.h>
#include <executorch/extension/llm/runner/audio.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace executorch::examples::foundation {

using ::CacheMode;
using ::example::IMemAlloc;
using ::example::KVManager;

struct QnnProfilePhaseTiming {
  long start_ms{0};
  long end_ms{0};
  long rss_kb_start{0};
  long rss_kb_end{0};
};

struct QnnProfileTokenTiming {
  int64_t token_idx{0};
  int64_t kv_pos{0};
  long start_ms{0};
  long end_ms{0};
  long rss_kb{0};
};

struct QnnProfileGenerateTimings {
  QnnProfilePhaseTiming vision_encode;
  QnnProfilePhaseTiming embedding_and_merging;
  QnnProfilePhaseTiming prefill;
  QnnProfilePhaseTiming decode;
  std::vector<QnnProfileTokenTiming> token_timings;
};

template <typename T>
class ProfiledQNNMultimodalRunner {
 public:
  explicit ProfiledQNNMultimodalRunner(
      std::unique_ptr<executorch::extension::Module> encoder,
      std::unique_ptr<executorch::extension::Module> tok_embedding,
      std::unique_ptr<executorch::extension::Module> text_decoder,
      const std::string& model_version,
      const std::string& tokenizer_path,
      const std::string& performance_output_path,
      const std::string& dump_logits_path,
      float temperature = 0.8f,
      int eval_mode = 1,
      bool shared_buffer = false,
      int ngram = 0,
      int window = 0,
      int gcap = 0);

  bool is_loaded() const;
  executorch::runtime::Error load();

  executorch::runtime::Error generate(
      const std::vector<executorch::extension::llm::MultimodalInput>& inputs,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {});

  executorch::runtime::Result<::example::ModelVersion> get_model_version();
  executorch::runtime::Result<executorch::runtime::MethodMeta>
  get_encoder_method_meta();

  const QnnProfileGenerateTimings& last_generate_timings() const {
    return last_generate_timings_;
  }

  int32_t context_len() const {
    return context_len_;
  }

  int64_t cur_pos() const {
    return cur_pos_;
  }

  size_t kv_cache_total_bytes() const {
    return kv_manager_ ? kv_manager_->total_cache_size_in_bytes() : 0;
  }

 private:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kUnsupported,
  };

  static constexpr const char* kEncoderForwardName = "forward";

  std::unique_ptr<executorch::extension::Module> encoder_;
  std::unique_ptr<executorch::extension::Module> tok_embedding_;
  std::unique_ptr<executorch::extension::Module> text_decoder_;

  int32_t context_len_{0};
  int ngram_{0};
  int window_{0};
  int gcap_{0};
  CacheMode cache_mode_{CacheMode::StaticCahce};
  int64_t cur_pos_{0};

  std::string tokenizer_path_;
  std::string performance_output_path_;
  std::string dump_logits_path_;
  float temperature_;
  EvalMode eval_mode_;
  bool shared_buffer_;

  ::example::ModelVersion model_version_;
  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager<T>> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<::example::DecoderRunner> decoder_runner_;
  std::unique_ptr<::example::MultimodalPromptProcessor<T>> prompt_processor_;
  std::unique_ptr<::example::MultimodalTokenGenerator<T>> token_generator_;
  std::unique_ptr<::example::EncoderRunner> encoder_runner_;
  std::unique_ptr<::example::TokenEmbeddingRunner> tok_embedding_runner_;
  std::unique_ptr<::example::TokenEmbeddingProcessor> tok_embedding_processor_;
  std::unique_ptr<::example::TokenEmbeddingProcessor> tok_embedding_generator_;
  std::unique_ptr<::example::MultimodalEmbeddingMerger> embedding_merger_;

  executorch::llm::Stats stats_;
  QnnProfileGenerateTimings last_generate_timings_;
};

} // namespace executorch::examples::foundation
