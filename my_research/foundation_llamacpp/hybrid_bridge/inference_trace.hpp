#pragma once

#include "common.h"
#include "foundation_token_io_format.hpp"

#include <fstream>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace streamingvlm::hybrid_bridge {

/** Same directory as `foundation_token_io.txt` (or cwd if path has no parent). */
inline std::string sibling_foundation_inference_tokens_path(const std::string& token_io_path) {
  namespace fs = std::filesystem;
  const fs::path p(token_io_path);
  const fs::path dir = p.has_parent_path() ? p.parent_path() : fs::path(".");
  return (dir / "foundation_inference_tokens.txt").lexically_normal().string();
}

/** Optional text dump aligned with HF reference question + GGUF inference token traces. */
struct inference_trace_collector {
  std::ofstream out;
  bool in_decode_ = false;
  std::string prefill_flat_;
  std::string decode_flat_;
  std::string hf_user_seg_ref_flat_;

  explicit inference_trace_collector(const std::string& path) : out(path) {}

  explicit operator bool() const {
    return out.is_open();
  }

  static void escape_piece(std::ostream& o, const std::string& piece) {
    for (unsigned char c : piece) {
      if (c == '\n') {
        o << "\\n";
      } else if (c == '\r') {
        o << "\\r";
      } else if (c == '\t') {
        o << "\\t";
      } else {
        o << c;
      }
    }
  }

  static std::string escape_piece_str(const std::string& piece) {
    std::ostringstream ps;
    escape_piece(ps, piece);
    return ps.str();
  }

  void write_hf_reference_question_literal(const std::string& hf_q) {
    if (!out.is_open()) return;
    out << "HF_OFFICIAL_QUESTION_LITERAL"
           " (same string you would assign to `question` in official InternVL `model.chat` single-image demos)\n"
        << "---BEGIN---\n"
        << hf_q << "\n---END---\n\n";
    out << "NOTE: HF_OFFICIAL_USER_SEGMENT_TOKENIZE matches full user-turn encode; "
           "PREFILL is the actual mtmd path (InternVL uses literal <image>\\n BPE + vision slots).\n\n";
    in_decode_ = false;
  }

  /** HF-aligned reference: tokenize the user-turn segment only (no system prompt), same bytes HF uses in chat template. */
  void write_hf_official_user_segment_reference(struct llama_context* lctx, const std::string& hf_q) {
    if (!out.is_open()) return;
    const std::string seg = internvl_hf_chat_template_user_segment_literal(hf_q);
    out << "HF_OFFICIAL_USER_SEGMENT_TOKENIZE"
           " (reference: common_tokenize on <|im_start|>user\\n + hf_question + <|im_end|>\\n; "
           "same BPE as HF tokenizer.encode on that segment; mtmd InternVL prefill now uses the same <image>\\n BPE.)\n"
        << "---SEGMENT_LITERAL---\n"
        << seg << "---END---\n";
    const std::vector<llama_token> toks = common_tokenize(lctx, seg, false, true);
    out << "---TOKEN_IDS (id\\tpiece) n=" << toks.size() << " ---\n";
    hf_user_seg_ref_flat_ =
        "--- HF_OFFICIAL_USER_SEGMENT (GGUF tokenizer on HF user-turn segment; for PyTorch compare tokenizer.encode)\n";
    for (llama_token t : toks) {
      const std::string piece = common_token_to_piece(lctx, t, true);
      out << static_cast<long long>(t) << '\t';
      escape_piece(out, piece);
      out << '\n';
      hf_user_seg_ref_flat_ +=
          std::to_string(static_cast<long long>(t)) + '\t' + escape_piece_str(piece) + '\n';
    }
    out << '\n';
  }

  void write_prefill_header() {
    if (!out.is_open()) return;
    out << "--- PREFILL (mtmd chunk order from this binary) ---\n";
    in_decode_ = false;
  }

  void chunk_text_begin(std::size_t idx, std::size_t n_tok) {
    if (!out.is_open()) return;
    out << "## CHUNK " << idx << " TEXT n_tokens=" << n_tok << "\n";
  }

  void token_line(llama_token t, const std::string& piece) {
    if (!out.is_open()) return;
    out << static_cast<long long>(t) << '\t';
    escape_piece(out, piece);
    out << '\n';
    const std::string esc = escape_piece_str(piece);
    const std::string one = std::to_string(static_cast<long long>(t)) + '\t' + esc + '\n';
    if (!in_decode_) {
      prefill_flat_ += one;
    } else {
      decode_flat_ += one;
    }
  }

  void chunk_image_begin(std::size_t idx, std::size_t n_tok, const char* cid) {
    if (!out.is_open()) return;
    out << "## CHUNK " << idx << " IMAGE n_placeholder_tokens=" << n_tok;
    if (cid != nullptr && cid[0]) {
      out << " mtmd_chunk_id=" << cid;
    }
    out << "\n";
    for (size_t i = 0; i < n_tok; ++i) {
      const std::string piece =
          "<VISION_KV_SLOT " + std::to_string(i + 1) + "/" + std::to_string(n_tok) + ">";
      out << static_cast<long long>(-1) << '\t' << piece << '\n';
      prefill_flat_ += std::string("-1\t") + piece + "\n";
    }
    out << "# (each slot: projected vision embedding into decoder KV; not a BPE vocab id)\n";
  }

  void decode_header() {
    if (!out.is_open()) return;
    out << "\n--- DECODE ---\n";
    in_decode_ = true;
  }

  /** Appended after `User:` / `Assistant:` block in `foundation_token_io.txt`. */
  std::string format_token_io_appendix() const {
    const std::string hf_blk = hf_user_seg_ref_flat_.empty()
        ? std::string()
        : (std::string("\n") + hf_user_seg_ref_flat_ + "\n");
    return hf_blk
        + std::string(
              "\n--- mtmd+GGUF tokens (flattened; includes chat specials like <|im_start|>, <img>; "
              "vision: id -1 = no discrete vocab token)\n"
              "--- PREFILL ---\n")
        + prefill_flat_ + std::string("--- DECODE ---\n") + decode_flat_;
  }
};

} // namespace streamingvlm::hybrid_bridge
