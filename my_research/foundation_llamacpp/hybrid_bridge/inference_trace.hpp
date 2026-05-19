#pragma once

#include "common.h"

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

/** Optional GGUF inference token traces (mtmd prefill + decode). */
struct inference_trace_collector {
  std::ofstream out;
  bool in_decode_ = false;
  std::string prefill_flat_;
  std::string decode_flat_;

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

  void append_prefill_trace_body(const std::string& body, const std::string& flat) {
    if (!out.is_open()) return;
    out << body;
    prefill_flat_ += flat;
    in_decode_ = false;
  }

  void chunk_image_begin(std::size_t idx, std::size_t n_tok, const char* cid) {
    chunk_image_begin(idx, n_tok, cid, 0);
  }

  void chunk_image_begin(std::size_t idx, std::size_t n_tok, const char* cid, std::size_t image_idx) {
    chunk_image_begin_visible(idx, n_tok, n_tok, cid, image_idx);
  }

  void chunk_image_begin_visible(
      std::size_t idx,
      std::size_t visible_tok,
      std::size_t nominal_tok,
      const char* cid,
      std::size_t image_idx) {
    if (!out.is_open()) return;
    out << "## CHUNK " << idx << " IMAGE image_index=" << (image_idx + 1)
        << " n_placeholder_tokens=" << visible_tok;
    if (nominal_tok != visible_tok) {
      out << " nominal_placeholder_tokens=" << nominal_tok;
    }
    if (cid != nullptr && cid[0]) {
      out << " mtmd_chunk_id=" << cid;
    }
    out << "\n";
    for (size_t i = 0; i < visible_tok; ++i) {
      const std::string piece = vision_slot_piece(i + 1);
      out << static_cast<long long>(-1) << '\t' << piece << '\n';
      prefill_flat_ += std::string("-1\t") + piece + "\n";
    }
    out << "# (each slot: projected vision embedding into decoder KV; not a BPE vocab id)\n";
  }

  static std::string vision_slot_piece(std::size_t one_based_idx) {
    return "<VISION_KV_SLOT " + std::to_string(one_based_idx) + ">";
  }

  void decode_header() {
    if (!out.is_open()) return;
    out << "\n--- DECODE ---\n";
    in_decode_ = true;
  }

  /** Appended after `User:` / `Assistant:` block in `foundation_token_io.txt`. */
  std::string format_token_io_appendix() const {
    return std::string(
               "\n--- mtmd+GGUF tokens (flattened; includes chat specials like <|im_start|>; "
               "vision: id -1 = no discrete vocab token)\n"
               "--- PREFILL ---\n")
        + prefill_flat_ + std::string("--- DECODE ---\n") + decode_flat_;
  }
};

} // namespace streamingvlm::hybrid_bridge
