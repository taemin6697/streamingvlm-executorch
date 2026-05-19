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
  std::string prefill_body_;
  std::string prefill_flat_;
  std::string decode_body_;
  std::string decode_flat_;
  std::size_t decode_token_count_ = 0;

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
    const std::string header =
        "## CHUNK " + std::to_string(idx) + " TEXT n_tokens=" + std::to_string(n_tok) + "\n";
    out << header;
    prefill_body_ += header;
  }

  void token_line(llama_token t, const std::string& piece) {
    if (!out.is_open()) return;
    const std::string esc = escape_piece_str(piece);
    const std::string one = std::to_string(static_cast<long long>(t)) + '\t' + esc + '\n';
    out << one;
    if (!in_decode_) {
      prefill_body_ += one;
      prefill_flat_ += one;
    } else {
      decode_body_ += one;
      decode_flat_ += one;
      ++decode_token_count_;
    }
  }

  void append_prefill_trace_body(const std::string& body, const std::string& flat) {
    if (!out.is_open()) return;
    out << body;
    prefill_body_ += body;
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
    std::string header = "## CHUNK " + std::to_string(idx) +
                         " IMAGE image_index=" + std::to_string(image_idx + 1) +
                         " n_placeholder_tokens=" + std::to_string(visible_tok);
    if (nominal_tok != visible_tok) {
      header += " nominal_placeholder_tokens=" + std::to_string(nominal_tok);
    }
    if (cid != nullptr && cid[0]) {
      header += std::string(" mtmd_chunk_id=") + cid;
    }
    header += "\n";
    out << header;
    prefill_body_ += header;
    for (size_t i = 0; i < visible_tok; ++i) {
      const std::string piece = vision_slot_piece(i + 1);
      const std::string line = std::string("-1\t") + piece + "\n";
      out << line;
      prefill_body_ += line;
      prefill_flat_ += line;
    }
    const std::string note = "# (each slot: projected vision embedding into decoder KV; not a BPE vocab id)\n";
    out << note;
    prefill_body_ += note;
  }

  static std::string vision_slot_piece(std::size_t one_based_idx) {
    return "<VISION_KV_SLOT " + std::to_string(one_based_idx) + ">";
  }

  void decode_header() {
    if (!out.is_open()) return;
    out << "\n--- DECODE ---\n";
    in_decode_ = true;
  }

  std::string decode_history_body(std::size_t chunk_idx) const {
    if (decode_body_.empty()) {
      return {};
    }
    return "## CHUNK " + std::to_string(chunk_idx) +
           " ASSISTANT_DECODE n_tokens=" + std::to_string(decode_token_count_) + "\n" +
           decode_body_;
  }

  const std::string& decode_flat() const {
    return decode_flat_;
  }

  std::size_t decode_token_count() const {
    return decode_token_count_;
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
