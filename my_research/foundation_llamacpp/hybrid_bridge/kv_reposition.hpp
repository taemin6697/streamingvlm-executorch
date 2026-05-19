#pragma once

#include "llama.h"

#include <string>

namespace streamingvlm {
namespace hybrid_bridge {

struct KvTokenRange {
  llama_pos begin = 0;
  llama_pos end = 0;

  llama_pos length() const {
    return end - begin;
  }
};

enum class KvPositionEncodingKind {
  Rope1D,
  MRope,
};

struct KvRepositionStrategy {
  KvPositionEncodingKind position_encoding = KvPositionEncodingKind::Rope1D;
  bool requires_k_shift_rebuild = true;
  bool supports_axis_aware_rewrite = false;
};

inline KvRepositionStrategy rope_1d_reposition_strategy() {
  return KvRepositionStrategy{KvPositionEncodingKind::Rope1D, true, false};
}

inline KvRepositionStrategy mrope_reposition_strategy_placeholder() {
  // Future Qwen-style M-RoPE support should preserve per-axis position metadata
  // before shifting/reapplying cached K. The current bridge only performs the
  // llama.cpp linear position shift used by 1D RoPE models such as InternVL3.
  return KvRepositionStrategy{KvPositionEncodingKind::MRope, true, true};
}

struct KvTailCompactionPlan {
  KvTokenRange removed;
  llama_pos sequence_end = 0;
  llama_pos tail_begin = 0;
  llama_pos tail_end = 0;
  llama_pos shift = 0;
  llama_pos compacted_sequence_end = 0;
};

struct KvTailInsertionPlan {
  llama_pos insert_pos = 0;
  llama_pos insert_len = 0;
  llama_pos sequence_end = 0;
  llama_pos tail_begin = 0;
  llama_pos tail_end = 0;
  llama_pos shift = 0;
  llama_pos expanded_sequence_end = 0;
};

inline void set_kv_reposition_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

inline bool validate_kv_range(
    KvTokenRange range,
    llama_pos sequence_end,
    std::string* error = nullptr) {
  if (range.begin < 0) {
    set_kv_reposition_error(error, "range begin must be non-negative");
    return false;
  }
  if (range.end < range.begin) {
    set_kv_reposition_error(error, "range end must be greater than or equal to begin");
    return false;
  }
  if (sequence_end < range.end) {
    set_kv_reposition_error(error, "range end exceeds sequence end");
    return false;
  }
  return true;
}

inline llama_pos compacted_position_after(KvTokenRange removed, llama_pos old_pos) {
  if (old_pos < removed.begin) {
    return old_pos;
  }
  if (old_pos < removed.end) {
    return -1;
  }
  return old_pos - removed.length();
}

inline llama_pos inserted_position_after(
    llama_pos insert_pos,
    llama_pos insert_len,
    llama_pos sequence_end,
    llama_pos old_pos) {
  if (old_pos < insert_pos) {
    return old_pos;
  }
  if (old_pos <= sequence_end) {
    return old_pos + insert_len;
  }
  return old_pos;
}

inline bool build_tail_compaction_plan(
    KvTokenRange removed,
    llama_pos sequence_end,
    KvTailCompactionPlan* out,
    std::string* error = nullptr) {
  if (out == nullptr) {
    set_kv_reposition_error(error, "output plan pointer is null");
    return false;
  }
  if (!validate_kv_range(removed, sequence_end, error)) {
    return false;
  }

  const llama_pos removed_len = removed.length();
  out->removed = removed;
  out->sequence_end = sequence_end;
  out->tail_begin = removed.end;
  out->tail_end = sequence_end;
  out->shift = -removed_len;
  out->compacted_sequence_end = sequence_end - removed_len;
  set_kv_reposition_error(error, "");
  return true;
}

inline bool build_tail_insertion_plan(
    llama_pos insert_pos,
    llama_pos insert_len,
    llama_pos sequence_end,
    KvTailInsertionPlan* out,
    std::string* error = nullptr) {
  if (out == nullptr) {
    set_kv_reposition_error(error, "output plan pointer is null");
    return false;
  }
  if (insert_pos < 0) {
    set_kv_reposition_error(error, "insert position must be non-negative");
    return false;
  }
  if (insert_len < 0) {
    set_kv_reposition_error(error, "insert length must be non-negative");
    return false;
  }
  if (sequence_end < insert_pos) {
    set_kv_reposition_error(error, "insert position exceeds sequence end");
    return false;
  }

  out->insert_pos = insert_pos;
  out->insert_len = insert_len;
  out->sequence_end = sequence_end;
  out->tail_begin = insert_pos;
  out->tail_end = sequence_end;
  out->shift = insert_len;
  out->expanded_sequence_end = sequence_end + insert_len;
  set_kv_reposition_error(error, "");
  return true;
}

inline bool build_rewrite_compaction_plan(
    KvTokenRange original,
    llama_pos rewritten_len,
    llama_pos sequence_end,
    KvTailCompactionPlan* out,
    std::string* error = nullptr) {
  if (rewritten_len < 0) {
    set_kv_reposition_error(error, "rewritten length must be non-negative");
    return false;
  }
  if (!validate_kv_range(original, sequence_end, error)) {
    return false;
  }
  if (rewritten_len > original.length()) {
    set_kv_reposition_error(error, "rewritten length cannot exceed original range length");
    return false;
  }

  return build_tail_compaction_plan(
      KvTokenRange{original.begin + rewritten_len, original.end},
      sequence_end,
      out,
      error);
}

inline bool apply_tail_compaction_plan(
    llama_memory_t memory,
    llama_seq_id seq_id,
    const KvTailCompactionPlan& plan,
    std::string* error = nullptr) {
  if (memory == nullptr) {
    set_kv_reposition_error(error, "llama memory pointer is null");
    return false;
  }
  if (!validate_kv_range(plan.removed, plan.sequence_end, error)) {
    return false;
  }
  if (plan.tail_begin != plan.removed.end ||
      plan.tail_end != plan.sequence_end ||
      plan.shift != -plan.removed.length() ||
      plan.compacted_sequence_end != plan.sequence_end - plan.removed.length()) {
    set_kv_reposition_error(error, "tail compaction plan does not match removed range");
    return false;
  }

  if (plan.removed.length() > 0 &&
      !llama_memory_seq_rm(memory, seq_id, plan.removed.begin, plan.removed.end)) {
    set_kv_reposition_error(error, "llama_memory_seq_rm failed");
    return false;
  }
  if (plan.tail_begin < plan.tail_end && plan.shift != 0) {
    llama_memory_seq_add(memory, seq_id, plan.tail_begin, plan.tail_end, plan.shift);
  }
  set_kv_reposition_error(error, "");
  return true;
}

inline bool apply_tail_insertion_plan(
    llama_memory_t memory,
    llama_seq_id seq_id,
    const KvTailInsertionPlan& plan,
    std::string* error = nullptr) {
  if (memory == nullptr) {
    set_kv_reposition_error(error, "llama memory pointer is null");
    return false;
  }
  if (plan.insert_pos < 0 || plan.insert_len < 0 || plan.sequence_end < plan.insert_pos) {
    set_kv_reposition_error(error, "invalid tail insertion plan");
    return false;
  }
  if (plan.tail_begin != plan.insert_pos ||
      plan.tail_end != plan.sequence_end ||
      plan.shift != plan.insert_len ||
      plan.expanded_sequence_end != plan.sequence_end + plan.insert_len) {
    set_kv_reposition_error(error, "tail insertion plan does not match insert range");
    return false;
  }

  if (plan.tail_begin < plan.tail_end && plan.shift != 0) {
    llama_memory_seq_add(memory, seq_id, plan.tail_begin, plan.tail_end, plan.shift);
  }
  set_kv_reposition_error(error, "");
  return true;
}

}  // namespace hybrid_bridge
}  // namespace streamingvlm
