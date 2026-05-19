#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace streamingvlm::hybrid_bridge {

struct StreamingPolicyConfig {
  std::string stream_mode = "on_demand";
  double window_sec = -1.0;
  int window_max_frames = 8;
};

inline std::string normalize_stream_mode_name(std::string mode, bool single_buffer = false) {
  if (single_buffer) {
    return "on_demand";
  }
  std::replace(mode.begin(), mode.end(), '-', '_');
  if (mode.empty() || mode == "single_buffer") {
    return "on_demand";
  }
  return mode;
}

template <typename Items>
inline Items evenly_limit_items(const Items& items, int limit) {
  if (limit <= 0) {
    throw std::invalid_argument("window_max_frames must be positive");
  }
  if (static_cast<int>(items.size()) <= limit) {
    return items;
  }
  if (limit == 1) {
    return Items{items.back()};
  }
  Items out;
  out.reserve(static_cast<size_t>(limit));
  const int last = static_cast<int>(items.size()) - 1;
  for (int i = 0; i < limit; ++i) {
    const int idx = static_cast<int>(std::llround(static_cast<double>(i) * last / (limit - 1)));
    out.push_back(items[static_cast<size_t>(idx)]);
  }
  return out;
}

template <typename Frame>
inline std::vector<Frame> select_prompt_frames(
    const StreamingPolicyConfig& policy,
    const std::vector<Frame>& available_frames,
    const Frame& current_frame,
    double prompt_timestamp_s) {
  const std::string stream_mode = normalize_stream_mode_name(policy.stream_mode);
  if (stream_mode == "on_demand") {
    return {current_frame};
  }

  if (stream_mode == "vision_prefill") {
    std::vector<Frame> selected;
    for (const Frame& frame : available_frames) {
      if (frame.timestamp_s <= prompt_timestamp_s) {
        selected.push_back(frame);
      }
    }
    if (selected.empty()) {
      selected.push_back(current_frame);
    }
    return selected;
  }

  if (stream_mode != "sliding_window") {
    throw std::invalid_argument("unsupported stream mode: " + stream_mode);
  }

  const double start_s = policy.window_sec > 0.0 ? prompt_timestamp_s - policy.window_sec : -1.0e30;
  std::vector<Frame> selected;
  for (const Frame& frame : available_frames) {
    if (frame.timestamp_s >= start_s && frame.timestamp_s <= prompt_timestamp_s) {
      selected.push_back(frame);
    }
  }
  if (selected.empty()) {
    for (auto it = available_frames.rbegin(); it != available_frames.rend(); ++it) {
      if (it->timestamp_s <= prompt_timestamp_s) {
        selected.push_back(*it);
        break;
      }
    }
  }
  if (selected.empty()) {
    selected.push_back(current_frame);
  }
  return evenly_limit_items(selected, policy.window_max_frames);
}

}  // namespace streamingvlm::hybrid_bridge
