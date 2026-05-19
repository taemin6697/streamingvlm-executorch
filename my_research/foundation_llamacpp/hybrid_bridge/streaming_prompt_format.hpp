#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace streamingvlm::hybrid_bridge {

enum class PromptFormatFamily {
  InternVL3,
  Qwen25VL,
};

struct PromptFormatProfile {
  std::string name = "internvl3";
  PromptFormatFamily family = PromptFormatFamily::InternVL3;
  std::string media_marker = "<__media__>";
  std::string frame_prefix = "Frame";
  std::string frame_separator = ": ";
  bool uses_mrope_positions = false;
};

inline std::string normalize_prompt_format_name(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    if (c == '-' || c == '.') {
      return '_';
    }
    return static_cast<char>(std::tolower(c));
  });
  if (name.empty() || name == "internvl" || name == "internvl3_instruct") {
    return "internvl3";
  }
  if (name == "qwen25vl" || name == "qwen2_5vl" ||
      name == "qwen2_5_vl" || name == "qwen2_5_vl_instruct") {
    return "qwen2_5_vl";
  }
  return name;
}

inline PromptFormatProfile prompt_format_profile(std::string name) {
  name = normalize_prompt_format_name(std::move(name));
  if (name == "qwen2_5_vl") {
    return PromptFormatProfile{
        "qwen2_5_vl",
        PromptFormatFamily::Qwen25VL,
        "<__media__>",
        "Frame",
        ": ",
        true,
    };
  }
  return PromptFormatProfile{
      "internvl3",
      PromptFormatFamily::InternVL3,
      "<__media__>",
      "Frame",
      ": ",
      false,
  };
}

template <typename Frame>
inline std::string build_stream_frame_prompt_line(
    const PromptFormatProfile& profile,
    const Frame& frame) {
  std::string out = profile.frame_prefix + std::to_string(frame.index + 1) + profile.frame_separator;
  const size_t n_tiles = std::max<size_t>(1, frame.tiles.size());
  for (size_t tile_i = 0; tile_i < n_tiles; ++tile_i) {
    out += profile.media_marker;
  }
  out += "\n";
  return out;
}

template <typename Frames>
inline std::string build_video_prompt_prefix(
    const PromptFormatProfile& profile,
    const Frames& frames) {
  std::string out;
  for (size_t frame_i = 0; frame_i < frames.size(); ++frame_i) {
    out += profile.frame_prefix + std::to_string(frame_i + 1) + profile.frame_separator;
    const size_t n_tiles = std::max<size_t>(1, frames[frame_i].tiles.size());
    for (size_t tile_i = 0; tile_i < n_tiles; ++tile_i) {
      out += profile.media_marker;
    }
    out += "\n";
  }
  return out;
}

template <typename Frames>
inline std::string build_stream_video_prompt_prefix(
    const PromptFormatProfile& profile,
    const Frames& frames) {
  std::string out;
  for (const auto& frame : frames) {
    out += build_stream_frame_prompt_line(profile, frame);
  }
  return out;
}

inline size_t next_non_frame_prefix_offset(
    const PromptFormatProfile& profile,
    const std::string& content) {
  size_t pos = 0;
  while (pos < content.size()) {
    const size_t line_start = pos;
    if (content.compare(pos, profile.frame_prefix.size(), profile.frame_prefix) != 0) {
      break;
    }
    pos += profile.frame_prefix.size();
    const size_t digits_begin = pos;
    while (pos < content.size() && std::isdigit(static_cast<unsigned char>(content[pos]))) {
      ++pos;
    }
    if (pos == digits_begin ||
        pos + profile.frame_separator.size() > content.size() ||
        content.compare(pos, profile.frame_separator.size(), profile.frame_separator) != 0) {
      pos = line_start;
      break;
    }
    pos += profile.frame_separator.size();
    bool saw_image_marker = false;
    while (content.compare(pos, profile.media_marker.size(), profile.media_marker) == 0) {
      saw_image_marker = true;
      pos += profile.media_marker.size();
    }
    if (!saw_image_marker || pos >= content.size() || content[pos] != '\n') {
      pos = line_start;
      break;
    }
    ++pos;
  }
  return pos;
}

inline std::string strip_stream_video_prompt_prefix(
    const PromptFormatProfile& profile,
    const std::string& content) {
  return content.substr(next_non_frame_prefix_offset(profile, content));
}

template <typename ChatHistory, typename Frames>
inline bool update_first_video_user_message(
    const PromptFormatProfile& profile,
    ChatHistory& chat_history,
    const Frames& frames) {
  for (auto& msg : chat_history) {
    if (msg.role == "user") {
      msg.content =
          build_stream_video_prompt_prefix(profile, frames) +
          strip_stream_video_prompt_prefix(profile, msg.content);
      return true;
    }
  }
  return false;
}

template <typename Frames>
inline std::string build_video_prompt(
    const PromptFormatProfile& profile,
    const Frames& frames,
    const std::string& raw_prompt) {
  return build_video_prompt_prefix(profile, frames) + raw_prompt;
}

}  // namespace streamingvlm::hybrid_bridge
