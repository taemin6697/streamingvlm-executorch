// InternVL Hugging Face reference: single-image chat uses literal
// question = '<image>\n' + task text (no "Frame1:" — that prefix is video-only).

#pragma once

#include <sstream>
#include <string>

namespace streamingvlm::hybrid_bridge {

inline bool internvl_prompt_has_hf_image_leader(const std::string & plain) {
  return plain.size() >= 7 && plain.compare(0, 7, "<image>") == 0;
}

// Matches OpenGVLab HF example single-image convention: '<image>\n' + prompt.
inline std::string internvl_hf_official_question_single_image(const std::string & plain_user_prompt_after_cli_minus_markers) {
  const std::string & p = plain_user_prompt_after_cli_minus_markers;
  if (internvl_prompt_has_hf_image_leader(p)) {
    return p;
  }
  return std::string("<image>\n") + p;
}

// Mirrors HF snippet: print(f'User: {question}\nAssistant: {response}')
inline std::string build_hf_user_assistant_echo(const std::string & hf_question, const std::string & assistant_text) {
  std::ostringstream o;
  o << "User: " << hf_question << "\nAssistant: " << assistant_text;
  return o.str();
}

// Same user-role *text* as HF `apply_chat_template` for one user turn: <|im_start|>user\n + question + <|im_end|>\n
// (`question` = internvl_hf_official_question_single_image(...)). Tokenize this in GGUF to match HF `encode` on that segment.
inline std::string internvl_hf_chat_template_user_segment_literal(const std::string & hf_question) {
  return std::string("<|im_start|>user\n") + hf_question + "<|im_end|>\n";
}

} // namespace streamingvlm::hybrid_bridge
