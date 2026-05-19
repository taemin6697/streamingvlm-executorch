#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

namespace streamingvlm::hybrid_bridge {

struct phase_recorder {
  long origin_ms = 0;
  std::ofstream out;

  phase_recorder(const std::string& path, long origin, const std::string& description)
      : origin_ms(origin) {
    if (!path.empty()) {
      out.open(path);
      if (!out.is_open()) {
        throw std::runtime_error("failed to open phase stats CSV: " + path);
      }
      out << "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
             "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
             "kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx\n";
      if (!description.empty()) {
        out << "# " << description << "\n";
      }
    }
  }

  void row(const char* row_type, long start_ms, long end_ms, int token_idx = 0) {
    if (!out.is_open()) {
      return;
    }
    const double start_s = static_cast<double>(start_ms - origin_ms) / 1000.0;
    const double end_s = static_cast<double>(end_ms - origin_ms) / 1000.0;
    const long total_ms = end_ms - start_ms;
    out << row_type << "," << start_s << "," << end_s
        << ",,," << total_ms << ",," << total_ms << ",,,,,,," << token_idx << "\n";
  }
};

inline const char* hybrid_decode_phase_description() {
  return "L_DecoderLoad: llama.cpp model/context/mmproj load  "
         "ExternalEmbeddingRead: .svlmemb read  LayoutTokenize: mtmd text/image layout  "
         "Mmproj: OpenCL mmproj projection for QNN pre-projector features  "
         "Prefill: combined text/image prompt eval  ImagePrefill: external image embedding decode  "
         "T_Prefill: text prompt eval  KVRepositionInsert: append late frame to restored video-prefix KV  "
         "KVRepositionTailShift: shift cached text tail in-place and reapply RoPE via llama.cpp K-shift before video-prefix KV insert  "
         "KVRepositionCompact: remove/close a KV gap and shift cached tail  D: one generated-token decode";
}

inline const char* opencl_phase_description() {
  return "L_DecoderRuntimeInit: llama.cpp args/OpenCL runtime init  "
         "L_DecoderLoad: llama.cpp text model/context/mmproj load  "
         "ImageLoad: input image load  LayoutTokenize: mtmd text/image layout  "
         "V_Encode: llama.cpp OpenCL InternVL vision pre-projector encode  "
         "Mmproj: llama.cpp OpenCL InternVL projector/mmproj  "
         "ImagePrefill: projected image embedding prefill  "
         "T_Prefill: text prompt prefill  D: one generated-token decode";
}

inline const char* vision_phase_description() {
  return "L_VisionLoad: ExecuTorch/QNN module load  ImageLoad: input tensor load  "
         "V_Encode: QNN projected vision embedding  EmbeddingFileWrite: .svlmemb write";
}

} // namespace streamingvlm::hybrid_bridge
