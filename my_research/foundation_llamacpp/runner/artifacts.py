from __future__ import annotations


HYBRID_PULL_ARTIFACTS = (
    "hybrid_vision_stdout.txt",
    "hybrid_decode_stdout.txt",
    "vision_output_stats.csv",
    "vision_phase_stats.csv",
    "decoder_phase_stats.csv",
    "foundation_token_io.txt",
    "foundation_inference_tokens.txt",
    "vision_embedding.svlmemb",
    "hybrid_projected_embedding.svlmemb",
    "media_manifest.json",
    "foundation_exit_code.txt",
    "vision_exit_code.txt",
    "decoder_exit_code.txt",
    "android_memory_timeline.csv",
)

HYBRID_STREAMING_PULL_ARTIFACTS = (
    "hybrid_streaming_stdout.txt",
    "opencl_streaming_stdout.txt",
    "foundation_output.txt",
    "stream_events.csv",
    "streaming_phase_stats.csv",
    "foundation_token_io.txt",
    "foundation_inference_tokens.txt",
    "stream_inference_tokens_*.txt",
    "media_manifest.json",
    "foundation_exit_code.txt",
    "android_memory_timeline.csv",
)

STANDALONE_PULL_ARTIFACTS = (
    "foundation_output.txt",
    "foundation_exit_code.txt",
    "foundation_phase_stats.csv",
    "foundation_token_io.txt",
    "foundation_inference_tokens.txt",
    "media_manifest.json",
    "opencl_projected_embedding.svlmemb",
    "android_memory_timeline.csv",
)

