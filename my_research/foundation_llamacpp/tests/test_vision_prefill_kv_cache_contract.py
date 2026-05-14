from pathlib import Path

from my_research.foundation_llamacpp.runner import cli as runner_cli


ROOT = Path(__file__).resolve().parents[3]
STREAMING_CPP = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp"
SPEC = ROOT / "docs/superpowers/specs/2026-05-13-vision-prefill-kv-cache-design.md"


def test_vision_prefill_uses_llama_sequence_state_cache():
    source = STREAMING_CPP.read_text()

    assert "VisionPrefillCache" in source
    assert "llama_state_seq_get_data_ext" in source
    assert "llama_state_seq_set_data_ext" in source
    assert "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE" in source


def test_vision_prefill_has_prompt_suffix_boundary_and_cache_phases():
    source = STREAMING_CPP.read_text()

    assert "SVLM_QUESTION_SENTINEL" in source
    assert "VisionPrefillCacheBuild" in source
    assert "VisionPrefillCacheSave" in source
    assert "VisionPrefillCacheRestore" in source
    assert "VisionPrefillCacheHit" in source
    assert "VisionPrefillCacheMiss" in source


def test_future_chunked_mode_name_is_documented():
    spec = SPEC.read_text()

    assert "--chunked-vision-prefill" in spec
    assert "--chunk-count" in spec


def test_streaming_timeline_aliases_vision_prefill_to_standard_lanes():
    source = STREAMING_CPP.read_text()

    assert runner_cli._streaming_timeline_phase_name("VisionPrefillV_Encode") == "V_Encode"
    assert runner_cli._streaming_timeline_phase_name("VisionPrefillMmproj") == "Mmproj"
    assert runner_cli._streaming_timeline_phase_name("VisionPrefillImagePrefill") == "ImagePrefill"
    assert runner_cli._streaming_timeline_phase_name("VisionPrefillT_Prefill") == "T_Prefill"
    assert runner_cli._streaming_timeline_phase_name("SingleBufferUpdate") == "SingleBufferUpdate"
    assert runner_cli._streaming_timeline_phase_name("D") == "D"

    assert runner_cli._streaming_timeline_phase_name("VisionPrefillCacheBuild") is None
    assert runner_cli._streaming_timeline_phase_name("VisionPrefillCacheSave") is None
    assert runner_cli._streaming_timeline_phase_name("VisionPrefillCacheRestore") is None

    assert "drop_pending_cache_updates" not in source


def test_vision_prefill_keeps_full_history_and_caches_every_frame():
    source = STREAMING_CPP.read_text()

    assert 'args.stream_mode == "vision_prefill"' in source
    assert 'mode == "sliding_window"' in source
    assert "return selected;" in source
    assert "StreamJobKind::CacheUpdate" in source
    assert "drop_pending_cache_updates" not in source


def test_vision_prefill_cache_build_encodes_frames_on_demand():
    source = STREAMING_CPP.read_text()
    build_fn = source.split("bool build_vision_prefill_cache(", 1)[1].split("\n}\n\nint run_single_buffer_prompt", 1)[0]

    assert "eval_streaming_chunks_with_on_demand_vision" in source
    assert "eval_streaming_chunks_with_on_demand_vision(" in build_fn
    assert "encoder.encode(bins)" not in build_fn
    assert "bins[image_chunk_idx]" in source


def test_sliding_window_keeps_multiturn_text_history():
    source = STREAMING_CPP.read_text()
    singleton_fn = source.split("bool is_singleton_video_mode(", 1)[1].split("\n}\n\nvoid reset_decode_context_for_singleton", 1)[0]

    assert 'args.stream_mode == "vision_prefill"' in singleton_fn
    assert 'args.stream_mode == "sliding_window"' not in singleton_fn
    assert 'if (args.stream_mode == "sliding_window") {\n    reset_decode_context_for_singleton(ctx);\n  }' not in source


def test_streaming_timeline_starts_at_stream_origin():
    assert runner_cli._streaming_timeline_origin(0.0, [5.0, 8.0], []) == 0.0
