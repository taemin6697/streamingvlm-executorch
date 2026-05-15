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

    assert runner_cli._phase_timeline_name("VisionPrefillV_Encode") == "V_Encode"
    assert runner_cli._phase_timeline_name("VisionPrefillMmproj") == "Mmproj"
    assert runner_cli._phase_timeline_name("VisionPrefillImagePrefill") == "ImagePrefill"
    assert runner_cli._phase_timeline_name("VisionPrefillT_Prefill") == "T_Prefill"
    assert runner_cli._phase_timeline_name("SingleBufferUpdate") == "SingleBufferUpdate"
    assert runner_cli._phase_timeline_name("D") == "Decode"

    assert runner_cli._phase_timeline_name("VisionPrefillCacheBuild") is None
    assert runner_cli._phase_timeline_name("VisionPrefillCacheSave") is None
    assert runner_cli._phase_timeline_name("VisionPrefillCacheRestore") is None

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


def test_vision_prefill_cache_build_restores_previous_snapshot_and_appends_one_frame():
    source = STREAMING_CPP.read_text()
    build_fn = source.split("bool build_vision_prefill_cache(", 1)[1].split("\n}\n\nint run_single_buffer_prompt", 1)[0]

    assert "build_formatted_incremental_vision_cache_append" in source
    assert "restore_vision_prefill_cache_state(ctx, cache, cache_phases, \"VisionPrefillCacheAppendRestore\")" in build_fn
    assert "std::vector<FrameRecord> append_frames{frames.back()}" in build_fn
    assert "bins_for_frames(append_frames)" in build_fn
    assert "layout_images_for_frames(append_frames)" in build_fn


def test_vision_prefill_preserves_chat_history_across_prompt_events():
    source = STREAMING_CPP.read_text()
    cache_struct = source.split("struct VisionPrefillCache {", 1)[1].split("\n};", 1)[0]
    restore_fn = source.split("bool restore_vision_prefill_cache_state(", 1)[1].split("\n}\n\nbool build_vision_prefill_cache", 1)[0]
    prompt_fn = source.split("int run_single_buffer_prompt(", 1)[1].split("\n#else\nint run_single_buffer_prompt", 1)[0]
    singleton_fn = source.split("bool is_singleton_video_mode(", 1)[1].split("\n}\n\nvoid reset_decode_context_for_singleton", 1)[0]

    assert "std::vector<common_chat_msg> chat_history" in cache_struct
    assert "std::string open_user_content" in cache_struct
    assert "bool open_user_prefix" in cache_struct
    assert "ctx.chat_history = cache.chat_history" in restore_fn
    assert "cache.chat_history = ctx.chat_history" in source
    assert 'args.stream_mode == "vision_prefill"' not in singleton_fn
    assert "vision_cache->open_user_prefix = false" in prompt_fn
    assert "save_vision_prefill_cache_state(ctx, *vision_cache, prompt_phases)" in prompt_fn


def test_vision_prefill_uses_global_stream_frame_labels_for_interleaved_turns():
    source = STREAMING_CPP.read_text()

    assert "build_stream_frame_prompt_line" in source
    assert '"Frame" + std::to_string(frame.index + 1) + ": "' in source


def test_streaming_cpp_uses_internvl_video_frame_labels_without_space():
    source = STREAMING_CPP.read_text()

    assert '"Frame" + std::to_string(frame_i + 1) + ": "' in source
    assert '"Frame" + std::to_string(frame.index + 1) + ": "' in source
    assert '"Frame " + std::to_string' not in source


def test_sliding_window_keeps_multiturn_text_history():
    source = STREAMING_CPP.read_text()
    singleton_fn = source.split("bool is_singleton_video_mode(", 1)[1].split("\n}\n\nvoid reset_decode_context_for_singleton", 1)[0]

    assert 'args.stream_mode == "vision_prefill"' not in singleton_fn
    assert 'args.stream_mode == "sliding_window"' not in singleton_fn
    assert 'if (args.stream_mode == "sliding_window") {\n    reset_decode_context_for_singleton(ctx);\n  }' not in source


def test_streaming_timeline_starts_at_stream_origin():
    assert runner_cli._phase_timeline_origin(0.0, [5.0, 8.0], [], stream_time=True) == 0.0
