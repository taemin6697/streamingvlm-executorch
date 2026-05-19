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
    assert runner_cli._phase_timeline_name("SingleBufferUpdate") == "OnDemandBufferUpdate"
    assert runner_cli._phase_timeline_name("OnDemandBufferUpdate") == "OnDemandBufferUpdate"
    assert runner_cli._phase_timeline_name("D") == "Decode"

    assert runner_cli._phase_timeline_name("VisionPrefillCacheBuild") is None
    assert runner_cli._phase_timeline_name("VisionPrefillCacheHostSave") is None
    assert runner_cli._phase_timeline_name("VisionPrefillCachePreempt") is None
    assert runner_cli._phase_timeline_name("VisionPrefillCacheSave") is None
    assert runner_cli._phase_timeline_name("VisionPrefillCacheRestore") is None
    assert runner_cli._phase_timeline_name("VisionPrefillCacheRollback") is None

    assert "drop_pending_cache_updates" in source


def test_vision_prefill_keeps_full_history_and_caches_every_frame():
    source = STREAMING_CPP.read_text()

    assert 'args.stream_mode == "vision_prefill"' in source
    assert 'mode == "sliding_window"' in source
    assert "return selected;" in source
    assert "StreamJobKind::CacheUpdate" in source
    assert "drop_pending_cache_updates(stream_jobs)" in source


def test_online_buffer_uses_latest_frame_at_processing_start():
    source = STREAMING_CPP.read_text()
    runner_source = (runner_cli.FOUNDATION_LLAMA / "runner" / "cli.py").read_text(encoding="utf-8")

    assert "bool online_buffer = false" in source
    assert "resolve_online_buffer_frames" in source
    assert "prompt_frame_lag_s" in source
    assert "stream_buffer_summary.txt" in source
    assert "committed_cache_fps=" in source
    assert "cache_worker_fps=" in source
    assert "note_committed_cache_update" in source
    assert "note_prompt_decode_job" in source
    assert "rm -f android_memory_timeline.csv" in runner_source
    assert "stream_buffer_summary.txt stream_response_*.txt" in runner_source


def test_latest_frame_only_drops_frame_cache_updates_arriving_while_worker_is_busy():
    source = STREAMING_CPP.read_text()
    runner_source = (runner_cli.FOUNDATION_LLAMA / "runner" / "cli.py").read_text(encoding="utf-8")

    assert "bool latest_frame_only = false" in source
    assert '"--latest-frame-only"' in source
    assert "latest_frame_only_arg" in runner_source
    assert "cache_worker_busy" in source
    assert "should_drop_cache_update_for_latest_frame_only" in source
    assert "LatestFrameOnlyCacheDrop" in source
    assert "latest_frame_only_dropped_cache_updates=" in source
    assert "cache_update_in_queue(stream_jobs)" in source
    assert "args.latest_frame_only && args.stream_mode == \"vision_prefill\"" in source


def test_vision_prefill_cache_build_encodes_frames_on_demand():
    source = STREAMING_CPP.read_text()
    build_fn = source.split("VisionPrefillCacheBuildStatus build_vision_prefill_cache(", 1)[1].split(
        "\n}\n\nint run_vision_prefill_prompt_from_committed_cache", 1
    )[0]

    assert "eval_streaming_chunks_with_on_demand_vision" in source
    assert "eval_streaming_chunks_with_on_demand_vision(" in build_fn
    assert "encoder.encode(bins)" not in build_fn
    assert "bins[image_chunk_idx]" in source


def test_vision_prefill_cache_build_restores_previous_snapshot_and_appends_next_missing_frame():
    source = STREAMING_CPP.read_text()
    build_fn = source.split("VisionPrefillCacheBuildStatus build_vision_prefill_cache(", 1)[1].split(
        "\n}\n\nint run_vision_prefill_prompt_from_committed_cache", 1
    )[0]

    assert "build_formatted_incremental_vision_cache_append" in source
    assert "restore_vision_prefill_cache_state(ctx, cache, cache_phases, \"VisionPrefillCacheAppendRestore\")" in build_fn
    assert "target_frames.resize(cached_prefix_size + 1)" in build_fn
    assert "target_frames.begin() + static_cast<std::ptrdiff_t>(cached_prefix_size)" in build_fn
    assert "target_frames.end()" not in build_fn.split("std::vector<FrameRecord> append_frames", 1)[1].split(");", 1)[0]
    assert "std::vector<FrameRecord> append_frames{target_frames.back()}" not in build_fn
    assert "bins_for_frames(append_frames)" in build_fn
    assert "layout_images_for_frames(append_frames)" in build_fn


def test_vision_prefill_prompt_uses_committed_cache_snapshot_without_video_fallback():
    source = STREAMING_CPP.read_text()
    prompt_fn = source.split("int run_vision_prefill_prompt_from_committed_cache(", 1)[1].split(
        "\n}\n\nint run_single_buffer_prompt", 1
    )[0]

    assert "vision_cache->frames" in prompt_fn
    assert "vision_cache->images" in prompt_fn
    assert "prefer_host_restore" in prompt_fn
    assert "vision_prefill_cache_matches" not in prompt_fn
    assert "encoder.encode(bins)" not in prompt_fn


def test_vision_prefill_cache_update_preempts_for_pending_prompt():
    source = STREAMING_CPP.read_text()
    build_fn = source.split("VisionPrefillCacheBuildStatus build_vision_prefill_cache(", 1)[1].split(
        "\n}\n\nint run_vision_prefill_prompt_from_committed_cache", 1
    )[0]
    mtmd_helper_header = (ROOT / "llama.cpp/tools/mtmd/mtmd-helper.h").read_text()

    assert "#include <atomic>" in source
    assert "std::atomic<int> pending_prompt_jobs" in source
    assert "pending_prompt_jobs.fetch_add" in source
    assert "pending_prompt_jobs.fetch_sub" in source
    assert "cache_preempt_requested" in source
    assert "VisionPrefillCachePreempt" in source
    assert "VisionPrefillCacheRollback" in source
    assert "host_state" in source
    assert "VisionPrefillCacheBuildStatus::Preempted" in build_fn
    assert "drop_pending_cache_updates(stream_jobs)" in source
    assert "mtmd_helper_decode_image_chunk_with_abort" in mtmd_helper_header
    assert "mtmd_helper_decode_image_chunk_with_abort_and_progress" in mtmd_helper_header
    assert "mtmd_decode_batch_callback" in mtmd_helper_header
    assert "preemptible_image_batch" in source
    assert "mtmd_helper_decode_image_chunk_with_abort_and_progress" in source
    assert "VisionPrefillImagePrefillBatch" in source


def test_partial_vision_prefill_commits_current_image_batch_instead_of_rollback():
    source = STREAMING_CPP.read_text()
    helper_header = (ROOT / "llama.cpp/tools/mtmd/mtmd-helper.h").read_text()
    helper_impl = (ROOT / "llama.cpp/tools/mtmd/mtmd-helper.cpp").read_text()
    build_fn = source.split("VisionPrefillCacheBuildStatus build_vision_prefill_cache(", 1)[1].split(
        "\n}\n\nint run_vision_prefill_prompt_from_committed_cache", 1
    )[0]

    assert "bool partial_vision_kv = false" in source
    assert '"--partial-vision-kv"' in source
    assert "VisionPrefillCacheBuildStatus::Partial" in source
    assert "partial_image_committed" in source
    assert "args.partial_vision_kv" in build_fn
    assert "save_vision_prefill_cache_state(ctx, next_cache, cache_phases, args.partial_vision_kv)" in build_fn
    assert "VisionPrefillCachePartialCommit" in source
    assert "rollback_vision_prefill_cache_build(ctx, cache, cache_phases, false)" in build_fn
    assert "new_n_past" in helper_header
    assert "*new_n_past = n_past + decoded_n_pos" in helper_impl
    assert "return 2" in helper_impl


def test_partial_vision_prefill_waits_for_one_image_batch_before_abort():
    source = STREAMING_CPP.read_text()

    assert "completed_image_batches" in source
    assert "require_completed_batch_before_abort" in source
    assert "*callback->completed_image_batches <= 0" in source


def test_partial_vision_prefill_commits_mutated_text_state_on_preempt():
    source = STREAMING_CPP.read_text()
    eval_fn = source.split("bool eval_streaming_chunks_with_on_demand_vision(", 1)[1].split(
        "\n}\n\nbool tokenize_formatted_text", 1
    )[0]

    assert "llm_state_mutated" in eval_fn
    assert "commit_partial_cache_preempt" in eval_fn
    assert "next_chunk_is_image" in eval_fn


def test_partial_vision_prefill_closes_current_frame_text_after_partial_image():
    source = STREAMING_CPP.read_text()
    eval_fn = source.split("bool eval_streaming_chunks_with_on_demand_vision(", 1)[1].split(
        "\n}\n\nbool tokenize_formatted_text", 1
    )[0]

    assert "drain_text_chunks_after_partial_image" in eval_fn
    assert "mtmd_input_chunk_get_type(drain_chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE" in eval_fn


def test_partial_vision_prefill_uses_live_cache_save_without_replacing_snapshot_mode():
    source = STREAMING_CPP.read_text()
    save_fn = source.split("bool save_vision_prefill_cache_state(", 1)[1].split(
        "\n}\n\nbool restore_vision_prefill_cache_state", 1
    )[0]

    assert "bool live_only" in save_fn
    assert "if (live_only)" in save_fn
    assert "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE" in save_fn
    assert "args.partial_vision_kv)" in source


def test_partial_vision_prefill_uses_ubatch_size_for_visible_chunks():
    source = STREAMING_CPP.read_text()

    assert "image_prefill_batch_size" in source
    assert "std::min<int32_t>(ctx.n_batch, image_prefill_batch_size)" in source
    assert "std::max(1, args.ubatch_size)" in source
    assert "k_preemptible_image_prefill_batch" not in source


def test_inference_trace_uses_committed_vision_slots_without_nominal_suffix():
    trace_source = (ROOT / "my_research/foundation_llamacpp/hybrid_bridge/inference_trace.hpp").read_text()
    streaming_source = STREAMING_CPP.read_text()

    assert "vision_slot_piece" in trace_source
    assert '"<VISION_KV_SLOT " + std::to_string(one_based_idx) + ">"' in trace_source
    assert '"/" + std::to_string(n_tok)' not in trace_source
    assert "chunk_image_begin_visible" in trace_source
    assert "nominal_placeholder_tokens" in trace_source

    assert "committed_image_tokens" in streaming_source
    assert "record_committed_image_tokens" in streaming_source
    assert "render_prefill_trace_for_chunks" in streaming_source
    assert "append_prefill_trace_body" in streaming_source
    assert "prefill_trace_next_chunk_idx" in streaming_source


def test_vision_prefill_preserves_chat_history_across_prompt_events():
    source = STREAMING_CPP.read_text()
    cache_struct = source.split("struct VisionPrefillCache {", 1)[1].split("\n};", 1)[0]
    build_fn = source.split("VisionPrefillCacheBuildStatus build_vision_prefill_cache(", 1)[1].split(
        "\n}\n\nint run_vision_prefill_prompt_from_committed_cache", 1
    )[0]
    restore_fn = source.split("bool restore_vision_prefill_cache_state(", 1)[1].split("\n}\n\nbool build_vision_prefill_cache", 1)[0]
    prompt_fn = source.split("int run_vision_prefill_prompt_from_committed_cache(", 1)[1].split(
        "\n}\n\nint run_single_buffer_prompt", 1
    )[0]
    singleton_fn = source.split("bool is_singleton_video_mode(", 1)[1].split("\n}\n\nvoid reset_decode_context_for_singleton", 1)[0]

    assert "std::vector<common_chat_msg> chat_history" in cache_struct
    assert "std::string open_user_content" in cache_struct
    assert "bool open_user_prefix" in cache_struct
    assert "std::string prefill_trace_body" in cache_struct
    assert "std::string prefill_trace_flat" in cache_struct
    assert "ctx.chat_history = cache.chat_history" in restore_fn
    assert "cache.chat_history = ctx.chat_history" in source
    assert 'args.stream_mode == "vision_prefill"' not in singleton_fn
    assert "next_cache.prefill_trace_body = cache.prefill_trace_body" in build_fn
    assert "tail_trace_body = vision_cache->prefill_trace_tail_body + suffix_trace.body" in prompt_fn
    assert "rebuild_prefill_trace_from_video_and_tail(*vision_cache)" in prompt_fn
    assert "decode_history_body(tail_next_chunk_idx)" in prompt_fn
    assert "vision_cache->open_user_prefix = false" in prompt_fn
    assert "save_vision_prefill_cache_state(ctx, *vision_cache, prompt_phases, args.partial_vision_kv)" in prompt_fn


def test_vision_prefill_inserts_late_frames_into_initial_video_prefix():
    source = STREAMING_CPP.read_text()
    cache_struct = source.split("struct VisionPrefillCache {", 1)[1].split("\n};", 1)[0]
    build_fn = source.split("VisionPrefillCacheBuildStatus build_vision_prefill_cache(", 1)[1].split(
        "\n}\n\nint run_vision_prefill_prompt_from_committed_cache", 1
    )[0]
    prompt_fn = source.split("int run_vision_prefill_prompt_from_committed_cache(", 1)[1].split(
        "\n}\n\nint run_single_buffer_prompt", 1
    )[0]

    assert "llama_pos video_prefix_insert_pos" in cache_struct
    assert "bool video_prefix_insert_pos_valid" in cache_struct
    assert "video_prefix_state_valid" in cache_struct
    assert "insert_frame_into_closed_video_prefix" in build_fn
    assert "restore_vision_prefill_video_prefix_state" in build_fn
    assert "replay_cached_conversation_tail_after_video_prefix" in build_fn
    assert "KVRepositionTailRestore" in build_fn
    assert "update_first_video_user_message" in source
    assert "save_vision_prefill_video_prefix_state(ctx, *vision_cache" in prompt_fn
    assert "vision_cache->video_prefix_insert_pos = video_prefix_insert_pos_before_suffix" in prompt_fn
    assert "next_cache.open_user_prefix = false" in build_fn
    assert "next_cache.open_user_content.clear()" in build_fn


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
