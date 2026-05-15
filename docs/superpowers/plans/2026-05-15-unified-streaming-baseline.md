# Unified Streaming Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify hybrid offline and streaming media runs behind `hybrid_streaming_decode`, add latest-only `--online-buffer`, and clean up user-facing mode names.

**Architecture:** Keep existing media preparation and artifact layout, but extend manifests so offline media can be consumed by the streaming binary. Add a small prompt-format abstraction in Python and matching C++ helpers. Use TDD contract tests for CLI normalization, manifest shape, runner binary selection, and online-buffer scheduling semantics.

**Tech Stack:** Python runner/tests, C++17 Android hybrid bridge, llama.cpp mtmd, ExecuTorch QNN vision encoder, pytest, CMake.

---

### Task 1: CLI Compatibility And Manifest Shape

**Files:**
- Modify: `my_research/foundation_llamacpp/runner/media.py`
- Modify: `my_research/foundation_llamacpp/runner/config.py`
- Modify: `my_research/foundation_llamacpp/runner/cli.py`
- Test: `my_research/foundation_llamacpp/tests/test_streaming_media.py`
- Test: `my_research/foundation_llamacpp/tests/test_result_artifact_layout.py`

- [ ] Add tests proving `on-demand` normalizes to `on_demand`, while `single-buffer` remains an alias.
- [ ] Add tests proving `--multi-image` and `--images` both populate `args.multi_image`/`args.images` compatibility data.
- [ ] Add manifest tests proving image, multi-image, and video manifests include `source_kind`, `frames`, `prompt_events`, and a fully formatted prompt.
- [ ] Implement canonical `on_demand` constants and aliases.
- [ ] Implement `--multi-image` CLI alias.
- [ ] Run `pytest -q my_research/foundation_llamacpp/tests/test_streaming_media.py my_research/foundation_llamacpp/tests/test_result_artifact_layout.py`.

### Task 2: Unified Hybrid Offline Runner

**Files:**
- Modify: `my_research/foundation_llamacpp/runner/cli.py`
- Modify: `my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp`
- Test: `my_research/foundation_llamacpp/tests/test_result_artifact_layout.py`
- Test: `my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py`

- [ ] Add tests proving hybrid image/multi-image/video pushes and executes `hybrid_streaming_decode`, not `hybrid_vision_dump + hybrid_decode`.
- [ ] Extend `Manifest` parsing with `source_kind`, `prompt`, and offline prompt event handling.
- [ ] Add `--media-mode image|multi-image|video|streaming` to `hybrid_streaming_decode`.
- [ ] Implement a non-streaming one-shot path inside `hybrid_streaming_decode` that loads the decoder/encoder once, encodes all manifest bins, evaluates the manifest prompt, generates output, and writes phase/token artifacts.
- [ ] Update Python remote script generation so hybrid offline media invokes `hybrid_streaming_decode`.
- [ ] Build `hybrid_streaming_decode`.

### Task 3: Latest-Only Online Buffer

**Files:**
- Modify: `my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp`
- Modify: `my_research/foundation_llamacpp/runner/cli.py`
- Modify: `my_research/foundation_llamacpp/runner/artifacts.py`
- Test: `my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py`
- Test: `my_research/foundation_llamacpp/tests/test_result_artifact_layout.py`

- [ ] Add static contract tests for `--online-buffer` parsing, coalesced cache jobs, and prompt frame selection at processing start.
- [ ] Add `Args::online_buffer`.
- [ ] In producer, keep shared `latest_frame` and `available_frames`.
- [ ] When `online_buffer` is true, coalesce pending cache updates and leave prompt frame selection to the consumer.
- [ ] In consumer, resolve prompt/cache frames from latest state just before processing.
- [ ] Write `stream_buffer_summary.txt` with input FPS, processed visual FPS, skipped cache updates, and prompt lag.
- [ ] Pull `stream_buffer_summary.txt` into `txt_json`.

### Task 4: Prompt Formatter Boundary

**Files:**
- Modify: `my_research/foundation_llamacpp/runner/media.py`
- Modify: `my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp`
- Test: `my_research/foundation_llamacpp/tests/test_streaming_media.py`
- Test: `my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py`

- [ ] Add Python formatter helpers for InternVL3 image, multi-image, and video prompts.
- [ ] Add C++ prompt-family field with InternVL3 helpers for stream frame/video prefixes.
- [ ] Add tests proving InternVL3 output remains unchanged.

### Task 5: Smoke Script, Docs, And 1B Q8 Validation

**Files:**
- Modify: `my_research/foundation_llamacpp/scripts/run_artifact_layout_1b_q8.sh`
- Modify: `my_research/foundation_llamacpp/docs/README.md`
- Modify: `my_research/foundation_llamacpp/docs/project_structure.md`
- Modify: `my_research/foundation_llamacpp/docs/for_cursor_llm_llamacpp_version2.md`

- [ ] Update the script to run the eight-run matrix, including online-buffer.
- [ ] Update README sections for `on-demand`, `--multi-image`, unified binary, and online-buffer metrics.
- [ ] Run Python tests and C++ build.
- [ ] Run `bash my_research/foundation_llamacpp/scripts/run_artifact_layout_1b_q8.sh`.
- [ ] Verify each run has `csv/`, `png/`, and `txt_json/`, and that streaming runs include `txt_json/stream_buffer_summary.txt`.

