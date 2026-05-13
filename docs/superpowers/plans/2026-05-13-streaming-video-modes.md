# Streaming Video Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `single-buffer`, `sliding-window`, and `vision-prefill` modes to the existing Android `--streaming-video` runner.

**Architecture:** Keep Python responsible for media sampling, manifest metadata, and CLI compatibility. Move frame-selection semantics into small pure helpers, then mirror the selection in the C++ streaming runner so prompt jobs can carry one or more frames.

**Tech Stack:** Python runner, pytest unit tests, C++17 llama.cpp/mtmd bridge.

---

### Task 1: Python Stream Mode Helpers

**Files:**
- Modify: `my_research/foundation_llamacpp/runner/media.py`
- Test: `my_research/foundation_llamacpp/tests/test_streaming_media.py`

- [ ] Add tests for mode normalization, window frame selection, even frame limiting, and video prompt rendering.
- [ ] Implement pure helpers in `runner/media.py`.
- [ ] Run `pytest my_research/foundation_llamacpp/tests/test_streaming_media.py -q`.

### Task 2: CLI And Manifest Wiring

**Files:**
- Modify: `my_research/foundation_llamacpp/runner/cli.py`
- Modify: `my_research/foundation_llamacpp/runner/media.py`

- [ ] Add `--stream-mode`, `--window-sec`, and `--window-max-frames`.
- [ ] Preserve `--single-buffer` as an alias.
- [ ] Write stream mode and window controls to `media_manifest.json`.
- [ ] Pass stream mode and window controls to `hybrid_streaming_decode` / `opencl_streaming_decode`.

### Task 3: C++ Multi-Frame Prompt Jobs

**Files:**
- Modify: `my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp`

- [ ] Parse stream mode and window controls.
- [ ] Change `PromptJob` from one frame to many frames.
- [ ] For `single-buffer`, preserve existing behavior and multi-turn state.
- [ ] For `sliding-window` and `vision-prefill`, reset decoder state before each prompt and format all selected frames as `Frame N:` video input.
- [ ] Support both QNN hybrid and OpenCL-only paths using existing multi-image evaluation helpers.

### Task 4: Documentation And Verification

**Files:**
- Modify: `my_research/foundation_llamacpp/docs/README.md`

- [ ] Document the new modes and example commands.
- [ ] Run the Python unit tests.
- [ ] Run a compile check if the local Android/C++ toolchain is available; otherwise report that it was not run.
