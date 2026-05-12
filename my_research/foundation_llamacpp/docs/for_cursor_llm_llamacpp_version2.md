# for_cursor_llm_llamacpp_version2

This is the active implementation log for the structured
`my_research/foundation_llamacpp` refactor. Use this file for new decisions,
workflow notes, validation results, and follow-up tasks. The older cumulative log
is retained under `docs/archive/for_cursor_llm_llamacpp.md`.

## 2026-05-11: Hybrid Bridge Refactor Baseline

- Refactor goal: separate media mode (`text`, `image`, `video_file`, future
  `streaming`) from backend mode (`cpu`, `opencl`, `hybrid_qnn_opencl`) while
  preserving current Android image/video behavior.
- `run_android_hybrid_bridge.py` remains the compatibility entrypoint, but new
  code should move into `my_research/foundation_llamacpp/runner/`.
- Added runner contracts:
  - `runner/config.py`: `MediaMode`, `BackendMode`, `PreparedMedia`, and mode
    normalization helpers.
  - `runner/media.py`: image/video preparation, InternVL tiling, and versioned
    `media_manifest.json` generation (`schema_version: 2`).
- `build-hybrid-host` is treated as generated/obsolete host sanity-build output.
  Prefer the active Android build directory for device runs and temporary
  `/tmp/...` host build dirs for compile checks.
- Known-good runtime baselines before this refactor:
  - image hybrid path works with QNN vision tower + OpenCL decoder.
  - 4-frame video hybrid: `4x256x4096` merged embedding, four IMAGE chunks.
  - 16-frame video hybrid: `16x256x4096` merged embedding, sixteen IMAGE chunks.

## 2026-05-11: Runner And Bridge Module Split

- Python runner split:
  - `runner/media.py` now owns image/video preparation and InternVL tiling.
  - `runner/config.py` defines explicit media/backend modes.
  - `runner/artifacts.py`, `runner/remote.py`, `runner/finalize.py`, and
    `runner/backends/*` provide the first package boundaries for artifact names,
    adb helpers, finalization helpers, and backend contracts.
  - `runner/cli.py` now owns the CLI/orchestration implementation.
  - `run_android_hybrid_bridge.py` is a compatibility wrapper that imports
    `my_research.foundation_llamacpp.runner.cli.main`.
- C++ bridge split:
  - `phase_trace.hpp` centralizes CSV phase row writing and phase descriptions.
  - `file_sync.hpp` centralizes ready/wait text-file synchronization.
  - `vision_encoder_et.{hpp,cpp}` owns ExecuTorch/QNN image loading and
    multi-image encode merging. `hybrid_vision_dump.cpp` is now a thin gflags
    wrapper around that module plus `.svlmemb`/stats output.
- Validation:
  - `python -m py_compile my_research/foundation_llamacpp/run_android_hybrid_bridge.py my_research/foundation_llamacpp/runner/*.py my_research/foundation_llamacpp/runner/backends/*.py`
    passed.
  - `cmake --build my_research/foundation_llamacpp/build-hybrid-android-opencl --target hybrid_decode opencl_phase_mtmd hybrid_vision_dump -j2`
    passed.
  - Android image smoke passed after one transient adb push protocol fault retry.
  - Android 4-frame video smoke passed.
  - Android 16-frame video smoke passed with `media_manifest.json`
    `schema_version: 2`, `source_kind: video`, `frame_bins: 16`,
    `vision_output_stats.csv` `output_dims: 16x256x4096`, and sixteen IMAGE
    chunks in `foundation_inference_tokens.txt`.
- Fix after smoke: ensure `runner/cli.py` recreates `result_dir` before writing
  `host_adb_output.txt`, and preserve `input_count` when finalizing
  `vision_output_stats.csv`.

## 2026-05-11: 8-Frame Video OpenCL/Hybrid Refactor Test

- Ran the refactored compatibility entrypoint with `--video
  my_research/foundation_llamacpp/sample_images/surveil_8.mp4`,
  `--num-segments 8`, `--max-num 1`, `--ctx-size 32768`, and f16 KV.
- OpenCL standalone (`--processor gpu`) completed with `foundation_exit_code.txt`
  `0` and eight IMAGE chunks in `foundation_inference_tokens.txt`.
- Hybrid QNN vision + OpenCL decoder (`--processor hybrid`) completed with
  `foundation_exit_code.txt` `0`.
- Hybrid artifact checks:
  - `media_manifest.json`: `schema_version: 2`, `source_kind: video`,
    `num_segments: 8`, `frame_bins: 8`, `layout_images: 8`,
    `num_patches_list: [1, 1, 1, 1, 1, 1, 1, 1]`.
  - `vision_output_stats.csv`: `output_dims: 8x256x4096`,
    `input_count: 8`, `output_values: 8388608`.
  - `foundation_inference_tokens.txt`: eight frame-prefixed IMAGE chunks.

## 2026-05-11: Project Structure Documentation

- Added `docs/project_structure.md` as the English structural guide for the
  refactored foundation llama.cpp project.
- The document explains:
  - top-level project layout and generated-output boundaries,
  - Python runner modules and responsibilities,
  - C++ bridge targets and shared helpers,
  - text, standalone OpenCL image/video, and hybrid QNN/OpenCL runtime flows,
  - result artifacts and validation points,
  - build notes and extension guidelines for future media/backend/streaming work,
  - remaining cleanup gaps after the first refactor.
- Linked the new document from `docs/README.md`.

## 2026-05-11: OpenCL 16-Frame Output Investigation

- Investigated why standalone OpenCL video inference generated only
  `Walking.<|im_end|>` while hybrid QNN vision + OpenCL decoder produced a
  longer scene description.
- Fixed an OpenCL trace bug in `hybrid_bridge/opencl_phase_mtmd.cpp`: loaded
  bitmaps now receive stable ids (`image_1`, `image_2`, ...), and
  `foundation_inference_tokens.txt` reports incrementing IMAGE `image_index`
  values instead of `image_index=1` for every frame.
- Rebuilt `opencl_phase_mtmd` and reran Q8_0 OpenCL 16-frame video. Validation:
  exit code `0`, sixteen IMAGE chunks, `image_index=1..16`,
  `mtmd_chunk_id=image_1..image_16`.
- Reran hybrid QNN+OpenCL with the same 16-frame prompt/settings. Both paths now
  show the same prompt-side token count (`4244`) and sixteen IMAGE chunks, so
  prompt layout/frame count mismatch is ruled out.
- Remaining behavior: OpenCL standalone still emits `Walking.<|im_end|>`, while
  hybrid emits a longer bank-office scene description. The discrepancy is in the
  vision feature path, not text quantization, prompt layout, or missing frames:
  standalone OpenCL runs llama.cpp/mtmd's full InternVL vision encoder+projector,
  while hybrid runs the ExecuTorch/QNN pre-projector vision tower and then the
  same GGUF mmproj/projector in `hybrid_decode`.
- `runner/artifacts.py` now also pulls `media_manifest.json` for standalone
  OpenCL/CPU runs so future result folders preserve the exact sampled frame
  manifest.

## 2026-05-11: OpenCL vs Hybrid Vision Embedding Comparison

- Added diagnostic projected-embedding dumps:
  - standalone OpenCL writes `opencl_projected_embedding.svlmemb` after
    `mtmd_encode_chunk()` and before image prefill.
  - hybrid writes `hybrid_projected_embedding.svlmemb` after external QNN
    pre-projector features are projected by the same GGUF `mmproj`.
- Rebuilt `opencl_phase_mtmd` and `hybrid_decode`, then reran the same F16
  single-image Golden Gate test under `results/log`.
- Both paths produced the same decoder-side shape: `1 x 256 x 896`, so the
  mismatch is not image-token count, prompt layout, or decoder embedding
  dimensionality.
- Numeric comparison of projected embeddings:
  - OpenCL projected stats: mean `-0.0393549`, std `0.598779`, L2 `287.393`.
  - Hybrid projected stats: mean `-0.0533506`, std `0.764133`, L2 `366.859`.
  - Global cosine similarity: `0.630686`.
  - Mean absolute difference: `0.471819`; RMS difference: `0.605839`; max
    absolute difference: `4.61126`.
  - Per-image-token cosine: mean `0.626368`, min `0.142897`, median
    `0.653571`, max `0.911484`.
- Interpretation: OpenCL and Hybrid are feeding substantially different visual
  embeddings into the same text decoder. Since Hybrid and the HF reference give
  semantically correct answers while OpenCL often hallucinates, the remaining
  bug is inside the standalone `llama.cpp`/`mtmd` InternVL vision path before
  decoder prefill: image preprocessing, InternVL graph implementation, pixel
  shuffle, or feature ordering.

## 2026-05-11: OpenCL InternVL Mmproj Timing Split

- Added project-specific timing instrumentation to split standalone OpenCL
  InternVL image encode into:
  - `V_Encode`: llama.cpp/mtmd InternVL vision tower through pixel shuffle,
    before the multi-modal projector.
  - `Mmproj`: llama.cpp/mtmd InternVL projector/mmproj-only graph.
- Implementation notes:
  - `clip_graph_internvl::build_preprojector()` factors the pre-projector graph
    out of the existing full InternVL graph.
  - `clip_image_encode_internvl_split()` runs the pre-projector graph, copies the
    `256 x 4096` feature tensor to host, then reuses the existing
    `clip_project_internvl_features()` projector-only path to produce
    decoder-side `256 x 896` embeddings.
  - `mtmd_encode_chunk_split_timing()` exposes this split for image chunks, and
    the project-local `opencl_phase_mtmd` wrapper records `V_Encode` and
    `Mmproj` as separate phase rows.
- Validation:
  - Rebuilt Android targets with
    `cmake --build my_research/foundation_llamacpp/build-hybrid-android-opencl --target opencl_phase_mtmd hybrid_decode hybrid_vision_dump -j$(nproc)`.
  - Reran README OpenCL single-image Golden Gate command (`ctx=32768`, Q8_0,
    f16 KV). Result folder:
    `results/log/InternVL3-1B-Instruct-Q8_0_opencl_ctx_32768_kv16`.
  - Added fixed-image warmup-before-measurement for both paths:
    - OpenCL `opencl_phase_mtmd` runs one split InternVL encode+mmproj pass on
      `sample_images/golden_gate_bridge_448.jpg` and discards it before
      recording measured `V_Encode`/`Mmproj`.
    - Hybrid QNN vision runs one `encoder.encode()` warmup on the same fixed
      Golden Gate bin. `hybrid_vision_dump` also writes that fixed warmup
      embedding, and `hybrid_decode` uses it to warm
      `mtmd_project_features()` before recording measured `Mmproj`.
  - Fixed-warmup OpenCL rows with projector scheduler/graph rebuilt per call:
    `V_Encode=726 ms`, `Mmproj=14 ms`, `ImagePrefill=47 ms`, first
    `T_Prefill=6 ms`, second `T_Prefill=219 ms`.
  - Fixed-warmup Hybrid rows on the same measured Golden Gate image: QNN
    `V_Encode=359 ms`, `Mmproj=48 ms`, `ImagePrefill=6 ms`, first
    `T_Prefill=6 ms`, second `T_Prefill=216 ms`.
  - Interpretation: fixed-image warmup makes OpenCL and Hybrid warmup inputs
    independent of the measured input/video. Both OpenCL split and hybrid
    external-feature projection now rebuild the projector-only scheduler/graph
    for each measured `Mmproj` call. A cached projector graph attempt was removed
    because it aborted with a ggml layout mismatch across calls.
  - Tested Hybrid `ImagePrefill` with an OpenCL-style `std::vector<float>` copy
    before `mtmd_helper_decode_image_chunk()`. `ImagePrefill` stayed low
    (`6 ms`), so the OpenCL-vs-Hybrid `ImagePrefill` gap is not caused simply by
    passing a copied host vector instead of the direct `mtmd_get_output_embd()`
    pointer.
- Timing semantics:
  - `ImagePrefill` / `I_Prefill` is not image encoding. It is the
    `llama_decode()` call that inserts already projected image embeddings into
    the LLM KV cache.
  - Bridge timing now calls `llama_synchronize()` immediately after every
    `llama_decode()` in image prefill, text prefill, and token decode. This is
    required for OpenCL because `llama_decode()` can enqueue work
    asynchronously; without the synchronize, image prefill cost can appear in
    the following `T_Prefill` or decode phase.
  - Validation on InternVL3-8B Q4_K_M at `ctx=1024`: before synchronization,
    Hybrid reported `ImagePrefill=16 ms` and following `T_Prefill=7759 ms`.
    After adding `llama_synchronize()`, the same run reported
    `ImagePrefill=3888 ms` and following `T_Prefill=657 ms`. OpenCL showed the
    same corrected distribution: `ImagePrefill=3967 ms`, following
    `T_Prefill=624 ms`.
  - Latest synchronized 8B rows:
    - OpenCL `ctx=1024`: `V_Encode=723 ms`, `Mmproj=19 ms`,
      first `T_Prefill=465 ms`, `ImagePrefill=3967 ms`, second
      `T_Prefill=624 ms`.
    - Hybrid `ctx=1024` using the existing QNN pre-projector vision encoder:
      QNN `V_Encode=407 ms`, `Mmproj=16 ms`, first `T_Prefill=460 ms`,
      `ImagePrefill=3888 ms`, second `T_Prefill=657 ms`.
  - The mtmd helper stdout line `image decoded ... in N ms` is printed inside
    `mtmd_helper_decode_image_chunk()` before the bridge-level synchronize, so
    it can remain tiny (`12-13 ms`) even when synchronized CSV phase rows show
    the true multi-second image prefill.
  - `T_Prefill` is text-token `llama_decode()` and may include logits for the
    last text chunk. Therefore image-token count alone does not guarantee that
    `I_Prefill` must be slower than `T_Prefill`.
- Upgrade-safety note: this touches original llama.cpp/mtmd files for a timing
  probe. Keep the change small and manage it as a local patch/debug hook when
  updating llama.cpp upstream.

## 2026-05-11: Documentation Sync For Warmed Timing

- Updated `docs/README.md` to document:
  - bridge-local warmup policy for OpenCL and Hybrid,
  - OpenCL `V_Encode` / `Mmproj` split semantics,
  - `ImagePrefill` vs `T_Prefill` timing semantics,
  - latest warmed Q8_0 Golden Gate OpenCL/Hybrid timing results,
  - projected embedding dump artifacts.
- Updated `docs/project_structure.md` to record:
  - the local llama.cpp/mtmd timing hook as an upstream patch/debug hook,
  - warmup responsibilities of `opencl_phase_mtmd`, `hybrid_vision_dump`, and
    `hybrid_decode`,
  - `Mmproj` phase semantics and projected embedding artifacts.

## 2026-05-12: Single-Buffer Streaming Video Mode

- Implemented file-backed streaming video simulation for `--streaming-video`
  with `--single-buffer`.
- Semantics:
  - host samples the input video at `--sampling-fps` and writes a streaming
    `media_manifest.json`;
  - Android-side runner replays sampled frames according to their stream
    timestamps;
  - `SingleBufferUpdate` replaces the current frame pointer with the latest
    sampled frame;
  - prompt events from `--time '[...]'` and `--prompt '["...", "..."]'` are
    captured at their stream timestamps and answered with the frame buffered at
    prompt arrival;
  - prompt execution is serialized, so later prompts can wait behind earlier
    prefill/decode, but their selected frame remains the one captured at arrival.
- `runner/media.py` now prepares `source_kind: streaming_video` manifests. In
  single-buffer mode it writes both:
  - `stream_frame_<idx>.png` for mtmd layout/tokenization;
  - `stream_frame_<idx>.bin` for QNN hybrid vision encoding.
- Streaming-specific manifest fields include:
  - `source_fps`,
  - `sampling_fps`,
  - `duration_s`,
  - `effective_duration_s`,
  - `max_video_time`,
  - `stream_mode: single_buffer`,
  - `prompt_events`,
  - per-frame `timestamp_s`, `video_frame_index`, and tile metadata.
- `runner/cli.py` validates streaming arguments:
  - `--streaming-video` is mutually exclusive with `--image` and `--video`;
  - `--sampling-fps` must be positive;
  - `--time` and JSON-list `--prompt` must have matching lengths;
  - `--max-video-time` / `--max_video_time` caps sampled duration.
- Result folder names now include `_streaming` for streaming runs, e.g.
  `InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_kv16`.
- Streaming artifacts:
  - `stream_events.csv`: frame enqueue, `SingleBufferUpdate`, prompt arrival,
    and prompt decode spans.
  - `streaming_phase_stats.csv`: setup, frame-buffer, vision, mmproj, prefill,
    and decode phase rows.
  - `foundation_proc.csv`: normalized copy of streaming phase rows.
  - `streaming_phase_timeline.png`: prompt timeline plot.
  - `stream_response_<idx>.txt`, `stream_token_io_<idx>.txt`,
    `stream_inference_tokens_<idx>.txt`: per-prompt output and token traces.
- Initial Q8 2B hybrid streaming validation:
  - command used `InternVL3-2B-Instruct-Q8_0.gguf`,
    `mmproj-InternVL3-2B-Instruct-Q8_0.gguf`,
    `--streaming-video sample_images/surveil_8.mp4`,
    `--single-buffer`, `--sampling-fps 1.0`, `--max_video_time 15`,
    prompts at `5s` and `8s`;
  - result:
    `my_research/foundation_llamacpp/results/log/InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_kv16/`;
  - `foundation_exit_code.txt=0`;
  - QNN `V_Encode` rows were about `376 ms` and `373 ms`;
  - Q8 produced normal text, unlike the Q4 2B run that repeated `</quad>`.

## 2026-05-12: Streaming Hybrid Uses QNN Vision Encoder

- The first single-buffer streaming prototype used the llama.cpp/OpenCL
  multimodal path for prompt handling. The hybrid processor path has now been
  corrected so `--processor hybrid --streaming-video --single-buffer` uses
  ExecuTorch/QNN vision encoding.
- Added reusable `VisionEncoderSession` in `vision_encoder_et.{hpp,cpp}`:
  - loads the ExecuTorch/QNN module once;
  - exposes `encode()` for per-frame image bin paths;
  - exposes `encode_with_optional_warmup()` for a single startup warmup;
  - preserves the existing `encode_images_with_executorch()` helper by
    implementing it through the session.
- `hybrid_streaming_decode` now:
  - loads QNN `VisionEncoderSession` once at stream startup;
  - warms it with the fixed Golden Gate bin when provided;
  - loads the llama.cpp/mmproj decode context once;
  - keeps chat history and KV state across prompt events;
  - for each prompt, QNN-encodes only the selected buffered `.bin` frame;
  - feeds the resulting pre-projector embedding through `eval_with_external_embedding()`;
  - then calls `generate_response()` without clearing chat/KV state.
- Timing bug fixed:
  - raw ExecuTorch/QNN phase timestamps used a different timer origin, causing
    very large values such as `1777880000` in `foundation_proc.csv`;
  - streaming now rebases QNN `L_VisionLoad`, `ImageLoad`, and `V_Encode`
    durations onto the llama.cpp `ggml_time_ms()` origin used by the rest of the
    run.
- Verified Q8/QNN streaming after the fix:
  - `L_VisionLoad` appears near `0s`;
  - prompt `ImageLoad` and `V_Encode` rows appear on the same stream execution
    timeline as `LayoutTokenize`, `Mmproj`, `ImagePrefill`, `T_Prefill`, and `D`;
  - no absolute-timestamp rows remain.

## 2026-05-12: Streaming Multi-Turn Chat And Token Traces

- Multi-turn streaming state:
  - `hybrid_streaming_decode` no longer clears llama memory, sampler state,
    `ctx.chat_history`, or `ctx.n_past` between prompts;
  - assistant messages are appended in the shared generation helpers, matching
    llama.cpp/mtmd multi-turn behavior;
  - validation prompt 1 `What did I ask earlier???` answered that prompt 0 was
    about the situation in the image.
- Raw token tracing bug fixed:
  - initial streaming trace code opened `foundation_inference_tokens.txt` for
    each prompt, so later prompts truncated earlier raw token traces;
  - streaming now writes per-turn raw traces to
    `stream_inference_tokens_<idx>.txt`;
  - `foundation_inference_tokens.txt` aggregates all per-turn raw traces with
    `===== stream prompt <idx> @ <time>s =====` headers.
- Follow-up flush/close fix:
  - the first aggregate implementation copied `stream_inference_tokens_<idx>.txt`
    while the trace writer still had the file open, so aggregate sections could
    be truncated even when per-turn files were complete;
  - `hybrid_streaming_decode` now closes the trace writer before reading the raw
    trace into the aggregate.
- `runner/cli.py::_pull_outputs()` now expands remote wildcard artifact names
  such as `stream_inference_tokens_*.txt`, so per-turn token traces are pulled
  into host result directories.

## 2026-05-12: OpenCL Single-Buffer Streaming

- Added OpenCL streaming support for
  `--processor gpu --streaming-video ... --single-buffer`.
- New C++ target:
  - `opencl_streaming_decode`, built from `hybrid_streaming_decode.cpp`;
  - compiles with `STREAMINGVLM_OPENCL_PHASE_MTMD_NO_MAIN=1`;
  - includes `opencl_phase_mtmd.cpp` in-process;
  - reuses llama.cpp/mtmd OpenCL full-vision encode, mmproj, prefill, and
    decode while preserving the same streaming event model as hybrid.
- Existing QNN target remains:
  - `hybrid_streaming_decode` compiles with
    `STREAMINGVLM_STREAMING_DECODE_USE_QNN=1` and
    `STREAMINGVLM_HYBRID_DECODE_NO_MAIN=1`;
  - it uses `VisionEncoderSession` and `hybrid_decode.cpp` in-process.
- Runner changes:
  - `--streaming-video` is now accepted for `--processor gpu` and
    `--processor hybrid`;
  - GPU streaming pushes/runs `opencl_streaming_decode`;
  - Hybrid streaming pushes/runs `hybrid_streaming_decode`;
  - both paths share `stream_events.csv`, `streaming_phase_stats.csv`,
    `foundation_proc.csv`, `streaming_phase_timeline.png`, and per-turn token
    trace finalization;
  - `HYBRID_STREAMING_PULL_ARTIFACTS` includes both
    `hybrid_streaming_stdout.txt` and `opencl_streaming_stdout.txt`.
- Build validation:
  - reconfigured `build-hybrid-android-opencl` after adding the new target;
  - built `opencl_streaming_decode` and `hybrid_streaming_decode` successfully.
- OpenCL streaming smoke:
  - command used Q8 2B, `--max_video_time 10`, prompts at `5s` and `8s`;
  - result:
    `my_research/foundation_llamacpp/results/log/InternVL3-2B-Instruct-Q8_0_opencl_ctx_4096_streaming_kv16/`;
  - `foundation_exit_code.txt=0`;
  - `foundation_proc.csv` contains OpenCL `V_Encode`, `Mmproj`,
    `ImagePrefill`, `T_Prefill`, and `D`;
  - second-answer quality did not recall the earlier question as cleanly as the
    hybrid QNN run, but execution and logs are correct.

## 2026-05-12: Streaming Timeline Plot Uses Stream Time

- Updated `runner/cli.py::_write_png_streaming_phase_timeline()` so the x-axis
  is stream/video time rather than first-prompt-relative time.
- The function reads `stream_events.csv`, derives the elapsed-time to video-time
  offset from the first frame/buffer event, and converts all phase rows before
  plotting.
- Prompt markers are labeled like `Prompt 0 @ 5.0s`.
- This avoids the confusing previous behavior where a prompt arriving at stream
  time `3s` or `5s` was displayed at x-axis `0s`.
- Regenerated current 2B Q8 hybrid and OpenCL streaming plots.

## 2026-05-12: 8B Hybrid Streaming Validation

- Ran hybrid single-buffer streaming with the available 8B weights:
  - text model:
    `llama.cpp/models/InternVL3-8B-Instruct-GGUF/InternVL3-8B-Instruct-Q4_K_M.gguf`;
  - mmproj:
    `llama.cpp/models/InternVL3-8B-Instruct-GGUF/mmproj-InternVL3-8B-Instruct-Q8_0.gguf`;
  - QNN vision:
    `my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte`;
  - `--ctx-size 4096`, f16 KV, prompts at `5s` and `8s`.
- Result:
  `my_research/foundation_llamacpp/results/log/InternVL3-8B-Instruct-Q4_K_M_hybrid_ctx_4096_streaming_kv16/`.
- Validation:
  - `foundation_exit_code.txt=0`;
  - QNN `V_Encode`: prompt 0 about `415 ms`, prompt 1 about `371 ms`;
  - `ImagePrefill`: prompt 0 about `3973 ms`, prompt 1 about `4800 ms`;
  - decode tokens mostly about `160-188 ms/token`;
  - prompt 1 answered that the earlier question asked about the situation in the
    image, confirming multi-turn state was preserved.
