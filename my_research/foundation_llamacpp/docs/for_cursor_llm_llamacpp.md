# for_cursor_llm_llamacpp

This is the main llama.cpp-side development log for the foundation work.
Keep it as an append-only record of implementation details, execution results, measurements,
and follow-up fixes that matter for future changes.

Use it to record:

- implementation decisions and follow-ups
- control output from `llama.cpp` or hybrid runtime runs
- memory breakdowns
- total runtime
- per-stage timing, such as image encode, prefill, and decode
- backend-specific observations for CPU, Vulkan, OpenCL, or other paths

## Patch Policy

- This file must be updated continuously whenever `my_research/foundation_llamacpp`
  is patched, rebuilt, rerun, debugged, or reorganized. Treat it as the durable
  handoff log for future agents.
- Add new findings here first instead of scattering them across multiple notes.
- When a run changes behavior, write the reason for the change and the observed effect.
- Keep the log cumulative: do not delete old notes unless they are clearly wrong.
- Prefer short dated entries with the command, backend, model, and key metrics.
- Store run artifacts under `my_research/foundation_llamacpp/results/log/<backend>/<model_name>/`.

## Entry Template

```text
date:
backend:
model:
input:
command:
result_dir:
memory:
timing:
output:
issues:
follow_up:
```

## Recent Runs

- 2026-04-30: Rebuilt `build-android-vulkan-noomp` and `build-android-opencl-noomp`,
  then re-ran `InternVL3-1B-Instruct-Q8_0` on Android through
  `run_android_llamacpp.py`.
- Vulkan still loads and offloads layers correctly, but decode fails with
  `vk::DeviceLostError: vk::Queue::submit: ErrorDeviceLost` on this device.
- OpenCL now runs on `QUALCOMM Adreno(TM) 830 (OpenCL 3.0 Adreno(TM) 830)` and
  completes successfully. The run produces the standard CSVs plus
  `memory_timeline_plot.png` and `phase_duration_stacked_bar.png` under
  `my_research/foundation_llamacpp/results/log/opencl/InternVL3-1B-Instruct-Q8_0/`.
- Result folders now use the GGUF stem as the model name so the quantization
  suffix stays visible in the path.
- The external memory sampler is intentionally kept outside `llama.cpp` and now
  samples `/proc/<pid>/status`, `/proc/<pid>/smaps_rollup`, and `/proc/meminfo`
  from Android so the memory timeline can capture RSS, PSS, dirty pages, and
  device memory counters without patching upstream code.
- The sampler was moved into the Android-side shell wrapper so very short runs
  still produce a dense timeline. A CPU rerun on `InternVL3-1B-Instruct-Q8_0`
  produced 66 memory samples in `android_memory_timeline.csv`, which is enough
  to plot the brief load/prefill window instead of a single sample.
- The `memory_timeline_plot.png` for llama.cpp is intentionally simplified to a
  single `MemAvailable` line, matching the xnnpack-style memory graph without
  ExecuTorch-specific phase breakdowns such as `L` or `V_Encode`.
- 2026-04-30: Added `hexagon` backend support to `run_android_llamacpp.py` for
  upstream llama.cpp Snapdragon/HTP runs. The runner now accepts
  `--backend hexagon`, defaults the device to `HTP0`, enables
  `GGML_HEXAGON_EXPERIMENTAL=1`, exports `ADSP_LIBRARY_PATH`, uses `--no-mmap`
  and `-fa on`, and defaults accelerator backends to `--n-gpu-layers 99`.
  Runtime `.so` files are pushed from both `build_dir/lib` and `build_dir/bin`
  so Snapdragon install/package layouts can be used without modifying
  upstream `llama.cpp`.
- 2026-04-30: Attempted to configure an Android Snapdragon build locally.
  `third_party/OpenCL-Headers` and
  `third_party/OpenCL-ICD-Loader/build-android/libOpenCL.so` are the existing
  OpenCL dependencies and work when passed as `OpenCL_INCLUDE_DIR` and
  `OpenCL_LIBRARY`; CMake found OpenCL 3.0 and included the OpenCL backend.
  The Hexagon/HTP configure still fails because this workspace does not have
  Hexagon SDK installed or exported (`HEXAGON_SDK_ROOT` is empty), so
  `libggml-hexagon.so` and `libggml-htp-v*.so` cannot be built yet.
- 2026-04-30: Extracted the Hexagon SDK from
  `ghcr.io/snapdragon-toolchain/arm64-android:v0.3` into
  `third_party/hexagon-sdk` without Docker by reading the GHCR OCI layers
  directly. The SDK root is now
  `third_party/hexagon-sdk/hexagon_sdk.json`, with Hexagon tools under
  `third_party/hexagon-sdk/tools/HEXAGON_Tools/19.0.04`.
- 2026-04-30: Local Snapdragon build succeeded in
  `llama.cpp/build-snapdragon-sdk2` with `GGML_OPENCL=ON` and
  `GGML_HEXAGON=ON`. Because symlinks were skipped during the OCI extraction,
  restored tool aliases inside the SDK: `hexagon-clang -> clang-19`,
  `hexagon-clang++ -> clang-19`, `hexagon-ar -> llvm-ar`, and
  `hexagon-link -> ld.qcld`. Built `llama-mtmd-cli`, `libggml-hexagon.so`,
  `libggml-opencl.so`, and HTP skels `libggml-htp-v73/v75/v79/v81.so`.
- 2026-04-30: Ran `InternVL3-1B-Instruct-Q8_0` through the new Hexagon path
  with `--device HTP0`, `--ctx-size 8192`, `--ubatch-size 256`, and 32 max
  new tokens. The run completed successfully under
  `my_research/foundation_llamacpp/results/log/hexagon/InternVL3-1B-Instruct-Q8_0/`.
  Key output: "The image shows two cats lying on a pink blanket, each with a
  remote control nearby." Timing: image encode 4208 ms, image decode 745 ms,
  prompt eval 5170.23 ms / 271 tokens (52.42 tok/s), decode 370.88 ms / 18
  runs (48.53 tok/s), total 5681.41 ms. Memory summary from llama.cpp:
  HTP0 model buffer 362.58 MiB, KV/context 96.0 MiB, compute 16.02 MiB.
- 2026-04-30: Hexagon/HTP was slower than the OpenCL run for the VLM path
  because the vision/CLIP graph still falls back for several unsupported ops.
  The log reports `clip_ctx: CLIP using HTP0 backend`, but also warns that
  flash attention is unsupported and that the CLIP graph uses unsupported HTP
  operators including `IM2COL`, `CONCAT`, `NORM`, `MUL_MAT`, and
  `FLASH_ATTN_EXT`. This creates CPU fallback, graph splitting, RPC/DMA, and
  scheduling overhead, so image encode was 4208 ms on HTP versus 745 ms in the
  earlier OpenCL run. Prefill was also long because the prompt eval includes
  256 image embedding tokens plus text tokens (271 total), and the HTP schedule
  split the bs=256 graph into 51 pieces. In short: decoder layers were offloaded
  (`offloaded 25/25 layers`), but VLM prefill is still limited by image-token
  batch size and CPU/HTP orchestration overhead.
- 2026-04-30: Started hybrid runtime Option A (split-process bridge) without
  modifying upstream `llama.cpp` or ExecuTorch. Added
  `my_research/foundation_llamacpp/hybrid_bridge/` with a small binary
  float32 embedding file format, `hybrid_vision_dump` for ExecuTorch QNN vision
  encoder output, and `hybrid_decode` for llama.cpp decoder injection through
  the public `mtmd_helper_decode_image_chunk()` path. The decoder still uses
  `mmproj` only to build the text/image token layout; the image embedding bytes
  come from the external QNN output file.
- 2026-04-30: Added `run_android_hybrid_bridge.py` to orchestrate the Android
  split process: preprocess one image to the existing InternVL 448x448 CHW
  float32 `.bin`, push QNN/llama runtime files, run `hybrid_vision_dump`, then
  run `hybrid_decode` with OpenCL and pull `hybrid_vision_stdout.txt`,
  `hybrid_decode_stdout.txt`, `vision_output_stats.csv`, and the bridge
  embedding file into `results/log/hybrid_bridge/<GGUF-stem>/`.
- 2026-04-30: Build validation notes for Option A. Host `hybrid_decode` builds
  successfully in `my_research/foundation_llamacpp/build-hybrid-host`. Android
  OpenCL/QNN bridge builds successfully in
  `my_research/foundation_llamacpp/build-hybrid-android-opencl` after matching
  the ExecuTorch QNN build API level (`ANDROID_PLATFORM=android-30`). Using
  `android-28` failed at link time because
  `libqnn_executorch_backend.so` referenced `memfd_create@LIBC_R`; the unified
  ExecuTorch QNN build was configured with `ANDROID_PLATFORM=android-30`, so
  the overlay bridge must use the same API level.
- 2026-04-30: Re-ran Option A after reconnecting ADB and successfully forced
  the decoder onto OpenCL. The key fix was to avoid the local ICD-loader
  `libOpenCL.so` in `/data/local/tmp/llama-vlm`; renaming it let the runtime use
  the device system OpenCL library, after which `--device GPUOpenCL` detected
  `QUALCOMM Adreno(TM) 830`. Results are under
  `my_research/foundation_llamacpp/results/log/hybrid_bridge_opencl/InternVL3-1B-Instruct-Q8_0/`.
  QNN vision produced `1x256x896` embeddings in 434 ms. `hybrid_decode` used
  OpenCL model/KV/compute buffers, decoded the external image embedding in 47
  ms, prompt eval was 267.20 ms / 271 tokens (1014.24 tok/s), decode was
  398.30 ms / 31 runs (77.83 tok/s), and total time was 1808.82 ms.
- 2026-04-30: Added the same result artifacts for hybrid bridge runs that the
  standalone llama.cpp runner produces: `foundation_output.txt`,
  `foundation_proc.csv`, `vision_output_stats.csv`, `android_memory_timeline.csv`,
  `memory_timeline_plot.png`, and `phase_duration_stacked_bar.png`. The memory
  timeline intentionally plots only Android `MemAvailable`, matching the earlier
  foundation plotting decision. A sampled rerun of the OpenCL hybrid bridge
  completed with QNN vision encode 376 ms, external embedding decode 48 ms,
  prompt eval 268.27 ms / 271 tokens, token decode 411.64 ms / 31 runs, and
  total llama.cpp time 1767.83 ms.
- 2026-04-30: Cleaned up hybrid bridge result naming. The canonical decoder log
  is now only `foundation_output.txt`, matching the standalone runner layout.
  The intermediate `hybrid_decode_stdout.txt` is used only on-device and removed
  from the local result directory after metrics/plots are generated. Manual
  rerun duplicates such as `hybrid_decode_opencl_stdout.txt` and
  `hybrid_opencl_exit_code.txt` were removed from the OpenCL hybrid result.
- 2026-04-30: Normalized the hybrid bridge decoder settings to match the
  standalone OpenCL baseline: `n_ctx=32768`, `n_batch=2048`, and `n_ubatch=512`.
  `run_android_hybrid_bridge.py` now defaults to those values and forwards
  `-ub/--ubatch-size` to `hybrid_decode`. The matched OpenCL hybrid rerun
  confirmed those settings in the log, with OpenCL KV buffer 384.00 MiB and
  OpenCL compute buffer 297.99 MiB. Timing: QNN vision encode 378 ms, external
  embedding decode/inject 46 ms, prompt eval 267.47 ms / 271 tokens, token
  decode 401.33 ms / 30 runs, total llama.cpp time 1800.31 ms.
- 2026-04-30: Expanded
  `docs/executorch_vision_llamacpp_decoder.md` from a feasibility note into the
  current end-to-end hybrid bridge guide. Added the split-process architecture,
  `.svlmemb` boundary, Android run procedure, canonical result artifacts,
  matched OpenCL/QNN timing interpretation, and troubleshooting notes. Important
  conclusion: InternVL3 `vision_encoder_pte` includes `vision_tower`,
  pixel-shuffle/downsample, and `multi_modal_projector`, producing projected
  `1x256x896` decoder-ready image embeddings. Therefore compare hybrid
  `QNN Vision Encoder` against standalone llama.cpp OpenCL `image slice encoded`,
  not against `image decoded` or prompt eval. Also documented that OpenCL phase
  counters can shift due to async queue/synchronization, so phase timing and
  external wall time must be interpreted separately.
- 2026-04-30: Further expanded
  `docs/executorch_vision_llamacpp_decoder.md` with a detailed build guide for
  the hybrid bridge. Documented dependency roles (llama.cpp OpenCL vs ExecuTorch
  QNN vs llama.cpp Hexagon SDK), host sanity build commands, Android
  OpenCL+QNN bridge CMake configure/build commands, expected targets/artifacts,
  runner-pushed libraries, and build troubleshooting. Key build requirements:
  keep host and Android build dirs separate, use `ANDROID_PLATFORM=android-30`
  for the bridge to match `executorch/build-android-unified`, pass
  `OpenCL_INCLUDE_DIR` and `OpenCL_LIBRARY` from `third_party`, and build both
  `hybrid_decode` and `hybrid_vision_dump` from the overlay without patching
  upstream sources.
- 2026-04-30: Clarified the patch policy for this file itself. This document is
  the persistent llama.cpp-side development/handoff log and must be updated
  continuously whenever code, build scripts, docs, run outputs, measurements, or
  troubleshooting decisions under `my_research/foundation_llamacpp` change.
- 2026-04-30: Added precise phase instrumentation for the hybrid bridge, similar
  to the foundation ExecuTorch runners. `hybrid_vision_dump` now writes
  `vision_phase_stats.csv` with `L_VisionLoad`, `ImageLoad`, `V_Encode`, and
  `EmbeddingFileWrite`. `hybrid_decode` now writes `decoder_phase_stats.csv`
  with `ExternalEmbeddingRead`, `L_DecoderLoad`, `LayoutTokenize`,
  `ImagePrefill`, text `T_Prefill`, and per-token `D` rows. The Android runner
  pulls both files, merges them into a foundation-style `foundation_proc.csv`,
  writes summary metrics to `foundation_summary.csv`, and regenerates
  `phase_duration_stacked_bar.png` and `memory_timeline_plot.png` from the
  precise phase rows when available. Also changed the default `--soc-model` to
  `SM8750`; using the old default `SM8550` pushed the v73 skel and caused QNN
  HTP load failure on the current device. Latest precise OpenCL hybrid run:
  QNN `V_Encode` 372 ms, decoder load 3197 ms, image prefill 46 ms, text prefill
  215 ms after an initial 7 ms text chunk, and per-token decode mostly 12-14 ms.
- 2026-04-30: Updated the hybrid bridge to enforce coordinated load-before-run
  timing. `hybrid_decode` now supports `--ready-path`, `--wait-for-embedding`,
  `--wait-timeout-ms`, and records `L_DecoderRuntimeInit` before
  `L_DecoderLoad`.
  `hybrid_vision_dump` now supports `--ready_path`, `--wait_path`, and
  `--wait_timeout_ms`; it loads the QNN module and image tensor first, writes a
  ready flag, then waits for the runner's `start_encode.flag` before `V_Encode`.
  `run_android_hybrid_bridge.py` now pushes and runs a remote shell script that
  starts decoder and vision processes together, waits until both are ready, then
  starts QNN encode. The merged `foundation_proc.csv` is sorted by phase start
  time and no longer appends decoder phases after the vision process. This makes
  `memory_timeline_plot.png` show loading first, then QNN encode, embedding
  handoff, prefill, and decode. Latest coordinated OpenCL hybrid run succeeded
  with exit code 0: `V_Encode` 369 ms, `L_DecoderLoad` 2966 ms,
  `ImagePrefill` 62 ms, prompt eval 278.20 ms / 271 tokens, token decode
  409.38 ms / 31 runs, and total llama.cpp time 2248.70 ms.
- 2026-04-30: Renamed the ambiguous `L_DecoderInit` phase to
  `L_DecoderRuntimeInit`. It represents llama.cpp argument parsing plus OpenCL
  runtime/device/kernel setup before GGUF model/context loading; it is not the
  decoder model load itself. The Android runner also now treats
  `foundation_proc.csv` as the canonical phase input for plotting: after merging
  raw `vision_phase_stats.csv` and `decoder_phase_stats.csv`, it writes
  `foundation_proc.csv`, reads it back, and generates `memory_timeline_plot.png`
  and `phase_duration_stacked_bar.png` from those canonical rows.
- 2026-04-30: Re-ran the coordinated hybrid OpenCL/QNN benchmark after the
  `L_DecoderRuntimeInit` rename. Result directory:
  `results/log/hybrid_bridge_opencl/InternVL3-1B-Instruct-Q8_0`. Exit code 0.
  `foundation_proc.csv` now shows `L_DecoderRuntimeInit` 13644 ms,
  `L_DecoderLoad` 2871 ms, `V_Encode` 369 ms, `ExternalEmbeddingRead` 3 ms,
  `ImagePrefill` 57 ms, and text prefill 10 ms + 210 ms. Summary metrics:
  prompt eval 276.44 ms / 271 tokens, token decode 410.27 ms / 31 runs, total
  llama.cpp time 2225.85 ms. Regenerated `memory_timeline_plot.png` and
  `phase_duration_stacked_bar.png` from the canonical `foundation_proc.csv`.
- 2026-04-30: Updated `phase_duration_stacked_bar.png` generation to exclude
  setup-only phases `L_DecoderRuntimeInit`, `ImageLoad`, and `LayoutTokenize`
  from the stacked runtime breakdown. These rows remain in `foundation_proc.csv`
  for traceability, and only the phase-duration plot aggregation filters them.
- 2026-04-30: Fixed `phase_duration_stacked_bar.png` ordering. The stacked bar
  now orders included phases by their first `elapsed_s_start` in
  `foundation_proc.csv` instead of a hardcoded phase list, so `L_DecoderLoad`
  appears before `V_Encode` when the coordinated load-before-run schedule is
  used.
- 2026-04-30: Further filtered `phase_duration_stacked_bar.png` to exclude load
  phases `L_VisionLoad` and `L_DecoderLoad` in addition to
  `L_DecoderRuntimeInit`, `ImageLoad`, and `LayoutTokenize`. The plot now focuses
  on execution phases (`V_Encode`, embedding handoff, prefill, and decode), while
  all load/setup rows remain in `foundation_proc.csv`.
- 2026-04-30: Added precise phase instrumentation for the standalone llama.cpp
  OpenCL path without modifying upstream llama.cpp. New overlay target:
  `hybrid_bridge/opencl_phase_mtmd.cpp`, built by the existing
  `hybrid_bridge/CMakeLists.txt` as `opencl_phase_mtmd`. It mirrors
  `llama-mtmd-cli` single-turn InternVL execution but records
  `foundation_phase_stats.csv` rows for `L_DecoderRuntimeInit`,
  `L_DecoderLoad`, `ImageLoad`, `LayoutTokenize`, `T_Prefill`, `V_Encode`,
  `ImagePrefill`, and per-token `D`. Updated `run_android_llamacpp.py` so
  backend `opencl` automatically uses `opencl_phase_mtmd` when present, pulls
  `foundation_phase_stats.csv`, writes canonical precise rows to
  `foundation_proc.csv`, writes metrics to `foundation_summary.csv`, and
  regenerates `phase_duration_stacked_bar.png` from the precise rows. Also
  stopped pushing the local `libOpenCL.so` from the pure llama.cpp runner because
  it broke Qualcomm system OpenCL discovery (`platform IDs not available`).
  Latest precise OpenCL run succeeded with exit code 0 in
  `results/log/opencl/InternVL3-1B-Instruct-Q8_0`: `V_Encode` 753 ms,
  `ImagePrefill` 7 ms, text prefill 6 ms + 215 ms, token decode rows mostly
  12-14 ms, prompt eval 981.44 ms / 271 tokens, token decode 382.91 ms / 29
  runs, total llama.cpp time 2425.68 ms.
- 2026-04-30: Corrected standalone OpenCL precise phase ordering. The first
  `opencl_phase_mtmd` implementation followed mtmd chunk order, which allowed a
  small text prefill chunk before `V_Encode`. Updated the overlay tool to scan
  image chunks after `LayoutTokenize`, run all `V_Encode` phases first, copy the
  projected embeddings, and then execute text/image prefill in chunk order using
  the precomputed embeddings. Latest rerun exit code 0:
  `L_DecoderRuntimeInit` 13850 ms, `L_DecoderLoad` 2827 ms, `V_Encode` 714 ms,
  text prefill 19 ms + 214 ms, `ImagePrefill` 36 ms, per-token decode mostly
  12-16 ms, prompt eval 269.20 ms / 271 tokens, token decode 377.38 ms / 29
  runs, total llama.cpp time 2444.37 ms. `foundation_proc.csv` now shows
  `V_Encode` before any `T_Prefill`.
- 2026-04-30: Expanded
  `docs/executorch_vision_llamacpp_decoder.md` into a detailed handoff guide for
  the current hybrid and standalone OpenCL measurement flows. Added a current
  implementation snapshot, file-by-file roles for `hybrid_vision_dump`,
  `hybrid_decode`, `hybrid_embedding_file`, `opencl_phase_mtmd`,
  `run_android_hybrid_bridge.py`, and `run_android_llamacpp.py`; documented
  load-before-run coordination, standalone OpenCL `V_Encode`-before-prefill
  ordering, CMake targets, host/Android build commands, canonical artifacts,
  plot filtering rules, OpenCL loader troubleshooting, and latest matched
  result numbers for hybrid QNN+OpenCL vs standalone OpenCL. This was done so a
  future agent can continue from the docs without reconstructing the chat
  history.

Suggested note format:

```text
model:
backend:
input:
command:
memory:
timing:
output:
issues:
```

Keep raw binaries, build outputs, and other large artifacts out of this folder.
