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

- 2026-05-11: Added first-class `--video` plumbing to the Android hybrid bridge.
  `run_android_hybrid_bridge.py` now accepts `--video`, `--num-segments`, and
  `--max-num` and prepares video as ordered InternVL frame/tile images. The host
  samples frames with the same center-biased segment math used by
  `my_research/test_python/internvl3_1b_video_chat.py`, applies the same
  dynamic tiling rules, writes `frame_XXXX_tile_YYYY.bin` tensors plus PNG
  layout images, and emits `media_manifest.json`. Hybrid execution still loads
  the decoder/mmproj and QNN encoder before `start_encode.flag`; then
  `hybrid_vision_dump --image_paths=...` encodes every frame/tile in order,
  writes one merged `.svlmemb`, and records one `V_Encode` phase per input.
  `hybrid_decode` now consumes external embeddings with a cursor so each mtmd
  IMAGE chunk gets the next slice instead of reusing the first buffer. The
  video prompt is `Frame1: <__media__>\n...` plus the raw prompt; mtmd expands
  each marker as InternVL `<img>` + vision slots + `</img>`. With `--max-num 1`
  this is exactly one image block per sampled frame; with `--max-num > 1`, each
  frame label contains multiple tile markers in frame/tile order. Phase plotting
  now exposes a combined `Prefill` row while retaining detailed `ImagePrefill`
  and `T_Prefill` rows in CSV traces.

- 2026-05-11: Trace dumps: no `HF_OFFICIAL_*` blocks; `User:` echoes raw `-p`. Removed
  `internvl_mtmd_prompt.hpp` (no HF `<image>` leader stripping on the bridge). InternVL vision wrapper in
  **`llama.cpp/tools/mtmd/mtmd.cpp`** uses **`<img>` … `</img>`** (`PROJECTOR_TYPE_INTERNVL`).
  Token-ID parity notes live only under **`my_research/foundation/docs/for_cursor_llm.md`** if needed.

- 2026-05-08: Updated the hybrid bridge from fused QNN vision+projector to
  **QNN vision tower only + llama.cpp/OpenCL mmproj + llama.cpp/OpenCL decoder**.
  The direct runner path is
  `run_android_hybrid_bridge.py --processor hybrid --vision <vision_tower_preproj_qnn.pte>`.
  The working artifact is
  `my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte`.
  It emits pre-projector features shaped `1 x 256 x 4096`; `hybrid_decode`
  logs `external embedding is pre-projector: tokens=256 feature_embd=4096 projected_embd=896`
  and applies the InternVL `mmproj` from the GGUF on OpenCL before image prefill.
- 2026-05-08: Important export pitfall: do **not** pass
  `--encoder-weights my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth`
  to `export_pre_projector_qnn.py`. That file contains decoder-style keys
  (`layers.*`, `tok_embeddings.*`) and no `vision_tower.*` keys. Earlier bad
  artifacts silently loaded no vision weights, produced all-zero `.svlmemb`
  (`mean/std/min/max = 0`), and hallucinated. `pre_projector.py` now fails fast
  if an explicit `--encoder-weights` file contains no vision-tower keys.
- 2026-05-08: Real-weight 16a8w QNN vision-only run succeeded on Android with
  result directory
  `my_research/foundation_llamacpp/results/log_realweights_quant_debug/InternVL3-1B-Instruct-Q8_0_hybrid_ctx_4096_kv16/`.
  `vision_embedding.svlmemb` stats were non-zero (`mean -0.1406`, `std 1.0058`,
  `norm 1039.9`) and the generated caption correctly described two cats on a
  pink blanket with remote controls. Use both `--force-push` and `--model-push`
  when swapping PTE artifacts because each artifact pushes to the same remote
  filename `vision_tower_preproj_qnn.pte`.
- 2026-05-08: Refactored the llama.cpp InternVL projector code so
  `clip_graph_internvl_projector_only` inherits from `clip_graph_internvl` and
  calls the same `build_projector()` helper used by the normal full OpenCL
  InternVL path. This keeps the external pre-projector bridge mathematically
  aligned with the existing OpenCL vision path.

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
- 2026-05-04: Reworked `docs/README.md` as the quick run guide for
  `foundation_llamacpp`. It now documents model/input paths, matched comparison
  parameters, result directory layout, CPU/OpenCL/Hybrid run commands, optional
  Vulkan/Hexagon command templates, phase names, latest matched OpenCL vs hybrid
  numbers, memory interpretation, and points readers to
  `docs/archive/executorch_vision_llamacpp_decoder.md` for build and
  implementation details.
- 2026-05-04: Archived older long-form notes under `docs/archive/`, leaving only
  `docs/README.md` and this development log in the `docs/` root. Updated the
  README document index and build-detail references to use the new archive paths.
- 2026-05-04: Unified Android CPU/OpenCL/Hybrid execution into
  `run_android_hybrid_bridge.py`. The script now exposes `--processor
  cpu|gpu|hybrid` and shares one flow for argument parsing, runtime file push,
  model caching, remote script execution, artifact pull, summary CSV generation,
  `foundation_proc.csv`, `memory_timeline_plot.png`, and
  `phase_duration_stacked_bar.png`. Processor-specific logic is limited to the
  command/artifact details: CPU uses `llama-mtmd-cli`, GPU uses
  `opencl_phase_mtmd` when present, and hybrid uses `hybrid_vision_dump` plus
  `hybrid_decode`. Removed the old `run_android_llamacpp.py` entry point so new
  CPU/GPU/Hybrid experiments use a single runner file.
- 2026-05-04: Added model push caching to the unified runner. Large model-like
  files (`--model`, `--mmproj`, and hybrid `vision_encoder_qnn.pte`) are pushed
  only when missing under `--remote-root`; `--model-push` / `--model_push`
  forces re-push. Runtime binaries, shared libraries, scripts, and input images
  are still pushed every run. The runner also keeps avoiding local
  `libOpenCL.so` by default unless `--push-opencl-loader` is set.
- 2026-05-04: Updated result naming to include the processor suffix directly
  under the results root. Initial names were
  `InternVL3-1B-Instruct-Q8_0_cpu`, `InternVL3-1B-Instruct-Q8_0_opencl`, and
  `InternVL3-1B-Instruct-Q8_0_hybrid`; later naming also appends
  `_ctx_<ctx_size>`. Verified 1B smoke tests for all three processors on Android
  with `--ctx-size 32768 --batch-size 2048 --ubatch-size 512`: CPU, GPU/OpenCL,
  and hybrid returned exit code 0 and generated the expected summary, memory
  timeline, and phase duration artifacts.
- 2026-05-04: Ran full unified-runner backend validation on connected Android
  device `R3KYC01FW1P` using InternVL3-1B Q8_0, sample cats image,
  `--n-predict 32 --ctx-size 32768 --batch-size 2048 --ubatch-size 512`.
  First CPU run used `--force-push --model-push` against
  `/data/local/tmp/streamingvlm_unified_full`, verifying real model/mmproj push.
  GPU/OpenCL and Hybrid reruns reused the same remote root and verified
  `[push-cache] keep remote model` for model/mmproj while runtime binaries/libs
  were pushed every run. All processors returned exit code 0 and produced
  `foundation_output.txt`, `foundation_summary.csv`, `foundation_proc.csv`,
  `android_memory_timeline.csv`, `memory_timeline_plot.png`, and
  `phase_duration_stacked_bar.png`. Key timings from `foundation_summary.csv`:
  CPU image encode 4880.0 ms, prompt eval 6110.56 ms, decode 362.59 ms, total
  6684.88 ms; OpenCL image encode 709.0 ms, prompt eval 267.45 ms, decode
  374.93 ms, total 2307.73 ms; Hybrid QNN vision encode 375.0 ms, prompt eval
  276.37 ms, decode 400.99 ms, total 2184.4 ms.
- 2026-05-04: Fixed an Android memory timeline alignment bug in
  `run_android_hybrid_bridge.py`. The remote memory sampler previously computed
  `elapsed_s` as `sample_idx * sample_interval`; because each `/proc` sampling
  iteration takes non-trivial shell time on Android, the memory timeline was
  compressed (for example OpenCL phase rows reached ~18 s while memory samples
  ended around ~7.45 s), causing phase overlays to bunch at the right edge of
  `memory_timeline_plot.png`. The sampler now records elapsed time from
  `/proc/uptime`, so memory samples use real device wall time. Regenerated CPU,
  OpenCL, and Hybrid full 1B results; corrected memory ranges are CPU 0.01-7.29
  s, OpenCL 0.01-18.31 s, Hybrid 0.00-11.92 s, all exit code 0.
- 2026-05-04: Normalized CPU fallback phase plotting. CPU uses upstream
  `llama-mtmd-cli`, which does not emit precise `foundation_phase_stats.csv`, so
  the runner previously fell back to a different `Runtime Breakdown` chart based
  on load/prompt/decode summary metrics. Added synthetic standalone phase rows
  from available llama.cpp summary timers (`V_Encode` from
  `image_slice_encoded_ms`, `ImagePrefill` from `image_decoded_ms`,
  `T_Prefill` from `prompt_eval_time_ms - image_decoded_ms`, and `Decode` from
  `decode_eval_time_ms`) so CPU `phase_duration_stacked_bar.png` uses the same
  `Precise Runtime Breakdown` visual style and `foundation_proc.csv` schema as
  OpenCL/Hybrid. Treat CPU synthetic rows as approximate because they are not
  emitted by internal exclusive C++ phase timers.
- 2026-05-04: Fixed CPU memory timeline after adding synthetic CPU phase rows.
  The synthetic rows are duration-only and not aligned to real wall-clock memory
  samples, so overlaying them on `memory_timeline_plot.png` produced misleading
  phase spans/labels. Updated `run_android_hybrid_bridge.py` so synthetic CPU
  rows are used for `phase_duration_stacked_bar.png` and `foundation_proc.csv`
  only; CPU `memory_timeline_plot.png` now stays as a pure `MemAvailable`
  timeline without synthetic phase overlays.
- 2026-05-04: Fixed stale local artifact handling during result pulls. If a
  previous run produced `foundation_phase_stats.csv` but a later pure CPU run
  does not, the old local file must not be reused. `_pull_outputs()` now removes
  each expected local artifact before attempting `adb pull`, preventing stale
  phase CSVs from contaminating regenerated plots. Also confirmed that pure CPU
  results have no `ggml_opencl` log lines and no `foundation_phase_stats.csv`.
  Keep `opencl_phase_mtmd` restricted to `--processor gpu`; using the OpenCL
  overlay binary for CPU initializes the OpenCL backend and is not a clean CPU
  baseline.
- 2026-05-04: Added `memory_usage_summary.txt` generation to
  `run_android_hybrid_bridge.py`. The metric is computed from
  `android_memory_timeline.csv` as
  `MemAvailable(first sample) - min(MemAvailable)` and reported in KiB/MiB with
  the start/min sample indices and elapsed timestamps. Generated summaries for
  the current InternVL3-1B Q8_0 results: CPU 827092 KiB / 807.707 MiB, OpenCL
  2150444 KiB / 2100.043 MiB, Hybrid 2990840 KiB / 2920.742 MiB.
- 2026-05-04: Added a new benchmark sample image under
  `my_research/foundation_llamacpp/sample_images/`. Downloaded a Golden Gate
  Bridge photo from Wikimedia Commons, center-cropped it to square, resized it
  to `448 x 448`, and saved it as
  `sample_images/golden_gate_bridge_448.jpg`. Updated
  `run_android_hybrid_bridge.py` default `--image` and `docs/README.md` CPU,
  OpenCL, and Hybrid command examples to use the new `sample_images/` path.
- 2026-05-04: Added forced generation and special-token I/O transcripts to the
  unified llama.cpp runner. New `run_android_hybrid_bridge.py` argument:
  `--force-generation N` / `--force_generation N`. For GPU/OpenCL
  `opencl_phase_mtmd` and Hybrid `hybrid_decode`, this sets `n_predict=N` and
  the C++ generation loop continues through EOS/EOG instead of breaking early.
  For pure CPU, the runner passes upstream `llama-mtmd-cli --ignore-eos`; CPU
  remains a clean CPU baseline and does not use the OpenCL overlay binary.
- 2026-05-04: Added `foundation_token_io.txt` output. GPU/OpenCL and Hybrid emit
  this directly from the instrumented C++ tools via `--token-io-path`, rendering
  the mtmd tokenized prompt with special tokens and `<IMG_CONTEXT>` placeholders
  plus generated token pieces with special tokens enabled. Pure CPU uses a
  Python fallback that reconstructs the prompt with 256 image context tokens and
  extracts generated text from `foundation_output.txt`. Verified short
  `--force-generation 8` runs for GPU/OpenCL, Hybrid, and CPU. GPU/Hybrid phase
  CSVs showed exactly 8 `D` rows and `foundation_token_io.txt` was pulled.
- 2026-05-04: Updated unified result directory naming to include context length.
  `run_android_hybrid_bridge.py` now writes results as
  `<model>_<processor>_ctx_<ctx_size>`, for example
  `InternVL3-1B-Instruct-Q8_0_opencl_ctx_32768`. Renamed the current CPU,
  OpenCL, and Hybrid result folders from the old names to the new
  `_ctx_32768` names and updated `docs/README.md` expected output paths.
- 2026-05-04: Improved system-wide memory usage measurement with a pre-run
  baseline window. `run_android_hybrid_bridge.py` now samples `MemAvailable`
  for `--baseline-window` seconds before launching the backend process
  (default `5.0`), writes those samples to `android_memory_timeline.csv` with
  negative `elapsed_s` values, and computes `memory_usage_summary.txt` as
  `baseline_avg_mem_available_kb - runtime_min_mem_available_kb`. The summary
  still includes legacy start-minus-runtime-min fields for comparison. This is
  the preferred metric when the goal is "how much did system-wide available
  memory actually drop during the run?" Existing results created before this
  patch do not contain baseline samples and should be regenerated for the new
  metric.
- 2026-05-04: Updated the three active `docs/README.md` run templates (CPU,
  OpenCL GPU, Hybrid) to pass `--baseline-window 5.0` explicitly. The runner
  already defaults to 5 seconds, but keeping the argument in the copy/paste
  commands makes the baseline memory policy obvious and prevents future command
  drift.
- 2026-05-04: `run_android_hybrid_bridge.py` accepts `--cache-type-k` /
  `--cache-type-v` (upstream llama.cpp names) and forwards them to GPU
  `opencl_phase_mtmd`, Hybrid `hybrid_decode`, and CPU `llama-mtmd-cli`.
  Optional helper `_cache_type_shell_suffix` appends them to the remote hybrid
  shell script. OpenCL GPU section in `docs/README.md` documents `q8_0` for 8-bit
  KV and points to `llama_kv_cache` lines in `foundation_output.txt`.
- 2026-05-04: `run_android_hybrid_bridge.py` accepts optional `--fit on|off` and
  forwards `--fit ...` to standalone GPU/CPU (`opencl_phase_mtmd` /
  `llama-mtmd-cli`) and Hybrid `hybrid_decode` (`_fit_shell_suffix`). Default is
  to omit the flag (binary default). Use `--fit off` when `common_fit_params`
  triggers OpenCL `SET_ROWS` on KV views (abort during init).
- 2026-05-04: OpenCL GPU command block in `docs/README.md` includes `--fit off`
  after q8 KV flags, plus a sentence on using it when OpenCL init aborts during
  `common_fit_params`.
- 2026-05-04: Added `scripts/run_opencl_ctx_sweep.sh`: loops ctx sizes
  512..32768; scales `--batch-size` / `--ubatch-size` when `ctx < 2048`; collects
  failures and exits non-zero.
- 2026-05-06: `run_opencl_ctx_sweep.sh` defaults to **InternVL3-1B Q8_0** (+ mmproj
  Q8_0). Override with `MODEL=` / `MMPROJ=`; refresh truncated device GGUF with
  `MODEL_PUSH=1` (optional `FORCE_PUSH=1`, wipes `remote-root` **each** runner
  invocation — heavy on full sweeps).
- 2026-05-06: Same script supports **`PROCESSOR=gpu`** (default OpenCL),
  **`PROCESSOR=cpu`** (`llama-mtmd-cli`, default build
  `LLAMA_BUILD_CPU=$ROOT/llama.cpp/build-android-cpu-noomp`), or
  **`PROCESSOR=both`** (GPU sweep then CPU). Separate default remote dirs:
  `REMOTE_ROOT_GPU`, `REMOTE_ROOT_CPU`.
- 2026-05-04: `run_android_hybrid_bridge.py` forwards llama.cpp RoPE/YaRN flags:
  `--rope-scaling`, `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`,
  `--yarn-orig-ctx`, `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-slow`,
  `--yarn-beta-fast` to GPU/CPU binaries and Hybrid `hybrid_decode`
  (`_rope_shell_suffix`). README OpenCL section documents HF YaRN mapping example.
- 2026-05-05: `scripts/plot_opencl_ctx_memory_series.py`: default `--plot-style usage`
  (single Y: system `actual_memory_used_from_baseline_avg_mib` plus GPUOpenCL **self**
  MiB from `common_memory_breakdown_print` in each run’s `foundation_output.txt`). X axis is
  **categorical** (tick labels `512`, `1024`, … at equal spacing; not a numeric ctx scale).
  `--plot-style dual`: twin Y + KV from log; `--plot-style avail-min`: min MemAvailable.
  Plain Y ticks via `ScalarFormatter(useOffset=False)`.
- Canvas (Cursor IDE):
  `~/.cursor/projects/workspace-streamingvlm/canvases/internvl3-8b-opencl-memavailable-vs-ctx.canvas.tsx`
  — single `LineChart` matching default PNG (system memory usage only).
- 2026-05-06: **Nested `llama.cpp/` git**: cherry-picked upstream draft PR
  [#21313](https://github.com/ggml-org/llama.cpp/pull/21313) as commit `cd6a04a01`
  (`opencl: flash attention optimizations … quantized KV cache`). Adds OpenCL FA prepass
  kernels (`flash_attn_pre_f16.cl`) and `ggml_cl_flash_attn_prepare_quantized_tensor`:
  Q4_0/Q8_0 KV dequant to fp16 temp buffers before FA (warns perf may be poor).
  **Rebuild** the Android OpenCL hybrid binaries after pulling this tree; the StreamingVLM
  repo ignores `llama.cpp/` — replicate this cherry-pick locally or merge upstream once landed.
  Android Adreno may still fail (PR [#21501](https://github.com/ggml-org/llama.cpp/issues/21501)).
- 2026-05-06: `run_opencl_ctx_sweep.sh` env **`CACHE_TYPE_K`** / **`CACHE_TYPE_V`**
  (default `f16`) forwarded to `--cache-type-k` / `--cache-type-v`. Source of truth: nested
  `llama.cpp/ggml/src/ggml-opencl/` (the old `foundation_llamacpp/kv_code/` mirror was removed as unused).
- 2026-05-06: `run_android_hybrid_bridge.py` passes **`--flash-attn`** (`on`|`off`|`auto`, aliases `-fa`),
  **`--no-kv-offload`**, and optional **`--disable-attn-kv-rotation`** (remote shell exports `LLAMA_ATTN_ROT_DISABLE=1`).
  `run_opencl_ctx_sweep.sh` forwards env **`FLASH_ATTN`**, **`NO_KV_OFFLOAD=1`**, **`DISABLE_ATTN_KV_ROTATION=1`**.
  **Default `WARMUP=1`** on the sweep (passes `--warmup`): stable `V_Encode` / `image slice encoded` vs cold OpenCL compile; **`WARMUP=0`** for fastest runs.
- 2026-05-06: **OpenCL `GGML_OP_SET_ROWS` + Q8_0 KV** — upstream only allowed F16/F32 destinations in
  `ggml_opencl_supports_op`, so quantized KV cache views aborted at `sched_reserve` (`SET_ROWS` on OpenCL buffer).
  Implemented **`kernel_set_rows_q8_0_i32/i64`** in `llama.cpp/ggml/src/ggml-opencl/kernels/set_rows.cl`
  (per-block quant matching `quantize_row_q8_0_ref`) and wired **`ggml_cl_set_rows`** / kernel load /
  `supports_op`. Rebuild Android OpenCL
  (`libggml-opencl.so`, `opencl_phase_mtmd`) after sync.
  **Device check (Adreno 830):** `sched_reserve` completes for q8 KV (no `SET_ROWS` abort).
- 2026-05-06 (post-upstream-sync fix): **KV q8 + `-fa on`** — two issues: (1)
  `GGML_OP_FLASH_ATTN_EXT` in `ggml_opencl_supports_op` only matched storage F32/F16, so **quantized KV**
  routed FA to **CPU** while K/V stayed OpenCL → **SIGSEGV on empty warmup**. **Fix:** treat quant K/V as
  logical F16/F32 (same rule as `ggml_cl_flash_attn`) and allow **F16 Q + F16-effective KV + F32 dst**
  (`ggml_flash_attn_ext` always allocates F32 output). (2) **`ggml_cl_flash_attn_prepare_quantized_tensor`**
  + **non-contiguous row pack** (see earlier bullet). Combined: InternVL3 1B OpenCL, ctx 8192,
  `--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --fit off --warmup` → exit 0 (splits ~2 vs ~50
  when FA incorrectly fell off OpenCL). Perf: each FA dequantizes KV to temp GPU buffers (one-time warn in log).
- 2026-05-06: **OpenCL FA + Q8 KV gibberish output** — `ggml_cl_flash_attn_prepare_quantized_tensor`
  read the GPU byte span and called `to_float` as if it were dense. K/V tensors are often **non-contiguous**
  after `ggml_permute` before FA, so dequant fed wrong bytes → garbage attention. **Fix:** when
  `!ggml_is_contiguous_0(tensor)`, pack each logical row (`nb[1..3]`, `ggml_row_size` along `ne[0]`)
  into a dense buffer, then `to_float`. Implemented in `llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp`.
- 2026-05-06: **`run_android_hybrid_bridge.py`** — default **skips llama empty warmup** (`--no-warmup` implied). Use **`--warmup`** to re-enable. First CLIP/vision encode still runs `clip_model_loader::warmup()` once to allocate graphs (log line `warmup: flash attention is ...`); that is separate from `--no-warmup`.
- 2026-05-06 (obsolete): **`kv_code/`** was a mirror of `llama.cpp/ggml/src/ggml-opencl/` for offline diff;
  **removed 2026-05** — patches live only under nested **`llama.cpp/`** plus docs in `docs/archive/q4_8_kvcache_implementation.md`.
- 2026-05-06: **`run_android_hybrid_bridge.py` result dir names** include KV cache slugs after
  `ctx_<N>`: e.g. `..._opencl_ctx_1024_kv8` for `q8_0`/`q8_0`, `..._kv4` for `q4_0`, `..._kv16` for `f16`/`f16`
  when K/V default to f16 (including when `--cache-type-*` omitted). Asymmetric K/V → `..._kv8_4`.
  `plot_opencl_ctx_memory_series.py` `ctx_from_dir` matches `_ctx_(\\d+)` before optional `_kv…`.
- 2026-05-06: **`scripts/run_opencl_ctx_sweep.sh`** always passes `--results-root` =
  `my_research/foundation_llamacpp/results/log` — run folders (`<model>_opencl_ctx_<N>_kv…`) live **directly**
  under that path; avoid introducing another dated parent directory under `results/`.
- 2026-05-06: **OpenCL `GGML_OP_SET_ROWS` + Q4_0 KV** — same wiring as Q8_0 (`supports_op`,
  `kernel_set_rows_q4_0_i32/i64`, `quantize_block_q4_0` in `set_rows.cl`).
  First device run produced **junk decode** (`char` saturate → wrong nibble saturation vs
  reference `quantize_row_q4_0_ref`: `(int8_t)(x + 8.5f)` then `(uint8_t)MIN(15, …)`).
  **Fix:** `kernel_set_rows_i32_as_int8_truncate` + `kernel_set_rows_q4_packed_nibble_ref`:
  truncate `x+8.5` to signed int8 semantics, clamp to `[0,15]`, write nibbles like C `(uint8_t)`.
  **Retest:** `CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0` OpenCL sweep (`run_opencl_ctx_sweep.sh`, `CTX_SIZES_OVERRIDE=1024`) can **exit 0** once `SET_ROWS`/graph issues are resolved — that only means **no crash / sched OK**, not caption quality.
  **User re-check (correcting earlier log):** **`InternVL3-1B-Instruct-Q8_0` weights + `_kv4`** and more generally **small VLM + `_kv4`** are **not** a safe “green” baseline — **Q4_K weights + `q4_0` KV** can still yield **bad decode** on device; treat **`_kv8` / `_kv16`** as the quality baselines until q4 KV is re-validated end-to-end per model.
- 2026-05-06: **`llama.cpp` upstream sync (subtree):** added remote `upstream-llama` →
  `https://github.com/ggml-org/llama.cpp.git`, committed removal of old vendored tree then
  `git subtree add --prefix=llama.cpp upstream-llama master --squash` (merge commit +
  squashed upstream). Local GGUF tree preserved by moving `llama.cpp/models` aside before
  `rm -rf llama.cpp/` and restoring after add. **Overlay re-applied after sync:** OpenCL
  `kernels/set_rows.cl` Q8_0/Q4_0 block helpers + `kernel_set_rows_{q8_0,q4_0}_{i32,i64}`;
  `ggml-opencl.cpp` `GGML_OP_SET_ROWS` in `ggml_opencl_supports_op`, `clCreateKernel`, and
  `ggml_cl_set_rows` branches. **Also port after slim upstream `ggml_cl_flash_attn`:** quantized-KV FA needs
  **`ggml_cl_flash_attn_prepare_quantized_tensor`** (+ row pack) and **`ggml_opencl_supports_op` FLASH_ATTN_EXT**
  extended for quant K/V so sched keeps FA on OpenCL (avoid CPU/GPU buffer mix segfault). Optional larger draft-PR
  extras: kv_pad / split / `flash_attn_pre_f16.cl` for perf, not required for basic correctness.
- **2026-05-06 (device sweep, Adreno / OpenCL):** Reported for **InternVL3-8B-Instruct-Q4_K_M** weights:
  **`CACHE_TYPE_K/V=q4_0` (`_kv4`)** decode broken vs **`q8_0` (`_kv8`)** / **`f16` (`_kv16`)** on the same prompt
  (historical note; see also OpenCL `prepare` fixes below). **Correction:** **`InternVL3-1B` + `_kv4`** is **not** reliably good either
  (**Q4_K + `q4_0` KV** can fail on 1B too); do not use 1B as proof that kv4 is “fine.”
- **2026-05-06:** **`ggml_cl_flash_attn_prepare_quantized_tensor`** (nested `llama.cpp/.../ggml-opencl.cpp`) —
  always build a **dense row-major slab** (`memcpy` each row using `tensor->nb[1..3]`) before
  `to_float` / `dequantize_row_*`. The old `if (!ggml_is_contiguous_0)` fast path passed **`host_raw`**
  straight through when “contiguous”; after permutes, **`nb[1]` may still exceed `ggml_row_size`**, so
  **`dequantize_row_q4_0`** (walks consecutive `block_q4_0`) read wrong bytes; Q8 path was less visibly
  wrong. **`clFinish(queue)`** before `clEnqueueReadBuffer` when only one OpenCL device (cross-backend
  `sync` is a no-op). Rebuild Android **`libggml-opencl.so`** and re-test **`_kv4`**.
- **2026-05-06:** **`GGML_OPENCL_DEBUG_KV`** — set `GGML_OPENCL_DEBUG_KV=1` on the host when calling
  `run_opencl_ctx_sweep.sh` / `run_android_hybrid_bridge.py`; the wrapper exports it on device.
  **`ggml-opencl.cpp`** then emits **throttled `GGML_LOG_WARN`** lines: (1)
  **`ggml_cl_flash_attn_prepare_quantized_tensor`** first 64 calls — packed-quant CRC, F32 sample min/max, NaNs,
  tensor name / `view_offs`; (2) **`FLASH_ATTN_EXT`** `supports_op` reject reasons (dims, dtypes); (3) first 96
  **`mul_mat` with `src0=Q4_0`** — name, `llama_kv_aos`, shapes. Standalone GPU runs log into
  **`foundation_output.txt`** (stdout+stderr merged). Hybrid runs: **`hybrid_decode_stdout.txt`** before finalize.
- **2026-05-06 (Android CPU KV A/B — InternVL3-8B, flash-attn):** Built **`build-android-cpu-noomp`**, ran `run_android_hybrid_bridge.py --processor cpu` → **`q8_0` KV** yields a normal Golden Gate caption; **`q4_0` KV** yields broken continuation (`The … 111…`). **`q4` KV + FA** degradation is **not OpenCL-only**. (Does **not** imply 1B is immune — see SET_ROWS / sweep correction above.)

  `--flash-attn on` + `--ctx-size 512` via `run_android_hybrid_bridge.py` (stub local GGUF paths, remote cache keeps
  real weights). **`foundation_output.txt`:** FA prepare lines for every layer show names like **`cache_*_lN (view) (permuted)`**,
  **nans=0**, plausible F32 ranges; **`FLASH_ATTN_EXT OpenCL unsupported`** absent (dk=dv=128 path accepted);
  **`mul_mat … Q4_0`** throttle lines absent in this workload. Decoder text still degraded vs F16 KV — next focus:
  OpenCL FA kernel vs temp strides / mask / `n_kv`, not KV empty names or naive prepare NaNs alone.

- **2026-05-06 (hybrid_decode):** **`mparams.warmup` is forced off** for mtmd/CLIP init. Hybrid uses
  `--external-embedding` from QNN; OpenCL must not run a dummy **ViT warmup** in `clip_init`. Text-side
  warmup remains controlled by **`common_params.warmup`** (`--warmup` / `--no-warmup` → `common_init_from_params`).
- **2026-05-08 (hybrid manifest/model size):** Old fused QNN `vision_encoder_pte` artifacts include
  `multi_modal_projector` and are model-size-specific. Pair a fused QNN manifest with the same-size
  GGUF/mmproj in llama.cpp. A 1B fused QNN vision PTE emits `256 x 896`, while an 8B decoder expects
  `256 x 3584`; mixing them triggers `embedding size mismatch`. The newer direct `--vision` path can
  instead use InternVL pre-projector PTEs (`1 x 256 x 4096`) because `hybrid_decode` now applies the
  GGUF mmproj before image prefill.

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
