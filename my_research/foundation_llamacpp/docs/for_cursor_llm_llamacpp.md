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
