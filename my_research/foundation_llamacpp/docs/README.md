# Foundation llama.cpp Runtime Guide

This directory contains the Android runner and bridge binaries used for
llama.cpp / ExecuTorch hybrid VLM experiments. This document is the quick run
guide. Build internals and historical notes live in
`archive/executorch_vision_llamacpp_decoder.md`. Dynamic KV implementation
details live in `archive/dynamic_kv_cache_implementation.md`, and the OpenCL
buffer/memory-architecture explanation lives in
`archive/dynamic_kv_opencl_buffer_memory_architecture.md`. Legacy
single-buffer details live in `archive/streaming_single_buffer_implementation.md`;
the current on-demand/sliding-window/KV vision-prefill details live in
`archive/streaming_sliding_window_and_vision_prefill.md`, and partial
vision-prefill KV preemption details live in
`archive/partial_vision_prefill_kv.md`. Future video-compression KV/RoPE
repositioning notes live in
`archive/kv_rope_reposition_for_video_compression.md`.

## Current Baseline

As of 2026-05-15, the hybrid media baseline is unified around
`hybrid_streaming_decode`:

```text
image          -> hybrid_streaming_decode --media-mode image
multi-image    -> hybrid_streaming_decode --media-mode multi-image
offline video  -> hybrid_streaming_decode --media-mode video
streaming      -> hybrid_streaming_decode --media-mode streaming
```

The old `hybrid_vision_dump + hybrid_decode` split remains in source for
comparison and older notes, but the default hybrid runner path now executes the
single unified binary for image, multi-image, video, and streaming runs.

User-facing CLI cleanup:

```text
--stream-mode on-demand
  Canonical name for the old single-buffer latest-frame streaming baseline.
  `--stream-mode single-buffer` and `--single-buffer` are accepted aliases.

--multi-image img1 img2 ...
  Canonical multi-image argument. `--images` remains a compatibility alias.

--online-buffer
  Streaming-only latest-buffer mode. Input frames continue to arrive at the
  configured sampling FPS. If processing is delayed, prompt/cache work selects
  frames from the latest buffer at processing start, and stale pending
  vision-prefill cache updates are coalesced.

--latest-frame-only
  Streaming vision-prefill cache-update policy. If a frame arrives while the
  worker is already handling an older cache update or prompt, the new frame's
  cache update is dropped. The next cache update starts only from a frame that
  arrives after the worker becomes idle.

--partial-vision-kv
  Hybrid vision-prefill preemption mode. If a prompt arrives while an image
  prefill cache update is running, the current image micro-batch is finished,
  the committed partial image KV is kept, and the answer starts without waiting
  for the rest of that frame image prefill. The commit granularity is
  controlled by `--ubatch-size`.

--kv-reposition-keep-latest-frames N
  Experimental hybrid vision-prefill compression hook. After each committed
  cache update, keep only the latest N frame vision-KV spans, remove older
  frame vision KV with `llama_memory_seq_rm`, and shift later cached tokens
  forward with `llama_memory_seq_add`. This is a KV/RoPE reposition experiment
  for future video compression; it preserves text/chat KV and compacts vision
  spans only.
```

## Common Setup

Run all commands from the workspace root:

```bash
cd /workspace/streamingvlm
```

The unified runner is:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py
```

Common arguments used by most runs:

```text
--processor cpu|gpu|hybrid
  cpu: llama.cpp CPU path.
  gpu: llama.cpp OpenCL path. Result folder uses `opencl`.
  hybrid: ExecuTorch QNN vision encoder + llama.cpp OpenCL decoder.

--llama-build-dir PATH
  Android llama.cpp build directory containing runtime binaries/libs.

--model PATH
  Text GGUF.

--mmproj PATH
  Multimodal projector GGUF. Required for image/video/hybrid runs. Omit for
  cpu/gpu text-only runs.

--prompt TEXT
  User prompt. In streaming mode this is a JSON list of prompts.

--n-predict N
  Maximum generated tokens.

--force-generation N
  Optional. Force exactly N generated tokens in instrumented OpenCL/hybrid paths.

--ctx-size N
  llama.cpp context length. In fixed-KV mode this also sets KV-cache allocation.

--dynamic-kv-cache --kv-init-size 1024 --kv-grow-step 1024
  Project-local prototype for the standard llama.cpp KV cache path. Logical
  context uses the model max context (`--ctx-size 0` internally), while physical
  KV starts at `kv-init-size` cells and grows by `kv-grow-step` cells on demand.
  On OpenCL KV buffers, grow migration uses `clEnqueueCopyBuffer` to copy K/V
  data directly device-to-device; it falls back to host tensor get/set only if
  the tensors are not OpenCL-backed. First validation is scoped to the
  OpenCL/hybrid streaming path and non-SWA single-sequence models. Paged KV is
  not active on `main`; the paged-KV prototype commits were reverted so this
  path stays a contiguous dynamic KV grow experiment.

--batch-size N / --ubatch-size N
  llama.cpp batch and micro-batch sizes.

--threads N
  CPU threads. Mostly relevant to CPU and host-side llama.cpp setup.

--gpu-layers N
  Number of layers offloaded to GPU/OpenCL. Use 99 for full offload where possible.

--device GPUOpenCL
  OpenCL device name used by the GPU/hybrid decoder path.

--cache-type-k TYPE / --cache-type-v TYPE
  KV-cache dtype, e.g. `f16` or `q8_0`.

--fit off
  Disables llama.cpp OpenCL automatic memory fitting. Useful when OpenCL init
  aborts in `common_fit_params`.

--baseline-window 5.0
  Samples Android MemAvailable for 5 seconds before execution.

--remote-root /data/local/tmp/streamingvlm_unified
  Android work directory.

--results-root my_research/foundation_llamacpp/results/log
  Host result directory root.

--force-push
  Recreate remote work directory contents.

--model-push
  Force re-push model-like files (`--model`, `--mmproj`, and hybrid `.pte`).
```

Result directory format:

```text
<GGUF_stem>_<processor>_ctx_<N>_<media>_kv<KV>
```

Examples:

```text
InternVL3-2B-Instruct-Q8_0_cpu_ctx_4096_text_kv16
InternVL3-2B-Instruct-Q8_0_opencl_ctx_4096_image_kv16
InternVL3-2B-Instruct-Q8_0_opencl_ctx_4096_multi_image_kv16
InternVL3-2B-Instruct-Q8_0_opencl_ctx_4096_video_kv16
InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_on_demand_kv16
InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_sliding_window_kv16
InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_vision_prefill_kv16
InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_vision_prefill_kv16_dynamic_online
InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_vision_prefill_kv16_dynamic_partialkv
InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_vision_prefill_kv16_dynamic_online_latest_frame_only
```

Each run directory is normalized into three report folders after finalization:

```text
csv/
  CSV tables such as foundation_proc.csv, foundation_summary.csv,
  android_memory_timeline.csv, stream_events.csv, and phase stats.

png/
  Plots such as memory_timeline_plot.png, phase_duration_stacked_bar.png,
  phase_timeline.png, and dynamic_kv_grow_breakdown_stacked_bar.png.

txt_json/
  Text and JSON logs such as foundation_output.txt, token traces, exit codes,
  host_adb_output.txt, media_manifest.json, and raw handoff/debug artifacts.
```

Important output files:

```text
txt_json/foundation_output.txt
  Model output transcript.

txt_json/foundation_token_io.txt
  Compact token input/output transcript.

txt_json/foundation_inference_tokens.txt
  Raw token trace. Streaming mode appends every prompt turn.

txt_json/stream_inference_tokens_<idx>.txt
  Per-turn raw token traces for streaming mode.

txt_json/run_command.txt
  Exact host-side command used for this run. This is written for image,
  multi-image, video, and streaming runs before artifacts are grouped.

csv/foundation_proc.csv
  Canonical phase timing CSV. Dynamic KV runs add `DynamicKVGrow` rows when
  physical KV capacity increases. For these rows, `kv_pos` is the old cell
  count, `kv_total` is the new cell count, `kv_estimated_used_kb` is the old
  KV MiB converted to KiB, and `kv_physical_committed_kb` is the new committed
  KV size. New runs record the full grow/retry window, including scheduler
  reserve, so retry-side `ImagePrefill` timing starts after `DynamicKVGrow`.
  Grow breakdown builds also add `DynamicKVGrowAlloc`,
  `DynamicKVGrowMetadata`, `DynamicKVGrowCopy`, and
  `DynamicKVGrowSchedulerReserve` rows. These are sub-spans inside the aggregate
  `DynamicKVGrow` window.

csv/streaming_phase_stats.csv / csv/stream_events.csv
  Streaming-only timing and event logs.

txt_json/stream_buffer_summary.txt
  Streaming-only buffer report. It records requested/observed input FPS,
  processed visual job FPS, skipped cache updates, and average prompt-frame
  lag. `--online-buffer` uses this file to show the split between input frame
  cadence and delayed processing cadence.

png/phase_timeline.png
  Common phase timeline plot. Image, multi-image, and offline-video runs use
  ready-relative time after bridge load/warmup; streaming runs use stream/video
  time from the first frame event.

txt_json/memory_usage_summary.txt / png/memory_timeline_plot.png
  Android system memory summary and plot.

png/memory_timeline_decode_window.png
  Zoomed memory plot from first `V_Encode` start to final decode end. Dynamic
  KV runs mark `DynamicKVGrow` with the cell and MiB growth detail.

png/dynamic_kv_grow_breakdown_stacked_bar.png
  Grow breakdown builds only. Separate stacked bar chart for the alloc,
  metadata, copy, and scheduler-reserve sub-spans inside each `DynamicKVGrow`
  window.

png/phase_duration_stacked_bar.png
  Common runtime duration stack. It uses the same visible phase labels as
  `phase_timeline.png`: `V_Encode`, `Mmproj`, `ImagePrefill`, `T_Prefill`,
  `KVRepositionCompact`, `DynamicKVGrow`, and `Decode`. Dynamic KV breakdown
  sub-rows are excluded from this aggregate plot and remain in
  `dynamic_kv_grow_breakdown_stacked_bar.png`.
```

Artifact layout smoke:

```bash
my_research/foundation_llamacpp/scripts/run_artifact_layout_1b_q8.sh
```

This runs the 1B Q8 hybrid baseline matrix: single image, multi-image, offline
video, streaming on-demand, streaming sliding-window, streaming vision-prefill,
streaming vision-prefill with dynamic KV growth, and streaming vision-prefill
with dynamic KV growth plus `--online-buffer`. Dynamic runs use
`--kv-init-size 512` and `--kv-grow-step 512`. The script deliberately does not
force-push models; the remote root must already contain the 1B Q8 GGUF, mmproj,
and vision PTE files.

## Single Text Input

Text-only mode is supported by `cpu` and `gpu`. It omits `--image`, `--video`,
`--streaming-video`, and `--mmproj`, and uses upstream `llama-completion`.

Hybrid text-only is not supported because `--processor hybrid` is defined as a
QNN vision + llama.cpp decoder path and requires visual input plus `--vision`.

### CPU

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor cpu \
  --llama-build-dir llama.cpp/build-android-cpu-noomp \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --prompt "Explain what a mobile visual assistant is in one sentence." \
  --n-predict 64 \
  --threads 4 \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

### GPU

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor gpu \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --prompt "Explain what a mobile visual assistant is in one sentence." \
  --n-predict 64 \
  --threads 4 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

### Hybrid

Not supported for text-only input. Use `cpu` or `gpu`.

### Text-Only Extra Arguments

```text
Required:
  --processor cpu|gpu
  --llama-build-dir PATH
  --model PATH
  --prompt TEXT

Do not pass:
  --mmproj
  --image
  --video
  --streaming-video
  --vision

Useful:
  --n-predict / --force-generation
  --ctx-size
  --cache-type-k / --cache-type-v
  --gpu-layers, --device, --fit off for GPU
```

## Image Input

Image mode is supported by `cpu`, `gpu`, and `hybrid`.

For InternVL experiments, keep benchmark images at `448 x 448` when comparing
backends. The runner also writes a `media_manifest.json` describing normalized
QNN inputs, layout images, and prompt metadata.

### CPU

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor cpu \
  --llama-build-dir llama.cpp/build-android-cpu-noomp \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
  --prompt "Describe this image briefly." \
  --n-predict 64 \
  --threads 4 \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

### GPU

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor gpu \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
  --prompt "Describe this image briefly." \
  --n-predict 64 \
  --threads 4 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

### Hybrid

Set these environment variables before hybrid runs:

```bash
export QNN_SDK_ROOT=/workspace/streamingvlm/executorch/backends/qualcomm/sdk/qnn/qairt/2.37.0.250724
export EXECUTORCH_ROOT=/workspace/streamingvlm/executorch
```

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
  --prompt "Describe this image briefly." \
  --n-predict 64 \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log \
  --force-push
```

### Image Extra Arguments

```text
Required:
  --image PATH
  --mmproj PATH

Hybrid required:
  --vision PATH or --manifest PATH
  QNN_SDK_ROOT in environment
  --soc-model SM8750|SM8650|...

Useful:
  --warmup-image PATH
    Fixed image used for bridge-local vision/mmproj warmup.

  --model-push
    Use when changing model/mmproj/PTE files that keep the same remote filename.

  --force-push
    Recreate remote workdir and avoid stale inputs.
```

## Multiple Image Input

Use `--multi-image` for InternVL-style multi-image prompts. The runner normalizes
each image as one 448 x 448 input, builds this prompt prefix, and writes the
same `csv/`, `png/`, `txt_json/` artifact folders as single-image runs:

```text
Image-1: <img>...</img>
Image-2: <img>...</img>
question text
```

Example:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor gpu \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --multi-image \
    my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
    my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
  --prompt "Compare these two images briefly." \
  --n-predict 64 \
  --threads 4 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

## Video Input

Offline video mode samples a fixed number of frames before running inference.
This is not streaming. The prompt sees all sampled frames at once:

```text
Frame1: <img>...</img>
Frame2: <img>...</img>
...
question text
```

Video mode is supported by `cpu`, `gpu`, and `hybrid`.

### CPU

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor cpu \
  --llama-build-dir llama.cpp/build-android-cpu-noomp \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --num-segments 8 \
  --max-num 1 \
  --prompt "Describe this video briefly." \
  --n-predict 64 \
  --threads 4 \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

### GPU

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor gpu \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --num-segments 8 \
  --max-num 1 \
  --prompt "Describe this video briefly." \
  --n-predict 64 \
  --threads 4 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

### Hybrid

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --num-segments 8 \
  --max-num 1 \
  --prompt "Describe this video briefly." \
  --n-predict 64 \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log \
  --force-push
```

### Video Extra Arguments

```text
--video PATH
  Input video file.

--num-segments N
  Number of uniformly sampled frames. Example: `8` means the model sees 8 frame
  groups at once.

--max-num N
  Max InternVL dynamic-preprocess tiles per sampled frame. Use `1` for one
  448x448 tile per frame and easier backend comparisons.

media_manifest.json
  Check this file for sampled frame indices, tile counts, and generated prompt
  layout.
```

## Streaming Video Input

Streaming video mode is supported by `--processor gpu` and `--processor hybrid`.

The host samples the video first and pushes frame files plus `media_manifest.json`
to Android. The device runner then replays those frames according to their
timestamps. `--stream-mode on-demand` keeps only the latest sampled frame.
`--stream-mode single-buffer` and `--single-buffer` are aliases for this legacy
baseline. `--stream-mode sliding-window` turns the recent sampled frames into
one video clip at prompt arrival while preserving multi-turn chat/KV state.
`--stream-mode vision-prefill` is the
hybrid KV-cache observation mode: every sampled frame saves a full-history
video-prefix KV snapshot. Frame 0 is built from scratch; later frames restore
the previous snapshot and append only the new frame's label/image KV before
saving the next snapshot. Prompt handling restores the matching snapshot before
evaluating only the text question suffix.

By default this is a deterministic file-backed streaming simulator where prompt
jobs capture their selected frame/window at request timestamp. Add
`--online-buffer` to model a real latest-buffer camera path: frame input cadence
and processing cadence are decoupled, stale pending cache updates are dropped,
and delayed prompt/cache work uses the latest frame/window at processing start.

Add `--partial-vision-kv` with hybrid `--stream-mode vision-prefill` when TTFT
has priority over finishing the current frame image prefill. If the prompt
arrives while a frame image prefill is running, the bridge finishes only the
current `--ubatch-size` image micro-batch, closes the image wrapper, and answers
from the committed partial KV. For the current InternVL3 one-tile setup, each
frame has 256 vision tokens, so `--ubatch-size 64` can answer from 64, 128, 192,
or 256 committed vision KV tokens.

Add `--kv-reposition-keep-latest-frames N` to compact old resident frame vision
KV after cache updates. The bridge records each committed frame image KV span,
removes spans older than the latest N frames, shifts later cached tokens forward
with llama.cpp's sequence-position API, and saves the compacted snapshot.
`streaming_phase_stats.csv` records `KVRepositionCompact`,
`foundation_inference_tokens.txt` records `KV_REPOSITION_COMPACT`, and
`stream_buffer_summary.txt` reports compaction counts and removed vision-token
totals.

### CPU

Not supported for `--streaming-video`.

### GPU

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor gpu \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --stream-mode on-demand \
  --sampling-fps 1.0 \
  --max_video_time 15 \
  --time '[5.0, 8.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???"]' \
  --max-num 1 \
  --n-predict 64 \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log \
  --force-push
```

### Hybrid

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-8B-Instruct-GGUF/InternVL3-8B-Instruct-Q4_K_M.gguf \
  --mmproj llama.cpp/models/InternVL3-8B-Instruct-GGUF/mmproj-InternVL3-8B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --stream-mode on-demand \
  --sampling-fps 1.0 \
  --max_video_time 60 \
  --time '[5.0, 8.0, 11.0, 14.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???", "What changed in the scene?", "Summarize the full situation so far."]' \
  --max-num 1 \
  --n-predict 64 \
  --ctx-size 4096 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log \
  --force-push
```

### Hybrid With Dynamic KV Cache

Normal dynamic grow run (`1024 -> 2048 -> ...` cells):

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-8B-Instruct-GGUF/InternVL3-8B-Instruct-Q4_K_M.gguf \
  --mmproj llama.cpp/models/InternVL3-8B-Instruct-GGUF/mmproj-InternVL3-8B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --stream-mode on-demand \
  --sampling-fps 1.0 \
  --max_video_time 60 \
  --time '[5.0, 8.0, 11.0, 14.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???", "What changed in the scene?", "Summarize the full situation so far."]' \
  --max-num 1 \
  --n-predict 64 \
  --dynamic-kv-cache \
  --kv-init-size 1024 \
  --kv-grow-step 1024 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log \
  --force-push
```

Large one-shot grow test (`1024 -> 16384` cells) to make the memory increase
easy to see:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --stream-mode on-demand \
  --sampling-fps 1.0 \
  --max_video_time 15 \
  --time '[5.0, 8.0, 11.0, 14.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???", "What changed in the scene?", "Summarize the full situation so far."]' \
  --max-num 1 \
  --n-predict 64 \
  --dynamic-kv-cache \
  --kv-init-size 1024 \
  --kv-grow-step 15360 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log/dynamic_grow_16384 \
  --force-push
```

The second run should report a `DynamicKVGrow` row like
`1024->16384/32768 cells; 28.00->448.00 MiB` in `foundation_proc.csv`, and the
same marker should appear in `phase_timeline.png` and
`memory_timeline_decode_window.png`. Current OpenCL builds should also log
`reset_capacity: dynamic KV data migration used device-to-device copy`; the
validated 2B Q8 hybrid run completed the `1024 -> 16384` grow in about
`202 ms` inside `llama_kv_cache::grow_to()`.

### Streaming Extra Arguments

```text
--streaming-video PATH
  Input video to sample and replay.

--single-buffer
  Backward-compatible alias for `--stream-mode on-demand`.

--stream-mode on-demand
  Existing latest-frame streaming baseline. Keep only the latest sampled frame as the
  current image buffer. Earlier frames are overwritten, not queued for later
  processing. This mode keeps llama.cpp chat/KV state across prompt events.

--stream-mode single-buffer
  Backward-compatible alias for `--stream-mode on-demand`.

--stream-mode sliding-window
  Sliding video-window baseline. Each prompt selects recent sampled frames,
  formats them as `Frame1: <__media__>` / `Frame2: <__media__>` / ... plus
  the question, then runs full vision encode, mmproj, image prefill, text
  prefill, and decode. It preserves llama.cpp chat/KV state across prompt
  events, so previous user/assistant turns remain visible. It does not reuse a
  cached image-prefix KV snapshot across prompts.

--stream-mode vision-prefill
  KV-level cached vision-prefill mode for hybrid streaming. As frames arrive,
  the bridge keeps an incremental KV snapshot for the currently open streaming
  user turn. The first frame is built from scratch. For each later frame, the
  bridge restores the previous cache, evaluates only the new global `FrameN:`
  label, QNN vision encode, mmproj, and ImagePrefill for that frame/tile, then
  saves the resulting llama sequence KV state. At prompt time it restores the
  latest matching KV snapshot, pre-fills only the text question suffix, decodes
  the answer, and saves the post-answer chat/KV state. Later frames then start
  the next user turn, so previous user/assistant turns remain visible.

--partial-vision-kv
  Hybrid vision-prefill only. This changes prompt preemption inside cache
  update work, not the outer streaming mode. When a prompt arrives during an
  image-prefill cache update, the current image micro-batch is allowed to
  finish and the remaining batches for that stale frame are skipped. The
  question then uses all committed KV up to that partial point. The partial
  size is `--ubatch-size`; with InternVL3 one-tile frames, `--ubatch-size 64`
  gives four possible 64-token image KV commit points inside the 256-token
  visual placeholder.

--chunked-vision-prefill
  Reserved future mode flag for independently reusable vision-prefill chunks.
  The planned `--chunk-count` argument will control how many frames are grouped
  into each cached KV chunk.

--window-sec SEC
  Optional prompt-time lookback window for sliding-window.
  If omitted, all sampled frames up to the prompt timestamp are eligible before
  `--window-max-frames` is applied. `vision-prefill` ignores this option and
  caches the full sampled frame history up to each frame.

--window-max-frames N
  Maximum frames used by one sliding-window prompt. If more frames are eligible,
  an even temporal subset is selected. `vision-prefill` ignores this option and
  caches the full sampled frame history up to each frame.

--sampling-fps FPS
  Sampling/replay rate. `1.0` means one buffer update per second. `0.5` means
  one update every two seconds.

--max-video-time SEC / --max_video_time SEC
  Optional cap on sampled video duration.

--time '[...]'
  JSON list of prompt arrival timestamps in seconds.

--prompt '["...", "..."]'
  JSON list of prompts. Must have the same length as `--time`.

--max-num N
  Max InternVL dynamic-preprocess tiles per sampled video frame. Single-buffer
  still uses one full-frame image; sliding-window and vision-prefill use the
  normal tiled video-frame path.

--dynamic-kv-cache --kv-init-size 1024 --kv-grow-step 1024
  Optional project prototype. The decoder advertises model-max logical context
  while initially allocating only 1024 KV cells, then grows by 1024 cells when
  the used KV no longer fits. Growth reallocates KV buffers and copies existing
  K/V data into the larger allocation. On OpenCL-backed tensors this copy is a
  device-to-device `clEnqueueCopyBuffer`; host tensor get/set is only the
  fallback path. Growth can still introduce a latency spike; it reduces
  reserved KV memory, not attention work. Paged KV is intentionally not part of
  the active `main` implementation.
  New builds write grow timestamps with the same `ggml_time_ms()` clock used by
  streaming phase timers, so retry-side prefill rows can be separated from grow
  time in `foundation_proc.csv` and `phase_timeline.png`. On the
  `codex/dynamic-kv-grow-breakdown` branch, stdout also records alloc,
  metadata, copy, and scheduler-reserve sub-spans. Those sub-spans stay out of
  the main streaming timeline and are visualized in
  `dynamic_kv_grow_breakdown_stacked_bar.png`.

--online-buffer
  Latest-buffer execution semantics for streaming. Frame input still follows
  `--sampling-fps`, but delayed work reads the newest buffered frame/window at
  processing start. In vision-prefill mode, pending stale cache-update jobs are
  coalesced so the worker spends time on the newest cache state instead of
  replaying every queued intermediate snapshot.

--latest-frame-only
  Streaming vision-prefill only. Frame cache updates arriving while the worker
  is busy are dropped instead of queued. This differs from prompt preemption:
  prompts may partially commit the current image KV before answering, but stale
  frame cache updates are simply ignored. `stream_events.csv` records
  `LatestFrameOnlyCacheDrop`, and `stream_buffer_summary.txt` records
  `latest_frame_only=true` plus `latest_frame_only_dropped_cache_updates=N`.

stream_events.csv
  Frame arrival, `OnDemandBufferUpdate`, prompt arrival, and prompt decode events.
  For multi-frame modes, prompt/decode rows use the last selected frame index.

streaming_phase_stats.csv / foundation_proc.csv
  Phase timing rows, including `V_Encode`, `Mmproj`, prefill, and decode.
  GPU streaming uses llama.cpp OpenCL vision encode. Hybrid streaming uses QNN
  vision encode and OpenCL mmproj/decode. In `vision-prefill`, cache work is
  reported with rows such as `VisionPrefillCacheBuild`,
  `VisionPrefillImagePrefill`, `VisionPrefillCacheSave`,
  `VisionPrefillCacheHit`, and `VisionPrefillCacheRestore`. Timeline plotting
  aliases `VisionPrefillV_Encode`, `VisionPrefillMmproj`,
  `VisionPrefillImagePrefill`, and `VisionPrefillT_Prefill` onto the normal
  lanes, while cache-management rows are hidden. With `--partial-vision-kv`,
  ImagePrefill can appear as multiple committed micro-batch spans for one
  frame; these spans represent the partial KV commit points used for prompt
  preemption.

phase_timeline.png
  Common phase timeline plot. For streaming runs, the x-axis uses stream/video
  time, so a prompt at 3s is labeled around 3s rather than being rebased to 0s.
  For image, multi-image, and offline-video runs, the bridge loads and warms up
  first, then the x-axis is rebased to that ready point.
```

Notes:

```text
OnDemandBufferUpdate
  The current frame pointer is replaced by the latest sampled frame. This is not
  popping from an accumulated queue.

Prompt wait
  Prompt events are captured at their stream timestamp, but prompt execution is
  serialized. If prompt 1 arrives while prompt 0 is decoding, prompt 1 waits.
  The selected image/window remains frozen at prompt arrival.

Multi-turn
  Streaming on-demand, sliding-window, and vision-prefill keep llama.cpp
  chat/KV state across prompt events. In sliding-window, only the visual input
  is bounded to the selected recent frame window. In vision-prefill, frames
  arriving before a prompt are cached as an open user turn; the prompt text
  closes that turn, the answer is appended to chat history, and later frames are
  cached under the next user turn.
  `foundation_inference_tokens.txt` appends all turns, and
  `stream_inference_tokens_<idx>.txt` stores each turn's raw trace.

Vision-prefill scheduling
  Frame arrivals are still logged as `OnDemandBufferUpdate` ticks at their stream
  timestamps, even while the consumer lane is busy building older cache
  snapshots. Cache jobs are serialized: one cache build restores the previous
  snapshot, appends one new frame's text/image KV, saves the next snapshot, and
  then moves to the next cache job or prompt job. This keeps the phase trace to
  one `VisionPrefillV_Encode` / `VisionPrefillImagePrefill` pair per sampled
  frame.

Partial vision-prefill scheduling
  With `--partial-vision-kv`, a prompt can interrupt cache work after the
  current image micro-batch commits. The answer uses only committed image KV,
  and uncommitted visual placeholder slots are not counted as present. For a
  256-token InternVL3 image and `--ubatch-size 64`, the cache can be visible to
  the prompt after 64, 128, 192, or 256 image tokens.

Token trace contract
  `foundation_inference_tokens.txt` and `stream_inference_tokens_<idx>.txt`
  show mtmd prefill chunks followed by decode tokens. Vision-prefill cache
  prompt turns replay the committed cached KV trace before the question suffix:
  prior frame prefill, prior user text, prior assistant decode tokens, and the
  currently open frame prefix are visible in sequence. This is a logical trace of
  the restored cache state; it makes follow-up questions auditable even though
  the decoder does not re-prefill those earlier tokens. Image slots are emitted
  as `<VISION_KV_SLOT N>`; partial image commits emit only the committed slots
  and add `nominal_placeholder_tokens=256` when fewer than the full image token
  count was committed.
```

Example sliding-window run:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --stream-mode sliding-window \
  --sampling-fps 1.0 \
  --window-sec 4.0 \
  --window-max-frames 8 \
  --time '[5.0, 8.0]' \
  --prompt '["What is happening?", "What changed?"]' \
  --max-num 1 \
  --n-predict 64 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750
```

Swap `--stream-mode vision-prefill` to run the KV snapshot cached
vision-prefill mode. In this mode the `--window-sec` and `--window-max-frames`
limits are ignored and each frame incrementally appends to the active streaming
turn cache.

Example vision-prefill run used for the current KV snapshot validation:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8_20sec.mp4 \
  --stream-mode vision-prefill \
  --sampling-fps 1.0 \
  --max-video-time 10 \
  --time '[5.0, 8.0]' \
  --prompt '["What is happening in this video window?", "What changed in the recent window?"]' \
  --max-num 1 \
  --n-predict 32 \
  --ctx-size 4096 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 0 \
  --remote-root /data/local/tmp/streamingvlm_2b_kv_cache \
  --results-root my_research/foundation_llamacpp/results/log/vision_prefill_kv_cache_2b_hybrid_frame_ordered
```

Validated incremental multi-turn result:
`results/log/red_panda_vision_prefill_multiturn_interleaved_2b_dynamic512_frame1/InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_vision_prefill_kv16_dynamic`
with `foundation_exit_code.txt=0`, fifteen `VisionPrefillCacheBuild` rows, four
`VisionPrefillCacheHit` rows, zero `VisionPrefillCacheMiss` rows, zero
`DynamicKVGrow` rows for the 512-cell dynamic-KV run, and fifteen
`VisionPrefillV_Encode` / `VisionPrefillImagePrefill` rows for fifteen sampled
frames. Prompt 1 answered that the previous question was about the red panda's
activity, confirming chat history is preserved.

### Merge Regression Script

Before merging feature branches that touch prompt formatting, media handoff,
streaming scheduling, KV cache behavior, or token tracing, run:

```bash
my_research/foundation_llamacpp/scripts/run_merge_regression_internvl_qwen.sh
```

`N_PREDICT` defaults to 64 for this script so multi-turn history checks have
enough room to produce meaningful assistant turns. Override it only when the
test goal is specifically short-output latency.

The script runs InternVL3-1B and Qwen2.5-VL-3B through single image,
multi-image, offline video, streaming on-demand, streaming sliding-window, and
streaming vision-prefill. Streaming runs use dynamic KV growth,
`--online-buffer`, a small `--ubatch-size`, and for vision-prefill also
`--partial-vision-kv` plus `--latest-frame-only`.

By default it writes under:

```text
my_research/foundation_llamacpp/results/log/merge_regression_internvl_qwen_<timestamp>/
```

The script intentionally avoids pushing model artifacts. Populate the remote
root once, or point each model family at its cached remote root:

```bash
INTERNVL_REMOTE_ROOT=/data/local/tmp/streamingvlm_smoke_modes \
QWEN_REMOTE_ROOT=/data/local/tmp/streamingvlm_qwen25 \
RESULTS_ROOT=my_research/foundation_llamacpp/results/log/merge_regression_manual \
my_research/foundation_llamacpp/scripts/run_merge_regression_internvl_qwen.sh
```

Example partial vision-prefill run, using 64-token image KV commit points:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation/results/model/qnn/internvl3_1b_hybrid_16p_16k_16a4w/vision_encoder_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --vision-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --executorch-build-dir executorch/build-android-unified \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8_20sec.mp4 \
  --stream-mode vision-prefill \
  --partial-vision-kv \
  --sampling-fps 1.0 \
  --max-video-time 20 \
  --time '[5.0, 8.0, 11.0, 14.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???", "What changed in the scene?", "Summarize the full situation so far."]' \
  --max-num 1 \
  --n-predict 64 \
  --ctx-size 32768 \
  --dynamic-kv-cache \
  --kv-init-size 512 \
  --kv-grow-step 512 \
  --batch-size 1024 \
  --ubatch-size 64 \
  --gpu-layers 99 \
  --threads 4 \
  --temperature 0.0 \
  --device GPUOpenCL \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --fit off \
  --remote-root /data/local/tmp/streamingvlm_1b_partial_vprefill \
  --results-root my_research/foundation_llamacpp/results/log/partial_vprefill_clean_surveillance_1b_q8_batch64_20s_4prompt
```

Validated partial result:
`results/log/partial_vprefill_clean_surveillance_1b_q8_batch64_20s_4prompt/InternVL3-1B-Instruct-Q8_0_hybrid_ctx_32768_streaming_vision_prefill_kv8_dynamic`
completed with `foundation_exit_code.txt=0`. Prompt 1 recovered the previous
question, and `phase_timeline.png` shows the 64-token image-prefill batch
boundaries used for partial KV preemption. A 2B Q8 retry is still pending
because the Android device disconnected during adb push before inference
started.

## Vision Tower Export

Use this flow to export the InternVL3 vision tower pre-projector to ExecuTorch
QNN. This artifact outputs visual features before the InternVL
`multi_modal_projector`; the hybrid decoder applies the GGUF `mmproj` on OpenCL.

```bash
export QNN_SDK_ROOT=/workspace/streamingvlm/executorch/backends/qualcomm/sdk/qnn/qairt/2.37.0.250724
export EXECUTORCH_ROOT=/workspace/streamingvlm/executorch
export ANDROID_NDK_ROOT=/opt/android-ndk-r26c
export LIBCXX_DIR=/opt/conda/envs/stream/lib/python3.11/site-packages/executorch/backends/qualcomm/sdk/libcxx-14.0.0
export PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LIBCXX_DIR:$EXECUTORCH_ROOT/build-x86/lib:${LD_LIBRARY_PATH:-}"

PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python -m my_research.foundation.models.internvl3.vision_encoder.export_pre_projector_qnn \
  --model-name internvl3_1b \
  --model-path OpenGVLab/InternVL3-1B-hf \
  --artifact-root my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750 \
  --soc-model SM8750 \
  --quant 16a8w \
  --calibration-images \
    my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
    my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
  --calibration-num 2
```

Expected artifact:

```text
my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte
```

Export notes:

```text
--quant 16a8w
  Current practical QNN quantization setting.

projector_included: false
  The exported PTE stops before the InternVL projector.

output shape
  Single image output is `1 x 256 x 4096` for the current InternVL3 1B vision
  tower pre-projector artifact.

Do not pass:
  --encoder-weights my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth

Reason:
  That file contains decoder-style keys and no `vision_tower.*` weights.
```

### Qwen2.5-VL Vision Encoder Export

Qwen2.5-VL uses dynamic image resize in the HF processor, so the QNN artifact
is exported for one concrete resolution/grid at a time. This artifact stops
before Qwen's `visual.patch_merger`, so it outputs pure vision block features.
Pick either a fixed `--image-size HEIGHT WIDTH` or an expected
`--image-tokens N`; when only tokens are given, the exporter infers a
near-square patch grid whose height/width are divisible by `spatial_merge_size`.

The exporter checks that the actual visual output token count equals
`grid_t * grid_h * grid_w` and records the result in
`vision_encoder_qnn_metadata.json`. For example, 448 x 448 becomes
`image_grid_thw = [1, 32, 32]` and outputs 1024 pre-merger tokens.

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python -m my_research.foundation.models.qwen2_5_vl.vision_encoder.export_qnn \
  --model-name qwen2_5_vl_3b \
  --model-path Qwen/Qwen2.5-VL-3B-Instruct \
  --artifact-root my_research/foundation_llamacpp/results/vision_models/qwen2_5_vl_3b_vision_encoder_premerger_qnn_1024tok_sm8750 \
  --soc-model SM8750 \
  --quant 16a8w \
  --image-tokens 1024 \
  --calibration-images \
    my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
    my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
  --calibration-num 2
```

Expected artifacts:

```text
my_research/foundation_llamacpp/results/vision_models/qwen2_5_vl_3b_vision_encoder_premerger_qnn_1024tok_sm8750/vision_encoder_qnn.pte
my_research/foundation_llamacpp/results/vision_models/qwen2_5_vl_3b_vision_encoder_premerger_qnn_1024tok_sm8750/vision_encoder_qnn_metadata.json
```

Validated export metadata:

```text
patch_merger_included: false
projector_included: false
output_shape: [1, 1024, 1280]
image_grid_thw: [1, 32, 32]
token_count_check: actual_tokens=1024, expected_tokens=1024, passed=true
```

For a host export run that leaves room for other jobs while giving QNN lowering
36 CPU cores, prefix the command with:

```bash
export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=36
export OPENBLAS_NUM_THREADS=36
export NUMEXPR_MAX_THREADS=36
export NUMEXPR_NUM_THREADS=36
export VECLIB_MAXIMUM_THREADS=36
export MALLOC_ARENA_MAX=4

taskset -c 0-35 python -m my_research.foundation.models.qwen2_5_vl.vision_encoder.export_qnn ...
```

Useful shape examples:

```text
--image-tokens 256
  Infers 224 x 224, image_grid_thw = [1, 16, 16], output tokens = 256.

--image-tokens 1024
  Infers 448 x 448, image_grid_thw = [1, 32, 32], output tokens = 1024.

--image-size 392 560 --image-tokens 1120
  Uses the explicit rectangular export size and fails early if the token count
  does not match.
```

## Phase Names

```text
L_VisionLoad
  ExecuTorch/QNN vision module load. Hybrid only.

L_DecoderRuntimeInit
  llama.cpp argument parsing and OpenCL runtime/device/kernel setup.

L_DecoderLoad
  llama.cpp model/context/mmproj load.

ImageLoad
  Input image/tensor load.

LayoutTokenize
  mtmd text/image layout construction.

V_Encode
  Vision tower encode. OpenCL path uses llama.cpp/mtmd; hybrid path uses QNN.

Mmproj
  InternVL multi-modal projector. Hybrid pre-projector features are projected by
  llama.cpp OpenCL mmproj before image prefill.

ImagePrefill
  Feed projected image embeddings into llama.cpp context/KV.

T_Prefill
  Text token prefill.

Prefill
  Combined prefill interval.

D
  One generated-token decode step.
```

## Etc

OpenCL note:

```text
Do not push local `libOpenCL.so` unless needed.
```

On the tested Qualcomm device, Android's system OpenCL loader discovers Adreno
correctly. Pushing a local ICD loader can cause:

```text
ggml_opencl: platform IDs not available
invalid device: GPUOpenCL
```

Use `--push-opencl-loader` only when intentionally testing a custom loader.

KV-cache note:

```text
--cache-type-k q8_0 --cache-type-v q8_0
```

uses 8-bit KV where supported. Result folders use `_kv8`. If a backend rejects
the dtype, model load will fail.

Long-context / YaRN note:

```bash
--ctx-size 131072 \
--rope-scaling yarn \
--rope-scale 4.0 \
--yarn-orig-ctx 32768
```

Use these only when the GGUF metadata does not already encode the intended RoPE
scaling or when you intentionally override it.

Build and implementation details:

```text
my_research/foundation_llamacpp/docs/archive/executorch_vision_llamacpp_decoder.md
my_research/foundation_llamacpp/docs/project_structure.md
```
