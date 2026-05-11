# Foundation Llama.cpp Project Structure

This document describes the current `my_research/foundation_llamacpp` code
layout after the hybrid bridge refactor. It is intended for future agents and
contributors who need to understand where to add image, video, streaming, CPU,
OpenCL, or hybrid QNN/OpenCL functionality.

## Design Goals

The project keeps project-specific mobile VLM runtime code outside upstream
`llama.cpp` and ExecuTorch sources. The main boundary is:

```text
Python runner
  prepares media, pushes binaries/models/assets to Android, launches remote
  scripts, pulls artifacts, and writes summaries.

C++ bridge binaries
  run llama.cpp, OpenCL multimodal prefill/decode, ExecuTorch/QNN vision encode,
  and external embedding handoff.

Upstream dependencies
  llama.cpp and ExecuTorch are treated as dependencies. Project-specific bridge
  code lives in this directory rather than being mixed into upstream trees.
```

Current note: the OpenCL InternVL timing split uses a small local llama.cpp/mtmd
debug hook (`clip_image_encode_internvl_split()` and
`mtmd_encode_chunk_split_timing()`) so OpenCL can report `V_Encode` and `Mmproj`
separately. Treat this as a project patch/debug hook when updating llama.cpp
upstream.

The current refactor separates three axes that were previously mixed together:

```text
media mode:
  text, image, video_file, future streaming

backend mode:
  cpu, opencl, hybrid_qnn_opencl

execution stage:
  media preparation, vision encode, prompt layout, prefill, decode, artifact
  finalization
```

## Top-Level Layout

```text
my_research/foundation_llamacpp/
  README.md
  run_android_hybrid_bridge.py
  runner/
  hybrid_bridge/
  docs/
  scripts/
  sample_images/
  results/
  build-hybrid-android-opencl/
```

`run_android_hybrid_bridge.py` is now only a compatibility entrypoint. It imports
`my_research.foundation_llamacpp.runner.cli.main` and exits with that return
code. Keep this path stable because README commands, old scripts, and user muscle
memory depend on it.

`runner/` contains the Python orchestration package. New runner-side code should
go here.

`hybrid_bridge/` contains project-owned C++ bridge binaries and shared bridge
helpers. These targets are built against upstream llama.cpp and, when enabled,
ExecuTorch.

`docs/` contains user-facing run guides, this structure document, the active
implementation log, and historical archived notes.

`results/` contains run outputs and exported model artifacts. Do not commit large
generated result logs or binary artifacts unless explicitly requested.

`build-hybrid-android-opencl/` is a generated Android build directory. Treat
build directories as disposable/generated outputs, not source.

## Python Runner Package

The Python runner lives under:

```text
my_research/foundation_llamacpp/runner/
  __init__.py
  cli.py
  config.py
  media.py
  remote.py
  artifacts.py
  finalize.py
  llama_args.py
  backends/
    __init__.py
    standalone.py
    hybrid_qnn_opencl.py
```

### `runner/cli.py`

`runner/cli.py` owns the current CLI and end-to-end orchestration:

```text
parse args
normalize/validate paths
derive media/backend modes
prepare media
push Android runtime files
build remote shell script
execute through adb
pull artifacts
finalize summaries and plots
```

This file is still the largest Python module. The compatibility wrapper calls
`main()` from here. Future cleanup should continue moving cohesive pieces out of
`cli.py` instead of adding more unrelated logic to it.

Important responsibilities still in `cli.py`:

- llama.cpp command rendering for standalone CPU/OpenCL.
- remote shell script construction for standalone and hybrid flows.
- result directory naming.
- memory timeline shell snippets.
- summary extraction from logs.
- final artifact pulling and post-processing.

### `runner/config.py`

`config.py` defines the explicit mode contracts:

```text
MediaMode.TEXT
MediaMode.IMAGE
MediaMode.VIDEO_FILE
MediaMode.STREAMING

BackendMode.CPU
BackendMode.OPENCL
BackendMode.HYBRID_QNN_OPENCL
```

It also defines `PreparedMedia`, the handoff object returned by media
preparation:

```text
frame_bins:
  preprocessed CHW float32 `.bin` files used by hybrid QNN vision encode

layout_images:
  PNG/JPG files passed to mtmd/llama.cpp for multimodal prompt layout

prompt:
  final prompt after media markers are inserted

metadata_path:
  path to generated `media_manifest.json`

num_patches_list:
  number of InternVL tiles per image or sampled frame

frame_indices:
  source video frame indices for video_file mode

source_kind:
  "image" or "video"
```

`MediaMode.STREAMING` is reserved for later incremental frame ingestion. The
current video mode is offline sampling from a video file.

### `runner/media.py`

`media.py` owns image and video preparation.

For image mode:

```text
input image
  -> load through transformers.image_utils.load_image
  -> normalize to ImageNet mean/std
  -> resize to 448 x 448
  -> write frame_0000.bin
  -> copy layout image
  -> write media_manifest.json
```

For video mode:

```text
input video
  -> decord.VideoReader
  -> uniformly sample --num-segments frame indices
  -> apply InternVL dynamic preprocessing per frame
  -> write one `.bin` and one layout `.png` per tile
  -> construct prompt:
       Frame 1: <__media__>
       Frame 2: <__media__>
       ...
       user prompt
  -> write media_manifest.json
```

The manifest currently uses:

```text
schema_version: 2
source_kind: image | video
source: original input path
frame_indices: sampled source frame indices
num_patches_list: tile count per image/frame
frame_bins: preprocessed binary inputs
layout_images: images used for mtmd prompt layout
frames: per-frame tile metadata for video
prompt: final media-expanded prompt
raw_prompt: original user prompt for video
```

For hybrid video with `--num-segments 8 --max-num 1`, the expected QNN bridge
embedding shape is:

```text
8 x 256 x 4096
```

### `runner/remote.py`

`remote.py` contains small adb and shell helpers:

```text
run()
adb_cmd()
push()
remote_exists()
pull_if_exists()
shell_join()
```

Keep raw adb subprocess handling here when adding reusable behavior. Avoid
duplicating push/pull/test-file helpers in backend code.

### `runner/artifacts.py`

`artifacts.py` defines canonical artifact pull lists.

Hybrid pulls:

```text
hybrid_vision_stdout.txt
hybrid_decode_stdout.txt
vision_output_stats.csv
vision_phase_stats.csv
decoder_phase_stats.csv
foundation_token_io.txt
foundation_inference_tokens.txt
vision_embedding.svlmemb
media_manifest.json
foundation_exit_code.txt
vision_exit_code.txt
decoder_exit_code.txt
android_memory_timeline.csv
```

Standalone pulls:

```text
foundation_output.txt
foundation_exit_code.txt
foundation_phase_stats.csv
foundation_token_io.txt
foundation_inference_tokens.txt
opencl_projected_embedding.svlmemb
media_manifest.json
android_memory_timeline.csv
```

Add new artifacts here first, then have the runner pull from this canonical list.

### `runner/finalize.py`

`finalize.py` currently holds generic CSV helpers. Much of the summary and plot
finalization still lives in `cli.py`; future refactors should move the remaining
summary logic here.

### `runner/llama_args.py`

`llama_args.py` contains reusable llama.cpp CLI rendering helpers. It currently
hosts RoPE/YaRN shell suffix rendering. Future shared command rendering should
move here so standalone and hybrid do not drift.

### `runner/backends/`

`backends/standalone.py` and `backends/hybrid_qnn_opencl.py` define backend marker
dataclasses.

Current backend behavior:

```text
cpu:
  llama.cpp text or multimodal through CPU settings

gpu:
  llama.cpp OpenCL path; video is handled as multiple sampled images by mtmd

hybrid:
  ExecuTorch/QNN vision process produces `.svlmemb`
  llama.cpp/OpenCL decoder process consumes that embedding
```

The hybrid flow intentionally remains a two-process coordinated run because QNN
vision loading/encode and OpenCL decoder loading have different runtime
constraints. Do not merge `hybrid_vision_dump` and `hybrid_decode` unless that
coordination requirement changes.

## C++ Hybrid Bridge

The bridge lives under:

```text
my_research/foundation_llamacpp/hybrid_bridge/
  CMakeLists.txt
  hybrid_decode.cpp
  opencl_phase_mtmd.cpp
  hybrid_vision_dump.cpp
  hybrid_embedding_file.h
  hybrid_embedding_file.cpp
  inference_trace.hpp
  file_sync.hpp
  phase_trace.hpp
  vision_encoder_et.hpp
  vision_encoder_et.cpp
```

### Targets

`hybrid_decode`

```text
Purpose:
  llama.cpp decoder process for hybrid QNN/OpenCL mode.

Inputs:
  text GGUF
  mmproj GGUF
  layout image list
  external `.svlmemb` embedding file from hybrid_vision_dump

Responsibilities:
  load llama.cpp model/context/mmproj
  wait for external embedding when requested
  tokenize mtmd prompt layout
  consume one external embedding slice per IMAGE chunk
  run combined prefill
  decode text
  write token/phase traces
```

`opencl_phase_mtmd`

```text
Purpose:
  standalone OpenCL multimodal runner with precise phase tracing.

Inputs:
  text GGUF
  mmproj GGUF
  image/layout list

Responsibilities:
  use llama.cpp/mtmd for vision encode and mmproj
  split InternVL OpenCL timing into V_Encode and Mmproj
  rebuild the projector-only scheduler/graph for each timed Mmproj call
  warm one fixed Golden Gate split encode+mmproj pass before recording timings
  prefill image/text chunks
  decode text
  write foundation_phase_stats.csv and token traces
```

`hybrid_vision_dump`

```text
Purpose:
  ExecuTorch/QNN vision process for hybrid mode.

Inputs:
  QNN/ExecuTorch encoder PTE
  one or more preprocessed CHW float32 `.bin` files

Responsibilities:
  load ExecuTorch module
  load image tensors
  warm one encoder.encode() pass on the fixed Golden Gate warmup input
  encode every input in order
  concatenate output features
  write `vision_embedding.svlmemb`
  write vision stats and phase rows
```

### Shared C++ Helpers

`hybrid_embedding_file.{h,cpp}`

Defines the bridge embedding file format used between QNN vision and llama.cpp
decoder:

```text
magic/version
shape rank and dimensions
raw float32 embedding values
```

For a video sampled as eight single-tile frames, the shape is:

```text
8 x 256 x 4096
```

`inference_trace.hpp`

Writes detailed token and chunk traces:

```text
## CHUNK N TEXT
## CHUNK N IMAGE image_index=M n_placeholder_tokens=...
decode token lines
```

Use `foundation_inference_tokens.txt` to verify that image/video prompt layout
matches the expected number of IMAGE chunks.

`file_sync.hpp`

Contains ready/wait text-file synchronization helpers. The hybrid remote script
uses small flag files so decoder and vision processes can load first, then begin
encode/decode coordination.

`phase_trace.hpp`

Contains `phase_recorder` and phase descriptions shared by bridge binaries. Phase
CSV rows use this schema:

```text
row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,
col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,
kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx
```

Important phase names:

```text
L_VisionLoad
ImageLoad
V_Encode
Mmproj
EmbeddingFileWrite
L_DecoderRuntimeInit
L_DecoderLoad
ExternalEmbeddingRead
LayoutTokenize
Prefill
ImagePrefill
T_Prefill
D
Decode
```

Phase semantics:

```text
V_Encode:
  OpenCL standalone: llama.cpp/mtmd InternVL vision tower through pixel shuffle,
  before the multi-modal projector.
  Hybrid: ExecuTorch/QNN pre-projector vision tower output.

Mmproj:
  InternVL projector/mmproj.
  OpenCL standalone: split out from the full llama.cpp/mtmd InternVL path.
  Hybrid: decoder-side llama.cpp OpenCL projection of QNN pre-projector features.

ImagePrefill:
  llama_decode() on already projected image embeddings. This is not image
  encoding, and it is not expected to scale purely from token count versus
  T_Prefill. Bridge phase timing synchronizes immediately after llama_decode()
  so asynchronous OpenCL work is charged to ImagePrefill instead of the next
  text/decode phase. The mtmd helper stdout line `image decoded ... in N ms`
  is printed before this bridge-level synchronize; use CSV/PNG phase rows for
  synchronized timing.

T_Prefill:
  llama_decode() on text token-id batches; the final text chunk may request
  logits. Bridge phase timing synchronizes immediately after llama_decode().
```

`vision_encoder_et.{hpp,cpp}`

Owns ExecuTorch/QNN vision encode logic:

```text
parse comma-separated image paths
validate group sizes
load ExecuTorch module
read expected tensor metadata
load CHW float32 image bins
run encoder for each input
validate all outputs share token count and feature dimension
concatenate outputs
return VisionEncodeResult
write vision_output_stats.csv
```

`hybrid_vision_dump.cpp` should remain a thin CLI wrapper around this module.

## Runtime Flows

### Text-Only CPU/OpenCL

```text
run_android_hybrid_bridge.py
  -> runner.cli
  -> no image/video
  -> llama-completion on Android
  -> pull foundation_output.txt and memory timeline
  -> finalize summaries
```

### Standalone Image/Video OpenCL

```text
run_android_hybrid_bridge.py
  -> runner.media.prepare_media
  -> push layout images and models
  -> opencl_phase_mtmd or llama-mtmd-cli on Android
  -> mtmd tokenizes prompt/media layout
  -> opencl_phase_mtmd warms one fixed Golden Gate split InternVL encode + mmproj pass
  -> llama.cpp performs measured vision encode + mmproj + prefill + decode
  -> pull foundation_phase_stats.csv, token trace, memory timeline
  -> finalize summaries and plots
```

Video is represented as ordered image frames. The prompt contains one
`<__media__>` marker per tile, and mtmd converts those media markers into model
specific image wrappers such as InternVL `<img>...</img>`.

### Hybrid QNN Vision + OpenCL Decoder

```text
run_android_hybrid_bridge.py
  -> runner.media.prepare_media
  -> push `.bin` frame/tile tensors and layout images
  -> start hybrid_decode in background
  -> start hybrid_vision_dump in background
  -> both processes signal ready
  -> coordinator touches start_encode.flag
  -> hybrid_vision_dump warms one QNN encoder.encode() pass on the fixed Golden Gate input
  -> hybrid_vision_dump writes vision_embedding.svlmemb
  -> hybrid_decode reads embedding and consumes slices per IMAGE chunk
  -> hybrid_decode warms mtmd_project_features() once with fixed Golden Gate QNN
     pre-projector features before measuring Mmproj on the actual input
  -> pull vision/decoder stats, traces, embedding file, memory timeline
  -> finalize summaries and plots
```

The key contract is that the order of `frame_bins`, `layout_images`, prompt media
markers, and `.svlmemb` slices must match. For single-tile frames:

```text
Frame 1 marker -> embedding slice 0
Frame 2 marker -> embedding slice 1
...
Frame N marker -> embedding slice N - 1
```

## Result Artifacts

Each run writes under:

```text
my_research/foundation_llamacpp/results/log/<model>_<backend>_ctx_<ctx>_kv<kv>/
```

Core artifacts:

```text
foundation_exit_code.txt:
  runner-level exit code; must be 0 for success

foundation_summary.csv:
  high-level backend, runtime, throughput, and memory summary

foundation_proc.csv:
  normalized phase rows used for runtime plots

foundation_output.txt:
  raw model output or decoder stdout copy

foundation_token_io.txt:
  user/assistant text plus trace appendix when available

foundation_inference_tokens.txt:
  detailed chunk/token trace; best file for checking media chunk count

android_memory_timeline.csv:
  sampled Android memory timeline

memory_usage_summary.txt:
  post-processed memory summary

phase_duration_stacked_bar.png:
  runtime phase plot when matplotlib is available

memory_timeline_plot.png:
  memory timeline plot when matplotlib is available
```

Hybrid-only artifacts:

```text
hybrid_vision_stdout.txt
hybrid_decode_stdout.txt
vision_output_stats.csv
vision_phase_stats.csv
decoder_phase_stats.csv
vision_embedding.svlmemb
hybrid_projected_embedding.svlmemb
vision_exit_code.txt
decoder_exit_code.txt
media_manifest.json
```

Standalone OpenCL artifacts:

```text
foundation_phase_stats.csv
opencl_projected_embedding.svlmemb
media_manifest.json
```

## Build Notes

Build the Android bridge from the active Android build directory:

```bash
cmake --build my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --target hybrid_decode opencl_phase_mtmd hybrid_vision_dump -j2
```

Use temporary host build directories for host compile checks. Do not rely on or
commit `build-hybrid-host`; it is obsolete generated output.

The CMake options are:

```text
HYBRID_BRIDGE_BUILD_LLAMA_DECODER:
  builds hybrid_decode and opencl_phase_mtmd against llama.cpp

HYBRID_BRIDGE_BUILD_EXECUTORCH_VISION:
  builds hybrid_vision_dump against ExecuTorch/QNN
```

## Extension Guidelines

When adding a new media mode:

```text
1. Add/extend MediaMode in runner/config.py.
2. Add media preparation in runner/media.py or a new media submodule.
3. Emit a versioned media_manifest.json.
4. Preserve ordered alignment between prompt media markers, layout images, and
   any generated tensor/embedding files.
5. Add artifact checks to README/docs when the mode becomes user-facing.
```

When adding a new backend:

```text
1. Add/extend BackendMode in runner/config.py.
2. Add backend-specific contract code under runner/backends/.
3. Keep adb push/pull helpers in runner/remote.py.
4. Keep canonical output names in runner/artifacts.py.
5. Avoid duplicating llama argument rendering; move shared rendering into
   runner/llama_args.py.
```

When changing C++ bridge behavior:

```text
1. Prefer adding project-owned helper files in hybrid_bridge/.
2. Avoid editing upstream llama.cpp or ExecuTorch unless there is no wrapper or
   overlay option.
3. If an upstream llama.cpp timing/debug hook is unavoidable, keep it minimal and
   document it as a project patch that must be checked during upstream updates.
4. Keep binaries thin: parse CLI flags, call shared modules, write artifacts.
5. Preserve existing target names because the Python runner and README commands
   call them directly.
6. Rebuild hybrid_decode, opencl_phase_mtmd, and hybrid_vision_dump.
7. Run at least one image or video smoke test when prompt/media layout changes.
```

When adding streaming later:

```text
1. Do not overload video_file semantics. Video_file is offline sampled frames.
2. Streaming should have an explicit state model for append/evict behavior.
3. Manifest schema should add timestamps or stream sequence ids.
4. Decoder context retention/eviction must be explicit; do not hide it inside
   image/video preparation helpers.
5. Start with interfaces and artifact contracts before optimizing scheduling.
```

## Known Current Gaps

The refactor established package boundaries but did not finish every possible
split. The main remaining cleanup opportunities are:

```text
runner/cli.py:
  still owns remote script construction, result naming, summary extraction, and
  most finalization logic.

runner/finalize.py:
  should eventually own all summary/plot post-processing.

runner/llama_args.py:
  should eventually own all shared llama.cpp argv/shell rendering.

runner/backends/:
  currently contains backend marker contracts; fuller backend planners can move
  here over time.

hybrid_decode.cpp and opencl_phase_mtmd.cpp:
  still duplicate some llama session, mtmd layout, prefill, and generation
  behavior. Future C++ splits should introduce modules such as
  llama_decode_session, mtmd_layout, prefill_engine, and generation while keeping
  target behavior unchanged.
```

