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
  text, image, video_file, streaming

backend mode:
  cpu, opencl, hybrid_qnn_opencl

execution stage:
  media preparation, vision encode, prompt layout, prefill, decode, artifact
  finalization
```

Current 2026-05-15 baseline:

```text
hybrid image / multi-image / offline video / streaming
  -> one Android binary: hybrid_streaming_decode

streaming mode names
  on_demand       canonical latest-frame baseline
  single_buffer   accepted legacy alias only
  sliding_window  selected recent-frame clip with multi-turn text/KV state
  vision_prefill  incremental KV-level visual prefix cache

online buffer
  optional --online-buffer decouples input frame cadence from processing
  cadence; delayed work uses the latest frame/window at processing start and
  vision-prefill coalesces stale pending cache jobs.

latest-frame-only
  optional --latest-frame-only for streaming vision_prefill drops frame cache
  updates that arrive while the worker is busy. The next cache update starts
  from the next frame arrival after the worker becomes idle.

partial vision prefill
  optional --partial-vision-kv for hybrid vision_prefill lets a prompt preempt
  image-prefill cache work after the current image micro-batch commits. The
  partial commit size is controlled by --ubatch-size.
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
- streaming flag forwarding, including `--stream-mode`, `--online-buffer`,
  `--latest-frame-only`, `--dynamic-kv-cache`, and `--partial-vision-kv`.
- memory timeline shell snippets.
- summary extraction from logs.
- final artifact pulling and post-processing.
- dynamic KV grow post-processing for streaming runs: parse llama.cpp grow logs,
  align them to the `streaming_phase_stats.csv` `clock_origin_ms`, split the
  aggregate `Prefill` row around `DynamicKVGrow`, and clip retry-side
  `ImagePrefill` timing so grow/retry overhead is shown as the black grow bar.

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
  "image", "multi_image", "video", or "streaming_video"
```
`MediaMode.STREAMING` is the file-backed streaming simulator path. It is
separate from `MediaMode.VIDEO_FILE`, which still means offline sampled frames
that are all presented to the model at once.

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

For video-file mode:

```text
input video
  -> decord.VideoReader
  -> uniformly sample --num-segments frame indices
  -> apply InternVL dynamic preprocessing per frame
  -> write one `.bin` and one layout `.png` per tile
  -> construct prompt:
       Frame1: <__media__>
       Frame2: <__media__>
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

For streaming video mode:

```text
input video
  -> decord.VideoReader
  -> sample frames at --sampling-fps up to optional --max-video-time
  -> write stream_frame_<idx>.png layout images
  -> in hybrid streaming modes, also write stream_frame_<idx>.bin QNN tensors
  -> record prompt events from --time and JSON-list --prompt
  -> write media_manifest.json with source_kind: streaming_video
```

Streaming manifest fields include:

```text
source_fps:
  source video FPS read by decord

sampling_fps:
  replay/sample FPS requested by CLI

duration_s / effective_duration_s / max_video_time:
  original and clipped stream duration

stream_mode:
  "on_demand", "sliding_window", or "vision_prefill"

frames:
  stream_frame index, timestamp_s, video_frame_index, num_patches, and tile files

prompt_events:
  prompt timestamp_s and prompt text
```

In `--stream-mode on-demand`, `OnDemandBufferUpdate` means the current frame
pointer is replaced with the latest sampled frame. It does not pop from an
accumulated queue. Prompt events normally capture the buffered frame at arrival
time; the actual prompt execution lane is serialized, so a prompt may wait
behind an earlier decode. `--single-buffer` and
`--stream-mode single-buffer` remain accepted aliases for this mode.

With `--online-buffer`, prompt/cache jobs do not freeze their selected frames at
request timestamp. They read the latest buffered frame/window when the consumer
actually starts the job, and stale pending vision-prefill cache updates are
coalesced.

With `--latest-frame-only`, frame cache updates use a stricter live-camera
policy. A frame arriving while the consumer is busy with an older cache update
or prompt does not queue a cache update. It is counted as
`LatestFrameOnlyCacheDrop`; the next cache update can only start from a frame
that arrives after the consumer becomes idle.

In `--stream-mode sliding-window`, prompt events capture a selected list of
sampled frames rather than one latest frame. The selection is bounded by
`--window-sec` and then evenly reduced to `--window-max-frames` when needed.
The selected frames behave like the current visual window, while the decoder
chat/KV state is preserved across prompt events for multi-turn text context.

In `--stream-mode vision-prefill`, every frame arrival enqueues a cache-update
job. The first cache-update builds frame 0 from scratch. Later cache updates
restore the previous snapshot, append only the newly arrived frame's global
`FrameN:` text/image KV to the currently open streaming user turn, save the
llama sequence state, and replace the previous cache snapshot. Prompt events
restore the latest matching snapshot, evaluate only the text suffix that closes
that user turn, decode the answer, and save the post-answer chat/KV state.
Later frame arrivals then begin the next streaming user turn. `--window-sec`
and `--window-max-frames` are intentionally ignored in this mode.

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

Streaming pulls:

```text
hybrid_streaming_stdout.txt
opencl_streaming_stdout.txt
foundation_output.txt
stream_events.csv
streaming_phase_stats.csv
foundation_token_io.txt
foundation_inference_tokens.txt
stream_inference_tokens_*.txt
media_manifest.json
foundation_exit_code.txt
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
  default media path uses one `hybrid_streaming_decode` process that owns both
  ExecuTorch/QNN vision encode and llama.cpp/OpenCL mmproj/decode
```

The legacy two-process flow (`hybrid_vision_dump` producing `.svlmemb` and
`hybrid_decode` consuming it) remains available in source for diagnostics and
historical comparisons, but the runner no longer uses it as the default hybrid
image/video path.

## C++ Hybrid Bridge

The bridge lives under:

```text
my_research/foundation_llamacpp/hybrid_bridge/
  CMakeLists.txt
  hybrid_decode.cpp
  opencl_phase_mtmd.cpp
  hybrid_streaming_decode.cpp
  kv_reposition_probe.cpp
  hybrid_vision_dump.cpp
  hybrid_embedding_file.h
  hybrid_embedding_file.cpp
  inference_trace.hpp
  file_sync.hpp
  kv_reposition.hpp
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

`hybrid_streaming_decode`

```text
Purpose:
  unified hybrid QNN/OpenCL runner for image, multi-image, offline video,
  on-demand streaming, sliding-window streaming, and vision-prefill streaming.

Compile mode:
  STREAMINGVLM_STREAMING_DECODE_USE_QNN=1
  STREAMINGVLM_HYBRID_DECODE_NO_MAIN=1

Inputs:
  text GGUF
  mmproj GGUF
  QNN/ExecuTorch vision PTE
  media_manifest.json
  frame/tile .bin inputs and PNG/JPG layout images

Responsibilities:
  load QNN VisionEncoderSession once
  optionally warm QNN vision once with fixed warmup bin
  load llama.cpp/mmproj decode context once
  for offline image/multi-image/video, encode all manifest bins once and run
  one prompt through the same decode context
  for streaming, replay sampled frame arrivals according to manifest timestamps
  maintain sampled-frame history and the latest-frame buffer
  capture prompt events against the correct frame/window snapshot
  for on-demand, QNN-encode only the selected frame per prompt
  for sliding-window, evaluate the selected window while preserving chat/KV state
  for vision-prefill, incrementally append frame KV into full-history snapshots
  with --online-buffer, select frames at processing start and coalesce stale
  pending cache updates
  run decoder-side mmproj, prefill, and decode
  preserve chat history and KV state in all streaming modes
  write stream events, phase rows, output, and per-turn token traces
```

`opencl_streaming_decode`

```text
Purpose:
  standalone OpenCL on-demand streaming runner.

Compile mode:
  STREAMINGVLM_OPENCL_PHASE_MTMD_NO_MAIN=1

Inputs:
  text GGUF
  mmproj GGUF
  streaming media_manifest.json
  stream_frame_<idx>.png files

Responsibilities:
  include opencl_phase_mtmd.cpp in-process
  load llama.cpp/mtmd OpenCL context once
  replay sampled frame arrivals with the same on-demand event model
  capture prompt events against the current buffered frame
  run llama.cpp/mtmd OpenCL vision encode + mmproj + prefill + decode
  preserve chat history and KV state across prompts
  write the same streaming artifacts as hybrid
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

`kv_reposition.hpp`

Defines the experimental KV tail-compaction contract for future video
compression. The helper builds and applies a `llama_memory_seq_rm` plus
`llama_memory_seq_add` plan so llama.cpp can update cached K RoPE positions
without re-prefilling the unchanged suffix. The current helper is intentionally
policy-free: compression decides which visual token range is removed or
rewritten, then this helper shifts the remaining logical positions.

`kv_reposition_probe.cpp`

Host-only validation binary for the KV/RoPE reposition helper. It compares a
compact reference prefill with a `prefix + removed + history` cache that has the
removed span deleted and the history tail shifted before a fresh suffix prefill.
Use it to check whether a proposed compression policy preserves practical
answers before wiring that policy into streaming video.

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
VisionEncoderSession:
  reusable ExecuTorch/QNN module session for streaming

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
Streaming code should use `VisionEncoderSession` instead of constructing a new
ExecuTorch module per prompt.

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
  -> run hybrid_streaming_decode --media-mode image|multi-image|video
  -> hybrid_streaming_decode warms QNN encoder.encode() once on fixed Golden Gate input
  -> hybrid_streaming_decode warms llama.cpp/mmproj decode context once
  -> encode all manifest bins in order
  -> consume slices per IMAGE chunk inside the same process
  -> pull unified phase rows, traces, output, and memory timeline
  -> finalize summaries and plots
```

The key contract is that the order of `frame_bins`, `layout_images`, prompt media
markers, and `.svlmemb` slices must match. For single-tile frames:

```text
Frame1 marker -> embedding slice 0
Frame2 marker -> embedding slice 1
...
FrameN marker -> embedding slice N - 1
```

The older `hybrid_vision_dump + hybrid_decode` two-process handoff still exists
for historical comparison, but it is no longer the default hybrid image/video
runner path.

### Streaming On-Demand OpenCL

```text
run_android_hybrid_bridge.py --processor gpu --streaming-video ... --stream-mode on-demand
  -> runner.media.prepare_streaming_video_media
  -> push streaming media_manifest.json and stream_frame_<idx>.png layout images
  -> run opencl_streaming_decode
  -> opencl_streaming_decode loads llama.cpp/mtmd OpenCL context once
  -> producer thread replays sampled frame timestamps
  -> OnDemandBufferUpdate replaces current frame pointer
  -> prompt job records current frame at arrival time
  -> consumer lane serially runs eval_message() and generate_response()
  -> output stream_events.csv, streaming_phase_stats.csv, per-turn token traces
  -> runner finalizes foundation_proc.csv, phase_duration_stacked_bar.png, and phase_timeline.png
```

OpenCL streaming uses llama.cpp/mtmd for `V_Encode` and `Mmproj`. It is useful
as a direct full-OpenCL baseline for the hybrid QNN streaming path.

### Streaming On-Demand Hybrid

```text
run_android_hybrid_bridge.py --processor hybrid --streaming-video ... --stream-mode on-demand
  -> runner.media.prepare_streaming_video_media
  -> push media_manifest.json, stream_frame_<idx>.png, stream_frame_<idx>.bin
  -> run hybrid_streaming_decode
  -> hybrid_streaming_decode loads VisionEncoderSession once
  -> hybrid_streaming_decode loads llama.cpp/mmproj context once
  -> producer thread replays sampled frame timestamps
  -> OnDemandBufferUpdate replaces current frame pointer
  -> prompt job records current frame at arrival time
  -> consumer lane serially QNN-encodes the selected .bin frame
  -> eval_with_external_embedding() runs decoder-side mmproj and prefill
  -> generate_response() decodes with preserved chat/KV state
  -> output stream_events.csv, streaming_phase_stats.csv, per-turn token traces
  -> runner finalizes foundation_proc.csv, phase_duration_stacked_bar.png, and phase_timeline.png
```

Hybrid streaming QNN phase timing is rebased onto the llama.cpp `ggml_time_ms()`
timeline. This avoids mixing ExecuTorch absolute timestamps with llama.cpp
elapsed timestamps.

### Streaming Sliding-Window Hybrid

```text
run_android_hybrid_bridge.py --processor hybrid --streaming-video ... --stream-mode sliding-window
  -> runner.media.prepare_streaming_video_media
  -> push media_manifest.json, stream_frame_<idx>.png, stream_frame_<idx>.bin
  -> run hybrid_streaming_decode
  -> producer thread replays sampled frame timestamps
  -> prompt job selects frames with timestamp <= prompt time
  -> optional --window-sec filters to recent frames
  -> --window-max-frames evenly reduces the selected frame list when needed
  -> build prompt as Frame1/Frame2/... plus the user question
  -> QNN-encode all selected frame/tile bins
  -> eval_with_external_embedding() runs mmproj, image prefill, text prefill
  -> generate_response() decodes while preserving prior text turns
```

This mode is the sliding-window baseline. It preserves prompt-arrival selection
semantics and multi-turn chat/KV state, but it does not reuse image-prefix KV.
It is useful for measuring how much latency remains when each prompt receives a
bounded recent video clip while the text conversation continues.

### Streaming Vision-Prefill Hybrid

```text
run_android_hybrid_bridge.py --processor hybrid --streaming-video ... --stream-mode vision-prefill
  -> runner.media.prepare_streaming_video_media
  -> push media_manifest.json, stream_frame_<idx>.png, stream_frame_<idx>.bin
  -> run hybrid_streaming_decode
  -> producer thread replays sampled frame timestamps
  -> every frame arrival enqueues a CacheUpdate job
  -> CacheUpdate selects all sampled frames up to that frame
  -> if frame 0, reset decoder context and tokenize the chat-formatted prefix
  -> otherwise restore the previous cache snapshot and tokenize only global FrameN
  -> walk mtmd chunks in order
  -> for text chunks, run VisionPrefillT_Prefill
  -> for image chunks, QNN-encode only the newly appended frame bin on demand
  -> run VisionPrefillMmproj and VisionPrefillImagePrefill immediately
  -> with --partial-vision-kv, expose ImagePrefill progress after each
     --ubatch-size image micro-batch and allow prompt preemption after the
     current batch commits
  -> save seq 0 with llama_state_seq_get_data_ext()
  -> prompt job restores the matching snapshot with llama_state_seq_set_data_ext()
  -> tokenize/evaluate only the formatted question suffix that closes the open user turn
  -> decode the answer
  -> save the post-answer state so later frames become the next user turn
```

This mode is a KV-level cached image-prefill observation path, not a chunk
composition algorithm. The current cache is one active snapshot per sampled
frame, built incrementally from the previous snapshot plus one new frame. The
cache also stores closed chat history plus the open streaming user content, so
prompt events behave as true multi-turn chat while still avoiding repeated
vision prefill for already cached frames. It ignores `--window-sec` and
`--window-max-frames`, because P0 must be able to use all cached frames before
the question boundary. Future
`--chunked-vision-prefill` should add independently reusable chunks controlled
by `--chunk-count` instead of changing this active-snapshot streaming-turn mode.

The cache build is intentionally frame-ordered. Earlier prototypes encoded all
selected bins first, which produced traces with several consecutive
`VisionPrefillV_Encode` rows before any image-prefill work. The current helper
`eval_streaming_chunks_with_on_demand_vision()` QNN-encodes one frame/tile only
when its IMAGE chunk is reached, then immediately runs mmproj and image prefill
before advancing to the next chunk.

With `--partial-vision-kv`, image prefill is still ordered, but the active
image chunk may be committed in smaller pieces. A prompt arriving during image
prefill does not wait for the remaining image batches. It waits for the current
batch to finish, closes the image wrapper with the following text chunks, and
then evaluates the question suffix from the partial KV state. For the current
InternVL3 one-tile path, a 256-token visual image with `--ubatch-size 64` can
therefore answer from 64, 128, 192, or 256 committed visual KV tokens.

## Result Artifacts

Each run writes under:

```text
my_research/foundation_llamacpp/results/log/<model>_<backend>_ctx_<ctx>_kv<kv>/
```

New runner outputs normalize each run directory into three subfolders:

```text
csv/
  CSV phase, summary, event, and Android memory tables

png/
  matplotlib plots

txt_json/
  stdout, generated text, token traces, exit codes, manifests, and binary
  handoff/debug artifacts
```

Core artifacts:

```text
txt_json/foundation_exit_code.txt:
  runner-level exit code; must be 0 for success

csv/foundation_summary.csv:
  high-level backend, runtime, throughput, and memory summary

csv/foundation_proc.csv:
  normalized phase rows used for runtime plots

txt_json/foundation_output.txt:
  raw model output or decoder stdout copy

txt_json/foundation_token_io.txt:
  user/assistant text plus trace appendix when available

txt_json/foundation_inference_tokens.txt:
  detailed chunk/token trace; best file for checking media chunk count.
  Streaming vision-prefill includes the committed cached visual prefix,
  question suffix, and decode tokens in order. Partial image KV emits only the
  committed `<VISION_KV_SLOT N>` entries.

txt_json/run_command.txt:
  exact host command used to produce the sample

csv/android_memory_timeline.csv:
  sampled Android memory timeline

txt_json/memory_usage_summary.txt:
  post-processed memory summary

png/phase_duration_stacked_bar.png:
  common runtime duration stack when matplotlib is available. It uses the same
  visible labels as `phase_timeline.png` and keeps dynamic-KV breakdown
  sub-spans out of the aggregate bar.

png/memory_timeline_plot.png:
  memory timeline plot when matplotlib is available
```

Streaming artifacts:

```text
csv/stream_events.csv:
  frame enqueue, OnDemandBufferUpdate, prompt arrival, and prompt decode spans

csv/streaming_phase_stats.csv:
  setup, buffer update, vision, mmproj, prefill, and decode phase rows. New
  hybrid streaming runs include a `clock_origin_ms` comment so stdout
  `DynamicKVGrow` logs can be aligned to the same `ggml_time_ms()` source.

png/phase_timeline.png:
  common phase timeline plot. Image, multi-image, and offline-video runs use
  ready-relative time after bridge load/warmup; streaming runs use stream/video
  time, not first-prompt relative time.

DynamicKVGrow:
  synthetic dynamic KV expansion row inserted during finalization. New runs use
  the full grow/retry window from `llama_context::decode()`, including
  `grow_to()`, KV migration, scheduler reserve, and retry preparation. For
  OpenCL-backed KV tensors, migration should log device-to-device
  `clEnqueueCopyBuffer` usage instead of CPU snapshot round-tripping. The active
  main branch uses contiguous dynamic KV grow only; paged KV prototype code was
  reverted and should not be assumed present.

retry-side ImagePrefill:
  if an `ImagePrefill` row overlaps the grow window, finalization clips the row
  start to the grow window end. This keeps image prefill compute in the blue bar
  and grow/retry overhead in the black `DynamicKVGrow` bar.

stream_response_<idx>.txt:
  assistant response for one prompt event

stream_token_io_<idx>.txt:
  compact per-turn user/assistant and token appendix

stream_inference_tokens_<idx>.txt:
  raw per-turn mtmd/GGUF token trace. Vision-prefill prompt turns replay
  committed cached image slots here before the prompt suffix.

foundation_inference_tokens.txt:
  aggregate of all streaming turns with prompt headers

txt_json/stream_buffer_summary.txt:
  requested and observed input FPS, processed visual FPS, skipped/coalesced
  cache updates, and prompt-frame lag. This is especially useful for
  `--online-buffer` runs because input frame arrival and processing throughput
  are intentionally decoupled.
```

Legacy hybrid two-process artifacts:

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

New default hybrid media/streaming runs use `hybrid_streaming_stdout.txt` and
the common `foundation_*`, `stream_events.csv`, and `streaming_phase_stats.csv`
artifacts instead.

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
  --target hybrid_decode opencl_phase_mtmd opencl_streaming_decode hybrid_streaming_decode hybrid_vision_dump -j2
```

Use temporary host build directories for host compile checks. Do not rely on or
commit `build-hybrid-host`; it is obsolete generated output.

The CMake options are:

```text
HYBRID_BRIDGE_BUILD_LLAMA_DECODER:
  builds hybrid_decode, opencl_phase_mtmd, opencl_streaming_decode, and the
  llama.cpp side of hybrid_streaming_decode

HYBRID_BRIDGE_BUILD_EXECUTORCH_VISION:
  builds hybrid_vision_dump and links QNN support into hybrid_streaming_decode
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
6. Rebuild `hybrid_streaming_decode` and `opencl_streaming_decode`; rebuild
   `hybrid_decode`/`hybrid_vision_dump` only when touching the legacy split
   bridge.
7. Run at least one image or video smoke test when prompt/media layout changes.
```

When changing streaming:

```text
1. Keep video_file and streaming semantics separate. Video_file is offline
   sampled frames; streaming is timestamped replay with prompt events.
2. Preserve the explicit state model. On-demand is latest-frame state,
   sliding-window is bounded visual-window state with retained text/chat KV, and
   vision-prefill is incrementally saved/restored streaming-turn KV with
   retained closed chat history.
3. Keep prompt arrival timestamp, selected buffered frame, and actual execution
   start distinguishable in stream_events.csv.
4. Decoder context retention/eviction must remain explicit. On-demand,
   sliding-window, and vision-prefill preserve chat history and KV across prompt
   events; vision-prefill restores a cached open user-turn prefix, evaluates
   only the text suffix, then saves the post-answer state for later frames.
   Partial vision-prefill is a preemption policy inside this same state model:
   it may shorten the latest frame image KV, but it must not drop closed chat
   history or pretend uncommitted vision slots exist.
5. Keep OpenCL and Hybrid streaming artifacts aligned so their timelines can be
   compared.
6. If adding persistent prefill or vision-encoder-only streaming, add new mode
   flags instead of changing `--stream-mode on-demand` semantics silently.
7. Keep dynamic KV and paged KV language separate. Current main means
   contiguous dynamic KV grow with OpenCL device-to-device migration; paged KV
   must be reintroduced only through a new design/branch.
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
