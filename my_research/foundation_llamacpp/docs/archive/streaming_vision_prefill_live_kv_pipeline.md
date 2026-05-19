# Streaming Vision-Prefill Live KV Pipeline

Date: 2026-05-19

This note summarizes the current `main` implementation of the live streaming
vision-prefill path after the recent refactor and multi-turn fixes. It is meant
as a code-level handoff document: where the behavior lives, what each option
changes, and how the latest 448x448 red-panda validation was run.

Related detailed notes:

```text
docs/archive/streaming_sliding_window_and_vision_prefill.md
docs/archive/partial_vision_prefill_kv.md
docs/archive/kv_rope_reposition_for_video_compression.md
docs/archive/streaming_research_refactor_model_policy_kv.md
```

## Goal

The target behavior for `--stream-mode vision-prefill` is:

```text
frames arrive over time
  -> cache worker keeps a KV snapshot for the video prefix
  -> prompt restores the newest committed snapshot
  -> prompt only prefills the text suffix and decodes
  -> later frames are inserted into the same first video user turn
  -> previous user/assistant chat turns remain visible
```

The important design decision is that the visual region and the later chat
region are treated as different logical areas:

```text
first user turn:
  Frame1: <image>
  Frame2: <image>
  ...
  FrameN: <image>
  first question

assistant turn:
  first answer

later user turns:
  follow-up questions only
```

When a new frame arrives after the first prompt has already closed the first
user turn, the runtime updates the first video user message with the new frame
prefix and preserves the closed chat tail.

## Public Options

The current live vision-prefill experiment usually combines these flags:

```text
--stream-mode vision-prefill
--online-buffer
--latest-frame-only
--partial-vision-kv
--dynamic-kv-cache --kv-init-size 512 --kv-grow-step 512
--ubatch-size 64
```

Their meanings are separate:

```text
--stream-mode vision-prefill
  Build and reuse decoder KV snapshots for the visual video prefix.

--online-buffer
  Use live streaming semantics. A job that starts late uses the frame state
  available at processing time, not a stale frame selected when the event was
  originally enqueued.

--latest-frame-only
  If the cache worker is busy while newer frames arrive, stale cache-update
  work is dropped. After the current job finishes, the next cache update uses
  the latest newly available frame.

--partial-vision-kv
  If a prompt arrives while image prefill is running, finish the current
  `--ubatch-size` image micro-batch, commit those visible vision KV tokens, and
  answer immediately from that partial cache.

--dynamic-kv-cache
  Start with a smaller KV capacity and grow the contiguous OpenCL KV buffer
  when needed. Current main uses device-to-device K/V migration for grow.
```

`--kv-reposition-keep-latest-frames N` is a separate compression experiment. It
is not required for the normal live vision-prefill runs. The live red-panda
validation below used `N=0`.

## Host-Side Changes

### `runner/prompt_formats.py`

This file defines model-family prompt profiles. `internvl3` remains the
validated default. `qwen2_5_vl` exists as an extension point so future Qwen
support can change prompt formatting without editing the streaming scheduler.

The runner now talks in an abstract media marker:

```text
<__media__>
```

The Android/mtmd side resolves that marker into the model-specific image chunk
layout.

### `runner/media.py`

The media pipeline prepares all input modes for the unified binary:

```text
image
multi-image
offline video
streaming video
```

For streaming video it:

```text
1. samples frames at --sampling-fps
2. writes PNG layout images for mtmd/image-token bookkeeping
3. writes QNN tensor .bin files for hybrid vision encoding
4. writes prompt_events into media_manifest.json
5. records prompt_format and stream_mode in the manifest
```

### `runner/cli.py`

The runner forwards the streaming research flags to
`hybrid_streaming_decode`:

```text
--stream-mode
--online-buffer
--latest-frame-only
--partial-vision-kv
--kv-reposition-keep-latest-frames
--prompt-format
```

It also writes run artifacts into the grouped layout:

```text
csv/
png/
txt_json/
```

`txt_json/run_command.txt` records the exact command for each run.

## Android Bridge Changes

### `hybrid_bridge/streaming_prompt_format.hpp`

This is the Android-side prompt profile boundary. The streaming bridge calls
helper functions instead of hard-coding InternVL strings in the middle of cache
scheduling:

```text
build_stream_frame_prompt_line()
build_video_prompt_prefix()
build_stream_video_prompt_prefix()
strip_stream_video_prompt_prefix()
update_first_video_user_message()
build_video_prompt()
```

For InternVL3, the active video prefix shape is:

```text
Frame1: <image>
Frame2: <image>
...
```

`update_first_video_user_message()` is the important multi-turn hook. When the
cached frame list grows, it rewrites only the video prefix inside the first user
message and keeps the original first question text.

### `hybrid_bridge/streaming_policy.hpp`

Frame selection lives here, not in prompt formatting:

```text
on-demand:
  latest/current frame only

sliding-window:
  recent bounded video window

vision-prefill:
  logical frame history comes from committed cache state
```

This split keeps future policies such as chunked prefill, retrieval, or
compressed frame windows out of the prompt formatter.

### `hybrid_bridge/kv_reposition.hpp`

This file owns KV position rewrite helpers and future strategy markers.

For the current InternVL3 path, 1D RoPE shifting is delegated to llama.cpp:

```text
llama_memory_seq_add()
llama_memory_seq_rm()
llama.cpp K-shift update
```

When a new frame must be inserted before a closed chat tail, the bridge builds a
tail insertion plan:

```text
old sequence:
  video prefix | first question + assistant + later chat tail

insert:
  video prefix + new frame | first question + assistant + later chat tail

operation:
  seq_add shifts the tail forward
  llama.cpp re-applies RoPE to cached K for the shifted tail
  the opened gap is filled by the new frame text/image KV
```

This is different from the future `keep_latest_frames` compression path. Normal
live vision-prefill uses the insertion part to keep the official video
multi-turn layout without replaying the unchanged chat tail.

M-RoPE is not implemented yet. `kv_reposition.hpp` exposes the placeholder
boundary so Qwen-style visual-axis position metadata can be added later without
rewriting stream scheduling.

### `hybrid_bridge/hybrid_streaming_decode.cpp`

This file still owns the end-to-end streaming state machine.

The main state object for cached vision-prefill is `VisionPrefillCache`. It
stores:

```text
frames / frame_indices / images
n_past
saved decoder state
host_state for dynamic KV restore
chat_history
open_user_prefix state
video_prefix_insert_pos
prefill token trace sections
per-frame VisionKvSpan records
```

The cache build path has three broad cases:

```text
1. First cache build before any prompt:
   build full open video user prefix from Frame1..FrameN.

2. Incremental append while the first user turn is still open:
   restore previous cache and append the next frame prefix directly.

3. Incremental append after the first prompt closed the video user turn:
   restore cache, shift the closed tail forward, fill the opened gap with the
   new frame prefix/image KV, update the first video user message, and save the
   new cache snapshot.
```

The prompt path is:

```text
1. restore newest committed VisionPrefillCache
2. if the video prefix is still open, close it with the current question
3. otherwise format the prompt as a normal follow-up user turn
4. T_Prefill only the text suffix
5. decode
6. append user/assistant messages to chat_history
7. save post-answer cache state
```

That gives the intended token-level shape for prompt 1 and later:

```text
<|im_start|>user
Frame1: <image>
Frame2: <image>
Frame3: <image>
What is the red panda doing?
<|im_end|>
<|im_start|>assistant
Eating from a stick.
<|im_end|>
<|im_start|>user
In the conversation history above, what was the user first question?
<|im_end|>
<|im_start|>assistant
```

## Partial Vision KV

Partial image prefill is implemented by decoding image chunks with batch-level
progress callbacks in the mtmd helper path. The runtime uses `--ubatch-size` as
the visible commit unit.

For InternVL3 one 448x448 one-tile frame normally maps to 256 vision tokens.

With `--ubatch-size 64`:

```text
batch 0 committed ->  64 visible vision KV tokens
batch 1 committed -> 128 visible vision KV tokens
batch 2 committed -> 192 visible vision KV tokens
batch 3 committed -> 256 visible vision KV tokens
```

If a prompt arrives during batch 1, batch 1 is allowed to finish and the cache
contains only the tokens that reached KV. The token trace prints only the
visible slots:

```text
<VISION_KV_SLOT 1>
...
<VISION_KV_SLOT 128>
```

It does not print fake slots up to 256.

## Online Buffer And Latest-Frame-Only

The live scheduling rule is:

```text
if worker is idle when a new frame arrives:
  start work for that new frame

if worker is busy when new frames arrive:
  do not queue every stale frame
  remember the newest frame
  after current work finishes, start from that newest frame
```

For prompt preemption, the rule is different:

```text
if prompt arrives during image prefill:
  finish the current image micro-batch
  commit partial KV
  stop the rest of that frame
  answer the prompt first
```

So stale background frames can be dropped, but a prompt can still consume the
current partial micro-batch because TTFT is the priority.

`stream_buffer_summary.txt` records both input and processing rates:

```text
observed_input_fps
processed_visual_jobs
processed_visual_fps
committed_cache_updates
committed_cache_fps
skipped_cache_updates
latest_frame_only_dropped_cache_updates
prompt_frame_lag_s_avg
```

## Timeline Artifacts

All current runs write:

```text
csv/streaming_phase_stats.csv
csv/stream_events.csv
png/phase_timeline.png
png/phase_duration_stacked_bar.png
txt_json/foundation_output.txt
txt_json/foundation_inference_tokens.txt
txt_json/stream_buffer_summary.txt
txt_json/run_command.txt
```

The timeline lanes are normalized around:

```text
V_Encode
Mmproj
ImagePrefill
T_Prefill
DynamicKVGrow
KVRepositionTailShift
KVRepositionCompact
Decode
```

For `--partial-vision-kv`, `ImagePrefill` can appear as multiple micro-batches.
For `--latest-frame-only`, gaps are expected when stale frame work is dropped.

## 448x448 Red-Panda Validation

The user requested a 448x448 streaming validation with InternVL3 1B and 2B
vision-prefill. The source video was downloaded from the InternVL example and
converted to:

```text
my_research/foundation_llamacpp/sample_images/red-panda_448x448.mp4
```

Validation output root:

```text
my_research/foundation_llamacpp/results/log/red_panda_448_vision_prefill_1b2b
```

Common run settings:

```text
--stream-mode vision-prefill
--online-buffer
--latest-frame-only
--partial-vision-kv
--sampling-fps 1.0
--max-video-time 20.0
--max-num 1
--time '[5.0, 8.0, 11.0, 14.0]'
--n-predict 64
--ctx-size 32768
--batch-size 1024
--ubatch-size 64
--dynamic-kv-cache --kv-init-size 512 --kv-grow-step 512
--cache-type-k f16 --cache-type-v f16
```

Observed status:

```text
InternVL3-1B-Instruct-Q8_0:
  foundation_exit_code.txt = 0
  input_frame_count = 21
  processed_visual_jobs = 15
  committed_cache_updates = 11

InternVL3-2B-Instruct-Q8_0:
  foundation_exit_code.txt = 0
  input_frame_count = 21
  processed_visual_jobs = 10
  committed_cache_updates = 6
```

The 2B run showed the healthier multi-turn behavior:

```text
prompt 0:
  user: What is the red panda doing?
  assistant: Eating from a stick.

prompt 1:
  user: In the conversation history above, what was the user first question?
  assistant: What is the red panda doing?

prompt 2:
  user: What changed in the scene?
  assistant: The red panda on the stick moved its head.

prompt 3:
  user: Summarize the full situation so far.
  assistant: Two red pandas are in an enclosure. One is eating from a stick,
             and the other is standing on its hind legs.
```

The 1B run also completed, but its final summary mixed in an incorrect
"black and white panda" phrase. That should be treated as model quality/noise,
not a streaming runtime failure, because prompt 1 still recovered the previous
question from chat history.

## Current Limitations

1. M-RoPE KV reposition is not implemented.
   The strategy boundary exists, but Qwen-style axis-aware visual position
   metadata still needs a real implementation.

2. `--kv-reposition-keep-latest-frames` is still experimental.
   It can remove old frame vision spans and shift later KV positions, but it is
   separate from the default live vision-prefill path.

3. Exact equivalence to full re-prefill is not guaranteed after KV shifts.
   llama.cpp can re-apply RoPE to cached K when positions shift, but hidden
   states in already-computed tail tokens were originally produced under the
   old context. This is acceptable for the current streaming observation work,
   but future compression should validate quality against a full-reprefill
   reference.

4. 1B answers can be visually weaker than 2B on the red-panda stream.
   Use 2B or 8B when judging whether the pipeline semantics are correct.
