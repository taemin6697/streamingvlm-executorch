# Streaming Sliding-Window And Vision-Prefill Modes

This note documents the two non-native streaming observation modes added after
the original single-buffer streaming baseline:

- `--stream-mode sliding-window`
- `--stream-mode vision-prefill`

The code lives in the foundation llama.cpp runner and bridge:

```text
my_research/foundation_llamacpp/runner/media.py
my_research/foundation_llamacpp/runner/cli.py
my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp
my_research/foundation_llamacpp/tests/test_streaming_media.py
my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py
```

Both modes use `--streaming-video`, so the host still samples a video file into
timestamped frames and the Android bridge still replays those frames. The
difference is the state model used when a prompt arrives.

## Mode Summary

```text
single-buffer:
  latest image only
  multi-turn chat/KV state is preserved
  prompt latency includes prompt-time vision encode and image prefill

sliding-window:
  selected recent frames become one video clip
  multi-turn chat/KV state is preserved across prompts
  prompt latency includes full prompt-time vision encode and image prefill

vision-prefill:
  every frame arrival saves an active streaming-turn KV snapshot
  frame 0 is built from scratch; later frames append one new global FrameN to
  the previous restored snapshot
  prompt handling closes the open user turn, decodes, and saves post-answer KV
  multi-turn chat/KV state is preserved across prompt events
  prompt latency restores cached image-prefix KV and evaluates only text suffix
```

`sliding-window` is the sliding-window baseline. `vision-prefill` is the current
KV-level image-prefill cache path. It is not yet the chunk-composition algorithm
planned for future `--chunked-vision-prefill`.

Both modes can run with `--dynamic-kv-cache`. In current `main`, dynamic KV means
contiguous standard KV grow with OpenCL device-to-device K/V migration. Paged KV
is not active; the prototype was reverted to keep this observation path focused.

## Host-Side Media Preparation

The streaming path starts in `runner/media.py`.

```text
input video
  -> decord.VideoReader
  -> sample at --sampling-fps until optional --max-video-time
  -> write stream_frame_<idx>.png layout images
  -> for hybrid modes, write stream_frame_<idx>.bin QNN tensors
  -> write prompt_events from --time and JSON-list --prompt
  -> write media_manifest.json
```

The manifest records:

```text
source_kind: streaming_video
stream_mode: single_buffer | sliding_window | vision_prefill
sampling_fps
duration_s
effective_duration_s
window_sec
window_max_frames
frames[]
prompt_events[]
```

`single-buffer` can use only layout images for OpenCL, but the hybrid streaming
modes need both PNG layout images and normalized CHW float32 `.bin` inputs for
ExecuTorch/QNN vision encoding.

## Runner CLI

`runner/cli.py` owns the public CLI contract:

```text
--single-buffer
  Backward-compatible alias for --stream-mode single-buffer.

--stream-mode single-buffer|sliding-window|vision-prefill
  Selects streaming state semantics.

--window-sec SEC
  Lookback window used only by sliding-window.

--window-max-frames N
  Maximum frame count used only by sliding-window.
```

`vision-prefill` intentionally ignores `--window-sec` and
`--window-max-frames`. It uses all sampled frames up to each cache-update frame.

Result directories include the stream mode for non-single-buffer runs:

```text
<model>_hybrid_ctx_4096_streaming_sliding_window_kv16
<model>_hybrid_ctx_4096_streaming_vision_prefill_kv16
```

## Android Bridge Scheduling

`hybrid_streaming_decode.cpp` uses one producer thread and one serialized
consumer lane.

The producer replays sampled frame timestamps. For each frame:

```text
StreamFrameEnqueue
SingleBufferUpdate
```

In `vision-prefill`, the producer also enqueues a `CacheUpdate` job for every
sampled frame. Prompt events are enqueued when their timestamp is reached.

The consumer lane processes jobs in queue order. This means prompt execution can
wait behind older cache builds or an earlier prompt decode. The selected
frame/window is still captured at the prompt's stream timestamp.

## Sliding-Window Details

`sliding-window` is a multi-turn video-window run. Each prompt receives a bounded
recent visual window, while the text conversation and decoder KV continue across
prompt events.

Frame selection:

```text
available_frames = all sampled frames seen so far
eligible = frames with timestamp <= prompt.timestamp_s
if --window-sec is set:
  eligible = frames with timestamp >= prompt.timestamp_s - window_sec
if len(eligible) > --window-max-frames:
  eligible = evenly sampled down to window_max_frames
```

The selected frames are formatted as a video prompt:

```text
Frame1: <__media__>
Frame2: <__media__>
...
<user question>
```

For every prompt, it performs the normal full visual prefill path:

```text
QNN V_Encode for selected frame bins
Mmproj
ImagePrefill
T_Prefill
D / Decode
```

The purpose is to measure a bounded recent video clip while preserving
multi-turn text state, but without reusing image-prefix KV.

Validated multi-turn behavior:

```text
prompt 0 @ 5s: What is this situation?
prompt 1 @ 8s: What did I ask earlier???

response to prompt 1:
  You asked about the activity in the video.
```

That run used 2B Q8 hybrid, `--dynamic-kv-cache`, `--kv-init-size 512`,
`--kv-grow-step 512`, `--window-sec 4.0`, and `--window-max-frames 8`. Because
sliding-window now preserves text/chat KV, dynamic KV grew from `512` to `5632`
cells over the four-prompt test rather than stopping at the earlier singleton
capacity.

## Vision-Prefill Details

`vision-prefill` is a hybrid-only incremental streaming-turn KV snapshot cache.

For every sampled frame, the bridge saves a cache representing the decoder state
after all sampled frames up to that frame have been consumed. Before the first
prompt, those frames are inside one open user turn:

```text
frame 0 cache: build [frame 0] from scratch
frame 1 cache: restore frame 0 cache, append [frame 1]
frame 2 cache: restore frame 1 cache, append [frame 2]
...
```

At prompt time, the prompt uses the cached snapshot matching the frame history
available at prompt arrival. The prompt text is evaluated as the suffix that
closes the open user turn. After decode, the user question and assistant answer
are stored in chat history and the closed post-answer state is saved. Later
frame arrivals append to a new open user turn, so this mode is multi-turn rather
than singleton.

The cache object stores:

```text
valid
frame_indices
images
state bytes from llama_state_seq_get_data_ext()
state_flags
n_past
chat_history
open_user_content
open_user_prefix
```

The cache path uses a sentinel to preserve the exact chat-template boundary:

```text
Frame1: <__media__>
Frame2: <__media__>
...
<SVLM_QUESTION_SENTINEL>
<user question>
```

The bridge formats this as the normal user message for the current chat history,
then splits the formatted string at `SVLM_QUESTION_SENTINEL`.

Cache build evaluates only the formatted prefix before the sentinel. Prompt
handling restores the saved sequence state and evaluates only the formatted
suffix after the sentinel.

## KV Save And Restore

Cache save:

```text
llama_state_seq_get_size_ext(ctx.lctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)
llama_state_seq_get_data_ext(ctx.lctx, state.data(), state.size(), 0, flags)
cache.n_past = ctx.n_past
```

Cache restore:

```text
reset_decode_context_for_singleton(ctx)
llama_state_seq_set_data_ext(ctx.lctx, cache.state.data(), cache.state.size(), 0, flags)
ctx.n_past = cache.n_past
ctx.chat_history = cache.chat_history
```

If the cache does not match or restore fails, the prompt records
`VisionPrefillCacheMiss` and falls back to the normal full prefill path.

## Frame-Ordered Cache Build

The current implementation intentionally builds the cache in token/chunk order.
This is important for the timeline.

The earlier implementation called:

```text
encoder.encode(bins)
```

for every selected bin first, then evaluated the full tokenized prefix. That
made traces look like several `VisionPrefillV_Encode` rows in a row, followed
later by image prefill rows.

The current implementation uses:

```text
eval_streaming_chunks_with_on_demand_vision()
```

It walks mtmd chunks in order:

```text
text chunk:
  VisionPrefillT_Prefill

image chunk:
  encoder.encode({bins[image_chunk_idx]})
  VisionPrefillV_Encode
  VisionPrefillMmproj
  VisionPrefillImagePrefill
  image_chunk_idx += 1
```

For incremental cache updates, each cache build after frame 0 first restores the
previous snapshot (`VisionPrefillCacheAppendRestore`), then evaluates only the
new frame's `FrameN:` text and image chunk.

```text
Frame label text prefill
frame 0 V_Encode
frame 0 Mmproj
frame 0 ImagePrefill
newline / next label text prefill
frame 1 V_Encode
frame 1 Mmproj
frame 1 ImagePrefill
...
```

One cache build completes before the next cache build or prompt job runs.

## Phase Rows

CSV rows include:

```text
VisionPrefillLayoutTokenize
VisionPrefillT_Prefill
VisionPrefillImageLoad
VisionPrefillV_Encode
VisionPrefillMmproj
VisionPrefillImagePrefill
VisionPrefillCacheSave
VisionPrefillCacheBuild
VisionPrefillCacheRestore
VisionPrefillCacheHit
VisionPrefillCacheMiss
```

Timeline plotting aliases these compute rows onto the standard lanes:

```text
VisionPrefillV_Encode      -> V_Encode
VisionPrefillMmproj        -> Mmproj
VisionPrefillImagePrefill  -> ImagePrefill
VisionPrefillT_Prefill     -> T_Prefill
```

Cache management rows remain in CSV but are hidden from the PNG timeline.
`SingleBufferUpdate` stays visible as a vertical tick marker so frame arrivals
can still be inspected.

## Validation

Repository checks:

```bash
pytest my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py \
  my_research/foundation_llamacpp/tests/test_streaming_media.py -q

python3 -m compileall my_research/foundation_llamacpp/runner \
  my_research/foundation_llamacpp/tests

cmake --build my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --target opencl_streaming_decode hybrid_streaming_decode -j2
```

Validated Android run:

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

Result:

```text
results/log/vision_prefill_kv_cache_2b_hybrid_frame_ordered/
  InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_vision_prefill_kv16
```

Older full-rebuild observation:

```text
foundation_exit_code.txt = 0
wall_time_s = 224.41
VisionPrefillCacheBuild = 11
VisionPrefillCacheHit = 2
VisionPrefillV_Encode = 66
VisionPrefillImagePrefill = 66
bad_consecutive_vencode = 0
```

The `66` vision/image-prefill rows correspond to full-history cache builds over
eleven frames:

```text
1 + 2 + 3 + ... + 11 = 66
```

Current incremental observation:

```text
result = results/log/stream_modes_2b_hybrid_dynamic512_npredict64/InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_vision_prefill_kv16_dynamic
foundation_exit_code.txt = 0
VisionPrefillCacheBuild = 16
VisionPrefillCacheAppendRestore = 15
VisionPrefillCacheHit = 4
VisionPrefillCacheRestore = 4
VisionPrefillV_Encode = 16
VisionPrefillImagePrefill = 16
DynamicKVGrow = 8
```

The `16` vision/image-prefill rows correspond to one newly appended frame per
sampled frame, not `1 + 2 + ... + 16`.

Interleaved multi-turn observation:

```text
result = results/log/red_panda_vision_prefill_multiturn_interleaved_2b_dynamic512_frame1/InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_vision_prefill_kv16_dynamic
foundation_exit_code.txt = 0
VisionPrefillCacheBuild = 15
VisionPrefillCacheAppendRestore = 14
VisionPrefillCacheHit = 4
VisionPrefillCacheMiss = 0
VisionPrefillCacheRestore = 4
VisionPrefillV_Encode = 15
VisionPrefillImagePrefill = 15
DynamicKVGrow = 0
```

Prompt 1 in that run asked `What did I ask earlier???` and the model answered
that the previous question was about the red panda's activity. This confirms the
cached vision-prefill path now carries text chat history across prompt events.

Additional closure validation:

```text
pytest my_research/foundation_llamacpp/tests/test_dynamic_kv_device_copy_contract.py \
  my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py \
  my_research/foundation_llamacpp/tests/test_streaming_media.py -q

cmake --build my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --target hybrid_streaming_decode -j2
```

The tracked main branch no longer contains paged-KV code or docs as active
behavior. Future paged-KV or true KV compression work should start from a new
branch/spec and should not silently change the semantics of these streaming
modes.

## Future Chunked Vision Prefill

The planned mode name is:

```text
--chunked-vision-prefill
```

The planned chunk-size argument is:

```text
--chunk-count
```

This should create independently reusable KV chunks, for example one frame per
chunk or two frames per chunk. That future mode should not silently change the
current `vision-prefill` semantics, which maintain one active streaming-turn
snapshot at each frame by incrementally appending the newest frame and saving
post-answer chat/KV state at prompt boundaries.
