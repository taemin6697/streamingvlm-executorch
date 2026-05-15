# Unified Streaming Baseline Design

## Goal

Build a baseline where hybrid image, multi-image, video, and streaming runs use the same `hybrid_streaming_decode` execution surface, with streaming mode names cleaned up and an optional latest-only `--online-buffer` mode.

## User-Facing Contract

- `--stream-mode on-demand` is the canonical name for the old single-buffer streaming baseline.
- `--stream-mode single-buffer` and `--single-buffer` remain accepted aliases for older commands.
- `--multi-image img1 img2 ...` is the canonical multi-image argument.
- `--images img1 img2 ...` remains accepted as a deprecated alias.
- `--online-buffer` enables latest-only stream buffering:
  - frame input continues at the configured sampling FPS;
  - the worker stores only the latest frame snapshot for delayed work;
  - queued stale cache-update jobs are coalesced;
  - prompt jobs choose frames at processing start time, not at request timestamp.

## Unified Binary Execution

Hybrid media runs use one Android binary:

- image: `hybrid_streaming_decode --media-mode image`
- multi-image: `hybrid_streaming_decode --media-mode multi-image`
- offline video: `hybrid_streaming_decode --media-mode video`
- streaming on-demand/sliding-window/vision-prefill: `hybrid_streaming_decode --media-mode streaming`

The old `hybrid_vision_dump + hybrid_decode` path remains in source for comparison, but the Python runner no longer uses it for the default hybrid media flow.

## Prompt Formatting

Prompt construction is routed through named model-family helpers. The first implementation is `internvl3`, preserving:

- image: `<image>\n{prompt}`
- multi-image: `Image-1: <image>\nImage-2: <image>\n{prompt}`
- video/streaming clips: `Frame1: <image>\nFrame2: <image>\n{prompt}`

Adding Qwen/Gemma later should require adding a formatter, not rewriting media preparation or streaming scheduling.

## Metrics

Streaming runs write `stream_buffer_summary.txt` with:

- requested input FPS from the manifest;
- observed input FPS from enqueue events;
- processed visual job count;
- average processed visual FPS;
- skipped/coalesced cache updates;
- prompt frame lag values.

## Validation Matrix

Run the 1B Q8 hybrid script over:

1. image
2. multi-image
3. video
4. streaming on-demand
5. streaming sliding-window
6. streaming vision-prefill
7. streaming vision-prefill + dynamic KV
8. streaming vision-prefill + dynamic KV + online-buffer

