# Streaming Video Modes Design

## Goal

Extend `--streaming-video` beyond the current latest-frame baseline so experiments can compare native single-frame streaming, sliding-window streaming, and cached-vision-prefill-style streaming with the same Android runner and artifacts.

## Modes

- `single-buffer`: existing baseline. Each prompt uses the latest sampled frame as one image.
- `sliding-window`: singleton baseline. Each prompt resets decoder/chat state, selects recent sampled frames, formats them as an InternVL-style video prompt, then runs full vision encode, image prefill, text prefill, and decode after the prompt.
- `vision-prefill`: singleton baseline. Each prompt resets decoder/chat state and uses multiple selected frames in the same InternVL-style video prompt. The first implementation keeps the selection/manifest/runner interface separate so frame-level precompute or KV-level prefill reuse can be added without changing CLI shape.

`sliding-window` and `vision-prefill` both use:

```text
Frame 1: <__media__>
Frame 2: <__media__>
...
<question>
```

## CLI

Add `--stream-mode {single-buffer,sliding-window,vision-prefill}`. Keep `--single-buffer` as a backward-compatible alias for `--stream-mode single-buffer`.

Add frame selection controls:

- `--window-sec`: prompt-time lookback window in seconds.
- `--window-max-frames`: maximum frames selected for one prompt. If more frames are available, choose an even temporal subset.

## Architecture

Python media prep remains responsible for sampling frames and writing `media_manifest.json`. The manifest records the selected stream mode and window controls.

The C++ streaming runner changes prompt jobs from one `FrameRecord` to a vector of `FrameRecord`. The producer still replays sampled frames over stream time, but when a prompt event fires it selects frames according to the mode. Prompt execution is serial.

Decoder state behavior:

- `single-buffer`: keep existing multi-turn behavior.
- `sliding-window`: reset before each prompt.
- `vision-prefill`: reset before each prompt.

This gives clean singleton latency baselines for the two video modes while preserving the original native streaming baseline.

## Testing

Add pure Python tests for stream-mode normalization, recent-window frame selection, max-frame downsampling, and video prompt text generation. Build-level verification should compile the hybrid bridge or at least compile-check the touched C++ target when Android toolchains are available.
