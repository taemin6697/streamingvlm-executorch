# Vision Prefill KV Cache Design

## Goal

Implement `vision-prefill` as a real KV-level image prefill cache for streaming video observation. Prompt-time latency should skip repeated image prefill by restoring a cached llama sequence state for the full sampled video history available at prompt time, then evaluating only the text question suffix before decoding.

## Mode Semantics

- `context-window`: resets before each prompt and evaluates the selected video window and text prompt together.
- `vision-prefill`: builds a cached KV snapshot for every sampled frame using all sampled frames up to that frame. At prompt time, it restores that full-history snapshot and evaluates only the text suffix. It ignores `--window-sec` and `--window-max-frames`.
- Future mode flag: `--chunked-vision-prefill`. This will cache independently reusable chunks of 1, 2, or more frames, controlled by a future argument named `--chunk-count`.

The first implementation intentionally covers only `vision-prefill` full-history snapshots. It does not compose per-frame or per-chunk KV fragments.

## Architecture

The hybrid streaming bridge owns the decoder context, QNN vision encoder, and prompt loop. The cache therefore lives in `hybrid_streaming_decode.cpp`, next to prompt scheduling, because it must share the same llama context that later restores the KV state.

A `VisionPrefillCache` stores:

- the selected frame ids for the cached window,
- the image paths used for mtmd layout,
- the saved sequence state bytes from `llama_state_seq_get_data_ext`,
- the `n_past` position after the image/video prefix,
- the formatted prefix and full prompt boundary metadata needed to evaluate the text suffix.

For every sampled frame in `vision-prefill`, the bridge builds a cache for all sampled frames up to that frame. Cache construction evaluates only the formatted video prefix with image chunks and QNN embeddings. It then saves seq 0 KV state, clears runtime state, restores that KV state at prompt handling, evaluates the text suffix with `logits_last=true`, and decodes.

## Prompt Boundary

The cache is not built from raw prompt text. It is built from the same chat-template formatted string that the final prompt would use.

The formatted user message contains a sentinel between the video prefix and the question. The cache path tokenizes/evaluates the formatted text before the sentinel and all image chunks in that prefix. The prompt path restores the saved KV and tokenizes/evaluates the formatted text after the sentinel. This keeps image chunk positions, chat template tokens, and text suffix positions aligned with the non-cached prompt.

## Error Handling

If cache restore fails, `vision-prefill` falls back to the existing full prefill path for that prompt and records a cache miss. If a selected frame has no QNN bin or layout image, the prompt fails with the same error behavior as the existing hybrid path. Unsupported OpenCL-only `vision-prefill` keeps the current full-prefill behavior until a separate cache path is added.

## Observability

The phase CSV records cache-specific rows:

- `VisionPrefillCacheBuild`
- `VisionPrefillCacheSave`
- `VisionPrefillCacheRestore`
- `VisionPrefillCacheHit`
- `VisionPrefillCacheMiss`

A successful cached prompt should not emit prompt-path `ImagePrefill` rows for the restored images. Image encoding and image prefill work move to cache-build rows.

## Testing

Unit tests check that source-level contracts for `vision-prefill` cache are present: state save/restore APIs, cache phases, sentinel boundary handling, and the documented future `--chunked-vision-prefill` flag name. Build verification compiles `hybrid_streaming_decode`. Device verification runs 2B Q8 hybrid on Android and confirms prompt output plus cache-hit phase rows.
