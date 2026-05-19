# Streaming Research Refactor: Model Profiles, Policies, and KV Extension Points

Date: 2026-05-19

This note records the extensibility refactor for the hybrid streaming research
runtime. The goal is to keep the existing InternVL3 behavior intact while
making future Qwen2.5-VL, Gemma-style VLMs, and M-RoPE/KV-reposition work easier
to add without editing the middle of the scheduling loops.

## Problem

The streaming bridge had three different concerns mixed together:

```text
model-family prompt layout
  FrameN / Image-N labels and image-marker placement

streaming research policy
  on-demand, sliding-window, and vision-prefill frame selection

KV position rewrite semantics
  current 1D RoPE compaction versus future M-RoPE axis-aware shifting
```

That was manageable while InternVL3 was the only target. It becomes brittle when
adding Qwen2.5-VL or later models because prompt formatting and positional
semantics can change while the streaming scheduler should stay the same.

## Host Prompt Formatter

`runner/prompt_formats.py` is the Python-side prompt profile registry.

```python
normalize_prompt_format("internvl3") -> "internvl3"
normalize_prompt_format("qwen2.5-vl") -> "qwen2_5_vl"
get_prompt_formatter("internvl3").video_prompt(...)
```

The current profiles are:

```text
internvl3:
  default validated profile
  image:       <__media__>\nquestion
  multi-image: Image-1: <__media__>\nImage-2: <__media__>\nquestion
  video:       Frame1: <__media__>\nFrame2: <__media__>\nquestion

qwen2_5_vl:
  registered extension profile
  currently uses the same abstract mtmd media marker at the runner boundary
  future work can specialize marker/prefix behavior here
```

`runner/media.py` now accepts `prompt_format` in image, multi-image, video, and
streaming media preparation. It writes the selected profile name into
`media_manifest.json`:

```json
{
  "prompt_format": "internvl3"
}
```

`runner/cli.py` exposes the user-facing option:

```bash
--prompt-format internvl3
--prompt-format qwen2_5_vl
```

The default remains `internvl3`.

## Android Streaming Prompt Profile

`hybrid_bridge/streaming_prompt_format.hpp` mirrors the host profile boundary
inside the Android streaming binary.

Key types:

```cpp
enum class PromptFormatFamily {
  InternVL3,
  Qwen25VL,
};

struct PromptFormatProfile {
  std::string name;
  PromptFormatFamily family;
  std::string media_marker;
  std::string frame_prefix;
  std::string frame_separator;
  bool uses_mrope_positions;
};
```

`hybrid_streaming_decode.cpp` still keeps its existing helper function names,
but those helpers now delegate to `PromptFormatProfile`:

```cpp
build_stream_frame_prompt_line()
build_video_prompt_prefix()
strip_stream_video_prompt_prefix()
update_first_video_user_message()
build_video_prompt()
```

This matters for vision-prefill because later frame updates may rewrite the
first video user message while preserving the user/assistant chat tail. The
rewrite should use the active model profile instead of hard-coded `"Frame"` and
`": "` strings.

## Streaming Policy Boundary

`hybrid_bridge/streaming_policy.hpp` owns frame selection:

```cpp
StreamingPolicyConfig policy;
policy.stream_mode = "sliding_window";
policy.window_sec = 4.0;
policy.window_max_frames = 8;
select_prompt_frames(policy, available_frames, current_frame, prompt_time);
```

Current behavior is unchanged:

```text
on_demand:
  use the current/live frame only

sliding_window:
  select frames whose timestamps are inside the lookback window,
  then evenly reduce to window_max_frames

vision_prefill:
  use all available frames up to the prompt timestamp as the logical cache
  history, with actual reuse handled by the vision-prefill KV cache
```

Future policies such as chunked vision-prefill, retrieval, or compressed-window
scheduling should be added in this policy helper first, then called from the
consumer loop.

## KV Reposition Strategy Boundary

`hybrid_bridge/kv_reposition.hpp` now defines a small strategy marker:

```cpp
enum class KvPositionEncodingKind {
  Rope1D,
  MRope,
};

struct KvRepositionStrategy {
  KvPositionEncodingKind position_encoding;
  bool requires_k_shift_rebuild;
  bool supports_axis_aware_rewrite;
};
```

Current InternVL3 streaming compaction remains the 1D RoPE path:

```text
llama_memory_seq_rm()
llama_memory_seq_add()
llama.cpp K-shift re-applies RoPE to cached K
```

The M-RoPE placeholder is deliberately explicit:

```cpp
mrope_reposition_strategy_placeholder()
```

Future Qwen-style M-RoPE support should preserve per-token visual axis position
metadata, then use that metadata when inverse/re-applying cached K after KV
compression or frame-prefix insertion. The current refactor does not implement
M-RoPE shifting; it makes the boundary visible so the implementation can be
added without touching prompt formatting or stream scheduling.

## Regression Script

`scripts/run_refactor_regression_internvl_1b2b.sh` is the targeted Android
regression suite for this refactor.

It runs InternVL3 1B and 2B Q8 in the unified hybrid binary across:

```text
single image
multi-image
offline video
streaming on-demand
streaming sliding-window
streaming vision-prefill
```

The streaming runs enable the research features that are most likely to be
affected by this refactor:

```text
--online-buffer
--dynamic-kv-cache --kv-init-size 512 --kv-grow-step 512
--partial-vision-kv and --latest-frame-only for vision-prefill
--ubatch-size 64 for streaming
```

The script intentionally avoids model pushes. It verifies that model, mmproj,
and QNN vision artifacts already exist under the remote root. It also waits for
adb to recover before each run, which matches the current phone/server daemon
setup used in the experiments.

Example:

```bash
DATA_ROOT=/workspace/streamingvlm \
INTERNVL_1B_REMOTE_ROOT=/data/local/tmp/streamingvlm_post_merge_1b \
INTERNVL_2B_REMOTE_ROOT=/data/local/tmp/streamingvlm_post_merge_2b \
RESULTS_ROOT=/workspace/streamingvlm/my_research/foundation_llamacpp/results/log/refactor_regression_internvl_1b2b \
my_research/foundation_llamacpp/scripts/run_refactor_regression_internvl_1b2b.sh
```

When the script completes, each run directory should contain:

```text
csv/
png/
txt_json/run_command.txt
txt_json/foundation_exit_code.txt
txt_json/foundation_inference_tokens.txt
```

## Extension Checklist

When adding a new VLM family:

```text
1. Add a host prompt profile in runner/prompt_formats.py.
2. Add the Android profile in streaming_prompt_format.hpp.
3. Mark whether the family needs M-RoPE position metadata.
4. Keep stream-mode scheduling inside streaming_policy.hpp.
5. Add a regression matrix entry before merging.
```

When adding a new streaming research method:

```text
1. Put frame selection semantics in streaming_policy.hpp.
2. Keep prompt layout through PromptFormatProfile.
3. Keep KV rewrite behavior behind kv_reposition.hpp strategy helpers.
4. Verify image, multi-image, offline video, and all streaming modes still run.
```
