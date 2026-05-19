# KV/RoPE Repositioning for Future Video Compression

This note records the first implementation step for compacting cached video
history without re-prefilling the unchanged suffix.

## Motivation

The streaming vision-prefill path can keep adding visual KV as new frames
arrive. Future video compression will reduce an older visual region, for
example replacing a 128-token or 256-token frame span with a shorter compressed
representation. After that rewrite, every later token should move to a smaller
logical position. Re-prefilling all later user/assistant text and later frames
would be too expensive, so the cache needs a KV-level reposition operation.

## Existing llama.cpp Mechanism

llama.cpp already exposes the needed primitive for one-dimensional RoPE models:

```cpp
llama_memory_seq_rm(mem, seq_id, p0, p1);
llama_memory_seq_add(mem, seq_id, p0, p1, delta);
```

`seq_rm` removes a logical position range from the sequence. `seq_add` changes
the logical position metadata of remaining tokens. Internally, changing
positions marks KV cells as shifted. On the next memory update/decode,
`llama_kv_cache::build_rope_shift()` applies the corresponding RoPE delta to
the cached K tensor.

The important split is:

```text
cached K
  position-dependent for RoPE models
  must be inverse/re-applied when logical positions change

V cache
  not RoPE-rotated in the standard decoder attention path
  can remain as value vectors while positions are shifted
```

So the first implementation should not manually rewrite all KV bytes. It should
change logical positions and let llama.cpp perform its cached K RoPE shift.

## Added Helper

`my_research/foundation_llamacpp/hybrid_bridge/kv_reposition.hpp` adds a small
policy-free helper:

```cpp
KvTailCompactionPlan plan;
std::string error;
build_tail_compaction_plan(KvTokenRange{128, 384}, 1024, &plan, &error);
apply_tail_compaction_plan(llama_get_memory(ctx), 0, plan, &error);
```

For the example above:

```text
remove positions:       [128, 384)
tail before compact:    [384, 1024)
tail shift:             -256
new logical end:        768
```

This is exactly the shape needed after dropping an old visual span. For true KV
compression where `[128, 384)` becomes, for example, 32 compressed tokens, the
caller first materializes or preserves the compressed replacement at
`[128, 160)`, then uses:

```cpp
build_rewrite_compaction_plan(KvTokenRange{128, 384}, 32, 1024, &plan, &error);
```

That removes `[160, 384)` and shifts the tail by `-224`.

## Host Probe

`my_research/foundation_llamacpp/hybrid_bridge/kv_reposition_probe.cpp` is a
small host-only validation binary. It compares two paths:

```text
reference:
  prefill prefix + history + suffix
  generate greedily

KV reposition:
  prefill prefix + removed + history
  remove [prefix_end, removed_end)
  shift history tail by -removed_len
  prefill the same suffix at the compacted position
  generate greedily
```

This intentionally avoids sampling from stale logits produced before the shift.
The first generated token comes after a fresh suffix prefill, so it exercises
the shifted cached K/V as context.

Observed host runs:

```text
InternVL3-1B-Instruct-Q8_0:
  top1_match=true
  reference_answer: The user asked about alpha.
  shifted_answer:   The user asked about alpha.
  logits_rms_delta=0.438441459

InternVL3-2B-Instruct-Q4_K_M:
  top1_match=true
  reference_answer: The user previously asked about alpha.
  shifted_answer:   The user previously asked about alpha.
  logits_rms_delta=0.473699338
```

The answers matched in these probes, but the logits were not identical. That is
expected: RoPE shift fixes positions for cached K, but any already-prefilled
tail token was originally computed while attending to the removed span. Exact
equivalence to a compacted re-prefill is therefore not guaranteed when the
unchanged tail was computed under the old context. This path should be treated
as a practical KV-level approximation unless the compressed/repositioned region
is before tokens whose hidden states are acceptable to preserve.

## Streaming Vision-Prefill Integration

The helper is now wired into the hybrid streaming vision-prefill path as an
explicit experiment:

```text
--kv-reposition-keep-latest-frames N
```

When this option is positive, every committed image prefill records the actual
decoder KV range occupied by that frame's vision tokens:

```cpp
struct VisionKvSpan {
  int frame_index;
  llama_pos begin;
  llama_pos end;
};
```

After a cache update commits, `compact_vision_prefill_cache_frames()` keeps the
latest `N` frame indices and removes older vision-KV spans. For each removed
span it builds a tail compaction plan, calls `llama_memory_seq_rm()` for the
removed range, calls `llama_memory_seq_add()` to shift the later cached tokens,
updates `ctx.n_past`, then saves the compacted cache snapshot. The removed
range is vision KV only; closed user/assistant text KV is preserved and shifted.

This is intentionally separate from the frame scheduling policy:

```text
--online-buffer
  pick the live latest frame at processing start

--latest-frame-only
  drop stale cache-update jobs while the worker is busy

--partial-vision-kv
  let prompt preemption commit the current image micro-batch

--kv-reposition-keep-latest-frames N
  after commits, physically remove older frame vision-KV positions from seq 0
```

`streaming_phase_stats.csv` records `KVRepositionCompact` rows. The token trace
also appends notes such as:

```text
## KV_REPOSITION_COMPACT removed_frames=1 removed_vision_tokens=256 keep_latest_frames=4
```

`stream_buffer_summary.txt` records:

```text
kv_reposition_keep_latest_frames
kv_reposition_compactions
kv_reposition_removed_frames
kv_reposition_removed_tokens
```

Observed Android runs on the 20 s surveillance stream with
`--online-buffer --latest-frame-only --partial-vision-kv
--kv-reposition-keep-latest-frames 4`:

```text
InternVL3-1B-Instruct-Q8_0:
  exit_code=0
  committed_cache_updates=15
  kv_reposition_compactions=11
  kv_reposition_removed_tokens=2816
  result:
    my_research/foundation_llamacpp/results/log/
    kv_rope_reposition_streaming_compact_1b_keep4_fixed/

InternVL3-2B-Instruct-Q8_0:
  exit_code=0
  committed_cache_updates=7
  kv_reposition_compactions=3
  kv_reposition_removed_tokens=768
  result:
    my_research/foundation_llamacpp/results/log/
    kv_rope_reposition_streaming_compact_2b_keep4/
```

## Boundaries and Limitations

This helper does not decide which frames or visual tokens to compress. It only
applies the position compaction once a compression policy has decided the
logical token range.

This helper does not physically shrink the KV allocation. It reduces active
logical positions and frees the removed range from the sequence metadata. A
separate memory compaction or dynamic-KV shrink path is needed if the goal is to
return OpenCL/driver memory to the system immediately after compression.

Current llama.cpp `seq_add` and `seq_div` are guarded by
`n_pos_per_embd() == 1`. That covers the InternVL/Qwen text-decoder style used
by the current InternVL streaming experiments. M-RoPE models such as Qwen2.5-VL
need a separate model position policy because visual positions are multi-axis.
The same idea still applies, but the mapping cannot use the current one-axis
`llama_memory_seq_add` contract blindly.

## Intended Integration

The future video-compression path should follow this order:

```text
1. decide old visual KV span to compress
2. create compressed replacement KV for the kept prefix/new summary span
3. remove obsolete positions with llama_memory_seq_rm
4. shift the unchanged tail with llama_memory_seq_add
5. let the next llama.cpp memory update apply cached K RoPE shift
6. update streaming trace metadata so Frame/Question ordering still matches
```

The key correctness test will be comparing logits after KV repositioning against
a reference run that directly prefills the compacted sequence.
