# Vulkan fp16 Correctness Debug

This folder contains focused experiments for the Vulkan true-fp16 export path.

The current hypothesis is not that InternVL3 cannot tolerate fp16. XNNPACK fp16
works as a reference. The issue is likely a Vulkan fp16 lowering/runtime
correctness difference around one of the decoder-sensitive paths.

First experiment:

```bash
SEQ_LEN=512 my_research/foundation/debug/vulkan_fp16_correctness/export_fp16_sdpa_kv_island.sh
```

This keeps `--use_sdpa_with_kv_cache` enabled and exports the decoder with
`--dtype fp16`, but forces transformed KV-cache buffers and update inputs to
fp32 through the foundation exporter debug flag:

```text
--vulkan_debug_fp32_kv_cache
```

If this artifact produces sane text while the previous full-fp16 Vulkan artifact
does not, the culprit is likely the fp16 KV-cache/update/SDPA path. If it still
produces corrupted text, the next suspects are Vulkan fp16 RMSNorm fusion,
softmax, logits, or mixed-dtype/layout transitions.

Second experiment:

```bash
SEQ_LEN=512 my_research/foundation/debug/vulkan_fp16_correctness/export_fp16_sdpa_portable.sh
```

This keeps `sdpa_with_kv_cache` enabled in the graph, keeps the fp32 KV-cache
island, but blocks `llama.sdpa_with_kv_cache` from Vulkan delegation:

```text
--vulkan_debug_block_sdpa_delegate
```

If this fixes generation, Vulkan's delegated `sdpa_with_kv_cache` path is the
culprit. If it still corrupts output, the issue is more likely another delegated
fp16 decoder op such as RMSNorm, softmax, linear/layout transition, or logits.
