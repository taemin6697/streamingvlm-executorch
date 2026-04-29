# Vulkan Memory Analysis

This note explains why Vulkan can appear to consume much more memory than
XNNPACK even when both artifacts are labeled `fp16`, and how to interpret the
memory logs under `my_research/foundation/results/log`.

## Summary

For memory-pressure analysis, use `mem_available_kb` from
`android_memory_timeline.csv`.

The useful system-level metric is:

```text
memory_used = max(mem_available_kb) - min(mem_available_kb)
```

This measures how much Android system-available memory disappears during the
run. It is different from process-local RSS.

Current 1B fp16 logs up to 8K show:

```text
XNNPACK 8K: ~2484 MiB available-memory drop
Vulkan 8K:  ~5779 MiB available-memory drop
```

The Vulkan increase is not mainly caused by a larger `.pte` model file. The
larger effect comes from GPU/KGSL allocations made by the Vulkan delegate.

## Why `self_rss_kb` Is Not Enough

`self_rss_kb` captures memory resident in the runner process. It is useful for
CPU-side memory, but it does not fully represent GPU driver allocations.

Vulkan uses GPU buffers and Qualcomm KGSL-managed graphics memory. These
allocations may not appear as process RSS in the same way CPU tensors do, but
they still reduce total system available memory.

Therefore:

```text
self_rss_kb          = process-local CPU-oriented RSS
mem_available_kb     = system-level available memory
gpu_total_kb         = GPU memory tracked by the system
kgsl_shmem_usage_kb  = Qualcomm KGSL shared memory usage
```

For system memory pressure, `mem_available_kb` is the more relevant metric.
For root-cause analysis, compare it with `gpu_total_kb` and
`kgsl_shmem_usage_kb`.

## Artifact Size Check

The Vulkan `.pte` artifacts are larger than XNNPACK, but only modestly.

For 8K:

```text
XNNPACK total: ~1795.9 MiB
  text_decoder_xnnpack.pte     ~946.8 MiB
  text_embedding_xnnpack.pte   ~259.2 MiB
  vision_encoder_xnnpack.pte   ~590.0 MiB

Vulkan total:  ~2055.3 MiB
  text_decoder_vulkan.pte      ~948.1 MiB
  text_embedding_vulkan.pte    ~518.4 MiB
  vision_encoder_vulkan.pte    ~588.7 MiB
```

The main file-size difference is `text_embedding_vulkan.pte`, which is roughly
2x the XNNPACK embedding file. This explains about 260 MiB of artifact-size
difference, not the multi-GB runtime memory gap.

## Vulkan `fp16` Is Not The Same Layout As XNNPACK `fp16`

XNNPACK `fp16` and Vulkan `fp16` are both labeled `fp16`, but they are not the
same runtime memory layout.

For Vulkan exports, the manifest records:

```json
{
  "vulkan_export_dtype": "fp32",
  "vulkan_force_fp16": true
}
```

This follows the upstream Vulkan export style: graph capture/export may use an
fp32-oriented path while Vulkan execution forces fp16 in the delegate. The
runtime can still allocate Vulkan storage buffers, staging buffers, descriptor
resources, and driver-managed GPU memory.

So the comparison is:

```text
XNNPACK fp16 = CPU/XNNPACK path with CPU-side packed weights and tensors
Vulkan fp16  = Vulkan delegate path with GPU buffers and KGSL allocations
```

Same precision label does not imply the same memory layout.

## Vulkan 8K Timeline

For `internvl3_vulkan_1b_8k_fp16`, the phase log shows:

```text
L:                   0.001s - 1.589s
V_Encode:            1.589s - 2.685s
EmbeddingAndMerging: 6.102s - 6.107s
T_Prefill:           6.102s - 6.516s
Decode:              6.516s onward
```

The large available-memory drop appears visually near the
Embedding/Prefill transition. However, most of the drop happens between
vision encode finishing and embedding/prefill starting:

```text
Interval: V_Encode end (~2.685s) to EmbeddingAndMerging start (~6.102s)

mem_available_kb drop:    ~4978 MiB
gpu_total_kb growth:      ~5391 MiB
kgsl_shmem_usage growth:  ~5392 MiB
self_rss_kb growth:        ~494 MiB
```

This means the cliff is mostly GPU/KGSL allocation, not CPU process RSS.

## Interpretation

The most likely explanation is:

1. The vision encoder finishes.
2. The runner moves toward text embedding / decoder prefill.
3. Vulkan delegate prepares large GPU-side resources for the delegated graphs.
4. GPU/KGSL allocations increase sharply.
5. Android `mem_available_kb` drops sharply.
6. Process `self_rss_kb` increases only moderately.

The exact allocation may include:

- Vulkan storage buffers for text embedding and decoder inputs.
- Staging buffers for CPU-to-GPU transfer.
- Delegate graph buffers for prefill/decode.
- Driver or KGSL-managed cached resources.
- Backend-specific packed or transformed weight layouts.

Because `gpu_total_kb` and `kgsl_shmem_usage_kb` closely track the available
memory drop, Vulkan GPU allocation is the dominant cause.

## Debug Scripts

Two debug scripts were added:

```bash
python my_research/foundation/debug/plot_backend_memory_by_seq.py
```

Outputs:

```text
my_research/foundation/debug/backend_memory_by_seq.png
my_research/foundation/debug/backend_memory_by_seq.csv
```

This plots XNNPACK and Vulkan memory usage by sequence length up to 8K using
`mem_available_kb`.

```bash
python my_research/foundation/debug/plot_memory_components_timeline.py
```

Outputs:

```text
my_research/foundation/debug/vulkan_8k_memory_components.png
```

This overlays available-memory drop, GPU memory growth, KGSL memory growth,
process RSS growth, and cached-memory changes on the same timeline.

## Takeaway

For paper or experiment wording:

```text
Vulkan reduces process-visible CPU RSS, but can create substantial GPU/KGSL
memory pressure as context length grows. Therefore, mobile memory evaluation
must track system-level available memory and GPU memory counters, not only
process RSS.
```

This distinction is important for mobile VLM deployment because GPU memory
pressure still affects system stability, thermal behavior, and OOM risk even
when process-local RSS looks small.
