# Paged KV OpenCL Prototype

## Goal

Dynamic contiguous KV starts with a small OpenCL KV buffer, but a grow event
snapshots KV, allocates a larger contiguous buffer, restores KV, reserves the
graph again, and retries the prompt. On the 2B Q8 hybrid streaming test,
`1024 -> 16384` cells produced a visible `DynamicKVGrow` spike around
`369-394 ms`.

Paged KV changes the addressing model:

```text
logical token index
  -> logical_page = index / page_size
  -> page_offset  = index % page_size
  -> physical_page = page_table[logical_page]
  -> physical token row = physical_page * page_size + page_offset
```

The current implementation focuses on removing the grow-time KV realloc/copy
spike in the llama.cpp/OpenCL path.

## Current Scope

```text
single sequence
standard non-SWA KV cache
OpenCL FlashAttention required
hybrid streaming and standalone OpenCL-compatible llama.cpp graph path
page size configured by --kv-page-size, validated at 256 cells
no eviction
no prefix sharing
no true KV compression yet
```

Unsupported modes fail early. Paged KV currently requires `--kv-init-size`,
`--kv-grow-step`, and FlashAttention.

## Implemented Files

```text
llama.cpp/include/llama.h
llama.cpp/common/arg.cpp
llama.cpp/common/common.h
llama.cpp/common/common.cpp
llama.cpp/src/llama-cparams.h
  CLI/common/context plumbing for --paged-kv-cache and --kv-page-size

llama.cpp/src/llama-context.cpp
llama.cpp/src/llama-model.cpp
  paged KV validation and initial physical size selection

llama.cpp/src/llama-kv-cells.h
  grow_to(n): metadata grow without clearing existing cell state

llama.cpp/src/llama-kv-cache.*
  page table metadata
  logical-to-physical cell mapping for K/V writes
  active cell capacity separate from reserved backing capacity
  metadata-only paged grow

llama.cpp/src/llama-graph.*
llama.cpp/ggml/include/ggml.h
llama.cpp/ggml/src/ggml.c
  page-table input attached to FlashAttention op

llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp
llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f16.cl
llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32.cl
llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl
  OpenCL FlashAttention receives the page table and translates K/V row indices

my_research/foundation_llamacpp/runner/cli.py
my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp
  paged KV args, ctx-size forwarding, result naming, and grow-row finalization
```

## Memory Layout In This Version

This version reserves one max-context OpenCL backing tensor at initialization:

```text
active cells:   --kv-init-size, e.g. 1024
backing cells:  --ctx-size,     e.g. 32768
logical cells:  --ctx-size,     e.g. 32768
page size:      --kv-page-size, e.g. 256
```

`reset_capacity` logs these separately:

```text
1024 active / 32768 backing / 32768 logical cells
```

Grow only extends active metadata:

```text
v_cells[s].grow_to(new_size)
allocate new logical page-table entries
do not call reset_capacity()
do not snapshot/restore KV
```

This is a deliberate tradeoff. It removes the grow-time realloc/copy spike, but
it does not reduce startup KV memory. For 2B f16 KV at `ctx=32768`, the backing
OpenCL KV buffer is `896 MiB`.

## Attention Path

K/V writes use page-mapped row indices:

```text
logical cell -> physical page row -> ggml_set_rows destination row
```

FlashAttention consumes the same page table. Kernel-side addressing replaces
the logical `k_idx` row with:

```text
physical_k_idx = page_table[k_idx / page_size] * page_size + (k_idx % page_size)
```

The K/V tensor is still one reserved OpenCL allocation in this version. The page
table is still useful because it removes the contiguous-view assumption in the
attention kernel and leaves room for later non-identity page remaps.

## Measurement

Command:

```bash
python3 my_research/foundation_llamacpp/runner/cli.py \
  --processor hybrid \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --manifest my_research/foundation/results/model/qnn/internvl3_2b_hybrid_16p_16k_16a4w/manifest.json \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --stream-mode single-buffer \
  --sampling-fps 1.0 \
  --max-video-time 15.0 \
  --time '[5.0, 8.0, 11.0, 14.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???", "What changed in the scene?", "Summarize the full situation so far."]' \
  --ctx-size 32768 \
  --paged-kv-cache \
  --kv-page-size 256 \
  --kv-init-size 1024 \
  --kv-grow-step 15360 \
  --flash-attn on \
  --n-predict 32 \
  --results-root my_research/foundation_llamacpp/results/log/paged_kv_full_2b_hybrid \
  --force-push
```

Result:

```text
results/log/paged_kv_full_2b_hybrid/InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_kv16_paged
return_code: 0
stream frames: 16
prompt events: 4
```

Grow logs:

```text
reset_capacity: OpenCL KV buffer size = 896.00 MiB
reset_capacity: size = 896.00 MiB (1024 active / 32768 backing / 32768 logical cells)
grow_to: paged KV grow metadata-only ... elapsed = 0.201 ms
decode: dynamic KV grow retry window ... clock_start_ms -> clock_end_ms = 113 ms
```

Comparison:

```text
contiguous dynamic baseline:
  1024 -> 16384 retry window: 369-394 ms
  grow_to allocation/copy: 232-246 ms
  initial OpenCL KV: 28 MiB

paged reserved-backing prototype:
  1024 -> 16384 retry window: 113 ms
  grow_to metadata update: 0.201 ms
  initial OpenCL KV backing: 896 MiB
```

Interpretation:

```text
The black grow bar is much smaller because the KV realloc/copy disappeared.
The remaining black bar is mainly scheduler reserve and retry preparation.
The memory cost moved to startup because this version reserves max backing.
```

## True Page Allocation Still To Do

This prototype is not yet a page-per-OpenCL-allocation allocator. A future
memory-saving paged KV needs either:

```text
1. a backend abstraction that lets one logical KV tensor be backed by multiple
   OpenCL page buffers, plus custom set_rows/attention kernels that consume
   that physical page list; or
2. a chunked backing-buffer allocator that grows by appending page chunks and
   lets kernels address chunk id + offset without copying old chunks.
```

OpenCL kernels cannot simply read a device-memory array of `cl_mem` handles, so
this needs explicit backend support. The current single-backing approach was
chosen to validate page-table attention and remove grow-time copies first.

## Future True KV Compression

For true KV compression, the logical history should become a compact sequence
after compression. For example, if `128` token vectors are compressed to `32`
vectors:

```text
logical history length decreases by 96
active pages decrease from 4 pages to 1 page when page_size = 32
subsequent attention length uses the compact length
future RoPE/position handling must be decided at the compression boundary
```

If compression must reduce system-visible memory, freeing internal page-table
entries is not enough. The runtime must run an explicit shrink/repack phase:

```text
1. synchronize outstanding GPU work
2. write compressed KV vectors into compact destination pages
3. device-to-device copy remaining live pages into a smaller backing allocation
4. rewrite page_table logical->physical entries
5. release the old larger backing allocation/chunks
```

That shrink/repack operation intentionally has a copy/allocation spike. It
should be scheduled at explicit compression boundaries, not hidden inside normal
decode.

The important invariant for that future mode is that `capacity pages` must
actually decrease when the system-visible allocation is expected to decrease.
Reducing only active pages shortens attention but does not return memory to the
OpenCL driver or OS.

## Difference From vLLM

vLLM's PagedAttention is a mature CUDA serving allocator with batching, block
tables, sharing, and eviction. This prototype applies the page-table addressing
idea inside llama.cpp/OpenCL for mobile streaming VLM experiments. The immediate
research question is how prompt-time latency behaves when visual context grows
and prompt arrival is uncertain.
