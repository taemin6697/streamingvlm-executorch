# Paged KV OpenCL Prototype

## Motivation

Dynamic contiguous KV reduces the initial OpenCL KV allocation, but growth
currently snapshots old KV through host memory, reallocates a larger contiguous
buffer, and restores the snapshot. That avoids reserving the maximum context at
startup but creates a visible latency spike.

Paged KV changes the ownership unit from one contiguous logical cache to fixed
KV pages:

```text
logical token position -> logical page -> physical page -> page offset
```

The first implementation slice wires the CLI, context parameters, standard
`llama_kv_cache` metadata, page allocation helper, and `llama-graph` page-table
input. Runtime execution is still guarded because OpenCL attention currently
reads contiguous K/V tensor views; correctness requires the attention op/kernel
to consume the page table when addressing K/V.

## Current Scope

```text
single sequence
standard non-SWA KV cache
OpenCL/hybrid target path
page size configured by --kv-page-size
no prefix sharing
no batching
no eviction
no true KV compression yet
```

Unsupported modes fail early. `--paged-kv-cache` cannot be combined with
`--dynamic-kv-cache`, recurrent memory, hybrid recurrent/attention memory,
SWA/iSWA, unified KV, or multiple sequences.

## Implemented Infrastructure

```text
llama.cpp/include/llama.h
  llama_context_params::{paged_kv_cache, kv_page_size}

llama.cpp/common/{arg.cpp,common.h,common.cpp}
  --paged-kv-cache and --kv-page-size parsing and forwarding

llama.cpp/src/llama-cparams.h
  internal cparams fields

llama.cpp/src/llama-kv-cache.*
  PagedKVBlockTable
  allocate_paged_kv_page()
  logical_pos_to_page_offset()
  build_input_kv_page_table()
  set_input_kv_page_table()

llama.cpp/src/llama-graph.*
  llm_graph_input_attn_kv::self_kv_page_table
  attn_inp_kv_page_table graph input creation/set path
```

The page table is intentionally metadata-only until the attention read path is
implemented. The graph now has a place to feed page-table data, but `get_k()`,
`get_v()`, and OpenCL FlashAttention still assume contiguous K/V storage.

## Next Kernel Step

The OpenCL FlashAttention kernels currently compute addresses like:

```text
k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1
v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1
```

Paged attention must replace `k_idx` with:

```text
logical_page  = k_idx / kv_page_size
page_offset   = k_idx % kv_page_size
physical_page = kv_page_table[logical_page]
physical_idx  = physical_page * kv_page_size + page_offset
```

That requires passing the page-table tensor to the ggml/OpenCL attention op, not
only storing it in llama.cpp metadata.

## Future True KV Compression

True KV compression should be handled as a page/segment transformation. For
example, compacting `128 -> 32` token vectors with a 32-slot physical page size:

```text
before:
  128 raw tokens = 4 active pages

after:
  32 compressed vectors = 1 active page
```

If compression must reduce system-visible memory, freeing internal pages is not
enough. The runtime must run a shrink/repack phase:

```text
1. synchronize outstanding GPU work
2. compress selected raw pages into fewer physical pages
3. device-to-device copy live pages into a smaller OpenCL allocation
4. rewrite the page table to the new physical page indices
5. release the old larger OpenCL buffer/chunks
```

In that future mode, `capacity pages` must decrease after compression, not only
`active pages`. This has a copy/allocation spike, so it should be explicit and
scheduled at known compression boundaries.

## Difference From vLLM

vLLM's PagedAttention is a mature CUDA server-side serving system with batching,
block tables, and memory sharing. This prototype applies the same broad memory
idea inside llama.cpp/OpenCL for a mobile SoC streaming VLM runtime, where the
main research question is latency and memory pressure during uncertain prompt
arrival and vision-prefix growth.
