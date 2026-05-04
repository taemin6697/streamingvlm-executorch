# llama.cpp Android VLM Memory Summary

This note summarizes the memory breakdown observed during the Android
`llama-mtmd-cli` smoke tests.

The llama.cpp logs report memory as:

```text
self = model + context + compute
```

For these VLM runs, `context` is mostly KV-cache allocation.

## Summary Table

| Experiment | Backend | Model | Context / KV | Compute | Total self |
|---|---:|---:|---:|---:|---:|
| SmolVLM-500M | CPU | `414 MiB` | `320 MiB` | `116 MiB` | `851 MiB` |
| SmolVLM-500M | Vulkan0 | `366 MiB` | `320 MiB` | `98 MiB` | `785 MiB` |
| SmolVLM-500M | OpenCL attempt, CPU fallback | `414 MiB` | `320 MiB` | `116 MiB` | `851 MiB` |
| InternVL3 1B, original `640x480` | CPU | `638 MiB` | `384 MiB` | `299 MiB` | `1322 MiB` |
| InternVL3 1B, resized `448x448` | CPU | `638 MiB` | `384 MiB` | `299 MiB` | `1322 MiB` |
| InternVL3 1B, original `640x480` | Vulkan0 | `500 MiB` | `384 MiB` | `297 MiB` | `1182 MiB` |
| InternVL3 1B, resized `448x448` | Vulkan0 | `500 MiB` | `384 MiB` | `297 MiB` | `1182 MiB` |

## Notes

- Resizing InternVL3 input from `640x480` to `448x448` reduced image tokens from
  `1280` to `256`, but did not reduce static context/KV memory.
- SmolVLM used `n_ctx = 8192`, producing a `320 MiB` KV cache.
- InternVL3 used `n_ctx = 32768`, producing a `384 MiB` KV cache.
- Vulkan moves most model/context/compute buffers to GPU memory, but host-side
  buffers still remain. InternVL3 Vulkan logs also reported about `205 MiB` of
  host self memory.
- The OpenCL run on Samsung Xclipse 940 detected OpenCL but rejected the GPU as
  unsupported, so memory matched the CPU fallback path.

## Interpretation

Image resizing mainly changes actual prefill token count and latency. It does
not shrink the preallocated KV-cache unless the run also changes context length
with an option such as `-c 2048`.
