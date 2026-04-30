# Hybrid Runtime Notes

This directory collects the llama.cpp and ExecuTorch hybrid experiments for
mobile streaming VLM research.

## Main Takeaway

For a streaming visual assistant, the most promising direction is a hybrid
runtime:

```text
Streaming vision path:
  ExecuTorch / QNN / Vulkan

Decoder and KV-control path:
  llama.cpp
```

ExecuTorch is better suited for mobile backend experiments and continuous vision
execution, especially when targeting Qualcomm QNN/HTP. llama.cpp is better
suited for decoder-side experiments because it exposes a compact, KV-centric
LLM runtime with direct KV-memory APIs.

## Runtime Terminology

The difference is not just a library difference. It is a runtime architecture
difference:

```text
ExecuTorch:
  general-purpose AOT static-graph runtime
  max-shape planned memory
  backend delegation through XNNPACK / Vulkan / QNN

llama.cpp:
  LLM-specialized dynamic token runtime
  runtime-managed KV cache
  token-level prefill/decode loop
```

Another concise way to describe it:

```text
AOT static graph
vs
runtime-managed token loop
```

ExecuTorch can be faster for fixed-shape vision/backend execution, but large
compiled context lengths can inflate planned activation/workspace memory.
llama.cpp keeps long-context memory closer to KV-cache capacity, but its
multimodal vision path and mobile accelerator control are weaker.

## Why This Matters For Streaming

Streaming VLMs need to process frames continuously and answer user queries with
low latency. That creates two different requirements:

- Vision encoding should use mobile accelerators efficiently, ideally QNN/HTP or
  a well-performing Vulkan path.
- Decoder memory should remain compact and controllable as visual context grows.

ExecuTorch gives the mobile backend control. llama.cpp gives easier decoder and
KV-cache control. A hybrid keeps both.

## Key Experiment Results

### SmolVLM-500M

SmolVLM-500M ran successfully on Android CPU and Vulkan with llama.cpp.

```text
CPU:
  image encode: 2688 ms
  image decode/prefill: 504 ms
  prompt eval: 23.96 tok/s
  decode: 26.51 tok/s

Vulkan:
  image encode: 14178 ms
  image decode/prefill: 5 ms
  prompt eval: 5.59 tok/s
  decode: 66.05 tok/s
```

Vulkan improved token decode but was worse for the short VLM prefill path on the
tested Samsung Xclipse 940 device.

### OpenCL

OpenCL built successfully, and the device exposed:

```text
Samsung Xclipse 940 (OpenCL 3.0)
```

However llama.cpp rejected it as unsupported and fell back to CPU:

```text
Unsupported GPU: Samsung Xclipse 940
no usable GPU found, --gpu-layers option will be ignored
```

On Qualcomm Adreno devices, especially Snapdragon 8 Gen 3 / 8 Elite, OpenCL is
worth testing because llama.cpp explicitly targets those GPUs.

### InternVL3 1B

InternVL3 1B ran correctly on Android CPU and Vulkan with llama.cpp.

Original COCO cats image:

```text
image size: 640 x 480
llama.cpp image tokens: 1280
```

The 1280 tokens came from InternVL dynamic high-resolution tiling:

```text
1280 = 5 tiles * 256 tokens
```

After resizing the input image to `448 x 448`, llama.cpp used the expected
single-tile count:

```text
image tokens: 256
```

CPU total latency dropped from `59.7 s` to `13.7 s`. Vulkan total latency dropped
from `253.8 s` to `62.3 s`, but Vulkan remained slower than CPU for
vision/prompt prefill on the tested device.

## Memory Summary

llama.cpp reports memory roughly as:

```text
self = model + context + compute
```

For these runs, `context` is mostly resident KV-cache memory.

```text
SmolVLM CPU:
  model 414 MiB, KV 320 MiB, compute 116 MiB, self 851 MiB

SmolVLM Vulkan:
  model 366 MiB, KV 320 MiB, compute 98 MiB, self 785 MiB

InternVL3 1B CPU:
  model 638 MiB, KV 384 MiB, compute 299 MiB, self 1322 MiB

InternVL3 1B Vulkan:
  model 500 MiB, KV 384 MiB, compute 297 MiB, self 1182 MiB
```

The KV-cache is resident in runtime memory. On CPU it is host DRAM. On Vulkan it
is GPU-visible memory, which is still unified DRAM on mobile SoCs.

Changing `-n` changes maximum generated tokens and therefore time/energy. It
does not significantly reduce allocated KV memory. To reduce KV memory in
llama.cpp, change context length with `-c`, for example:

```bash
-c 2048
```

## ExecuTorch vs llama.cpp Memory Behavior

ExecuTorch 16K artifacts can consume much more memory than llama.cpp because
ExecuTorch stores an AOT max-shape graph memory plan:

```text
weights/constants
+ KV cache
+ planned activation arena
+ XNNPACK/Vulkan/QNN delegate workspace
+ packed/copied backend buffers
+ method-specific planned memory
```

llama.cpp is more compact for decoder memory because it is an LLM-specialized
runtime:

```text
weights
+ KV cache
+ reusable compute buffer
```

This is why a 16K ExecuTorch artifact can use several GB while llama.cpp with a
large context can stay closer to KV-cache capacity.

## Document Index

- `executorch_vision_llamacpp_decoder.md`:
  Feasibility note for feeding ExecuTorch vision embeddings into llama.cpp's
  decoder path.
- `llamacpp_android_cpu_vlm_smoke.md`:
  SmolVLM-500M Android CPU smoke test.
- `llamacpp_android_vulkan_vlm_smoke.md`:
  SmolVLM-500M Android Vulkan smoke test.
- `llamacpp_android_opencl_vlm_attempt.md`:
  Android OpenCL build and unsupported Xclipse fallback.
- `llamacpp_android_internvl3_1b_smoke.md`:
  InternVL3 1B CPU/Vulkan tests, including the `448 x 448` resize follow-up.
- `llamacpp_android_memory_summary.md`:
  Backend memory breakdown for the llama.cpp Android VLM runs.
- `aot_static_graph_vs_runtime_token_loop.md`:
  Discussion note on ExecuTorch AOT memory growth, llama.cpp runtime token
  execution, and why the split matters for streaming VLMs.

## Current Recommendation

Use ExecuTorch as the primary runtime for mobile vision/backend experiments:

```text
vision encoder:
  ExecuTorch QNN / Vulkan / XNNPACK
```

Use llama.cpp as the decoder/KV-control prototype:

```text
decoder:
  llama.cpp CPU / Vulkan / OpenCL where supported
  runtime-managed KV-cache experiments
```

The long-term streaming design should validate an embedding boundary rather than
a KV-cache boundary:

```text
ExecuTorch vision embeddings -> llama.cpp decoder prefill
```

Direct cross-runtime KV-cache transfer is not practical because KV layout,
position handling, quantization, and backend ownership are runtime-specific.
