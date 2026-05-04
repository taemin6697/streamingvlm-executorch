# llama.cpp Android InternVL3 1B Smoke Test

This note records Android `llama-mtmd-cli` smoke tests for
`ggml-org/InternVL3-1B-Instruct-GGUF` with the same COCO cats image used in the
SmolVLM tests.

## Model Files

Downloaded from:

```text
ggml-org/InternVL3-1B-Instruct-GGUF
```

Files:

```text
InternVL3-1B-Instruct-Q8_0.gguf                 644 MiB
mmproj-InternVL3-1B-Instruct-Q8_0.gguf          318 MiB
```

Local path:

```text
/workspace/streamingvlm/llama.cpp/models/InternVL3-1B-Instruct-GGUF/
```

Device path:

```text
/data/local/tmp/llama-internvl3/
```

## Important Model Shape

InternVL3 1B is much heavier than SmolVLM-500M for image prefill:

```text
text model context length: 32768
vision image size: 448 x 448
vision patch size: 14
image embedding tokens: 1280
```

By comparison, the SmolVLM-500M run used only `64` image tokens. This is the
main reason InternVL3 1B has much higher prompt/prefill latency.

## CPU Run

Command:

```bash
adb shell 'cd /data/local/tmp/llama-internvl3 && chmod +x llama-mtmd-cli && LD_LIBRARY_PATH=. ./llama-mtmd-cli \
  -m InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --image image.jpg \
  -p "Describe this image briefly." \
  -n 64 \
  -t 4'
```

Observed output:

```text
The image shows two cats lying on a pink blanket next to two remote controls. One cat is on the left side, and the other is on the right side of the blanket.
```

Timing:

```text
image slice encoded in 44818 ms
image decoded in 11527 ms
prompt eval: 1295 tokens, 22.83 tok/s
decode eval: 36 runs, 18.06 tok/s
total time: 59728.92 ms
```

Memory/context:

```text
n_ctx = 32768
CPU KV buffer size = 384.00 MiB
CPU compute buffer size = 299.74 MiB
```

## Vulkan Run

Command:

```bash
adb shell 'cd /data/local/tmp/llama-internvl3 && chmod +x llama-mtmd-cli && LD_LIBRARY_PATH=. ./llama-mtmd-cli \
  -m InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --image image.jpg \
  -p "Describe this image briefly." \
  -n 64 \
  -t 4 \
  --n-gpu-layers 99'
```

Vulkan was used for both the text model and vision encoder:

```text
load_tensors: offloaded 25/25 layers to GPU
llama_kv_cache: Vulkan0 KV buffer size = 384.00 MiB
clip_ctx: CLIP using Vulkan0 backend
```

Observed output:

```text
The image shows two cats lying on a pink blanket next to two remote controls. The cats appear relaxed and comfortable, with one cat lying on its side and the other lying on its back.
```

Timing:

```text
image slice encoded in 194521 ms
image decoded in 44501 ms
prompt eval: 1295 tokens, 5.15 tok/s
decode eval: 38 runs, 34.56 tok/s
total time: 253792.90 ms
```

Memory/context:

```text
n_ctx = 32768
Vulkan0 model buffer size = 500.56 MiB
Vulkan0 KV buffer size = 384.00 MiB
Vulkan0 compute buffer size = 297.99 MiB
```

## CPU vs Vulkan

On the tested Samsung Xclipse 940 device:

```text
CPU total:     59.7 s
Vulkan total: 253.8 s
```

Vulkan decode was faster:

```text
CPU decode:     18.06 tok/s
Vulkan decode:  34.56 tok/s
```

But Vulkan image encode and image prefill were much slower:

```text
CPU image encode:      44.8 s
Vulkan image encode:  194.5 s

CPU image prefill:     11.5 s
Vulkan image prefill:  44.5 s
```

Conclusion:

```text
InternVL3 1B works in llama.cpp on Android CPU and Vulkan, but on this device
Vulkan is not practical for the VLM prefill path. The GPU only helped token
decode.
```

This reinforces the hybrid direction: use a faster dedicated vision path
where possible, and treat llama.cpp's decoder/KV APIs as the main attraction
rather than assuming llama.cpp Vulkan is a good vision encoder backend.

## 448x448 Resize Follow-up

The original COCO cats image is `640 x 480`. llama.cpp's InternVL preprocessor
uses dynamic high-resolution tiling for images larger than the native
`448 x 448` size, which is why the first run produced `1280` image tokens:

```text
1280 = 5 tiles * 256 tokens
```

To match the ExecuTorch-style single-frame input, the image was resized to
exactly `448 x 448`:

```text
/workspace/streamingvlm/my_research/foundation/sample_coco_cats_448.jpg
```

For a single `448 x 448` InternVL3 tile:

```text
patch grid = 448 / 14 = 32
n_merge = 2
merged grid = 16 x 16
image tokens = 256
```

### CPU 448x448

The resized image correctly reduced image tokens:

```text
decoding image batch 1/1, n_tokens_batch = 256
```

Timing:

```text
image slice encoded in 8961 ms
image decoded in 2149 ms
prompt eval: 271 tokens, 23.96 tok/s
decode eval: 40 runs, 21.41 tok/s
total time: 13653.77 ms
```

Compared with the original `640 x 480` image:

```text
CPU total: 59.7 s -> 13.7 s
image tokens: 1280 -> 256
```

### Vulkan 448x448

The resized image also reduced Vulkan image tokens:

```text
decoding image batch 1/1, n_tokens_batch = 256
```

Timing:

```text
image slice encoded in 47550 ms
image decoded in 497 ms
prompt eval: 271 tokens, 4.48 tok/s
decode eval: 36 runs, 40.27 tok/s
total time: 62346.23 ms
```

Compared with the original `640 x 480` image:

```text
Vulkan total: 253.8 s -> 62.3 s
image tokens: 1280 -> 256
```

Conclusion:

```text
Resizing to 448x448 confirms that the earlier 1280-token run came from
InternVL dynamic tiling. On this Xclipse 940 device, Vulkan remains slower than
CPU for vision/prompt prefill even after reducing the image to one tile, but
Vulkan token decode is faster.
```
