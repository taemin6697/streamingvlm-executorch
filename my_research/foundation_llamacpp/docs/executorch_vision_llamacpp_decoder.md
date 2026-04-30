# ExecuTorch Vision Encoder + llama.cpp Decoder Hybrid

This note records the feasibility of a hybrid runtime where ExecuTorch runs the
vision encoder continuously, then llama.cpp consumes the resulting vision
embeddings for decoder prefill and generation when the user asks a question.

## Goal

The target flow is:

```text
Streaming phase:
  frame -> ExecuTorch vision_encoder.pte -> vision embeddings -> local cache

User query phase:
  cached vision embeddings + text prompt
  -> llama.cpp decoder prefill
  -> llama.cpp token generation
  -> llama.cpp KV-memory APIs for cache management experiments
```

This is different from transferring ExecuTorch KV-cache into llama.cpp. KV-cache
layout, RoPE handling, layer ordering, quantization, and backend buffer ownership
are runtime-specific, so direct KV-cache transfer between ExecuTorch and
llama.cpp is not a practical target. The practical boundary is the **vision
embedding**, not the KV-cache.

## llama.cpp VLM Entry Point

llama.cpp's current multimodal path lives under:

```text
llama.cpp/tools/mtmd/
```

The relevant runtime flow is:

```text
mtmd_tokenize()
  -> split prompt into text chunks and media chunks

mtmd_helper_eval_chunks()
  -> text chunk: llama_decode() with token ids
  -> image/audio chunk:
       mtmd_encode_chunk()
       mtmd_get_output_embd()
       mtmd_helper_decode_image_chunk()

generate_response()
  -> llama_decode() token by token
```

The key injection point is:

```cpp
int32_t mtmd_helper_decode_image_chunk(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past);
```

This helper already accepts a `float * encoded_embd` pointer and builds a
`llama_batch` with:

```cpp
batch.tokens = nullptr;
batch.embd   = encoded_embd;
```

So llama.cpp already has a path for feeding external embeddings into the text
decoder. Today those embeddings are produced by llama.cpp's `mmproj`; for the
hybrid experiment, they could instead come from ExecuTorch.

## Compatibility Requirements

The ExecuTorch vision output must match the llama.cpp decoder contract:

- Shape must be `[n_image_tokens, llama_model_n_embd_inp(model)]`.
- Values must be in the same embedding space as the llama.cpp text decoder.
- The model/tokenizer/chat template must be the same family, for example
  InternVL-style `<img> ... </img>` formatting for InternVL.
- The number of image tokens must match the llama.cpp `mtmd_input_chunk`
  metadata used for positions and batching.
- For M-RoPE models such as Qwen-VL style models, per-token 2D/temporal
  positions must also match. InternVL-style normal position handling is a
  simpler first target.

For InternVL, llama.cpp already has a `PROJECTOR_TYPE_INTERNVL` path with:

```text
<img> ... image embeddings ... </img>
```

That makes InternVL a reasonable first candidate if the ExecuTorch vision
encoder output is already projected into the text decoder hidden dimension.

## Android Feasibility

Android is feasible, but should not be the first validation step.

A native Android binary or app would need to link both runtimes:

```text
Android process
  ExecuTorch runtime
    -> run vision_encoder.pte
    -> read CPU-accessible output embedding buffer

  llama.cpp runtime
    -> pass embedding buffer to mtmd_helper_decode_image_chunk()
    -> generate with llama_decode()
```

Vulkan is also possible on the llama.cpp side with `GGML_VULKAN=ON`. In that
case, the external embedding still enters as a CPU pointer and llama.cpp uploads
it internally to the selected backend. This introduces a copy, but it keeps the
integration simple.

The harder Android issues are practical:

- binary size and dependency conflicts from linking ExecuTorch and llama.cpp
- memory pressure from two runtimes in one process
- thread and backend resource contention
- dtype conversion if ExecuTorch produces fp16 while llama.cpp expects fp32
- keeping model, tokenizer, prompt format, and embedding layout aligned

## Prototype Plan

Start on Linux before Android:

1. Run llama.cpp VLM normally and dump the `mtmd_get_output_embd()` output for a
   known image.
2. Run the ExecuTorch vision encoder on the same image and dump its output.
3. Compare shape, dtype, token count, and cosine similarity if both are expected
   to be equivalent.
4. Add a small llama.cpp-side test path that bypasses `mtmd_encode_chunk()` and
   feeds an external embedding blob into `mtmd_helper_decode_image_chunk()`.
5. Once Linux works, port the same path to Android CPU llama.cpp.
6. Enable Android Vulkan for llama.cpp decoder.
7. Replace the embedding blob with live ExecuTorch vision output.

## Why This Is Useful

This hybrid path separates two research concerns:

- ExecuTorch can continue to be used for mobile-oriented vision encoder
  experiments, including XNNPACK/QNN/Vulkan backend comparison.
- llama.cpp can be used for decoder-side KV-memory experiments because it has
  runtime memory APIs such as `llama_memory_seq_rm`, `llama_memory_seq_cp`,
  `llama_memory_seq_add`, `llama_memory_seq_div`, and state save/load APIs.

This does not solve cross-runtime KV-cache transfer, but it gives a plausible
route to combine ExecuTorch vision execution with llama.cpp's more accessible
decoder/KV runtime.
