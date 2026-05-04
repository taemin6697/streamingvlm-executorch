# AOT Static Graph vs Runtime Token Loop

This note summarizes the discussion about why ExecuTorch memory grows sharply
with long compiled context lengths, why llama.cpp behaves differently, and what
that means for streaming VLM design.

## Short Answer

For long-context VLM streaming, the important distinction is:

```text
ExecuTorch:
  AOT static graph runtime
  max-context graph memory planning

llama.cpp:
  LLM-specialized runtime token loop
  runtime-managed KV cache and reusable compute buffers
```

This is not only a library difference. It is a runtime architecture difference.

## What Grows In ExecuTorch

When ExecuTorch exports or compiles a decoder with a larger `max_context_len`,
the growing part is mostly the text decoder execution plan.

The rough order is:

```text
1. text_decoder planned activation / workspace / intermediate arena
2. KV-cache
3. backend delegate workspace and packed buffers
4. input/output tensor buffers
```

Weights are mostly independent of context length. The vision encoder is also
mostly independent of text context length.

The risky part is the text decoder because attention and prefill-related
intermediate tensors are planned around the maximum shape that the exported graph
must support.

```text
ExecuTorch decoder memory
= weights/constants
+ KV-cache
+ graph-planned activation arena
+ attention/prefill intermediate buffers
+ backend delegate workspace
+ packed/copied backend buffers
```

The KV-cache is large and predictable:

```text
KV ~= layers * 2(K,V) * n_kv_heads * head_dim * max_context_len * dtype_size
```

However, if observed ExecuTorch memory is much larger than this KV estimate, the
remaining growth is usually from:

```text
decoder graph memory plan
+ activation/intermediate buffers
+ XNNPACK/Vulkan/QNN workspace
```

In short, the phrase "graph preparation memory" is mostly correct, but the more
precise term is:

```text
max-shape decoder activation/workspace memory
```

## What Happens If ExecuTorch Prefills 8192 Tokens

There are two different cases:

```text
1. compile/export with max_context_len = 8192
2. actually prefill 8192 tokens
```

Case 1 already increases baseline memory because the graph is prepared for the
maximum supported context.

Case 2 can further increase runtime peak memory because the long prefill path
activates large attention, mask, projection, reshape, transpose, and backend
workspace buffers.

So for ExecuTorch:

```text
large compiled context:
  baseline memory increases

large actual prefill:
  peak memory and latency can both increase
```

This makes long-context decoder streaming hard in a general AOT graph runtime.

## Why llama.cpp Does Not Grow The Same Way

llama.cpp does not compile the decoder into one large max-context graph. It is a
C/C++ LLM runtime that calls `llama_decode()` with the current input batch.

For example, if the runtime is started with:

```text
n_ctx = 8192
```

that means the KV-cache can hold up to 8192 token positions. It does not mean
every call computes 8192 tokens.

If the actual prompt is 100 tokens:

```text
prefill:
  llama_decode(batch of 100 tokens)
```

During generation, llama.cpp usually decodes one new token at a time:

```text
decode step 1:
  llama_decode(batch of 1 token)

decode step 2:
  llama_decode(batch of 1 token)
```

Past tokens are not recomputed. Their K/V tensors are already resident in the
KV-cache and are referenced by the new token.

So `n_ctx` is mainly capacity:

```text
n_ctx = maximum KV-cache capacity
actual compute = current batch/token count
```

This is why a good description for llama.cpp is:

```text
runtime-managed token loop
```

or:

```text
LLM-specialized dynamic runtime
```

Here, "dynamic" does not mean PyTorch eager-style dynamic graph construction. It
means the runtime handles token count, batch size, positions, and KV-cache state
at execution time.

## What Happens If llama.cpp Prefills 8192 Tokens

If llama.cpp is already running with:

```text
n_ctx = 8192
```

then KV-cache capacity has already been allocated. Prefilling 100 tokens or 8192
tokens does not change the KV-cache capacity allocation much.

What does change strongly is:

```text
latency
energy
thermal pressure
```

llama.cpp can also split long prefill into smaller chunks using runtime batch
controls such as `n_batch` and `n_ubatch`.

Conceptually:

```text
8192-token prompt
-> process 512 tokens
-> process 512 tokens
-> process 512 tokens
-> ...
```

This keeps memory closer to:

```text
weights
+ KV-cache capacity
+ reusable compute buffer
```

The biggest memory lever in llama.cpp is usually context length `-c`, not output
length `-n`.

```text
-c 8192   -> KV-cache capacity for 8192 positions
-n 128    -> generate up to 128 new tokens
-n 1024   -> generate up to 1024 new tokens
```

Changing `-n` mostly changes time and energy. Changing `-c` changes KV-cache
capacity and therefore memory.

## Why AOT Is Still Useful

AOT is not bad. It is useful when the workload is shape-stable and the backend
can optimize ahead of time.

ExecuTorch AOT can know the following before runtime:

```text
operator graph
tensor shape ranges
dtype
backend target
memory plan
operator placement
```

This enables:

```text
op fusion
layout conversion
weight packing
delegate-specific memory planning
NPU/GPU-friendly graph lowering
low runtime overhead
```

This is especially valuable for mobile accelerators such as:

```text
QNN / HTP
Vulkan
XNNPACK
```

The vision encoder is a good AOT target because its input shape is often fixed:

```text
image size: 448 x 448
vision tokens: 256
```

That is why ExecuTorch can be a better fit than llama.cpp for mobile vision
backend experiments.

## Streaming VLM Implication

Streaming VLM has two different workloads:

```text
continuous vision encoding:
  fixed-ish shape, accelerator-friendly

long-context decoder / KV-cache management:
  variable length, runtime stateful, memory-sensitive
```

Therefore, the current recommended split is:

```text
vision encoder:
  ExecuTorch AOT backend path
  QNN / Vulkan / XNNPACK experiments

decoder:
  llama.cpp runtime-managed token loop
  KV-cache experiments and long-context control
```

The practical integration boundary is not KV-cache transfer. KV-cache layout,
position handling, quantization, and backend ownership are runtime-specific.

The practical boundary is:

```text
ExecuTorch vision embeddings -> llama.cpp decoder prefill
```

## Terminology To Use

Recommended terms:

```text
ExecuTorch:
  AOT static-graph runtime
  general graph runtime
  max-shape graph memory planning

llama.cpp:
  LLM-specialized runtime
  runtime-managed token loop
  runtime-managed KV-cache
  dynamic token runtime
```

Avoid saying only "static vs dynamic" without context, because "dynamic" can be
confused with PyTorch eager dynamic graph execution.

The clearer contrast is:

```text
AOT static graph
vs
runtime-managed token loop
```
