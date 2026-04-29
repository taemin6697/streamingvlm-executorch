# Mobile Backend Flow

This guide keeps ExecuTorch as a clean upstream dependency and places project-specific VLM code
under `my_research/foundation`.

Supported backend targets:

- `xnnpack`: CPU backend, currently implemented.
- `vulkan`: Android GPU backend, currently implemented for export/run experiments.
- `qnn`: Qualcomm NPU backend, currently implemented.

## 1. Directory

```text
/workspace/streamingvlm/
  executorch/                 # clean upstream ExecuTorch checkout
  my_research/
    foundation/
      README.md
      CMakeLists.txt
      cli.py
      export.py
      manifest.py
      docs/
        mobile_backend_flow.md
      scripts/
        build_backend_and_runner.sh
      exporters/
        xnnpack.py
        qnn.py
        # Vulkan export is handled by the XNNPACK/Vulkan split exporter path.
      models/
        internvl3/              # project-local InternVL3 model/export helpers
      host/
        launcher.py
      runner/
        backend.h
        xnnpack_qnn_runner.cpp
        xnnpack_backend.cpp
        qnn_backend.cpp
        # Vulkan currently shares the XNNPACK split runner implementation.
      results/
        model/
          hf/                   # local HF model/checkpoint inputs
          xnnpack/              # XNNPACK exported artifacts
          vulkan/               # Vulkan exported artifacts
          qnn/                  # QNN exported artifacts
        log/                    # run/build logs and pulled outputs
```

Runtime artifacts use a common manifest contract:

```text
artifact_root/
  manifest.json
  tokenizer/
    tokenizer.json
  models/
    vision_encoder.pte
    text_embedding.pte
    text_decoder.pte
```

Important manifest fields:

- `backend`: `xnnpack`, `vulkan`, or `qnn`
- `runner_type`: currently `multimodal_split`
- `paths.vision_encoder_pte`
- `paths.text_embedding_pte`
- `paths.text_decoder_pte`
- `paths.tokenizer_path`

## 2. Environment / Backend Build Setup

Start from the project root:

```bash
cd /workspace/streamingvlm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stream

export PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch
export ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-/opt/android-ndk-r26c}
```

For QNN:

```bash
export QNN_SDK_ROOT=/path/to/qnn_sdk
```

Check the foundation CLI:

```bash
python -m my_research.foundation.cli --help
```

### 2.1 Build Backend Tree and Runner with Superbuild

Use `unified` as the default build path. It builds one Android ExecuTorch tree
with XNNPACK, Vulkan, and QNN enabled, then builds the project-local foundation
runner against that tree.

```bash
cd /workspace/streamingvlm

# Default: XNNPACK + Vulkan + QNN in one Android build tree and one runner
my_research/foundation/scripts/build_backend_and_runner.sh unified

# Legacy/debug split builds. Use these only when isolating backend build issues.
# XNNPACK + Vulkan backend tree, then foundation runner
my_research/foundation/scripts/build_backend_and_runner.sh xnnpack-vulkan

# QNN backend tree, then foundation runner
my_research/foundation/scripts/build_backend_and_runner.sh qnn

# Build both groups sequentially
my_research/foundation/scripts/build_backend_and_runner.sh all
```

Useful overrides:

- `JOBS=16`
- `SKIP_ET_BUILD=1` to rebuild only the foundation runner
- `SKIP_RUNNER_BUILD=1` to build only the ExecuTorch backend tree
- `EXECUTORCH_ROOT=/path/to/executorch`
- `ANDROID_NDK_ROOT=/path/to/android-ndk`

## 3. Runner Build

The default runner location is:

```text
/workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner
```

Legacy split runner locations, used only for backend-isolation debugging, are:

```text
/workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner
/workspace/streamingvlm/executorch/build-android/foundation/xnnpack_qnn_runner
```

If only runner code changed, rebuild just the runner:

```bash
cd /workspace/streamingvlm

SKIP_ET_BUILD=1 my_research/foundation/scripts/build_backend_and_runner.sh unified

# Legacy/debug split rebuilds:
SKIP_ET_BUILD=1 my_research/foundation/scripts/build_backend_and_runner.sh xnnpack-vulkan
SKIP_ET_BUILD=1 my_research/foundation/scripts/build_backend_and_runner.sh qnn
```

## 4. Model Export

Export after the target backend build tree exists.

KV-cache allocation must be controlled at export time. For memory experiments,
export artifacts with KV quantization enabled:

- XNNPACK/Vulkan: pass `--quantize_kv_cache`. This uses upstream int8 per-token
  `QuantizedKVCache`.
- QNN: use a quantized QNN path and pass `--qnn_kv_quant 8`. Do not use true QNN
  `fp16` compile for memory-controlled KV baselines because that path skips
  PTQ/QDQ and cannot apply the 8-bit KV annotation.

Without these flags, AoT export can reserve full-precision KV-cache buffers
based on `max_context_len`, so the runtime memory footprint is not controlled by
the intended KV-cache budget.

### 4.1 XNNPACK Export

```bash
cd /workspace/streamingvlm

python -m my_research.foundation.cli export \
  --backend xnnpack \
  --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_xnnpack_1b_8k_fp16_test \
  --decoder_model internvl3_1b \
  --model_path /workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --checkpoint /workspace/streamingvlm/my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth \
  --max_seq_len 8192 \
  --max_context_len 8192 \
  --dtype fp32 \
  --vision_quant fp16 \
  --decoder_quant fp16 \
  --embedding_quant fp16 \
  --quantize_kv_cache
```

### 4.2 Vulkan Export

Vulkan uses the same split exporter as XNNPACK. The stable path is still
`--dtype fp32` or the existing validated fp16 artifacts. Experimental
`--dtype fp16` now records `vulkan_export_dtype=fp16`, but this path has shown
generation correctness issues and should be treated as a debugging artifact.

```bash
cd /workspace/streamingvlm

PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp16_4kv \
  --decoder_model internvl3_1b \
  --model_path /workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --checkpoint /workspace/streamingvlm/my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth \
  --max_seq_len 8192 \
  --max_context_len 8192 \
  --dtype fp32 \
  --vision_quant fp16 \
  --decoder_quant fp16 \
  --embedding_quant fp16 \
  --decoder_input_mode embeddings \
  --dynamic_shape \
  --use_sdpa_with_kv_cache \
  --quantize_kv_cache
```

Current Vulkan component support:

- Vision: `fp16` only in the regular exporter.
- Text embedding: `fp16` only in the regular exporter.
- Text decoder:
  - `fp16`
  - `vulkan_8w` / `8w`: Vulkan PT2E linear weight-only int8
  - `4w`: upstream Llama `qmode` source-transform weight-only int4
  - `8da8w`: upstream Llama `qmode` source-transform int8 dynamic activation + int8 weight
  - `8da4w`: upstream Llama `qmode` source-transform int8 dynamic activation + int4 weight
- Dynamic shape is supported and is the default.
- KV-cache quantization: use `--quantize_kv_cache` for memory-controlled
  artifacts. This is upstream int8 per-token `QuantizedKVCache`.

Important: `8da4w`/`8da8w` must follow the upstream Llama `-qmode` path, not a
Vulkan PT2E graph quantizer path. The PT2E dynamic-activation Vulkan path leaves
`quantized_decomposed.quantize_per_tensor.tensor` outside the Vulkan partition
with dynamic shapes and fails during `to_executorch()`. The `qmode` path lowers
to Vulkan ops such as `et_vk.linear_dq8ca_q4gsw` and has exported/run
successfully for the 512-token decoder-only test.

These Vulkan quant modes apply to decoder `linear` operators. They do not
quantize KV-cache; use `--quantize_kv_cache` separately for KV-cache memory.

InternVL3 Vulkan vision uses a local overlay for backend compatibility:

- Attention is rewritten as explicit `bmm -> softmax -> bmm`.
- MLP GELU is rewritten as a tanh-form primitive expression because the Vulkan
  `aten.gelu` path produced NaNs in deeper vision layers.

This fix is part of the default Vulkan export path; `_fix` artifact names are no
longer required for new exports.

### 4.3 QNN Export

```bash
cd /workspace/streamingvlm

PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python -m my_research.foundation.cli export \
  --backend qnn \
  --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/qnn/internvl3_1b_qnn_512_16a16w \
  --decoder_model internvl3_1b \
  -b executorch/build-android-unified \
  -s R3KYC01FW1P \
  -m SM8750 \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 512 \
  --max_context_len 512 \
  --dtype fp32 \
  --vision_quant 16a16w \
  --decoder_quant 16a16w \
  --embedding_quant 16a16w \
  --qnn_kv_quant 8 \
  --prompts "Can you describe this image?" \
  --image_path "http://images.cocodataset.org/val2017/000000039769.jpg"
```

QNN export supports `fp16`, `16a16w`, `16a8w`, `16a4w`, `16a4w_block`,
`8a8w`, and `8a4w`. In this project, QNN `fp16` now means QNN HTP fp16
compile precision via `generate_htp_compiler_spec(use_fp16=True)`. Use explicit
`16a16w` when you want the old quantized 16a16w QDQ path.

QNN decoder KV I/O can be forced to 8-bit on quantized paths with
`--qnn_kv_quant 8`. This appends Qualcomm's `annotate_kv_8bit` to the decoder
quant recipe. It is not available for the true QNN `fp16` compile path because
that path skips PTQ/QDQ.

For memory-controlled QNN baselines, prefer a quantized mode such as `16a8w` or
`16a16w` plus `--qnn_kv_quant 8`. Treat true QNN `fp16` as a compiler precision
experiment, not as a KV-memory-controlled baseline.

The current QNN export flow requires a connected Android device and SoC model
at compile time (`--device` and `--model`).

### 4.4 Batch Export

Use the matrix script for repeated context-length exports:

```bash
cd /workspace/streamingvlm

PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
EXPORT_MODELS="internvl3_1b" \
EXPORT_LENGTHS="512 1024 2048 4096 8192 16384" \
XNNPACK_EXTRA_ARGS="--quantize_kv_cache" \
my_research/foundation/scripts/export_internvl3_matrix.sh xnnpack
```

QNN `16a16w` matrix:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
EXPORT_MODELS="internvl3_1b" \
EXPORT_LENGTHS="512 1024 2048 4096 8192 16384" \
QNN_DEVICE=R3KYC01FW1P \
QNN_SOC_MODEL=SM8750 \
QNN_BUILD_PATH=/workspace/streamingvlm/executorch/build-android-unified \
QNN_QUANT=16a16w \
QNN_EXTRA_ARGS="--qnn_kv_quant 8" \
my_research/foundation/scripts/export_internvl3_matrix.sh qnn
```

QNN true HTP fp16 matrix for compiler-precision debugging only. Do not use this
as a memory-controlled KV baseline because `--qnn_kv_quant 8` is not available
on the true fp16 path:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
EXPORT_MODELS="internvl3_1b" \
EXPORT_LENGTHS="512 1024 2048 4096 8192 16384" \
QNN_DEVICE=R3KYC01FW1P \
QNN_SOC_MODEL=SM8750 \
QNN_BUILD_PATH=/workspace/streamingvlm/executorch/build-android-unified \
QNN_QUANT=fp16 \
my_research/foundation/scripts/export_internvl3_matrix.sh qnn
```

Vulkan fp16 matrix:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
EXPORT_MODELS="internvl3_1b" \
EXPORT_LENGTHS="512 1024 2048 4096 8192 16384" \
VULKAN_DTYPE=fp16 \
VULKAN_VISION_QUANT=fp16 \
VULKAN_DECODER_QUANT=fp16 \
VULKAN_EMBEDDING_QUANT=fp16 \
VULKAN_DECODER_INPUT_MODE=embeddings \
VULKAN_EXTRA_ARGS="--quantize_kv_cache" \
my_research/foundation/scripts/export_internvl3_matrix.sh vulkan
```

Useful overrides:

- `EXPORT_LENGTHS="1024 2048 4096"`
- `EXPORT_MODELS="internvl3_1b internvl3_2b"`
- `QNN_DEVICE=...`
- `QNN_BUILD_PATH=...`
- `QNN_SOC_MODEL=SM8750`
- `VULKAN_DECODER_QUANT=8da4w`
- `XNNPACK_EXTRA_ARGS="--quantize_kv_cache"`
- `VULKAN_EXTRA_ARGS="--quantize_kv_cache"`
- `QNN_EXTRA_ARGS="--qnn_kv_quant 8"` for quantized QNN modes
- `SKIP_EXISTING=1`

## 5. Inspect / Run

### 5.1 Inspect Manifest

```bash
python -m my_research.foundation.cli inspect-manifest \
  /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_xnnpack_1b_2k/manifest.json
```

Check that these paths exist:

- `vision_encoder_pte`
- `text_embedding_pte`
- `text_decoder_pte`
- `tokenizer_path`

### 5.2 Run XNNPACK

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_xnnpack_1b_2k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --force_generate_token 128 \
  --temperature 0.0 \
  --save_log
```

Run examples omit `--seq_len` so the launcher reads the artifact maximum from
`manifest.export.max_seq_len`. `--force_generate_token 128` is optional but useful
for profiling: it ignores EOS and stop tokens and generates exactly 128 decode
tokens. Without it, `--seq_len` remains the normal maximum generation length and
EOS can stop generation early.

For XNNPACK, dynamic and static artifacts use different runner paths:

- Dynamic shape artifact: prompt prefill can run with the actual prompt length,
  and decode uses 1-token inputs.
- Static shape artifact: matches upstream ExecuTorch static KV-cache behavior.
  Parallel prefill is disabled, so the runner feeds one token/image embedding
  row at a time. Re-export static artifacts after this behavior change; older
  `_static` artifacts may have been fixed to a 256-token example shape.

`--save_log` stores run output and memory logs under:

```text
my_research/foundation/results/log/<backend>/<artifact_dir_name>/
  foundation_output.txt
  foundation_proc.csv
  android_memory_timeline.csv
  vision_output_stats.csv          # if vision output dump support is enabled
  vision_output_0000_f32.bin       # if vision output dump support is enabled
  memory_timeline_plot.png
```

The launcher keeps a model-specific cache on the Android device to avoid re-pushing large `.pte`
files every run:

```text
/data/local/tmp/foundation_runner/<artifact_dir_name>/
```

Add `--force_push` when you need to refresh the cached runner/model files.

### 5.3 Run Vulkan

Full Vulkan image run:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --force_generate_token 128 \
  --temperature 0.0 \
  --save_log
```

The unified runner links upstream `portable_ops_lib` when available so XNNPACK
and Vulkan fallback operators are registered even when QNN is enabled. It also
whole-archives `custom_ops` for Llama custom ops such as `llama::custom_sdpa.out`
and `llama::update_cache.out`.

Text-only Vulkan run with an embedding-input artifact:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --questions "Briefly explain what a vision-language model is." \
  --force_generate_token 128 \
  --temperature 0.0 \
  --save_log
```

Hybrid image run using XNNPACK vision and Vulkan text embedding/decoder:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_hybrid_xnnpack_vision_vulkan_embedding_decoder_fp16_1k/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --force_generate_token 128 \
  --temperature 0.0 \
  --save_log
```

This hybrid path is still useful as a comparison baseline, but full Vulkan image
runs should now work after the Vulkan-friendly GELU overlay fix.

### 5.4 Run QNN

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/qnn/internvl3_1b_qnn_2k_16a8w/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  -b executorch/build-android-unified \
  -s R3KYC01FW1P \
  -m SM8750 \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --force_generate_token 128 \
  --temperature 0.0 \
  --save_log
```

For video input, replace `--image` with `--video /workspace/streamingvlm/sample.mp4`.

## 6. Current Status

- XNNPACK path: implemented.
- QNN path: implemented.
- Vulkan path: implemented for export/run experiments. Text-only Vulkan decoder
  runs work, and full image Vulkan runs now work after the InternVL3 vision
  attention/GELU overlay fixes.
- Unified XNNPACK + Vulkan + QNN build path: implemented and now the default
  runner path under `executorch/build-android-unified`. XNNPACK, Vulkan, and QNN
  image runs have all been smoke-tested with the unified runner.
- The common artifact contract assumes split PTEs: vision encoder, text embedding, text decoder.
- The current runner path is batch-oriented. A full streaming loop is not implemented yet.
- ExecuTorch source should remain clean. Project-specific changes belong in `my_research/foundation`.
