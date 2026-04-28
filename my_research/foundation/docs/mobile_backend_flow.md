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

Use this as the default build path. The script builds the upstream ExecuTorch backend tree first,
then builds the project-local foundation runner against that tree.

```bash
cd /workspace/streamingvlm

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

The superbuild script already builds the runner. The expected runner locations are:

```text
/workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner
/workspace/streamingvlm/executorch/build-android/foundation/xnnpack_qnn_runner
```

If only runner code changed, rebuild just the runner:

```bash
cd /workspace/streamingvlm

SKIP_ET_BUILD=1 my_research/foundation/scripts/build_backend_and_runner.sh xnnpack-vulkan
SKIP_ET_BUILD=1 my_research/foundation/scripts/build_backend_and_runner.sh qnn
```

## 4. Model Export

Export after the target backend build tree exists.

### 4.1 XNNPACK Export

```bash
cd /workspace/streamingvlm

python -m my_research.foundation.cli export \
  --backend xnnpack \
  --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_xnnpack_1b_2k_fp16 \
  --decoder_model internvl3_1b \
  --model_path /workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --checkpoint /workspace/streamingvlm/my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth \
  --max_seq_len 1024 \
  --max_context_len 1024 \
  --dtype fp16 \
  --vision_quant fp16 \
  --decoder_quant fp16 \
  --embedding_quant fp16
```

### 4.2 Vulkan Export

Vulkan uses the same split exporter as XNNPACK. `--dtype fp16` maps to
`vulkan_export_dtype=fp32` plus Vulkan `force_fp16=true`, matching upstream
Llama-style Vulkan export behavior.

```bash
cd /workspace/streamingvlm

PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_fp16 \
  --decoder_model internvl3_1b \
  --model_path /workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --checkpoint /workspace/streamingvlm/my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth \
  --max_seq_len 1024 \
  --max_context_len 1024 \
  --dtype fp16 \
  --vision_quant fp16 \
  --decoder_quant fp16 \
  --embedding_quant fp16 \
  --decoder_input_mode embeddings \
  --dynamic_shape \
  --use_sdpa_with_kv_cache
```

Current Vulkan component support:

- Vision: `fp16` only in the regular exporter.
- Text embedding: `fp16` only in the regular exporter.
- Text decoder: `fp16` or `8da4w`.
- Dynamic shape is supported and is the default.

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
  -b executorch/build-android \
  -s R3CX50FQ62L \
  -m SM8750 \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 512 \
  --max_context_len 512 \
  --dtype fp32 \
  --vision_quant 16a16w \
  --decoder_quant 16a16w \
  --embedding_quant 16a16w \
  --prompts "Can you describe this image?" \
  --image_path "http://images.cocodataset.org/val2017/000000039769.jpg"
```

QNN quantization modes use Qualcomm-style names: `16a16w`, `16a8w`,
`16a4w`, `16a4w_block`, `8a8w`, and `8a4w`. For QNN only, legacy
`fp16` is treated as an alias for `16a16w`, which makes comparisons with
XNNPACK fp16 exports explicit.

The current QNN export flow requires a connected Android device and SoC model
at compile time (`--device` and `--model`).

### 4.4 Batch Export

Use the matrix script for repeated context-length exports:

```bash
cd /workspace/streamingvlm

PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
EXPORT_MODELS="internvl3_1b" \
EXPORT_LENGTHS="512 1024 2048 4096 8192 16384" \
my_research/foundation/scripts/export_internvl3_matrix.sh xnnpack
```

QNN `16a16w` matrix:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
EXPORT_MODELS="internvl3_1b" \
EXPORT_LENGTHS="512 1024 2048 4096 8192 16384" \
QNN_DEVICE=R3CX50FQ62L \
QNN_SOC_MODEL=SM8750 \
QNN_QUANT=16a16w \
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
my_research/foundation/scripts/export_internvl3_matrix.sh vulkan
```

Useful overrides:

- `EXPORT_LENGTHS="1024 2048 4096"`
- `EXPORT_MODELS="internvl3_1b internvl3_2b"`
- `QNN_DEVICE=...`
- `QNN_BUILD_PATH=...`
- `QNN_SOC_MODEL=SM8750`
- `VULKAN_DECODER_QUANT=8da4w`
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
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_xnnpack_1b_1k_fp16_static/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner \
  --device R3CX50FQ62L \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

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
  --runner_binary /workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner \
  --device R3CX50FQ62L \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

Text-only Vulkan run with an embedding-input artifact:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner \
  --device R3CX50FQ62L \
  --questions "Briefly explain what a vision-language model is." \
  --seq_len 128 \
  --temperature 0.0 \
  --save_log
```

Hybrid image run using XNNPACK vision and Vulkan text embedding/decoder:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_hybrid_xnnpack_vision_vulkan_embedding_decoder_fp16_1k/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner \
  --device R3CX50FQ62L \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

This hybrid path is still useful as a comparison baseline, but full Vulkan image
runs should now work after the Vulkan-friendly GELU overlay fix.

### 5.4 Run QNN

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/qnn/internvl3_1b_qnn_2k_16a8w/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android/foundation/xnnpack_qnn_runner \
  -b executorch/build-android \
  -s R3CX50FQ62L \
  -m SM8750 \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 2048 \
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
- The common artifact contract assumes split PTEs: vision encoder, text embedding, text decoder.
- The current runner path is batch-oriented. A full streaming loop is not implemented yet.
- ExecuTorch source should remain clean. Project-specific changes belong in `my_research/foundation`.
