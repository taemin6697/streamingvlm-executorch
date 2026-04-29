# Unified VLM Foundation

Unified VLM Foundation is the project-local mobile execution layer for running
split vision-language models with ExecuTorch on Android. It keeps ExecuTorch as
an upgradeable upstream dependency and puts research-specific build scripts,
exporters, manifests, host launchers, and runners under `my_research/foundation`.

The current target model is InternVL3-1B, exported as three `.pte` programs:
vision encoder, text embedding, and text decoder. The Android runner can load the
same manifest contract across XNNPACK, Vulkan, and QNN experiments.

## Highlights

- Unified Android runner for XNNPACK, Vulkan, and QNN:
  `executorch/build-android-unified/foundation/xnnpack_qnn_runner`
- Project-local overlays only. The original ExecuTorch source should stay clean.
- Backend-specific exporters for XNNPACK, Vulkan, and Qualcomm QNN.
- Manifest-based artifact layout so host launch commands do not need to know
  individual `.pte` filenames.
- Android host launcher support for pushing models, QNN runtime libraries,
  frames, logs, and memory traces.
- QNN quantization experiments including `fp16`, `16a16w`, `16a8w`, `16a4w`,
  `16a4w_block`, `8a8w`, and `8a4w`.

## Repository Layout

```text
my_research/foundation/
  cli.py                         # export/run/inspect command entry point
  CMakeLists.txt                 # project-local Android runner build
  exporters/
    xnnpack.py                   # XNNPACK and Vulkan split export path
    qnn.py                       # QNN export overlay around Qualcomm examples
  host/
    launcher.py                  # Android push/run/log collection
  runner/
    xnnpack_qnn_runner.cpp       # shared multimodal runner executable
    xnnpack_backend.cpp
    qnn_backend.cpp
  scripts/
    build_backend_and_runner.sh  # Android backend tree + runner superbuild
    export_internvl3_matrix.sh   # length/quant export matrix helper
  docs/
    mobile_backend_flow.md       # detailed operational guide
    for_cursor_llm.md            # implementation history and caveats
```

## Setup

From the repository root:

```bash
cd /workspace/streamingvlm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stream

export PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch
export ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-/opt/android-ndk-r26c}
export QNN_SDK_ROOT=/path/to/qnn_sdk   # required for QNN export/run
```

Check the CLI:

```bash
python -m my_research.foundation.cli --help
```

## Build The Unified Runner

Use the unified build as the default path. It builds one Android ExecuTorch tree
with XNNPACK, Vulkan, and QNN enabled, then builds the foundation runner against
that tree.

```bash
my_research/foundation/scripts/build_backend_and_runner.sh unified
```

Expected runner:

```text
/workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner
```

If only runner code changed:

```bash
SKIP_ET_BUILD=1 my_research/foundation/scripts/build_backend_and_runner.sh unified
```

## Export Examples

### XNNPACK

```bash
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

### Vulkan

```bash
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_2k_fp16 \
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

### QNN

For QNN, `fp16` means QNN HTP fp16 compile precision. Use explicit `16a16w`
when you want the quantized 16a16w QDQ path.

```bash
python -m my_research.foundation.cli export \
  --backend qnn \
  --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/qnn/internvl3_1b_qnn_512_16a8w \
  --decoder_model internvl3_1b \
  -b executorch/build-android-unified \
  -s R3KYC01FW1P \
  -m SM8750 \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 512 \
  --max_context_len 512 \
  --dtype fp32 \
  --vision_quant 16a8w \
  --decoder_quant 16a8w \
  --embedding_quant 16a8w \
  --prompts "Can you describe this image?" \
  --image_path "http://images.cocodataset.org/val2017/000000039769.jpg"
```

## Run On Android

All backends use the same runner binary. QNN runs also need the unified build
tree path so the launcher can push `libqnn_executorch_backend.so`.

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/qnn/internvl3_1b_qnn_512_16a8w/manifest.json \
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

Run examples omit `--seq_len` so the launcher uses `manifest.export.max_seq_len`
as the generation limit. `--force_generate_token 128` is useful for profiling:
it makes the runner emit exactly 128 decode tokens even if EOS appears earlier;
it also stops after 128 tokens if EOS never appears.

For XNNPACK or Vulkan, use the matching manifest and the same unified runner:

```bash
python -m my_research.foundation.cli run \
  --manifest /path/to/xnnpack_or_vulkan/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --force_generate_token 128 \
  --temperature 0.0 \
  --save_log
```

Logs and plots are pulled to:

```text
my_research/foundation/results/log/<backend>/<artifact_name>/
```

## QNN Quantization Notes

Supported QNN modes:

```text
fp16        # HTP fp16 compile precision
16a16w      # quantized 16-bit activation, 16-bit weight
16a8w       # quantized 16-bit activation, 8-bit weight
16a4w
16a4w_block
8a8w        # typical INT8 activation + INT8 weight mode
8a4w
```

Use `16a8w` as the current practical QNN baseline unless an experiment needs a
specific quantization mode. In local smoke tests, `16a8w` produced normal
InternVL3 captions for 512 and 2K artifacts, while the quantized `16a16w` 512
artifact generated an abnormal short output.

## Detailed Documentation

- `docs/mobile_backend_flow.md`: full build, export, run, and troubleshooting flow.
- `docs/dynamic_shape_kv_cache.md`: dynamic shape and KV-cache notes.
- `docs/connect_android.md`: Android connection commands.
- `docs/for_cursor_llm.md`: implementation history, known fixes, and future-agent notes.

## Development Principles

- Do not modify upstream ExecuTorch unless there is no practical alternative.
- Prefer wrappers, overlays, local scripts, and project-local CMake changes.
- Do not commit generated artifacts, build outputs, QNN SDK files, or large model
  binaries.
- Keep `docs/mobile_backend_flow.md` and `docs/for_cursor_llm.md` updated when
  changing workflows or fixing backend issues.
