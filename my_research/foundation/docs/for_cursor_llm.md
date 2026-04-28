# Foundation Implementation Tracking

This document is for Cursor/LLM implementation tracking. Keep it updated whenever the foundation
layer changes. The goal is to preserve the reasoning behind each change so future edits can avoid
re-discovering the same context.

## Core Rule

ExecuTorch should remain a clean upstream dependency.

- Do not add project-specific code directly under `executorch/` unless there is no practical
  alternative.
- Put project-specific Python, C++, scripts, docs, and model adapters under
  `my_research/foundation/`.
- If an upstream ExecuTorch or Qualcomm file must be changed, prefer tracking that change as a
  small patch rather than copying an entire directory.

## Current Layout

```text
my_research/foundation/
  README.md
  CMakeLists.txt
  cli.py
  export.py
  manifest.py
  docs/
    mobile_backend_flow.md
    for_cursor_llm.md
  scripts/
    build_backend_and_runner.sh
    export_internvl3_matrix.sh
  exporters/
    xnnpack.py
    qnn.py
  models/
    internvl3/
      1b_config.json
      2b_config.json
      8b_config.json
      convert_weights.py
      vision_encoder/
        model.py
        export_xnnpack.py
  host/
    launcher.py
  runner/
    backend.h
    xnnpack_qnn_runner.cpp
    xnnpack_backend.cpp
    qnn_backend.cpp
  results/
    model/
      hf/
      xnnpack/
      vulkan/
      qnn/
    log/
```

## Changes Already Made

### 1. Moved Foundation Out of ExecuTorch

The original foundation code was treated as if it lived under:

```text
executorch/examples/models/foundation/
```

It is now project-local:

```text
my_research/foundation/
```

Updated imports from:

```python
executorch.examples.models.foundation...
```

to:

```python
my_research.foundation...
```

Affected files include:

- `my_research/foundation/cli.py`
- `my_research/foundation/export.py`
- `my_research/foundation/exporters/__init__.py`
- `my_research/foundation/exporters/xnnpack.py`
- `my_research/foundation/host/launcher.py`
- `my_research/foundation/__init__.py`

### 2. C++ Runner Now Uses Local Headers

The runner code no longer includes foundation headers through ExecuTorch paths.

Changed includes from:

```cpp
#include <executorch/examples/models/foundation/runner/backend.h>
```

to:

```cpp
#include "backend.h"
```

Affected files:

- `runner/xnnpack_qnn_runner.cpp`
- `runner/xnnpack_backend.cpp`
- `runner/qnn_backend.cpp`

### 3. CMake Treats ExecuTorch as External Dependency

`my_research/foundation/CMakeLists.txt` now accepts:

- `EXECUTORCH_ROOT`: clean upstream ExecuTorch checkout
- `EXECUTORCH_BUILD_DIR`: already-built Android ExecuTorch build tree

This lets foundation runner build against ExecuTorch without editing ExecuTorch CMake files.

### 4. Added Superbuild Script

Added:

```text
my_research/foundation/scripts/build_backend_and_runner.sh
```

Purpose:

- Build ExecuTorch backend tree first.
- Then build foundation runner against that tree.
- Keep ExecuTorch source untouched.

Supported modes:

```bash
my_research/foundation/scripts/build_backend_and_runner.sh xnnpack-vulkan
my_research/foundation/scripts/build_backend_and_runner.sh qnn
my_research/foundation/scripts/build_backend_and_runner.sh all
```

Useful overrides:

```bash
JOBS=16
SKIP_ET_BUILD=1
SKIP_RUNNER_BUILD=1
EXECUTORCH_ROOT=/path/to/executorch
ANDROID_NDK_ROOT=/path/to/android-ndk
```

### 5. Added InternVL3 Export Matrix Script

Added:

```text
my_research/foundation/scripts/export_internvl3_matrix.sh
```

Purpose:

- Batch export InternVL3 artifacts for model sizes `internvl3_1b`, `internvl3_2b`, and
  `internvl3_8b`.
- Compile/export the default length matrix `512`, `1024`, `2048`, `4096`, `8192`, and
  `16384`.
- Use the project-local CLI module `my_research.foundation.cli`, not the old
  `executorch.examples.models.foundation.cli`.
- Write outputs under the current results layout:
  - `my_research/foundation/results/model/qnn`
  - `my_research/foundation/results/model/xnnpack`

Behavior:

- Supports `all`, `qnn`, and `xnnpack` modes.
- Looks for local HF/checkpoint inputs under
  `my_research/foundation/results/model/hf`.
- Automatically passes local `--model_path` and `--checkpoint` if the expected files exist.
- Supports `SKIP_EXISTING=1` to skip artifact directories that already contain
  `manifest.json`.

### 6. Backend Build Strategy

Build trees are intentionally separated:

```text
executorch/build-android-xnnpack-vulkan
executorch/build-android
```

Rationale:

- XNNPACK and Vulkan can share one Android build tree.
- QNN uses Qualcomm build script and QNN SDK/runtime assumptions, so keep it separate.
- This avoids backend/op registration conflicts and makes updates easier.

### 7. Documentation Reorganized

Main guide:

```text
my_research/foundation/docs/mobile_backend_flow.md
```

Current order:

1. Directory
2. Environment / Backend Build Setup
3. Runner Build
4. Model Export
5. Inspect / Run

Model/result paths are standardized under:

```text
my_research/foundation/results/model/
my_research/foundation/results/log/
```

Examples:

```text
results/model/hf/
results/model/xnnpack/
results/model/vulkan/
results/model/qnn/
results/log/xnnpack/
results/log/vulkan/
results/log/qnn/
```

### 8. Local InternVL3 Model Adapter

The clean upstream ExecuTorch checkout does not contain the custom InternVL3 vision encoder package
that the old exporter expected:

```text
executorch.examples.models.internvl3.vision_encoder
```

Restored the previous custom InternVL3 code into:

```text
my_research/foundation/models/internvl3/
```

Updated XNNPACK exporter imports to use foundation-local model code:

```python
import my_research.foundation.models.internvl3 as internvl3_pkg
from my_research.foundation.models.internvl3 import convert_weights
from my_research.foundation.models.internvl3.vision_encoder.model import (
    load_vision_encoder,
)
```

### 9. Avoided Qualcomm Package Init Side Effects

`models/internvl3/vision_encoder/model.py` originally imported:

```python
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import InternVL3Encoder
from executorch.examples.qualcomm.oss_scripts.llama.model.vision_encoder import InternVL3VisionEncoder
```

That caused `executorch.examples.qualcomm.oss_scripts.llama.__init__` to run, which pulled in
unrelated model packages and triggered a `torchao/torchtune` import mismatch.

Current workaround:

- Define a minimal local `InternVL3Encoder` adapter.
- Load Qualcomm `model/vision_encoder.py` directly from file with `importlib.util`.
- Avoid importing the Qualcomm llama package initializer.

File:

```text
my_research/foundation/models/internvl3/vision_encoder/model.py
```

### 10. InternVL3 Text Decoder Model Class Mapping

ExecuTorch's upstream Llama exporter does not recognize:

```text
internvl3_1b
internvl3_2b
internvl3_8b
```

It failed with:

```text
ValueError: internvl3_1b is not a valid Llama model.
```

Foundation keeps the external variant name as `internvl3_*`, but maps the internal
`llm_config.base.model_class` passed to the upstream Llama exporter to `llama3_2`.

Added helper:

```python
def _llama_export_model_class(decoder_model: str) -> str:
    if decoder_model.startswith("internvl3_"):
        return "llama3_2"
    return decoder_model
```

Applied in:

- `my_research/foundation/exporters/xnnpack.py`
- `my_research/foundation/models/internvl3/export_xnnpack_multimodal.py`

### 11. Run Output Log Path

Changed:

```text
my_research/foundation/host/launcher.py
```

Previous behavior:

- Pulled Android runner output to `foundation_output.txt` in the current shell working directory.
- This made repeated runs overwrite each other and did not follow the documented `results/log`
  layout.

Current behavior:

- Pulls Android runner output to:

```text
my_research/foundation/results/log/<backend>/<artifact_dir_name>/foundation_output.txt
```

Example:

```text
my_research/foundation/results/log/xnnpack/internvl3_xnnpack_1b_1k_fp16/foundation_output.txt
```

Applies to:

- XNNPACK unified runner flow.
- QNN unified runner flow.

### 12. Android Device Cache for Run Artifacts

Changed:

```text
my_research/foundation/cli.py
my_research/foundation/host/launcher.py
```

Purpose:

- Avoid re-pushing large `.pte` files, tokenizer files, runner binary, and QNN libraries on every
  Android run.
- Keep repeated run latency dominated by input frame upload and model execution instead of ADB file
  transfer.

Current behavior:

- Uses a model/artifact-specific Android cache directory:

```text
/data/local/tmp/foundation_runner/<artifact_dir_name>/
```

- If a required remote file already exists, the launcher skips `adb push`.
- The `xnnpack_qnn_runner` binary is always pushed, because it changes often during development and
  stale runner cache can hide newly added behavior such as logging.
- Input frames are still refreshed every run because image/video inputs can change.
- Added CLI flag:

```bash
python -m my_research.foundation.cli run ... --force_push
```

- `--force_push` removes/re-populates the model-specific device cache and pushes all files again.

### 13. Runner Memory and Phase Logging

Changed:

```text
my_research/foundation/runner/internal_memory_sampler.h
my_research/foundation/runner/xnnpack_backend.cpp
my_research/foundation/runner/qnn_backend.cpp
my_research/foundation/runner/xnnpack_qnn_runner.cpp
my_research/foundation/host/launcher.py
my_research/foundation/host/memory_plot.py
my_research/foundation/docs/mobile_backend_flow.md
```

Purpose:

- Restore the old foundation `--save_log` behavior without modifying ExecuTorch source.
- Keep memory/phase logs under the current project-local results layout.

Current behavior:

- `xnnpack_qnn_runner --save_log` writes:

```text
foundation_proc.csv
android_memory_timeline.csv
```

- `internal_memory_sampler.h` samples:
  - `/proc/self/smaps_rollup`
  - `/proc/meminfo`
  - `get_rss_bytes()`
  - optional backend KV metrics fields
- The Python launcher pulls logs to:

```text
my_research/foundation/results/log/<backend>/<artifact_dir_name>/
```

- When `matplotlib` is available, `host/memory_plot.py` generates:

```text
memory_timeline_plot.png
phase_duration_stacked_bar.png
```

- `phase_duration_stacked_bar.png` is a single stacked bar that breaks total runtime into phase
  durations from `foundation_proc.csv`.

Notes:

- XNNPACK logs phase rows for load, vision encode, embedding/merge, prefill, decode, and token
  decode.
- QNN logs load and whole generate-call timing; detailed QNN KV resident byte metrics from the old
  custom Qualcomm runner are not available in the clean upstream `QNNMultimodalRunner` API, so the
  current QNN sampler records process/system memory fields and leaves KV fields empty.

### 14. Backend-Scoped Log Directories

Changed:

```text
my_research/foundation/host/launcher.py
my_research/foundation/docs/mobile_backend_flow.md
```

Current behavior:

- Run logs are grouped by backend first, then artifact name:

```text
my_research/foundation/results/log/xnnpack/<artifact_dir_name>/
my_research/foundation/results/log/vulkan/<artifact_dir_name>/
my_research/foundation/results/log/qnn/<artifact_dir_name>/
```

Reason:

- Keeps XNNPACK, Vulkan, and QNN run outputs separate even when artifact names are similar.

### 15. XNNPACK Static Shape Export Option

Changed:

```text
my_research/foundation/cli.py
my_research/foundation/exporters/xnnpack.py
my_research/foundation/scripts/export_internvl3_matrix.sh
```

Purpose:

- Allow exporting XNNPACK artifacts with dynamic sequence shapes disabled for comparison.

CLI:

```bash
python -m my_research.foundation.cli export ... --disable_dynamic_shape
```

Batch script:

```bash
DYNAMIC_SHAPE=0 my_research/foundation/scripts/export_internvl3_matrix.sh xnnpack
```

Behavior:

- Default remains dynamic shape enabled.
- `DYNAMIC_SHAPE=0` writes XNNPACK artifacts with a `_static` suffix after the dtype tag.
- Static-shape artifacts may not run with the current unified runner without padding or matching
  the fixed export sequence length, because the runner calls the same decoder with both prefill
  length and decode length 1. This path is primarily for export/runtime investigation.

### 16. InternVL3 Batch Export Length Range

Changed:

```text
my_research/foundation/scripts/export_internvl3_matrix.sh
```

Purpose:

- Keep default batch exports bounded to the requested mobile context range.

Behavior:

- Matrix export default lengths remain `512 1024 2048 4096 8192 16384`.
- `32768` is not included by default.
- `EXPORT_LENGTHS="..."` can still override the default range.

### 17. Dynamic Shape and KV-Cache Explanation Doc

Changed:

```text
my_research/foundation/docs/dynamic_shape_kv_cache.md
```

Purpose:

- Add a Korean explanation of what XNNPACK dynamic shape changes during prefill/decode.
- Clarify that decode can be called with one new token, while KV-cache capacity and memory
  planning remain tied to `max_context_len`.
- Record why multiple context-length artifacts are still useful on mobile even with dynamic shape.

### 18. QNN HTP Skel Library and ADSP Path for Android Runs

Changed:

```text
my_research/foundation/host/launcher.py
```

Problem:

- QNN Android run failed during `runner.load()` with:

```text
QnnDsp <E> DspTransport.openSession qnn_open failed, 0x80000406
QnnDsp <E> Failed to load skel, error: 1002
QnnDsp <E> Transport layer setup failed: 14001
Fail to configure Qnn device
```

Cause:

- The launcher pushed `lib/aarch64-android/libQnn*.so` but did not push the HTP skel library from
  `lib/hexagon-v<arch>/unsigned/libQnnHtpV<arch>Skel.so`.
- The Android command only set `LD_LIBRARY_PATH=.`. QNN HTP/DSP transport also needs the skel search
  path exposed through `ADSP_LIBRARY_PATH`.

Fix:

- Added SoC to HTP arch mapping (`SM8750 -> 79`, `SM8650 -> 75`, `SM8550 -> 73`, `SM8450 -> 69`,
  `SM8350 -> 68`) in `_AdbWorkspace`.
- `push_qnn_libs()` now pushes the matching `libQnnHtpV<arch>Skel.so` when present.
- QNN runner command now exports `LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.`.
- Prompt strings are shell-quoted when forwarded to the Android runner.

Note:

- If the selected `QNN_SDK_ROOT` does not contain the matching skel library, the launcher prints a
  warning and the QNN run can still fail with the same transport error. In that case, install/copy
  the full QNN SDK Hexagon runtime folder for the device arch.

### 19. Runner Frame Path Fix for Single Image Inputs

Changed:

```text
my_research/foundation/runner/xnnpack_qnn_runner.cpp
my_research/foundation/runner/qnn_backend.cpp
my_research/foundation/runner/xnnpack_backend.cpp
```

Problem:

- QNN run progressed past backend initialization but failed while loading the input image:

```text
Failed to open input file: /frame_0000.bin
```

Cause:

- The launcher pushes single-image input as `frame_0000.bin` in the runner workspace.
- `normalize_frame_dir("frame_0000.bin")` returned an empty parent directory.
- `frame_path("", 0)` concatenated `"/frame_0000.bin"`, turning the relative file into an invalid
  absolute root path.

Fix:

- `normalize_frame_dir()` now returns `"."` when the frame file is in the current directory.
- `frame_path()` now uses `std::filesystem::path(dir) / "frame_%04d.bin"` and no longer prefixes
  the frame name with `/`.

Verification:

- Rebuilt QNN foundation runner with:

```bash
SKIP_ET_BUILD=1 JOBS=8 my_research/foundation/scripts/build_backend_and_runner.sh qnn
```

- Smoke-tested on device with `seq_len=320`; QNN reached vision encode, prompt prefill, and decode
  successfully.

## Current Known Status

### Implemented

- Foundation CLI entry:
  ```bash
  python -m my_research.foundation.cli
  ```
- XNNPACK exporter path is being actively fixed.
- QNN exporter path exists and treats Qualcomm code as dependency.
- XNNPACK/QNN runner build succeeds against Android build trees.
- Superbuild script exists.
- Docs use `results/model` and `results/log`.

### Planned

- Vulkan exporter:
  ```text
  my_research/foundation/exporters/vulkan.py
  ```
- Vulkan runner backend:
  ```text
  my_research/foundation/runner/vulkan_backend.cpp
  ```
- CLI support for:
  ```bash
  --backend vulkan
  ```
- CMake wiring for Vulkan runner backend.
- Full streaming loop. Current runner path is still batch-oriented.

## Recent Errors and Fixes

### Error: Missing InternVL3 Vision Encoder

Command failed with:

```text
ModuleNotFoundError: No module named 'executorch.examples.models.internvl3.vision_encoder'
```

Fix:

- Restored custom InternVL3 code into `my_research/foundation/models/internvl3`.
- Updated imports in XNNPACK exporter.

### Error: Qualcomm Import Side Effect

Importing the vision encoder caused:

```text
ImportError: cannot import name 'Int8DynamicActivationInt4WeightConfig'
```

Fix:

- Avoid Qualcomm package `__init__`.
- Load only the required Qualcomm `vision_encoder.py` file directly.

### Error: Invalid Llama Model

XNNPACK export failed with:

```text
ValueError: internvl3_1b is not a valid Llama model.
```

Fix:

- Map `internvl3_*` to `llama3_2` only for the upstream Llama exporter model class.
- Keep manifest/export variant as `internvl3_*`.

### Error: `executorch.__file__` Is `None`

XNNPACK export failed while loading the local InternVL3 vision encoder:

```text
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

Cause:

- `executorch` can be imported as a namespace package, where `executorch.__file__` is `None`.
- The previous root discovery assumed `Path(executorch.__file__)`.

Fix:

- Added `_executorch_root()` in `models/internvl3/vision_encoder/model.py`.
- Root discovery now checks, in order:
  1. Project-local `/workspace/streamingvlm/executorch`
  2. `EXECUTORCH_ROOT`
  3. `executorch.__file__`
  4. `executorch.__path__`
- This keeps the local project checkout preferred over the old `/workspace/stream` checkout.

### Error: QNN Export Import Fails Through `torchtune` / `torchao`

QNN export failed while importing:

```python
from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS
```

The Qualcomm llama package initializer pulled in model converters such as Granite/Phi, which import
`torchtune`. The current nightly `torchao==0.18.0.dev20260427+cpu` was incompatible with the pinned
ExecuTorch example `torchtune` commit and failed with missing symbols such as:

```text
torchao.dtypes.nf4tensor
torchao.quantization.Int8DynamicActivationInt4WeightConfig
```

Environment repair:

```bash
python -m pip install --force-reinstall --no-deps torchao==0.16.0
```

Verified versions:

```text
torchao==0.16.0
torchtune==0.0.0
torch==2.11.0+cu129
executorch==1.3.0a0+bf64fa1
```

Verified imports:

- `from torchao.dtypes.nf4tensor import NF4Tensor`
- `from torchao.quantization import Int8DynamicActivationInt4WeightConfig`
- `import torchtune.training.quantization`
- `from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS`

## 20. QNN Detailed Phase Profiling Overlay

Implemented a local QNN multimodal runner overlay so QNN logs can expose phase
rows similar to XNNPACK without modifying upstream ExecuTorch.

What changed:

- Added `my_research/foundation/runner/foundation_qnn_multimodal_runner.{h,cpp}`.
- Wired `my_research/foundation/runner/qnn_backend.cpp` to use
  `ProfiledQNNMultimodalRunner` instead of upstream `example::QNNMultimodalRunner`.
- Added the overlay source to `my_research/foundation/CMakeLists.txt`.
- `foundation_proc.csv` for QNN now records:
  - `L`
  - `V_Encode`
  - `EmbeddingAndMerging`
  - `T_Prefill`
  - `Decode`
  - per-token callback rows as `D`

Why:

- `/workspace/stream` had the desired QNN profiling behavior, but it achieved
  this by modifying files under `executorch/examples/qualcomm/...`.
- The project rule is to keep ExecuTorch clean, so the modified flow was copied
  into a local overlay under `my_research/foundation/runner`.

Notes:

- QNN `dispatch_inputs()` can execute text embedding before image encoding for
  prompts like `<image>...`; the CSV writer adjusts `EmbeddingAndMerging` so the
  reported row does not double-count `V_Encode` in the stacked phase graph.
- Per-token `D` rows are based on token callback timing from the local overlay.
  They are useful for token position/KV growth, but not as exact as modifying
  upstream `TokenGenerator` internals. Upstream ExecuTorch remains untouched.

Verification:

```bash
SKIP_ET_BUILD=1 JOBS=8 my_research/foundation/scripts/build_backend_and_runner.sh qnn
```

Smoke test run:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/qnn/internvl3_qnn_1b_1k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android/foundation/xnnpack_qnn_runner \
  -b executorch/build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

Observed output:

- `my_research/foundation/results/log/qnn/internvl3_qnn_1b_1k_fp16/foundation_proc.csv`
  contains `V_Encode`, `EmbeddingAndMerging`, `T_Prefill`, `Decode`, and `D` rows.
- `phase_duration_stacked_bar.png` and `memory_timeline_plot.png` are regenerated.

Follow-up output-format fix:

- QNN initially wrote only the assistant answer to `foundation_output.txt`, while
  XNNPACK writes the echoed InternVL3 prompt plus the answer.
- Updated `qnn_backend.cpp` to write the same prompt echo format as XNNPACK:
  `<|im_start|>user:`, `FrameN: <img><IMG_CONTEXT>...</img>`, user question,
  `<|im_start|>assistant`, then the generated answer.
- The original prompt list is copied before `prepare_messages()` because upstream
  QNN chat-template preparation mutates prompts by prepending `<image>` tokens for
  model input. The saved output should show the user-facing prompt, not the
  internal QNN dispatch placeholder.

## 21. XNNPACK Static Shape Runner Semantics

Corrected the XNNPACK static-shape interpretation to match upstream ExecuTorch
LLM runner behavior.

What changed:

- `my_research/foundation/exporters/xnnpack.py`
  - Dynamic export still uses a representative `sample_seq_len=min(256, max_seq_len)`.
  - Static export now uses `sample_seq_len=1`.
  - Manifest export metadata records `enable_dynamic_shape` and
    `static_prefill_mode`.
- `my_research/foundation/runner/xnnpack_backend.cpp`
  - Reads `enable_dynamic_shape` from the decoder PTE constant method.
  - Dynamic path remains unchanged: embed/merge the full prompt and call decoder
    once for prefill.
  - Static path now disables parallel prefill and feeds one token at a time.
  - For `<IMG_CONTEXT>` tokens, the runner feeds the corresponding vision hidden
    row directly instead of calling token embedding.
  - Static path validates that `text_embedding_xnnpack.pte` has shape `[1,1]`;
    old `_static` artifacts fixed to `[1,256]` must be re-exported.

Why:

- Upstream ExecuTorch static KV-cache LLM runners do not feed
  `max_context_len`-wide tensors for XNNPACK decode.
- With `enable_dynamic_shape=false`, upstream disables parallel prefill and
  performs sequential 1-token steps.
- QNN fixed-AR/static behavior is different and should not be used as the
  XNNPACK static contract.

Verification:

```bash
SKIP_ET_BUILD=1 JOBS=8 my_research/foundation/scripts/build_backend_and_runner.sh xnnpack-vulkan
```

Dynamic XNNPACK smoke run still passes:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_xnnpack_1b_1k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

Next step for static comparison:

- Re-export static XNNPACK artifacts with `DYNAMIC_SHAPE=0`; existing old static
  artifacts with 256-token text embedding shape are incompatible with the fixed
  static runner contract.

## 22. QNN Quant Mode Selection Overlay

Added project-local QNN quant mode selection without modifying upstream
ExecuTorch/Qualcomm sources.

What changed:

- `my_research/foundation/exporters/qnn.py`
  - Added QNN quant mode normalization for `fp16`, `16a16w`, `16a8w`,
    `16a4w`, `16a4w_block`, `8a8w`, and `8a4w`.
  - For QNN, `fp16` is treated as an alias for `16a16w`.
  - Injects local decoder and vision quant recipe classes into the
    `internvl3_1b` config only for the duration of `qnn_compile()`.
  - Wraps `llm_wrappers.make_quantizer()` during export so the token embedding
    quantizer follows `--embedding_quant` instead of upstream's hard-coded
    `QuantDtype.use_16a8w`.
- `my_research/foundation/cli.py`
  - Updated quant argument help text to show QNN-style quant names.
- `my_research/foundation/docs/mobile_backend_flow.md`
  - Updated the QNN export example to use `16a16w`.

Why:

- XNNPACK `fp16` artifacts are effectively 16-bit activation / 16-bit weight
  for comparison purposes.
- Upstream Qualcomm InternVL3 recipes default to `use_16a8w`, so a command that
  looked like `--decoder_quant fp16` still printed `use_16a8w/PTQ`.
- Explicit `16a16w` export is needed for a fairer XNNPACK-vs-QNN precision
  comparison.

Verification:

```bash
python -m py_compile my_research/foundation/cli.py my_research/foundation/exporters/qnn.py
```

Recipe override smoke check:

```bash
python - <<'PY'
from my_research.foundation.exporters.qnn import _qnn_quant_overrides
from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS
cfg = SUPPORTED_LLM_MODELS['internvl3_1b']
with _qnn_quant_overrides(cfg, decoder_quant='16a16w', vision_quant='16a16w', embedding_quant='16a16w'):
    print(cfg.quant_recipe().default_quant_dtype.name)
    print(cfg.vision_encoder.quant_recipe().default_quant_dtype.name)
PY
```

## 23. QNN Matrix Export Quant Suffix

Updated `my_research/foundation/scripts/export_internvl3_matrix.sh` so QNN matrix
exports encode the effective quant mode in artifact directory names.

What changed:

- Added QNN matrix environment overrides:
  - `QNN_QUANT`
  - `QNN_VISION_QUANT`
  - `QNN_DECODER_QUANT`
  - `QNN_EMBEDDING_QUANT`
- QNN artifact roots now end with a quant suffix, for example:
  - `internvl3_1b_hybrid_16p_2k_16a16w`
  - `internvl3_1b_hybrid_16p_2k_16a8w`
- The shell script normalizes `fp16` to `16a16w` for artifact names, matching the
  QNN exporter alias behavior.
- If QNN component quant modes differ, the suffix becomes
  `v<vision>_d<decoder>_e<embedding>`.

Why:

- QNN exports with different quant recipes should not share visually ambiguous
  artifact directory names.
- Existing result analysis uses artifact directory names as run identifiers.

Verification:

```bash
bash -n my_research/foundation/scripts/export_internvl3_matrix.sh
```

## 24. Vulkan Dynamic Shape Export Overlay

Implemented a foundation-local Vulkan export overlay for InternVL3 dynamic-shape
decoder export, then later removed the operator-avoidance parts after device-side
Vulkan runs triggered a GPU-driver fatal reboot.

What changed:

- `my_research/foundation/exporters/xnnpack.py`
  - Keeps Vulkan `dtype_override` aligned with upstream Llama Vulkan export:
    `fp32` export dtype with optional Vulkan `force_fp16`.
  - Keeps SDPA KV-cache disabled for this InternVL3 Vulkan path because the
    custom SDPA/KV-cache fusion path fails earlier in Vulkan lowering.
  - Adds a local Vulkan preprocess patch for options already checked by
    ExecuTorch's Vulkan backend but not parsed by upstream `parse_compile_spec`,
    notably `skip_memory_planning`.
  - Removed the temporary operator-avoidance overlays:
    Vulkan preprocess pass no-ops, Vulkan-only RoPE `get_freqs()` monkey patch,
    Vulkan-only dynamic `Tensor.narrow()` monkey patch, and the `_skip_dim_order`
    compile-config workaround.

Why:

- Dynamic shape is required for Vulkan exports in this project, but the
  operator-avoidance overlays produced a graph that could export and then caused
  a probable GPU-driver fatal reboot on device.
- Removing the overlays puts the foundation path back closer to upstream
  behavior for comparison and avoids carrying custom graph rewrites while
  testing standard Llama Vulkan export.

Verification:

```bash
python -m py_compile my_research/foundation/exporters/xnnpack.py
```

Earlier successful dynamic Vulkan export before removing the operator-avoidance
overlays:

```bash
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_512_fp32 \
  --decoder_model internvl3_1b \
  --model_path my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --checkpoint my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth \
  --max_seq_len 512 \
  --max_context_len 512 \
  --dtype fp32 \
  --vision_quant fp16 \
  --decoder_quant fp16 \
  --embedding_quant fp16 \
  --no-use_sdpa_with_kv_cache
```

Generated files:

- `vision_encoder_vulkan.pte` (~1.2GB)
- `text_embedding_vulkan.pte` (~519MB)
- `text_decoder_vulkan.pte` (~1.9GB)
- `manifest.json`

Notes:

- The earlier produced manifest records `enable_dynamic_shape: true`,
  `vulkan_export_dtype: fp32`, `vulkan_force_fp16: false`, and
  `use_sdpa_with_kv_cache: false`.
- Token embedding currently does not partition into Vulkan because
  `aten.embedding` reports unsupported args, but the split artifact is still
  produced. The decoder and vision encoder contain Vulkan delegate partitions.
- ExecuTorch source was not modified.

Upstream Llama Vulkan smoke tests after removing the foundation operator
avoidance overlays:

```bash
python -m examples.models.llama.export_llama \
  --model stories110m \
  --checkpoint executorch/examples/models/llama/params/demo_rand_params.pth \
  --params executorch/examples/models/llama/params/demo_config.json \
  -d fp32 \
  --vulkan \
  -qmode 8da4w \
  -G 64 \
  --max_seq_length 128 \
  --max_context_length 128 \
  -kv \
  --use_sdpa_with_kv_cache \
  --metadata '{"get_bos_id":1,"get_eos_ids":[2]}' \
  --output-dir tmp_vulkan_llama \
  --output_name stories110m_vulkan_demo_8da4w.pte
```

Result:

- Failed during upstream Vulkan preprocessing in `FuseBatchNormPass`.
- Root error was still dynamic RoPE slicing:
  `GuardOnDataDependentSymNode: Could not guard on data-dependent expression
  u176 + 3 > 128`, while executing `aten.slice_copy.Tensor` from
  `examples/models/llama/rope.py` `freqs_cos.narrow(...)`.
- This confirms the same dynamic RoPE `item() -> narrow()` issue appears in the
  clean upstream Llama Vulkan path in the current environment; it was not caused
  by the foundation-only avoidance code.

## 25. Vulkan Decoder-First Export Order

Changed the foundation split exporter so text decoder lowering runs before vision
encoder and text embedding lowering.

What changed:

- `my_research/foundation/exporters/xnnpack.py`
  - The decoder `ExportedProgram` is now lowered and saved first.
  - Vision encoder loading/export/lowering is deferred until after
    `text_decoder_<backend>.pte` is successfully produced.
  - Text embedding lowering is also deferred until after decoder success.
  - Vulkan now allows `decoder_quant=8da4w` with `text_group_size=64`,
    `dtype=fp16` mapped to upstream-style `fp32 + force_fp16`, and
    `use_sdpa_with_kv_cache=True` without forcibly disabling it.
- `my_research/foundation/cli.py`
  - `--text_group_size` default is now `None`; Vulkan `8da4w` defaults to 64,
    other paths default to 128 in the exporter.

Why:

- Vulkan enablement is currently blocked in the text decoder, not the vision
  encoder. Running vision first costs several minutes before reaching the real
  failure.
- Decoder-first export makes Vulkan debugging much faster and avoids writing
  partial vision/embedding artifacts when decoder lowering fails.

Verification:

```bash
python -m py_compile my_research/foundation/exporters/xnnpack.py my_research/foundation/cli.py
```

Decoder-first smoke command:

```bash
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_128_decoder_first_smoke \
  --decoder_model internvl3_1b \
  --model_path my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --max_seq_len 128 \
  --max_context_len 128 \
  --dtype fp16 \
  --vision_quant fp16 \
  --decoder_quant 8da4w \
  --embedding_quant fp16 \
  --text_group_size 64 \
  --use_sdpa_with_kv_cache
```

Result:

- No vision or embedding PTE was written before decoder failure.
- Decoder reached Vulkan preprocessing directly and failed in the known split
  decoder SDPA fusion path:
  `AssertionError: match.update_key_cache_node is not None` in
  `executorch/backends/vulkan/patterns/sdpa.py`.
- ExecuTorch source was not modified.

Follow-up token-id decoder experiment:

- Updated Vulkan decoder export to use a token-id input wrapper instead of the
  embeddings-input wrapper:
  - XNNPACK/non-Vulkan keeps `InternVL3EmbeddingTextDecoder`.
  - Vulkan uses `InternVL3TokenTextDecoder`, calling
    `decoder(token_ids, {"input_pos": input_pos})`.
  - Vulkan forces `use_sdpa_with_kv_cache=True`.
  - Manifest metadata records `decoder_input_mode`.
- Smoke command:

```bash
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_128_token_id_sdpa \
  --decoder_model internvl3_1b \
  --model_path my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --max_seq_len 128 \
  --max_context_len 128 \
  --dtype fp16 \
  --vision_quant fp16 \
  --decoder_quant 8da4w \
  --embedding_quant fp16 \
  --text_group_size 64 \
  --use_sdpa_with_kv_cache
```

Result:

- Token-id path was applied; Vulkan logs show `aten.embedding.default` in the
  decoder graph.
- Vulkan still found a single subgraph, then failed in the same SDPA replacement
  assertion:
  `AssertionError: match.update_key_cache_node is not None`.
- Interpretation: the blocker is no longer just embeddings-input split. InternVL3
  uses a Qwen-style text architecture/cache update graph that still does not
  match the upstream Llama Vulkan SDPA pattern expected by
  `replace_custom_sdpa_with_causal_sdpa`.

Follow-up SDPA matcher overlay:

- Added a foundation-local monkey patch inside `_vulkan_preprocess_option_patch`
  to make `executorch.backends.vulkan.patterns.sdpa.CausalSDPAMatch` search
  cache-node ancestors when direct `key_cache_node.users` /
  `value_cache_node.users` do not contain `llama::update_cache`.
- This keeps ExecuTorch source clean while allowing the Vulkan SDPA replacement
  to find InternVL3/Qwen-style cache update nodes.
- Successful smoke command:

```bash
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_128_token_id_sdpa_cachepatch \
  --decoder_model internvl3_1b \
  --model_path my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --max_seq_len 128 \
  --max_context_len 128 \
  --dtype fp16 \
  --vision_quant fp16 \
  --decoder_quant 8da4w \
  --embedding_quant fp16 \
  --text_group_size 64 \
  --use_sdpa_with_kv_cache
```

Generated files:

- `text_decoder_vulkan.pte` (~799MB)
- `text_embedding_vulkan.pte` (~519MB)
- `vision_encoder_vulkan.pte` (~589MB)
- `manifest.json`

Important logs:

- Decoder:
  - `Found 1 Vulkan subgraphs to be partitioned.`
  - Vulkan partition includes `sdpa_with_kv_cache.default`,
    `et_vk.apply_rotary_emb_hf.default`, and
    `et_vk.linear_dq8ca_q4gsw.default`.
- Vision:
  - `Found 73 Vulkan subgraphs to be partitioned.`
- Embedding:
  - `aten.embedding.default` still does not partition to Vulkan due to unsupported
    args, matching upstream Llama behavior.

Follow-up text-only run support:

- User wanted to run the `max_seq_len=128` Vulkan artifact without an image because
  the image placeholder tokens exceed the context budget.
- Added a local runner CLI flag, `--decoder_input_mode`, and propagated it through
  `ManifestData`.
- Updated the Android launcher to allow text-only XNNPACK/Vulkan runs when the
  artifact uses `decoder_input_mode=token_ids`; it now skips frame extraction and
  omits `--image_path`.
- Updated `xnnpack_backend.cpp` so token-id decoder artifacts skip vision/text
  embedding modules and feed `Long` token tensors directly into the decoder for
  both prefill and decode.
- Rebuilt with:

```bash
SKIP_ET_BUILD=1 JOBS=8 my_research/foundation/scripts/build_backend_and_runner.sh xnnpack-vulkan
```

- Smoke run succeeded without `--image`:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_128_token_id_sdpa_cachepatch/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --questions "Describe yourself briefly." \
  --seq_len 32 \
  --temperature 0.0 \
  --save_log
```

- Output/logs were pulled to
  `my_research/foundation/results/log/vulkan/internvl3_vulkan_1b_128_token_id_sdpa_cachepatch/`.

### Vulkan 1K Token-ID Export

User requested a 1024-token Vulkan build after the 128-token text-only smoke run.

Command:

```bash
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_token_id_sdpa_cachepatch \
  --decoder_model internvl3_1b \
  --model_path my_research/foundation/results/model/hf/InternVL3-1B-hf \
  --max_seq_len 1024 \
  --max_context_len 1024 \
  --dtype fp16 \
  --vision_quant fp16 \
  --decoder_quant 8da4w \
  --embedding_quant fp16 \
  --text_group_size 64 \
  --use_sdpa_with_kv_cache
```

Result:

- Export succeeded.
- Output directory:
  `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_token_id_sdpa_cachepatch/`
- Generated files:
  - `text_decoder_vulkan.pte` (~800MB)
  - `text_embedding_vulkan.pte` (~519MB)
  - `vision_encoder_vulkan.pte` (~589MB)
  - `manifest.json`
- Manifest confirms:
  - `max_seq_len=1024`
  - `max_context_len=1024`
  - `enable_dynamic_shape=true`
  - `decoder_input_mode=token_ids`
  - `use_sdpa_with_kv_cache=true`
  - `decoder_quant=8da4w`
  - `text_group_size=64`
- Export logs:
  - Decoder: `Found 1 Vulkan subgraphs to be partitioned.`
  - Decoder Vulkan ops include `sdpa_with_kv_cache.default` and
    `et_vk.linear_dq8ca_q4gsw.default`.
  - Vision: `Found 73 Vulkan subgraphs to be partitioned.`

### Hybrid XNNPACK Vision + Vulkan Decoder Experiment

User asked whether unsupported/problematic Vulkan image-side work could be sent
through XNNPACK while keeping the decoder on GPU.

Setup:

- Vision encoder: XNNPACK 512 fp16 artifact.
- Text embedding: XNNPACK 512 fp16 artifact.
- Text decoder: Vulkan 512 embeddings artifact with `8da4w`,
  `use_sdpa_with_kv_cache=true`, `decoder_input_mode=embeddings`.
- Created manifest:
  `my_research/foundation/results/model/vulkan/internvl3_hybrid_xnnpack_vision_embedding_vulkan_decoder_512/manifest.json`

Runner changes:

- Added Vulkan decoder input casting in `xnnpack_backend.cpp`, because the
  Vulkan embeddings decoder expects Float input while the XNNPACK embedding and
  vision artifacts produce Half embeddings.
- Kept image frame input cast to Half for the existing fp16 vision artifacts.

Findings:

- Full-Vulkan image path still exits before useful progress, likely during
  Vulkan vision/embedding pipeline initialization or early execution.
- Hybrid path reaches all expected phases:
  - `V_Encode` completed with XNNPACK vision.
  - `EmbeddingAndMerging` completed.
  - `T_Prefill` completed on Vulkan decoder.
  - Decode steps ran on Vulkan decoder.
- Running with `seq_len=320` on a 512 context artifact failed after `kv_pos`
  exceeded 512:

```text
tensor sizes requires a larger texture than the current one
```

- This was not an Adreno crash; it was context capacity overflow:
  prompt/image tokens were 278, so 320 generated tokens needs roughly 598 total
  context slots.
- Re-running with `seq_len=200` succeeded.
- Output/logs:
  `my_research/foundation/results/log/vulkan/internvl3_hybrid_xnnpack_vision_embedding_vulkan_decoder_512/`

## Update Checklist for Future Changes

When modifying implementation:

- Update this file with:
  - What changed
  - Why it changed
  - Which files changed
  - Any error messages that motivated the change
  - Whether ExecuTorch source was touched
- Keep `docs/mobile_backend_flow.md` aligned with actual commands.
- Keep `README.md` short and point to the detailed docs.
- Prefer adding adapters under `my_research/foundation` over patching `executorch/`.