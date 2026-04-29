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
  - Historical note: at this point, QNN `fp16` was treated as an alias for
    `16a16w`. Section 28 changes this behavior so `fp16` means HTP fp16 compile
    precision and explicit `16a16w` keeps the quantized 16a16w path.
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
- Historical note: this originally normalized `fp16` to `16a16w` for artifact
  names. Section 28 changes this so `fp16` and `16a16w` produce distinct artifact
  tags.
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

## 26. Vulkan-Friendly InternVL3 Vision Attention Overlay

User asked to make the InternVL3 vision encoder Vulkan-compatible by avoiding the
generic vision SDPA decomposition that produced unsupported Vulkan ops.

What changed:

- `my_research/foundation/models/internvl3/vision_encoder/model.py`
  - Added `VulkanFriendlyInternVLVisionAttention`.
  - Replaces InternVL vision self-attention with explicit `bmm + softmax + bmm`.
  - Adds `replace_vision_attention_for_vulkan()` and
    `load_vision_encoder(..., vulkan_friendly_attention=True)`.
- `my_research/foundation/exporters/xnnpack.py`
  - Vulkan vision export now requests `vulkan_friendly_attention=True`.
  - XNNPACK/QNN behavior is unchanged.

Verification:

- Vision graph no longer contains repeated `aten::scaled_dot_product_attention`.
- Repeated attention-mask unsupported ops disappeared:
  `logical_not`, `any.dim`, `eq.Scalar`, `full_like`, and attention-path
  `expand_copy`.
- Vision-only Vulkan export succeeded:
  `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_vision_attention_overlay_smoke/vision_encoder_vulkan.pte`
- Size: ~589MB (`617255810` bytes).
- Vulkan partitioner reported `Found 1 Vulkan subgraphs to be partitioned.`
- One remaining skip was `aten.expand_copy.default` on shape `[1, 1, 1024]`,
  likely CLS token expansion before `cat`, not the repeated attention mask path.
- ExecuTorch source and Transformers site-packages were not modified.

### Vulkan Vision Runtime Input Dtype Fix

Full image Vulkan run with the attention-overlay vision artifact reached the
runner but failed during vision execution:

```text
Input 0 has unexpected scalar type: expected Float but was Half.
```

Cause:

- `xnnpack_backend.cpp` always converted preprocessed image frames from Float to
  Half before calling the vision module.
- This is correct for existing XNNPACK fp16 vision artifacts, but Vulkan
  force-fp16 artifacts keep a Float input schema and cast inside the Vulkan
  delegate.

Fix:

- `my_research/foundation/runner/xnnpack_backend.cpp`
  - Keep frame input as Float when `manifest_.backend == "vulkan"`.
  - Continue casting to Half for XNNPACK fp16 vision artifacts.

Next step:

```bash
SKIP_ET_BUILD=1 JOBS=8 my_research/foundation/scripts/build_backend_and_runner.sh xnnpack-vulkan
```

Then rerun the same full image Vulkan command.

### Embedding-Input Text-Only Vulkan Runs

Observation:

- Embedding-input artifacts do not intrinsically require an image. If no image is
  supplied, the prompt contains no `<IMG_CONTEXT>` tokens, so the runner can use
  `text_embedding_pte -> text_decoder_pte` directly.
- The previous launcher/runner incorrectly required `--image_path` whenever
  `decoder_input_mode != token_ids`.

Changed:

- `my_research/foundation/host/launcher.py`
  - Allows XNNPACK/Vulkan text-only runs for both `token_ids` and `embeddings`
    decoder input modes.
- `my_research/foundation/runner/xnnpack_qnn_runner.cpp`
  - Removed the `--image_path` requirement for embedding-input decoder artifacts.
- `my_research/foundation/runner/xnnpack_backend.cpp`
  - Removed `frame_dir` / `frame_count` assertions for embedding-input artifacts.
  - Loads the vision module only when `frame_count > 0`.
  - Still loads the text embedding module for embedding-input text-only runs.

Next step:

```bash
SKIP_ET_BUILD=1 JOBS=8 my_research/foundation/scripts/build_backend_and_runner.sh xnnpack-vulkan
```

Then an embedding-input Vulkan artifact can run text-only by omitting `--image`.

### Vulkan 1K FP16 Text-Only Smoke

User compiled a 1K Vulkan FP16 artifact and ran a text-only prompt.

Observed output:

```text
<|im_start|>user:
Where is capital of Korea?<|im_end|>
<|im_start|>assistant
The capital of South Korea is Seoul.<|im_end|>
```

Result:

- Text-only Vulkan FP16 path produced a correct answer.
- Output log:
  `my_research/foundation/results/log/vulkan/internvl3_vulkan_1b_1k_fp16/foundation_output.txt`

### Vulkan Vision NaN Root Cause and GELU Fix

Problem:

- Full Vulkan image runs initially produced `!` immediately after prefill.
- Dumping the vision encoder output showed the Vulkan vision tensor
  `[1, 256, 896]` was entirely NaN:
  `nan_count=229376`.
- The same image with XNNPACK vision and Vulkan text embedding/decoder produced
  a normal answer, so the failure was isolated to the Vulkan vision encoder.
- Vulkan fp32 vision export also produced NaNs, so the issue was not simply
  fp16 overflow from `force_fp16`.

Debug changes:

- `my_research/foundation/runner/xnnpack_qnn_runner.cpp`
  - Added `--vision_only` so the Android runner can execute only the vision
    encoder.
- `my_research/foundation/runner/backend.h`
  - Added `UnifiedRunConfig::vision_only`.
- `my_research/foundation/runner/xnnpack_backend.cpp`
  - In `vision_only` mode, loads and runs only the vision module.
  - Dumps `vision_output_stats.csv` and `vision_output_0000_f32.bin` when
    `--save_log` is supplied.
- `my_research/foundation/host/launcher.py`
  - Added `vision_only` plumbing and pulls the vision dump artifacts.
  - Skips pushing text embedding/decoder/tokenizer files in vision-only runs.
- Debug scripts under `my_research/foundation/debug/vision_compare/`:
  - `compare_dumped_vision_outputs.py`
  - `export_vulkan_vision_fp32.py`
  - `export_vulkan_vision_prefix_fix.py`
  - `export_vulkan_vision_stage_fix.py`

Isolation results:

- `prefix0`, `prefix1`, `prefix4`, `prefix8`, `prefix10`, `prefix11` were
  numerically valid.
- `prefix12` produced NaNs.
- Layer 11 attention-only was valid.
- Layer 11 full block produced NaNs.
- Therefore the first failure was isolated to the layer-11 MLP path, not the
  rewritten attention path.

Fix:

- `my_research/foundation/models/internvl3/vision_encoder/model.py`
  - Kept the Vulkan-friendly attention overlay as explicit
    `bmm -> softmax -> bmm`.
  - Added `contiguous()` around the attention permute/reshape/bmm boundaries to
    avoid Vulkan layout ambiguity.
  - Added `VulkanFriendlyGELU`, a tanh-form GELU written with primitive ops.
  - `replace_vision_attention_for_vulkan()` now also replaces each vision MLP
    `activation_fn` with `VulkanFriendlyGELU`.

Verification:

- Exported through the regular CLI path, without `_fix`-specific code:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_fp16_fix \
  --decoder_model internvl3_1b \
  --model_path my_research/foundation/results/model/hf/InternVL3-1B-hf \
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

- Full image run succeeded:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_fp16_fix/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-xnnpack-vulkan/foundation/xnnpack_qnn_runner \
  --device R3CX50FQ62L \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log \
  --force_push
```

- Output:

```text
Two cats are sleeping on a pink blanket with a remote control nearby.<|im_end|>
```

- Vision stats after the fix:
  - shape `[1, 256, 896]`
  - mean `-0.0566381`
  - min/max `-7.58984 / 6.57031`
  - no NaNs
- XNNPACK-vs-Vulkan fixed vision dump comparison:
  - cosine `0.9986975`
  - mean abs diff `0.0320589`
- Result logs:
  - `my_research/foundation/results/log/vulkan/internvl3_vulkan_1b_1k_fp16_fix/`

Next:

- The GELU fix is now part of the default Vulkan vision overlay. Future Vulkan
  exports do not need `_fix` artifact names.

## 27. Unified XNNPACK + Vulkan + QNN Android Build

User wanted a single build tree and runner that links XNNPACK, Vulkan, and QNN
together for future hybrid-system experiments.

Final status:

- `my_research/foundation/scripts/build_backend_and_runner.sh unified` is now the
  default practical build path.
- The unified runner at
  `executorch/build-android-unified/foundation/xnnpack_qnn_runner` has been
  verified with XNNPACK, Vulkan, and QNN 2K InternVL3 artifacts on device
  `R3KYC01FW1P`.
- The older split build trees remain useful only for backend-isolation debugging:
  - `executorch/build-android-xnnpack-vulkan`
  - `executorch/build-android`

What changed:

- `my_research/foundation/scripts/build_backend_and_runner.sh`
  - Added `unified` mode.
  - Configures `executorch/build-android-unified` with:
    - `EXECUTORCH_BUILD_QNN=ON`
    - `EXECUTORCH_BUILD_XNNPACK=ON`
    - `EXECUTORCH_BUILD_VULKAN=ON`
    - LLM runner, module, tensor, flat tensor, named data map, quantized,
      optimized, and LLM kernels enabled.
  - Builds the existing foundation runner against that unified tree.
- `my_research/foundation/CMakeLists.txt`
  - Previous QNN path avoided linking upstream `portable_ops_lib` to prevent
    duplicate portable kernel registration in QNN-only builds.
  - Unified builds need XNNPACK/Vulkan fallback kernels too, so the runner now
    links upstream `portable_ops_lib` whenever that target exists, even when
    `qnn_executorch_backend` also exists.
  - Links `custom_ops` with whole-archive in the QNN/unified path so Llama custom
    ops such as `llama::custom_sdpa.out` and `llama::update_cache.out` are
    registered.
  - Keeps a small selected portable op library fallback only for QNN build trees
    that do not expose `portable_ops_lib`. That fallback currently includes:
    - `aten::expand_copy.out`
    - `aten::embedding.out`
  - Keeps `custom_ops` independent from that selected portable fallback to avoid
    registering the same kernel twice.
- `my_research/foundation/docs/mobile_backend_flow.md`
  - Changed the default build/run flow to unified.
  - Kept split build paths documented only as legacy/debug fallback paths.

Why:

- The paper direction needs one runtime binary that can host heterogeneous
  execution choices instead of selecting a backend-specific runner binary.
- The foundation CMake was already written to link any available
  `xnnpack_backend`, `vulkan_backend`, and `qnn_executorch_backend` targets in
  the selected ExecuTorch build tree.
- XNNPACK and Vulkan artifacts still contain portable/custom fallback ops for
  pieces that are not delegated. A unified binary must therefore preserve both:
  - portable kernels such as `aten::gelu.out`, `aten::native_layer_norm.out`,
    `aten::where.self_out`, `aten::embedding.out`, and
    `dim_order_ops::_to_dim_order_copy.out`
  - Llama custom ops such as `llama::custom_sdpa.out` and
    `llama::update_cache.out`

Verification:

```bash
JOBS=8 my_research/foundation/scripts/build_backend_and_runner.sh unified
```

Result:

- Build succeeded.
- Runner output:
  `executorch/build-android-unified/foundation/xnnpack_qnn_runner`
- Vulkan image run with unified runner succeeded:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_2k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

- Output:
  `Two cats are sleeping on a pink blanket with a remote control nearby.<|im_end|>`
- XNNPACK image run with unified runner succeeded:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/xnnpack/internvl3_xnnpack_1b_2k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

- XNNPACK output:
  `Two cats are sleeping on a pink blanket with a remote control nearby.<|im_end|>`

- QNN image run with unified runner and build tree succeeded:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/qnn/internvl3_1b_qnn_2k_16a8w/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  -b executorch/build-android-unified \
  -s R3KYC01FW1P \
  -m SM8750 \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

- QNN output:
  `Two cats sleeping on a pink blanket with a remote control nearby.<|im_end|>`
- Vulkan was re-run after the later XNNPACK/QNN link fixes and still succeeded:

```bash
python -m my_research.foundation.cli run \
  --manifest /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_2k_fp16/manifest.json \
  --runner_binary /workspace/streamingvlm/executorch/build-android-unified/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --seq_len 320 \
  --temperature 0.0 \
  --save_log
```

- Latest Vulkan output:
  `Two cats are sleeping on a pink blanket with a remote control nearby.<|im_end|>`
- ExecuTorch source was not modified.
- CMake warnings about unavailable unrelated backends such as CoreML, MPS, MLX,
  and OpenVINO are expected because those targets are not enabled in this
  Android build.

Errors fixed during unified Vulkan run:

- Missing `aten::expand_copy.out` when the unified runner did not preserve the
  portable fallback op registration.
- Duplicate `_adaptive_avg_pool2d` registration when trying to whole-archive all
  portable ops.
- Duplicate `expand_copy` registration when the selected op lib was linked
  through both `custom_ops` and the runner.
- Missing `aten::embedding.out`; fixed by adding it to the selected portable op
  list.
- XNNPACK unified run then missed many portable fallback ops; fixed by linking
  upstream `portable_ops_lib` when available.
- XNNPACK unified run then missed `llama::custom_sdpa.out` and
  `llama::update_cache.out`; fixed by whole-archiving `custom_ops` in the
  QNN/unified runner path.

Implementation notes:

- Do not whole-archive all portable ops as a local generated library in the
  unified runner. That caused duplicate registration such as:
  `Re-registering aten::_adaptive_avg_pool2d.out`.
- Do not link the same generated selected portable op library both through
  `custom_ops` and directly through the runner. That caused duplicate
  `aten::expand_copy.out` registration.
- Prefer upstream `portable_ops_lib` in unified trees because it already has the
  broader set of fallback kernels needed by XNNPACK/Vulkan artifacts.
- Use the generated selected portable fallback only for QNN-only build trees
  where `portable_ops_lib` is absent.
- From this point, examples in `mobile_backend_flow.md` should use
  `executorch/build-android-unified` and
  `executorch/build-android-unified/foundation/xnnpack_qnn_runner` by default.

## 28. QNN `fp16` Now Uses HTP FP16 Compile Precision

User asked that specifying `fp16` for QNN should use ExecuTorch/Qualcomm's real
HTP fp16 compile path instead of being mapped to quantized `16a16w`.

What changed:

- `my_research/foundation/exporters/qnn.py`
  - Removed the old `fp16 -> 16a16w` alias behavior.
  - `fp16` is now preserved as its own mode.
  - If all QNN components are `fp16`, export uses a project-local wrapper around
    upstream `examples/qualcomm/oss_scripts/llama/llama.py` compile flow.
  - The wrapper creates HTP compile specs with
    `generate_htp_compiler_spec(use_fp16=True)`, which sets QNN HTP precision to
    `QnnExecuTorchHtpPrecision.kHtpFp16`.
  - PTQ is disabled for this path by not calling
    `MultiModalManager.quantize()`. The upstream quant recipe remains available
    during model construction because `TextDecoder` uses it to derive static KV
    IO metadata.
  - The encoder, token embedding, and decoder graphs are lowered directly with
    fp16 HTP compile specs.
  - Mixed `fp16` plus quantized component modes are rejected for now. Use either
    all `fp16` for HTP fp16 compile or explicit quant modes such as `16a8w` and
    `16a16w` for the old QDQ path.
- `my_research/foundation/cli.py`
  - Updated QNN quant help text: `fp16` now means HTP fp16 compile precision;
    explicit `16a16w` means quantized 16a16w.
- `my_research/foundation/scripts/export_internvl3_matrix.sh`
  - QNN artifact tags now keep `fp16` as `fp16` instead of normalizing it to
    `16a16w`.
- `my_research/foundation/docs/mobile_backend_flow.md`
  - Updated the QNN quantization mode explanation and added a QNN true HTP fp16
    matrix export example.

Why:

- ExecuTorch Qualcomm has two different concepts:
  - `QuantDtype.use_16a16w`: PTQ/QDQ quantized graph.
  - `generate_htp_compiler_spec(use_fp16=True)`: QNN HTP fp16 runtime compile
    precision.
- The old alias made `--decoder_quant fp16` produce a quantized `16a16w` graph,
  which is not the same as QNN HTP fp16 compile.
- The `internvl3_1b_qnn_512_16a16w` artifact ran but produced `1-1`, while
  `internvl3_1b_qnn_512_16a8w` produced a normal caption. This made the
  distinction practically important.

Important caveat:

- This is an upgrade-safe overlay. It does not modify ExecuTorch source.
- The true fp16 path is newly implemented and still needs a full export/run
  verification. If export fails, inspect assumptions in the local compile
  wrapper against the current upstream `llama.py` flow.
- First attempted implementation set `decoder_model_config.quant_recipe=None`,
  but that failed during `TextDecoder` initialization with:
  `'NoneType' object has no attribute 'get_kv_io_bit_width'`. Keep the recipe
  object available and skip the quantize phase instead.
- Second failure during true fp16 export:
  `'LlamaModelWithoutEmbedding' object has no attribute 'graph'` inside
  `HybridTextDecoder._encoding_override()`.
  - Cause: skipping PTQ leaves decoder/prefill as eager modules; upstream
    `_encoding_override()` assumes quantize converted them to graph modules.
  - Fix: in the local true-fp16 wrapper only, temporarily patch
    `HybridTextDecoder._encoding_override` to a no-op. The override aligns QDQ
    encodings, which do not exist in the true fp16 path.

## 29. Foundation GitHub README Expansion

User asked to turn `my_research/foundation/README.md` into a GitHub-facing
project overview instead of a short placeholder.

What changed:

- Rewrote `my_research/foundation/README.md` with:
  - project overview and design goal
  - repository layout
  - environment setup
  - unified Android runner build command
  - XNNPACK, Vulkan, and QNN export examples
  - Android run examples
  - QNN quantization mode notes
  - links to detailed docs
  - development principles about keeping ExecuTorch clean

Why:

- The README should explain the whole foundation layer to someone landing on the
  repository from GitHub.
- It should point users to the unified runner path as the default workflow while
  leaving detailed troubleshooting in `docs/mobile_backend_flow.md`.

## 30. Root Git Ignore Policy

User asked whether the GitHub repo should include only code and exclude
`executorch` and generated `results`.

Decision:

- Keep upstream ExecuTorch out of git with root `.gitignore` entry `executorch/`.
- Keep generated model artifacts, logs, plots, and run outputs out of git with
  `my_research/foundation/results/`.
- Also ignore common generated files: Python caches, CMake outputs, local envs,
  logs, CSV profiler data, and large model/runtime binaries such as `.pte`,
  `.pth`, `.safetensors`, `.onnx`, `.so`, and `.a`.

Why:

- The repo should track project-local code, scripts, CMake, and docs.
- ExecuTorch should remain an external upstream checkout so it can be updated
  cleanly.
- Foundation results are reproducible outputs and can be very large.

## 31. Forced Decode Token Count

User asked for a runner argument that forces decoding to produce exactly a fixed
number of tokens, ignoring EOS if it appears early and stopping at that count if
EOS never appears.

What changed:

- Added CLI/runner option `--force_generate_token N`.
- `my_research/foundation/cli.py`
  - Added `run --force_generate_token`.
  - Passes the value through to `run_with_manifest()`.
- `my_research/foundation/host/launcher.py`
  - Passes `--force_generate_token=N` to the Android runner for QNN, XNNPACK,
    and Vulkan manifests.
- `my_research/foundation/runner/backend.h`
  - Added `UnifiedRunConfig::force_generate_token`.
- `my_research/foundation/runner/xnnpack_qnn_runner.cpp`
  - Added gflags option `--force_generate_token`.
- `my_research/foundation/runner/qnn_backend.cpp`
  - When forced, sets `GenerationConfig.ignore_eos=true` and uses the forced
    count as `seq_len`.
- `my_research/foundation/runner/xnnpack_backend.cpp`
  - When forced, ignores EOS and text stop markers inside the manual decode loop
    and stops after exactly the forced count.
- `my_research/foundation/README.md` and
  `my_research/foundation/docs/mobile_backend_flow.md`
  - Documented the option.

Behavior:

- Without `--force_generate_token`, existing behavior remains unchanged:
  `--seq_len` is the maximum decode length and EOS/stop markers can stop early.
- With `--force_generate_token 128`, the runner decodes exactly 128 tokens,
  regardless of EOS.

Follow-up:

- User wanted run examples to omit `--seq_len`, letting the launcher read the
  default generation cap from `manifest.export.max_seq_len`, and to include
  `--force_generate_token 128`.
- Updated `my_research/foundation/host/launcher.py` so XNNPACK/Vulkan now match
  QNN behavior and default to `manifest.export.max_seq_len` when `--seq_len` is
  not provided.
- Updated `my_research/foundation/docs/mobile_backend_flow.md` and
  `my_research/foundation/README.md` run examples to remove explicit
  `--seq_len` and add `--force_generate_token 128`.

## 32. Backend Memory-vs-Sequence Debug Plot

User asked for a debug Python script under `my_research/foundation/debug` that
plots XNNPACK and Vulkan memory usage against sequence length up to 8K using
logs under `my_research/foundation/results/log/{xnnpack,vulkan}`.

What changed:

- Added `my_research/foundation/debug/plot_backend_memory_by_seq.py`.
- The script reads each artifact's `android_memory_timeline.csv`.
- It includes 512, 1K, 2K, 4K, and 8K artifacts, excluding 16K by default.
- Default memory column: `mem_available_kb`, per user correction.
- Default metric: `max_minus_min`, because lower available memory means higher
  system memory consumption.
- It also supports `--metric first_minus_min` if the exact first-minus-minimum
  baseline comparison is needed.
- Generated:
  - `my_research/foundation/debug/backend_memory_by_seq.png`
  - `my_research/foundation/debug/backend_memory_by_seq.csv`

Observed with current logs:

- XNNPACK `mem_available_kb` max-minus-min grows from about 1856 MiB at 512 to
  about 2484 MiB at 8K.
- Vulkan `mem_available_kb` max-minus-min grows from about 2059 MiB at 512 to
  about 5779 MiB at 8K.

Follow-up Vulkan 8K component analysis:

- Added `my_research/foundation/debug/plot_memory_components_timeline.py`.
- Generated `my_research/foundation/debug/vulkan_8k_memory_components.png`.
- The plot overlays:
  - available-memory drop
  - `gpu_total_kb` growth
  - `kgsl_shmem_usage_kb` growth
  - process `self_rss_kb` growth
  - cached-memory growth
- The Vulkan 8K available-memory drop closely tracks `gpu_total_kb` and
  `kgsl_shmem_usage_kb`, not process RSS.
- Important interpretation:
  - The large drop appears visually near the Embedding/Prefill transition, but
    numerically most of it happens between `V_Encode` end (~2.685s) and
    `EmbeddingAndMerging` start (~6.102s).
  - In that interval, `mem_available_kb` drops by about 4978 MiB while
    `gpu_total_kb` grows by about 5391 MiB and `kgsl_shmem_usage_kb` grows by
    about 5392 MiB.
  - `self_rss_kb` grows by only about 494 MiB in the same interval.
- Conclusion: the apparent embedding-time memory cliff is mainly Vulkan
  GPU/KGSL allocation for delegated graph buffers / storage / staging around
  text embedding and decoder prefill preparation, not CPU process RSS.

Documentation:

- Added `my_research/foundation/docs/vulkan_memory_analysis.md`.
- The doc explains:
  - why `mem_available_kb` is the preferred system memory-pressure metric
  - why process `self_rss_kb` under-reports Vulkan GPU allocations
  - why Vulkan `fp16` and XNNPACK `fp16` have different runtime memory layouts
  - why the 8K Vulkan cliff is attributed to GPU/KGSL allocations around the
    embedding/prefill transition
  - how to regenerate the debug plots

## 33. Experimental Vulkan fp16 Export Dtype

User wanted `vulkan_export_dtype` itself to be `fp16` instead of the prior
compatibility path where `--dtype fp16` produced an fp32 export graph plus
`vulkan_force_fp16=true`.

What changed:

- Updated `my_research/foundation/exporters/xnnpack.py` only.
- The foundation Vulkan partitioner wrapper now accepts `dtype="fp16"` as well
  as `fp32`/`None`.
- `export_dtype` now follows the requested CLI dtype for Vulkan instead of
  forcing every Vulkan export graph to fp32.
- `vulkan_force_fp16` is still set when `--dtype fp16`, so the Vulkan delegate
  continues to receive the fp16 compile option.
- Added a scoped foundation-only context patch for
  `executorch.extension.llm.custom_ops.custom_ops._validate_params` during
  Vulkan fp16 lowering. This lets `llama.sdpa_with_kv_cache` meta validation
  accept fp16/fp32 floating tensors.

Why:

- The earlier path was chosen because upstream ExecuTorch's general Vulkan
  helper documents fp32-only dtype override and recommends `force_fp16`.
- For memory experiments, we need a stricter artifact where the exported graph
  metadata and manifest record `vulkan_export_dtype: fp16`.

Caveat:

- This is an experimental project overlay. If Vulkan lowering or runtime fails,
  the fallback is to restore the previous `fp32 + force_fp16` export path.
- No original ExecuTorch source was modified.

Verification:

- First attempt with `vulkan_export_dtype=fp16` failed in Vulkan lowering:
  `AssertionError: Expected key to be float32 but got torch.float16`.
  - Cause: upstream Python meta validation for `llama.sdpa_with_kv_cache`
    hard-coded fp32 even though the Vulkan op registry declares floating-point
    input support.
  - Fix: patch `_validate_params` only inside the foundation Vulkan fp16 lowering
    context.
- Re-export then succeeded for:
  `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp16_exportdtype_fp16_v2`.
- The generated manifest records:
  - `dtype: fp16`
  - `vulkan_export_dtype: fp16`
  - `vulkan_force_fp16: true`
- Runtime verification on Android failed before generation:
  `Vulkan QueryPool: Exceeded the maximum number of queries allowed by the queryPool (4096)!`
  from `executorch/backends/vulkan/runtime/vk_api/QueryPool.cpp`.
  - The failure happens both with default manifest `--seq_len=8192` and with
    explicit `--seq_len 320`.
  - It also happens without `--save_log`, so this is not the foundation log
    collection path.
  - Interpretation: the fp16 export graph runs through Vulkan lowering, but at
    runtime it emits more timestamp/profile query writes than the current
    ExecuTorch Vulkan query pool size allows.
  - Next practical fix, if this artifact must run, is to rebuild the Vulkan
    runtime with a larger `VULKAN_QUERY_POOL_SIZE` compile definition or find a
    runtime/build option that disables shader timestamp profiling.
- Increased the unified Android build's Vulkan query pool size:
  - `my_research/foundation/scripts/build_backend_and_runner.sh` now accepts
    `VULKAN_QUERY_POOL_SIZE` and defaults it to `65536`.
  - It passes `-DCMAKE_CXX_FLAGS="-DVULKAN_QUERY_POOL_SIZE=${VULKAN_QUERY_POOL_SIZE}"`
    to XNNPACK+Vulkan and unified ExecuTorch CMake configure.
  - Rebuilt with:
    `JOBS=8 VULKAN_QUERY_POOL_SIZE=65536 my_research/foundation/scripts/build_backend_and_runner.sh unified`.
- Runtime verification after the rebuild:
  - Command used the unified runner, fp16-export-dtype 8K Vulkan artifact,
    `--seq_len 320`, `--force_generate_token 128`, and `--force_push`.
  - The QueryPool abort was resolved and the run exited with code 0.
  - Output file:
    `my_research/foundation/results/log/vulkan/internvl3_vulkan_1b_8k_fp16_exportdtype_fp16_v2/foundation_output.txt`.
  - The generated text was corrupted/repetitive (`Deb`, `EDITOR`, `false`,
    etc.), so this artifact now runs but is not behaviorally correct yet.
  - This suggests the experimental `vulkan_export_dtype=fp16` graph path may
    still have numerical/lowering/runtime correctness issues compared with the
    previous `fp32 export graph + force_fp16` path.

## 34. KV-Cache Quantization Controls

User asked to expose upstream `--quantize_kv_cache` and to check whether QNN can
force KV-cache/KV I/O to 8-bit independently from the main component quant mode.

What changed:

- `my_research/foundation/cli.py`
  - Added export flag `--quantize_kv_cache`.
  - Added export flag `--qnn_kv_quant {default,8}`.
- `my_research/foundation/exporters/xnnpack.py`
  - Wires `--quantize_kv_cache` to `llm_config.model.quantize_kv_cache`.
  - Records `quantize_kv_cache` in manifest export metadata.
  - Records `quant.kv_cache = "int8"` when enabled.
- `my_research/foundation/exporters/qnn.py`
  - Adds `qnn_kv_quant` to QNN export metadata.
  - When `--qnn_kv_quant 8` is used on a quantized QNN path, the project-local
    decoder quant recipe appends Qualcomm's `annotate_kv_8bit`.
  - This forces `StaticLLMQuantRecipe.get_kv_io_bit_width()` to return 8 for the
    decoder recipe even when the main decoder quant mode is not `8a8w`.
  - `--qnn_kv_quant 8` is rejected for the true QNN fp16 compile path because
    that path skips PTQ/QDQ and therefore does not have a KV 8-bit annotation
    path.

Support matrix:

- XNNPACK/Vulkan:
  - `--quantize_kv_cache` uses upstream `QuantizedKVCache`.
  - Upstream currently supports int8 per-token KV-cache only; no bit-width
    argument exists.
  - This is wired but not yet export/run verified for InternVL3 foundation
    artifacts.
- QNN:
  - Default behavior follows the selected QNN quant recipe.
  - `--qnn_kv_quant 8` forces decoder KV I/O to 8-bit on quantized paths such as
    `16a8w`, `16a16w`, `16a4w`, `16a4w_block`, `8a8w`, and `8a4w`.
  - True QNN `fp16` compile cannot currently use this flag.

Follow-up documentation/workflow update:

- User clarified that all memory experiments should quantize KV-cache so AoT
  KV allocation is controlled by the intended KV budget.
- Updated `my_research/foundation/docs/mobile_backend_flow.md`:
  - Added an export-time KV quantization principle near the top of the export
    section.
  - XNNPACK/Vulkan examples now include `--quantize_kv_cache`.
  - QNN example includes `--qnn_kv_quant 8`.
  - True QNN fp16 is explicitly documented as a compiler-precision experiment,
    not a memory-controlled KV baseline.
- Updated `my_research/foundation/scripts/export_internvl3_matrix.sh`:
  - Added `QNN_EXTRA_ARGS`, `XNNPACK_EXTRA_ARGS`, and `VULKAN_EXTRA_ARGS`.
  - These are appended to each backend export command, so batch exports can pass
    `--quantize_kv_cache` or `--qnn_kv_quant 8` without adding new hard-coded
    script flags.

## 35. Vulkan Linear Quantization Modes

User pointed to the upstream ExecuTorch Vulkan backend docs, which state that
Vulkan supports quantized linear layers with:

- 8-bit or 4-bit weights and FP32/FP16 activations.
- 8-bit or 4-bit weights with 8-bit dynamically quantized activations.

What changed:

- `my_research/foundation/exporters/xnnpack.py`
  - Added foundation-local Vulkan decoder quant mode normalization.
  - Opens the following Vulkan decoder modes:
    - `vulkan_8w` / `8w`
    - `vulkan_4w` / `4w`
    - `vulkan_8da8w` / `8da8w`
    - `vulkan_8da4w` / `8da4w`
  - Uses `executorch.backends.vulkan.quantizer.VulkanQuantizer` directly with
    `get_symmetric_quantization_config(is_dynamic=..., weight_bits=...)`.
  - Keeps Vulkan vision and text embedding at `fp16`.
  - Keeps these modes separate from KV-cache quantization; use
    `--quantize_kv_cache` for KV-cache.
- `my_research/foundation/cli.py`
  - Updated `--decoder_quant` help text to list Vulkan modes and aliases.
- `my_research/foundation/docs/mobile_backend_flow.md`
  - Updated the Vulkan support list to include the new decoder linear
    quantization modes.

Notes:

- Upstream LLM export only exposes `vulkan_8w` through `--pt2e_quantize`, but
  the Vulkan quantizer implementation already accepts `weight_bits=4/8` and
  `is_dynamic=True/False`. The foundation exporter now maps user-friendly mode
  names to that quantizer directly.
- These modes are wired but still need artifact export/run verification for
  InternVL3.

## 36. Vulkan true-fp16 Correctness Debug Harness

User clarified that XNNPACK fp16 works, so the issue should be treated as a
Vulkan fp16 backend correctness problem rather than generic LLM fp16
sensitivity. User also clarified that `sdpa_with_kv_cache` should remain enabled
and should accept fp16 inputs.

What changed:

- Added debug folder:
  - `my_research/foundation/debug/vulkan_fp16_correctness/`
  - `README.md`
  - `export_fp16_sdpa_kv_island.sh`
- Added CLI/debug flag:
  - `--vulkan_debug_fp32_kv_cache`
- Added foundation-local exporter helper:
  - `_force_custom_kv_cache_fp32()` in
    `my_research/foundation/exporters/xnnpack.py`

Behavior:

- Only active when explicitly requested.
- Keeps `--use_sdpa_with_kv_cache` enabled for Vulkan.
- Keeps decoder export dtype as fp16.
- Forces transformed CustomKVCache-like buffers to fp32 and casts update inputs
  to the cache dtype before calling the original update implementation.
- Records debug metadata in the manifest:
  - `vulkan_debug_fp32_kv_cache`
  - `vulkan_debug_fp32_kv_cache_modules`

First artifact exported successfully:

- Command:
  `SEQ_LEN=512 bash my_research/foundation/debug/vulkan_fp16_correctness/export_fp16_sdpa_kv_island.sh`
- Output:
  `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_512_fp16_sdpa_kv_fp32_island`
- Manifest records:
  - `dtype: fp16`
  - `vulkan_export_dtype: fp16`
  - `vulkan_force_fp16: true`
  - `use_sdpa_with_kv_cache: true`
  - `vulkan_debug_fp32_kv_cache: true`
  - `vulkan_debug_fp32_kv_cache_modules: 24`
- Export completed successfully. Runtime correctness still needs to be tested on
  Android.

Runtime debug results:

- First run failed before execution because the runner hard-coded Vulkan decoder
  embedding inputs to fp32:
  `Input 0 has unexpected scalar type: expected Half but was Float`.
- Fixed `my_research/foundation/runner/xnnpack_backend.cpp`:
  - Reads decoder `forward` input 0 dtype from method metadata.
  - Casts merged prompt embeddings and decode-step embeddings to that dtype.
  - This preserves fp32 for old stable Vulkan artifacts and uses fp16 for true
    fp16 decoder artifacts.
- Rebuilt unified runner:
  `JOBS=8 VULKAN_QUERY_POOL_SIZE=65536 my_research/foundation/scripts/build_backend_and_runner.sh unified`.
- Ran:
  `internvl3_vulkan_1b_512_fp16_sdpa_kv_fp32_island`
  - Execution succeeded.
  - Output remained corrupted with the same `Deb`/`EDITOR` pattern.
  - Conclusion: forcing KV-cache/update inputs to fp32 is not enough.

Second debug experiment:

- Added `--vulkan_debug_block_sdpa_delegate`.
  - Keeps `sdpa_with_kv_cache` in the graph.
  - Blocks `torch.ops.llama.sdpa_with_kv_cache.default` from Vulkan delegation.
  - Lets the custom/portable op path execute it outside Vulkan while the rest of
    the decoder remains Vulkan-lowered.
- Added script:
  `my_research/foundation/debug/vulkan_fp16_correctness/export_fp16_sdpa_portable.sh`
- Exported and ran:
  `internvl3_vulkan_1b_512_fp16_sdpa_portable_fp32_kv`
  - Execution succeeded.
  - Output was still identical/corrupted.
  - Conclusion: Vulkan-delegated `sdpa_with_kv_cache` is not the primary cause.
    The next suspects are other Vulkan fp16 decoder ops such as RMSNorm/native
    layer norm, softmax, linear/layout transitions, or logits.

## Superseded InternVL3 Vulkan-Friendly Text Decoder Overlay

Context:

- Vulkan decoder export hit data-dependent symbolic guard failures in upstream
  Llama decoder helpers:
  - RoPE used `input_pos[-1].item()` and `narrow(...)`.
  - Attention mask slicing used another `input_pos[-1].item()` + `narrow(...)`.
- A previous broad `narrow` patch was unsafe, so this fix is foundation-local
  and scoped to InternVL3 text decoder construction.

Changes:

- Added, then removed `my_research/foundation/models/internvl3/text_decoder.py`.
  - `make_vulkan_friendly_text_decoder(model)` patches an already-built
    ExecuTorch Llama-style decoder in-place.
  - RoPE dynamic cache positions use tensor `index_select` instead of Python
    scalar `.item()` + `narrow`.
  - Attention dynamic mask slicing avoids `narrow`; for the custom
    `sdpa_with_kv_cache` path it passes `None` when the SDPA module does not
    require a mask.
- Export wiring:
  - Initially applied automatically for `internvl3_*` decoder models after
    `_prepare_for_llama_export(...)`.
  - This wiring was removed after matching the upstream Llama Vulkan tutorial's
    `qmode=8da4w` source-transform path, which exports successfully without
    changing the model code.
- Added, then removed eager validation script:
  - `my_research/foundation/debug/vulkan_fp16_correctness/internvl3_text_decoder_eager_smoke.py`

Validation:

- fp32 eager smoke:
  `PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch python my_research/foundation/debug/vulkan_fp16_correctness/internvl3_text_decoder_eager_smoke.py --dtype fp32 --max_seq_len 128 --max_context_len 128 --max_new_tokens 8`
  - Completed successfully.
  - Generated: `The capital of South Korea is Seoul.`
- fp16 eager smoke:
  same command with `--dtype fp16`.
  - Completed successfully.
  - Generated: `The capital of South Korea is Seoul.`

Notes:

- This did not modify upstream ExecuTorch source.
- Current exporter behavior intentionally uses the original upstream Llama-style
  InternVL3 decoder model. The overlay files were removed to avoid confusion.

## Vulkan 8da4w Correct Export Path

Context:

- User pointed out the official ExecuTorch tutorial:
  "Exporting Llama 3.2 1B/3B Instruct to ExecuTorch Vulkan and running on
  device" uses:
  `--vulkan -qmode 8da4w -G 64`.
- Earlier foundation code routed Vulkan `8da4w` through a custom
  `VulkanQuantizer(is_dynamic=True, weight_bits=4)` path. That generated
  `quantized_decomposed.quantize_per_tensor.tensor` graph ops, which the Vulkan
  partitioner skipped under dynamic shapes. Export then failed in
  `to_executorch()` with:
  `RuntimeError: Missing out variants: {'quantized_decomposed::quantize_per_tensor'}`.

Fix:

- Updated `my_research/foundation/exporters/xnnpack.py`:
  - `vulkan_8w` / `8w` still uses the Vulkan PT2E weight-only quantizer.
  - `4w`, `8da8w`, and `8da4w` now use the upstream Llama `qmode`
    source-transform path via `llm_config.quantization.qmode`.
  - Default Vulkan `8da4w` `text_group_size` is now `64`, matching the tutorial.
  - Removed automatic InternVL3 decoder overlay application.
- Updated `my_research/foundation/cli.py` help text to distinguish the Vulkan
  PT2E `vulkan_8w` path from the Llama `qmode` paths.
- Updated `my_research/foundation/docs/mobile_backend_flow.md` with the corrected
  quantization semantics.

Validation:

- Command:
  `PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch python -m my_research.foundation.cli export --backend vulkan --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_512_fp32_8da4w_qmode_decoder_only --decoder_only_from /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_512_fp16 --decoder_model internvl3_1b --model_path /workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf --checkpoint /workspace/streamingvlm/my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth --max_seq_len 512 --max_context_len 512 --dtype fp32 --vision_quant fp16 --decoder_quant 8da4w --embedding_quant fp16 --decoder_input_mode embeddings --dynamic_shape --use_sdpa_with_kv_cache`
- Export result:
  - Succeeded.
  - Vulkan partitioner reported `Found 1 Vulkan subgraphs to be partitioned`.
  - Partition included `et_vk.linear_dq8ca_q4gsw.default`,
    `torchao.choose_qparams_affine.default`, `et_vk.rms_norm.default`,
    `sdpa_with_kv_cache.default`, and RoPE-related Vulkan ops.
- Android run:
  - Artifact:
    `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_512_fp32_8da4w_qmode_decoder_only`
  - Runner:
    `executorch/build-android-unified/foundation/xnnpack_qnn_runner`
  - Device:
    `R3KYC01FW1P`
  - Output began with the correct image description:
    `Two cats are sleeping on pink bedding with two remote controls placed near them.`
  - Forced 64 tokens continued beyond the concise answer, as expected.

8192-token validation:

- Command:
  `PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch python -m my_research.foundation.cli export --backend vulkan --artifact_root /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp32_8da4w_qmode_decoder_only --decoder_only_from /workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp16 --decoder_model internvl3_1b --model_path /workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf --checkpoint /workspace/streamingvlm/my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth --max_seq_len 8192 --max_context_len 8192 --dtype fp32 --vision_quant fp16 --decoder_quant 8da4w --embedding_quant fp16 --decoder_input_mode embeddings --dynamic_shape --use_sdpa_with_kv_cache`
- Export result:
  - Succeeded.
  - Artifact:
    `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp32_8da4w_qmode_decoder_only`
  - Reused vision/embedding/tokenizer from:
    `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp16`
  - Manifest records `max_seq_len: 8192`, `max_context_len: 8192`,
    `dtype: fp32`, `decoder: 8da4w`, `text_group_size: 64`.
  - Vulkan partitioner reported `Found 1 Vulkan subgraphs to be partitioned`.
  - Partition included `et_vk.linear_dq8ca_q4gsw.default`,
    `torchao.choose_qparams_affine.default`, `et_vk.rms_norm.default`,
    `sdpa_with_kv_cache.default`, and RoPE-related Vulkan ops.
- Android run result:
  - Command used the unified runner with the COCO cats image, prompt
    `Describe this image briefly using around 10 words.`, and
    `--force_generate_token 128`.
  - Push and tokenizer loading succeeded.
  - Runner exited with status `255` immediately after tokenizer load.
  - After failure, `adb devices -l` returned an empty device list, so the phone
    disconnected/restarted before logs could be pulled.
  - Earlier retry with `--force_generate_token 64` showed the same exit-255 and
    ADB disconnect behavior.
  - Current interpretation: the 8k `fp32 + qmode 8da4w` Vulkan artifact exports
    cleanly but is not runtime-stable on device. The failure appears before
    normal foundation output/log files are produced, likely during model load,
    Vulkan preparation, or early prefill memory allocation.

Follow-up mixed manifest and text-only validation:

- Created a diagnostic manifest:
  `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp16_vision_embedding_8da4w_decoder/manifest.json`
- Purpose:
  - Keep `vision_encoder_pte`, `text_embedding_pte`, and `tokenizer_path`
    pointing directly at the existing 8k fp16 artifact:
    `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp16`
  - Point only `text_decoder_pte` at the newly exported 8k `8da4w` qmode
    decoder:
    `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp32_8da4w_qmode_decoder_only/text_decoder_vulkan.pte`
- Multimodal run with image:
  - Confirmed push paths used fp16 vision/embedding/tokenizer and only the
    `8da4w` decoder.
  - Still failed with exit status `255` immediately after tokenizer load, and
    ADB temporarily lost the device.
- Text-only run:
  - Command omitted `--image`, so the foundation launcher set `text_only=True`
    and did not pass `--image_path` to `xnnpack_qnn_runner`.
  - Succeeded on device with prompt
    `Write one short sentence about mobile AI.`
  - Output was written to:
    `my_research/foundation/results/log/vulkan/internvl3_vulkan_1b_8k_fp16_vision_embedding_8da4w_decoder/foundation_output.txt`
  - `vision_output_stats.csv` contained only its header, confirming the vision
    encoder path was not executed.
  - Current interpretation: the 8k `8da4w` decoder itself can load and generate
    text on Vulkan. The crash is tied to the multimodal path, most likely the
    large fp16 vision/embedding load, vision execution, embedding merge, or
    combined memory pressure rather than decoder-only text generation.

Vulkan qmode `8da4w` with explicit `--vulkan-force-fp16`:

- Added foundation CLI support for `--vulkan-force-fp16` /
  `--vulkan_force_fp16`.
  - This keeps `--dtype fp32` as the export graph dtype.
  - It sets `llm_config.backend.vulkan.force_fp16 = True` and passes
    `force_fp16=True` to the Vulkan partitioner.
- Updated `my_research/foundation/exporters/xnnpack.py` so Vulkan force-fp16
  is enabled either when `--dtype fp16` is requested or when the new explicit
  force flag is set.
- Per user request, an initial sequential 512/4096/8192 export was stopped.
  Then all existing unlimited-thread export jobs were killed and restarted with
  max 32 thread environment variables:
  `OMP_NUM_THREADS=32`, `MKL_NUM_THREADS=32`, `OPENBLAS_NUM_THREADS=32`,
  `NUMEXPR_NUM_THREADS=32`, `TORCH_NUM_THREADS=32`.
- 4096 export was cancelled per user request.
- Successful exports:
  - `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_512_fp32_8da4w_qmode_forcefp16_decoder_only`
  - `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp32_8da4w_qmode_forcefp16_decoder_only`
- Both manifests record:
  - `dtype: fp32`
  - `vulkan_export_dtype: fp32`
  - `vulkan_force_fp16: true`
  - `decoder: 8da4w`
  - `text_group_size: 64`
  - `enable_dynamic_shape: true`
- 8k text-only Android run:
  - Artifact:
    `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_8k_fp32_8da4w_qmode_forcefp16_decoder_only`
  - Command omitted `--image`, so the launcher did not pass `--image_path`.
  - Initial ADB push attempt failed while copying the decoder PTE with
    `no response`; after reconnecting the device, rerun succeeded.
  - Output was written to:
    `my_research/foundation/results/log/vulkan/internvl3_vulkan_1b_8k_fp32_8da4w_qmode_forcefp16_decoder_only/foundation_output.txt`
  - The generated answer began:
    `AI is used to perform tasks that would be difficult if not impossible to perform by humans.`
  - `vision_output_stats.csv` contained only its header, confirming no vision
    encoder execution in this text-only run.

512 Vulkan decoder-only `--quantize_kv_cache` attempt:

- Command attempted:
  `--backend vulkan --max_seq_len 512 --max_context_len 512 --dtype fp32 --vulkan-force-fp16 --decoder_quant 8da4w --quantize_kv_cache --decoder_only_from .../internvl3_vulkan_1b_512_fp16`
- Initial failure:
  - Upstream `replace_kv_cache_with_quantized_kv_cache()` needs
    `quantized_decomposed.quantize_per_token.out`.
  - Existing Android build dirs had `EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=OFF`,
    so no host `quantized_ops_aot_lib` was available for export-time
    `torch.ops.load_library()`.
  - Error:
    `AssertionError: Expected 1 library but got 0`.
- Built host target:
  `cmake --build /workspace/streamingvlm/executorch/build-x86 --target quantized_ops_aot_lib -j 36`
  - Produced:
    `/workspace/streamingvlm/executorch/build-x86/kernels/quantized/libquantized_ops_aot_lib.so`
- Exporter update:
  - Added a foundation-local search-path helper in
    `my_research/foundation/exporters/xnnpack.py`.
  - When `--quantize_kv_cache` is requested, it copies the host AOT library to a
    clean temp directory and appends that directory to `executorch.__path__` so
    upstream's recursive lookup sees exactly one `quantized_ops_aot_lib` match.
- Retry status:
  - The AOT op registration issue was resolved and export progressed into
    Vulkan partitioning.
  - Vulkan partitioner reported many skipped int8 KV-cache/update-cache nodes
    because Vulkan does not support those int8/custom quantized KV ops.
  - The command was interrupted before completion.
  - Artifact directory exists but no `manifest.json` was produced:
    `my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_512_fp32_8da4w_qmode_forcefp16_kv8_decoder_only`

## llama.cpp Vulkan VLM README

Context:

- User cloned `llama.cpp` under `/workspace/streamingvlm/llama.cpp` and asked
  for a README describing how to run VLMs with Vulkan from that tree.

Changes:

- Added `llama.cpp/README_VULKAN_VLM.md`.
  - Documents Linux Vulkan build with `-DGGML_VULKAN=ON`.
  - Documents Android NDK cross-compile with Vulkan enabled.
  - Explains llama.cpp multimodal flow through `libmtmd`,
    `llama-mtmd-cli`, `llama-server`, model GGUF, and `mmproj` GGUF.
  - Includes host and Android execution examples, `--list-devices`,
    `-ngl 99`, `--device`, CPU fallback, and debugging checks.

Notes:

- This is a local documentation file for the research workspace, not an
  upstream contribution.

## ExecuTorch Vision + llama.cpp Decoder Hybrid Note

Context:

- User asked whether ExecuTorch could run the vision encoder continuously and
  pass the resulting vision embeddings into llama.cpp's decoder when a user
  query arrives.
- The motivation is to keep ExecuTorch for mobile vision backend experiments
  while using llama.cpp's more accessible runtime KV-memory APIs for decoder
  cache manipulation experiments.

Changes:

- Added `hybrid_docs/executorch_vision_llamacpp_decoder.md`.
- The note documents:
  - llama.cpp's VLM flow under `tools/mtmd`
  - `mtmd_helper_decode_image_chunk()` as the external embedding injection point
  - Android feasibility and constraints
  - why embedding transfer is realistic but cross-runtime KV-cache transfer is
    not practical
  - a Linux-first prototype plan before Android integration

Notes:

- No ExecuTorch source was modified.
- No llama.cpp source was modified.

## llama.cpp Android VLM Hybrid Docs

Context:

- User explored llama.cpp as an Android VLM runtime and as a possible decoder
  target for ExecuTorch vision-encoder output.
- Detailed commands, logs, timings, memory, and backend-specific notes now live
  under `hybrid_docs/` instead of this foundation tracker.

Changes:

- Added/updated:
  - `hybrid_docs/README.md`
  - `hybrid_docs/executorch_vision_llamacpp_decoder.md`
  - `hybrid_docs/llamacpp_android_cpu_vlm_smoke.md`
  - `hybrid_docs/llamacpp_android_vulkan_vlm_smoke.md`
  - `hybrid_docs/llamacpp_android_opencl_vlm_attempt.md`
  - `hybrid_docs/llamacpp_android_internvl3_1b_smoke.md`
  - `hybrid_docs/llamacpp_android_memory_summary.md`
- Added sample images:
  - `my_research/foundation/sample_coco_cats.jpg`
  - `my_research/foundation/sample_coco_cats_448.jpg`

Key outcomes:

- SmolVLM-500M ran successfully on Android CPU and Vulkan.
- OpenCL built successfully but rejected `Samsung Xclipse 940`, so it fell back
  to CPU on the tested device.
- InternVL3 1B ran successfully on Android CPU and Vulkan.
- llama.cpp InternVL dynamic tiling explained the `1280` image-token run:
  original `640x480` input became `5 * 256` tokens. Resizing to `448x448`
  reduced it to `256` tokens.
- On the tested Xclipse 940 device, Vulkan improved token decode but was slower
  for VLM vision/prompt prefill.
- Memory details are summarized in
  `hybrid_docs/llamacpp_android_memory_summary.md`.
- `hybrid_docs/README.md` now acts as the high-level index and conclusion page
  for runtime terminology, streaming-task direction, memory behavior, backend
  outcomes, and links to the detailed hybrid notes.
- Added `hybrid_docs/aot_static_graph_vs_runtime_token_loop.md` to preserve the
  follow-up discussion about ExecuTorch AOT max-shape memory growth,
  llama.cpp's runtime-managed token loop, 8192-token prefill behavior, and why
  AOT remains useful for fixed-shape mobile accelerator paths.

Notes:

- No ExecuTorch source was modified.
- No llama.cpp source was modified.
- Local external checkouts/build/model artifacts live under ignored paths such
  as `llama.cpp/` and `third_party/`.

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