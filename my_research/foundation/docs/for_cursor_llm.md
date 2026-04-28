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