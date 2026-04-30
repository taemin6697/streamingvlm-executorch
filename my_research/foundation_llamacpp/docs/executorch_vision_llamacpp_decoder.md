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

## Implemented Prototype: Option A Split-Process Bridge

The current working prototype uses a split-process bridge. It intentionally does
not modify upstream ExecuTorch or upstream llama.cpp.

```text
Process 1: hybrid_vision_dump
  image bin
  -> ExecuTorch Module(vision_encoder_qnn.pte)
  -> QNN/HTP delegated InternVL vision path
  -> projected image embeddings
  -> vision_embedding.svlmemb

Process 2: hybrid_decode
  vision_embedding.svlmemb + original image file + text prompt
  -> llama.cpp mtmd_tokenize() only for text/image chunk layout
  -> mtmd_helper_decode_image_chunk() with external embedding pointer
  -> llama.cpp OpenCL decoder prefill and token generation
```

The bridge lives under:

```text
my_research/foundation_llamacpp/hybrid_bridge/
```

Important files:

```text
hybrid_embedding_file.h/.cpp
  Small binary `.svlmemb` file format for projected image embeddings.

hybrid_vision_dump.cpp
  ExecuTorch-side tool. Runs the QNN vision encoder and writes `.svlmemb`.

hybrid_decode.cpp
  llama.cpp-side tool. Reads `.svlmemb`, builds the mtmd layout, and injects
  external embeddings through mtmd_helper_decode_image_chunk().

CMakeLists.txt
  Overlay build. Pulls in ExecuTorch and llama.cpp without editing upstream.

run_android_hybrid_bridge.py
  Android orchestration: preprocess image, push binaries/models/libs, run both
  processes, sample memory, pull logs, generate CSVs and plots.
```

## Current Implementation Snapshot

As of 2026-04-30, this workspace has two related but separate Android paths:

```text
1. Hybrid bridge:
   ExecuTorch QNN vision encoder + llama.cpp OpenCL decoder

2. Standalone OpenCL precise baseline:
   llama.cpp OpenCL vision encoder/projector + llama.cpp OpenCL decoder
```

Both paths are implemented as project-specific overlays under
`my_research/foundation_llamacpp`. Upstream `llama.cpp` and upstream
ExecuTorch source files are intentionally not modified.

### Hybrid Bridge Files

```text
my_research/foundation_llamacpp/hybrid_bridge/hybrid_vision_dump.cpp
  ExecuTorch/QNN vision process. Loads `vision_encoder_qnn.pte`, runs the
  InternVL3 projected vision encoder, and writes a `.svlmemb` embedding file.

my_research/foundation_llamacpp/hybrid_bridge/hybrid_decode.cpp
  llama.cpp decoder process. Loads GGUF model + mmproj, reads `.svlmemb`, uses
  mtmd only to recover text/image layout, then feeds the external embedding into
  `mtmd_helper_decode_image_chunk()`.

my_research/foundation_llamacpp/hybrid_bridge/hybrid_embedding_file.h/.cpp
  The split-process embedding file format. Current payload is float32 projected
  embeddings with shape metadata.

my_research/foundation_llamacpp/run_android_hybrid_bridge.py
  Android runner for the split-process bridge. It preprocesses the image, pushes
  runtime files, launches both processes through a remote shell script, samples
  memory, pulls phase CSVs/logs, and generates result plots.
```

The hybrid runner enforces **load-before-run** ordering:

```text
1. Start `hybrid_decode` and `hybrid_vision_dump` together.
2. `hybrid_decode` loads llama.cpp runtime/model/mmproj/context, then writes
   `decoder_ready.flag`.
3. `hybrid_vision_dump` loads the ExecuTorch/QNN module and image tensor, then
   writes `vision_ready.flag`.
4. The remote coordinator creates `start_encode.flag` only after both ready flags
   exist.
5. QNN `V_Encode` starts after both model-loading paths are complete.
6. `.svlmemb` is written, decoder reads it, then image/text prefill and token
   decode run.
```

This ordering is important for timeline interpretation. Earlier prototypes ran
vision first and loaded the decoder afterward, which made memory and phase plots
misleading for a streaming system that should keep the decoder loaded before a
query arrives.

### Standalone OpenCL Precise Baseline Files

```text
my_research/foundation_llamacpp/hybrid_bridge/opencl_phase_mtmd.cpp
  Project overlay for precise standalone llama.cpp OpenCL measurement. It mirrors
  the single-turn `llama-mtmd-cli` InternVL flow but records foundation-style
  phase rows. It is a separate tool and does not patch upstream llama.cpp.

my_research/foundation_llamacpp/run_android_llamacpp.py
  Android runner for standalone llama.cpp backends. For `--backend opencl`, when
  `opencl_phase_mtmd` exists in the build dir, the runner uses it instead of
  upstream `llama-mtmd-cli` and pulls `foundation_phase_stats.csv`.
```

`opencl_phase_mtmd` deliberately separates phases that upstream
`mtmd_helper_eval_chunks()` normally hides inside one helper call:

```text
LayoutTokenize
  mtmd_tokenize() splits prompt into text and image chunks.

V_Encode
  mtmd_encode_chunk() runs llama.cpp's OpenCL vision encoder and InternVL
  projector. This corresponds to the old log line `image slice encoded in ...`.

ImagePrefill
  mtmd_helper_decode_image_chunk() feeds already-projected image embeddings into
  the LLM/KV context. This corresponds to the old log line
  `image decoded (batch 1/1) in ...`.

T_Prefill
  mtmd_helper_eval_chunk_single() runs text-token prefill chunks.

D
  One generated-token llama_decode() call per row.
```

The first `opencl_phase_mtmd` version followed mtmd chunk order exactly, which
allowed a small text prefill chunk before image encoding. That was confusing for
the intended streaming interpretation. The current version scans image chunks
after tokenization, runs all `V_Encode` phases first, copies the projected
embeddings, then performs text/image prefill in layout order using those
precomputed embeddings:

```text
current OpenCL precise ordering:
  L_DecoderRuntimeInit
  L_DecoderLoad
  ImageLoad
  LayoutTokenize
  V_Encode
  T_Prefill / ImagePrefill
  D
```

This makes standalone OpenCL comparable with the hybrid bridge, where QNN
`V_Encode` also happens before decoder prefill.

## What the QNN Vision Encoder Contains

For InternVL3, the ExecuTorch `vision_encoder_pte` is not only the raw vision
tower. It includes the projector path needed to produce decoder-ready image
embeddings.

The upstream Qualcomm model wrapper defines:

```text
InternVL3VisionEncoder
  vision_tower = InternVLVisionModel(config.vision_config)
  multi_modal_projector = InternVLMultiModalProjector(config)
```

Its forward path is:

```text
pixel_values
-> vision_tower(...).last_hidden_state
-> remove CLS token for default feature select strategy
-> pixel_shuffle/downsample
-> reshape to image token sequence
-> multi_modal_projector(...)
-> projected image embeddings
```

The observed QNN output shape for InternVL3-1B is:

```text
1 x 256 x 896
```

That matches llama.cpp's decoder input embedding dimension for the
`InternVL3-1B-Instruct-Q8_0` text model:

```text
256 image tokens x 896 decoder embedding dim
```

Therefore, the fair phase-level comparison is:

```text
Pure llama.cpp OpenCL:
  image slice encoded
    ~= InternVL vision tower + pixel shuffle/downsample + mm projector

Hybrid:
  QNN Vision Encoder
    ~= InternVL vision tower + pixel shuffle/downsample + multi_modal_projector
```

The llama.cpp `image decoded` line is different. It means the already projected
image embeddings are being decoded/prefilled into the LLM context. It is not the
vision encoder or projector.

## Embedding File Boundary

The current bridge boundary is a file, not shared memory:

```text
QNN output
-> CPU-accessible tensor
-> vision_embedding.svlmemb on Android local storage
-> hybrid_decode reads the file
-> CPU vector<float>
-> llama.cpp image chunk embedding pointer
-> llama.cpp uploads/uses it through the selected backend
```

The `.svlmemb` file contains:

```text
magic/version
dtype = float32
n_dims
n_values
shape[]
float32 values[]
```

This is deliberately simple for the first prototype. It makes the boundary easy
to inspect and preserves upgrade safety because neither runtime needs upstream
patches. It is not the final low-latency path. For a production-style bridge,
replace the file round trip with one of:

```text
same-process integration
memfd/ashmem/shared memory
native Android IPC with shared buffer handles
```

## Build Guide

The bridge is an overlay CMake project. Build it from
`my_research/foundation_llamacpp/hybrid_bridge/`; do not edit upstream
`llama.cpp` or upstream ExecuTorch sources.

### Build Inputs and Dependency Roles

The hybrid bridge uses two independent dependency stacks:

```text
llama.cpp OpenCL stack
  purpose:
    Build and run the decoder side, including ggml OpenCL kernels.
  key files:
    llama.cpp/
    third_party/OpenCL-Headers/
    third_party/OpenCL-ICD-Loader/build-android/libOpenCL.so

ExecuTorch QNN stack
  purpose:
    Build and run the QNN-delegated vision side.
  key files:
    executorch/
    executorch/build-android-unified/
    QNN_SDK_ROOT/lib/aarch64-android/libQnn*.so
    QNN_SDK_ROOT/lib/hexagon-v*/unsigned/libQnnHtpV*Skel.so
```

Do not confuse the OpenCL SDK files with QNN SDK files:

```text
OpenCL-Headers + libOpenCL.so
  Used to compile/link llama.cpp GGML_OPENCL.

QNN_SDK_ROOT / QAIRT / QNN SDK
  Used by ExecuTorch Qualcomm backend and the QNN vision PTE runtime.

Hexagon SDK for llama.cpp HTP backend
  Used by llama.cpp GGML_HEXAGON, not by this QNN-vision/OpenCL-decoder bridge.
```

The current successful Android bridge build cache used:

```text
CMAKE_TOOLCHAIN_FILE = /opt/android-ndk-r26c/build/cmake/android.toolchain.cmake
ANDROID_ABI          = arm64-v8a
ANDROID_PLATFORM     = android-30
EXECUTORCH_BUILD_DIR = /workspace/streamingvlm/executorch/build-android-unified
GGML_OPENCL          = ON
GGML_OPENMP          = OFF
OpenCL include       = /workspace/streamingvlm/third_party/OpenCL-Headers
OpenCL library       = /workspace/streamingvlm/third_party/OpenCL-ICD-Loader/build-android/libOpenCL.so
```

### CMake Targets

The overlay defines three project-specific targets:

```text
hybrid_decode
  llama.cpp-side decoder bridge.
  Depends on llama-common and mtmd.
  Can be built on host for API validation or Android for real runs.

hybrid_vision_dump
  ExecuTorch-side QNN vision embedding dumper.
  Requires an Android ExecuTorch build with Qualcomm/QNN backend support.
  Only meaningful for Android QNN runs.

opencl_phase_mtmd
  Standalone llama.cpp OpenCL precise measurement tool.
  Depends on llama-common and mtmd.
  Used by `run_android_llamacpp.py --backend opencl` when present.
  Does not use ExecuTorch or QNN.
```

Relevant CMake options:

```text
HYBRID_BRIDGE_BUILD_LLAMA_DECODER=ON
  Build llama.cpp-side overlay tools (`hybrid_decode` and `opencl_phase_mtmd`).

HYBRID_BRIDGE_BUILD_EXECUTORCH_VISION=ON
  Build hybrid_vision_dump.

LLAMA_CPP_ROOT=/workspace/streamingvlm/llama.cpp
  llama.cpp checkout. Defaults correctly from the overlay layout.

EXECUTORCH_ROOT=/workspace/streamingvlm/executorch
  ExecuTorch checkout. Defaults correctly from the overlay layout.

EXECUTORCH_BUILD_DIR=/workspace/streamingvlm/executorch/build-android-unified
  Existing ExecuTorch Android build/install tree used by find_package(executorch).
```

### Host Sanity Build

Use the host build first to verify that the llama.cpp external-embedding decoder
bridge compiles and links. This does not build or run QNN vision.

```bash
cmake \
  -S /workspace/streamingvlm/my_research/foundation_llamacpp/hybrid_bridge \
  -B /workspace/streamingvlm/my_research/foundation_llamacpp/build-hybrid-host \
  -DHYBRID_BRIDGE_BUILD_EXECUTORCH_VISION=OFF \
  -DHYBRID_BRIDGE_BUILD_LLAMA_DECODER=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build \
  /workspace/streamingvlm/my_research/foundation_llamacpp/build-hybrid-host \
  --target hybrid_decode opencl_phase_mtmd \
  -j2
```

Expected artifact:

```text
my_research/foundation_llamacpp/build-hybrid-host/hybrid_decode
my_research/foundation_llamacpp/build-hybrid-host/opencl_phase_mtmd
```

If this fails, fix the llama.cpp-side include/link issue first before attempting
Android QNN. The known fixes already captured in the overlay are:

```text
LLAMA_BUILD_COMMON=ON
LLAMA_BUILD_TOOLS=ON
include llama.cpp/common
include llama.cpp/tools/mtmd
include llama.cpp/vendor
include llama.cpp/ggml/include and ggml/src
link llama-common and mtmd
```

### Android OpenCL + QNN Bridge Build

This is the build used for the working hybrid run.

Set environment variables:

```bash
export ANDROID_NDK_ROOT=/opt/android-ndk-r26c
export QNN_SDK_ROOT=/workspace/streamingvlm/executorch/backends/qualcomm/sdk/qnn/qairt/2.37.0.250724
```

If your shell uses `ANDROID_NDK` instead of `ANDROID_NDK_ROOT`, either export
both or replace the path in the CMake command explicitly.

Configure:

```bash
cmake \
  -S /workspace/streamingvlm/my_research/foundation_llamacpp/hybrid_bridge \
  -B /workspace/streamingvlm/my_research/foundation_llamacpp/build-hybrid-android-opencl \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DHYBRID_BRIDGE_BUILD_EXECUTORCH_VISION=ON \
  -DHYBRID_BRIDGE_BUILD_LLAMA_DECODER=ON \
  -DEXECUTORCH_BUILD_DIR=/workspace/streamingvlm/executorch/build-android-unified \
  -DGGML_OPENCL=ON \
  -DGGML_OPENMP=OFF \
  -DOpenCL_INCLUDE_DIR=/workspace/streamingvlm/third_party/OpenCL-Headers \
  -DOpenCL_LIBRARY=/workspace/streamingvlm/third_party/OpenCL-ICD-Loader/build-android/libOpenCL.so \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
```

Build:

```bash
cmake --build \
  /workspace/streamingvlm/my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --target hybrid_decode hybrid_vision_dump opencl_phase_mtmd \
  -j2
```

Expected artifacts:

```text
my_research/foundation_llamacpp/build-hybrid-android-opencl/hybrid_decode
my_research/foundation_llamacpp/build-hybrid-android-opencl/hybrid_vision_dump
my_research/foundation_llamacpp/build-hybrid-android-opencl/opencl_phase_mtmd
```

Depending on the generator and CMake layout, the binaries may also appear under
a `bin/` subdirectory. The Python runner checks both locations.

Useful cache checks:

```bash
grep -E 'ANDROID_PLATFORM|ANDROID_ABI|GGML_OPENCL|HYBRID_BRIDGE|EXECUTORCH_BUILD_DIR|OpenCL' \
  my_research/foundation_llamacpp/build-hybrid-android-opencl/CMakeCache.txt
```

Expected values:

```text
ANDROID_ABI=arm64-v8a
ANDROID_PLATFORM=android-30
GGML_OPENCL=ON
HYBRID_BRIDGE_BUILD_EXECUTORCH_VISION=ON
HYBRID_BRIDGE_BUILD_LLAMA_DECODER=ON
EXECUTORCH_BUILD_DIR=/workspace/streamingvlm/executorch/build-android-unified
```

### What the Runner Pushes

`run_android_hybrid_bridge.py` pushes:

```text
hybrid_vision_dump
hybrid_decode
vision_encoder_qnn.pte
InternVL3-1B-Instruct-Q8_0.gguf
mmproj-InternVL3-1B-Instruct-Q8_0.gguf
preprocessed frame_0000.bin
layout image jpg
QNN SDK libQnn*.so
QNN HTP skel libQnnHtpV*Skel.so
libqnn_executorch_backend.so
llama.cpp runtime .so files from build bin/lib directories
```

The remote process sets:

```text
LD_LIBRARY_PATH=.
ADSP_LIBRARY_PATH=.
```

This keeps the QNN runtime and HTP skel lookup local to the remote work
directory.

### Build Troubleshooting

#### `arg.h: No such file or directory`

Cause:

```text
hybrid_decode includes llama.cpp common headers, but the overlay did not expose
llama.cpp/common.
```

Fix:

```text
Add llama.cpp/common to target_include_directories(hybrid_decode).
```

The current CMake already does this.

#### `nlohmann/json_fwd.hpp: No such file or directory`

Cause:

```text
llama.cpp common headers require vendored nlohmann/json.
```

Fix:

```text
Add llama.cpp/vendor to target_include_directories(hybrid_decode).
```

The current CMake already does this.

#### `cannot find -lllama-common` or `cannot find -lmtmd`

Cause:

```text
llama.cpp common/tool libraries were not built by the subdirectory configure.
```

Fix:

```text
Force LLAMA_BUILD_COMMON=ON and LLAMA_BUILD_TOOLS=ON before add_subdirectory(llama.cpp).
```

The current CMake already does this.

#### `memfd_create@LIBC_R` Link Error

Symptom:

```text
ld.lld: error: undefined reference due to --no-allow-shlib-undefined:
memfd_create@LIBC_R
```

Cause:

```text
The bridge was configured with an older Android API level than the prebuilt
ExecuTorch QNN backend.
```

Fix:

```text
Use ANDROID_PLATFORM=android-30, matching executorch/build-android-unified.
```

#### OpenCL Header or Library Not Found

Symptoms:

```text
Could NOT find OpenCL
CL/opencl.h not found
OpenCL_LIBRARY missing
```

Fix:

```bash
-DOpenCL_INCLUDE_DIR=/workspace/streamingvlm/third_party/OpenCL-Headers
-DOpenCL_LIBRARY=/workspace/streamingvlm/third_party/OpenCL-ICD-Loader/build-android/libOpenCL.so
```

#### QNN Runtime Loads but Vision PTE Fails

Check that the runner pushed:

```text
libQnn*.so from QNN_SDK_ROOT/lib/aarch64-android
libQnnHtpV*Skel.so from QNN_SDK_ROOT/lib/hexagon-v*/unsigned
libqnn_executorch_backend.so from executorch/build-android-unified/backends/qualcomm
```

Also check:

```text
ADSP_LIBRARY_PATH=.
LD_LIBRARY_PATH=.
```

If the device SoC differs, set the runner `--soc-model` so it chooses the right
HTP skel version:

```text
SM8750 -> hexagon-v79
SM8650 -> hexagon-v75
SM8550 -> hexagon-v73
SM8450 -> hexagon-v69
SM8350 -> hexagon-v68
```

#### Local `libOpenCL.so` Breaks Runtime Device Detection

This is a runtime issue caused by a build artifact, but it is common after
pushing all local runtime libraries.

Symptom:

```text
ggml_opencl: platform IDs not available
warning: no usable GPU found
```

Fix used on the device:

```bash
adb shell 'cd /data/local/tmp/llama-vlm && mv libOpenCL.so libOpenCL.so.bak'
```

Then use:

```text
--device GPUOpenCL
```

Reason:

```text
On this Qualcomm Android device, the system OpenCL library detected Adreno
correctly, while the pushed ICD loader interfered with platform discovery.
```

### Rebuild Discipline

If changing CMake options such as Android API level, OpenCL on/off, or
ExecuTorch build directory, prefer a fresh build directory. Do not mix host and
Android CMake caches.

Recommended separate directories:

```text
build-hybrid-host
  host-only decoder compile/link validation

build-hybrid-android-opencl
  Android QNN vision + OpenCL llama.cpp decoder
```

Build outputs should stay out of git. Do not commit build directories, pushed
Android runtime libraries, GGUF files, PTE files, or `.svlmemb` binaries.

## Precise Phase Instrumentation

Both the hybrid bridge and the standalone OpenCL baseline now record phase CSVs
in the same style as the foundation ExecuTorch runners.

The canonical schema is:

```text
row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,
col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,
kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx
```

The most important columns today are `row_type`, `elapsed_s_start`,
`elapsed_s_end`, `total_ms`, and `token_idx`. Memory/KV columns are preserved in
the schema for compatibility with the foundation runner style, but the current
llama.cpp overlay tools do not populate them yet.

### Hybrid Phase Files

Vision process:

```text
vision_phase_stats.csv

L_VisionLoad
  ExecuTorch Module construction and encoder.load().

ImageLoad
  Load preprocessed CHW float32 image and build the input tensor.

V_Encode
  QNN projected InternVL image embedding generation.

EmbeddingFileWrite
  Write projected embeddings to `.svlmemb`.
```

Decoder process:

```text
decoder_phase_stats.csv

L_DecoderRuntimeInit
  llama.cpp argument parsing plus OpenCL runtime/device/kernel setup before
  model/context construction. This is not GGUF model loading.

L_DecoderLoad
  llama.cpp model/context load plus mtmd/mmproj context setup.

ExternalEmbeddingRead
  Read `.svlmemb` into CPU memory.

LayoutTokenize
  Load the layout image and run mtmd_tokenize() to recover text/image chunks.

ImagePrefill
  Feed external projected image embeddings through mtmd_helper_decode_image_chunk().

T_Prefill
  Text chunk eval through mtmd_helper_eval_chunk_single().

D
  One generated-token llama_decode() call per row.
```

The hybrid Python runner pulls both phase files and merges them into:

```text
foundation_proc.csv
```

The summary metrics that used to live in `foundation_proc.csv` are now written
to:

```text
foundation_summary.csv
```

When phase rows are available, `phase_duration_stacked_bar.png` and
`memory_timeline_plot.png` are generated from the precise phase rows. This makes
the hybrid plots closer to the foundation ExecuTorch plots.

The Android runner now enforces coordinated load-before-run ordering. It pushes a
remote shell script, starts `hybrid_decode` and `hybrid_vision_dump` together,
waits for both to create ready flags, then creates `start_encode.flag`. The
vision process loads the QNN module and image tensor before it writes
`vision_ready.flag`, so `V_Encode` only starts after decoder init/load and vision
load are both complete.

Important caveat: Option A is split-process. `vision_phase_stats.csv` and
`decoder_phase_stats.csv` each use their own process-local zero time. The runner
therefore no longer appends decoder phases after vision phases; it keeps the
process-local timelines aligned from the common remote-script launch and sorts
the merged `foundation_proc.csv` by phase start. This is good for visualizing the
intended load-first schedule, but it still does not expose every shell/ADB/file
handoff gap as a separate row. For strict end-to-end latency, also report the
external wall time and eventually add parent-process phase rows around process
launch and handoff.

Latest coordinated OpenCL hybrid run:

```text
result:
  my_research/foundation_llamacpp/results/log/hybrid_bridge_opencl/InternVL3-1B-Instruct-Q8_0/

L_DecoderRuntimeInit  = 13644 ms
L_DecoderLoad         = 2871 ms
V_Encode              = 369 ms
EmbeddingFileWrite    = 2 ms
ExternalEmbeddingRead = 3 ms
LayoutTokenize        = 9 ms
ImagePrefill          = 57 ms
T_Prefill             = 10 ms + 210 ms
D per generated token = mostly 12-15 ms
prompt eval           = 276.44 ms / 271 tokens
token decode          = 410.27 ms / 31 runs
llama.cpp total       = 2225.85 ms
```

### Standalone OpenCL Phase Files

For standalone OpenCL, `run_android_llamacpp.py --backend opencl` uses
`opencl_phase_mtmd` when that binary exists in the build directory. The runner
pulls:

```text
foundation_phase_stats.csv
```

and writes canonical precise phase rows to:

```text
results/log/opencl/InternVL3-1B-Instruct-Q8_0/foundation_proc.csv
```

Summary metrics are written to:

```text
results/log/opencl/InternVL3-1B-Instruct-Q8_0/foundation_summary.csv
```

The standalone OpenCL phases are:

```text
L_DecoderRuntimeInit
  llama.cpp argument parsing plus OpenCL runtime/device/kernel setup before
  GGUF model/context construction.

L_DecoderLoad
  llama.cpp text model/context load plus mtmd/mmproj vision context setup.

ImageLoad
  Load the input image into an mtmd bitmap.

LayoutTokenize
  mtmd_tokenize() layout construction.

V_Encode
  llama.cpp OpenCL vision encoder + InternVL projector.

ImagePrefill
  Feed projected OpenCL image embeddings into llama.cpp context/KV.

T_Prefill
  Text chunk prefill.

D
  One generated-token llama_decode() call per row.
```

Important ordering detail:

```text
The standalone OpenCL precise tool runs all V_Encode phases before any
T_Prefill/ImagePrefill phase.
```

This is intentional. Upstream `mtmd_helper_eval_chunks()` normally follows chunk
order, so a small text chunk before the image marker can run before image encode.
For the streaming/mobile comparison, we want the same conceptual ordering as the
hybrid path: vision encoding happens before decoder prefill. Therefore
`opencl_phase_mtmd` tokenizes first, scans image chunks, runs `mtmd_encode_chunk`
for each image, copies the projected embeddings, then performs text/image prefill
in layout order using those precomputed embeddings.

Latest precise standalone OpenCL run:

```text
result:
  my_research/foundation_llamacpp/results/log/opencl/InternVL3-1B-Instruct-Q8_0/

L_DecoderRuntimeInit = 13850 ms
L_DecoderLoad        = 2827 ms
V_Encode             = 714 ms
ImagePrefill         = 36 ms
T_Prefill            = 19 ms + 214 ms
D per generated token = mostly 12-16 ms
prompt eval          = 269.20 ms / 271 tokens
token decode         = 377.38 ms / 29 runs
llama.cpp total      = 2444.37 ms
```

### Phase Plot Filtering

`foundation_proc.csv` should keep all rows for traceability. The current
`phase_duration_stacked_bar.png` intentionally filters setup/loading phases so
the figure focuses on execution:

```text
excluded from phase_duration_stacked_bar.png:
  L_DecoderRuntimeInit
  L_DecoderLoad
  L_VisionLoad
  ImageLoad
  LayoutTokenize
```

Rows that remain visible in the stacked bar include:

```text
V_Encode
EmbeddingFileWrite
ExternalEmbeddingRead
ImagePrefill
T_Prefill
Decode
```

`Decode` is the sum of per-token `D` rows. The stacked bar order is based on the
first `elapsed_s_start` of each included phase, not a hardcoded display order.

`memory_timeline_plot.png` remains a `MemAvailable` timeline. For the hybrid
runner, phase spans can be overlaid on the memory timeline. For standalone
OpenCL, the plot is currently simpler and primarily validates the run-level
memory trend.

## Android Hybrid Run From Scratch

The expected inputs are:

```text
ExecuTorch QNN manifest:
  my_research/foundation/results/model/qnn/internvl3_1b_qnn_1k_16a8w/manifest.json

llama.cpp text model:
  llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf

llama.cpp mmproj:
  llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf

sample image:
  my_research/foundation_llamacpp/sample_coco_cats_448.jpg
```

Required environment:

```bash
export QNN_SDK_ROOT=/path/to/qairt-or-qnn-sdk
export ANDROID_NDK=/path/to/android-ndk
```

The Android bridge build must use the same minimum Android API level as the
ExecuTorch QNN backend build. In this workspace the working value was:

```text
ANDROID_PLATFORM=android-30
```

Using `android-28` failed because `libqnn_executorch_backend.so` referenced
`memfd_create@LIBC_R`.

The runner defaults are intentionally aligned with the standalone OpenCL
baseline:

```text
n_ctx    = 32768
n_batch  = 2048
n_ubatch = 512
```

Use those defaults when comparing phase times. Changing them changes the OpenCL
KV and compute buffers and can make prompt eval numbers incomparable.

Typical Android run:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --manifest my_research/foundation/results/model/qnn/internvl3_1b_qnn_1k_16a8w/manifest.json \
  --vision-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_coco_cats_448.jpg \
  --prompt "Describe this image briefly." \
  --n-predict 32 \
  --ctx-size 32768 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --results-root my_research/foundation_llamacpp/results/log/hybrid_bridge_opencl
```

The result directory should be:

```text
my_research/foundation_llamacpp/results/log/hybrid_bridge_opencl/InternVL3-1B-Instruct-Q8_0/
```

Canonical result artifacts:

```text
foundation_output.txt
  Canonical llama.cpp decoder stdout.

foundation_exit_code.txt
  Final decoder process exit code.

foundation_proc.csv
  Canonical precise phase rows for plotting and comparison.

foundation_summary.csv
  Run-level summary metrics parsed from logs.

vision_output_stats.csv
  QNN vision input/output shape and phase metrics.

android_memory_timeline.csv
  External Android /proc/meminfo sampling. Use MemAvailable for plots.

memory_timeline_plot.png
  MemAvailable timeline only.

phase_duration_stacked_bar.png
  Phase-level runtime breakdown.

hybrid_vision_stdout*.txt
  Raw ExecuTorch/QNN vision process log. This is useful for debugging and is
  separate from the canonical decoder stdout.
```

Do not keep duplicate decoder logs locally. The decoder stdout should be
canonicalized as `foundation_output.txt`. Intermediate names such as
`hybrid_decode_stdout.txt` are on-device implementation details.

## Android Standalone OpenCL Precise Run From Scratch

Use this path when you want a pure llama.cpp OpenCL baseline with the same
phase-level detail as the hybrid bridge.

Prerequisite:

```text
my_research/foundation_llamacpp/build-hybrid-android-opencl/opencl_phase_mtmd
```

Build it with:

```bash
cmake --build \
  /workspace/streamingvlm/my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --target opencl_phase_mtmd \
  -j2
```

Run:

```bash
python3 my_research/foundation_llamacpp/run_android_llamacpp.py \
  --backend opencl \
  --build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_coco_cats_448.jpg \
  --prompt "Describe this image briefly." \
  --max-new-tokens 32 \
  --threads 4 \
  --n-gpu-layers 99 \
  --device GPUOpenCL \
  --ctx-size 32768 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --device-workdir /data/local/tmp/llama-vlm \
  --results-root my_research/foundation_llamacpp/results/log
```

Expected result directory:

```text
my_research/foundation_llamacpp/results/log/opencl/InternVL3-1B-Instruct-Q8_0/
```

Important artifacts:

```text
foundation_output.txt
  Canonical stdout from `opencl_phase_mtmd`.

foundation_phase_stats.csv
  Raw phase CSV emitted by the Android binary.

foundation_proc.csv
  Canonical precise phase rows. This is what plots should consume.

foundation_summary.csv
  Run-level summary metrics parsed from stdout.

phase_duration_stacked_bar.png
  Execution-only phase breakdown. Loading/setup phases are filtered from the
  stacked bar but remain in `foundation_proc.csv`.

memory_timeline_plot.png
  MemAvailable timeline.
```

OpenCL loader warning:

```text
Do not push the local `third_party/OpenCL-ICD-Loader/.../libOpenCL.so` to the
device by default.
```

On the tested Qualcomm device, pushing the local ICD loader caused:

```text
ggml_opencl: platform IDs not available
error while handling argument "--device": invalid device: GPUOpenCL
```

The fixed `run_android_llamacpp.py` intentionally avoids pushing the local
`libOpenCL.so`, allowing the Android system Qualcomm OpenCL loader to discover
Adreno correctly.

## Current Matched OpenCL Result

Latest matched-settings hybrid run:

```text
result:
  my_research/foundation_llamacpp/results/log/hybrid_bridge_opencl/InternVL3-1B-Instruct-Q8_0/

decoder settings:
  n_ctx    = 32768
  n_batch  = 2048
  n_ubatch = 512

buffers:
  OpenCL KV buffer      = 384.00 MiB
  OpenCL compute buffer = 297.99 MiB

timing:
  V_Encode / QNN vision           = 369 ms
  Embedding file write            = 2 ms
  External embedding read         = 3 ms
  ImagePrefill                    = 57 ms
  T_Prefill                       = 10 ms + 210 ms
  Token Decode                    = mostly 12-15 ms/token
  Prompt Eval                     = 276.44 ms / 271 tokens
  Token Decode total              = 410.27 ms / 31 runs
  llama.cpp total                 = 2225.85 ms
```

Standalone llama.cpp OpenCL baseline with matched decoder settings and precise
phase ordering:

```text
result:
  my_research/foundation_llamacpp/results/log/opencl/InternVL3-1B-Instruct-Q8_0/

V_Encode             = 714 ms
ImagePrefill         = 36 ms
T_Prefill            = 19 ms + 214 ms
Token Decode         = mostly 12-16 ms/token
Prompt Eval          = 269.20 ms / 271 tokens
Token Decode total   = 377.38 ms / 29 runs
llama.cpp total      = 2444.37 ms
```

Interpretation:

```text
QNN V_Encode 369 ms
  Compare against standalone OpenCL `V_Encode 714 ms`.

ExternalEmbeddingRead 3 ms
  Read the `.svlmemb` file into CPU memory. This is not ADB transfer.

ImagePrefill 57 ms
  Pass projected image embeddings into llama.cpp/KV through
  mtmd_helper_decode_image_chunk().

Prompt Eval / T_Prefill
  llama.cpp decoder-side text/image prefill counters. They are not vision
  encoder time.
```

## Timing Interpretation and Caveats

The phase times in `foundation_output.txt` are llama.cpp internal counters. They
are useful for diagnosis, but they are not always equal to externally observed
wall time.

Especially on OpenCL/GPU backends:

```text
OpenCL work can be queued asynchronously.
Some synchronization cost can appear in a later phase.
Vision and decoder work can share the same GPU/backend resources.
```

This matters for the standalone OpenCL run. The standalone path runs CLIP/vision
and decoder on the same OpenCL device:

```text
image -> OpenCL vision/mmproj -> OpenCL decoder prefill -> OpenCL token decode
```

The hybrid path separates the heavy vision stage onto QNN:

```text
image -> QNN vision/projector -> file embedding -> OpenCL decoder
```

So a shorter hybrid prompt eval does not mean QNN directly accelerated prompt
eval. It likely means the decoder's OpenCL path is less affected by preceding
vision/mmproj work on the same GPU queue/resources.

Use this convention in analysis:

```text
End-to-end/QoS:
  Prefer external wall time and user-visible response latency.

Phase diagnosis:
  Use llama.cpp phase counters, but explain possible async/synchronization
  movement between adjacent OpenCL phases.

Vision comparison:
  Compare QNN Vision Encoder against OpenCL image slice encoded.

Decoder comparison:
  Compare image decoded, prompt eval, and token decode only after matching
  n_ctx, n_batch, n_ubatch, n_predict, prompt, model, and image.
```

## Troubleshooting

### OpenCL Device Not Found

Symptom:

```text
ggml_opencl: platform IDs not available
warning: no usable GPU found, --gpu-layers option will be ignored
```

Cause seen in this workspace:

```text
A pushed local libOpenCL.so ICD loader in the remote working directory
interfered with Android's system Qualcomm OpenCL stack.
```

Fix:

```bash
adb shell 'cd /data/local/tmp/llama-vlm && mv libOpenCL.so libOpenCL.so.bak'
```

Then pass:

```text
--device GPUOpenCL
```

Expected log:

```text
ggml_opencl: selected platform: 'QUALCOMM Snapdragon(TM)'
ggml_opencl: device: 'QUALCOMM Adreno(TM) 830 ...'
```

If using the Python runner, prefer not to push a local `libOpenCL.so` unless the
target device actually requires it. For this device, the system OpenCL library
worked better.

### Android Link Error: memfd_create

Symptom:

```text
undefined reference due to --no-allow-shlib-undefined: memfd_create@LIBC_R
```

Cause:

```text
The hybrid Android build used an older API level than the ExecuTorch QNN backend.
```

Fix:

```text
Configure the bridge build with ANDROID_PLATFORM=android-30.
```

### Duplicate Logs

Symptom:

```text
foundation_output.txt
hybrid_decode_stdout.txt
hybrid_decode_opencl_stdout.txt
```

Meaning:

```text
These are duplicate decoder stdout files from manual reruns and intermediate
on-device filenames.
```

Fix:

```text
Keep only foundation_output.txt as the canonical decoder stdout.
Keep foundation_exit_code.txt as the canonical exit code.
Keep hybrid_vision_stdout*.txt only for QNN vision process debugging.
```

The runner now removes local `hybrid_decode_stdout.txt` after finalizing metrics.

### Prompt Eval Not Matching Between Runs

First check that these are identical:

```text
model GGUF
mmproj GGUF
prompt
image
n_predict
n_ctx
n_batch
n_ubatch
gpu layers
device
```

For the matched comparison in this workspace:

```text
n_ctx    = 32768
n_batch  = 2048
n_ubatch = 512
```

If those match but standalone OpenCL prompt eval is still slower, remember that
standalone OpenCL performs vision/mmproj on the same OpenCL device immediately
before decoder prefill. Due to OpenCL queue synchronization and resource reuse,
some preceding work or synchronization can be charged to the next llama.cpp
decode/prompt-eval counter.

### External Embedding Shape Mismatch

Symptom:

```text
embedding size mismatch
```

Expected for InternVL3-1B:

```text
embedding shape = 1 x 256 x 896
n_values        = 229376
```

The decoder checks:

```text
embedding.values.size() == n_image_tokens * llama_model_n_embd_inp(model)
```

If this fails, the likely causes are:

```text
wrong model/mmproj pair
wrong image preprocessing size
wrong PTE variant
projector not included in the external encoder
dynamic tiling mismatch changing image token count
```

### ADB or Remote State Problems

If ADB hangs or stale files persist:

```bash
adb kill-server
adb start-server
adb devices
```

Then clear only the intended remote work directory:

```bash
adb shell 'rm -rf /data/local/tmp/streamingvlm_hybrid_bridge'
```

Do not delete unrelated device files or model directories unless they are known
to be disposable.

## Next Steps

The split-process file bridge proves the runtime boundary. The next useful
improvements are:

```text
1. Add an apples-to-apples repeated-run script for standalone OpenCL vs hybrid.
2. Record true external wall-clock phases around both processes.
3. Replace `.svlmemb` file handoff with shared memory or same-process handoff.
4. Add an explicit "projected vision embedding" naming convention in plots.
5. Investigate whether llama.cpp can expose stricter OpenCL synchronization
   markers for phase timing validation without upstream modification.
```
