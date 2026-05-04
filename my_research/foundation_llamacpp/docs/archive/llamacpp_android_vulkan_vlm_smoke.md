# llama.cpp Android Vulkan VLM Smoke Test

This note records the working Android Vulkan path for running the same
SmolVLM-500M `llama-mtmd-cli` smoke test used in the CPU-only note.

## Result

Vulkan works on the tested Android device.

The run used:

```text
Vulkan0 (Samsung Xclipse 940)
```

The text model was offloaded to Vulkan:

```text
load_tensors: offloaded 33/33 layers to GPU
llama_kv_cache: Vulkan0 KV buffer size = 320.00 MiB
```

The multimodal projector and vision encoder also used Vulkan:

```text
clip_ctx: CLIP using Vulkan0 backend
```

The output was correct:

```text
In this image, we can see two cats on the bed. There is a remote on the bed.
```

## Build Dependencies

The Android NDK provided `libvulkan.so` and C Vulkan headers, but the llama.cpp
Vulkan backend also needed Khronos C++ and SPIR-V headers:

```text
vulkan/vulkan.hpp
spirv/unified1/spirv.hpp
```

For a local smoke build, clone them outside committed project sources:

```bash
cd /workspace/streamingvlm
mkdir -p third_party
git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git third_party/Vulkan-Headers
git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git third_party/SPIRV-Headers
```

`third_party/` is ignored because these are external dependency checkouts.

## Build

Build llama.cpp for Android with Vulkan enabled and OpenMP disabled:

```bash
cd /workspace/streamingvlm/llama.cpp

cmake -B build-android-vulkan-noomp \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_OPENMP=OFF \
  -DCMAKE_CXX_FLAGS="-I/workspace/streamingvlm/third_party/Vulkan-Headers/include -I/workspace/streamingvlm/third_party/SPIRV-Headers/include"

cmake --build build-android-vulkan-noomp --target llama-mtmd-cli -j
```

The important build output should include:

```text
-- Vulkan found
-- Including Vulkan backend
[100%] Built target llama-mtmd-cli
```

## Push Files

Use the same device directory as the CPU smoke test:

```bash
adb shell 'mkdir -p /data/local/tmp/llama-vlm'

cd /workspace/streamingvlm/llama.cpp

adb push build-android-vulkan-noomp/bin/llama-mtmd-cli /data/local/tmp/llama-vlm/
adb push build-android-vulkan-noomp/bin/*.so /data/local/tmp/llama-vlm/
```

The pushed Vulkan build includes `libggml-vulkan.so` in addition to the common
llama.cpp shared libraries.

The model directory and sample image can be reused from the CPU smoke test:

```text
/data/local/tmp/llama-vlm/SmolVLM-500M-Instruct-GGUF/
/data/local/tmp/llama-vlm/image.jpg
```

## Run

Use `--n-gpu-layers 99` to offload the model to Vulkan:

```bash
adb shell 'cd /data/local/tmp/llama-vlm && chmod +x llama-mtmd-cli && LD_LIBRARY_PATH=. ./llama-mtmd-cli \
  -m SmolVLM-500M-Instruct-GGUF/SmolVLM-500M-Instruct-Q8_0.gguf \
  --mmproj SmolVLM-500M-Instruct-GGUF/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  --image image.jpg \
  -p "Describe this image briefly." \
  -n 64 \
  -t 4 \
  --n-gpu-layers 99'
```

## Observed Timing

Observed timings on the tested device:

```text
image slice encoded in 14178 ms
image decoded in 5 ms
prompt eval: 82 tokens, 5.59 tok/s
decode eval: 21 runs, 66.05 tok/s
total time: 15530.12 ms
```

Interpretation:

- Decode token generation became faster than CPU-only.
- Vision encoding was much slower in this smoke run than the earlier CPU-only
  result, even though it used Vulkan. This may be first-run shader/pipeline
  overhead, backend scheduling overhead, or an unfavorable vision graph mapping.
- For the StreamingVLM hybrid direction, Vulkan llama.cpp is useful to test, but
  vision latency should be measured over repeated runs before assuming it is
  better than CPU for the mmproj/vision path.

## OpenCL Note

llama.cpp also has a GGML OpenCL backend (`-DGGML_OPENCL=ON`), with optional
Adreno-oriented kernels. It was not tested here.

On Android it is less plug-and-play than Vulkan because it depends on OpenCL
headers/libraries at build time and a compatible vendor OpenCL runtime on the
device. Vulkan should be the first GPU path to use unless a specific device or
experiment requires OpenCL.
