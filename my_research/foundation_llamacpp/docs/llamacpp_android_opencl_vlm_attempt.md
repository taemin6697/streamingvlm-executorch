# llama.cpp Android OpenCL VLM Attempt

This note records the Android OpenCL attempt for the same SmolVLM-500M
`llama-mtmd-cli` smoke test.

## Result

The Android OpenCL build succeeded, but the tested device did not run llama.cpp
through OpenCL.

Runtime detected the device OpenCL platform:

```text
ggml_opencl: selected platform: 'Samsung Mobile GPU Platform'
ggml_opencl: device: 'Samsung Xclipse 940 (OpenCL 3.0)'
```

But llama.cpp rejected it:

```text
Unsupported GPU: Samsung Xclipse 940
ggml_opencl: drop unsupported device.
warning: no usable GPU found, --gpu-layers option will be ignored
```

So the final execution fell back to CPU:

```text
load_tensors: CPU_Mapped model buffer size = 414.86 MiB
llama_kv_cache: CPU KV buffer size = 320.00 MiB
clip_ctx: CLIP using CPU backend
```

## Why

llama.cpp's OpenCL backend is primarily developed and documented for Qualcomm
Adreno GPUs, especially:

```text
Adreno 750 (Snapdragon 8 Gen 3)
Adreno 830 (Snapdragon 8 Elite)
Adreno X85 (Snapdragon X Elite)
```

The tested phone exposes OpenCL for `Samsung Xclipse 940`, but llama.cpp does
not accept it as a usable OpenCL backend device.

## Build Dependencies

The Android NDK did not include OpenCL headers. Build dependencies were added
under ignored local dependency checkouts:

```bash
cd /workspace/streamingvlm
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git third_party/OpenCL-Headers
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git third_party/OpenCL-ICD-Loader
```

Build the Android OpenCL ICD loader:

```bash
cd /workspace/streamingvlm

cmake -S third_party/OpenCL-ICD-Loader \
  -B third_party/OpenCL-ICD-Loader/build-android \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=/workspace/streamingvlm/third_party/OpenCL-Headers \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DANDROID_STL=c++_shared

cmake --build third_party/OpenCL-ICD-Loader/build-android -j
```

## Build llama.cpp

```bash
cd /workspace/streamingvlm/llama.cpp

cmake -B build-android-opencl-noomp \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DGGML_OPENMP=OFF \
  -DOpenCL_INCLUDE_DIR=/workspace/streamingvlm/third_party/OpenCL-Headers \
  -DOpenCL_LIBRARY=/workspace/streamingvlm/third_party/OpenCL-ICD-Loader/build-android/libOpenCL.so

cmake --build build-android-opencl-noomp --target llama-mtmd-cli -j
```

## Push Files

```bash
adb shell 'mkdir -p /data/local/tmp/llama-vlm-opencl'

cd /workspace/streamingvlm/llama.cpp

adb push build-android-opencl-noomp/bin/llama-mtmd-cli /data/local/tmp/llama-vlm-opencl/
adb push build-android-opencl-noomp/bin/*.so /data/local/tmp/llama-vlm-opencl/
adb push /workspace/streamingvlm/third_party/OpenCL-ICD-Loader/build-android/libOpenCL.so \
  /data/local/tmp/llama-vlm-opencl/

adb shell 'cp -r /data/local/tmp/llama-vlm/SmolVLM-500M-Instruct-GGUF /data/local/tmp/llama-vlm-opencl/'
adb shell 'cp /data/local/tmp/llama-vlm/image.jpg /data/local/tmp/llama-vlm-opencl/image.jpg'
```

## Run

```bash
adb shell 'cd /data/local/tmp/llama-vlm-opencl && chmod +x llama-mtmd-cli && LD_LIBRARY_PATH=. ./llama-mtmd-cli \
  -m SmolVLM-500M-Instruct-GGUF/SmolVLM-500M-Instruct-Q8_0.gguf \
  --mmproj SmolVLM-500M-Instruct-GGUF/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  --image image.jpg \
  -p "Describe this image briefly." \
  -n 64 \
  -t 4 \
  --n-gpu-layers 99'
```

## Observed Fallback Timing

Because OpenCL was rejected, these are CPU fallback timings:

```text
image slice encoded in 2717 ms
image decoded in 512 ms
prompt eval: 82 tokens, 23.99 tok/s
decode eval: 23 runs, 24.84 tok/s
total time: 4679.77 ms
```

This closely matches the previous CPU-only smoke test and confirms OpenCL was
not used for acceleration on this device.
