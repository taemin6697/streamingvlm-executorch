# llama.cpp Android CPU VLM Smoke Test

This note records the working Android CPU path for running a small llama.cpp
VLM with `llama-mtmd-cli`.

## Tested Model

Small VLM:

```text
ggml-org/SmolVLM-500M-Instruct-GGUF
```

Files used on device:

```text
/data/local/tmp/llama-vlm/
  llama-mtmd-cli
  libggml-base.so
  libggml-cpu.so
  libggml.so
  libllama-common.so
  libllama.so
  libmtmd.so
  image.jpg
  SmolVLM-500M-Instruct-GGUF/
    SmolVLM-500M-Instruct-Q8_0.gguf
    mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
```

The sample image was:

```text
/workspace/streamingvlm/my_research/foundation/sample_coco_cats.jpg
```

## Build

The first Android CPU build produced binaries that depended on `libomp.so`.
The device did not have that library, so the working build disables OpenMP:

```bash
cd /workspace/streamingvlm/llama.cpp

cmake -B build-android-cpu-noomp \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=OFF

cmake --build build-android-cpu-noomp --target llama-mtmd-cli -j
```

This still uses llama.cpp's GGML CPU backend. It is not ExecuTorch XNNPACK,
QNN, or Vulkan.

## Push Files

Create the device workspace:

```bash
adb shell 'mkdir -p /data/local/tmp/llama-vlm'
```

Push the executable and required shared libraries:

```bash
cd /workspace/streamingvlm/llama.cpp

adb push \
  build-android-cpu-noomp/bin/llama-mtmd-cli \
  build-android-cpu-noomp/bin/libggml-base.so \
  build-android-cpu-noomp/bin/libggml-cpu.so \
  build-android-cpu-noomp/bin/libggml.so \
  build-android-cpu-noomp/bin/libllama-common.so \
  build-android-cpu-noomp/bin/libllama.so \
  build-android-cpu-noomp/bin/libmtmd.so \
  /data/local/tmp/llama-vlm/
```

Push the model directory:

```bash
adb push models/SmolVLM-500M-Instruct-GGUF \
  /data/local/tmp/llama-vlm/SmolVLM-500M-Instruct-GGUF
```

Push the sample image:

```bash
adb push /workspace/streamingvlm/my_research/foundation/sample_coco_cats.jpg \
  /data/local/tmp/llama-vlm/image.jpg
```

## Run

Use `LD_LIBRARY_PATH=.` so Android can find the llama.cpp shared libraries in
the same directory as `llama-mtmd-cli`.

```bash
adb shell 'cd /data/local/tmp/llama-vlm && LD_LIBRARY_PATH=. ./llama-mtmd-cli \
  -m SmolVLM-500M-Instruct-GGUF/SmolVLM-500M-Instruct-Q8_0.gguf \
  --mmproj SmolVLM-500M-Instruct-GGUF/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  --image image.jpg \
  -p "Describe this image briefly." \
  -n 64 \
  -t 4'
```

## Observed Output

The run succeeded and produced:

```text
In this image we can see two cats are lying on the bed. We can also see a remote on the bed.
```

Observed timing:

```text
image slice encoded in 2688 ms
image decoded in 504 ms
prompt eval: 82 tokens, 23.96 tok/s
decode eval: 24 runs, 26.51 tok/s
total time: 4592.95 ms
```

## Errors Fixed

Initial run failed with:

```text
CANNOT LINK EXECUTABLE "./llama-mtmd-cli": library "libllama-common.so" not found
```

Fix:

- Push llama.cpp shared libraries from the Android build `bin/` directory.
- Run with `LD_LIBRARY_PATH=.`

Second run failed with:

```text
CANNOT LINK EXECUTABLE "./llama-mtmd-cli": library "libomp.so" not found
```

Fix:

- Rebuild with `-DGGML_OPENMP=OFF`.
- Push the no-OpenMP binaries and libraries.

## Notes

- This is a CPU-only smoke test.
- No Vulkan option is used.
- `-t 4` sets CPU threads. Try `-t 6` or `-t 8` if thermal and scheduling
  behavior are acceptable.
- Use the Q8 model and Q8 mmproj first. The F16 files are larger and slower on
  CPU.
