# Hybrid Runtime Notes

This directory collects the llama.cpp and ExecuTorch hybrid experiments for
mobile streaming VLM research. Use this file as the quick run guide. For build
details and implementation internals, see
`archive/executorch_vision_llamacpp_decoder.md`.

## Current Recommendation

For a mobile streaming visual assistant, use a hybrid runtime:

```text
streaming vision path:
  ExecuTorch QNN / Vulkan / XNNPACK

decoder and KV-control path:
  llama.cpp CPU / OpenCL / Vulkan / Hexagon
```

The practical runtime boundary is projected **vision embeddings**, not KV-cache.
Direct cross-runtime KV transfer is not practical because KV layout, RoPE
positioning, quantization, and backend buffer ownership differ by runtime.

## Models And Inputs

The current main model is InternVL3 1B with Q8_0 llama.cpp weights:

```text
text GGUF:
  llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf

mmproj GGUF:
  llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf

sample image:
  my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg

QNN vision manifest:
  my_research/foundation/results/model/qnn/internvl3_1b_qnn_1k_16a8w/manifest.json
```

Sample images live under `my_research/foundation_llamacpp/sample_images/`.
The current default is a center-cropped and resized Golden Gate Bridge photo from
Wikimedia Commons. Keep benchmark sample images resized to `448 x 448`. InternVL
dynamic tiling can turn larger images into multiple tiles; for apples-to-apples
measurements we use one tile:

```text
image tokens = 256
decoder embedding dim = 896
projected vision embedding shape = 1 x 256 x 896
```

The Q8_0 suffix is part of the model identity and should remain visible in result
paths:

```text
InternVL3-1B-Instruct-Q8_0
```

## Common Parameters

Use the unified Android runner and matched decoder settings when comparing CPU,
OpenCL GPU, and hybrid runs:

```text
context length:
  --ctx-size 32768

batch size:
  --batch-size 2048

micro-batch size:
  --ubatch-size 512

new tokens:
  --n-predict 32

force generation:
  --force-generation 64   # optional; continue until exactly 64 generated tokens

memory baseline:
  --baseline-window 5.0   # average MemAvailable from -5s to 0s before execution

prompt:
  "Describe this image briefly."
```

Changing `--ctx-size` changes KV-cache allocation. Changing `--n-predict`
changes generation length and runtime, but it does not meaningfully reduce
allocated KV memory.

`--force-generation N` overrides `--n-predict` for the run. GPU/OpenCL and
Hybrid continue through EOS/EOG in the instrumented overlay binaries. CPU uses
upstream `llama-mtmd-cli` with `--ignore-eos`, so it emits `N` tokens but the CPU
token transcript is reconstructed by the Python runner from stdout.

## Result Layout

Unified runner writes each run under `--results-root` using the **text GGUF stem**,
`--processor` (`cpu`, `opencl` for GPU, or `hybrid`), `--ctx-size`, and KV dtypes:

```text
<GGUF_stem>_<processor>_ctx_<N>_kv<KV>
```

`<KV>` comes from `--cache-type-k` / `--cache-type-v` (when omitted, both default to
`f16` in the folder slug → `_kv16`). Examples for InternVL3-1B Q8_0 at ctx `32768`
with default FP16 KV:

```text
my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_cpu_ctx_32768_kv16/
my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_opencl_ctx_32768_kv16/
my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_hybrid_ctx_32768_kv16/
```

Important artifacts:

```text
foundation_output.txt
  Canonical stdout log.

foundation_token_io.txt
  Prompt/response token text with special tokens and image placeholder tokens.
  Use this when you need a compact input/output transcript like the ExecuTorch
  foundation runner output.

foundation_exit_code.txt
  Process exit code.

foundation_proc.csv
  Canonical phase rows or summary rows, depending on backend/tooling.

foundation_summary.csv
  Run-level summary metrics when precise phase rows are available.

foundation_phase_stats.csv
  Raw standalone OpenCL precise phase CSV emitted on device.

vision_phase_stats.csv / decoder_phase_stats.csv
  Raw hybrid phase CSVs emitted by the two bridge processes.

android_memory_timeline.csv
  Android memory samples. Use MemAvailable for memory plots. By default, the
  first 5 seconds are pre-run baseline samples with negative elapsed_s values.

memory_usage_summary.txt
  System-wide memory usage summary computed as baseline average MemAvailable
  minus minimum runtime MemAvailable.

memory_timeline_plot.png
  MemAvailable timeline.

phase_duration_stacked_bar.png
  Runtime phase breakdown.
```

## Run CPU

Use CPU for a simple correctness baseline. This path uses upstream
`llama-mtmd-cli` through the unified Android runner.

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor cpu \
  --llama-build-dir llama.cpp/build-android-cpu-noomp \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
  --prompt "Describe this image briefly." \
  --n-predict 32 \
  --force-generation 64 \
  --threads 4 \
  --ctx-size 32768 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

Expected output directory:

```text
my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_cpu_ctx_32768_kv16/
```

Using a different GGUF (for example InternVL3-8B) changes `<GGUF_stem>` in the folder
name accordingly.

## Run OpenCL GPU

Use OpenCL for the standalone llama.cpp GPU baseline on Qualcomm Adreno devices.
When `opencl_phase_mtmd` exists in the build directory, the runner automatically
uses it instead of upstream `llama-mtmd-cli` to produce precise phase rows.

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor gpu \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
  --prompt "Describe this image briefly." \
  --n-predict 32 \
  --force-generation 64 \
  --threads 4 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --ctx-size 32768 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

Expected output directory for the command above (`--ctx-size 32768`, FP16 KV):

```text
my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_opencl_ctx_32768_kv16/
```

Changing `--ctx-size` or `--cache-type-k` / `--cache-type-v` updates the `_ctx_<N>` / `_kv…`
segments in the folder name (for example `_ctx_512_kv16` or `_ctx_32768_kv8`).

HF에서 `rope_scaling: { "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768 }` 인 체크포인트를 그대로 반영해 실행하는 예입니다. **롱컨텍스트 상한(약 128k)** 을 쓰려면 `--ctx-size 131072`처럼 올리면 되고, 메모리가 빠듯하면 `32768`로 두고 아래 RoPE 플래그만 맞춰도 됩니다.

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor gpu \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
  --prompt "Describe this image briefly." \
  --n-predict 32 \
  --force-generation 64 \
  --threads 4 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --ctx-size 131072 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --temperature 0.0 \
  --rope-scaling yarn \
  --rope-scale 4.0 \
  --yarn-orig-ctx 32768 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log
```

YaRN 예제 결과 디렉터리 예시:

```text
my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_opencl_ctx_131072_kv16/
```

GGUF 메타에 동일한 RoPE 설정이 이미 들어 있으면 위 `--rope-scaling` / `--rope-scale` / `--yarn-orig-ctx` 는 생략해도 됩니다. 추가로 조정할 때만 `--rope-freq-base`, `--rope-freq-scale`, `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-slow`, `--yarn-beta-fast` 를 붙이면 됩니다.

KV-cache를 8비트로 쓰려면 `--cache-type-k q8_0 --cache-type-v q8_0`처럼 지정하면 됩니다 (upstream llama.cpp `common_params`와 동일한 플래그명으로 그대로 전달됩니다). 결과 로그의 `llama_kv_cache` 줄에서 `K (q8_0)`, `V (q8_0)`로 표시되는지 확인하면 됩니다. 디바이스·백엔드 조합에 따라 해당 타입이 거부되면 로드 시 에러가 날 수 있습니다. OpenCL에서 초기화 단계(`common_fit_params`)가 `SET_ROWS` 등으로 abort 할 때는 `--fit off`로 자동 메모리 맞춤을 끄고 다시 시도하면 됩니다 (runner가 `opencl_phase_mtmd`에 그대로 넘깁니다).

Important OpenCL note:

```text
Do not push the local OpenCL ICD loader (`libOpenCL.so`) to the device by
default.
```

On the tested Qualcomm device, the Android system OpenCL loader discovers Adreno
correctly. Pushing the local ICD loader caused:

```text
ggml_opencl: platform IDs not available
invalid device: GPUOpenCL
```

The unified runner avoids pushing that local loader unless
`--push-opencl-loader` is explicitly set.

## Run Hybrid QNN Vision + OpenCL Decoder

Use the hybrid bridge for the main streaming-system experiment:

```text
ExecuTorch QNN:
  Fused PTE (vision tower + multi_modal_projector), or vision-tower-only PTE whose
  features are projected by the llama.cpp GGUF mmproj before decoder prefill.

llama.cpp OpenCL:
  layout tokenize + image/text prefill + token decode
```

The QNN manifest and the llama.cpp model/mmproj must use the same InternVL size.
For example, a 1B QNN manifest emits image embeddings sized for the 1B decoder
(`256 x 896` for the standard 256 image tokens). Do not pair that manifest with
an 8B GGUF/mmproj (`256 x 3584` expected), or `hybrid_decode` will stop with an
embedding size mismatch.

Typical run (**OpenCL FP16 KV cache**, K/V 모두 `f16`). KV cache를 8bit로
바꾸려면 `--cache-type-k q8_0 --cache-type-v q8_0 --fit off`를 사용하면
됩니다(결과 폴더는 `_kv8` 접미사).

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --image my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
  --prompt "Describe this image briefly." \
  --n-predict 32 \
  --force-generation 64 \
  --ctx-size 32768 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log \
  --force-push \
  --model-push
```

This `--vision` artifact is the 16a8w QNN vision-tower-only export:
`projector_included: false`, output shape `1 x 256 x 4096`. Because it stops
before InternVL3 `multi_modal_projector`, the decoder bridge must apply mmproj
before image prefill for this command to run end-to-end.

Expected output directory:

```text
my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_hybrid_ctx_32768_kv16/
```

The unified runner caches only model-like files (`--model`, `--mmproj`, and the
hybrid QNN `.pte`) on device. If they already exist under `--remote-root`, it
skips pushing them; pass `--model-push` to force re-push. Use `--force-push`
when changing the remote workdir contents wholesale. Runtime binaries,
shared libraries, scripts, and input images are always pushed so rebuilds are
reflected immediately. In hybrid mode, the runner starts the QNN vision process
and llama.cpp decoder process together, waits until both are loaded, then starts
QNN `V_Encode`.

### Vision-Tower-Only Artifacts

The current QNN `vision_encoder_pte` is fused: it runs InternVL3 `vision_tower`
and the model-size-specific `multi_modal_projector`, then writes embeddings in
the decoder input dimension. `run_android_hybrid_bridge.py --processor hybrid`
now accepts either the old manifest or a direct PTE:

```bash
--vision my_research/foundation_llamacpp/results/vision_models/<artifact>/vision_tower_preproj_qnn.pte
```

For experiments that need the raw visual features before that projector, use the
separate pre-projector export path:

```bash
export QNN_SDK_ROOT=/workspace/streamingvlm/executorch/backends/qualcomm/sdk/qnn/qairt/2.37.0.250724
export EXECUTORCH_ROOT=/workspace/streamingvlm/executorch
export ANDROID_NDK_ROOT=/opt/android-ndk-r26c
export LIBCXX_DIR=/opt/conda/envs/stream/lib/python3.11/site-packages/executorch/backends/qualcomm/sdk/libcxx-14.0.0
export PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LIBCXX_DIR:$EXECUTORCH_ROOT/build-x86/lib:${LD_LIBRARY_PATH:-}"

PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python -m my_research.foundation.models.internvl3.vision_encoder.export_pre_projector_qnn \
  --model-name internvl3_1b \
  --model-path OpenGVLab/InternVL3-1B-hf \
  --artifact-root my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750 \
  --soc-model SM8750 \
  --quant 16a8w \
  --calibration-images \
    my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
    my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg \
  --calibration-num 2
```

Do not pass `--encoder-weights my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth`
for this export: that file is a decoder-style checkpoint and contains no
`vision_tower.*` weights. The exporter now fails fast if `--encoder-weights`
does not contain vision-tower keys.

Expected artifact:

```text
my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte
```

Full exports write under:

```text
my_research/foundation_llamacpp/results/vision_models/<model>_vision_tower_preproj_qnn_<soc>/
```

and include `vision_tower_preproj_qnn_metadata.json` with
`projector_included: false`. These artifacts are intentionally not the same as
the old fused `vision_encoder_pte`: the output is still in vision hidden space,
so `hybrid_decode` needs a decoder-side mmproj/projector step before image
prefill. The current `hybrid_decode` supports this InternVL pre-projector path:
it detects `1 x 256 x 4096`, runs the GGUF `mmproj` through OpenCL, then feeds
the resulting `256 x 896` image embeddings into decoder prefill.

## Optional Backends

Older Vulkan and Hexagon command templates are kept in
`archive/executorch_vision_llamacpp_decoder.md` for historical reference. The
active unified runner currently exposes only the comparison modes needed for the
main experiment:

```text
--processor cpu
--processor gpu      # OpenCL GPU
--processor hybrid   # ExecuTorch QNN vision + llama.cpp OpenCL decoder
```

## Phase Names

Precise OpenCL and hybrid runs use these phase names:

```text
L_DecoderRuntimeInit
  llama.cpp argument parsing and OpenCL runtime/device/kernel setup.

L_DecoderLoad
  llama.cpp model/context/mmproj load.

L_VisionLoad
  ExecuTorch/QNN vision module load. Hybrid only.

ImageLoad
  Input image/tensor load.

LayoutTokenize
  mtmd_tokenize() text/image layout construction.

V_Encode
  Vision encoder (+ projector when fused), 또는 tower-only QNN 출력 후 디코더 쪽 mmproj.
  OpenCL: llama.cpp OpenCL vision path.
  Hybrid: ExecuTorch QNN `.pte` (매니페스트 또는 `--vision`).

EmbeddingFileWrite
  Write `.svlmemb`. Hybrid only.

ExternalEmbeddingRead
  Read `.svlmemb`. Hybrid only.

ImagePrefill
  Feed projected image embeddings into llama.cpp context/KV.

T_Prefill
  Text chunk prefill.

D
  One generated-token llama_decode() call.
```

`phase_duration_stacked_bar.png` filters load/setup phases so the figure focuses
on execution. The full trace remains in `foundation_proc.csv`.

## Current Matched Results

아래 숫자는 **동일 디코더 설정**(예: `n_ctx=32768`, `n_batch=2048`, `n_ubatch=512`)으로
잡은 과거 측정값입니다. 결과 폴더 경로는 러너 버전에 따라 예전
`results/log/opencl/<stem>/` 같은 하위 디렉터리 대신, 위 **Result Layout** 규칙의
`..._<stem>_opencl_ctx_32768_kv16/` 형태를 씁니다.

Latest standalone OpenCL precise run:

```text
result (representative folder name today):
  my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_opencl_ctx_32768_kv16/

V_Encode             = 714 ms
ImagePrefill         = 36 ms
T_Prefill            = 19 ms + 214 ms
token decode         = mostly 12-16 ms/token
prompt eval          = 269.20 ms / 271 tokens
token decode total   = 377.38 ms / 29 runs
llama.cpp total      = 2444.37 ms
```

Latest hybrid QNN vision + OpenCL decoder run:

```text
result (representative folder name today):
  my_research/foundation_llamacpp/results/log/InternVL3-1B-Instruct-Q8_0_hybrid_ctx_32768_kv16/

V_Encode / QNN vision = 369 ms
ImagePrefill          = 57 ms
T_Prefill             = 10 ms + 210 ms
token decode          = mostly 12-15 ms/token
prompt eval           = 276.44 ms / 271 tokens
token decode total    = 410.27 ms / 31 runs
llama.cpp total       = 2225.85 ms
```

Interpretation:

```text
Compare OpenCL V_Encode against hybrid QNN V_Encode.
Compare ImagePrefill/T_Prefill against decoder-side prefill behavior.
Do not compare QNN vision time against llama.cpp prompt eval directly.
```

## Memory Summary

llama.cpp reports memory roughly as:

```text
self = model + context + compute
```

For InternVL3 1B Q8_0 matched runs:

```text
OpenCL model buffer = about 500 MiB
OpenCL KV buffer    = 384 MiB at ctx 32768
OpenCL compute buf  = about 298 MiB for decoder
```

The KV-cache is resident runtime memory. On mobile SoCs, GPU-visible memory is
still unified DRAM, so OpenCL/Vulkan allocations affect system memory pressure.

## Build Documentation

This README intentionally focuses on running experiments. For CMake configure,
target descriptions, QNN library pushing, troubleshooting, and implementation
details, read:

```text
my_research/foundation_llamacpp/docs/archive/executorch_vision_llamacpp_decoder.md
```

## Document Index

- `for_cursor_llm_llamacpp.md`: Append-only development log for future agents.
- `archive/executorch_vision_llamacpp_decoder.md`: Detailed hybrid bridge and
  standalone OpenCL precise measurement guide.
- `archive/llamacpp_android_cpu_vlm_smoke.md`: SmolVLM-500M Android CPU smoke
  test.
- `archive/llamacpp_android_vulkan_vlm_smoke.md`: SmolVLM-500M Android Vulkan smoke
  test.
- `archive/llamacpp_android_opencl_vlm_attempt.md`: Android OpenCL build and
  unsupported Xclipse fallback.
- `archive/llamacpp_android_internvl3_1b_smoke.md`: InternVL3 1B CPU/Vulkan
  tests and the `448 x 448` resize follow-up.
- `archive/llamacpp_android_memory_summary.md`: Backend memory breakdown for
  earlier llama.cpp Android VLM runs.
- `archive/aot_static_graph_vs_runtime_token_loop.md`: ExecuTorch AOT memory
  growth vs llama.cpp runtime token execution.
