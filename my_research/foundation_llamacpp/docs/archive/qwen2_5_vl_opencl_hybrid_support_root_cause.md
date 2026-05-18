# Qwen2.5-VL OpenCL/Hybrid 지원 원인 분석

문서 목적: Qwen2.5-VL을 Android OpenCL + QNN hybrid 경로에서 붙일 때
출력이 틀어졌던 원인, 실제 수정 지점, 검증 결과를 나중에 다시 추적할 수
있게 정리한다. 이 문서는 line number 대신 파일명과 심볼명 기준으로 읽는다.

## 1. 목표

Qwen2.5-VL-3B-Instruct GGUF를 다음 두 경로에서 정상 동작시키는 것이 목표였다.

```text
full OpenCL GPU:
  Qwen decoder GGUF + Qwen mmproj GGUF + llama.cpp mtmd vision/mmproj
  모두 OpenCL backend에서 실행

QNN hybrid:
  QNN vision encoder(pre-merger) -> llama.cpp Qwen mmproj/patch merger
  -> OpenCL decoder
```

중요한 제약은 CPU fallback 없이 동작해야 한다는 점이다. CPU vision/mmproj는
원인 분리용 diagnostic으로만 사용한다.

## 2. 관찰된 증상

Golden Gate Bridge 이미지를 넣고 "What landmark or structure is shown in
this image? Answer briefly."를 물었을 때 다음처럼 경로별 결과가 달랐다.

```text
CPU vision/mmproj + OpenCL decoder:
  golden gate bridge

full CPU GGUF:
  golden gate bridge

full OpenCL GPU, Qwen CLIP FlashAttention auto/on:
  Empire State Building

QNN pre-merger hybrid, post_ln 누락 상태:
  이미지가 들어가긴 하지만 visual embedding이 틀어져 답변 품질이 깨짐
```

이 결과 때문에 decoder GGUF, Q8/F16 양자화, 이미지 자체, Qwen prompt format이
주 원인은 아니라고 판단했다. 같은 이미지와 같은 prompt가 CPU mtmd 경로에서는
정답을 냈기 때문이다.

## 3. 원인 1: Qwen2.5-VL OpenCL CLIP FlashAttention

### 현상

Qwen2.5-VL full OpenCL GPU 경로에서 CLIP vision backend도 OpenCL로 올라간
상태였지만, 기본 FlashAttention auto/on에서는 Golden Gate 이미지를 Empire
State Building으로 잘못 인식했다.

검증 로그:

```text
results/log/qwen25_clean_opencl_f16_all_gpu_single_image_golden_gate_manual
```

핵심 로그 형태:

```text
load_tensors: offloaded 37/37 layers to GPU
clip_ctx: CLIP using OpenCL backend
assistant: Empire State Building
```

반대로 같은 Qwen Q8/F16 GGUF를 host CPU 또는 CPU mtmd vision/mmproj로 돌리면
정상적으로 `golden gate bridge`가 나왔다.

### 실제 원인

Qwen2.5-VL vision encoder는 window attention pattern을 사용한다. 현재
llama.cpp mtmd의 OpenCL CLIP FlashAttention 경로는 이 graph를 받아 실행은
하지만, Adreno OpenCL에서 Qwen2.5-VL window attention mask가 들어간 visual
embedding을 올바르게 만들지 못했다.

따라서 원인은 다음이 아니었다.

```text
아님: 이미지 resize/preprocess 문제
아님: Qwen chat template/special token 문제
아님: decoder Q8/F16 양자화 문제
아님: OpenCL decoder 자체 문제
맞음: Qwen2.5-VL CLIP OpenCL FlashAttention correctness 문제
```

### 수정

파일:

```text
llama.cpp/tools/mtmd/clip.cpp
```

심볼:

```text
clip_model_loader::warmup(clip_ctx &, const clip_image_f32_batch &)
```

Qwen2.5-VL + OpenCL backend + window attention pattern이면 CLIP FlashAttention을
강제로 끈다.

```cpp
const bool is_opencl_backend =
    ctx_clip.backend &&
    std::strcmp(ggml_backend_name(ctx_clip.backend), "OpenCL") == 0;
const bool is_qwen25_window_attn =
    ctx_clip.model.proj_type == PROJECTOR_TYPE_QWEN25VL &&
    ctx_clip.model.hparams.n_wa_pattern > 0;

if (is_opencl_backend && is_qwen25_window_attn &&
    ctx_clip.flash_attn_type != CLIP_FLASH_ATTN_TYPE_DISABLED) {
    LOG_WRN("%s: disabling CLIP FlashAttention for Qwen2.5-VL on OpenCL; using non-FA OpenCL vision attention for correctness\n",
            __func__);
    ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
}
```

이 수정은 CPU fallback이 아니다. CLIP graph는 계속 OpenCL에서 실행하고,
attention 구현만 FlashAttention 대신 일반 OpenCL op path로 보낸다.

### 검증

로그:

```text
results/log/qwen25_opencl_flash_attn_guard_golden_gate/qwen_q8_all_gpu_guard_golden_gate_stdout.txt
```

핵심 결과:

```text
load_tensors: offloaded 37/37 layers to GPU
clip_ctx: CLIP using OpenCL backend
warmup: disabling CLIP FlashAttention for Qwen2.5-VL on OpenCL; using non-FA OpenCL vision attention for correctness
warmup: flash attention is disabled
image slice encoded in 1914 ms
assistant: golden gate bridge
```

F16 full OpenCL도 `--ctx-size 1024`에서는 같은 guard로 `golden gate bridge`를
냈다. 다만 F16 + ctx4096은 OpenCL model buffer가 약 5.9 GiB까지 올라가면서
ADB/device reset이 발생할 수 있었다. 이는 correctness 문제가 아니라 device
memory/driver pressure 이슈로 본다.

## 4. 원인 2: QNN pre-merger hybrid의 post_ln 누락

### QNN artifact 구조

현재 Qwen2.5-VL QNN artifact는 patch merger를 제외한 순수 pre-merger vision
encoder 출력이다.

```text
artifact:
  results/vision_models/qwen2_5_vl_3b_vision_encoder_premerger_qnn_1024tok_sm8750/vision_encoder_qnn.pte

output:
  [1, 1024, 1280]

metadata:
  patch_merger_included: false
  projector_included: false
  image_grid_thw: [1, 32, 32]
```

448x448 입력에서 patch size 14이므로 pre-merger token은 32 x 32 = 1024개다.
Qwen patch merger가 2x2 patch를 묶으면 decoder에 들어가는 image token은
1024 / 4 = 256개가 된다.

### 처음 의심한 것

처음에는 window reorder mismatch를 의심했다. Qwen2.5-VL은 vision block과
patch merger 사이에 window attention용 ordering을 사용하기 때문에, 외부 QNN
feature를 바로 patch merger에 넣을 때 `window_idx` / `inv_window_idx`가 틀리면
출력이 깨질 수 있다.

하지만 projector-only graph에는 이미 다음 경로가 들어가 있었다.

```text
pre-merger feature
  -> inv_window_idx reorder
  -> patch merger MLP
  -> window_idx restore
```

즉 window reorder 자체가 최종 원인은 아니었다.

### 실제 원인

Qwen patch merger는 MLP 전에 layer norm을 수행한다. HF 기준으로는
`visual.patch_merger.ln_q`에 해당하고, llama.cpp mtmd에서는 `model.post_ln_w`
/ `model.post_ln_b`로 로드된다.

여기서 `post_ln`은 "vision encoder 전체 뒤에 붙는 임의의 후처리"가 아니라,
Qwen patch merger가 자신의 MLP 입력으로 기대하는 정규화 단계다. Qwen2.5-VL의
vision block 출력은 아직 patch 단위의 hidden state이고, patch merger는 2x2
patch를 묶어 decoder embedding 차원으로 보내기 전에 이 hidden state에 norm을
걸어 분포를 맞춘다. 그래서 `post_ln`은 다음 경계에 위치한다.

```text
vision block output, pre-merger feature
  -> post_ln / visual.patch_merger.ln_q
  -> 2x2 patch merge reshape
  -> patch merger MLP
  -> decoder image embedding
```

full mtmd graph는 vision block 뒤 patch merger 전에 이 post norm을 적용한다.
하지만 QNN pre-merger feature를 받아 patch merger만 수행하는
`clip_graph_qwen2vl_projector_only` 경로에는 이 norm이 빠져 있었다.
shape은 여전히 맞기 때문에 crash가 나지 않고, 대신 "그럴듯하지만 다른 이미지
embedding"이 만들어지는 형태로 품질이 깨졌다.

정리하면:

```text
full llama.cpp Qwen vision:
  vision blocks -> post_ln -> patch merger -> decoder embedding

기존 QNN hybrid projector-only:
  QNN pre-merger feature -> patch merger -> decoder embedding
  post_ln 누락

수정 후:
  QNN pre-merger feature -> post_ln -> patch merger -> decoder embedding
```

### 수정

파일:

```text
llama.cpp/tools/mtmd/clip.cpp
```

심볼:

```text
clip_graph_qwen2vl_projector_only::build()
```

external pre-merger feature tensor를 만든 직후, patch merge reshape 전에
`model.post_ln_w`가 있으면 동일하게 norm을 적용한다.

```cpp
if (model.post_ln_w) {
    norm_type norm_t = proj_type == PROJECTOR_TYPE_QWEN25VL
        ? NORM_TYPE_RMS
        : NORM_TYPE_NORMAL;
    embeddings = build_norm(embeddings, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
}
```

Qwen2.5-VL은 RMS norm을 사용하므로 `PROJECTOR_TYPE_QWEN25VL`에서는
`NORM_TYPE_RMS`를 쓴다.

### 검증

실행 경로:

```text
QNN vision pre-merger
  -> llama.cpp Qwen mmproj/patch merger on OpenCL
  -> Qwen decoder on OpenCL
```

로그:

```text
results/log/qwen25_hybrid_qnn_premerger_postln_golden_gate/Qwen2.5-VL-3B-Instruct-Q8_0_hybrid_ctx_4096_image_kv16
```

핵심 결과:

```text
load_tensors: offloaded 37/37 layers to GPU
clip_ctx: CLIP using OpenCL backend
external embedding slice is pre-projector: feature_tokens=1024 image_tokens=256 feature_embd=1280 projected_embd=2048 consumed_floats=1310720/1310720
assistant: golden gate bridge
```

측정값:

```text
V_Encode:     272 ms
Mmproj:        26 ms
ImagePrefill: 1994 ms
```

## 5. runner 정책 변경

파일:

```text
my_research/foundation_llamacpp/runner/cli.py
```

기존에는 Qwen-VL hybrid 실행에서 CPU vision fallback이 기본처럼 동작할 수
있었다. 이러면 hybrid QNN projector-only 경로의 correctness 문제를 숨기기
때문에 기본값을 바꿨다.

현재 정책:

```text
기본:
  Qwen QNN hybrid path 사용
  mmproj offload enabled
  CPU vision fallback disabled

diagnostic:
  --qwen-cpu-vision-fallback 을 명시할 때만 CPU mtmd vision/mmproj 사용
```

관련 옵션:

```text
--qwen-cpu-vision-fallback
--no-qwen-cpu-vision-fallback
--mmproj-offload
```

Qwen 지원을 확인할 때는 `--no-qwen-cpu-vision-fallback --mmproj-offload`를
명시해서 CPU path가 섞이지 않도록 보는 것이 좋다.

## 6. 테스트

계약 테스트:

```bash
pytest -q /workspace/streamingvlm/my_research/foundation_llamacpp/tests/test_qwen_premerger_hybrid_contract.py
```

결과:

```text
3 passed
```

Python syntax check:

```bash
python3 -m py_compile \
  /workspace/streamingvlm/my_research/foundation_llamacpp/runner/cli.py \
  /workspace/streamingvlm/my_research/foundation/models/qwen2_5_vl/vision_encoder/model.py
```

Android QNN hybrid 실행은 exit 0이고 Golden Gate 이미지를 올바르게 인식했다.

## 7. 운영상 주의점

- Qwen2.5-VL + OpenCL CLIP에서는 FlashAttention을 다시 켜면 안 된다. window
  attention mask correctness를 OpenCL FA kernel에서 별도로 고치기 전까지는
  non-FA OpenCL attention path가 안전하다.
- QNN artifact가 pre-merger인지 post-merger인지 반드시 metadata로 확인해야
  한다. pre-merger artifact라면 llama.cpp 쪽에서 post_ln + patch merger를
  수행해야 한다.
- 448x448 Qwen2.5-VL pre-merger feature는 1024 tokens이고, patch merger 뒤
  decoder image token은 256 tokens다.
- CPU fallback은 원인 분리용으로만 사용한다. 결과가 맞더라도 QNN hybrid
  경로가 검증된 것은 아니다.
- F16 full OpenCL은 correctness guard 이후에도 메모리 압박이 크다. Q8 + F16
  KV-cache 조합이 현재 Android 검증 기준으로 더 현실적이다.
