# OpenCL 양자 KV (`Q8_0` / `Q4_0`) + Flash Attention — 구현·재현 가이드 (StreamingVLM)

**문서 목적:** 안드로이드 **OpenCL**에서 `llama.cpp`의 `--cache-type-k` / `--cache-type-v` 로 **양자 KV**를 쓸 때 필요한 **코드 변경**, **스케줄러 이슈**, **실행 규칙**, **처음부터 재구성 절차**를 한곳에 모은다.

**정본(canonical) 코드:** `streamingvlm` 루트의 **`llama.cpp/`** (Git에 포함; **`llama.cpp/models/`** 는 `.gitignore`). 예전 `foundation_llamacpp/kv_code/` 미러는 **삭제됨**.

**라인 번호**는 업스트림 머지마다 흔들리므로 문서에서는 **심볼/파일 검색**을 기준으로 한다.

**최종 정리 시점 (워크스페이스 기준):**  
- `GGML_OP_SET_ROWS` → Q8_0 / Q4_0 OpenCL 커널 + `supports_op`  
- `GGML_OP_FLASH_ATTN_EXT` → 양자 KV **dequant 임시 버퍼** + **`supports_op` 에서 양자 KV 허용**(스케줄러가 CPU로 FA를 보내지 않도록)  
- `ggml_flash_attn_ext` 의 **dst 타입은 항상 F32** → OpenCL `supports_op` 에 **F16 Q + 논리적 F16 K/V + F32 dst** 패턴 필요  
- 결과 폴더 슬러그: **f16 KV → `_kv16`**, `q8_0` → **`_kv8`**  
- 스윕 스크립트: **`my_research/foundation_llamacpp/results/log/` 바로 아래**에 런 디렉터리만 생성 (중간 `log_*_sweep_*` 부모 폴더 없음)  
- 표준 실험 플래그(사용자 규칙): **`--flash-attn on`, `--fit on`, `--warmup`**

---

## 목차

1. [문제 배경](#1-문제-배경)
2. [업스트림과 StreamingVLM 오버레이의 관계](#2-업스트림과-streamingvlm-오버레이의-관계)
3. [패치가 건드리는 파일 요약](#3-패치가-건드리는-파일-요약)
4. [`SET_ROWS` 의미 · OpenCL 디스패치](#4-set_rows-의미--opencl-디스패치)
5. [`set_rows.cl` (Q8 / Q4)](#5-set_rowscl-q8--q4)
6. [`ggml_cl_set_rows` 호스트 측](#6-ggml_cl_set_rows-호스트-측)
7. [`ggml_opencl_supports_op`](#7-ggml_opencl_supports_op)
   - 7.4 [필수: `GGML_OP_FLASH_ATTN_EXT` + 양자 KV](#74-필수-ggml_op_flash_attn_ext--양자-kv)
8. [Flash Attention + 양자 KV: 준비 함수](#8-flash-attention--양자-kv-준비-함수)
9. [CPU 레퍼런스 대응 (Q4)](#9-cpu-레퍼런스-대응-q4)
10. [CMake / 커널 임베드](#10-cmake--커널-임베드)
11. [빌드·디바이스 실행·로그 규칙](#11-빌드디바이스-실행로그-규칙)
12. [foundation_llamacpp 브리지 (`opencl_phase_mtmd`)](#12-foundation_llamacpp-브리지-opencl_phase_mtmd)
13. [증상 → 원인](#13-증상--원인)
14. [한 블록 단위 검증](#14-한-블록-단위-검증)
15. [업스트림 리베이스 체크리스트](#15-업스트림-리베이스-체크리스트)
16. [Greenfield: 처음부터 재구성](#16-greenfield-처음부터-재구성)
17. [부록: 내부 노트](#17-부록-내부-노트)

---

## 1. 문제 배경

1. **양자 KV**를 쓰면 그래프에서 **F32 행**과 **KV 버퍼 뷰** 사이에 **`GGML_OP_SET_ROWS`** (scatter + quantize)가 들어간다. OpenCL에서는 **GPU `cl_mem`** 상에서 동작해야 한다.
2. OpenCL이 `SET_ROWS` 대상 **`dst` 타입으로 Q8_0/Q4_0을 허용하지 않거나** 커널이 없으면 **예약(`sched_reserve`) 또는 실행 단계**에서 실패한다.
3. **`--flash-attn on`** 이면 **`GGML_OP_FLASH_ATTN_EXT`** 가 KV를 읽는다. **저장은 q8_0**이어도 FA 커널은 **연속 F16/F32 K·V**를 기대하므로, OpenCL 측에서 **GPU → 호스트 dequant → 임시 선형 버퍼 → 다시 GPU** 같은 **준비 경로**가 필요할 수 있다.
4. **치명적 버그(2026-05 정리):** `ggml_opencl_supports_op` 가 **양자 K/V인 FA**를 “미지원”으로 두면, 스케줄러가 FA를 **CPU**에 붙이고 K/V 텐서는 **OpenCL 버퍼**에 남긴다. CPU FA는 **호스트 포인터**를 기대하므로 **`common_init_from_params` 빈 워밍업 단계에서 SIGSEGV**가 난다. → **반드시 OpenCL이 양자 KV FA를 지원한다고 선언**하고, 실행 시 **prepare 경로**로 맞춘다.
5. **`ggml_permute` 등으로 K/V가 `ne[0]` 방향 비연속**이면, GPU 바이트 덤프를 통째로 `to_float` 하면 **행이 어긋나** 디코드가 **무의미 반복 토큰**으로 붕괴한다 → **행 단위 pack** 후 dequant.

---

## 2. 업스트림과 StreamingVLM 오버레이의 관계

- **참고 Draft PR:** [llama.cpp#21313](https://github.com/ggml-org/llama.cpp/pull/21313) (OpenCL FA·양자 KV 관련 아이디어; 본 워크스페이스는 **전부를 그대로 머지하지 않을 수 있음**).
- **`ggml-org/llama.cpp` `master`** 를 subtree 등으로 가져온 뒤, **StreamingVLM에서 필요한 최소 diff**만 `llama.cpp/ggml/src/ggml-opencl/` 등에 유지한다.
- **대형 PR의 kv_pad / split / `flash_attn_pre_f16.cl`** 류는 **정확성에 필수는 아님**(최소 패치는 supports + prepare + SET_ROWS). **성능** 개선용으로만 선택 이식.

---

## 3. 패치가 건드리는 파일 요약

| 구분 | 경로 | 내용 |
|------|------|------|
| SET_ROWS 커널 | `llama.cpp/ggml/src/ggml-opencl/kernels/set_rows.cl` | F32 행 → `block_q8_0` / `block_q4_0` GPU quant |
| OpenCL 디스패치 | `llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp` | `supports_op`(SET_ROWS, **FLASH_ATTN_EXT**), `ggml_cl_set_rows`, **`ggml_cl_flash_attn`** + **`ggml_cl_flash_attn_prepare_quantized_tensor`**, `clCreateKernel` |
| 커널 목록 | `llama.cpp/ggml/src/ggml-opencl/CMakeLists.txt` | `set_rows` 가 `GGML_OPENCL_KERNELS` 에 포함 |

**foundation_llacampp 쪽 (브리지·실행)**

| 구분 | 경로 |
|------|------|
| Android 브리지 CMake | `my_research/foundation_llamacpp/hybrid_bridge/CMakeLists.txt` |
| 위상·페이즈 도구 | `my_research/foundation_llamacpp/hybrid_bridge/opencl_phase_mtmd.cpp` |
| 단일 디바이스 실행 | `my_research/foundation_llamacpp/run_android_hybrid_bridge.py` |
| ctx 스윕 | `my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh` |

---

## 4. `SET_ROWS` 의미 · OpenCL 디스패치

**선언:** `ggml_set_rows(ctx, a, b, c)` (`ggml.c` 검색).

- 결과 텐서는 **`a`의 뷰**; `src[0]=b`(F32 행들), `src[1]=c`(인덱스 I32/I64), `src[2]=a`(대상 KV 뷰, 레거시 순서 주의).

**OpenCL** `ggml_cl_set_rows(backend, src0, src1, dst)`:

- **`dst->type`** 이 Q8_0/Q4_0/F16/F32 에 따라 다른 커널.

---

## 5. `set_rows.cl` (Q8 / Q4)

**경로:** `llama.cpp/ggml/src/ggml-opencl/kernels/set_rows.cl`

- **Q8_0:** `block_q8_0`, `kernel_set_rows_quantize_block_q8_0`, `kernel_set_rows_q8_0_i32` / `_i64`
- **Q4_0:** `QK4_0_KV` 등으로 다른 커널과 매크로 충돌 방지; **`kernel_set_rows_i32_as_int8_truncate`**, **`kernel_set_rows_q4_packed_nibble_ref`** 로 CPU `quantize_row_q4_0_ref` 와 같은 **truncate + `(int8_t)(x+8.5f)` + `MIN(15, …)`** 니블 적재
- **워크그룹:** 한 `ne01` 행을 여러 WI가 **quant 블록** 단위로 나눔 (`nblk0` 루프)

(세부 줄 단위 테이블은 이전판과 동일하게 커널 파일을 따라가며 확인.)

---

## 6. `ggml_cl_set_rows` 호스트 측

**검색:** `static void ggml_cl_set_rows`

- 차원/`nb`/블록 수 `nblk0 = ne0 / ggml_blck_size(dst_type)` 설정 후 `clSetKernelArg`, global/local NDRange.

---

## 7. `ggml_opencl_supports_op`

### 7.1 `GGML_OP_SET_ROWS`

**검색:** `case GGML_OP_SET_ROWS:`

- `src[0]` F32, `src[1]` I32/I64, **`op->type`(dst)** 가 F16/F32/**Q8_0**/ **Q4_0** 일 때 true (프로젝트 패치 기준).

### 7.2 `GET_ROWS` + SoA

**`GGML_OP_GET_ROWS`** 에서 **`Q4_0` + `GGML_OPENCL_SOA_Q`** 등은 **가중치 SoA** 경로와 혼동되면 안 됨. KV **`SET_ROWS`** 와 문제 성격이 다르다.

### 7.3 커널 로드

**검색:** `kernel_set_rows_q8_0_i64`, `program_set_rows` — Q8/Q4 커널 핸들을 `clCreateKernel` 로 등록.

### 7.4 필수: `GGML_OP_FLASH_ATTN_EXT` + 양자 KV

**검색:** `case GGML_OP_FLASH_ATTN_EXT:` 안의 `k_logical_type` / `is_f16_f16_out_f32`.

- **`ggml_flash_attn_ext()`** 로 만든 노드의 **결과 타입은 항상 F32** (`ggml_new_tensor(..., GGML_TYPE_F32, ...)`).
- 예전 OpenCL 매칭이 **`q`(F16) + `k`/`v`(Q8 저장) + `op`(F32)** 인 경우를 **전부 거부**했고, 결과적으로 **[증상]** FA가 CPU로 떨어짐 → **SIGSEGV**.
- **수정 요지:**
  - K/V 가 **양자**이면 **같은 양자형** 등 제약 검사.
  - **논리 타입:** `ggml_is_quantized(k)` 이면 `q` 가 F16이면 **논리 K/V는 F16**, 아니면 F32.
  - **허용 조합 예:**  
    - `q`/`k`/`v` 저장형이 허용된 F32/F16 조합  
    - **추가:** **`q` F16 + 논리 K/V F16 + **`op->type == F32`** (`is_f16_f16_out_f32`)** ← llama-graph 기본 FA 노드 패턴과 맞춤.

이와 짝으로 **`ggml_cl_flash_attn`** 에서 **`k_logical_type`** 으로 mixed/f16 커널 선택을 맞추고, **`ggml_cl_flash_attn_prepare_quantized_tensor`** 로 실제 디바이스/stride를 교체한다.

---

## 8. Flash Attention + 양자 KV: 준비 함수

**구조체:** `ggml_cl_flash_attn_temp_buffer` (임시 `cl_mem`, 소멸자에서 `clReleaseMemObject`)

**함수:** `ggml_cl_flash_attn_prepare_quantized_tensor(...)` (**검색으로 위치 확인**)

- **비양자:** 즉시 `false` 반환 → **포인터/stride 변경 없음**
- **양자:**
  1. `sync_with_other_backends` 후 `clEnqueueReadBuffer` 전체 `ggml_nbytes`
  2. **`!ggml_is_contiguous_0(tensor)`** 이면 `(i1,i2,i3)` 이중 루프로 **`row_off = i1*nb1+i2*nb2+i3*nb3`**, **`ggml_row_size(type, ne[0])`** 바이트만 `memcpy` 해 **dense 양자 슬랩** 구축
  3. `to_float` → `kv_target_type`(Q가 F16이면 F16, 아니면 F32)으로 선형 버퍼
  4. `clCreateBuffer` + `clEnqueueWriteBuffer`; **`data_device`/`offset`/linear `nb1..3`** 갱신
  5. 성능 경고 1회: dequant 및 임시 버퍼

**호출부:** **`ggml_cl_flash_attn`** 내부에서 **커널 선택 이후**, `clSetKernelArg` 전에 **`k`/`v` 각각 호출**(항상 호출되며, 비양자면 no-op).

**참고:** 업스트림 full PR의 **KV pad preload 커널** 등은 **본 최소 스택에는 없음**—stride만 임시 선형 버퍼에 맞추면 현재 단일 FA 커널 경로로 동작 확인됨.

---

## 9. CPU 레퍼런스 대응 (Q4)

**CPU:** `ggml-quants.c` **`quantize_row_q4_0_ref`**

OpenCL **`kernel_set_rows_quantize_block_q4_0`** + 니블/트렁케이트 헬퍼와 **바이트 단위** 비교로 검증.

---

## 10. CMake / 커널 임베드

**파일:** `llama.cpp/ggml/src/ggml-opencl/CMakeLists.txt`

- **`set_rows`** 가 **`GGML_OPENCL_KERNELS`** 리스트에 있어야 함.
- 빌드 시 `embed_kernel.py` → `set_rows.cl.h` 등 자동 생성.

---

## 11. 빌드·디바이스 실행·로그 규칙

### 11.1 Android OpenCL 브리지 빌드

- **CMake 소스 트리:** `my_research/foundation_llamacpp/hybrid_bridge` (여기서 `add_subdirectory(llama.cpp)`).
- 툴체인 예: Android NDK, `ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM` 적절히, **`GGML_OPENCL=ON`**.

```bash
cmake -S my_research/foundation_llamacpp/hybrid_bridge -B build-hybrid-android-opencl \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
cmake --build build-hybrid-android-opencl -j"$(nproc)"
```

중요 산출물: `libggml-opencl.so`, **`opencl_phase_mtmd`**, `hybrid_decode`, … (`build-hybrid-android-opencl` 또는 `bin/` 아래 레이아웃은 빌드 설정에 따름).

### 11.2 단발 실행 (`run_android_hybrid_bridge.py`)

**권장 플래그 (프로덕션 실험):**

- `--flash-attn on`
- `--fit on`
- `--warmup`

예 (GPU / InternVL):

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor gpu \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/.../InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/.../mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --ctx-size 8192 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --fit on --warmup
```

양자 KV:

```bash
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --fit on --warmup
```

### 11.3 ctx 스윕 (`run_opencl_ctx_sweep.sh`)

- **기본 ctx:** `512 1024 2048 4096 8192 16384 32768`
- **`--results-root` 는 항상**  
  **`${REPO}/my_research/foundation_llamacpp/results/log`**  
  에 고정되어 있음 (**중간 `log_*_sweep_*` 부모 디렉터리 사용 안 함**).
- **`FIT` 미지정**이면: **양자 KV**일 때만 스크립트가 기본 **`FIT=off`** 를 넣음(과거 SET_ROWS+fit 회피). **사용자 규칙이 `fit on` 이면** 반드시 **`FIT=on`** 을 명시.

```bash
FIT=on FLASH_ATTN=on WARMUP=1 PROCESSOR=gpu \
  CACHE_TYPE_K=f16 CACHE_TYPE_V=f16 \
  ./my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh

FIT=on FLASH_ATTN=on WARMUP=1 PROCESSOR=gpu \
  CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 \
  ./my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh
```

### 11.4 결과 디렉터리 이름 (`--results-root` 직속)

`_result_model_name` (`run_android_hybrid_bridge.py` 검색):

- 형식: **`<모델스템>_opencl_ctx_<N>_kv<슬러그>`**
- **`_result_kv_slug_part`:** `q8_0`→`8`, `q4_0`→`4`, **`f16`/`fp16`→`16`** ⇒ 접미사 **`_kv16`**, **`_kv8`** (예전 `_kvf16` 명명 폐기)
- 사용자가 추가로 **`results/log/InternVL3-1B_kv16/`** 처럼 그룹 폴더를 **수동으로** 만들 수 있으나, **스크립트 기본은 `results/log/<한 단계만>`**.

### 11.5 로그에서 확인할 것

- **`llama_context: flash_attn = enabled`**
- **양자 KV + FA:** (최대 1회) `ggml_cl_flash_attn_prepare_quantized_tensor: ... dequantizes ...`
- **`sched_reserve: graph splits`** — 양자 FA가 OpenCL로 붙으면 **split 수가 과도하게 크지 않은 편**(이전 버그에서는 CPU 분배로 폭증)
- **`foundation_exit_code.txt` → 0**

---

## 12. foundation_llamacpp 브리지 (`opencl_phase_mtmd`)

`llama.cpp` **업스트림**이 `common` 라이브러리 타깃 이름을 **`llama-common`** 으로 바꾼 뒤:

- **`hybrid_bridge/CMakeLists.txt`:**  
  `target_link_libraries(opencl_phase_mtmd PRIVATE llama-common mtmd Threads::Threads)`  
  (`common` 타깃명 사용 불가).
- **`opencl_phase_mtmd.cpp`:** `base_callback_data` 제거 등 API 변화 대응  
  → **`std::optional<common_debug_cb_user_data> mtmd_debug_graph_cb`**  
  → 디버그 시 `emplace()`, `mparams.cb_eval_user_data = &*mtmd_debug_graph_cb`,  
    `mparams.cb_eval = common_debug_cb_eval` (`mtmd-cli` 패턴 정렬).  
  **`#include "debug.h"`** 필요.

브리지는 **ExecuTorch를 직접 수정하지 않는** 레이아웃 규칙과 일치하게 `hybrid_bridge/` 아래에만 둔다.

---

## 13. 증상 → 원인

| 증상 | 우선 확인 |
|------|-----------|
| `sched_reserve` / `SET_ROWS` 실패 | `supports_op`(SET_ROWS), `set_rows.cl` 커널 등록 |
| 빈 워밍업 **SIGSEGV** (exit 139) | **`FLASH_ATTN_EXT` 의 `supports_op` 가 양자 KV를 거절** → CPU FA + GPU 버퍼 |
| 가독한 영어 대신 문자 반복·깨짐 | **`prepare_quantized_tensor` 행 pack** 여부 (`ggml_is_contiguous_0`) |
| Q4만 깨짐 | `quantize_row_q4_0_ref` vs OpenCL 니블/`d` |
| `_kv16` 이름이 안 맞음 | `_result_kv_slug_part` 에서 **`f16`→`16`** |

---

## 14. 한 블록 단위 검증

1. 동일 **32원소 FP32 블록**을 CPU quant 와 GPU `SET_ROWS` 한 행 결과로 비교.
2. 필요 시 해당 행만 `clEnqueueReadBuffer` 로 읽어 **_hex dump**.

---

## 15. 업스트림 리베이스 체크리스트

1. **`ggml-opencl.cpp`** 충돌: `supports_op`(SET_ROWS + **FLASH_ATTN_EXT**), `ggml_cl_set_rows`, **`ggml_cl_flash_attn`**, **prepare**.
2. **`set_rows.cl`** vs `quantize_row_*_ref`.
3. 커널 리스트 CMake.
4. **Android 재빌드** 후 **kv16 / kv8** 각각 최소 한 ctx에서 **`--flash-attn on --fit on --warmup`** 스모크.

---

## 16. Greenfield: 처음부터 재구성

1. **저장소:** `streamingvlm` 클론; **`llama.cpp/models/`** 는 Git에 없음 — GGUF/mmproj는 로컬에 둠.
2. **업스트림 싱크:** (정책에 따라) subtree `llama.cpp` 또는 수동 카피 후 **패치 재적용**.
3. **`llama.cpp/ggml/src/ggml-opencl/` 에 적용:**
   - `kernels/set_rows.cl` (Q8/Q4 브랜치)
   - **`ggml-opencl.cpp`** (SET_ROWS + FA prepare + **`FLASH_ATTN_EXT supports_op`**)
   - `CMakeLists.txt` 커널 목록.
4. **NDK 빌드** (11절): `hybrid_bridge` 소스 디렉터리, **`GGML_OPENCL=ON`**.
5. **ADB** 디바이스 연결 확인 후 **`run_android_hybrid_bridge.py`** 단발 → **`foundation_exit_code.txt` 0**.
6. **스윕** (선택): `FIT=on FLASH_ATTN=on WARMUP=1` + `CACHE_TYPE_*` 로 **전 ctx 구간**.
7. **플롯:** `scripts/plot_opencl_ctx_memory_series.py` — `--parent results/log`; **`ctx_from_dir`** 는 `_ctx_(\d+)` 라 **`_kv16` 접미사와 무관하게 ctx 추출 가능**.

---

## 17. 부록: 내부 노트

- **`my_research/foundation_llamacpp/docs/for_cursor_llm_llamacpp.md`** — 일자별 이슈·커맨드 누적.
- **`my_research/foundation/docs/for_cursor_llm.md`** — foundation 레이어와 교차 참고.

---

*심벌 검색 키워드:* `GGML_OP_SET_ROWS`, `GGML_OP_FLASH_ATTN_EXT`, `ggml_cl_set_rows`, `ggml_cl_flash_attn_prepare_quantized_tensor`, `ggml_cl_flash_attn`, `kernel_set_rows_q4_0_i32`, `quantize_row_q4_0_ref`, `_result_kv_slug_part`.
