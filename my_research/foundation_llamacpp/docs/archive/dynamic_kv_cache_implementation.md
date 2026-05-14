# Dynamic KV Cache 구현 설명

문서 목적: `llama.cpp` 기반 Android hybrid streaming 경로에 추가한 project-local dynamic KV cache prototype을 코드 단위에서 다시 추적할 수 있게 정리한다. 이 문서는 line number가 아니라 파일명과 심볼명 기준으로 읽는다. upstream `llama.cpp`가 바뀌면 line number는 쉽게 달라질 수 있다.

## 1. 최종 동작 모델

기존 fixed KV mode에서는 `--ctx-size 4096` 같은 값이 logical context 길이이면서 동시에 KV tensor의 physical allocation 크기였다. 따라서 실제로 아직 1024 token 정도만 사용해도, `ctx-size` 전체에 해당하는 K/V buffer가 시작 시점에 잡혔다.

현재 `main`의 active 구현은 standard llama.cpp KV cache를 대상으로 한
contiguous dynamic KV grow이다. Paged KV prototype은 실험 후 revert했으며,
현재 main에는 page-table attention path나 max-context backing allocation trick을
남기지 않는다. 이 문서는 contiguous dynamic KV와 OpenCL device-to-device
migration만 설명한다. OpenCL buffer가 SoC memory hierarchy에서 어디에 잡히는지
설명한 문서는 `dynamic_kv_opencl_buffer_memory_architecture.md`를 참고한다.

Dynamic KV prototype은 아래 플래그로 켠다.

```bash
--dynamic-kv-cache --kv-init-size 1024 --kv-grow-step 1024
```

동작은 다음과 같다.

1. `--dynamic-kv-cache`가 켜지면 runner는 `--ctx-size`를 직접 넘기지 않고 llama.cpp 쪽 logical context를 모델 최대 context로 둔다.
2. `--kv-init-size`는 최초 physical KV capacity가 된다.
3. prefill/decode 중 현재 batch가 physical KV capacity 안에 들어가지 않으면 standard KV cache를 grow한다.
4. grow 시점에는 기존 K/V tensor와 metadata를 보존하고, K/V tensor와 backend buffer를 새 capacity로 다시 만든 뒤 기존 byte range를 새 buffer로 옮긴다. OpenCL-backed KV에서는 `clEnqueueCopyBuffer`를 사용해 device-to-device로 복사하며, OpenCL fast path가 불가능할 때만 host tensor get/set으로 fallback한다.
5. backend scheduler reserve를 다시 수행하고, 실패했던 batch를 재시도한다.
6. grow event는 stdout log에 남고, Python finalizer가 이를 `foundation_proc.csv`의 `DynamicKVGrow` row로 backfill한다.

중요한 한계: 이 기능은 reserved KV memory를 줄이는 실험이다. 실제 accumulated KV length가 커질수록 attention work가 늘어나는 문제를 없애지는 않는다.

## 2. 실행 명령

일반 grow 검증 명령:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --single-buffer \
  --sampling-fps 1.0 \
  --max_video_time 15 \
  --time '[5.0, 8.0, 11.0, 14.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???", "What changed in the scene?", "Summarize the full situation so far."]' \
  --max-num 1 \
  --n-predict 64 \
  --dynamic-kv-cache \
  --kv-init-size 1024 \
  --kv-grow-step 1024 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log \
  --force-push
```

큰 memory jump를 보기 위한 one-shot grow 명령:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --single-buffer \
  --sampling-fps 1.0 \
  --max_video_time 15 \
  --time '[5.0, 8.0, 11.0, 14.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???", "What changed in the scene?", "Summarize the full situation so far."]' \
  --max-num 1 \
  --n-predict 64 \
  --dynamic-kv-cache \
  --kv-init-size 1024 \
  --kv-grow-step 15360 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 5.0 \
  --remote-root /data/local/tmp/streamingvlm_unified \
  --results-root my_research/foundation_llamacpp/results/log/dynamic_grow_16384 \
  --force-push
```

## 3. 변경 파일 요약

`llama.cpp` public/internal parameter plumbing:

- `llama.cpp/include/llama.h`: `llama_context_params`에 `kv_init_size`, `kv_grow_step`, `dynamic_kv_cache` 추가.
- `llama.cpp/src/llama-cparams.h`: internal `llama_cparams`에 같은 값을 추가.
- `llama.cpp/common/common.h`: CLI-facing `common_params`에 같은 값을 추가.
- `llama.cpp/common/arg.cpp`: `--dynamic-kv-cache`, `--kv-init-size`, `--kv-grow-step` 파싱 추가.
- `llama.cpp/common/common.cpp`: `common_params` 값을 `llama_context_params`로 전달.

`llama.cpp` runtime/grow implementation:

- `llama.cpp/src/llama-context.cpp`: context 생성 시 dynamic KV validation/padding/logging 추가, `decode()`의 memory slot 실패 경로에 grow-and-retry 추가.
- `llama.cpp/src/llama-model.cpp`: dynamic KV unsupported memory type guard 추가, standard `llama_kv_cache` 생성 시 initial physical capacity와 logical capacity를 분리해서 전달.
- `llama.cpp/src/llama-memory.h`: `llama_memory_i`에 grow capability interface 추가.
- `llama.cpp/src/llama-kv-cache.h`: `llama_kv_cache`에 logical/physical capacity 분리, cache type/offload 보존 멤버, grow/query/reset method 선언 추가.
- `llama.cpp/src/llama-kv-cache.cpp`: `reset_capacity()`, `grow_to()`, OpenCL device-to-device tensor byte copy fast path, fallback host tensor copy, physical/logical query 구현.
- `llama.cpp/src/llama-kv-cache-iswa.cpp`: iSWA wrapper의 `llama_kv_cache` constructor call에 fixed logical size 인자 추가.
- `llama.cpp/src/llama-memory-hybrid.cpp`: hybrid memory wrapper의 `llama_kv_cache` constructor call에 fixed logical size 인자 추가.

Project runner/bridge:

- `my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp`: Android streaming bridge에서 dynamic KV flags를 parse하고 llama.cpp runner argv로 forwarding.
- `my_research/foundation_llamacpp/runner/cli.py`: Python CLI flags, remote shell command generation, result folder suffix, `DynamicKVGrow` parsing, CSV/plot generation 추가.

Documentation/logs:

- `my_research/foundation_llamacpp/docs/README.md`: dynamic KV command와 output artifact 설명 추가.
- `my_research/foundation_llamacpp/docs/for_cursor_llm_llamacpp_version2.md`: 구현/검증 로그 추가.
- `my_research/foundation_llamacpp/docs/archive/streaming_single_buffer_implementation.md`: streaming 구현 문서에 dynamic KV validation section 추가.
- `my_research/foundation_llamacpp/docs/archive/dynamic_kv_opencl_buffer_memory_architecture.md`: OpenCL buffer allocation, device-to-device copy, memory bus 관점 설명 추가.
- `my_research/foundation/docs/for_cursor_llm.md`: workspace-level 누적 로그 추가.

## 4. llama.cpp parameter plumbing

### 4.1 Public context params

`llama.cpp/include/llama.h`의 `llama_context_params`에 아래 field를 추가했다.

- `kv_init_size`: dynamic KV mode에서 처음 할당할 physical KV cell 수. `0`이면 disabled/invalid.
- `kv_grow_step`: grow할 때마다 늘릴 physical KV cell step. `0`이면 disabled/invalid.
- `dynamic_kv_cache`: dynamic mode enable flag.

이 값은 public API 구조체에 들어가므로 downstream caller가 직접 `llama_context_default_params()`를 쓰더라도 default가 필요하다. 그래서 `llama.cpp/src/llama-context.cpp`의 `llama_context_default_params()`에도 기본값을 추가했다.

### 4.2 Common CLI params

`llama.cpp/common/common.h`의 `common_params`에도 같은 값을 추가했다. `llama.cpp/common/arg.cpp`는 아래 세 옵션을 처리한다.

- `--dynamic-kv-cache`
- `--kv-init-size N`
- `--kv-grow-step N`

`--dynamic-kv-cache`가 들어오면 `params.dynamic_kv_cache = true`로 설정하고, logical context는 model max를 쓰기 위해 `params.n_ctx = 0`로 둔다. 이 처리는 사용자가 fixed `--ctx-size`와 dynamic mode를 동시에 생각하지 않아도 되게 하기 위한 것이다.

`llama.cpp/common/common.cpp`의 `common_context_params_to_llama()`는 `common_params`의 세 값을 `llama_context_params`에 복사한다.

### 4.3 Internal cparams

`llama.cpp/src/llama-context.cpp` constructor는 `llama_context_params`에서 `llama_cparams`로 값을 복사한다.

`dynamic_kv_cache`가 켜져 있으면 validation을 수행한다.

- `n_seq_max == 1`이어야 한다.
- `kv_unified == false`이어야 한다.
- `kv_init_size`와 `kv_grow_step`은 0이 아니어야 한다.
- `kv_init_size`는 logical context보다 클 수 없고, 256 pad를 적용한다.
- `kv_grow_step`도 256 pad를 적용한다.

현재 prototype이 single non-unified sequence만 지원하는 이유는 K/V byte layout과 `v_cells` metadata 보존을 단순하게 유지하기 위해서다. streaming 실험 경로도 single chat sequence이므로 이 범위에서 먼저 검증한다.

## 5. logical context와 physical KV capacity 분리

핵심 변경은 `llama_kv_cache` constructor의 의미를 바꾼 것이다.

기존에는 `kv_size` 하나가 physical tensor shape이면서 logical context limit 역할을 했다. 변경 후에는 다음처럼 나뉜다.

- `kv_size`: 지금 실제로 할당하는 physical cell 수.
- `logical_kv_size`: dynamic mode에서 최대로 grow할 수 있는 logical cell 수.

`llama.cpp/src/llama-model.cpp`의 standard KV cache 생성 경로에서 dynamic mode이면:

- physical `kv_size` = `cparams.kv_init_size`
- logical `logical_kv_size` = `cparams.n_ctx_seq`

fixed mode이면:

- physical `kv_size` = `cparams.n_ctx_seq`
- logical `logical_kv_size` = `cparams.n_ctx_seq`

`llama.cpp/src/llama-kv-cache-iswa.cpp`와 `llama.cpp/src/llama-memory-hybrid.cpp`는 dynamic grow 대상이 아니므로 physical과 logical을 같은 값으로 넘긴다.

## 6. unsupported memory type guard

`llama.cpp/src/llama-model.cpp`에서 아래 memory type에는 dynamic KV를 금지했다.

- recurrent memory
- hybrid memory
- SWA/iSWA memory

`llama.cpp/src/llama-context.cpp`에서는 unified KV와 multi-sequence도 금지했다.

이 guard는 silent wrong behavior를 막기 위한 것이다. 특히 SWA/iSWA나 hybrid memory는 여러 KV object를 조합하거나 sliding/window semantics를 갖기 때문에 단순 `llama_kv_cache::reset_capacity()`로 안전하게 다룰 수 없다.

## 7. grow interface 확장

`llama.cpp/src/llama-memory.h`의 `llama_memory_i`에 아래 virtual method를 추가했다.

- `can_grow()`
- `grow_to(uint32_t new_size)`
- `get_physical_size()`
- `get_logical_size()`

기본 구현은 모두 unsupported/no-op이다. 이렇게 해서 `llama_context::decode()`는 memory object가 정확히 어떤 구현인지 downcast하지 않고 grow 가능 여부만 묻는다.

standard `llama_kv_cache`만 이 interface를 override한다.

## 8. `llama_kv_cache` 내부 변경

### 8.1 새 멤버

`llama.cpp/src/llama-kv-cache.h`에 다음 멤버를 추가했다.

- `logical_kv_size`: dynamic mode의 max logical capacity.
- `offload`: K/V buffer를 device backend로 offload할지 여부.
- `cache_type_k`, `cache_type_v`: reallocation 때 같은 dtype으로 K/V tensor를 다시 만들기 위한 저장값.

또한 constructor에서 받은 `filter`/`reuse` callback을 `filter_cb`/`reuse_cb` 멤버로 보존한다. grow 때 layer filtering/reuse 구조를 동일하게 재생성해야 하기 때문이다.

### 8.2 `reset_capacity()`

`llama.cpp/src/llama-kv-cache.cpp`의 `reset_capacity(uint32_t new_size, bool copy_existing)`가 실제 tensor/buffer 재생성을 담당한다.

동작 순서:

1. `copy_existing == true`이면 기존 `ctxs_bufs`, `layers`, `v_cells`, `v_heads`를 임시 변수로 move해서 old buffer lifetime을 유지한다.
2. 기존 live `ctxs_bufs`, `v_cells`, `layers`, `map_layer_ids` 등을 비운다.
3. 새 `kv_size` 기준으로 `v_heads`, `v_cells`, K/V tensors를 재생성한다.
4. backend buffer type은 기존 `offload` 값에 따라 CPU 또는 model layer device buffer를 사용한다.
5. K/V dtype은 기존 `cache_type_k`, `cache_type_v`를 사용한다.
6. backend buffer를 allocate하고 clear한다.
7. `copy_existing == true`이면 `v_cells` metadata를 grow하고, K/V tensor bytes를 old tensor에서 new tensor로 복사한다. OpenCL-backed tensor는 `clEnqueueCopyBuffer`로 device-to-device copy를 수행한다.

초기 constructor도 중복 초기화 코드를 들고 있지 않고 `reset_capacity(kv_size, false)`를 호출한다.

### 8.3 K/V data migration

grow는 device/OpenCL buffer를 새로 만들기 때문에 기존 K/V 내용을 보존해야 한다. 현재 구현은 아래 순서로 처리한다.

- `llama_kv_cache::copy_existing_data_from()`: old/new layer 목록을 비교하고 K/V byte range를 복사한다.
- `llama_kv_copy_tensor_bytes()`: OpenCL fast path를 먼저 시도하고, 실패하면 `ggml_backend_tensor_get()` / `ggml_backend_tensor_set()` fallback을 사용한다.
- `ggml_backend_opencl_tensor_copy_bytes()`: old/new tensor가 같은 OpenCL device buffer type이면 `clEnqueueCopyBuffer`로 GPU driver queue 안에서 직접 복사한다.
- `llama_kv_cells::grow_to()`: 기존 cell metadata를 유지하고 새 cell 영역만 empty state로 초기화한다.

K/V tensor layout은 K와 non-transposed V는 앞쪽 `old_size` rows가 contiguous하므로 한 번에 복사한다. transposed V는 embedding lane마다 stride가 `kv_size`에 의존하므로 lane 단위로 `old_size` elements를 old offset에서 new offset으로 옮긴다.

중요: 더 큰 OpenCL buffer를 "제자리에서 확장"하는 것은 OpenCL buffer API로는 불가능하므로 새 buffer allocation 자체는 여전히 필요하다. 다만 기존 방식처럼 CPU DRAM snapshot을 왕복하지 않고, OpenCL command queue의 buffer-to-buffer copy로 old KV를 새 KV에 복사한다.

### 8.4 `grow_to()`

`llama_kv_cache::grow_to(uint32_t new_size)`는 다음을 수행한다.

1. `new_size`를 `n_pad`로 pad하고 `logical_kv_size`를 넘지 않게 clamp한다.
2. 이미 충분히 크면 true를 반환한다.
3. single stream, non-SWA인지 다시 확인한다.
4. grow 시작 log를 남긴다.
5. `reset_capacity(new_size, true)`로 재할당/복원을 수행한다.
6. elapsed time을 ms로 log한다.

stdout 예:

```text
grow_to: growing dynamic KV cache: old = 1024, new = 16384, logical = 32768
reset_capacity:     OpenCL KV buffer size =   448.00 MiB
reset_capacity: size =  448.00 MiB ( 16384/ 32768 cells,  28 layers,  1/1 seqs), K (f16):  224.00 MiB, V (f16):  224.00 MiB
reset_capacity: dynamic KV data migration used device-to-device copy
grow_to: dynamic KV grow completed in 202.135 ms
```

## 9. grow trigger point

`llama.cpp/src/llama-context.cpp`의 `llama_context::decode()`에서 batch allocation/memory slot 준비가 실패하면 기존에는 warning을 찍고 실패했다.

Dynamic KV mode에서는 이 실패 지점에서 다음을 수행한다.

1. `memory->can_grow()` 확인.
2. old physical size와 logical max size 읽기.
3. `requested = min(logical_size, old_size + max(kv_grow_step, current_batch_tokens))` 계산.
4. `memory->grow_to(requested)` 호출.
5. grow가 성공하면 scheduler reserve가 다시 필요하므로 `sched_need_reserve = true`로 두고 `sched_reserve()` 호출.
6. `did_optimize = false`로 두고 loop를 `continue`해서 같은 batch를 재시도.

이 위치를 선택한 이유는 "실제로 더 큰 KV slot이 필요한 순간"에만 grow하도록 하기 위해서다. 사전에 token 수를 추정해서 grow하는 방식보다 변경 범위가 작고, existing failure path와 잘 맞는다.

## 10. Android bridge forwarding

`my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp`의 `Args`에 아래 값을 추가했다.

- `dynamic_kv_cache`
- `kv_init_size`
- `kv_grow_step`

argument parser는 아래 flags를 인식한다.

- `--dynamic-kv-cache`
- `--kv-init-size`
- `--kv-grow-step`

`build_decoder_argv()`는 dynamic mode가 아니면 기존처럼 `--ctx-size <ctx_size>`를 넘긴다. dynamic mode이면 `--ctx-size`를 넘기지 않고 아래만 넘긴다.

```text
--dynamic-kv-cache --kv-init-size <N> --kv-grow-step <N>
```

이렇게 한 이유는 dynamic mode에서 logical context를 llama.cpp/model max로 두기 위해서다.

## 11. Python runner 변경

### 11.1 CLI flags

`my_research/foundation_llamacpp/runner/cli.py`의 `main()`에 아래 argparse option을 추가했다.

- `--dynamic-kv-cache`
- `--kv-init-size`
- `--kv-grow-step`

기본값은 `kv_init_size=1024`, `kv_grow_step=1024`이다.

### 11.2 remote shell suffix

`_ctx_dynamic_kv_shell_suffix()`를 추가했다. 이 함수는 remote Android shell command를 만들 때 context 관련 인자를 결정한다.

- fixed mode: `-c <ctx_size>`
- dynamic mode: `--dynamic-kv-cache --kv-init-size <N> --kv-grow-step <N>`

`_build_hybrid_remote_script()`와 `_build_hybrid_streaming_remote_script()`가 이 suffix를 사용한다.

### 11.3 result folder suffix

`_result_model_name()`에 `dynamic_kv_cache` 인자를 추가했다. dynamic mode이면 result directory 끝에 `_dynamic`을 붙인다.

예:

```text
InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_kv16_dynamic
```

주의: folder name의 `ctx_32768`은 logical context 표시다. physical KV는 `foundation_proc.csv`의 `DynamicKVGrow` row나 stdout의 `reset_capacity` log를 봐야 한다.

## 12. `DynamicKVGrow` artifact 생성

### 12.1 stdout parsing

`runner/cli.py`에 `_dynamic_kv_rows_from_stdout()`를 추가했다. 이 함수는 `hybrid_streaming_stdout.txt` 또는 hybrid stdout에서 아래 log를 찾는다.

```text
grow_to: growing dynamic KV cache: old = <old>, new = <new>, logical = <logical>, clock_ms = <ggml_time_ms>
reset_capacity: size = <MiB> MiB
grow_to: dynamic KV grow completed in <ms> ms, clock_ms = <ggml_time_ms>
```

`hybrid_streaming_decode.cpp`는 `streaming_phase_stats.csv` header comment에
`clock_origin_ms`를 쓴다. 새 빌드에서 생성되는 grow log의 `clock_ms`는 같은
`ggml_time_ms()` clock이다. 따라서 `_dynamic_kv_rows_from_stdout()`는
`clock_origin_ms`가 있으면 grow start/end를 phase recorder와 같은 clock으로
변환한다.

기존 결과처럼 `clock_ms`가 없는 stdout은 fallback으로 주변 Android log
timestamp와 `load_tensors:` timestamp를 사용한다. 이 경우 phase timer와 clock
source가 달라서 retry 경계가 약간 어긋나 보일 수 있다.

### 12.2 retry 기준 prefill 분리

Dynamic KV grow는 llama.cpp 내부에서 prefill batch를 넣으려다가 physical KV
capacity가 부족할 때 발생한다. 성공한 prefill은 grow 이후 scheduler reserve를
다시 수행한 뒤 같은 batch를 retry하면서 진행된다.

그래서 finalizer는 `DynamicKVGrow`와 겹치는 aggregate `Prefill` row를 grow
구간과 겹치지 않게 split한다. fine-grained `ImagePrefill`/`T_Prefill`/`Mmproj`
row가 grow window와 걸치면, 새 clock 기반 결과에서는 retry 가능한 grow end
시각부터 시작하도록 clip한다. 완전히 grow window 안에 찍힌 legacy row는 failed
attempt 계측일 수 있으므로 해석에 주의해야 한다.

### 12.3 CSV row schema

`_write_phase_csv()`의 header 설명에 `DynamicKVGrow`를 추가했다.

`DynamicKVGrow` row는 아래처럼 해석한다.

- `row_type`: `DynamicKVGrow`
- `elapsed_s_start`: grow 시작 추정 시각
- `elapsed_s_end`: grow 완료 시각
- `col_a_ms`, `total_ms`: grow latency
- `kv_pos`: old physical cell count
- `kv_total`: new physical cell count
- `kv_estimated_used_kb`: old physical KV MiB를 KiB로 변환한 값
- `kv_total_kb`: new physical KV MiB를 KiB로 변환한 값
- `kv_physical_committed_kb`: new committed physical KV size
- `token_idx`: 사람이 읽기 쉬운 detail string

예:

```text
DynamicKVGrow,41.154632,41.427575,,,273,,273,1024,16384,,28672,458752,458752,1024->16384/32768 cells; 28.00->448.00 MiB
```

## 13. plot 변경

`runner/cli.py`의 `_phase_colors()`에 `DynamicKVGrow` 색을 추가했다.

`_write_png_streaming_phase_timeline()`의 visible phase list에도 `DynamicKVGrow`를 넣었다. 이 plot에서는 grow duration을 `KV +<ms>ms` 형태로 표시한다.

최신 구현에서는 `DynamicKVGrow`가 단순 `llama_kv_cache::grow_to()` 시간만
뜻하지 않는다. `llama_context::decode()`가 `memory->grow_to()` 성공 후
`sched_reserve()`를 완료한 시점까지 `dynamic KV grow retry window`로 기록하고,
finalizer는 이 full window를 검정색 grow bar로 사용한다. 따라서 검정색에는
slot 부족 후 KV grow, 기존 KV copy/restore, backend scheduler reserve, retry
준비 비용이 포함된다. 파란색 `ImagePrefill`은 이 window 이후부터 시작하도록
clip되어 retry-side 실제 image prefill 계산에 가깝게 표시된다.

`_write_png_memory_timeline_decode_window()`를 새로 추가했다. 이 plot은 기존 `memory_timeline_plot.png`와 별개로 생성된다.

범위:

- 시작: 첫 `V_Encode`의 `elapsed_s_start`
- 끝: 마지막 `D` 또는 `Decode`의 `elapsed_s_end`

표시:

- `MemAvailable (MiB)`
- 가능하면 `KgslShmemUsage (MiB)`
- `V_Encode`, `ImagePrefill`, `T_Prefill`, `Mmproj`, `DynamicKVGrow` phase span
- phase legend
- `DynamicKVGrow` label에는 `1024->16384/32768 cells; 28.00->448.00 MiB` 같은 detail 표시

파일명:

```text
memory_timeline_decode_window.png
```

## 14. validation 결과

### 14.1 1024-step grow

결과 폴더:

```text
my_research/foundation_llamacpp/results/log/InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_kv16_dynamic
```

조건:

- model: `InternVL3-2B-Instruct-Q8_0.gguf`
- processor: `hybrid`
- dynamic flags: `--dynamic-kv-cache --kv-init-size 1024 --kv-grow-step 1024`
- prompts: 4 turns
- output: `foundation_exit_code.txt = 0`

관찰:

```text
DynamicKVGrow,38.670695,38.748724,,,78,,78,1024,2048,,28672,57344,57344,1024->2048/32768 cells; 28.00->56.00 MiB
```

즉:

- `1024 -> 2048 cells`
- `28.00 -> 56.00 MiB`
- grow latency: 약 `78 ms`

### 14.2 1024 to 16384 one-shot grow

결과 폴더:

```text
my_research/foundation_llamacpp/results/log/dynamic_kv_device_copy_2b_hybrid_retry/InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_kv16_dynamic
```

조건:

- dynamic flags: `--dynamic-kv-cache --kv-init-size 1024 --kv-grow-step 15360`
- prompts: 4 turns
- output: `foundation_exit_code.txt = 0`

관찰:

```text
DynamicKVGrow,30.427000,30.726000,,,299,,299,1024,16384,,28672,458752,458752,1024->16384/32768 cells; 28.00->448.00 MiB
```

stdout:

```text
grow_to: growing dynamic KV cache: old = 1024, new = 16384, logical = 32768
reset_capacity:     OpenCL KV buffer size =   448.00 MiB
reset_capacity: size =  448.00 MiB ( 16384/ 32768 cells,  28 layers,  1/1 seqs), K (f16):  224.00 MiB, V (f16):  224.00 MiB
reset_capacity: dynamic KV data migration used device-to-device copy
grow_to: dynamic KV grow completed in 202.135 ms
```

즉:

- `1024 -> 16384 cells`
- `28.00 -> 448.00 MiB`
- 증가량: `+420 MiB`
- grow latency: 약 `202 ms` (`llama_kv_cache::grow_to()` internal time), full grow/retry window은 runner finalizer의 `DynamicKVGrow` row 기준으로 해석한다.

`android_memory_timeline.csv`에서도 grow 직후 `MemAvailable` 하락 구간이 더 명확하게 보인다. 다만 Android 전체 memory metric은 allocator/cache/driver 영향이 섞이므로, 정확한 KV allocation 크기는 `foundation_proc.csv`의 `kv_physical_committed_kb`와 stdout의 `reset_capacity` log를 기준으로 해석하는 것이 좋다.

### 14.3 five-prompt full grow/retry window validation

목적: grow가 발생한 prompt 이후에도 graph/scheduler caching이 계속 느려지는지,
또는 grow prompt에서만 일회성 spike가 생기는지 분리한다.

비교 조건:

- no-grow baseline: `--dynamic-kv-cache --kv-init-size 16384 --kv-grow-step 15360`
- grow run: `--dynamic-kv-cache --kv-init-size 1024 --kv-grow-step 15360`
- prompts: 5 turns at `5s/8s/11s/14s/17s`
- video: `surveil_8_20sec.mp4`

결과:

```text
init-16384 P4:
  ImagePrefill = 2144 ms
  Prefill      = 2647 ms
  Decode avg   = 57.0 ms/token

grow full-window P4:
  DynamicKVGrow = 394 ms
  ImagePrefill  = 2215 ms
  Prefill       = 2480 ms after grow + 223 ms before grow
  Decode avg    = 57.0 ms/token

init-16384 P5:
  ImagePrefill = 2851 ms
  Prefill      = 3430 ms
  Decode avg   = 60.8 ms/token

grow full-window P5:
  ImagePrefill = 2886 ms
  Prefill      = 3457 ms
  Decode avg   = 60.9 ms/token
```

해석:

- P4의 one-time grow/retry 비용은 `DynamicKVGrow=394 ms`로 분리된다.
- P4의 retry-side `ImagePrefill=2215 ms`는 init-16384의 `2144 ms`와 가까워졌다.
- P5는 no-grow와 grow run이 거의 동일하다. 따라서 dynamic grow가 이후 prompt의
  graph/scheduler cache를 지속적으로 망가뜨리는 현상은 관찰되지 않았다.

## 15. 성능 해석

Dynamic KV가 줄이는 것은 "처음부터 예약되는 KV buffer capacity"이다. 예를 들어 2B Q8 f16 KV 기준:

- `1024 cells`: 약 `28 MiB`
- `2048 cells`: 약 `56 MiB`
- `4096 cells`: 약 `112 MiB`
- `16384 cells`: 약 `448 MiB`

하지만 token decode latency는 physical capacity가 아니라 actual used KV length에 더 직접적으로 영향을 받는다. attention kernel은 현재 token이 attend해야 하는 accumulated past K/V 범위를 처리하기 때문이다.

따라서 dynamic KV는:

- early stage memory reservation을 줄이는 데 유효하다.
- grow 시점에 reallocation/copy/scheduler reserve latency spike가 생긴다.
- long context에서 per-token attention cost 증가를 없애지는 않는다.

## 16. 구현상 주의점

- 이 변경은 upstream `llama.cpp` source를 직접 수정한다. 프로젝트 원칙상 가능한 한 patch로 유지하고, upstream update 시 conflict를 작게 관리해야 한다.
- `llama_kv_cache::reset_capacity()`는 device buffer를 재생성한다. OpenCL backend에서는 grow latency가 새 buffer allocation과 device-to-device copy를 포함한다. OpenCL tensor 조건이 맞지 않으면 host fallback copy가 사용될 수 있다.
- K/V migration은 tensor byte layout에 의존한다. upstream에서 KV tensor shape, transposed V layout, stream view 생성 방식이 바뀌면 `copy_existing_data_from()`을 재검증해야 한다.
- 현재 parser는 stdout log format에 의존한다. `grow_to:` 또는 `reset_capacity:` log string이 바뀌면 `_dynamic_kv_rows_from_stdout()`도 같이 수정해야 한다.
- current implementation은 single stream/single sequence 중심이다. multi-session, beam/parallel sequence, unified KV, SWA/iSWA로 확장하려면 memory object별 semantics를 다시 설계해야 한다.
- 결과 폴더명 `_dynamic`은 dynamic mode 여부만 표시한다. `kv-grow-step` 값까지 폴더명에 들어가지는 않으므로, 비교 실험은 `--results-root`를 다르게 주는 것이 안전하다.

## 17. 앞으로 확장한다면

가능한 후속 작업:

1. grow threshold를 "prepare failure 후"가 아니라 "다음 prompt prefill 예상 token 수 기준"으로 사전 grow하도록 바꾸기.
2. `foundation_proc.csv`에 grow 전후 Android memory sample을 가까운 timestamp 기준으로 같이 기록하기.
3. KV capacity step을 adaptive policy로 바꾸기. 예: 작은 turn에서는 1024, long prefill 직전에는 4096 이상.
4. OpenCL buffer reallocation latency를 줄이기 위해 async copy/event chaining, copy granularity, grow 직전 사전 reserve 가능성 검토.
5. dynamic KV를 CPU fixed path에서도 검증하고, OpenCL/hybrid 외 backend로 확대.

## 18. Main Branch Closure State

현재 main에는 아래 상태로 정리되어 있다.

```text
active:
  contiguous dynamic KV grow
  OpenCL device-to-device K/V migration
  DynamicKVGrow timeline/finalizer instrumentation
  sliding-window multi-turn streaming

reverted / not active:
  paged KV cache prototype
  page-table OpenCL attention path
```

마지막 main 정리에서는 원격 main에 들어가 있던 paged-KV prototype 커밋 두 개를
revert한 뒤 dynamic-KV/device-copy streaming 작업만 merge했다. 따라서 이후
문서나 실험에서 "dynamic KV"라고 부르는 것은 이 문서의 contiguous grow 방식을
의미한다.
