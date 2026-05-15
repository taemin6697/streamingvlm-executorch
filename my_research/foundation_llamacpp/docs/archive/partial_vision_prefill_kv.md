# Partial Vision Prefill KV 구현 설명

문서 목적: `--stream-mode vision-prefill` 위에 추가한
`--partial-vision-kv` 구현을 코드 단위에서 다시 추적할 수 있게 정리한다.
이 문서는 line number 대신 파일명과 심볼명 기준으로 읽는다. upstream
`llama.cpp`나 bridge 코드가 바뀌면 line number는 쉽게 달라질 수 있다.

## 1. 문제 정의

기존 `vision-prefill`은 프레임이 들어올 때마다 현재 streaming user turn의
visual prefix KV를 미리 만든다. 프롬프트가 들어오면 가장 최신 snapshot을
restore하고, 질문 text suffix만 prefill한 뒤 decode한다.

문제는 이미지 프리필 중 프롬프트가 들어오는 경우다.

```text
frame cache update running:
  V_Encode -> Mmproj -> ImagePrefill(256 vision tokens)

prompt arrives:
  기존 full-frame 정책: ImagePrefill 전체가 끝날 때까지 기다리거나 rollback
  partial 정책: 현재 image micro-batch만 끝내고 바로 답변
```

TTFT를 최우선으로 보려면 이미지 프리필 전체 256 토큰을 반드시 기다릴 필요가
없다. 현재 구현의 목표는 `--ubatch-size` 단위로 이미 KV에 들어간 image
token만 prompt에서 보이게 하고, 아직 처리하지 않은 image token은 버리는
것이다.

InternVL3 현재 one-tile frame은 보통 256 vision tokens이다.

```text
--ubatch-size 64:
  batch 0 commit ->  64 vision KV tokens visible
  batch 1 commit -> 128 vision KV tokens visible
  batch 2 commit -> 192 vision KV tokens visible
  batch 3 commit -> 256 vision KV tokens visible
```

## 2. 최종 동작 모델

`--partial-vision-kv`는 독립 streaming mode가 아니다. 기존 hybrid
`--stream-mode vision-prefill`의 cache-update preemption policy를 바꾸는
옵션이다.

```text
normal vision-prefill:
  cache update starts
  if prompt appears before commit, rollback/preempt cache update
  prompt restores last complete snapshot

partial vision-prefill:
  cache update starts
  if prompt appears during image prefill, finish current image micro-batch
  commit that partial image KV
  close the image wrapper text chunks
  prompt evaluates question suffix from the partial KV state
```

중요한 점:

- partial commit은 "KV에 실제로 들어간 token"만 사용한다.
- `VISION_KV_SLOT 97..256`처럼 처리하지 않은 placeholder를 있다고 가정하지
  않는다.
- closed chat history는 유지한다.
- prompt 이후에는 stale frame work를 계속 마무리하지 않는다. 지나간 frame은
  live stream에서는 버려도 된다.
- normal `vision-prefill` snapshot path는 유지한다. partial mode만 live
  committed metadata를 사용한다.

## 3. CLI 변경

### `runner/cli.py`

추가된 public flag:

```text
--partial-vision-kv
--partial_vision_kv
```

구현 지점:

```text
parser.add_argument("--partial-vision-kv", "--partial_vision_kv", ...)
```

remote shell 생성 시 `hybrid_streaming_decode`에 그대로 전달한다.

```text
partial_vision_kv_arg =
  "--partial-vision-kv" if args.partial_vision_kv else ""

./hybrid_streaming_decode {online_buffer_arg} {partial_vision_kv_arg} ...
```

timeline plotting alias도 추가했다.

```text
"VisionPrefillImagePrefillBatch": "ImagePrefill"
```

이 덕분에 partial image batch span은 기존 `ImagePrefill` lane 위에 그려진다.
CSV에는 raw phase name이 남고, PNG에서는 다른 방법론과 비교하기 쉽도록
`ImagePrefill`로 합쳐 보인다.

## 4. mtmd helper 변경

파일:

```text
llama.cpp/tools/mtmd/mtmd-helper.h
llama.cpp/tools/mtmd/mtmd-helper.cpp
```

기존에는 image chunk prefill을 외부에서 보면 하나의 긴 작업처럼만 볼 수
있었다. partial preemption을 하려면 "몇 번째 image batch까지 실제로
llama_decode에 들어갔는지"를 알아야 한다.

### 새 callback 타입

```cpp
typedef void (*mtmd_decode_batch_callback)(
    int32_t batch_idx,
    int32_t n_batches,
    int32_t n_tokens_batch,
    int64_t start_ms,
    int64_t end_ms,
    void * user_data);
```

### 새 helper

```cpp
mtmd_helper_decode_image_chunk_with_abort_and_progress(...)
```

역할:

1. image/audio chunk를 `n_batch` 단위로 나눈다.
2. 각 batch마다 `llama_decode()`를 실행한다.
3. batch가 끝날 때마다 `on_batch_done()`을 호출한다.
4. batch가 끝날 때마다 `*new_n_past`를 갱신한다.
5. abort callback이 true를 반환하면 `2`를 반환한다.

핵심 상태 갱신:

```cpp
decoded_n_pos += n_tokens_batch;
*new_n_past = n_past + decoded_n_pos;
```

full chunk가 끝난 경우에는 기존처럼 chunk 전체 position 수로 보정한다.

```cpp
decoded_n_pos = mtmd_input_chunk_get_n_pos(chunk);
*new_n_past = n_past + decoded_n_pos;
```

기존 함수들은 새 helper를 감싸도록 유지했다.

```text
mtmd_helper_decode_image_chunk()
  -> mtmd_helper_decode_image_chunk_with_abort()
  -> mtmd_helper_decode_image_chunk_with_abort_and_progress()
```

따라서 기존 호출부는 깨지지 않고, partial이 필요한 호출부만 progress
callback을 사용한다.

## 5. hybrid_streaming_decode 변경

파일:

```text
my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp
```

### Args

`Args`에 옵션을 추가했다.

```cpp
bool partial_vision_kv = false;
```

파서에서 두 spelling을 받는다.

```cpp
--partial-vision-kv
--partial_vision_kv
```

### Cache build status

기존 cache build status에 partial을 추가했다.

```cpp
enum class VisionPrefillCacheBuildStatus {
  Ok,
  Failed,
  Preempted,
  Partial,
};
```

`Partial`은 "cache update가 완전 성공은 아니지만, prompt가 사용할 수 있는
partial image KV를 commit했다"는 뜻이다. event detail은 `partial`로 기록된다.

## 6. Preemption callback 구조

prompt arrival은 worker thread 밖 producer path에서 감지된다. vision-prefill
prompt가 enqueue되면 pending prompt count를 올린다.

```cpp
std::atomic<int> pending_prompt_jobs{0};
pending_prompt_jobs.fetch_add(1, std::memory_order_release);
```

cache update 쪽은 이 값을 보고 중단 가능 여부를 판단한다.

```cpp
cache_preempt_requested(pending_prompt_jobs)
```

partial mode에서는 "프롬프트가 왔으니 즉시 멈춤"이 아니라
"최소 하나의 image micro-batch가 끝난 뒤 멈춤"이어야 한다. 그래서 callback
state에 아래 필드를 추가했다.

```cpp
struct CachePreemptDecodeCallback {
  const std::atomic<int>* pending_prompt_jobs;
  int32_t* completed_image_batches;
  bool require_completed_batch_before_abort;
};
```

핵심 조건:

```cpp
if (require_completed_batch_before_abort &&
    completed_image_batches != nullptr &&
    *completed_image_batches <= 0) {
  return false;
}
```

이 조건 때문에 prompt가 image batch 0 시작 직후 들어와도 batch 0은 끝까지
돌고, 그 뒤에 abort된다.

## 7. ImagePrefill batch progress 기록

partial mode에서는 image prefill 전체를 하나의 긴 bar로 보면 안 된다. 실제로
preempt 가능한 지점은 micro-batch boundary이기 때문이다.

추가된 progress state:

```cpp
struct ImagePrefillBatchProgress {
  phase_recorder* phases;
  int32_t* completed_image_batches;
};
```

callback:

```cpp
image_prefill_batch_progress_callback(...)
```

역할:

- `completed_image_batches`를 증가시킨다.
- `VisionPrefillImagePrefillBatch` phase row를 기록한다.

```text
VisionPrefillImagePrefillBatch
  -> plot alias: ImagePrefill
```

따라서 timeline에서는 image prefill이 64-token 단위 block으로 보인다.

## 8. eval_streaming_chunks_with_on_demand_vision 변경

이 함수는 vision-prefill cache build 중 formatted chunks를 순서대로 실행한다.
partial 구현을 위해 인자가 늘었다.

```cpp
bool eval_streaming_chunks_with_on_demand_vision(
    ...,
    int32_t image_prefill_batch_size,
    bool allow_partial_image_commit,
    bool* partial_image_committed,
    const std::atomic<int>* pending_prompt_jobs,
    bool* preempted);
```

### image batch size

partial image prefill batch size는 CLI의 `--ubatch-size`에서 온다.

```cpp
const int32_t image_prefill_batch_size = std::max(1, args.ubatch_size);
const int32_t preemptible_image_batch =
    std::min<int32_t>(ctx.n_batch, image_prefill_batch_size);
```

예를 들어 `-b 1024 -ub 64`이면 전체 llama batch는 1024지만 image prefill은
64-token 단위로 visible commit point가 생긴다.

### partial abort 처리

image chunk decode 호출은 새 mtmd helper를 사용한다.

```cpp
mtmd_helper_decode_image_chunk_with_abort_and_progress(...)
```

반환값 의미:

```text
0: image chunk full success
2: abort requested
else: decode failure
```

abort가 발생했고 `new_n_past > image_n_past_before`이면 이미 일부 image KV가
llama context에 들어간 상태다.

```cpp
if (allow_partial_image_commit && new_n_past > image_n_past_before) {
  ctx.n_past = new_n_past;
  llm_state_mutated = true;
  drain_text_chunks_after_partial_image(i + 1);
  *partial_image_committed = true;
}
```

`ctx.n_past`를 partial 위치로 전진시키는 것이 핵심이다. 이 값이 prompt에서
실제로 사용할 KV 길이가 된다.

### text chunk drain

partial image commit 직후 바로 prompt suffix를 붙이면 image wrapper가 닫히지
않은 상태가 될 수 있다. 그래서 image chunk 뒤에 이어지는 text chunks를
다음 image chunk가 나오기 전까지 drain한다.

```cpp
drain_text_chunks_after_partial_image(i + 1)
```

이 함수는 다음 IMAGE chunk를 만나면 멈춘다.

```cpp
if (mtmd_input_chunk_get_type(drain_chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
  break;
}
```

즉, 현재 frame의 partial image KV와 그 뒤의 닫는 text만 commit하고, 다음
frame image까지 억지로 진행하지 않는다.

### text-only mutation preempt

partial mode에서는 image batch뿐 아니라 이미 text chunk가 KV를 바꾼 뒤 prompt가
들어오는 경우도 있다. 이를 위해 `llm_state_mutated`와
`commit_partial_cache_preempt()`를 둔다.

```cpp
if (allow_partial_image_commit &&
    llm_state_mutated &&
    cache_preempt_requested(pending_prompt_jobs)) {
  *partial_image_committed = true;
  return true;
}
```

이는 "이미 context를 바꿨는데 rollback하지 않고 현재 committed prefix를
cache로 삼겠다"는 경로다.

## 9. cache save / restore 변경

기존 normal vision-prefill은 seq state를 저장한다.

```cpp
llama_state_seq_get_data_ext(...)
llama_state_seq_set_data_ext(...)
```

partial mode는 여기서 별도 정책이 필요했다. 아직 처리하지 않은 vision slot을
snapshot에 있다고 착각하면 이후 prompt token trace와 실제 KV state가 어긋난다.

그래서 `save_vision_prefill_cache_state()`에 `live_only` 인자를 추가했다.

```cpp
bool save_vision_prefill_cache_state(
    decode_context& ctx,
    VisionPrefillCache& cache,
    phase_recorder& phases,
    bool live_only = false)
```

`live_only == true`일 때:

```cpp
cache.state.clear();
cache.host_state.clear();
cache.state_flags = 0;
cache.host_state_flags = 0;
cache.n_past = ctx.n_past;
cache.chat_history = ctx.chat_history;
```

즉 partial mode에서는 현재 llama context에 실제로 살아있는 committed KV 위치와
chat history를 기준으로 기록한다. normal mode는 기존 seq-state snapshot 저장
경로를 그대로 탄다.

호출부:

```cpp
save_vision_prefill_cache_state(ctx, next_cache, cache_phases, args.partial_vision_kv)
save_vision_prefill_cache_state(ctx, *vision_cache, prompt_phases, args.partial_vision_kv)
```

## 10. build_vision_prefill_cache 변경

`build_vision_prefill_cache()`는 cache update job 하나를 수행한다.

partial mode에서 핵심 흐름:

```text
1. previous cache restore 또는 fresh prefix build
2. tokenize formatted prefix
3. eval_streaming_chunks_with_on_demand_vision(..., partial enabled)
4. if preempted and partial_image_committed:
     save current live cache
     row VisionPrefillCachePartialCommit
     return Partial
5. if preempted and no committed KV:
     rollback
     return Preempted
6. otherwise full cache save
```

코드 레벨 핵심:

```cpp
if (preempted) {
  if (partial_image_committed) {
    save_vision_prefill_cache_state(ctx, next_cache, cache_phases, args.partial_vision_kv);
    next_cache.valid = true;
    cache_phases.row("VisionPrefillCachePartialCommit", build_start_ms, now_ms());
    cache = std::move(next_cache);
    return VisionPrefillCacheBuildStatus::Partial;
  }
  rollback_vision_prefill_cache_build(...);
  return VisionPrefillCacheBuildStatus::Preempted;
}
```

partial이 아닌 경우에는 기존처럼 prompt가 들어오면 rollback/preempt한다.

```cpp
if (cache_preempt_requested(pending_prompt_jobs) && !args.partial_vision_kv) {
  rollback...
  return Preempted;
}
```

## 11. online buffer와의 관계

`--online-buffer`에서는 frame input cadence와 processing cadence가 다르다.
cache update가 밀려 있으면 stale cache update는 버릴 수 있다. live camera
모델에서는 지나간 frame을 전부 복원하는 것보다 최신 상태와 prompt TTFT가 더
중요하기 때문이다.

producer path:

```cpp
if (args.online_buffer) {
  buffer_stats.skipped_cache_updates += drop_pending_cache_updates(stream_jobs);
}
```

cache build path에서는 이미 cache에 있는 prefix 뒤에 최대 다음 missing frame
하나만 append한다.

```cpp
if (cached_prefix_size > 0 && target_frames.size() > cached_prefix_size + 1) {
  target_frames.resize(cached_prefix_size + 1);
}
```

이는 delayed cache worker가 과거 missing frame 전체를 따라잡느라 prompt를
막지 않게 하기 위한 정책이다.

## 12. prompt 처리와 multi-turn 보존

prompt job은 committed cache를 기준으로 질문 suffix를 평가한다.

```text
restore latest cache
split formatted question suffix
eval suffix text
decode answer
append user/assistant to chat history
save post-answer state
```

partial mode에서도 prompt 후 state 저장은 같은 함수로 간다.

```cpp
save_vision_prefill_cache_state(ctx, *vision_cache, prompt_phases, args.partial_vision_kv)
```

그래서 prompt 1에서 "What did I ask earlier???"를 물으면 prompt 0의 user turn과
assistant answer가 chat history에 남아 있어야 한다.

## 13. trace / artifact 변경

추가되거나 의미가 중요해진 phase:

```text
VisionPrefillImagePrefillBatch
  image micro-batch 단위 prefill span

VisionPrefillCachePartialCommit
  prompt preemption 때문에 full frame이 아니라 partial KV를 cache로 commit한 span

VisionPrefillCachePreempt
  pending prompt 때문에 cache update가 중단 지점에 도달한 marker
```

plot alias:

```text
VisionPrefillImagePrefillBatch -> ImagePrefill
```

stream event detail:

```text
VisionPrefillCacheBuild status detail:
  ok | preempted | partial | miss
```

buffer stats:

```cpp
if status == Ok or Partial:
  note_committed_cache_update(...)
else if status == Preempted:
  skipped_cache_updates += 1
```

따라서 partial commit은 "실제로 처리된 visual job"으로 계산된다.

## 14. contract tests

파일:

```text
my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py
my_research/foundation_llamacpp/tests/test_phase_plot_contract.py
```

주요 contract:

- mtmd helper header에 abort/progress helper가 존재해야 한다.
- `new_n_past`가 image batch 진행 후 갱신되어야 한다.
- `--partial-vision-kv` flag와 `VisionPrefillCacheBuildStatus::Partial`이
  존재해야 한다.
- prompt preemption은 최소 하나의 image batch가 끝난 뒤 가능해야 한다.
- partial image commit 후 text chunk drain이 있어야 한다.
- `--ubatch-size`가 visible image chunk granularity로 쓰여야 한다.
- `VisionPrefillImagePrefillBatch`는 plot에서 `ImagePrefill`로 alias되어야
  한다.

## 15. 검증 상태

현재 clean partial implementation merge 후 전체 foundation_llamacpp tests:

```text
pytest -q my_research/foundation_llamacpp/tests
51 passed
```

1B Q8 surveillance partial run:

```text
--stream-mode vision-prefill
--partial-vision-kv
--ubatch-size 64
--dynamic-kv-cache --kv-init-size 512 --kv-grow-step 512
prompts: 5s / 8s / 11s / 14s
```

결과 위치:

```text
my_research/foundation_llamacpp/results/log/
  partial_vprefill_clean_surveillance_1b_q8_batch64_20s_4prompt/
    InternVL3-1B-Instruct-Q8_0_hybrid_ctx_32768_streaming_vision_prefill_kv8_dynamic/
```

관찰:

- prompt 1이 이전 질문을 회수했다.
- timeline에 64-token image-prefill batch boundary가 보인다.
- partial mode에서도 multi-turn chat state는 유지된다.

2B Q8 검증은 adb device disconnect 때문에 inference가 시작되지 않았다. 해당
시도는 모델 검증으로 취급하지 않는다.

## 16. 현재 한계와 후속 확장

### M-RoPE 모델

현재 partial `new_n_past` batch 갱신은 `n_tokens_batch` 기준이다. full chunk가
끝나면 `mtmd_input_chunk_get_n_pos(chunk)`로 보정하지만, partial 상태에서
M-RoPE 모델을 정식 지원하려면 batch별 position span을 별도로 계산해야 한다.
현재 검증 범위는 InternVL3 non-M-RoPE streaming path다.

### true chunked vision prefill

이 구현은 `--chunked-vision-prefill`이 아니다. 현재는 active streaming user
turn snapshot 안에서 partial commit을 허용한다. 미래의 chunked mode는 1-frame,
2-frame 같은 독립 reusable KV chunk를 만들어 composition하는 별도 mode로
가야 한다.

### future vision token control

지금은 llama/mmproj image chunk의 256 visual placeholder 중 일부 batch만
commit하는 방식이다. 나중에 vision encoder 단계에서 token 수 자체를 제어하게
되면, 이 문서의 partial commit 정책은 "encoder output token count"와
"LLM image placeholder count"를 같이 맞추는 방향으로 확장해야 한다.

## 17. 실행 예시

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation/results/model/qnn/internvl3_1b_hybrid_16p_16k_16a4w/vision_encoder_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --vision-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --executorch-build-dir executorch/build-android-unified \
  --model llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8_20sec.mp4 \
  --stream-mode vision-prefill \
  --partial-vision-kv \
  --sampling-fps 1.0 \
  --max-video-time 20 \
  --time '[5.0, 8.0, 11.0, 14.0]' \
  --prompt '["What is this situation?", "What did I ask earlier???", "What changed in the scene?", "Summarize the full situation so far."]' \
  --max-num 1 \
  --n-predict 64 \
  --ctx-size 32768 \
  --dynamic-kv-cache \
  --kv-init-size 512 \
  --kv-grow-step 512 \
  --batch-size 1024 \
  --ubatch-size 64 \
  --gpu-layers 99 \
  --threads 4 \
  --temperature 0.0 \
  --device GPUOpenCL \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --fit off \
  --remote-root /data/local/tmp/streamingvlm_1b_partial_vprefill \
  --results-root my_research/foundation_llamacpp/results/log/partial_vprefill_clean_surveillance_1b_q8_batch64_20s_4prompt
```
