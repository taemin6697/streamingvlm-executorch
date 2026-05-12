# Single-Buffer Streaming Video 구현 설명

문서 목적: `my_research/foundation_llamacpp`에 추가한 streaming video mode가 코드 단위에서 어떻게 구현되어 있는지 정리한다. 특히 `--streaming-video --single-buffer`가 host Python runner, Android remote script, C++ streaming runner, QNN/OpenCL backend, artifact finalization을 거쳐 어떻게 동작하는지 재구성할 수 있게 남긴다.

이 문서는 line number가 아니라 파일명과 심볼명 기준으로 읽는다. upstream merge나 refactor로 줄 번호는 쉽게 바뀔 수 있다.

## 1. 최종 동작 모델

현재 구현된 streaming mode는 실제 카메라 입력을 직접 받는 online pipeline이 아니라, video file을 timestamp가 있는 stream처럼 replay하는 file-backed simulator이다.

사용자는 다음 형태로 실행한다.

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8.mp4 \
  --single-buffer \
  --sampling-fps 1.0 \
  --time '[5.0, 8.0]' \
  --prompt '["What is happening in this image?", "What did I ask earlier???"]' \
  ...
```

핵심 semantics는 아래와 같다.

1. Host에서 video를 `--sampling-fps`로 sampling한다.
2. 각 sampled frame에 stream timestamp를 붙여 `media_manifest.json`에 기록한다.
3. Android runner가 manifest를 읽고 frame timestamp 순서대로 replay한다.
4. `SingleBufferUpdate` 시점마다 current buffer는 최신 frame 하나로 교체된다.
5. `--time`으로 지정한 prompt timestamp가 도착하면, 그 순간 current buffer에 있던 frame을 prompt job에 고정한다.
6. prompt decode는 single consumer lane에서 직렬 실행한다. 따라서 P1 실행 시작은 P0 prefill/decode 때문에 밀릴 수 있다.
7. 그래도 P1이 사용할 image는 P1 timestamp 도착 당시 buffer에 있던 frame이다.
8. decoder context와 chat history는 prompt마다 초기화하지 않는다. multi-turn 질문이 이전 turn을 참조할 수 있다.

중요한 점은 `SingleBufferUpdate`가 queue에서 오래된 frame을 하나씩 pop하는 구조가 아니라는 것이다. 단일 슬롯에 latest frame pointer를 계속 덮어쓴다.

## 2. 변경 파일 요약

Streaming 구현은 project-owned 파일 중심으로 추가했다. 원칙적으로 ExecuTorch upstream source는 직접 수정하지 않았다.

주요 Python runner 변경:

- `runner/media.py`: streaming video sampling과 `media_manifest.json` 생성.
- `runner/cli.py`: CLI argument, validation, Android script generation, streaming artifact pull/finalize, timeline plot.
- `runner/artifacts.py`: streaming artifact pull list 추가.

주요 C++ bridge 변경:

- `hybrid_bridge/CMakeLists.txt`: `hybrid_streaming_decode`, `opencl_streaming_decode` target 추가.
- `hybrid_bridge/hybrid_streaming_decode.cpp`: Android-side streaming event loop와 prompt execution 구현.
- `hybrid_bridge/vision_encoder_et.{hpp,cpp}`: reusable `VisionEncoderSession` 추가.
- `hybrid_bridge/hybrid_decode.cpp`: streaming QNN path에서 external embedding eval/generation helper를 재사용할 수 있게 일부 helper를 공유.
- `hybrid_bridge/opencl_phase_mtmd.cpp`: OpenCL streaming binary에서 main 중복 없이 include할 수 있게 compile guard 적용.

문서/사용자 가이드 변경:

- `docs/README.md`: text/image/video/streaming 실행 모드별 명령 정리.
- `docs/project_structure.md`: streaming media mode, C++ target, artifacts, runtime flow 반영.
- `docs/for_cursor_llm_llamacpp_version2.md`: 구현 로그와 검증 결과 기록.
- `my_research/foundation/docs/for_cursor_llm.md`: workspace-level cumulative log 업데이트.

## 3. Host CLI 변경

### 3.1 CLI argument

`runner/cli.py`에 streaming 전용 argument를 추가했다.

주요 인자:

- `--streaming-video` / `--streaming_video`: streaming replay에 사용할 source video.
- `--single-buffer` / `--single_buffer`: latest frame 하나만 유지하는 mode.
- `--sampling-fps` / `--sampling_fps`: video에서 frame을 sampling할 FPS.
- `--max-video-time` / `--max_video_time`: sampling할 최대 stream duration.
- `--time`: prompt arrival timestamp JSON list.
- `--prompt`: streaming mode에서는 prompt string 하나가 아니라 JSON list로 해석한다.

관련 심볼:

- `runner/cli.py::_parse_json_list_arg`
- `runner/cli.py::_parse_streaming_prompt_events`
- `runner/cli.py::main`

`_parse_streaming_prompt_events()`는 `--time`과 `--prompt`를 JSON list로 parse하고 길이가 다르면 종료한다. prompt event는 `{"time": float, "prompt": str}` 형태로 저장하고 timestamp 순서로 정렬한다.

### 3.2 mode validation

`runner/cli.py::main`에서 media input은 상호 배타적으로 검사한다.

```text
--image
--video
--streaming-video
```

셋 중 하나만 사용할 수 있다. streaming mode에서는 `--sampling-fps`가 positive value여야 하고 `--time`이 필요하다.

현재 streaming 지원 backend:

- `--processor hybrid`: QNN vision encoder + llama.cpp/OpenCL decoder.
- `--processor gpu`: llama.cpp/mtmd OpenCL full vision + decoder.
- `--processor cpu`: streaming 미지원.

## 4. Host Media Preparation

Streaming media preparation은 `runner/media.py::prepare_streaming_video_media()`에서 담당한다.

입력:

- source video path
- temporary work directory
- `sampling_fps`
- parsed `prompt_events`
- `max_num`
- optional `max_video_time`
- `single_buffer`

처리 흐름:

1. `decord.VideoReader`로 video를 연다.
2. source FPS와 frame count를 읽는다.
3. `effective_duration_s = min(duration_s, max_video_time)`로 sampling 범위를 정한다.
4. `idx / sampling_fps` 형태의 target timestamps를 만든다.
5. 각 timestamp를 source frame index로 변환한다.
6. single-buffer mode에서는 sampled frame마다 하나의 `.png`와 하나의 `.bin`을 만든다.

single-buffer output naming:

```text
stream_frame_0000.png
stream_frame_0000.bin
stream_frame_0001.png
stream_frame_0001.bin
...
media_manifest.json
```

`.png`는 mtmd layout/image tokenization에 사용된다. `.bin`은 QNN hybrid path에서 ExecuTorch vision encoder input으로 사용된다. 둘 다 같은 original frame에서 나온다.

single-buffer mode에서는 tile을 여러 개 만들지 않고 `num_patches = 1`로 둔다. 즉 현재 구현은 latest sampled frame 하나를 single image로 보는 baseline이다.

## 5. Streaming Manifest

`media_manifest.json`은 Android runner가 읽는 유일한 stream schedule이다.

중요 필드:

```json
{
  "schema_version": 2,
  "source_kind": "streaming_video",
  "source_fps": 30.0,
  "sampling_fps": 1.0,
  "duration_s": 20.0,
  "effective_duration_s": 10.0,
  "max_video_time": 10.0,
  "stream_mode": "single_buffer",
  "frames": [
    {
      "stream_frame": 0,
      "timestamp_s": 0.0,
      "video_frame_index": 0,
      "num_patches": 1,
      "tiles": [
        {
          "bin": "stream_frame_0000.bin",
          "layout_image": "stream_frame_0000.png"
        }
      ]
    }
  ],
  "prompt_events": [
    {
      "time": 5.0,
      "prompt": "What is happening in this image?"
    }
  ]
}
```

여기서 `timestamp_s`는 sampled frame의 stream/video time이고, `prompt_events[*].time`은 prompt arrival time이다.

## 6. Android Remote Script

`runner/cli.py::_build_hybrid_streaming_remote_script()`가 streaming용 Android shell script를 만든다. 이름은 hybrid로 시작하지만 GPU OpenCL streaming도 같은 builder를 공유한다.

핵심 분기:

```text
if args.processor == "hybrid":
  runner_bin = "hybrid_streaming_decode"
  stdout_name = "hybrid_streaming_stdout.txt"
  pass --encoder-path and --warmup-image-path
else:
  runner_bin = "opencl_streaming_decode"
  stdout_name = "opencl_streaming_stdout.txt"
  do not pass QNN encoder args
```

script가 실행하는 C++ runner command는 공통적으로 다음 artifact path를 넘긴다.

```text
--stream-manifest media_manifest.json
--stream-events-path stream_events.csv
--phase-stats-path streaming_phase_stats.csv
--output foundation_output.txt
--token-io-path foundation_token_io.txt
```

memory timeline은 기존 run과 동일하게 Android shell loop에서 `android_memory_timeline.csv`로 sampling한다.

## 7. Artifact Pull And Finalization

Streaming artifact pull list는 `runner/artifacts.py::HYBRID_STREAMING_PULL_ARTIFACTS`에 있다.

주요 artifact:

- `hybrid_streaming_stdout.txt`
- `opencl_streaming_stdout.txt`
- `foundation_output.txt`
- `stream_events.csv`
- `streaming_phase_stats.csv`
- `foundation_token_io.txt`
- `foundation_inference_tokens.txt`
- `stream_inference_tokens_*.txt`
- `media_manifest.json`
- `foundation_exit_code.txt`
- `android_memory_timeline.csv`

처음에는 wildcard artifact를 못 가져왔기 때문에 per-turn token trace가 remote에는 있어도 host result directory에 없을 수 있었다. 이를 위해 `runner/cli.py::_pull_outputs()`에 `*`, `?`, `[`가 들어간 artifact name을 remote shell glob으로 listing한 뒤 각각 pull하는 로직을 추가했다.

Finalization은 `runner/cli.py::_finalize_hybrid_streaming_outputs()`가 담당한다.

하는 일:

1. `hybrid_streaming_stdout.txt` 또는 `opencl_streaming_stdout.txt`를 `foundation_output.txt`로 보존한다.
2. `stream_events.csv`에서 frame count와 prompt count를 계산한다.
3. `streaming_phase_stats.csv`를 normalized `foundation_proc.csv`로 변환한다.
4. memory summary와 memory plot을 만든다.
5. phase stacked bar를 만든다.
6. streaming 전용 `streaming_phase_timeline.png`를 만든다.

## 8. Timeline Plot Fix

`runner/cli.py::_write_png_streaming_phase_timeline()`은 streaming prompt timeline plot을 만든다.

처음 구현에서는 x-axis가 첫 prompt 근처로 rebasing되어 prompt 0이 3초 또는 5초에 도착해도 0초처럼 보였다. 이를 고치기 위해 `stream_events.csv`의 첫 `StreamFrameEnqueue` 또는 `SingleBufferUpdate`를 읽어서 elapsed time과 video time 사이 offset을 계산한다.

변환 함수:

```text
stream_time = elapsed_s - stream_origin_elapsed + stream_origin_video
```

이제 plot x-axis label은 `Stream Time (s)`이고 prompt marker는 `Prompt 0 @ 5.0s`처럼 표시된다.

## 9. CMake Target 구조

Streaming binary는 `hybrid_bridge/CMakeLists.txt`에서 추가했다.

추가 target:

```text
hybrid_streaming_decode
opencl_streaming_decode
```

둘 다 source file은 `hybrid_streaming_decode.cpp`를 공유한다. compile definition으로 QNN path와 OpenCL path를 나눈다.

Hybrid QNN target:

```text
STREAMINGVLM_HYBRID_DECODE_NO_MAIN=1
STREAMINGVLM_STREAMING_DECODE_USE_QNN=1
```

이 target은 `hybrid_decode.cpp`를 include해서 external embedding eval/generation helper를 재사용하고, ExecuTorch build가 켜져 있을 때 `vision_encoder_et.cpp`와 QNN backend를 link한다.

OpenCL target:

```text
STREAMINGVLM_OPENCL_PHASE_MTMD_NO_MAIN=1
```

이 target은 `opencl_phase_mtmd.cpp`를 include해서 llama.cpp/mtmd OpenCL vision encode, mmproj, prefill, decode helper를 재사용한다.

이 구조 덕분에 streaming event loop는 하나의 파일에 유지하면서 backend-specific prompt execution만 compile-time branch로 갈라진다.

## 10. C++ Streaming Runner

핵심 파일은 `hybrid_bridge/hybrid_streaming_decode.cpp`이다.

### 10.1 Conditional include

파일 상단에서 QNN build와 OpenCL build를 compile definition으로 구분한다.

```cpp
#if defined(STREAMINGVLM_STREAMING_DECODE_USE_QNN)
#include "vision_encoder_et.hpp"
#include "hybrid_decode.cpp"
#else
#include "opencl_phase_mtmd.cpp"
#endif
```

QNN build는 `VisionEncoderSession`과 `hybrid_decode.cpp` helper를 사용한다. OpenCL build는 `opencl_phase_mtmd.cpp` helper를 사용한다.

### 10.2 Manifest parser

구조체:

- `TileRecord`
- `FrameRecord`
- `PromptEvent`
- `Manifest`

관련 함수:

- `parse_manifest`
- `object_blocks_in_array`
- `find_number_after`
- `find_string_after`

현재 C++ parser는 full JSON library를 쓰지 않고 manifest에서 필요한 key만 찾는 작은 parser이다. Android binary dependency를 늘리지 않기 위한 선택이다. Manifest schema를 바꾸면 이 parser도 같이 업데이트해야 한다.

### 10.3 EventWriter

`EventWriter`는 `stream_events.csv`를 쓴다.

CSV schema:

```text
event,frame_idx,prompt_idx,video_time_s,elapsed_s_start,elapsed_s_end,detail
```

주요 event:

- `StreamFrameEnqueue`: sampled frame이 stream replay에 들어옴.
- `SingleBufferUpdate`: latest image buffer가 해당 frame으로 교체됨.
- `StreamPromptPrefill`: prompt arrival이 current frame과 binding됨.
- `StreamDecode`: 해당 prompt job의 실제 execution span.

이 event log를 보면 prompt arrival과 실제 decode start가 다를 수 있음을 확인할 수 있다.

### 10.4 Context loading

`load_single_buffer_decoder_context()`는 첫 layout image를 warm image처럼 넣어 llama.cpp/mtmd decode context를 한 번 만든다.

QNN build에서는 `load_single_buffer_encoder_context()`가 추가로 실행된다.

```text
VisionEncoderSession(args.encoder_path)
optional fixed warmup image encode
```

중요한 점은 prompt마다 QNN module을 다시 load하지 않는다는 것이다. streaming에서는 encoder session과 decoder context를 stream lifetime 동안 유지한다.

## 11. Producer / Consumer Scheduling

`hybrid_streaming_decode.cpp::main`은 하나의 producer thread와 main consumer loop로 구성된다.

Producer 역할:

1. manifest frames를 timestamp 순서로 순회한다.
2. `args.realtime`이면 frame timestamp delta만큼 sleep한다.
3. `current_frame = frame`으로 single buffer를 갱신한다.
4. 현재 frame timestamp 이하인 prompt event를 모두 prompt job queue에 넣는다.
5. `StreamFrameEnqueue`와 `SingleBufferUpdate` event를 기록한다.
6. 모든 frame replay 후 남은 prompt는 마지막 current frame에 binding한다.

Consumer 역할:

1. condition variable로 prompt job을 기다린다.
2. queue에서 하나씩 prompt job을 pop한다.
3. `StreamPromptPrefill` row를 기록한다.
4. backend-specific `run_single_buffer_prompt()`를 호출한다.
5. per-prompt phase file을 global `streaming_phase_stats.csv`에 append한다.
6. response를 `foundation_output.txt`에 prompt section 형태로 append한다.
7. `StreamDecode` event를 기록한다.

이 구조 때문에 prompt execution은 직렬이다. P0가 오래 걸리면 P1 execution start는 밀린다. 하지만 P1 job에는 producer가 P1 arrival time에 본 `current_frame`이 copy되어 들어가므로, P1 selected frame은 유지된다.

## 12. Hybrid QNN Prompt Execution

QNN path의 `run_single_buffer_prompt()`는 compile flag `STREAMINGVLM_STREAMING_DECODE_USE_QNN`이 켜진 경우 사용된다.

처리 순서:

1. prompt job의 frame에서 `.bin`과 `.png`를 꺼낸다.
2. prompt text에 mtmd default marker가 없으면 앞에 붙인다.
3. per-turn trace writer를 `stream_inference_tokens_<idx>.txt`로 연다.
4. `VisionEncoderSession::encode({bin})`으로 selected frame 하나를 QNN encode한다.
5. QNN `ImageLoad`와 `V_Encode` duration을 prompt phase recorder에 기록한다.
6. `VisionEncodeResult` 값을 `EmbeddingFile`로 감싼다.
7. `eval_with_external_embedding(ctx, prompt_text, {image}, embedding, ...)`을 호출한다.
8. `generate_response()`로 text decode를 수행한다.
9. `stream_response_<idx>.txt`와 `stream_token_io_<idx>.txt`를 쓴다.
10. trace writer를 닫은 뒤 raw trace를 aggregate `foundation_inference_tokens.txt`에 append한다.

QNN timestamp fix도 여기에서 처리한다. `VisionEncoderSession` 내부 timestamp는 ExecuTorch timer origin이고, llama.cpp phase CSV는 `ggml_time_ms()` origin이다. 그래서 raw QNN timestamp를 그대로 쓰지 않고, `vision_start_ms = now_ms()`를 기준으로 QNN duration만 누적해 `ImageLoad`와 `V_Encode` row를 만든다.

## 13. OpenCL Prompt Execution

OpenCL path의 `run_single_buffer_prompt()`는 QNN compile flag가 없을 때 사용된다.

처리 순서:

1. prompt job의 frame에서 `.png` layout image를 꺼낸다.
2. prompt text에 mtmd default marker를 붙인다.
3. per-turn trace writer를 연다.
4. `eval_message(ctx, msg, {image}, ...)`를 호출한다.
5. `eval_message()` 내부에서 llama.cpp/mtmd OpenCL vision encode, `Mmproj`, image prefill, text prefill이 수행된다.
6. `generate_response()`로 text decode를 수행한다.
7. QNN path와 동일하게 response, token IO, aggregate token trace를 쓴다.

OpenCL streaming은 QNN `.bin`을 사용하지 않는다. `stream_frame_<idx>.png`만 있으면 된다.

## 14. Multi-Turn State

Streaming prompt마다 decoder context를 새로 만들지 않는다. `decode_context`는 stream lifetime 동안 한 번 load되고 모든 prompt가 같은 context를 공유한다.

따라서 다음 상태가 유지된다.

- llama KV cache
- sampler state
- `ctx.chat_history`
- `ctx.n_past`

초기 구현에서 multi-turn 확인을 위해 prompt 1을 `What did I ask earlier???`로 실행했다. Hybrid QNN streaming은 prompt 0의 질문이 image 상황에 관한 것이었다고 답해, 이전 turn state가 유지됨을 확인했다.

## 15. Token Trace Aggregation

문제:

처음에는 `foundation_inference_tokens.txt`를 prompt마다 직접 열어 쓰면서 이전 prompt trace가 덮어써졌다. 이후 aggregate 방식으로 바꿨지만, per-turn writer가 flush/close되기 전에 aggregate가 읽으면서 trace가 잘리는 문제가 있었다.

현재 방식:

1. prompt별 raw trace는 `stream_inference_tokens_<idx>.txt`에 쓴다.
2. generation이 끝나면 `trace_writer.reset()`으로 파일을 닫는다.
3. 닫힌 raw trace를 `foundation_inference_tokens.txt`에 append한다.
4. Host pull 단계에서 `stream_inference_tokens_*.txt` wildcard도 가져온다.

Aggregate section format:

```text
===== stream prompt 0 @ 5s =====
image: stream_frame_0005.png
user: What is happening in this image?

... raw token trace ...
```

## 16. Phase CSV

Streaming runner는 `streaming_phase_stats.csv`를 직접 쓴다.

Header는 기존 foundation phase CSV와 맞춘다.

```text
row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,
col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,
kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx
```

Streaming-specific row:

- `SingleBufferUpdate`
- `StreamPromptPrefill`

Backend phase row:

- `L_VisionLoad`
- `L_DecoderRuntimeInit`
- `L_DecoderLoad`
- `ImageLoad`
- `V_Encode`
- `LayoutTokenize`
- `Mmproj`
- `ImagePrefill`
- `T_Prefill`
- `D`

`runner/cli.py::_finalize_hybrid_streaming_outputs()`가 이 파일을 `foundation_proc.csv`로 복사/정규화해서 기존 plot 함수들이 읽을 수 있게 한다.

## 17. QNN VisionEncoderSession

`vision_encoder_et.{hpp,cpp}`에 `VisionEncoderSession`을 추가한 이유는 streaming에서 prompt마다 ExecuTorch module을 다시 load하면 안 되기 때문이다.

Interface:

```cpp
class VisionEncoderSession {
 public:
  explicit VisionEncoderSession(const std::string& encoder_path);
  VisionEncodeResult encode(const std::vector<std::string>& image_paths);
  VisionEncodeResult encode_with_optional_warmup(
      const std::vector<std::string>& image_paths,
      const std::string& warmup_image_path = "");
};
```

기존 one-shot helper `encode_images_with_executorch()`는 유지했다. 내부 구현만 `VisionEncoderSession`을 만들어 `encode_with_optional_warmup()`을 호출하도록 바꿨다. 그래서 기존 `hybrid_vision_dump` flow는 깨지지 않는다.

## 18. README / Result Directory / Naming

Streaming run result directory에는 `_streaming` suffix가 들어간다.

예시:

```text
results/log/InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_kv16/
results/log/InternVL3-2B-Instruct-Q8_0_opencl_ctx_4096_streaming_kv16/
results/log/InternVL3-8B-Instruct-Q4_K_M_hybrid_ctx_4096_streaming_kv16/
```

`docs/README.md`는 실행 모드를 아래 기준으로 재정리했다.

- single text input
- image input
- offline video input
- streaming video input
- vision tower export
- phase names
- etc

Streaming section에는 hybrid와 OpenCL GPU command가 모두 들어가 있다.

## 19. 검증 결과

Q8 2B hybrid streaming:

- model: `InternVL3-2B-Instruct-Q8_0.gguf`
- backend: QNN vision + OpenCL decoder
- prompts: 5s, 8s
- result: `InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_kv16`
- `foundation_exit_code.txt = 0`
- QNN `V_Encode` 약 370ms대
- multi-turn prompt가 이전 질문을 참조하는 것을 확인

Q8 2B OpenCL streaming:

- model: `InternVL3-2B-Instruct-Q8_0.gguf`
- backend: llama.cpp/mtmd OpenCL full vision + decoder
- result: `InternVL3-2B-Instruct-Q8_0_opencl_ctx_4096_streaming_kv16`
- `foundation_exit_code.txt = 0`
- `foundation_proc.csv`에 OpenCL `V_Encode`, `Mmproj`, `ImagePrefill`, `T_Prefill`, `D` 기록

8B hybrid streaming:

- text model: `InternVL3-8B-Instruct-Q4_K_M.gguf`
- mmproj: `mmproj-InternVL3-8B-Instruct-Q8_0.gguf`
- QNN vision: realweights SM8750 pre-projector PTE
- result: `InternVL3-8B-Instruct-Q4_K_M_hybrid_ctx_4096_streaming_kv16`
- `foundation_exit_code.txt = 0`
- QNN `V_Encode` 약 415ms / 371ms
- `ImagePrefill` 약 3973ms / 4800ms
- prompt 1이 이전 질문 내용을 기억함

Dynamic KV prototype validation:

- flags: `--dynamic-kv-cache --kv-init-size 1024 --kv-grow-step 1024`
- target: standard llama.cpp KV cache path used by 2B Q8 hybrid streaming
- fixed baseline: `InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_streaming_kv16`
  - `foundation_exit_code.txt = 0`
  - OpenCL KV buffer `112 MiB`, `4096/4096` cells
  - `ImagePrefill`: `1081, 1421, 1761, 2115 ms`
- dynamic run: `InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_kv16_dynamic`
  - `foundation_exit_code.txt = 0`
  - logical context `32768`
  - initial OpenCL KV buffer `28 MiB`, `1024/32768` cells
  - one grow: `1024 -> 2048` cells, OpenCL KV `56 MiB`
  - grow log: `dynamic KV grow completed in 78.029 ms`
  - `ImagePrefill`: `1077, 1427, 1769, 2386 ms`
- Dynamic KV reduces reserved KV memory. It does not reduce attention compute;
  decode/prefill latency still grows with the actual accumulated `n_kv`.

## 20. 현재 한계

현재 구현은 intentional baseline이다.

- Real camera input이 아니라 file-backed replay이다.
- `--single-buffer`만 구현되어 있다.
- Persistent vision KV prefill이나 historical frame cache는 아직 없다.
- Single-buffer mode는 latest frame 하나만 사용하므로 긴 temporal history를 보존하지 않는다.
- Prompt execution은 single consumer lane이라 이전 prompt가 오래 걸리면 다음 prompt execution은 밀린다.
- OpenCL streaming은 실행/logging baseline으로 동작하지만, 답변 품질은 QNN hybrid path와 다를 수 있다.
- C++ manifest parser는 full JSON parser가 아니라 필요한 key만 읽는다. Manifest schema 변경 시 parser 동기화가 필요하다.

## 21. 다음에 수정할 때 체크리스트

Streaming media schema를 바꿀 때:

1. `runner/media.py::prepare_streaming_video_media()`를 수정한다.
2. `hybrid_streaming_decode.cpp::parse_manifest()`도 같이 수정한다.
3. `docs/project_structure.md`와 이 문서를 갱신한다.

새 streaming backend를 추가할 때:

1. `hybrid_bridge/CMakeLists.txt`에 target을 추가한다.
2. `hybrid_streaming_decode.cpp`에 compile definition branch 또는 새 runner를 추가한다.
3. `runner/cli.py::_build_hybrid_streaming_remote_script()`에서 runner binary 선택을 확장한다.
4. `runner/artifacts.py::HYBRID_STREAMING_PULL_ARTIFACTS`에 stdout/artifact를 추가한다.
5. `runner/cli.py::_finalize_hybrid_streaming_outputs()`가 backend name을 구분할 수 있게 한다.

Token trace를 바꿀 때:

1. Per-turn raw file을 먼저 완성한다.
2. Writer를 close/reset한다.
3. Aggregate `foundation_inference_tokens.txt`에 append한다.
4. Wildcard pull list에 per-turn file pattern이 있는지 확인한다.

Timeline을 바꿀 때:

1. `stream_events.csv`에서 prompt arrival과 execution span을 구분해 유지한다.
2. Plot x-axis는 elapsed time이 아니라 stream/video time으로 유지한다.
3. Prompt marker label은 requested prompt timestamp가 보이게 한다.

ExecuTorch/QNN 쪽을 바꿀 때:

1. Upstream ExecuTorch source를 직접 수정하지 않는다.
2. 가능하면 `vision_encoder_et.{hpp,cpp}` wrapper/session 안에서 처리한다.
3. QNN raw timer origin과 llama.cpp `ggml_time_ms()` origin을 섞지 않는다. Phase CSV에는 duration을 llama.cpp origin 위에 rebasing해서 기록한다.
