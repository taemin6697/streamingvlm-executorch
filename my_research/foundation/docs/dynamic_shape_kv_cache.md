# Dynamic Shape와 KV-Cache 정리

이 문서는 XNNPACK export에서 `dynamic_shape`를 켰을 때 실제로 무엇이 달라지고, 왜 `max_context_len`이 커질수록 decode latency와 메모리가 영향을 받는지 정리한다.

## 핵심 결론

`dynamic_shape`는 주로 **입력 sequence length에 따른 연산량을 줄이는 기능**이다.

하지만 KV-cache와 ExecuTorch memory planning은 여전히 **`max_context_len` 기준의 최대 capacity**를 잡기 때문에, 메모리 효율은 크게 좋아지지 않는다.

즉 다음처럼 이해하면 된다.

```text
dynamic_shape ON:
  입력 길이: 실제 prefill/decode 길이에 맞게 가변 가능
  decode 입력: 보통 새 토큰 1개 단위로 호출 가능
  KV-cache capacity: max_context_len 전체 기준
  메모리 footprint: max_context_len에 강하게 묶임
```

## Decode는 1개씩 하나, 전체 context를 하나?

decode 단계에서 모델에 새로 들어가는 입력은 보통 **새 토큰 1개**다.

예를 들어 dynamic shape가 켜져 있으면 decoder input embedding은 대략 다음처럼 작게 들어갈 수 있다.

```text
embeddings: [batch=1, seq=1, hidden]
input_pos:  [1]
```

따라서 decode 때 매번 `max_context_len` 전체 토큰을 입력으로 다시 넣는 것은 아니다.

다만 attention은 새 토큰의 Query가 이전에 쌓인 Key/Value를 참조해야 한다. 그래서 내부에는 다음과 같은 KV-cache buffer가 존재한다.

```text
K cache: [layers, batch, heads, max_context_len, head_dim]
V cache: [layers, batch, heads, max_context_len, head_dim]
```

실제로 의미 있는 토큰은 현재까지 채워진 길이만큼이지만, buffer capacity 자체는 export 시 정한 `max_context_len`에 맞춰 잡힌다.

## Dynamic Shape가 줄이는 것

dynamic shape가 켜져 있으면 prefill이나 decode 호출에서 입력 sequence length를 고정하지 않아도 된다.

예를 들어 같은 exported artifact에서 다음처럼 서로 다른 입력 길이를 받을 수 있다.

```text
prefill: seq = 실제 prompt/image token 길이
decode:  seq = 1
```

이 덕분에 새로 들어온 입력 token 처리, embedding, 일부 decoder 연산은 실제 입력 길이에 맞게 줄어들 수 있다.

특히 decode에서 새 토큰 1개만 넣을 수 있다는 점은 static input shape 대비 중요한 차이다.

## Dynamic Shape가 줄이지 못하는 것

dynamic shape가 켜져 있어도 KV-cache의 최대 저장 공간은 동적으로 줄어들지 않는다.

`max_context_len=1024`로 export한 artifact와 `max_context_len=16384`로 export한 artifact는 내부 capacity가 다르다.

```text
1k artifact:
  KV-cache capacity ~= 1024 tokens

16k artifact:
  KV-cache capacity ~= 16384 tokens
```

따라서 실제 decode 입력이 1 token이라도, 16k artifact는 더 큰 KV-cache와 memory plan을 가진다.

이 때문에 `max_context_len`이 커지면 다음이 증가할 수 있다.

- 전체 RSS 메모리
- KV-cache 관련 buffer 크기
- memory planning에서 잡는 working memory
- cache 접근 및 backend 내부 처리 overhead
- token당 decode latency

## Dynamic Shape ON/OFF 비교

```text
dynamic_shape ON:
  입력 sequence length 가변 가능
  decode를 1 token shape로 호출 가능
  연산량 일부 감소
  KV-cache capacity는 max_context_len 기준
  메모리는 max_context_len에 크게 의존

dynamic_shape OFF:
  입력 sequence length가 export example shape에 고정
  prefill 길이와 decode 길이 1을 같은 decoder PTE로 처리하기 어려울 수 있음
  KV-cache capacity는 역시 max_context_len 기준
  메모리는 max_context_len에 크게 의존
```

결론적으로 dynamic shape는 **연산 효율 개선**에 가깝고, KV-cache 메모리 자체를 줄이는 기능은 아니다.

## 왜 여러 길이별 artifact가 필요한가?

하나의 큰 `max_context_len=16k` artifact를 모든 상황에 쓰면 짧은 context에서도 16k capacity의 비용을 일부 부담할 수 있다.

반대로 512, 1k, 2k, 4k, 8k, 16k처럼 길이별 artifact를 만들면, 실제 필요한 context 길이에 맞는 artifact를 선택할 수 있다.

```text
짧은 대화/짧은 영상:
  512 또는 1k artifact 사용
  메모리와 decode latency를 낮게 유지

긴 context가 필요한 경우:
  8k 또는 16k artifact 사용
  더 많은 과거 정보를 유지하지만 메모리/latency 비용 증가
```

모바일 환경에서는 메모리와 latency가 중요하므로, 하나의 큰 artifact만 쓰는 것보다 context length별 artifact를 준비하고 상황에 맞게 선택하는 전략이 더 적합하다.

## 현재 프로젝트에서의 의미

`my_research/foundation/scripts/export_internvl3_matrix.sh`는 기본적으로 다음 길이의 artifact를 만든다.

```text
512 1024 2048 4096 8192 16384
```

XNNPACK에서는 기본적으로 dynamic shape가 켜져 있다.

static shape 비교가 필요하면 다음처럼 실행한다.

```bash
DYNAMIC_SHAPE=0 my_research/foundation/scripts/export_internvl3_matrix.sh xnnpack
```

다만 static shape artifact는 현재 unified runner에서 바로 실행 비교가 어려울 수 있다. runner가 같은 decoder artifact로 prefill과 decode를 모두 호출하는데, static shape에서는 입력 sequence length가 고정되기 때문이다.
