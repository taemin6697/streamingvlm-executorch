# Dynamic KV OpenCL Buffer Memory Architecture

이 문서는 dynamic KV grow가 모바일 SoC에서 어떤 메모리 영역을 사용하고,
grow 순간 어떤 data movement가 생기는지 컴퓨터 구조 관점에서 정리한다. 현재
대상은 foundation llama.cpp hybrid streaming 경로다.

## Current State

```text
Vision encoder:
  ExecuTorch / QNN

LLM decoder:
  llama.cpp / OpenCL

KV cache:
  llama.cpp standard KV cache
  OpenCL backend buffer로 offload

Active implementation:
  contiguous OpenCL KV buffer reallocation
  OpenCL device-to-device K/V migration

Not active:
  Paged KV cache prototype
```

Paged KV는 실험 후 main에서 revert했다. 현재 main의 dynamic KV는 standard
llama.cpp KV cache를 작게 시작한 뒤 필요할 때 더 큰 contiguous OpenCL buffer로
재할당하는 방식이다.

## Short Answer

Dynamic KV grow가 크게 잡는 메모리는 일반 C++ heap이 아니라
`ggml_backend_buffer`다. Hybrid/OpenCL 경로에서는 이 buffer가 OpenCL
device-accessible buffer로 잡힌다.

```text
KV 실제 K/V tensor storage:
  OpenCL backend buffer

KV cell metadata / tensor metadata:
  CPU host memory

grow 중 기본 K/V migration:
  old OpenCL buffer -> new OpenCL buffer
  clEnqueueCopyBuffer

fallback migration:
  old backend tensor -> host memory -> new backend tensor
  ggml_backend_tensor_get / ggml_backend_tensor_set
```

교수님께 짧게 설명하면:

```text
Dynamic KV grow는 logical context는 모델 최대치로 유지하되, 실제 K/V tensor의
physical capacity만 작게 시작합니다. Hybrid/OpenCL 경로에서 K/V tensor 본체는
일반 C++ heap이 아니라 OpenCL backend buffer에 잡힙니다. Grow가 발생하면 더 큰
OpenCL buffer를 새로 만들고, 기존 K/V 내용은 CPU를 거치지 않는
clEnqueueCopyBuffer device-to-device copy로 새 buffer에 옮깁니다. OpenCL fast
path가 불가능한 경우에만 host tensor get/set fallback이 사용됩니다.
```

## Mobile SoC Memory Model

모바일 SoC는 discrete GPU 시스템처럼 CPU DRAM과 GPU VRAM이 따로 있지 않은
경우가 일반적이다.

```text
Desktop discrete GPU:
  CPU system DRAM
  GPU VRAM

Mobile SoC:
  CPU / GPU / NPU가 같은 LPDDR system DRAM 공유
```

하지만 물리 DRAM을 공유한다고 해서 모든 allocation이 같은 방식으로 관리되는
것은 아니다.

```text
User process virtual memory:
  C++ heap
  std::vector
  stack
  mmap region

Driver-managed memory:
  OpenCL buffer object
  GPU-accessible allocation
  GPU IOMMU mapping
```

그림으로 보면:

```text
          System LPDDR DRAM
        +--------------------+
        | model weights      |  <- OpenCL/GPU backend buffer
        | KV cache buffer    |  <- OpenCL/GPU backend buffer
        | normal heap        |  <- CPU heap
        | stack              |
        +--------------------+

 CPU MMU:
   process heap / stack / mmap을 CPU virtual address로 mapping

 GPU IOMMU:
   OpenCL buffer object를 GPU virtual address로 mapping
```

OpenCL buffer는 물리적으로 LPDDR 위에 있을 수 있지만, C++ heap object가 아니다.
OpenCL runtime/driver가 소유하고 command queue, event, synchronization을 통해
접근한다.

## OpenCL Buffer Allocation Path

llama.cpp dynamic KV grow에서는 `llama_kv_cache::reset_capacity()`가 새 K/V
tensor metadata를 만들고, 아래 경로로 backend buffer를 할당한다.

```cpp
ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft)
```

여기서 `buft`가 CPU backend buffer type이면 host memory 계열로 잡히고,
OpenCL backend buffer type이면 OpenCL runtime/driver가 device-accessible
buffer object를 생성한다.

```text
llama.cpp
  -> ggml OpenCL backend
  -> OpenCL runtime
  -> GPU driver
  -> kernel memory allocator / dma-buf / ION 계열 allocation
  -> system DRAM pages 확보
  -> GPU IOMMU page table에 mapping
  -> GPU kernel이 GPU virtual address로 접근
```

정확한 allocator 이름은 기기/드라이버 구현에 따라 달라질 수 있다. 중요한 점은
OpenCL buffer가 process heap의 `malloc` block으로 확장되는 것이 아니라 driver
관리 allocation이라는 점이다.

## Dynamic KV Grow Sequence

OpenCL buffer는 `realloc()`처럼 제자리 확장할 수 없다. 따라서 grow는 새 buffer를
만들고 기존 내용을 옮기는 방식이다.

현재 main의 기본 OpenCL fast path:

```text
1. old KV objects lifetime 보존
   old ctxs_bufs / layers / v_cells / v_heads를 임시 owner로 move

2. 새 physical kv_size로 metadata 재생성
   ggml_context metadata는 host memory
   K/V tensor descriptors도 host-side metadata

3. 새 backend buffer 할당
   new OpenCL KV buffer allocate

4. 기존 K/V bytes를 old OpenCL buffer에서 new OpenCL buffer로 복사
   clEnqueueCopyBuffer

5. v_cells metadata grow
   기존 cell state 유지
   새 cell 영역은 empty state로 초기화

6. scheduler reserve 재수행
   실패했던 batch를 retry
```

Fallback path:

```text
old backend tensor
  -> ggml_backend_tensor_get
  -> host temporary bytes
  -> ggml_backend_tensor_set
  -> new backend tensor
```

Fallback은 OpenCL tensor 조건이 맞지 않거나 OpenCL helper가 실패한 경우만
사용한다. 정상 OpenCL-backed KV에서는 로그가 아래처럼 찍힌다.

```text
reset_capacity: dynamic KV data migration used device-to-device copy
```

## Memory Pressure During Grow

Grow 순간에는 적어도 다음 자원들이 겹친다.

```text
old OpenCL KV buffer
new OpenCL KV buffer
temporary host metadata
backend scheduler compute buffers
```

OpenCL fast path에서는 K/V payload를 CPU heap으로 왕복하지 않는다. 다만 old와
new OpenCL buffer가 동시에 살아 있어야 하므로, grow 순간 peak memory pressure는
여전히 커질 수 있다.

```text
old capacity = 1024 cells  ->  28 MiB
new capacity = 16384 cells -> 448 MiB
temporary overlap can approach old + new KV buffer lifetime
```

실제 Android memory metric은 allocator cache, DMA heap pool, driver deferred
free의 영향을 받는다. 정확한 KV capacity는 stdout의 `reset_capacity` log와
`foundation_proc.csv`의 `kv_physical_committed_kb`를 우선해서 본다.

## Memory Bus Perspective

Decode 중 attention은 현재 `n_kv` 범위의 K/V를 읽는다. fixed KV라도 max ctx
전체를 매번 읽는 것이 아니라 현재 사용된 KV 길이, 보통 padding된 `n_kv`까지만
읽고 계산한다.

```text
fixed KV:
  allocation = max ctx
  read/compute = current n_kv

dynamic KV:
  allocation = current physical KV capacity
  read/compute = current n_kv
```

GPU kernel의 read path는 대략 다음과 같다.

```text
GPU compute unit
  -> GPU cache / buffer cache
  -> NoC / interconnect
  -> memory controller
  -> LPDDR
```

Dynamic KV가 줄이는 것은 주로 초기/resident KV capacity다. 실제 per-token
decode latency는 현재 token이 attend해야 하는 accumulated `n_kv`가 커질수록
증가한다.

## Grow Traffic

OpenCL fast path의 K/V migration traffic:

```text
old OpenCL KV buffer
  -> clEnqueueCopyBuffer
  -> new OpenCL KV buffer
```

같은 LPDDR 위의 device-accessible allocation 사이 copy이므로, CPU register로
데이터를 읽어와 다시 쓰는 구조가 아니다. 그러나 memory controller 관점에서는
old buffer read와 new buffer write traffic이 발생한다.

Grow latency에는 다음이 포함된다.

```text
new OpenCL buffer allocation
new KV clear
old K/V -> new K/V device copy
backend scheduler reserve
failed batch retry preparation
```

실측 예:

```text
grow_to: growing dynamic KV cache: old = 1024, new = 16384, logical = 32768
reset_capacity:     OpenCL KV buffer size =   448.00 MiB
reset_capacity: dynamic KV data migration used device-to-device copy
grow_to: dynamic KV grow completed in 202.135 ms
DynamicKVGrow finalizer row: 299 ms
```

`grow_to()` internal time은 allocation/copy 중심이고, finalizer의
`DynamicKVGrow` row는 scheduler reserve와 retry preparation까지 포함한 더 넓은
window다.

## Cache Coherency

모바일 SoC에서 CPU와 GPU가 같은 LPDDR을 공유해도 cache hierarchy는 다를 수 있다.

```text
CPU:
  CPU L1/L2/L3 cache

GPU:
  GPU cache hierarchy

Shared:
  system interconnect
  memory controller
  LPDDR
```

OpenCL fast path는 device-to-device copy이므로 CPU가 K/V payload를 직접 읽고
쓰지 않는다. OpenCL runtime/driver가 command queue ordering과 synchronization을
보장한다. Fallback tensor get/set path에서는 backend가 내부적으로 read/write
buffer synchronization과 필요한 cache maintenance를 담당한다.

우리 코드에서는 직접 cache flush/invalidate를 호출하지 않는다.

## Fixed KV vs Dynamic KV

```text
fixed KV:
  context 생성 시 max ctx만큼 OpenCL KV buffer를 잡음
  메모리 footprint가 처음부터 큼
  attention read/compute는 current n_kv 기준

dynamic KV:
  kv-init-size만큼 OpenCL KV buffer를 잡음
  부족하면 더 큰 OpenCL KV buffer로 재할당
  grow 순간 allocation + device copy + scheduler reserve spike가 있음
  attention read/compute는 current n_kv 기준
```

2B Q8 f16 KV 기준 문서화된 값:

```text
1024 cells:   약  28 MiB
2048 cells:   약  56 MiB
4096 cells:   약 112 MiB
16384 cells:  약 448 MiB
```

## Why Not Paged KV Now

Paged KV는 page table addressing으로 grow-time realloc/copy를 줄이는 방향으로
실험했지만, 현재 연구 흐름에서는 제외했다. main에는 해당 prototype을 남기지
않는다.

현재 선택지는:

```text
Keep:
  contiguous dynamic KV grow
  OpenCL device-to-device migration
  clear DynamicKVGrow timeline instrumentation

Do not keep:
  paged KV page-table attention path
  max-context backing allocation trick
```

이 결정은 "초기 memory reservation을 줄이는 dynamic KV" 관찰을 깨끗하게 유지하기
위한 것이다. Paged KV나 true KV compression은 나중에 별도 branch/spec으로 다시
열어야 한다.
