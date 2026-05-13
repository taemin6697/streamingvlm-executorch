# Vision Prefill KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vision-prefill` restore cached image-prefix llama KV at prompt time instead of recomputing image prefill.

**Architecture:** Add a window snapshot cache inside the hybrid streaming bridge. Cache construction evaluates the chat-formatted video prefix with QNN image embeddings, saves seq 0 state, then prompt handling restores that state and evaluates only the formatted text suffix before decoding.

**Tech Stack:** C++17, llama.cpp mtmd helper APIs, llama sequence state APIs, QNN hybrid bridge, pytest for repository contract tests, Android hybrid runner for device verification.

---

### Task 1: Contract Tests

**Files:**
- Create: `my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py`
- Read: `my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp`
- Read: `docs/superpowers/specs/2026-05-13-vision-prefill-kv-cache-design.md`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STREAMING_CPP = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp"
SPEC = ROOT / "docs/superpowers/specs/2026-05-13-vision-prefill-kv-cache-design.md"


def test_vision_prefill_uses_llama_sequence_state_cache():
    source = STREAMING_CPP.read_text()

    assert "VisionPrefillCache" in source
    assert "llama_state_seq_get_data_ext" in source
    assert "llama_state_seq_set_data_ext" in source
    assert "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE" in source


def test_vision_prefill_has_prompt_suffix_boundary_and_cache_phases():
    source = STREAMING_CPP.read_text()

    assert "SVLM_QUESTION_SENTINEL" in source
    assert "VisionPrefillCacheBuild" in source
    assert "VisionPrefillCacheSave" in source
    assert "VisionPrefillCacheRestore" in source
    assert "VisionPrefillCacheHit" in source
    assert "VisionPrefillCacheMiss" in source


def test_future_chunked_mode_name_is_documented():
    spec = SPEC.read_text()

    assert "--chunked-vision-prefill" in spec
    assert "--chunk-count" in spec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py -q`

Expected: FAIL because `VisionPrefillCache` and cache phases are not implemented yet.

### Task 2: KV Cache Implementation

**Files:**
- Modify: `my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp`

- [ ] **Step 1: Add cache data structures and prompt formatting helpers**

Add a `VisionPrefillCache` struct near `PromptJob` support code. Add a sentinel constant named `SVLM_QUESTION_SENTINEL`. Add helpers that build full formatted prompt content and split formatted prompt into cache prefix and text suffix.

- [ ] **Step 2: Add low-level external embedding chunk evaluator**

Extract the chunk-evaluation loop from `eval_with_external_embedding` into a streaming-local helper that accepts an already-tokenized chunk list, an embedding file, a seq id, `logits_last`, and a phase label prefix. This helper must pass the target seq id to `mtmd_helper_decode_image_chunk` and `mtmd_helper_eval_chunk_single`.

- [ ] **Step 3: Build and save the vision prefill cache**

For `vision-prefill` prompt jobs in the QNN build, reset the decode context, QNN-encode the selected frame bins, tokenize only the formatted video prefix with images, evaluate it into seq 0, then save seq 0 with:

```cpp
std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx.lctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
const size_t copied = llama_state_seq_get_data_ext(ctx.lctx, state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
```

Record `VisionPrefillCacheBuild` and `VisionPrefillCacheSave` rows.

- [ ] **Step 4: Restore cache and evaluate only text suffix**

Before decoding, clear seq 0, restore the cached state with `llama_state_seq_set_data_ext`, set `ctx.n_past` to cached `n_past`, tokenize/evaluate the formatted suffix without image bitmaps, then call `generate_response`.

- [ ] **Step 5: Fallback on miss or restore failure**

If no cache is available, if the frame key does not match, or if state restore returns zero, emit `VisionPrefillCacheMiss` and call the existing `eval_with_external_embedding` path.

### Task 3: Verification

**Files:**
- Modify: `my_research/foundation_llamacpp/docs/README.md`

- [ ] **Step 1: Run the contract test**

Run: `pytest my_research/foundation_llamacpp/tests/test_vision_prefill_kv_cache_contract.py -q`

Expected: PASS.

- [ ] **Step 2: Run existing streaming media tests**

Run: `pytest my_research/foundation_llamacpp/tests/test_streaming_media.py -q`

Expected: PASS.

- [ ] **Step 3: Build Android hybrid streaming target**

Run: `cmake --build my_research/foundation_llamacpp/build-hybrid-android-opencl --target hybrid_streaming_decode -j2`

Expected: target builds successfully.

- [ ] **Step 4: Run 2B Q8 hybrid device test**

Run the Android hybrid runner with:

```bash
python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
  --processor hybrid \
  --vision my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte \
  --llama-build-dir my_research/foundation_llamacpp/build-hybrid-android-opencl \
  --model llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf \
  --mmproj llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf \
  --streaming-video my_research/foundation_llamacpp/sample_images/surveil_8_20sec.mp4 \
  --stream-mode vision-prefill \
  --sampling-fps 1.0 \
  --max-video-time 10 \
  --window-sec 4.0 \
  --window-max-frames 8 \
  --time '[5.0, 8.0]' \
  --prompt '["What is happening in this video window?", "What changed in the recent window?"]' \
  --max-num 1 \
  --n-predict 32 \
  --ctx-size 4096 \
  --gpu-layers 99 \
  --device GPUOpenCL \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --fit off \
  --soc-model SM8750 \
  --baseline-window 0 \
  --remote-root /data/local/tmp/streamingvlm_2b_kv_cache \
  --results-root my_research/foundation_llamacpp/results/log/vision_prefill_kv_cache_2b_hybrid
```

Expected: return code 0, generated responses exist, phase CSV contains `VisionPrefillCacheHit` and `VisionPrefillCacheRestore`, and cached prompt paths do not record repeated prompt-path `ImagePrefill` rows.
