# InternVL3 Partial Vision Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an InternVL3 HF experiment that prunes visual tokens by reference attention score, benchmarks latency at retained-token ratios, and writes attention visualizations.

**Architecture:** Keep the experiment self-contained in `my_research/test_python/internvl3_partial_prefill.py`. Separate pure helper functions from model execution so tests can validate selection, attention aggregation, prompt sizing, and heatmap grid behavior without loading InternVL3.

**Tech Stack:** Python 3, PyTorch, Transformers 4.x remote code, Pillow, torchvision transforms, pytest.

---

### Task 1: Pure Helper Tests

**Files:**
- Create: `my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py`
- Create: `my_research/test_python/internvl3_partial_prefill.py`

- [ ] **Step 1: Write tests for ratio parsing and top-k selection**

```python
import torch

from my_research.test_python.internvl3_partial_prefill import (
    parse_ratios,
    select_topk_indices,
)


def test_parse_ratios_accepts_percent_values():
    assert parse_ratios("10,30,100") == [10, 30, 100]


def test_select_topk_indices_preserves_original_order_after_ranking():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])

    selected = select_topk_indices(scores, 50)

    assert selected.tolist() == [1, 3]
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py -q`

Expected: FAIL because `internvl3_partial_prefill.py` does not exist yet.

### Task 2: Attention and Grid Tests

**Files:**
- Modify: `my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py`
- Modify: `my_research/test_python/internvl3_partial_prefill.py`

- [ ] **Step 1: Add tests for attention aggregation and grid sizing**

```python
import pytest
import torch

from my_research.test_python.internvl3_partial_prefill import (
    aggregate_visual_attention,
    square_grid_size,
)


def test_aggregate_visual_attention_uses_text_queries_only():
    attention = torch.zeros(1, 1, 5, 5)
    attention[0, 0, 3, 1] = 0.7
    attention[0, 0, 4, 2] = 0.3

    scores = aggregate_visual_attention(
        attentions=(attention,),
        visual_positions=torch.tensor([1, 2]),
        text_positions=torch.tensor([3, 4]),
    )

    assert torch.allclose(scores, torch.tensor([0.35, 0.15]))


def test_square_grid_size_returns_side_for_square_token_count():
    assert square_grid_size(256) == 16


def test_square_grid_size_rejects_non_square_token_count():
    with pytest.raises(ValueError):
        square_grid_size(250)
```

- [ ] **Step 2: Run tests and verify failures**

Run: `pytest my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py -q`

Expected: FAIL because helper functions are not implemented.

### Task 3: Minimal Helper Implementation

**Files:**
- Create: `my_research/test_python/internvl3_partial_prefill.py`

- [ ] **Step 1: Implement pure helpers**

Implement:

```python
def parse_ratios(text: str) -> list[int]
def select_topk_indices(scores: torch.Tensor, ratio_percent: int) -> torch.Tensor
def aggregate_visual_attention(attentions, visual_positions, text_positions) -> torch.Tensor
def square_grid_size(token_count: int) -> int
```

- [ ] **Step 2: Run helper tests**

Run: `pytest my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py -q`

Expected: PASS.

### Task 4: InternVL3 Execution Path

**Files:**
- Modify: `my_research/test_python/internvl3_partial_prefill.py`

- [ ] **Step 1: Add CLI and image preprocessing**

Add args for `--image`, `--model`, `--question`, `--ratios`, `--results-root`, `--max-new-tokens`, `--bf16`, `--no-bf16`, and `--dry-run`.

- [ ] **Step 2: Reproduce InternVL prompt expansion**

Use the model remote conversation template to build the expanded prompt and replace `<image>` with `<img><IMG_CONTEXT>*N</img>`.

- [ ] **Step 3: Build injected embeddings**

Tokenize the expanded prompt, obtain `inputs_embeds` from the LLM embedding layer, run `model.extract_feature(pixel_values)`, and replace `<IMG_CONTEXT>` positions with visual embeddings.

- [ ] **Step 4: Run reference prefill**

Call the language model forward with `inputs_embeds`, `attention_mask`, `use_cache=True`, `output_attentions=True`, and `return_dict=True`. Aggregate attention scores over visual positions.

- [ ] **Step 5: Run each partial ratio**

For each ratio, select top-k visual embeddings, rebuild an expanded prompt with that many `<IMG_CONTEXT>` tokens, run generation with `inputs_embeds`, record latency, and decode the output.

### Task 5: Visualization and Reports

**Files:**
- Modify: `my_research/test_python/internvl3_partial_prefill.py`

- [ ] **Step 1: Write heatmap overlays**

Use Pillow to resize normalized attention grids to the source image size and overlay a red heatmap. Write `attention_heatmap_full.png`.

- [ ] **Step 2: Write selected-token masks**

For each ratio, convert selected token indices to a grid mask and overlay blue/transparent cells. Write `selected_tokens_<ratio>.png`.

- [ ] **Step 3: Write machine-readable and HTML summaries**

Write `summary.json`, `summary.tsv`, `selected_tokens.json`, `response_<ratio>.txt`, and `summary.html`.

### Task 6: Verification

**Files:**
- Read: `my_research/test_python/internvl3_partial_prefill.py`
- Read: `my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py`

- [ ] **Step 1: Run helper tests**

Run: `pytest my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py -q`

Expected: PASS.

- [ ] **Step 2: Run dry-run CLI**

Run: `python my_research/test_python/internvl3_partial_prefill.py --dry-run --image my_research/foundation_llamacpp/sample_images/surveil_8.jpg`

Expected: exit 0 and print the planned ratios without model loading. If the sample jpg does not exist, use any local image path from the repository or create no file changes and report the missing smoke asset.

- [ ] **Step 3: Optional GPU benchmark**

Run the script with a real image and GPU if CUDA and model weights are available:

```bash
python my_research/test_python/internvl3_partial_prefill.py \
  --image my_research/foundation_llamacpp/sample_images/surveil_8_frame.jpg \
  --question "Describe this image briefly." \
  --ratios 10,30,50,70,80,100 \
  --max-new-tokens 64
```

Expected: outputs are written under `my_research/test_python/results/partial_prefill_*`.
