# InternVL3 Partial Vision Prefill Handoff

This folder contains a Hugging Face InternVL3 research script for partial
visual-token prefill. The goal is to reduce the number of visual embeddings
inserted into the LLM prompt while keeping enough object detail and background
context for useful generation.

Main script:

```text
my_research/test_python/internvl3_partial_prefill.py
```

Helper tests:

```text
my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py
```

## Current Recommendation

Use InternVL's image-encoder attention, not LLM text-to-image attention:

```bash
--importance-source vit-patch-rollout
```

This is the default. It is CLS-free: the script removes the CLS row/column,
runs patch-to-patch attention rollout across vision layers, then averages
incoming source-patch flow.

For streaming/mobile-style compression experiments, use:

```bash
--selection-policy object-topk-residual-cluster-pool
```

This keeps high-importance object/detail tokens as original embeddings and
compresses every non-selected residual token into background summary regions.
The visualization has no uncovered cells: red cells are original object tokens,
blue cells are pooled residual/background summaries.

## Quick Commands

Plain top-k baseline:

```bash
python my_research/test_python/internvl3_partial_prefill.py \
  --image my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
  --question "Describe this image with concrete visual details." \
  --ratios 10,30,100 \
  --max-new-tokens 96
```

Object tokens plus exactly 5 background summary tokens:

```bash
python my_research/test_python/internvl3_partial_prefill.py \
  --image my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
  --question "Describe this image with concrete visual details." \
  --ratios 10,30,100 \
  --selection-policy object-topk-residual-cluster-pool \
  --background-count-policy fixed \
  --background-fixed-clusters 5 \
  --max-new-tokens 96
```

Object tokens plus automatic background summary count:

```bash
python my_research/test_python/internvl3_partial_prefill.py \
  --image my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
  --question "Describe this image with concrete visual details." \
  --ratios 10,30,100 \
  --selection-policy object-topk-residual-cluster-pool \
  --background-count-policy cc \
  --max-new-tokens 96
```

## Core Algorithm

InternVL3-1B uses a `448x448` image by default. Its vision patch grid is
`32x32` because the patch size is 14. The LLM receives `256` visual embeddings,
arranged as a `16x16` visual-token grid.

The recommended scoring path is:

```text
1. Capture InternVL image-encoder self-attention from every vision layer.
2. Remove CLS row and CLS column.
3. Average attention heads.
4. Add identity to model residual attention flow.
5. Row-normalize each layer matrix.
6. Multiply matrices across layers.
7. Score each source patch by mean incoming patch-to-patch rollout flow.
8. Average-pool the 32x32 patch scores to the 16x16 IMG_CONTEXT grid.
```

The recommended selection path is:

```text
1. Normalize the 16x16 visual-token importance map.
2. Apply light Gaussian smoothing.
3. Use Otsu thresholding to split high-attention islands from context.
4. Select object/detail tokens from the islands by top-k score.
5. Treat every non-selected token as residual/background.
6. Merge nearby residual tokens into background regions.
7. Average each region's visual embeddings into one summary embedding.
8. Insert object embeddings plus background summary embeddings into the LLM prompt.
```

## Background Count Policies

Fixed percent of the full visual-token grid:

```bash
--background-count-policy fixed \
--background-ratio-percent 5
```

For InternVL3-1B, 5% of 256 means 13 background summary tokens.

Exact fixed count:

```bash
--background-count-policy fixed \
--background-fixed-clusters 5
```

For a 30% budget, this gives:

```text
77 total retained visual embeddings
72 original object/detail embeddings
5 pooled background summary embeddings
```

Automatic connected-component count:

```bash
--background-count-policy cc
```

The automatic policy first selects object tokens, then computes residual
connected components. It estimates:

```text
k_bg = sum_components ceil(component_size / background_target_tokens_per_cluster)
```

Then it clamps the value:

```text
background_min_clusters <= k_bg <= min(background_max_clusters, background_max_fraction * total_budget)
```

Default automatic settings:

```text
background_target_tokens_per_cluster = 12
background_min_clusters = 4
background_max_clusters = 16
background_max_fraction = 0.25
```

On the cat sample, automatic `cc` selected 6 background summaries for 10% and
16 background summaries for 30%. Qualitatively, 30% auto was better than a
fixed 5-background-token run because it preserved more scene layout/context.

## Selection Policies

```bash
--selection-policy topk
```

Keeps only original visual tokens by top-k score. This is the simplest baseline.

```bash
--selection-policy object-topk-residual-cluster-pool
```

Recommended. Keeps object/detail top-k tokens and pools every non-object token
into residual background clusters.

```bash
--selection-policy island-background-merge-pool
```

Older comparison. Keeps object/detail tokens from attention islands and pools
only non-island background tokens.

```bash
--selection-policy island-context-pool
```

Older comparison. Pools background using a coarse rectangular grid.

## Output Files

Each run is saved under:

```text
my_research/test_python/results/partial_prefill_<image_stem>_<UTC_TIMESTAMP>/
```

Important files:

```text
summary.html
summary.tsv
summary.json
meta.json
selected_tokens.json
response_<ratio>.txt
selected_tokens_<ratio>.png
attention_heatmap_full.png
attention_100_side_by_side_style.png
vit_patch_rollout_attention_side_by_side.png
vit_patch_rollout_attention_downsampled_16_side_by_side.png
```

`summary.tsv` includes:

```text
ratio_percent
retained_visual_tokens
object_original_token_count
background_summary_token_count
visual_source_coverage_count
visual_source_missing_count
prefill_ms
prefill_reduction_percent
generation_latency_ms
response
```

`selected_tokens.json` includes the full score vector, selected object indices,
background summary source indices, island mask, and source coverage metadata.

## Visualization Meaning

For `object-topk-residual-cluster-pool`:

```text
red cells  = original object/detail visual tokens sent to the LLM
blue cells = residual/background source tokens pooled into summary embeddings
bold lines = boundaries between object tokens and background clusters
```

If `visual_source_coverage_count` is 256 and `visual_source_missing_count` is
0, every original visual token either survived as an object token or contributed
to a background summary.

## Implementation Notes

InternVL's remote Hugging Face code exposes vision outputs, but its current
`InternAttention.forward()` does not return attention weights. The script
temporarily patches each vision-layer `_naive_attn()` method during a no-grad
vision forward pass and captures:

```text
attention = softmax((q * scale) @ k.T)
```

The patch is restored immediately after capture.

The partial-prefill path does not call `model.generate(pixel_values=...)`.
Instead it:

```text
1. extracts visual embeddings once with model.extract_feature(pixel_values)
2. selects or pools visual embeddings
3. rebuilds the prompt with fewer IMG_CONTEXT tokens
4. injects embeddings into inputs_embeds
5. calls model.language_model.generate(inputs_embeds=..., attention_mask=...)
```

This avoids re-encoding the full image during partial generation.

## Known Limits

- Pooled background policies currently require one square image tile.
- GPU timing is noisy for these small prefill lengths; use several images and
  repeated runs before making speed claims.
- Very low ratios such as 1%, 5%, and sometimes 10% can miss secondary objects.
- `llm-attention` is kept for debugging but showed boundary/corner artifacts on
  the cat sample.

## Verification

Run helper tests:

```bash
pytest my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py -q
```

Compile check:

```bash
python -m py_compile my_research/test_python/internvl3_partial_prefill.py
```

Dry-run without loading the model:

```bash
python my_research/test_python/internvl3_partial_prefill.py \
  --dry-run \
  --ratios 10,30,100 \
  --selection-policy object-topk-residual-cluster-pool \
  --background-count-policy cc
```
