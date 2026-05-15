# InternVL3 Partial Vision Prefill

This folder contains a Hugging Face InternVL3 research script for reducing the
visual tokens inserted into the LLM prefill. The current method keeps the
attention-island object tokens as original visual embeddings and compresses the
remaining scene/background tokens into spatially connected k-means summaries.

Main files:

```text
my_research/test_python/internvl3_partial_prefill.py
my_research/test_python/tests/test_internvl3_partial_prefill_helpers.py
```

## Current Method

Recommended command:

```bash
python my_research/test_python/internvl3_partial_prefill.py \
  --image my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg \
  --question "Describe this image with concrete visual details." \
  --ratios 10,20,30,40,50,60,70,80,90,100 \
  --importance-source vit-patch-rollout \
  --selection-policy object-budget-background-kmeans \
  --object-budget-percent 90 \
  --max-new-tokens 96
```

The default importance source is already `vit-patch-rollout`. It is CLS-free:
the script removes the CLS row/column from the image encoder self-attention,
runs patch-to-patch attention rollout across vision layers, and downsamples the
result to the `16x16` InternVL visual-token grid.

`object-budget-background-kmeans` interprets each ratio as the final total
visual embedding budget:

```text
final visual embeddings = original object tokens + background summary tokens
```

So `--ratios 30` on a 256-token image means the LLM receives at most
`ceil(256 * 30 / 100) = 77` visual embeddings.

## Budget Rule

Definitions:

```text
N = number of original visual tokens, usually 256 for InternVL3-1B at 448x448
r = ratio percent
o = object_budget_percent, default 90
M = number of tokens inside the attention island
```

Budget calculation:

```text
k_total = ceil(N * r / 100)
k_bg_initial = ceil(k_total * (100 - o) / 100)
k_obj_budget = k_total - k_bg_initial
k_obj = min(k_obj_budget, M)
k_bg = k_total - k_obj
```

If the object budget is larger than the attention island, the unused object
budget is transferred to background summaries. This is intentional. At high
ratios, all object tokens are already preserved, so extra budget should improve
scene/background coverage instead of being wasted.

For the cat sample where `N=256` and the attention island has `M=118` tokens:

| ratio | total budget | object originals | background summaries |
| ---: | ---: | ---: | ---: |
| 10 | 26 | 23 | 3 |
| 20 | 52 | 46 | 6 |
| 30 | 77 | 69 | 8 |
| 40 | 103 | 92 | 11 |
| 50 | 128 | 115 | 13 |
| 60 | 154 | 118 | 36 |
| 70 | 180 | 118 | 62 |
| 80 | 205 | 118 | 87 |
| 90 | 231 | 118 | 113 |
| 100 | 256 | 118 | 138 |

At `100%`, every original visual token is represented: 118 survive as object
tokens and the remaining 138 are represented as background summary tokens. When
the requested background count equals the residual token count, the script uses
singleton regions, so this is equivalent to keeping all original embeddings.

## Algorithm

Scoring path:

```text
1. Capture InternVL image-encoder self-attention from every vision layer.
2. Drop CLS row and CLS column.
3. Average attention heads.
4. Add identity to model residual attention flow.
5. Row-normalize each layer matrix.
6. Multiply matrices across layers.
7. Score each source patch by mean incoming patch-to-patch rollout flow.
8. Average-pool the 32x32 ViT patch map to the 16x16 IMG_CONTEXT grid.
```

Selection path:

```text
1. Normalize the 16x16 visual-token importance map to 0..1.
2. Apply light Gaussian smoothing.
3. Use Otsu thresholding to create an attention-island mask.
4. Remove tiny island fragments and fill tiny holes.
5. Select object tokens from the island by top-k attention score.
6. Treat every non-selected token as residual/background source.
7. Cluster the residual field with spatially connected k-means.
8. Mean-pool each background region into one summary embedding.
9. Rebuild the prompt with object embeddings plus summary embeddings.
```

The source coverage target is always the full original visual grid:

```text
visual_source_coverage_count = 256
visual_source_missing_count = 0
```

This means every original visual token either survives directly as an object
token or contributes to one background summary.

## Important Parameters

| parameter | default | current use |
| --- | ---: | --- |
| `--importance-source` | `vit-patch-rollout` | CLS-free InternVL image-encoder rollout. |
| `--selection-policy` | `topk` | Set to `object-budget-background-kmeans` for the current method. |
| `--ratios` | `10,30,50,70,80,100` | Total final visual embedding budget percent. |
| `--object-budget-percent` | `90` | Percent of each total budget initially reserved for object tokens. |
| `--background-min-region-size` | `2` | Merge background regions smaller than this many source tokens. |
| `--background-spatial-weight` | `1.0` | Weight for normalized `(row, col)` in background k-means. |
| `--background-score-weight` | `0.35` | Weight for attention score in background k-means. |
| `--input-size` | `448` | InternVL image size. |
| `--max-num` | `1` | Number of image tiles; pooled policies currently expect one square tile. |
| `--max-new-tokens` | `64` | Generation length for benchmark responses. |
| `--bf16` / `--no-bf16` | `bf16` | Model dtype switch. |

Legacy background parameters such as `--background-count-policy`,
`--background-fixed-clusters`, and `--background-ratio-percent` are only used by
older comparison policies. They do not drive the current
`object-budget-background-kmeans` method.

## Background K-Means

Background clustering uses features of the form:

```text
[row * background_spatial_weight,
 col * background_spatial_weight,
 normalized_attention_score * background_score_weight]
```

The current budgeted method sets the target cluster count directly:

```text
target_background_regions = k_bg
```

After k-means, labels are post-processed so regions are spatially connected:

```text
1. Split disconnected islands that received the same k-means label.
2. Merge regions smaller than background_min_region_size into nearby regions.
3. If there are too many regions, merge nearby low-cost neighbors.
4. If there are too few regions, split large regions until the target is met.
5. If k_bg equals the number of residual tokens, return singleton regions.
```

Each final region stores its source token indices and contributes one mean
embedding to the LLM prefill.

## Visualization

For the current method:

```text
red cells  = original object/detail visual tokens sent to the LLM
blue cells = source tokens represented by background summaries
bold lines = boundaries between object cells and background regions
```

Useful generated files:

```text
selected_tokens_<ratio>.png
selected_tokens_grid_10_to_100_object90_background_rest_kmeans.png
attention_heatmap_full.png
vit_patch_rollout_attention_side_by_side.png
vit_patch_rollout_attention_downsampled_16_side_by_side.png
```

Latest reference run used while stabilizing the current method:

```text
my_research/test_python/results/partial_prefill_sample_coco_cats_448_20260515_053438/
```

Panel image:

```text
my_research/test_python/results/partial_prefill_sample_coco_cats_448_20260515_053438/selected_tokens_grid_10_to_100_object90_background_rest_kmeans.png
```

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
```

Important `summary.tsv` fields:

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

`selected_tokens.json` contains the full score vector, selected object indices,
background summary source indices, island mask, and coverage metadata.

## Legacy Policies

`topk`

Keeps only original visual tokens by top-k score. This is the simplest baseline
and does not preserve full source coverage.

`object-topk-residual-cluster-pool`

Older budgeted policy. It keeps object/detail top-k tokens and pools every
non-selected token into merged residual regions. Background count is controlled
by `--background-count-policy`.

`object-all-background-kmeans`

Keeps every attention-island token as an original embedding and compresses only
the non-island background. It uses auto-k elbow selection instead of the ratio
budget rule.

`island-background-merge-pool`

Older comparison policy. Keeps selected island tokens and pools only non-island
background tokens.

`island-context-pool`

Older comparison policy. Pools background with a coarse rectangular grid.

## Implementation Notes

InternVL's Hugging Face remote code exposes vision outputs, but its current
`InternAttention.forward()` does not return attention weights. The script
temporarily patches each vision-layer `_naive_attn()` during a no-grad vision
forward pass and captures:

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
  --importance-source vit-patch-rollout \
  --selection-policy object-budget-background-kmeans \
  --object-budget-percent 90
```
