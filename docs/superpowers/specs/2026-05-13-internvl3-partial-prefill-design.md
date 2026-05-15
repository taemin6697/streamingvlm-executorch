# InternVL3 Partial Vision Prefill Design

## Goal

Build an English CLI experiment named `internvl3_partial_prefill.py` that measures how much decode latency can be reduced when InternVL3 receives only the most important subset of its visual tokens. The experiment should also visualize which image regions the reference run considered important.

## Scope

The script targets Hugging Face `OpenGVLab/InternVL3-1B` remote code through `transformers`. It is an analysis and benchmarking tool, not a model patch. It should support a single image as the primary accuracy/visualization path and may also accept a short video path by reusing the existing frame sampling helpers.

The retained visual token ratios are `10, 30, 50, 70, 80, 100` percent by default.

## InternVL3 Model Path

InternVL3 expands each `<image>` placeholder into:

```text
<img><IMG_CONTEXT>...<IMG_CONTEXT></img>
```

The number of context tokens is `model.num_image_token * num_patches`. In InternVL3-1B with the default 448 input size, 14 patch size, and 0.5 downsample ratio, this is 256 visual tokens per image tile. The script must not hard-code 256 or 16x16. It must derive token count from `model.num_image_token` and derive the visualization grid from `sqrt(model.num_image_token)` when the token count is square.

For each run, the script will:

1. Build the same chat-template prompt as `InternVLChatModel.chat`.
2. Run `model.extract_feature(pixel_values)` to obtain visual embeddings.
3. Inject those embeddings into the `<IMG_CONTEXT>` token positions in `inputs_embeds`.
4. Run one reference prefill with `output_attentions=True`.
5. Score each visual token by the attention it receives from text/query tokens.
6. Rebuild prompts with fewer `<IMG_CONTEXT>` positions and inject only selected visual embeddings.
7. Run generation for each retained ratio and record latency plus decoded response.

## Importance Score

The default importance score is **CLS-excluded patch-to-patch attention rollout** from InternVL's image encoder. The script captures image-encoder self-attention, removes the CLS row/column, averages heads, adds identity for residual flow, row-normalizes, multiplies patch attention matrices across layers, then scores each source patch by its average contribution to final patch tokens. For InternVL3-1B this produces a `32x32` patch map, which is average-pooled to the `16x16` LLM visual-token grid.

The older LLM text-to-visual attention score remains available as a comparison/debug mode, but it is not recommended as the default because it can over-rank visual-block boundary tokens. The older CLS rollout score remains available as `vit-rollout` for comparison.

For CLS-free comparison, the script also provides patch-only image-encoder scores:

- `vit-patch-incoming`: average incoming patch-to-patch attention after excluding CLS.
- `vit-patch-rollout`: CLS-excluded patch-to-patch attention rollout. It propagates patch attention across layers with residual identity, then scores each source patch by averaging its contribution across final patch tokens. This is the preferred baseline for future CLS-free vision encoder experiments.

## Partial Prefill Semantics

The experiment physically reduces the visual token count in the prompt and input embedding sequence before prefill. It does not zero out unused tokens. This better approximates the target mobile benefit because the language model sees a shorter prefill sequence.

Selected visual embeddings must stay in their original order after top-k selection so positional order remains monotonic. The selected indices and scores are saved for analysis.

The script also supports object/detail plus background-summary policies. The current preferred variant is `--selection-policy object-topk-residual-cluster-pool`: high-attention island top-k tokens keep original visual embeddings, while every non-selected residual token is assigned to merged adjacent graph regions until the requested background summary count is reached. This removes uncovered cells from the selection visualization. For example, a `50%` run with `--background-ratio-percent 5` keeps `45%` original object/detail tokens and `5%` merged residual/background summary tokens. The older `island-background-merge-pool` and `island-context-pool` policies remain available for comparison.

For automatic background count selection, `--background-count-policy cc` derives the background summary count from residual connected components. It sums `ceil(component_size / target_tokens_per_cluster)` over components, then clamps the result by minimum clusters, maximum clusters, and a maximum fraction of the total retained budget.

## Visualization

The script will write PNG visualizations under the run directory:

- `attention_heatmap_full.png`: continuous attention heatmap over the image tile.
- `selected_tokens_10.png`, `selected_tokens_30.png`, etc.: top-k binary masks overlaid on the image tile.
- `summary.html`: a compact report with latency, token counts, responses, and generated visualizations.

For the default InternVL3-1B configuration, the 256 visual tokens map to a 16x16 grid. This is derived from the model configuration. If the grid is not square, the script still writes score tables and skips image-grid overlays with a warning.

## Outputs

Each run directory contains:

- `summary.json`
- `summary.tsv`
- `selected_tokens.json`
- `response_<ratio>.txt`
- visualization PNG files
- `summary.html`

## Testing

Tests should avoid loading the model. They cover pure behavior:

- building top-k masks from attention scores,
- preserving original index order after top-k selection,
- deriving square heatmap grid sizes,
- aggregating attentions from text queries to visual positions,
- producing prompt token counts for each retained ratio.

A `--dry-run` path should exercise the CLI output planning without model loading.
