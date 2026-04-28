# InternVL3 Vision Overlay Comparison

This folder checks whether the local Vulkan-friendly InternVL3 vision attention
overlay changes PyTorch-level numerics before export.

Run from the repository root:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python my_research/foundation/debug/vision_compare/compare_pytorch_vision_overlay.py
```

The script compares:

- Original `load_vision_encoder(model_path)`
- Overlay `load_vision_encoder(model_path, vulkan_friendly_attention=True)`

Default output:

- `results/summary.json`

To also save tensors for deeper inspection:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python my_research/foundation/debug/vision_compare/compare_pytorch_vision_overlay.py \
  --save_tensors
```

The first run showed the overlay is numerically close to the original in
PyTorch: `max_abs_diff ~= 1.38e-4`, `mean_abs_diff ~= 1.4e-6`, and cosine
similarity is effectively `1.0`.

To compare the export-like variants:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python my_research/foundation/debug/vision_compare/analyze_vision_export_differences.py
```

This writes `results/vision_export_differences.json`. The first run showed:

- Original fp16 PyTorch vs overlay fp32-schema PyTorch remains close:
  `max_abs_diff ~= 8.6e-2`, `mean_abs_diff ~= 1.4e-3`, cosine `~= 0.999995`.
- Original fp16 vs overlay fp16 is also close: cosine `~= 0.999985`.
- XNNPACK export-like graph starts from original attention and fp16 input, while
  Vulkan export-like graph starts from the bmm/softmax overlay and float input
  schema with `force_fp16`.
- After partitioning, XNNPACK leaves many ops portable; Vulkan delegates almost
  the whole graph, with only one `expand_copy` portable.

To compare actual Android runner vision outputs, run both backends with
`--save_log`. The runner writes:

- `vision_output_stats.csv`
- `vision_output_0000_f32.bin`

Then compare the dumps:

```bash
PYTHONPATH=/workspace/streamingvlm:/workspace/streamingvlm/executorch \
python my_research/foundation/debug/vision_compare/compare_dumped_vision_outputs.py \
  --reference my_research/foundation/results/log/xnnpack/internvl3_hybrid_xnnpack_vision_vulkan_embedding_decoder_fp16_1k/vision_output_0000_f32.bin \
  --candidate my_research/foundation/results/log/vulkan/internvl3_vulkan_1b_1k_fp16/vision_output_0000_f32.bin \
  --output my_research/foundation/debug/vision_compare/results/xnnpack_vs_vulkan_android_vision_diff.json
```

The first Android dump comparison showed the Vulkan vision output is all NaN:

- XNNPACK vision: dtype `Half`, shape `[1, 256, 896]`, `nan_count=0`
- Vulkan vision: dtype `Float`, shape `[1, 256, 896]`, `nan_count=229376`
