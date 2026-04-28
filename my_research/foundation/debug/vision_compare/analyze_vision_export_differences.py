from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import EdgeCompileConfig, to_edge
from torchvision import transforms
from transformers.image_utils import load_image

from my_research.foundation.models.internvl3.vision_encoder.model import (
    load_vision_encoder,
)


def _preprocess_image(image_source: str, dtype: torch.dtype) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    image = load_image(image_source)
    return transform(image).unsqueeze(0).to(dtype)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    t = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "l2_norm": float(t.norm().item()),
        "nan_count": int(torch.isnan(t).sum().item()),
        "inf_count": int(torch.isinf(t).sum().item()),
    }


def _diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().float()
    cand = candidate.detach().float()
    diff = ref - cand
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "l2_norm": float(diff.norm().item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(ref.flatten(), cand.flatten(), dim=0).item()
        ),
    }


def _graph_op_counts(model: torch.nn.Module, example_input: torch.Tensor) -> dict[str, object]:
    with torch.no_grad():
        exported = torch.export.export(model, (example_input,), strict=False)
    ops = [
        str(node.target)
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    return {
        "input_dtype": str(example_input.dtype),
        "op_counts": dict(sorted(Counter(ops).items())),
        "total_call_function_ops": len(ops),
    }


def _partition_counts(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    backend: str,
) -> dict[str, object]:
    with torch.no_grad():
        exported = torch.export.export(model, (example_input,), strict=False)
    edge = to_edge(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    if backend == "xnnpack":
        partitioner = XnnpackPartitioner()
    elif backend == "vulkan":
        partitioner = VulkanPartitioner({"require_dynamic_shapes": True, "force_fp16": True})
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    partition_result = partitioner.partition(edge.exported_program())
    tagged = partition_result.tagged_exported_program
    delegated = 0
    portable = 0
    delegated_ops: Counter[str] = Counter()
    portable_ops: Counter[str] = Counter()
    for node in tagged.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if "delegation_tag" in node.meta:
            delegated += 1
            delegated_ops[target] += 1
        else:
            portable += 1
            portable_ops[target] += 1
    return {
        "backend": backend,
        "delegated_call_function_ops": delegated,
        "portable_call_function_ops": portable,
        "delegated_ops": dict(sorted(delegated_ops.items())),
        "portable_ops": dict(sorted(portable_ops.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare original vs Vulkan-overlay InternVL3 vision export differences."
    )
    parser.add_argument(
        "--model_path",
        default="/workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf",
    )
    parser.add_argument(
        "--image",
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
    )
    parser.add_argument(
        "--output_dir",
        default="/workspace/streamingvlm/my_research/foundation/debug/vision_compare/results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_float = _preprocess_image(args.image, torch.float32)
    x_half = x_float.to(torch.float16)

    base_fp32 = load_vision_encoder(args.model_path).eval()
    overlay_fp32 = load_vision_encoder(
        args.model_path,
        vulkan_friendly_attention=True,
    ).eval()
    base_fp16 = load_vision_encoder(args.model_path).eval().to(torch.float16)
    overlay_fp16 = (
        load_vision_encoder(args.model_path, vulkan_friendly_attention=True)
        .eval()
        .to(torch.float16)
    )

    with torch.no_grad():
        outputs = {
            "base_fp32": base_fp32(x_float),
            "overlay_fp32": overlay_fp32(x_float),
            "base_fp16": base_fp16(x_half),
            "overlay_fp16": overlay_fp16(x_half),
        }

    summary = {
        "model_path": args.model_path,
        "image": args.image,
        "outputs": {name: _tensor_stats(value) for name, value in outputs.items()},
        "diffs_vs_base_fp32": {
            name: _diff_stats(outputs["base_fp32"], value)
            for name, value in outputs.items()
            if name != "base_fp32"
        },
        "diffs_between_export_like_variants": {
            "xnnpack_base_fp16_vs_vulkan_overlay_fp32_schema": _diff_stats(
                outputs["base_fp16"], outputs["overlay_fp32"]
            ),
            "base_fp16_vs_overlay_fp16": _diff_stats(
                outputs["base_fp16"], outputs["overlay_fp16"]
            ),
        },
        "graphs": {
            "xnnpack_export_like_base_fp16": _graph_op_counts(base_fp16, x_half),
            "vulkan_export_like_overlay_fp32_schema": _graph_op_counts(
                overlay_fp32, x_float
            ),
        },
        "partitions": {
            "xnnpack_export_like_base_fp16": _partition_counts(
                base_fp16, x_half, "xnnpack"
            ),
            "vulkan_export_like_overlay_fp32_schema": _partition_counts(
                overlay_fp32, x_float, "vulkan"
            ),
        },
    }

    summary_path = output_dir / "vision_export_differences.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
