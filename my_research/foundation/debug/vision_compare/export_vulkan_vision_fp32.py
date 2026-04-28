from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass

from my_research.foundation.exporters.xnnpack import _lower_split_program
from my_research.foundation.models.internvl3.vision_encoder.model import (
    load_vision_encoder,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Vulkan vision encoder only.")
    parser.add_argument(
        "--model_path",
        default="/workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf",
    )
    parser.add_argument(
        "--artifact_root",
        default="/workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_vision_fix",
    )
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    vision_encoder = load_vision_encoder(
        args.model_path,
        vulkan_friendly_attention=True,
    ).eval()
    example_inputs = vision_encoder.get_example_inputs()

    with torch.no_grad():
        exported = torch.export.export(vision_encoder, example_inputs, strict=False)

    vulkan_force_fp16 = args.dtype == "fp16"
    edge = _lower_split_program(
        exported,
        backend="vulkan",
        enable_dynamic_shape=True,
        dtype="fp32",
        vulkan_force_fp16=vulkan_force_fp16,
    )
    exec_config = ExecutorchBackendConfig(
        extract_delegate_segments=True,
        passes=[QuantFusionPass()],
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    )
    pte_path = artifact_root / f"vision_encoder_vulkan_{args.dtype}_fix.pte"
    with pte_path.open("wb") as f:
        edge.to_executorch(exec_config).write_to_file(f)

    metadata = {
        "model_path": args.model_path,
        "vision_encoder_pte": str(pte_path),
        "backend": "vulkan",
        "vulkan_force_fp16": vulkan_force_fp16,
        "dtype": args.dtype,
        "vulkan_friendly_attention": True,
        "fix": "contiguous attention tensors",
    }
    (artifact_root / "vision_fp32_export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
