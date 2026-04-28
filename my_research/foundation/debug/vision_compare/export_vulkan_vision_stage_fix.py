from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass

from my_research.foundation.exporters.xnnpack import _lower_split_program
from my_research.foundation.models.internvl3.vision_encoder.model import load_vision_encoder


class InternVL3VisionStage(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, layer_idx: int, stage: str):
        super().__init__()
        self.encoder = encoder
        self.layer_idx = layer_idx
        self.stage = stage

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encoder.vision_tower.layernorm(hidden_states)
        vision_features = hidden_states[:, 1:, :]
        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)
        vision_features = self.encoder.pixel_shuffle(
            vision_features,
            scale_factor=self.encoder.config.downsample_ratio,
        )
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])
        return self.encoder.multi_modal_projector(vision_features)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.encoder.vision_tower.embeddings(pixel_values)
        for idx, layer in enumerate(self.encoder.vision_tower.encoder.layer):
            if idx < self.layer_idx:
                hidden_states = layer(hidden_states)
                continue
            if idx > self.layer_idx:
                break

            residual = hidden_states
            hidden_states_norm = layer.layernorm_before(hidden_states)
            attn_output, _ = layer.attention(hidden_states_norm)
            hidden_states = residual + layer.dropout(attn_output)
            if self.stage == "attention":
                break

            residual = hidden_states
            hidden_states = layer.layernorm_after(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + layer.dropout(hidden_states)
            break
        return self._project(hidden_states)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Vulkan vision stage model.")
    parser.add_argument(
        "--model_path",
        default="/workspace/streamingvlm/my_research/foundation/results/model/hf/InternVL3-1B-hf",
    )
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--stage", choices=["attention", "full"], required=True)
    parser.add_argument(
        "--artifact_root",
        default="/workspace/streamingvlm/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_1k_vision_stage_fix",
    )
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    encoder = load_vision_encoder(args.model_path, vulkan_friendly_attention=True).eval()
    model = InternVL3VisionStage(encoder, args.layer_idx, args.stage).eval()
    example_inputs = encoder.get_example_inputs()

    with torch.no_grad():
        exported = torch.export.export(model, example_inputs, strict=False)

    edge = _lower_split_program(
        exported,
        backend="vulkan",
        enable_dynamic_shape=True,
        dtype="fp32",
        vulkan_force_fp16=args.dtype == "fp16",
    )
    exec_config = ExecutorchBackendConfig(
        extract_delegate_segments=True,
        passes=[QuantFusionPass()],
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    )
    name = f"vision_encoder_vulkan_layer{args.layer_idx}_{args.stage}_{args.dtype}_fix.pte"
    pte_path = artifact_root / name
    with pte_path.open("wb") as f:
        edge.to_executorch(exec_config).write_to_file(f)

    metadata = {
        "backend": "vulkan",
        "dtype": args.dtype,
        "layer_idx": args.layer_idx,
        "stage": args.stage,
        "vision_encoder_pte": str(pte_path),
    }
    (artifact_root / f"layer{args.layer_idx}_{args.stage}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
