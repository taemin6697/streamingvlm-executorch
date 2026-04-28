from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import transforms
from transformers.image_utils import load_image

from my_research.foundation.models.internvl3.vision_encoder.model import (
    load_vision_encoder,
)


def _preprocess_image(image_source: str) -> torch.Tensor:
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
    return transform(image).unsqueeze(0).float()


def _stats(tensor: torch.Tensor) -> dict[str, float | list[int] | str]:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch InternVL3 vision output before/after Vulkan attention overlay."
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
    parser.add_argument("--save_tensors", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_values = _preprocess_image(args.image)
    base = load_vision_encoder(args.model_path).eval()
    overlay = load_vision_encoder(
        args.model_path,
        vulkan_friendly_attention=True,
    ).eval()

    with torch.no_grad():
        base_out = base(pixel_values)
        overlay_out = overlay(pixel_values)

    diff = (base_out - overlay_out).float()
    cosine = torch.nn.functional.cosine_similarity(
        base_out.float().flatten(),
        overlay_out.float().flatten(),
        dim=0,
    )
    summary = {
        "model_path": args.model_path,
        "image": args.image,
        "base": _stats(base_out),
        "overlay": _stats(overlay_out),
        "diff": {
            "max_abs": float(diff.abs().max().item()),
            "mean_abs": float(diff.abs().mean().item()),
            "l2_norm": float(diff.norm().item()),
            "cosine": float(cosine.item()),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved summary: {summary_path}")

    if args.save_tensors:
        torch.save(
            {
                "pixel_values": pixel_values.cpu(),
                "base_out": base_out.cpu(),
                "overlay_out": overlay_out.cpu(),
                "diff": diff.cpu(),
            },
            output_dir / "vision_compare_tensors.pt",
        )
        print(f"Saved tensors: {output_dir / 'vision_compare_tensors.pt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
