#!/usr/bin/env python3

"""Export Qwen2.5-VL vision encoder to ExecuTorch QNN.

The exported module takes one fixed-size normalized CHW image tensor and embeds
the matching Qwen `image_grid_thw` internally. Qwen's dynamic-resolution
behavior is represented by choosing a fixed export resolution or token count
when the PTE is produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchBackendType
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from my_research.foundation.models.qwen2_5_vl.vision_encoder.model import (
    describe_vision_tower_output,
    load_vision_tower,
)


_DEFAULT_HF_MODELS = {
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}

_QWEN_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
_QWEN_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def _parse_image_size(values: Optional[list[int]]) -> Optional[tuple[int, int]]:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--image-size expects HEIGHT WIDTH.")
    return int(values[0]), int(values[1])


def _default_artifact_root(model_name: str, soc_model: str, image_tokens: int) -> Path:
    return (
        Path("my_research/foundation_llamacpp/results/vision_models")
        / f"{model_name}_vision_encoder_qnn_{image_tokens}tok_{soc_model.lower()}"
    )


def _load_calibration_data(
    calibration_sources: Optional[List[str]],
    *,
    image_height: int,
    image_width: int,
    num_samples: int,
) -> list[tuple[torch.Tensor]]:
    samples: list[tuple[torch.Tensor]] = []
    if calibration_sources:
        try:
            from torchvision import transforms
            from transformers.image_utils import load_image

            transform = transforms.Compose(
                [
                    transforms.Resize((image_height, image_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_QWEN_IMAGE_MEAN, std=_QWEN_IMAGE_STD),
                ]
            )
            for src in calibration_sources[:num_samples]:
                image = load_image(src)
                samples.append((transform(image).unsqueeze(0),))
        except Exception as exc:
            print(f"[qwen2.5-vl-export-qnn] Calibration image load failed ({exc}); using random inputs.")
            samples = []

    if not samples:
        for _ in range(num_samples):
            samples.append((torch.randn(1, 3, image_height, image_width, dtype=torch.float32),))
    return samples


def _write_metadata(
    path: Path,
    *,
    model_name: str,
    model_path: str,
    soc_model: str,
    quant: str,
    calibration_images: Optional[list[str]],
    output_summary: dict[str, int | list[int]],
) -> None:
    metadata = {
        "model_name": model_name,
        "model_path": model_path,
        "artifact_type": "qwen2_5_vl_vision_encoder_qnn",
        "backend": "qnn",
        "soc_model": soc_model,
        "quantization": quant,
        "calibration_images": calibration_images or [],
        "projector_included": False,
        "patch_merger_included": False,
        "projector_note": "Qwen2.5-VL visual.patch_merger is excluded; outputs are pure pre-merger vision block features.",
        "output_name": "vision_encoder_qnn.pte",
        "input_format": "normalized_chw_float32_qwen2_vl",
        "image_mean": _QWEN_IMAGE_MEAN,
        "image_std": _QWEN_IMAGE_STD,
        "output": output_summary,
        "token_count_check": {
            "expected_tokens": output_summary["expected_tokens"],
            "actual_tokens": output_summary["num_tokens"],
            "passed": output_summary["expected_tokens"] == output_summary["num_tokens"],
        },
        "notes": (
            "Qwen2.5-VL supports dynamic image resolution in the HF processor, but this QNN PTE "
            "is exported for one fixed image size/grid. Re-export with --image-size or --image-tokens "
            "to target another visual token count."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main(
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    artifact_root: Optional[str] = None,
    encoder_weights: Optional[str] = None,
    soc_model: Optional[str] = None,
    quant: Optional[str] = None,
    image_size: Optional[tuple[int, int]] = None,
    image_tokens: Optional[int] = None,
    calibration_images: Optional[List[str]] = None,
    calibration_num: Optional[int] = None,
    shape_only: bool = False,
    generate_etrecord: bool = False,
) -> None:
    parser = argparse.ArgumentParser(description="Export Qwen2.5-VL vision encoder to QNN")
    parser.add_argument(
        "--model-name",
        choices=sorted(_DEFAULT_HF_MODELS),
        default="qwen2_5_vl_3b",
        help="Named Qwen2.5-VL model used for defaults.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="HuggingFace model id or local path. Defaults from --model-name.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Output directory. Default: my_research/foundation_llamacpp/results/vision_models/<model>_vision_encoder_qnn_<tokens>tok_<soc>",
    )
    parser.add_argument("--encoder-weights", default=None)
    parser.add_argument("--soc-model", default="SM8750")
    parser.add_argument(
        "--quant",
        choices=("fp16", "16a8w"),
        default="16a8w",
        help="QNN quantization mode.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Fixed export image size after preprocessing. Must be divisible by patch_size * spatial_merge_size.",
    )
    parser.add_argument(
        "--image-tokens",
        type=int,
        default=256,
        help="Expected visual token count. If --image-size is omitted, infer the closest-to-square fixed size.",
    )
    parser.add_argument(
        "--calibration-images",
        nargs="+",
        default=None,
        help="Calibration image paths/URLs for 16a8w PTQ. Defaults to random example input if omitted.",
    )
    parser.add_argument("--calibration-num", type=int, default=8)
    parser.add_argument("--shape-only", action="store_true")
    parser.add_argument("--generate-etrecord", action="store_true")
    args = parser.parse_args()

    if model_name is not None:
        args.model_name = model_name
    if model_path is not None:
        args.model_path = model_path
    if artifact_root is not None:
        args.artifact_root = artifact_root
    if encoder_weights is not None:
        args.encoder_weights = encoder_weights
    if soc_model is not None:
        args.soc_model = soc_model
    if quant is not None:
        args.quant = quant
    if image_size is not None:
        args.image_size = list(image_size)
    if image_tokens is not None:
        args.image_tokens = image_tokens
    if calibration_images is not None:
        args.calibration_images = calibration_images
    if calibration_num is not None:
        args.calibration_num = calibration_num
    if shape_only:
        args.shape_only = True
    if generate_etrecord:
        args.generate_etrecord = True

    resolved_model_path = args.model_path or _DEFAULT_HF_MODELS[args.model_name]
    resolved_image_size = _parse_image_size(args.image_size)

    print(f"[qwen2.5-vl-export-qnn] Loading {resolved_model_path}...")
    model = load_vision_tower(
        resolved_model_path,
        image_size=resolved_image_size,
        image_tokens=args.image_tokens,
        encoder_weights=args.encoder_weights,
    )
    output_summary = describe_vision_tower_output(model)
    print(f"[qwen2.5-vl-export-qnn] Output summary: {output_summary}")

    token_count = int(output_summary["num_tokens"])
    artifact_dir = (
        Path(args.artifact_root)
        if args.artifact_root
        else _default_artifact_root(args.model_name, args.soc_model, token_count)
    ).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _write_metadata(
        artifact_dir / "vision_encoder_qnn_metadata.json",
        model_name=args.model_name,
        model_path=resolved_model_path,
        soc_model=args.soc_model,
        quant=args.quant,
        calibration_images=args.calibration_images,
        output_summary=output_summary,
    )
    if args.shape_only:
        print(f"[qwen2.5-vl-export-qnn] Shape-only metadata saved: {artifact_dir}")
        return

    example_inputs = model.get_example_inputs()
    if args.quant == "16a8w":
        print("[qwen2.5-vl-export-qnn] Applying QNN 16a8w PTQ...")
        with torch.no_grad():
            calibration_data = _load_calibration_data(
                args.calibration_images,
                image_height=int(output_summary["image_height"]),
                image_width=int(output_summary["image_width"]),
                num_samples=args.calibration_num,
            )
            example_inputs = calibration_data[0]
            exported = torch.export.export(model, example_inputs, strict=False).module()
            quantizer = make_quantizer(
                quant_dtype=QuantDtype.use_16a8w,
                per_channel_linear=True,
                backend=QnnExecuTorchBackendType.kHtpBackend,
                soc_model=args.soc_model,
            )
            prepared = prepare_pt2e(exported, quantizer)
            for calibration_input in calibration_data:
                prepared(*calibration_input)
            model = convert_pt2e(prepared)

    backend_options = generate_htp_compiler_spec(use_fp16=args.quant == "fp16")
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[args.soc_model],
        backend_options=backend_options,
        shared_buffer=True,
    )
    delegated_program = to_edge_transform_and_lower_to_qnn(
        model,
        example_inputs,
        compile_spec,
        generate_etrecord=args.generate_etrecord,
    )
    exec_prog = delegated_program.to_executorch(ExecutorchBackendConfig())
    if args.generate_etrecord:
        exec_prog.get_etrecord().save(str(artifact_dir / "etrecord.bin"))
    save_pte_program(exec_prog, "vision_encoder_qnn", str(artifact_dir))
    print(f"[qwen2.5-vl-export-qnn] Saved: {artifact_dir / 'vision_encoder_qnn.pte'}")


if __name__ == "__main__":
    main()
