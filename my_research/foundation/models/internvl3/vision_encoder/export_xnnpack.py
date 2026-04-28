# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
InternVL3 비전 인코더(InternViT-300M + MLP projector)를 XNNPACK CPU용으로 export.

QNN 버전과 동일하게 8-bit PTQ 양자화 지원 (--quantize).
사용:
    python -m my_research.foundation.models.internvl3.vision_encoder.export_xnnpack
    python -m my_research.foundation.models.internvl3.vision_encoder.export_xnnpack \\
        --quantize --output ./vision_encoder_xnnpack_q8.pte
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from my_research.foundation.models.internvl3.vision_encoder.model import (
    load_vision_encoder,
)

# QNN InternVL3Encoder와 동일: 448x448
_DEFAULT_CALIBRATION_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def _load_calibration_data(
    calibration_sources: Optional[List[str]],
    img_size: int = 448,
    num_samples: int = 8,
) -> List[tuple]:
    """Calibration용 (pixel_values,) 리스트 생성."""
    samples = []
    if calibration_sources:
        try:
            from torchvision import transforms
            from transformers.image_utils import load_image

            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            for src in calibration_sources[:num_samples]:
                img = load_image(src)
                pv = transform(img).unsqueeze(0)  # (1, 3, H, W)
                samples.append((pv,))
        except Exception as e:
            print(f"[export] Calibration 이미지 로드 실패 ({e}), 랜덤 사용")
            samples = []

    if not samples:
        for _ in range(num_samples):
            samples.append(
                (torch.randn(1, 3, img_size, img_size, dtype=torch.float32),)
            )
    return samples


def main(
    model_path: str = None,
    output: str = None,
    encoder_weights: str = None,
    quantize: bool = False,
    calibration_images: List[str] = None,
    calibration_num: int = 8,
):
    parser = argparse.ArgumentParser(
        description="InternVL3 vision encoder XNNPACK export"
    )
    parser.add_argument(
        "--model_path",
        default="OpenGVLab/InternVL3-1B-hf",
        help="HuggingFace model id or local path (default: OpenGVLab/InternVL3-1B-hf)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PTE path (default: vision_encoder_xnnpack.pte in cwd)",
    )
    parser.add_argument(
        "--encoder_weights",
        default=None,
        help="Pre-extracted vision encoder .safetensors (faster load, skip full model)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="8-bit PTQ (QNN 16a8w와 유사, XNNPACK symmetric 8-bit)",
    )
    parser.add_argument(
        "--calibration_images",
        default=None,
        nargs="+",
        help="Calibration 이미지 경로/URL (미지정 시 랜덤, 기본 8장)",
    )
    parser.add_argument(
        "--calibration_num",
        type=int,
        default=8,
        help="Calibration 샘플 수 (기본 8)",
    )
    args = parser.parse_args()

    if model_path is not None:
        args.model_path = model_path
    if output is not None:
        args.output = output
    if encoder_weights is not None:
        args.encoder_weights = encoder_weights
    if quantize:
        args.quantize = True
    if calibration_images is not None:
        args.calibration_images = calibration_images
    if calibration_num is not None:
        args.calibration_num = calibration_num

    encoder_weights = getattr(args, "encoder_weights", None)

    out_path = (
        Path(args.output).resolve()
        if args.output
        else Path.cwd() / "vision_encoder_xnnpack.pte"
    )
    if out_path.suffix != ".pte":
        out_path = out_path.with_suffix(".pte")
    if args.quantize and "_q8" not in out_path.stem:
        out_path = out_path.parent / f"{out_path.stem}_q8.pte"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading vision encoder from {args.model_path}...")
    encoder = load_vision_encoder(
        args.model_path, encoder_weights=encoder_weights
    )
    example_inputs = encoder.get_example_inputs()

    if args.quantize:
        print("[export] 8-bit PTQ 양자화 (XNNPACK symmetric)...")
        calibration_sources = args.calibration_images or [_DEFAULT_CALIBRATION_URL]
        calibration_data = _load_calibration_data(
            calibration_sources,
            img_size=448,
            num_samples=args.calibration_num,
        )

        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        ep = torch.export.export(encoder, example_inputs, strict=False)
        prepared = prepare_pt2e(ep.module(), quantizer)
        for inp in calibration_data:
            prepared(*inp)
        encoder = convert_pt2e(prepared)
        example_inputs = calibration_data[0]
        ep = torch.export.export(encoder, example_inputs, strict=False)
        print("[export] 양자화 완료, XNNPACK으로 lowering...")
    else:
        ep = torch.export.export(encoder, example_inputs, strict=False)

    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=not args.quantize,
            _skip_dim_order=True,
        ),
    )
    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    save_pte_program(exec_prog, out_path.stem, str(out_path.parent))
    print(f"[export] Saved: {out_path.parent / (out_path.stem + '.pte')}")


if __name__ == "__main__":
    main()
