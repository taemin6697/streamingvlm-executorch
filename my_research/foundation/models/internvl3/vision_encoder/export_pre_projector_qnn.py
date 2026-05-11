#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export InternVL3 vision tower pre-projector features to QNN.

This produces a QNN PTE for only the InternVL vision tower path up through
CLS removal + pixel_shuffle/downsample. It intentionally excludes
multi_modal_projector so projector/mmproj can be handled by the decoder side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchBackendType
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    InternVL3EncoderQuantRecipe,
)
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from my_research.foundation.models.internvl3.vision_encoder.pre_projector import (
    describe_pre_projector_output,
    load_vision_pre_projector,
)
from my_research.foundation.models.internvl3.vision_encoder.export_xnnpack import (
    _load_calibration_data,
)


_DEFAULT_HF_MODELS = {
    "internvl3_1b": "OpenGVLab/InternVL3-1B-hf",
    "internvl3_2b": "OpenGVLab/InternVL3-2B-hf",
    "internvl3_8b": "OpenGVLab/InternVL3-8B-hf",
}


def _default_artifact_root(model_name: str, soc_model: str) -> Path:
    return (
        Path("my_research/foundation_llamacpp/results/vision_models")
        / f"{model_name}_vision_tower_preproj_qnn_{soc_model.lower()}"
    )


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
        "artifact_type": "internvl3_vision_tower_pre_projector_qnn",
        "backend": "qnn",
        "soc_model": soc_model,
        "quantization": quant,
        "calibration_images": calibration_images or [],
        "projector_included": False,
        "output_name": "vision_tower_preproj_qnn.pte",
        "image_size": 448,
        "output": output_summary,
        "notes": (
            "Output is after InternVL vision_tower CLS removal + pixel_shuffle/downsample "
            "and before multi_modal_projector. This PTE is not compatible with current "
            "hybrid_decode unless decoder-side mmproj/projector is applied before image prefill."
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
    calibration_images: Optional[List[str]] = None,
    calibration_num: Optional[int] = None,
    shape_only: bool = False,
    generate_etrecord: bool = False,
) -> None:
    parser = argparse.ArgumentParser(
        description="Export InternVL3 pre-projector vision tower to QNN"
    )
    parser.add_argument(
        "--model-name",
        choices=sorted(_DEFAULT_HF_MODELS),
        default="internvl3_1b",
        help="Named InternVL3 model size used for defaults.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="HuggingFace model id or local path. Defaults from --model-name.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Output directory. Default: my_research/foundation_llamacpp/results/vision_models/<model>_vision_tower_preproj_qnn_<soc>",
    )
    parser.add_argument("--encoder-weights", default=None)
    parser.add_argument("--soc-model", default="SM8750")
    parser.add_argument(
        "--quant",
        choices=("fp16", "16a8w"),
        default="16a8w",
        help="QNN quantization mode. Default matches existing InternVL3 QNN vision exports.",
    )
    parser.add_argument(
        "--calibration-images",
        nargs="+",
        default=None,
        help="Calibration image paths/URLs for 16a8w PTQ. Defaults to random example input if omitted.",
    )
    parser.add_argument(
        "--calibration-num",
        type=int,
        default=8,
        help="Maximum number of calibration images to use.",
    )
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
    if calibration_images is not None:
        args.calibration_images = calibration_images
    if calibration_num is not None:
        args.calibration_num = calibration_num
    if shape_only:
        args.shape_only = True
    if generate_etrecord:
        args.generate_etrecord = True

    resolved_model_path = args.model_path or _DEFAULT_HF_MODELS[args.model_name]
    artifact_dir = (
        Path(args.artifact_root)
        if args.artifact_root
        else _default_artifact_root(args.model_name, args.soc_model)
    ).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export-preproj-qnn] Loading {resolved_model_path}...")
    model = load_vision_pre_projector(
        resolved_model_path,
        encoder_weights=args.encoder_weights,
    ).eval()
    output_summary = describe_pre_projector_output(model)
    print(f"[export-preproj-qnn] Output summary: {output_summary}")
    _write_metadata(
        artifact_dir / "vision_tower_preproj_qnn_metadata.json",
        model_name=args.model_name,
        model_path=resolved_model_path,
        soc_model=args.soc_model,
        quant=args.quant,
        calibration_images=args.calibration_images,
        output_summary=output_summary,
    )
    if args.shape_only:
        print(f"[export-preproj-qnn] Shape-only metadata saved: {artifact_dir}")
        return

    example_inputs = model.get_example_inputs()
    if args.quant == "16a8w":
        print("[export-preproj-qnn] Applying InternVL3 encoder 16a8w PTQ...")
        with torch.no_grad():
            calibration_data = _load_calibration_data(
                args.calibration_images,
                img_size=448,
                num_samples=args.calibration_num,
            )
            example_inputs = calibration_data[0]
            exported = torch.export.export(model, example_inputs, strict=False).module()
            quantizer = make_quantizer(
                backend=QnnExecuTorchBackendType.kHtpBackend,
                soc_model=args.soc_model,
            )
            quantizer.set_recipe(InternVL3EncoderQuantRecipe(verbose=True).recipe)
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
    save_pte_program(exec_prog, "vision_tower_preproj_qnn", str(artifact_dir))
    print(f"[export-preproj-qnn] Saved: {artifact_dir / 'vision_tower_preproj_qnn.pte'}")


if __name__ == "__main__":
    main()
