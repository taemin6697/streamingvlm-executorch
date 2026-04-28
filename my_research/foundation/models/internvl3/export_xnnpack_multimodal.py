#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from my_research.foundation.models.internvl3 import convert_weights
from my_research.foundation.manifest import (
    FOUNDATION_MANIFEST_FILENAME,
    build_manifest,
    write_manifest,
)
from my_research.foundation.models.internvl3.vision_encoder.model import (
    load_vision_encoder,
)
from executorch.examples.models.llama.export_llama_lib import (
    _prepare_for_llama_export,
    get_quantizer_and_quant_params,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import (
    ConstraintBasedSymShapeEvalPass,
    HintBasedSymShapeEvalPass,
)
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from executorch.extension.llm.export.config.llm_config import DtypeOverride, LlmConfig
from transformers import AutoTokenizer

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

_DEFAULT_HF_MODELS = {
    "internvl3_1b": "OpenGVLab/InternVL3-1B-hf",
    "internvl3_2b": "OpenGVLab/InternVL3-2B-hf",
    "internvl3_8b": "OpenGVLab/InternVL3-8B-hf",
}

_DEFAULT_PARAMS = {
    "internvl3_1b": "1b_config.json",
    "internvl3_2b": "2b_config.json",
    "internvl3_8b": "8b_config.json",
}

_DEFAULT_CALIBRATION_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def _llama_export_model_class(decoder_model: str) -> str:
    if decoder_model.startswith("internvl3_"):
        return "llama3_2"
    return decoder_model


class _ModelName(str):
    @property
    def value(self) -> str:
        return str(self)


def _default_params_path(model_name: str) -> Path:
    return Path(__file__).resolve().parent / _DEFAULT_PARAMS[model_name]


def _load_calibration_data(
    calibration_sources: Optional[List[str]],
    img_size: int = 448,
    num_samples: int = 8,
) -> List[Tuple[torch.Tensor]]:
    samples = []
    if calibration_sources:
        try:
            from torchvision import transforms
            from transformers.image_utils import load_image

            transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            for src in calibration_sources[:num_samples]:
                img = load_image(src)
                pixel_values = transform(img).unsqueeze(0)
                samples.append((pixel_values,))
        except Exception as exc:
            logging.warning(
                "Calibration 이미지 로드 실패 (%s). 랜덤 텐서를 사용합니다.", exc
            )
            samples = []

    if not samples:
        for _ in range(num_samples):
            samples.append(
                (torch.randn(1, 3, img_size, img_size, dtype=torch.float32),)
            )
    return samples


def _resolve_text_checkpoint(args) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
    if args.checkpoint:
        return str(Path(args.checkpoint).resolve()), None

    hf_model_path = Path(args.model_path)
    if not hf_model_path.exists():
        raise ValueError(
            "--checkpoint 를 주지 않는 경우 --model_path 는 로컬 InternVL3 모델 디렉터리여야 합니다."
        )

    tmp_dir = tempfile.TemporaryDirectory(prefix="internvl3_text_ckpt_")
    checkpoint_path = Path(tmp_dir.name) / "internvl3_text_decoder_meta.pth"
    logging.info("Converting InternVL3 text checkpoint from %s ...", hf_model_path)
    convert_weights(str(hf_model_path), str(checkpoint_path))
    return str(checkpoint_path), tmp_dir


def _build_llm_config(args, checkpoint_path: str) -> LlmConfig:
    llm_config = LlmConfig()
    llm_config.base.model_class = _ModelName(
        _llama_export_model_class(args.decoder_model)
    )
    llm_config.base.checkpoint = checkpoint_path
    llm_config.base.params = str(Path(args.params).resolve())
    llm_config.base.metadata = json.dumps(
        {
            "get_bos_id": args.bos_token_id,
            "get_eos_ids": args.eos_ids,
        }
    )

    llm_config.model.dtype_override = DtypeOverride(args.dtype)
    llm_config.model.use_kv_cache = True
    llm_config.model.use_sdpa_with_kv_cache = args.use_sdpa_with_kv_cache
    llm_config.model.enable_dynamic_shape = True

    llm_config.export.max_seq_length = args.max_seq_len
    llm_config.export.max_context_length = args.max_context_len
    llm_config.export.output_dir = str(Path(args.output).resolve().parent)
    llm_config.export.output_name = Path(args.output).name

    # Decoder/embedding quantization: fp16 = none
    llm_config.quantization.qmode = (
        args.decoder_quant if args.decoder_quant != "fp16" else None
    )
    llm_config.quantization.group_size = args.text_group_size
    llm_config.quantization.embedding_quantize = (
        args.embedding_quant if args.embedding_quant != "fp16" else None
    )

    llm_config.backend.xnnpack.enabled = True
    llm_config.backend.xnnpack.extended_ops = True
    return llm_config


def _create_text_decoder_export(
    eager_model: torch.nn.Module,
    llm_config: LlmConfig,
    dtype: DType,
):
    class InternVL3TextDecoder(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, embeddings, input_pos):
            return self.decoder(None, {"input_pos": input_pos}, embeddings)

    # Export text_decoder with a representative prefill-length embedding tensor.
    # XNNPACK multimodal runners pass a single start position for prefill/decode,
    # while the embedding sequence length varies with image/text content.
    sample_seq_len = min(256, llm_config.export.max_seq_length)
    token_ids = torch.arange(1, sample_seq_len + 1, dtype=torch.long).unsqueeze(0)
    sample_embeddings = eager_model.tok_embeddings(token_ids)
    sample_input_pos = torch.tensor([0], dtype=torch.long)
    seq_dim = torch.export.Dim(
        "seq_dim", min=1, max=llm_config.export.max_seq_length
    )
    dynamic_shapes = ({1: seq_dim}, {0: 1})

    manager = LLMEdgeManager(
        model=InternVL3TextDecoder(eager_model),
        modelname="internvl3_text_decoder",
        max_seq_len=llm_config.export.max_seq_length,
        dtype=dtype,
        use_kv_cache=True,
        example_inputs=(sample_embeddings, sample_input_pos),
        dynamic_shapes=dynamic_shapes,
    )
    _, quantizers, _ = get_quantizer_and_quant_params(llm_config)
    manager = manager.export().pt2e_quantize(quantizers)
    with torch.no_grad():
        return torch.export.export(
            manager.pre_autograd_graph_module,
            manager.example_inputs,
            dynamic_shapes=manager._get_dynamic_shape(),
            strict=True,
        )


def _create_token_embedding_export(
    eager_model: torch.nn.Module,
    max_seq_len: int,
    dtype: torch.dtype,
):
    # Use a representative prompt length so memory planning does not overfit to
    # a 2-token sample while still keeping the method bounded-dynamic.
    sample_seq_len = min(256, max_seq_len)
    sample_tokens = torch.arange(1, sample_seq_len + 1, dtype=torch.long).unsqueeze(0)
    token_dim = torch.export.Dim("token_dim_1", min=1, max=max_seq_len)
    emb = eager_model.tok_embeddings
    if dtype != torch.float32:
        emb = emb.to(dtype)
    with torch.no_grad():
        return torch.export.export(
            emb,
            (sample_tokens,),
            dynamic_shapes=[{1: token_dim}],
            strict=True,
        )


def _create_vision_encoder_export(args):
    encoder = load_vision_encoder(
        args.model_path, encoder_weights=args.encoder_weights
    ).eval()
    example_inputs = encoder.get_example_inputs()

    if args.vision_quant == "8a8w":
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        calibration_sources = args.calibration_images or [_DEFAULT_CALIBRATION_URL]
        calibration_data = _load_calibration_data(
            calibration_sources,
            img_size=448,
            num_samples=args.calibration_num,
        )
        prepared_ep = torch.export.export(encoder, example_inputs, strict=False)
        from torchao.quantization.pt2e import move_exported_model_to_eval
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        prepared = prepare_pt2e(prepared_ep.module(), quantizer)
        for inp in calibration_data:
            prepared(*inp)
        encoder = convert_pt2e(prepared)
        move_exported_model_to_eval(encoder)
        example_inputs = calibration_data[0]
    elif args.vision_quant == "fp16":
        encoder = encoder.to(torch.float16)
        inp = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
        example_inputs = tuple(
            x.to(torch.float16) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in inp
        )

    with torch.no_grad():
        return torch.export.export(encoder, example_inputs, strict=False)


def _tokenizer_metadata(tokenizer) -> Tuple[int, List[int]]:
    eos_ids = []
    for token in ("<|im_end|>", tokenizer.eos_token):
        if token is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            continue
        if token_id not in eos_ids:
            eos_ids.append(int(token_id))

    if not eos_ids and tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = eos_ids[0] if eos_ids else 0
    return int(bos_token_id), eos_ids


VISION_ENCODER_PTE = "vision_encoder_xnnpack"
TEXT_EMBEDDING_PTE = "text_embedding_xnnpack"
TEXT_DECODER_PTE = "text_decoder_xnnpack"


def _write_artifacts(args, tokenizer, output_dir: Path, pte_paths: dict):
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = artifact_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    legacy_manifest = {
        "vision_encoder_pte": str(pte_paths["vision_encoder"]),
        "text_embedding_pte": str(pte_paths["text_embedding"]),
        "text_decoder_pte": str(pte_paths["text_decoder"]),
        "tokenizer_path": str(tokenizer_dir / "tokenizer.json"),
        "decoder_model": args.decoder_model,
        "model_source": args.model_path,
        "max_seq_len": args.max_seq_len,
        "max_context_len": args.max_context_len,
        "dtype": args.dtype,
        "vision_quant": args.vision_quant,
        "decoder_quant": args.decoder_quant,
        "embedding_quant": args.embedding_quant,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(legacy_manifest, indent=2), encoding="utf-8"
    )

    foundation_manifest = build_manifest(
        artifact_root=output_dir,
        backend="xnnpack",
        variant=args.decoder_model,
        runner_type="multimodal_combined" if args.single_pte else "multimodal_split",
        vision_encoder_pte=Path(pte_paths["vision_encoder"]),
        text_embedding_pte=Path(pte_paths["text_embedding"]),
        text_decoder_pte=Path(pte_paths["text_decoder"]),
        tokenizer_path=tokenizer_dir / "tokenizer.json",
        combined_pte=(
            Path(pte_paths["text_decoder"])
            if args.single_pte
            else None
        ),
        export={
            "max_seq_len": args.max_seq_len,
            "max_context_len": args.max_context_len,
            "dtype": args.dtype,
            "model_source": args.model_path,
        },
        quant={
            "vision": args.vision_quant,
            "decoder": args.decoder_quant,
            "embedding": args.embedding_quant,
        },
        runtime={
            "decoder_model_version": "internvl3",
            "preferred_runner": "unified_vlm_runner",
        },
    )
    write_manifest(foundation_manifest, output_dir / FOUNDATION_MANIFEST_FILENAME)
    return artifact_dir


def _export_single_pte(args, vision_encoder_ep, token_embedding_ep, text_decoder_ep,
                       metadata, use_decoder_quant, output_path):
    """Export as single combined multi-method PTE (legacy)."""
    text_decoder_partitioner = [
        XnnpackPartitioner(
            config_precisions=(
                ConfigPrecisionType.DYNAMIC_QUANT if use_decoder_quant
                else ConfigPrecisionType.FP32
            ),
            per_op_mode=True,
        ),
        XnnpackPartitioner(),
    ]
    lowered = to_edge_transform_and_lower(
        {
            "vision_encoder": vision_encoder_ep,
            "token_embedding": token_embedding_ep,
            "text_decoder": text_decoder_ep,
        },
        partitioner={
            "vision_encoder": [XnnpackPartitioner()],
            "token_embedding": [XnnpackPartitioner()],
            "text_decoder": text_decoder_partitioner,
        },
        constant_methods=metadata,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    prog = lowered.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[QuantFusionPass()],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={
                "vision_encoder": ConstraintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": ConstraintBasedSymShapeEvalPass(),
            },
        )
    )
    with open(output_path, "wb") as f:
        prog.write_to_file(f)
    return {"vision_encoder": output_path, "text_embedding": output_path, "text_decoder": output_path}


def export_multimodal(args):
    output_path = Path(args.output).resolve()
    if args.single_pte:
        if output_path.suffix != ".pte":
            output_path = output_path.with_suffix(".pte")
        output_dir = output_path.parent
    else:
        if output_path.suffix == ".pte":
            output_dir = output_path.parent
        else:
            output_dir = output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    args.bos_token_id, args.eos_ids = _tokenizer_metadata(tokenizer)

    checkpoint_path, temp_dir = _resolve_text_checkpoint(args)
    try:
        dtype = DType[args.dtype]
        torch_dtype = dtype.to_torch_dtype()
        use_decoder_quant = args.decoder_quant != "fp16"

        llm_config = _build_llm_config(args, checkpoint_path)
        text_edge_manager = _prepare_for_llama_export(llm_config)
        eager_text_model = text_edge_manager.model

        vision_encoder_ep = _create_vision_encoder_export(args)
        token_embedding_ep = _create_token_embedding_export(
            eager_text_model, args.max_seq_len, torch_dtype
        )
        text_decoder_ep = _create_text_decoder_export(
            eager_text_model, llm_config, dtype
        )

        metadata = {
            "get_bos_id": args.bos_token_id,
            "get_eos_ids": args.eos_ids,
            "get_max_seq_len": args.max_seq_len,
            "get_max_context_len": args.max_context_len,
            "enable_dynamic_shape": True,
            "use_kv_cache": True,
        }

        if args.single_pte:
            pte_paths = _export_single_pte(
                args, vision_encoder_ep, token_embedding_ep, text_decoder_ep,
                metadata, use_decoder_quant, output_path
            )
            logging.info("Saved single XNNPACK PTE to %s", output_path)
        else:
            pte_paths = {}
            exec_config = ExecutorchBackendConfig(
                extract_delegate_segments=True,
                passes=[QuantFusionPass()],
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            )

            # 1. Vision encoder
            vision_edge = to_edge_transform_and_lower(
                vision_encoder_ep,
                partitioner=[XnnpackPartitioner()],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            vision_pte_path = output_dir / f"{VISION_ENCODER_PTE}.pte"
            with open(vision_pte_path, "wb") as f:
                vision_edge.to_executorch(exec_config).write_to_file(f)
            pte_paths["vision_encoder"] = vision_pte_path
            logging.info("Saved vision encoder to %s", vision_pte_path)

            # 2. Token embedding
            emb_edge = to_edge_transform_and_lower(
                token_embedding_ep,
                partitioner=[XnnpackPartitioner()],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            emb_pte_path = output_dir / f"{TEXT_EMBEDDING_PTE}.pte"
            with open(emb_pte_path, "wb") as f:
                emb_edge.to_executorch(exec_config).write_to_file(f)
            pte_paths["text_embedding"] = emb_pte_path
            logging.info("Saved token embedding to %s", emb_pte_path)

            # 3. Text decoder
            text_decoder_partitioner = [
                XnnpackPartitioner(
                    config_precisions=(
                        ConfigPrecisionType.DYNAMIC_QUANT if use_decoder_quant
                        else ConfigPrecisionType.FP32
                    ),
                    per_op_mode=True,
                ),
                XnnpackPartitioner(),
            ]
            decoder_edge = to_edge_transform_and_lower(
                text_decoder_ep,
                partitioner=text_decoder_partitioner,
                constant_methods=metadata,
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            decoder_pte_path = output_dir / f"{TEXT_DECODER_PTE}.pte"
            with open(decoder_pte_path, "wb") as f:
                decoder_edge.to_executorch(exec_config).write_to_file(f)
            pte_paths["text_decoder"] = decoder_pte_path
            logging.info("Saved text decoder to %s", decoder_pte_path)

        artifact_dir = _write_artifacts(args, tokenizer, output_dir, pte_paths)
        logging.info("Tokenizer artifacts saved under %s", artifact_dir)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export InternVL3 multimodal program for Android CPU/XNNPACK"
    )
    parser.add_argument(
        "--decoder_model",
        default="internvl3_1b",
        choices=sorted(_DEFAULT_HF_MODELS.keys()),
        help="InternVL3 variant to export.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Hugging Face model id or local directory for tokenizer and vision encoder.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Meta-format text decoder checkpoint produced by convert_weights.py.",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="InternVL3 params json. Defaults to the config bundled with this repo.",
    )
    parser.add_argument(
        "--encoder_weights",
        default=None,
        help="Optional standalone vision encoder weights file.",
    )
    parser.add_argument(
        "--output",
        default="internvl3_xnnpack",
        help="Output path: directory for 3 PTE (default), or .pte path when --single_pte.",
    )
    parser.add_argument(
        "--single_pte",
        action="store_true",
        help="Export as single combined .pte (legacy, for internvl3_xnnpack_runner). Default: 3 separate PTE files.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length compiled into the decoder.",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=1024,
        help="Maximum context length compiled into the decoder.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "fp32"],
        help="Base dtype for all components (default: fp16).",
    )
    parser.add_argument(
        "--vision_quant",
        default="fp16",
        choices=["fp16", "8a8w"],
        help="Vision encoder: fp16 (no quant) or 8a8w.",
    )
    parser.add_argument(
        "--decoder_quant",
        default="fp16",
        choices=["fp16", "8da4w", "8da8w", "int8"],
        help="Text decoder: fp16 (no quant) or 8da4w/8da8w/int8.",
    )
    parser.add_argument(
        "--embedding_quant",
        default="fp16",
        choices=["fp16", "4,32"],
        help="Token embedding: fp16 (no quant) or 4,32.",
    )
    parser.add_argument(
        "--text_group_size",
        type=int,
        default=128,
        help="Group size for text decoder quantization (when decoder_quant != fp16).",
    )
    parser.add_argument(
        "--calibration_images",
        nargs="+",
        default=None,
        help="Calibration images for vision PTQ.",
    )
    parser.add_argument(
        "--calibration_num",
        type=int,
        default=8,
        help="Number of calibration samples for vision PTQ.",
    )
    parser.add_argument(
        "--use_sdpa_with_kv_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the sdpa_with_kv_cache custom op for the text decoder.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to Hugging Face loaders.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = _DEFAULT_HF_MODELS[args.decoder_model]
    if args.params is None:
        args.params = str(_default_params_path(args.decoder_model))
    export_multimodal(args)


if __name__ == "__main__":
    main()
