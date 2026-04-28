# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Foundation-native XNNPACK export. Does not call export_xnnpack_multimodal."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
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
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from executorch.extension.llm.export.config.llm_config import DtypeOverride, LlmConfig
from transformers import AutoTokenizer

import my_research.foundation.models.internvl3 as internvl3_pkg
from my_research.foundation.models.internvl3 import convert_weights
from my_research.foundation.models.internvl3.vision_encoder.model import (
    load_vision_encoder,
)
from my_research.foundation.manifest import (
    FOUNDATION_MANIFEST_FILENAME,
    build_manifest,
    write_manifest,
)

LOG = logging.getLogger(__name__)

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

VISION_ENCODER_PTE = "vision_encoder_xnnpack"
TEXT_EMBEDDING_PTE = "text_embedding_xnnpack"
TEXT_DECODER_PTE = "text_decoder_xnnpack"


def _llama_export_model_class(decoder_model: str) -> str:
    if decoder_model.startswith("internvl3_"):
        return "llama3_2"
    return decoder_model


class _ModelName(str):
    @property
    def value(self) -> str:
        return str(self)


def _default_params_path(model_name: str) -> Path:
    return Path(internvl3_pkg.__file__).resolve().parent / _DEFAULT_PARAMS[model_name]


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
            LOG.warning("Calibration 이미지 로드 실패 (%s). 랜덤 텐서를 사용합니다.", exc)
            samples = []

    if not samples:
        for _ in range(num_samples):
            samples.append((torch.randn(1, 3, img_size, img_size, dtype=torch.float32),))
    return samples


def _resolve_text_checkpoint(
    model_path: str, checkpoint: Optional[str], artifact_root: Path
) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
    if checkpoint:
        return str(Path(checkpoint).resolve()), None

    hf_model_path = Path(model_path)
    if not hf_model_path.exists():
        raise ValueError(
            "--checkpoint 를 주지 않는 경우 --model_path 는 로컬 InternVL3 모델 디렉터리여야 합니다."
        )

    tmp_dir = tempfile.TemporaryDirectory(prefix="internvl3_text_ckpt_")
    checkpoint_path = Path(tmp_dir.name) / "internvl3_text_decoder_meta.pth"
    LOG.info("Converting InternVL3 text checkpoint from %s ...", hf_model_path)
    convert_weights(str(hf_model_path), str(checkpoint_path))
    return str(checkpoint_path), tmp_dir


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


def _split_pte_names(backend: str) -> Tuple[str, str, str]:
    if backend == "vulkan":
        return (
            "vision_encoder_vulkan",
            "text_embedding_vulkan",
            "text_decoder_vulkan",
        )
    return (VISION_ENCODER_PTE, TEXT_EMBEDDING_PTE, TEXT_DECODER_PTE)


@contextlib.contextmanager
def _vulkan_preprocess_option_patch() -> Iterator[None]:
    """Teach ExecuTorch Vulkan preprocess about bool options it already checks."""
    import executorch.backends.vulkan.vulkan_preprocess as vulkan_preprocess
    import executorch.backends.vulkan.patterns.sdpa as vulkan_sdpa

    original_parse_compile_spec = vulkan_preprocess.parse_compile_spec
    original_causal_sdpa_match_init = vulkan_sdpa.CausalSDPAMatch.__init__

    def parse_compile_spec_with_foundation_options(compile_specs):
        options = original_parse_compile_spec(compile_specs)
        for spec in compile_specs:
            if spec.key in {
                "skip_memory_planning",
                "skip_tag_memory_metadata",
                "small_texture_limits",
            }:
                options[spec.key] = bool.from_bytes(spec.value, byteorder="little")
        return options

    def cache_ancestor_nodes(node, depth: int = 4):
        if depth < 0 or not isinstance(node, torch.fx.Node):
            return
        yield node
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                yield from cache_ancestor_nodes(arg, depth - 1)
            elif isinstance(arg, (tuple, list)):
                for item in arg:
                    yield from cache_ancestor_nodes(item, depth - 1)

    def find_update_cache_node_from_ancestors(node):
        for candidate in cache_ancestor_nodes(node):
            for user in candidate.users:
                if vulkan_sdpa.is_update_cache_node(user):
                    return user
        return None

    def causal_sdpa_match_with_ancestor_cache_search(self, custom_sdpa_node):
        original_causal_sdpa_match_init(self, custom_sdpa_node)
        if not self.match_found:
            return
        if self.update_key_cache_node is None:
            self.update_key_cache_node = find_update_cache_node_from_ancestors(
                self.key_cache_node
            )
            if self.update_key_cache_node is not None:
                self.key_projection_node = self.update_key_cache_node.args[0]
        if self.update_value_cache_node is None:
            self.update_value_cache_node = find_update_cache_node_from_ancestors(
                self.value_cache_node
            )
            if self.update_value_cache_node is not None:
                self.value_projection_node = self.update_value_cache_node.args[0]

    vulkan_preprocess.parse_compile_spec = parse_compile_spec_with_foundation_options
    vulkan_sdpa.CausalSDPAMatch.__init__ = causal_sdpa_match_with_ancestor_cache_search
    try:
        yield
    finally:
        vulkan_preprocess.parse_compile_spec = original_parse_compile_spec
        vulkan_sdpa.CausalSDPAMatch.__init__ = original_causal_sdpa_match_init


def _foundation_vulkan_partitioner(
    *, dtype: str, enable_dynamic_shape: bool, force_fp16: bool
):
    assert (
        dtype == "fp32" or dtype is None
    ), "Vulkan backend does not support non fp32 dtype_override; use force_fp16."
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    return VulkanPartitioner(
        {"require_dynamic_shapes": enable_dynamic_shape, "force_fp16": force_fp16}
    )


def _lower_split_program(
    exported_program,
    *,
    backend: str,
    enable_dynamic_shape: bool,
    dtype: str,
    vulkan_force_fp16: bool = False,
    xnnpack_partitioner=None,
    vulkan_fallback_partitioner=None,
    constant_methods=None,
):
    partitioner = xnnpack_partitioner
    if backend == "vulkan":
        partitioner = [
            _foundation_vulkan_partitioner(
                dtype=dtype,
                enable_dynamic_shape=enable_dynamic_shape,
                force_fp16=vulkan_force_fp16,
            )
        ]
        if vulkan_fallback_partitioner:
            partitioner.extend(vulkan_fallback_partitioner)
        with _vulkan_preprocess_option_patch():
            return to_edge_transform_and_lower(
                exported_program,
                partitioner=partitioner,
                constant_methods=constant_methods,
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
    return to_edge_transform_and_lower(
        exported_program,
        partitioner=partitioner,
        constant_methods=constant_methods,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )


def export_xnnpack(args: argparse.Namespace) -> int:
    """Foundation-native XNNPACK/Vulkan export. Split PTE only."""
    output_dir = Path(args.artifact_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = getattr(args, "backend", "xnnpack")
    if backend not in {"xnnpack", "vulkan"}:
        raise SystemExit(f"Unsupported split backend: {backend}")

    max_context_len = getattr(args, "max_context_len", None) or args.max_seq_len

    model_path = args.model_path or _DEFAULT_HF_MODELS.get(args.decoder_model)
    if not model_path:
        raise SystemExit(
            f"Unsupported decoder_model for {backend}: {args.decoder_model}. "
            f"Use one of {list(_DEFAULT_HF_MODELS.keys())} or set --model_path."
        )
    if backend == "vulkan":
        supported_vulkan_decoder_quant = {"fp16", "8da4w"}
        if args.vision_quant != "fp16" or args.embedding_quant != "fp16":
            raise SystemExit(
                "Vulkan foundation export currently keeps vision/embedding quant as fp16."
            )
        if args.decoder_quant not in supported_vulkan_decoder_quant:
            raise SystemExit(
                "Vulkan foundation export currently supports decoder_quant fp16 or 8da4w."
            )

    params_path = args.params or str(_default_params_path(args.decoder_model))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=getattr(args, "trust_remote_code", True)
    )
    bos_token_id, eos_ids = _tokenizer_metadata(tokenizer)

    checkpoint_path, temp_dir = _resolve_text_checkpoint(
        model_path, args.checkpoint, output_dir
    )
    try:
        requested_dtype = args.dtype
        vulkan_force_fp16 = backend == "vulkan" and requested_dtype == "fp16"
        export_dtype = "fp32" if backend == "vulkan" else requested_dtype
        dtype = DType[export_dtype]
        torch_dtype = dtype.to_torch_dtype()
        use_decoder_quant = args.decoder_quant != "fp16"

        llm_config = LlmConfig()
        llm_config.base.model_class = _ModelName(
            _llama_export_model_class(args.decoder_model)
        )
        llm_config.base.checkpoint = checkpoint_path
        llm_config.base.params = str(Path(params_path).resolve())
        llm_config.base.metadata = json.dumps(
            {"get_bos_id": bos_token_id, "get_eos_ids": eos_ids}
        )
        llm_config.model.dtype_override = DtypeOverride(export_dtype)
        llm_config.model.use_kv_cache = True
        use_sdpa_with_kv_cache = getattr(args, "use_sdpa_with_kv_cache", True)
        if backend == "vulkan":
            use_sdpa_with_kv_cache = True
        llm_config.model.use_sdpa_with_kv_cache = use_sdpa_with_kv_cache
        enable_dynamic_shape = getattr(args, "dynamic_shape", True)
        llm_config.model.enable_dynamic_shape = enable_dynamic_shape
        llm_config.export.max_seq_length = args.max_seq_len
        llm_config.export.max_context_length = max_context_len
        llm_config.export.output_dir = str(output_dir.parent)
        llm_config.export.output_name = output_dir.name
        llm_config.quantization.qmode = (
            args.decoder_quant if args.decoder_quant != "fp16" else None
        )
        text_group_size = getattr(args, "text_group_size", None)
        if text_group_size is None:
            text_group_size = 64 if backend == "vulkan" and args.decoder_quant == "8da4w" else 128
        llm_config.quantization.group_size = text_group_size
        llm_config.quantization.embedding_quantize = (
            args.embedding_quant if args.embedding_quant != "fp16" else None
        )
        if backend == "xnnpack":
            llm_config.backend.xnnpack.enabled = True
            llm_config.backend.xnnpack.extended_ops = True
        elif backend == "vulkan":
            llm_config.backend.vulkan.enabled = True
            llm_config.backend.vulkan.force_fp16 = vulkan_force_fp16

        text_edge_manager = _prepare_for_llama_export(llm_config)
        eager_text_model = text_edge_manager.model

        vision_quant = getattr(args, "vision_quant", "fp16")
        calibration_num = getattr(args, "calibration_num", 8)
        calibration_images = getattr(args, "calibration_images", None)

        requested_decoder_input_mode = getattr(args, "decoder_input_mode", "token_ids")

        # Upstream ExecuTorch static KV-cache runners disable parallel prefill
        # and feed one token per decoder step. Dynamic exports keep a larger
        # representative sample while allowing the sequence dim to vary.
        sample_seq_len = (
            args.max_seq_len
            if backend == "vulkan"
            and requested_decoder_input_mode == "embeddings"
            and enable_dynamic_shape
            else min(256, args.max_seq_len) if enable_dynamic_shape else 1
        )
        token_ids = torch.arange(1, sample_seq_len + 1, dtype=torch.long).unsqueeze(0)
        token_emb = eager_text_model.tok_embeddings
        if torch_dtype != torch.float32:
            token_emb = token_emb.to(torch_dtype)
        token_dim = torch.export.Dim("token_dim_1", min=1, max=args.max_seq_len)
        token_embedding_dynamic_shapes = (
            [{1: token_dim}] if enable_dynamic_shape else None
        )
        with torch.no_grad():
            token_embedding_ep = torch.export.export(
                token_emb,
                (token_ids,),
                dynamic_shapes=token_embedding_dynamic_shapes,
                strict=True,
            )

        class InternVL3EmbeddingTextDecoder(torch.nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, embeddings, input_pos):
                return self.decoder(None, {"input_pos": input_pos}, embeddings)

        class InternVL3TokenTextDecoder(torch.nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, token_ids, input_pos):
                return self.decoder(token_ids, {"input_pos": input_pos})

        sample_embeddings = eager_text_model.tok_embeddings(token_ids)
        sample_input_pos = torch.tensor([0], dtype=torch.long)
        seq_dim = torch.export.Dim("seq_dim", min=1, max=llm_config.export.max_seq_length)
        if backend == "vulkan" and requested_decoder_input_mode == "token_ids":
            decoder_model = InternVL3TokenTextDecoder(eager_text_model)
            decoder_example_inputs = (token_ids, sample_input_pos)
            decoder_dynamic_shapes = (
                ({1: seq_dim}, {0: 1}) if enable_dynamic_shape else None
            )
            decoder_input_mode = "token_ids"
        else:
            decoder_model = InternVL3EmbeddingTextDecoder(eager_text_model)
            decoder_example_inputs = (sample_embeddings, sample_input_pos)
            decoder_dynamic_shapes = (
                ({1: seq_dim}, {0: 1}) if enable_dynamic_shape else None
            )
            decoder_input_mode = "embeddings"
        manager = LLMEdgeManager(
            model=decoder_model,
            modelname="internvl3_text_decoder",
            max_seq_len=llm_config.export.max_seq_length,
            dtype=dtype,
            use_kv_cache=True,
            example_inputs=decoder_example_inputs,
            enable_dynamic_shape=enable_dynamic_shape,
            dynamic_shapes=decoder_dynamic_shapes,
        )
        _, quantizers, _ = get_quantizer_and_quant_params(llm_config)
        manager = manager.export().pt2e_quantize(quantizers)
        with torch.no_grad():
            text_decoder_ep = torch.export.export(
                manager.pre_autograd_graph_module,
                manager.example_inputs,
                dynamic_shapes=manager._get_dynamic_shape(),
                strict=True,
            )

        metadata = {
            "get_bos_id": bos_token_id,
            "get_eos_ids": eos_ids,
            "get_max_seq_len": args.max_seq_len,
            "get_max_context_len": max_context_len,
            "enable_dynamic_shape": enable_dynamic_shape,
            "use_kv_cache": True,
            "use_sdpa_with_kv_cache": use_sdpa_with_kv_cache,
            "decoder_input_mode": decoder_input_mode,
        }

        exec_config = ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[QuantFusionPass()],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )

        vision_pte_name, emb_pte_name, decoder_pte_name = _split_pte_names(backend)

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
        decoder_edge = _lower_split_program(
            text_decoder_ep,
            backend=backend,
            enable_dynamic_shape=enable_dynamic_shape,
            dtype=export_dtype,
            vulkan_force_fp16=vulkan_force_fp16,
            xnnpack_partitioner=text_decoder_partitioner,
            constant_methods=metadata,
        )
        decoder_delegate = backend
        decoder_pte_path = output_dir / f"{decoder_pte_name}.pte"
        with open(decoder_pte_path, "wb") as f:
            decoder_edge.to_executorch(exec_config).write_to_file(f)
        LOG.info("Saved text decoder to %s", decoder_pte_path)

        vision_encoder = load_vision_encoder(
            model_path, encoder_weights=getattr(args, "encoder_weights", None)
        ).eval()
        example_inputs = vision_encoder.get_example_inputs()

        if vision_quant == "8a8w":
            quantizer = XNNPACKQuantizer()
            quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
            calibration_sources = calibration_images or [_DEFAULT_CALIBRATION_URL]
            calibration_data = _load_calibration_data(
                calibration_sources, img_size=448, num_samples=calibration_num
            )
            prepared_ep = torch.export.export(vision_encoder, example_inputs, strict=False)
            from torchao.quantization.pt2e import move_exported_model_to_eval
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

            prepared = prepare_pt2e(prepared_ep.module(), quantizer)
            for inp in calibration_data:
                prepared(*inp)
            vision_encoder = convert_pt2e(prepared)
            move_exported_model_to_eval(vision_encoder)
            example_inputs = calibration_data[0]
        elif vision_quant == "fp16" and backend != "vulkan":
            vision_encoder = vision_encoder.to(torch.float16)
            inp = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
            example_inputs = tuple(
                x.to(torch.float16) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                for x in inp
            )

        with torch.no_grad():
            vision_encoder_ep = torch.export.export(vision_encoder, example_inputs, strict=False)

        vision_edge = _lower_split_program(
            vision_encoder_ep,
            backend=backend,
            enable_dynamic_shape=enable_dynamic_shape,
            dtype=export_dtype,
            vulkan_force_fp16=vulkan_force_fp16,
            xnnpack_partitioner=[XnnpackPartitioner()],
            vulkan_fallback_partitioner=(
                [XnnpackPartitioner()] if getattr(args, "vulkan_xnnpack_fallback", False) else None
            ),
        )
        vision_pte_path = output_dir / f"{vision_pte_name}.pte"
        with open(vision_pte_path, "wb") as f:
            vision_edge.to_executorch(exec_config).write_to_file(f)
        LOG.info("Saved vision encoder to %s", vision_pte_path)

        emb_edge = _lower_split_program(
            token_embedding_ep,
            backend=backend,
            enable_dynamic_shape=enable_dynamic_shape,
            dtype=export_dtype,
            vulkan_force_fp16=vulkan_force_fp16,
            xnnpack_partitioner=[XnnpackPartitioner()],
            vulkan_fallback_partitioner=(
                [XnnpackPartitioner()] if getattr(args, "vulkan_xnnpack_fallback", False) else None
            ),
        )
        emb_pte_path = output_dir / f"{emb_pte_name}.pte"
        with open(emb_pte_path, "wb") as f:
            emb_edge.to_executorch(exec_config).write_to_file(f)
        LOG.info("Saved token embedding to %s", emb_pte_path)

        artifact_dir = output_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir = artifact_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)
        tokenizer_path = tokenizer_dir / "tokenizer.json"

        foundation_manifest = build_manifest(
            artifact_root=output_dir,
            backend=backend,
            variant=args.decoder_model,
            runner_type="multimodal_split",
            vision_encoder_pte=vision_pte_path,
            text_embedding_pte=emb_pte_path,
            text_decoder_pte=decoder_pte_path,
            tokenizer_path=tokenizer_path,
            export={
                "max_seq_len": args.max_seq_len,
                "max_context_len": max_context_len,
                "dtype": args.dtype,
                "vulkan_export_dtype": export_dtype if backend == "vulkan" else None,
                "vulkan_force_fp16": vulkan_force_fp16 if backend == "vulkan" else None,
                "model_source": model_path,
                "enable_dynamic_shape": enable_dynamic_shape,
                "use_sdpa_with_kv_cache": use_sdpa_with_kv_cache,
                "decoder_input_mode": decoder_input_mode,
                "text_group_size": text_group_size,
                "static_prefill_mode": (
                    "parallel_dynamic" if enable_dynamic_shape else "sequential_1_token"
                ),
                "vulkan_xnnpack_fallback": getattr(args, "vulkan_xnnpack_fallback", False),
            },
            quant={
                "vision": vision_quant,
                "decoder": args.decoder_quant,
                "embedding": args.embedding_quant,
            },
            runtime={
                "decoder_model_version": "internvl3",
                "preferred_runner": "xnnpack_qnn_runner",
                "split_runner_backend": "xnnpack" if backend == "xnnpack" else "vulkan",
                "vision_delegate": backend,
                "embedding_delegate": backend,
                "decoder_delegate": decoder_delegate,
            },
        )
        write_manifest(foundation_manifest, output_dir / FOUNDATION_MANIFEST_FILENAME)
        LOG.info("Saved manifest to %s", output_dir / FOUNDATION_MANIFEST_FILENAME)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    return 0
