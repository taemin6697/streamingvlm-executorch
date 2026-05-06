# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Foundation-native QNN export that treats Qualcomm llama code as a dependency."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from my_research.foundation.manifest import (
    FOUNDATION_MANIFEST_FILENAME,
    build_manifest,
    write_manifest,
)


_QNN_QUANT_MODES = {
    "fp16",
    "16a16w",
    "16a8w",
    "16a4w",
    "16a4w_block",
    "8a8w",
    "8a4w",
}


def _normalize_qnn_quant(name: str | None, *, component: str) -> str:
    key = (name or "fp16").lower()
    if key not in _QNN_QUANT_MODES:
        supported = ", ".join(sorted(_QNN_QUANT_MODES))
        raise SystemExit(
            f"Unsupported QNN {component} quant mode: {name}. "
            f"Supported modes: {supported}"
        )
    return key


def _ensure_project_qnn_models_registered() -> None:
    """Register project-only QNN model variants without editing ExecuTorch files."""

    def _register_internvl3_variant_qnn(
        model_name: str,
        *,
        repo_id: str,
        config_filename: str,
        config_cls_name: str,
    ) -> None:
        from executorch.examples.models.internvl3 import (
            convert_weights as convert_internvl3_weights,
        )
        from executorch.examples.qualcomm.oss_scripts import llama as qnn_llama
        from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
        from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
            DECODER_MODEL_VERSION,
            VISION_ENCODER,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
            InternVL3Encoder,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
            LlamaModelWithoutEmbedding,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
            InternVL3_1B_QuantRecipe,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import (
            IMG_TOKEN,
            VLM_SPECIAL_TOKENS,
        )

        if model_name in qnn_llama.SUPPORTED_LLM_MODELS:
            return

        DECODER_MODEL_VERSION[model_name] = "internvl3"
        qnn_llama.LLM_VARIANT_ARCHS[model_name] = LlamaModelWithoutEmbedding
        VLM_SPECIAL_TOKENS[model_name] = {
            IMG_TOKEN: "<IMG_CONTEXT>",
            "fake_wrap_start": "<img>",
            "fake_wrap_end": "</img>",
        }

        params_path = str(
            Path(__file__).resolve().parents[1]
            / "models"
            / "internvl3"
            / config_filename
        )

        rid = repo_id
        ppath = params_path

        @dataclass(init=False, frozen=True)
        class _InternVL3QNNVariant(LLMModelConfig):
            repo_id: str = rid
            params_path: str = ppath
            convert_weights = convert_internvl3_weights
            decoder_model_version = "internvl3"
            transform_weight = False
            instruct_model = True
            num_sharding = 1
            masked_softmax = True
            seq_mse_candidates = 0
            r1 = False
            r2 = False
            r3 = False
            quant_recipe = InternVL3_1B_QuantRecipe

        _InternVL3QNNVariant.__name__ = config_cls_name
        setattr(_InternVL3QNNVariant, VISION_ENCODER, InternVL3Encoder)
        qnn_llama.SUPPORTED_LLM_MODELS[model_name] = _InternVL3QNNVariant()

    _register_internvl3_variant_qnn(
        "internvl3_8b",
        repo_id="OpenGVLab/InternVL3-8B-hf",
        config_filename="8b_config.json",
        config_cls_name="InternVL3_8B",
    )
    _register_internvl3_variant_qnn(
        "internvl3_2b",
        repo_id="OpenGVLab/InternVL3-2B-hf",
        config_filename="2b_config.json",
        config_cls_name="InternVL3_2B",
    )


def _qnn_quant_dtype(name: str):
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

    return getattr(QuantDtype, f"use_{name}")


def _qnn_quant_recipe_class(
    name: str,
    *,
    op_target,
    default_granularity,
    target_granularity,
    base_cls,
    observer,
    force_kv_8bit: bool = False,
    target_extra_kwargs: dict | None = None,
):
    quant_dtype = _qnn_quant_dtype(name)

    class FoundationQNNQuantRecipe(base_cls):
        default_quant_dtype = quant_dtype

        def __init__(self, verbose: bool = False):
            super().__init__()
            from executorch.backends.qualcomm.quantizer.quant_recipe import QuantRecipe

            self.recipe = QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=observer,
                granularity=default_granularity,
                verbose=verbose,
            ).add_node_target(
                {op_target},
                self.default_quant_dtype,
                False,
                act_observer=observer,
                granularity=target_granularity,
                extra_kwargs=target_extra_kwargs,
            )
            if force_kv_8bit:
                from executorch.backends.qualcomm.quantizer.custom_annotation import (
                    annotate_kv_8bit,
                )

                self.recipe.custom_quant_annotations.append(annotate_kv_8bit)

    FoundationQNNQuantRecipe.__name__ = (
        f"FoundationQNN_{name}_{op_target.name().replace('::', '_')}_Recipe"
    )
    return FoundationQNNQuantRecipe


@contextlib.contextmanager
def _qnn_quant_overrides(
    decoder_model_config,
    *,
    decoder_quant: str,
    vision_quant: str,
    embedding_quant: str,
    qnn_kv_quant: str,
) -> Iterator[None]:
    import torch
    from executorch.backends.qualcomm.quantizer.quant_recipe import QuantGranularity
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
    from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
        EncoderQuantRecipe,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
        StaticLLMQuantRecipe,
    )
    from torchao.quantization.pt2e import MinMaxObserver

    original_decoder_recipe = getattr(decoder_model_config, "quant_recipe", None)
    original_vision_recipe = None
    vision_config = getattr(decoder_model_config, "vision_encoder", None)
    if vision_config is not None:
        original_vision_recipe = getattr(vision_config, "quant_recipe", None)
    original_make_quantizer = llm_wrappers.make_quantizer

    decoder_recipe = _qnn_quant_recipe_class(
        decoder_quant,
        op_target=torch.ops.aten.conv2d.default,
        default_granularity=QuantGranularity.PER_TENSOR,
        target_granularity=(
            QuantGranularity.PER_BLOCK
            if decoder_quant == "16a4w_block"
            else QuantGranularity.PER_CHANNEL
        ),
        base_cls=StaticLLMQuantRecipe,
        observer=MinMaxObserver,
        force_kv_8bit=qnn_kv_quant == "8",
        target_extra_kwargs=(
            {"block_size": (1, 32, 1, 1)} if decoder_quant == "16a4w_block" else None
        ),
    )
    vision_recipe = _qnn_quant_recipe_class(
        vision_quant,
        op_target=torch.ops.aten.linear.default,
        default_granularity=QuantGranularity.PER_TENSOR,
        target_granularity=(
            QuantGranularity.PER_BLOCK
            if vision_quant == "16a4w_block"
            else QuantGranularity.PER_CHANNEL
        ),
        base_cls=EncoderQuantRecipe,
        observer=MinMaxObserver,
        target_extra_kwargs=(
            {"block_size": (1, 32)} if vision_quant == "16a4w_block" else None
        ),
    )
    embedding_quant_dtype = _qnn_quant_dtype(embedding_quant)

    def foundation_make_quantizer(*args, **kwargs):
        quant_dtype = kwargs.get("quant_dtype", None)
        if quant_dtype == _qnn_quant_dtype("16a8w"):
            kwargs = dict(kwargs)
            kwargs["quant_dtype"] = embedding_quant_dtype
        return original_make_quantizer(*args, **kwargs)

    object.__setattr__(decoder_model_config, "quant_recipe", decoder_recipe)
    if vision_config is not None:
        setattr(vision_config, "quant_recipe", vision_recipe)
    llm_wrappers.make_quantizer = foundation_make_quantizer
    try:
        yield
    finally:
        object.__setattr__(decoder_model_config, "quant_recipe", original_decoder_recipe)
        if vision_config is not None:
            setattr(vision_config, "quant_recipe", original_vision_recipe)
        llm_wrappers.make_quantizer = original_make_quantizer


@contextlib.contextmanager
def _qnn_true_fp16_overrides(decoder_model_config) -> Iterator[None]:
    """Reserve a scoped hook point for QNN HTP fp16 compile precision."""
    try:
        yield
    finally:
        pass


def _qnn_compile_true_fp16(
    qnn_compile,
    args,
    decoder_model_config,
    pte_filenames,
    _tokenizer,
    _calibration_data,
    is_multimodal,
) -> None:
    """Compile through QNN HTP fp16 without editing ExecuTorch's llama.py."""
    globals_ = qnn_compile.__globals__
    os = globals_["os"]
    MultiModalManager = globals_["MultiModalManager"]
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
        HybridTextDecoder,
    )

    get_soc_to_chipset_map = globals_["get_soc_to_chipset_map"]
    generate_htp_compiler_spec = globals_["generate_htp_compiler_spec"]
    generate_qnn_executorch_compiler_spec = globals_[
        "generate_qnn_executorch_compiler_spec"
    ]
    AUDIO_ENCODER = globals_["AUDIO_ENCODER"]
    TEXT_ENCODER = globals_["TEXT_ENCODER"]
    VISION_ENCODER = globals_["VISION_ENCODER"]
    TOK_EMBEDDING = globals_["TOK_EMBEDDING"]
    TEXT_DECODER = globals_["TEXT_DECODER"]
    TOK_EMBEDDING_GRAPH_NAMES = globals_["TOK_EMBEDDING_GRAPH_NAMES"]
    DECODER_GRAPH_NAMES = globals_["DECODER_GRAPH_NAMES"]

    os.makedirs(args.artifact, exist_ok=True)
    multi_modal_mgr = MultiModalManager(control_args=args, config=decoder_model_config)
    compile_specs = {
        AUDIO_ENCODER: None,
        TEXT_ENCODER: None,
        VISION_ENCODER: None,
        TOK_EMBEDDING: None,
        TEXT_DECODER: None,
    }
    for modality in compile_specs:
        if is_multimodal and modality in {AUDIO_ENCODER, TEXT_ENCODER, VISION_ENCODER}:
            backend_options = generate_htp_compiler_spec(use_fp16=True)
            compile_specs[modality] = generate_qnn_executorch_compiler_spec(
                soc_model=get_soc_to_chipset_map()[args.soc_model],
                backend_options=backend_options,
                shared_buffer=not args.enable_x86_64,
            )
        elif is_multimodal and modality == TOK_EMBEDDING:
            backend_options = generate_htp_compiler_spec(
                use_fp16=True,
                use_weight_sharing=not args.enable_x86_64,
            )
            compile_specs[modality] = [
                generate_qnn_executorch_compiler_spec(
                    soc_model=get_soc_to_chipset_map()[args.soc_model],
                    backend_options=backend_options,
                    shared_buffer=not args.enable_x86_64,
                )
            ] * len(TOK_EMBEDDING_GRAPH_NAMES)
        elif modality == TEXT_DECODER:
            backend_options = generate_htp_compiler_spec(
                use_fp16=True,
                use_multi_contexts=decoder_model_config.num_sharding > 1,
                use_weight_sharing=not args.enable_x86_64,
            )
            compile_specs[modality] = [
                generate_qnn_executorch_compiler_spec(
                    soc_model=get_soc_to_chipset_map()[args.soc_model],
                    backend_options=backend_options,
                    shared_buffer=not args.enable_x86_64,
                    use_mha2sha=True,
                )
            ] * len(DECODER_GRAPH_NAMES)

    # Do not call MultiModalManager.quantize() for true fp16. That method applies
    # PTQ/QDQ recipes before lowering; fp16 should lower the eager modules
    # directly with HTP fp16 compiler specs. Hybrid encoding override only aligns
    # QDQ encodings, so it must be skipped when no QDQ graph exists.
    original_encoding_override = HybridTextDecoder._encoding_override

    def _skip_encoding_override(self, decode_model, prefill_model):
        return None

    HybridTextDecoder._encoding_override = _skip_encoding_override
    try:
        multi_modal_mgr.compile(compile_specs=compile_specs, pte_filenames=pte_filenames)
    finally:
        HybridTextDecoder._encoding_override = original_encoding_override


def export_qnn(args: argparse.Namespace) -> int:
    """Export QNN artifacts without modifying ExecuTorch's Qualcomm sources."""
    if not args.build_path or not args.device or not args.model:
        raise SystemExit(
            "QNN export에는 --build_path (-b), --device (-s), --model (-m) 이 필요합니다."
        )

    from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS
    from executorch.examples.qualcomm.oss_scripts.llama.dataset import DatasetBuilder
    from executorch.examples.qualcomm.oss_scripts.llama import decoder_constants
    from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
    from executorch.examples.qualcomm.oss_scripts.llama.llama import compile as qnn_compile

    _ensure_project_qnn_models_registered()

    audio_encoder = decoder_constants.AUDIO_ENCODER
    text_encoder = decoder_constants.TEXT_ENCODER
    text_decoder = decoder_constants.TEXT_DECODER
    vision_encoder = decoder_constants.VISION_ENCODER
    text_embedding = getattr(
        decoder_constants,
        "TEXT_EMBEDDING",
        getattr(decoder_constants, "TOK_EMBEDDING", "tok_embedding"),
    )

    if args.decoder_model not in SUPPORTED_LLM_MODELS:
        raise SystemExit(
            f"Unsupported decoder_model for QNN: {args.decoder_model}. "
            f"Use one of {list(SUPPORTED_LLM_MODELS.keys())}."
        )

    decoder_model_config = SUPPORTED_LLM_MODELS[args.decoder_model]
    vision_quant = _normalize_qnn_quant(
        getattr(args, "vision_quant", "fp16"), component="vision"
    )
    decoder_quant = _normalize_qnn_quant(
        getattr(args, "decoder_quant", "fp16"), component="decoder"
    )
    embedding_quant = _normalize_qnn_quant(
        getattr(args, "embedding_quant", "fp16"), component="embedding"
    )
    qnn_true_fp16 = {vision_quant, decoder_quant, embedding_quant} == {"fp16"}
    qnn_kv_quant = getattr(args, "qnn_kv_quant", "default")
    if qnn_kv_quant == "8" and qnn_true_fp16:
        raise SystemExit(
            "QNN --qnn_kv_quant 8 requires a quantized QNN path. The true fp16 "
            "compile path skips PTQ/QDQ, so KV 8-bit annotation is not available."
        )
    if "fp16" in {vision_quant, decoder_quant, embedding_quant} and not qnn_true_fp16:
        raise SystemExit(
            "QNN fp16 now means HTP fp16 compile precision and must be used for "
            "vision, decoder, and embedding together. Use explicit 16a16w for "
            "the old 16a16w quantized path."
        )
    # llama.py: max_context_len defaults to max_seq_len; must be >= prefill_ar_len
    max_context_len = getattr(args, "max_context_len", None) or args.max_seq_len
    if max_context_len < args.max_seq_len:
        raise SystemExit(
            f"max_context_len ({max_context_len}) >= max_seq_len ({args.max_seq_len}) 필요"
        )

    if max_context_len < args.prefill_ar_len:
        raise SystemExit(
            f"max_context_len ({max_context_len}) >= prefill_ar_len ({args.prefill_ar_len}) 필요"
        )

    model_mode = getattr(args, "model_mode", "hybrid")
    if model_mode == "kv":
        pte_filename = "kv_llama_qnn"
    elif model_mode == "hybrid":
        pte_filename = "hybrid_llama_qnn"
    elif model_mode == "lookahead":
        pte_filename = "lookahead_llama_qnn"
    else:
        raise SystemExit(f"Unknown model_mode: {model_mode}")

    # MultiModalManager expects all modalities; unused ones (audio/text encoder) get placeholder
    if "internvl" in args.decoder_model.lower():
        pte_filenames = {
            audio_encoder: f"{audio_encoder}_qnn",  # unused for InternVL3
            text_encoder: f"{text_encoder}_qnn",  # unused for InternVL3
            text_decoder: pte_filename,
            vision_encoder: f"{vision_encoder}_qnn",
            text_embedding: f"{text_embedding}_qnn",
        }
    else:
        pte_filenames = {
            audio_encoder: f"{audio_encoder}_qnn",
            text_encoder: f"{text_encoder}_qnn",
            text_decoder: pte_filename,
            vision_encoder: f"{vision_encoder}_qnn",
            text_embedding: f"{text_embedding}_qnn",
        }

    qnn_args = argparse.Namespace(
        artifact=str(Path(args.artifact_root).resolve()),
        decoder_model=args.decoder_model,
        model_mode=model_mode,
        prefill_ar_len=args.prefill_ar_len,
        max_seq_len=args.max_seq_len,
        max_context_len=max_context_len,
        dtype_override=args.dtype,
        vision_quant=vision_quant,
        decoder_quant=decoder_quant,
        embedding_quant=embedding_quant,
        embedding_quantize=None,
        backend="htp",
        soc_model=args.model,
        model=args.model,
        build_folder=args.build_path,
        build_path=args.build_path,
        device=args.device,
        enable_x86_64=getattr(args, "enable_x86_64", False),
        prompt=args.prompts or ["Can you describe this image?"],
        system_prompt=getattr(args, "system_prompt", ""),
        tokenizer_model=getattr(args, "tokenizer_model", None),
        tokenizer_bin=getattr(args, "tokenizer_bin", None),
        image_path=([args.image_path] if getattr(args, "image_path", None) else []),
        audio_path=[],
        params=getattr(args, "params", None),
        checkpoint=getattr(args, "checkpoint", None),
        model_path=getattr(args, "model_path", None),
        window=getattr(args, "window", 8),
        ngram=getattr(args, "ngram", 5),
        gcap=getattr(args, "gcap", 8),
        temperature=0.0,
        shared_buffer=False,
        use_attention_sink=None,
        verbose=False,
        pre_gen_pte=None,
        compile_only=True,
        eval_methods=["prompt_eval"],
        tasks=None,
        limit=1,
        num_fewshot=0,
        quant_recipe_suggestion=False,
        calibration_num_threads=0,
        ip=None,
        port=-1,
    )
    tokenizer_wrapper = TokenizerWrapper(qnn_args, decoder_model_config)
    runtime_tokenizer_path, tokenizer, chat_template = (
        tokenizer_wrapper.get_runtime_tokenizer(
            qnn_args.tokenizer_model, qnn_args.tokenizer_bin
        )
    )

    dataset_builder = DatasetBuilder(qnn_args, decoder_model_config, tokenizer_wrapper)
    calibration_data = dataset_builder.prepare_calibration_dataset(
        qnn_args.prompt, chat_template
    )

    text_decoder_pte_path = Path(qnn_args.artifact) / f"{pte_filenames[text_decoder]}.pte"
    encoder_pte_path = Path(qnn_args.artifact) / f"{pte_filenames[vision_encoder]}.pte"
    text_embedding_pte_path = Path(qnn_args.artifact) / f"{pte_filenames[text_embedding]}.pte"

    is_multimodal = any(
        hasattr(decoder_model_config, modality)
        for modality in (vision_encoder, audio_encoder)
    )

    if qnn_true_fp16:
        with _qnn_true_fp16_overrides(decoder_model_config):
            _qnn_compile_true_fp16(
                qnn_compile,
                qnn_args,
                decoder_model_config,
                pte_filenames,
                tokenizer,
                calibration_data,
                is_multimodal,
            )
    else:
        with _qnn_quant_overrides(
            decoder_model_config,
            decoder_quant=decoder_quant,
            vision_quant=vision_quant,
            embedding_quant=embedding_quant,
            qnn_kv_quant=qnn_kv_quant,
        ):
            qnn_compile(
                qnn_args,
                decoder_model_config,
                pte_filenames,
                tokenizer,
                calibration_data,
                is_multimodal,
            )

    manifest = build_manifest(
        artifact_root=Path(qnn_args.artifact),
        backend="qnn",
        variant=args.decoder_model,
        vision_encoder_pte=encoder_pte_path,
        text_embedding_pte=text_embedding_pte_path,
        text_decoder_pte=text_decoder_pte_path,
        tokenizer_path=Path(runtime_tokenizer_path),
        export={
            "max_seq_len": args.max_seq_len,
            "max_context_len": max_context_len,
            "prefill_ar_len": args.prefill_ar_len,
            "model_mode": model_mode,
            "soc_model": args.model,
            "qnn_fp16_compile": qnn_true_fp16,
            "qnn_kv_quant": qnn_kv_quant,
            "qnn_quant_note": (
                "fp16 uses QNN HTP fp16 compile precision; use 16a16w for the old quantized path"
            ),
        },
        quant={
            "dtype": args.dtype,
            "vision": vision_quant,
            "decoder": decoder_quant,
            "embedding": embedding_quant,
            "kv_cache": "8" if qnn_kv_quant == "8" else None,
        },
    )
    write_manifest(manifest, Path(qnn_args.artifact) / FOUNDATION_MANIFEST_FILENAME)

    return 0
