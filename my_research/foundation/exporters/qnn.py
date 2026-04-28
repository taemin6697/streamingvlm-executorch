# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Foundation-native QNN export that treats Qualcomm llama code as a dependency."""

from __future__ import annotations

import argparse
from pathlib import Path

from my_research.foundation.manifest import (
    FOUNDATION_MANIFEST_FILENAME,
    build_manifest,
    write_manifest,
)


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
        vision_quant=getattr(args, "vision_quant", "fp16"),
        decoder_quant=getattr(args, "decoder_quant", "fp16"),
        embedding_quant=getattr(args, "embedding_quant", "fp16"),
        embedding_quantize=(
            args.embedding_quant if args.embedding_quant != "fp16" else None
        ),
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
        },
        quant={
            "dtype": args.dtype,
            "vision": getattr(args, "vision_quant", "fp16"),
            "decoder": getattr(args, "decoder_quant", "fp16"),
            "embedding": getattr(args, "embedding_quant", "fp16"),
        },
    )
    write_manifest(manifest, Path(qnn_args.artifact) / FOUNDATION_MANIFEST_FILENAME)

    return 0
