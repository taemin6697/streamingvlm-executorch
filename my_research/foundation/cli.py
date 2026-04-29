from __future__ import annotations

import argparse
from pathlib import Path

from my_research.foundation.export import export_with_backend
from my_research.foundation.host.launcher import run_with_manifest
from my_research.foundation.manifest import load_manifest


def _cmd_export(args: argparse.Namespace) -> int:
    return export_with_backend(args)


def _cmd_inspect(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    print(f"backend: {manifest.backend}")
    print(f"model_family: {manifest.model_family}")
    print(f"variant: {manifest.variant}")
    print(f"runner_type: {manifest.runner_type}")
    print("paths:")
    for key, value in manifest.paths.items():
        print(f"  {key}: {value}")
    if manifest.export:
        print("export:")
        for key, value in manifest.export.items():
            print(f"  {key}: {value}")
    if manifest.quant:
        print("quant:")
        for key, value in manifest.quant.items():
            print(f"  {key}: {value}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    eval_mode = args.eval_mode
    if eval_mode is None:
        eval_mode = 1 if manifest.backend == "qnn" else 0
    return run_with_manifest(
        Path(args.manifest),
        build_path=args.build_path,
        device=args.device,
        model=args.model,
        image=args.image,
        video=args.video,
        questions=args.questions,
        timestamps=args.timestamps,
        seq_len=args.seq_len,
        force_generate_token=args.force_generate_token,
        temperature=args.temperature,
        eval_mode=eval_mode,
        save_log=args.save_log,
        stream=args.stream,
        runner_binary=args.runner_binary,
        force_push=args.force_push,
        vision_only=args.vision_only,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified foundation CLI for QNN/XNNPACK/Vulkan multimodal export and run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="Export foundation artifacts natively for QNN, XNNPACK, or Vulkan.",
    )
    export_parser.add_argument("--backend", required=True, choices=["qnn", "xnnpack", "vulkan"])
    export_parser.add_argument("--artifact_root", required=True)
    export_parser.add_argument("--decoder_model", required=True)
    export_parser.add_argument("--max_seq_len", type=int, default=1024)
    export_parser.add_argument(
        "--max_context_len",
        type=int,
        default=None,
        help="미지정 시 max_seq_len과 동일. prefill_ar_len 이상이어야 함.",
    )
    export_parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    export_parser.add_argument(
        "--vision_quant",
        default="fp16",
        help=(
            "Backend-specific vision quant mode. For QNN, fp16 uses HTP fp16 "
            "compile precision; explicit 16a16w uses the quantized 16a16w path. "
            "Other QNN modes: 16a8w, 16a4w, 16a4w_block, 8a8w, 8a4w."
        ),
    )
    export_parser.add_argument(
        "--decoder_quant",
        default="fp16",
        help=(
            "Backend-specific decoder quant mode. For QNN, fp16 uses HTP fp16 "
            "compile precision; explicit 16a16w uses the quantized 16a16w path. "
            "Other QNN modes: 16a8w, 16a4w, 16a4w_block, 8a8w, 8a4w. "
            "For Vulkan: fp16 and vulkan_8w use the Vulkan PT2E quantizer; "
            "4w/8da8w/8da4w use the upstream Llama qmode source-transform path."
        ),
    )
    export_parser.add_argument(
        "--embedding_quant",
        default="fp16",
        help=(
            "Backend-specific embedding quant mode. For QNN, fp16 uses HTP fp16 "
            "compile precision; explicit 16a16w uses the quantized 16a16w path. "
            "Other QNN modes: 16a8w, 16a4w, 16a4w_block, 8a8w, 8a4w."
        ),
    )
    export_parser.add_argument("--model_path", default=None)
    export_parser.add_argument("--checkpoint", default=None)
    export_parser.add_argument("--params", default=None)
    export_parser.add_argument("--encoder_weights", default=None)
    export_parser.add_argument("--calibration_images", nargs="+", default=None)
    export_parser.add_argument("--calibration_num", type=int, default=8)
    export_parser.add_argument(
        "--text_group_size",
        type=int,
        default=None,
        help="TorchAo text weight group size. Defaults to 64 for Vulkan 8da4w, otherwise 128.",
    )
    export_parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    export_parser.add_argument(
        "--use_sdpa_with_kv_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    export_parser.add_argument(
        "--quantize_kv_cache",
        action="store_true",
        help=(
            "For XNNPACK/Vulkan exports, use upstream int8 per-token quantized "
            "KV-cache. This is separate from QNN KV I/O quantization."
        ),
    )
    export_parser.add_argument(
        "--qnn_kv_quant",
        choices=["default", "8"],
        default="default",
        help=(
            "For QNN quantized exports, override decoder KV I/O quantization. "
            "'8' forces annotate_kv_8bit; 'default' follows the quant recipe."
        ),
    )
    export_parser.add_argument(
        "--decoder_input_mode",
        choices=["token_ids", "embeddings"],
        default="token_ids",
        help=(
            "Vulkan decoder input mode. token_ids matches upstream Llama-style export; "
            "embeddings is required for InternVL image-feature merging."
        ),
    )
    export_parser.add_argument(
        "--vulkan_xnnpack_fallback",
        action="store_true",
        help="For Vulkan export, lower unsupported non-decoder subgraphs with XNNPACK.",
    )
    export_parser.add_argument(
        "--vulkan-force-fp16",
        "--vulkan_force_fp16",
        action="store_true",
        dest="vulkan_force_fp16",
        help=(
            "For Vulkan export, keep the requested export dtype but force fp16 "
            "storage/compute in the Vulkan delegate. This allows fp32 graph export "
            "with Vulkan force-fp16."
        ),
    )
    export_parser.add_argument(
        "--vulkan_debug_fp32_kv_cache",
        action="store_true",
        help=(
            "Debug only: keep Vulkan fp16 decoder export but force transformed "
            "KV-cache buffers and update inputs to fp32."
        ),
    )
    export_parser.add_argument(
        "--vulkan_debug_block_sdpa_delegate",
        action="store_true",
        help=(
            "Debug only: keep sdpa_with_kv_cache in the graph, but block it "
            "from Vulkan delegation so the custom op runs outside Vulkan."
        ),
    )
    export_parser.add_argument(
        "--decoder_only_from",
        default=None,
        help=(
            "Export only the text decoder and reuse vision/embedding/tokenizer "
            "artifacts from this existing foundation artifact root or manifest."
        ),
    )
    export_parser.add_argument(
        "--dynamic_shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dynamic sequence shapes for XNNPACK/Vulkan text embedding/decoder export.",
    )
    export_parser.add_argument(
        "--disable_dynamic_shape",
        action="store_false",
        dest="dynamic_shape",
        help="Disable dynamic sequence shapes for XNNPACK/Vulkan text embedding/decoder export.",
    )
    export_parser.add_argument("-b", "--build_path", default=None)
    export_parser.add_argument("-s", "--device", default=None)
    export_parser.add_argument("-m", "--model", default=None)
    export_parser.add_argument("--prompts", nargs="+", default=None)
    export_parser.add_argument("--system_prompt", default="")
    export_parser.add_argument("--tokenizer_model", default=None)
    export_parser.add_argument("--tokenizer_bin", default=None)
    export_parser.add_argument("--image_path", default=None)
    export_parser.add_argument(
        "--model_mode",
        default="hybrid",
        choices=["kv", "hybrid", "lookahead"],
    )
    export_parser.add_argument("--prefill_ar_len", type=int, default=32)
    export_parser.add_argument("--ngram", type=int, default=5)
    export_parser.add_argument("--window", type=int, default=8)
    export_parser.add_argument("--gcap", type=int, default=8)
    export_parser.add_argument(
        "--enable_x86_64",
        action="store_true",
        help="QNN x86 emulator build flag.",
    )
    export_parser.set_defaults(func=_cmd_export)

    inspect_parser = subparsers.add_parser("inspect-manifest", help="Print manifest details.")
    inspect_parser.add_argument("manifest")
    inspect_parser.set_defaults(func=_cmd_inspect)

    run_parser = subparsers.add_parser("run", help="Run using a foundation manifest.")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("-b", "--build_path", default=None)
    run_parser.add_argument("-s", "--device", default=None)
    run_parser.add_argument("-m", "--model", default=None)
    run_parser.add_argument("--runner_binary", default=None)
    run_parser.add_argument("--image", default=None)
    run_parser.add_argument("--video", default=None)
    run_parser.add_argument("--questions", nargs="+", default=None)
    run_parser.add_argument("--timestamps", nargs="+", type=float, default=None)
    run_parser.add_argument("--seq_len", type=int, default=None)
    run_parser.add_argument(
        "--force_generate_token",
        type=int,
        default=None,
        help=(
            "If set, generate exactly this many tokens and ignore EOS/stop "
            "tokens. Overrides --seq_len for generation length."
        ),
    )
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument(
        "--eval_mode",
        type=int,
        default=None,
        help="0=KV, 1=Hybrid, 2=Lookahead. 미지정 시 QNN→1, XNNPACK→0 자동.",
    )
    run_parser.add_argument("--stream", action="store_true")
    run_parser.add_argument("--save_log", action="store_true")
    run_parser.add_argument(
        "--vision_only",
        action="store_true",
        help="Run only the vision encoder and dump vision output logs.",
    )
    run_parser.add_argument(
        "--force_push",
        action="store_true",
        help="Re-push runner/model files even when cached on the Android device.",
    )
    run_parser.set_defaults(func=_cmd_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
