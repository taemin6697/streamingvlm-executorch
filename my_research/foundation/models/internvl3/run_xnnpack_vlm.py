#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from transformers.image_utils import load_image

_MODEL_CONFIGS = {
    "internvl3_1b": ("OpenGVLab/InternVL3-1B-hf", 448, 448, "<IMG_CONTEXT>"),
    "internvl3_2b": ("OpenGVLab/InternVL3-2B-hf", 448, 448, "<IMG_CONTEXT>"),
    "internvl3_8b": ("OpenGVLab/InternVL3-8B-hf", 448, 448, "<IMG_CONTEXT>"),
}


def _run(cmd, *, check=True, capture_output=False):
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _adb_base(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _infer_tokenizer_path(model_path: Path) -> Path:
    artifact_dir = model_path.parent / f"{model_path.stem}_artifacts" / "tokenizer"
    return artifact_dir / "tokenizer.json"


def _load_processor(model_name: str):
    from transformers import AutoProcessor

    model_id, _, _, _ = _MODEL_CONFIGS[model_name]
    return AutoProcessor.from_pretrained(model_id)


def _preprocess_image(image_source: str, out_path: Path, processor, model_name: str):
    from PIL import Image

    _, img_h, img_w, image_token = _MODEL_CONFIGS[model_name]
    if image_source.startswith(("http://", "https://")):
        image = load_image(image_source)
    else:
        image = Image.open(image_source).convert("RGB")

    try:
        inputs = processor(
            text=image_token,
            images=[image],
            return_tensors="pt",
            crop_to_patches=False,
            size={"height": img_h, "width": img_w},
        )
    except TypeError:
        inputs = processor(
            text=image_token,
            images=[image],
            return_tensors="pt",
        )

    pixel_values = inputs["pixel_values"].detach().cpu().numpy()
    if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
        pixel_values = pixel_values[0]
    pixel_values.astype(np.float32).tofile(out_path)


def extract_image(image_path: str, out_dir: Path, model_name: str) -> int:
    processor = _load_processor(model_name)
    out_path = out_dir / "frame_0000.bin"
    _preprocess_image(image_path, out_path, processor, model_name)
    print(f"[internvl3_xnnpack] 이미지 1프레임 → {out_path}")
    return 1


def extract_frames(
    video_path: str,
    fps: float,
    out_dir: Path,
    model_name: str,
    max_frames: int | None = None,
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "[internvl3_xnnpack] opencv-python 이 필요합니다: pip install opencv-python"
        ) from exc

    from PIL import Image

    processor = _load_processor(model_name)
    _, img_h, img_w, image_token = _MODEL_CONFIGS[model_name]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"[internvl3_xnnpack] 비디오를 열 수 없습니다: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(native_fps / fps))
    frame_count = 0
    native_idx = 0

    print(
        f"[internvl3_xnnpack] 비디오 FPS={native_fps:.1f}, "
        f"샘플링 간격={frame_interval}프레임"
    )

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if native_idx % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            out_path = out_dir / f"frame_{frame_count:04d}.bin"
            try:
                inputs = processor(
                    text=image_token,
                    images=[image],
                    return_tensors="pt",
                    crop_to_patches=False,
                    size={"height": img_h, "width": img_w},
                )
            except TypeError:
                inputs = processor(
                    text=image_token,
                    images=[image],
                    return_tensors="pt",
                )

            pixel_values = inputs["pixel_values"].detach().cpu().numpy()
            if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
                pixel_values = pixel_values[0]
            pixel_values.astype(np.float32).tofile(out_path)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
        native_idx += 1

    cap.release()
    print(f"[internvl3_xnnpack] {frame_count}개 프레임 → {out_dir}")
    return frame_count


def push_and_run(args, frame_count: int, staging_dir: Path):
    adb = _adb_base(args.serial)
    remote_root = args.device_workdir.rstrip("/")
    remote_frames = f"{remote_root}/frames"
    remote_model = f"{remote_root}/{Path(args.model_path).name}"
    remote_tokenizer = f"{remote_root}/{Path(args.tokenizer_path).name}"
    remote_runner = f"{remote_root}/{Path(args.runner_binary).name}"
    remote_output = f"{remote_root}/output.txt"

    _run(adb + ["shell", "mkdir", "-p", remote_root, remote_frames])
    _run(adb + ["push", str(args.model_path), remote_model])
    _run(adb + ["push", str(args.tokenizer_path), remote_tokenizer])
    _run(adb + ["push", str(args.runner_binary), remote_runner])
    _run(adb + ["push", str(staging_dir), remote_frames])
    _run(adb + ["shell", "chmod", "+x", remote_runner])

    shell_cmd = [
        remote_runner,
        f"--model_path={remote_model}",
        f"--tokenizer_path={remote_tokenizer}",
        f"--frame_dir={remote_frames}/{staging_dir.name}",
        f"--frame_count={frame_count}",
        f"--question={args.question}",
        f"--max_new_tokens={args.max_new_tokens}",
        f"--temperature={args.temperature}",
        f"--output_path={remote_output}",
    ]
    if args.cpu_threads is not None:
        shell_cmd.append(f"--cpu_threads={args.cpu_threads}")

    remote_dump = f"{remote_root}/input_dump.txt" if args.dump_input else None
    if remote_dump:
        shell_cmd.append(f"--dump_input_path={remote_dump}")

    if getattr(args, "save_log", False):
        shell_cmd.append("--save_log")

    print("[internvl3_xnnpack] adb shell 실행 중...")
    remote_cmd = " ".join(shlex.quote(part) for part in shell_cmd)
    _run(adb + ["shell", remote_cmd])

    local_output = Path(args.local_output or "internvl3_xnnpack_output.txt").resolve()
    _run(adb + ["pull", remote_output, str(local_output)])
    print(f"[internvl3_xnnpack] 출력 저장: {local_output}")

    # Pull proc.csv, mem.csv, tokens.csv (save_log 시)
    for suffix in [".proc.csv", ".mem.csv", ".tokens.csv"]:
        try:
            local_extra = Path(str(local_output) + suffix)
            _run(adb + ["pull", remote_output + suffix, str(local_extra)])
            print(f"[internvl3_xnnpack] {suffix} 저장: {local_extra}")
        except Exception:
            pass

    if remote_dump and args.dump_input:
        local_dump = Path(args.dump_input).resolve()
        _run(adb + ["pull", remote_dump, str(local_dump)])
        print(f"[internvl3_xnnpack] 입력 덤프 저장: {local_dump}")

    return local_output


def _make_param_folder_name(args) -> str:
    """파라미터 기반 폴더명 생성 (save_log 하위용, QNN stream_vlm과 유사)."""
    input_type = "video" if args.video else ("image" if args.image else "text")
    run_type = "batch"  # CPU는 배치만 지원
    lazy = "lazy" if getattr(args, "lazy_kv_alloc", False) else "nolazy"
    fps_tag = str(args.fps).replace(".", "p")
    frame_tag = (
        f"frames{args.decode_after_frames}"
        if getattr(args, "decode_after_frames", None) is not None
        else "framesall"
    )
    parts = [
        args.decoder_model,
        "cpu",
        run_type,
        f"seq{args.seq_len}",
        input_type,
        f"fps{fps_tag}",
        frame_tag,
        f"eval{getattr(args, 'eval_mode', 0)}",
        lazy,
    ]
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in "_".join(parts)
    )
    return safe


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run InternVL3 multimodal XNNPACK example on Android CPU."
    )
    parser.add_argument(
        "--decoder_model",
        default="internvl3_1b",
        choices=sorted(_MODEL_CONFIGS.keys()),
        help="InternVL3 variant used during export.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=Path,
        help="Combined multimodal .pte generated by export_xnnpack_multimodal.py",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        type=Path,
        help="Tokenizer json path. Defaults to sibling *_artifacts/tokenizer/tokenizer.json",
    )
    parser.add_argument(
        "--runner_binary",
        required=True,
        type=Path,
        help="Built Android runner binary (internvl3_xnnpack_runner).",
    )
    parser.add_argument(
        "-s", "--serial",
        default=None,
        dest="serial",
        help="adb device serial (QNN -s와 동일).",
    )
    parser.add_argument(
        "--device_workdir",
        default="/data/local/tmp/internvl3_xnnpack",
        help="Remote adb working directory.",
    )
    parser.add_argument("--image", default=None, help="Single image path or URL.")
    parser.add_argument("--video", default=None, help="Video path for batch mode.")
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame sampling rate when --video is used.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Question asked after image/video prefill (단일). --questions와 호환.",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="질문 리스트 (QNN stream_vlm과 동일). 지정 시 --question 무시.",
    )
    parser.add_argument(
        "--decode_after_frames",
        type=int,
        default=None,
        help="배치 모드: 첫 N프레임만 사용. 미지정 시 전체 프레임.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="최대 시퀀스 길이 (기본 2048).",
    )
    parser.add_argument(
        "--eval_mode",
        type=int,
        default=0,
        help="0=KV, 1=Hybrid 등 (호환용, CPU에서는 무시될 수 있음).",
    )
    parser.add_argument(
        "--lazy_kv_alloc",
        action="store_true",
        help="KV 캐시 lazy 할당 (호환용, CPU runner에서 지원 시).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to the runner.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=-1,
        help="CPU thread count for the runner.",
    )
    parser.add_argument(
        "--local_output",
        default=None,
        help="Local file to store the generated response.",
    )
    parser.add_argument(
        "--dump_input",
        default=None,
        metavar="PATH",
        help="Dump C++ runner 실제 입력 (full prompt + input_ids) to file.",
    )
    parser.add_argument(
        "--save_log",
        action="store_true",
        help="my_save/save_log/파라미터폴더/ 하위에 txt, proc.csv, mem.csv, tokens.csv 저장.",
    )
    parser.add_argument(
        "--save_log_dir",
        default=None,
        help="save_log 저장 기준 경로 (기본: workspace/my_save/save_log).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if (args.image is None) == (args.video is None):
        raise SystemExit("정확히 하나의 입력만 지정해야 합니다: --image 또는 --video")

    # 질문: --questions 우선, 없으면 --question, 둘 다 없으면 기본값
    if args.questions:
        args.question = args.questions[0]
    elif args.question is None:
        args.question = "Can you describe this image?"

    args.model_path = args.model_path.resolve()
    if args.tokenizer_path is None:
        args.tokenizer_path = _infer_tokenizer_path(args.model_path)
    args.tokenizer_path = args.tokenizer_path.resolve()
    args.runner_binary = args.runner_binary.resolve()

    for required_path in (args.model_path, args.tokenizer_path, args.runner_binary):
        if not required_path.exists():
            raise SystemExit(f"필수 파일을 찾을 수 없습니다: {required_path}")

    # save_log: my_save/save_log/파라미터폴더/stream_output.txt
    if getattr(args, "save_log", False):
        stream_root = Path(__file__).resolve().parents[4]
        base_dir = (
            Path(args.save_log_dir).resolve()
            if getattr(args, "save_log_dir", None)
            else stream_root / "my_save" / "save_log"
        )
        param_folder = _make_param_folder_name(args)
        out_dir = base_dir / param_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        args.local_output = str(out_dir / "stream_output.txt")
        print(f"[internvl3_xnnpack] save_log: {args.local_output}")

    with tempfile.TemporaryDirectory(prefix="internvl3_xnnpack_frames_") as tmpdir:
        frame_dir = Path(tmpdir)
        if args.image:
            frame_count = extract_image(args.image, frame_dir, args.decoder_model)
        else:
            extracted = extract_frames(
                args.video, args.fps, frame_dir, args.decoder_model
            )
            # decode_after_frames: 배치 모드에서 사용할 프레임 수 제한
            if getattr(args, "decode_after_frames", None) is not None:
                frame_count = min(extracted, args.decode_after_frames)
            else:
                frame_count = extracted
        if frame_count <= 0:
            raise SystemExit("전처리된 프레임이 없습니다.")

        local_output = push_and_run(args, frame_count, frame_dir)
        if local_output.exists():
            print(local_output.read_text(encoding="utf-8", errors="ignore"))


if __name__ == "__main__":
    main()
