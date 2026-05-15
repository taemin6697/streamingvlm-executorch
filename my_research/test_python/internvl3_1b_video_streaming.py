#!/usr/bin/env python3
"""
InternVL3 (HF) — **VStream-QA streaming (online / RVS)** benchmark driver.

Dataset: `IVGSZ/VStream-QA` on Hugging Face (annotations under `vstream-realtime/`).
Streaming subsets: **RVS-Ego**, **RVS-Movie** (`test_qa_ego4d.json`, `test_qa_movienet.json`).
Offline VS-* uses `vstream/` — use `internvl3_1b_video_chat.py` + clipped mp4 if needed.

Frame packs (included in the HF dataset snapshot):
  - `vstream-realtime/movienet_frames_online.zip` → unzip → `movienet_frames/<video_id>/*.jpg`
  - `vstream-realtime/ego4d_frames_online.part*` (~26GB split zip) → merge/unzip → frame folders
    (see dataset README; Ego4d *license* for raw video is separate from this pack).

Requires transformers 4.x (`pip install 'transformers>=4.45,<5'`), torch, decord (optional for `--video`).

Examples:
  pip install huggingface_hub
  python internvl3_1b_video_streaming.py --download-vstream-only

  # After unzip movienet_frames_online.zip under vstream-realtime/
  python internvl3_1b_video_streaming.py --vstream-source movienet --sample-limit 2

  # Raw mp4 fallback (surveillance demo)
  python internvl3_1b_video_streaming.py --video path/to.mp4 --num-segments 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import transformers
from transformers import AutoModel, AutoTokenizer

try:
    from decord import VideoReader, cpu as decord_cpu
except ImportError:
    VideoReader = None
    decord_cpu = None


def _require_transformers_v4() -> None:
    major_str = transformers.__version__.split(".")[0]
    try:
        major = int(major_str)
    except ValueError:
        return
    if major >= 5:
        sys.exit(
            f"Incompatible transformers {transformers.__version__}. "
            "OpenGVLab InternVL (trust_remote_code) expects transformers 4.x.\n"
            "Fix: pip install 'transformers>=4.45,<5'"
        )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = REPO_ROOT / "my_research/foundation_llamacpp/sample_images/surveil_8.mp4"
RESULTS_PARENT = REPO_ROOT / "my_research/test_python/results"
VSTREAM_DEFAULT_ROOT = REPO_ROOT / "my_research/test_python/data/VStream-QA"
VSTREAM_REALTIME = "vstream-realtime"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def download_vstream_qa(destination: Path) -> Path:
    """Snapshot `IVGSZ/VStream-QA` (dataset repo) via huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from e

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="IVGSZ/VStream-QA",
        repo_type="dataset",
        local_dir=str(destination),
    )
    return destination


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def uniform_sample_indices(n_frames: int, num_segments: int) -> np.ndarray:
    """Same spacing idea as video `get_index`: centers in each temporal bin."""
    if n_frames <= 0:
        return np.array([], dtype=np.int64)
    if num_segments <= 0:
        num_segments = 1
    max_frame = n_frames - 1
    seg_size = float(max_frame + 1) / num_segments
    return np.array(
        [min(max_frame, int((seg_size / 2) + np.round(seg_size * idx))) for idx in range(num_segments)],
        dtype=np.int64,
    )


def movienet_jpg_sort_key(path: Path) -> tuple[int, int]:
    m = re.match(r"shot_(\d+)_img_(\d+)\.(jpg|jpeg)$", path.name, re.I)
    if not m:
        return (10**9, 99)
    return int(m.group(1)), int(m.group(2))


def list_movienet_frames_in_qa_interval(folder: Path, start_fn: str, end_fn: str) -> list[Path]:
    jpgs = sorted(folder.glob("*.jpg"), key=movienet_jpg_sort_key)
    names = [p.name for p in jpgs]
    if start_fn not in names or end_fn not in names:
        raise FileNotFoundError(
            f"MovieNet folder {folder}: missing start={start_fn!r} or end={end_fn!r} "
            f"(have {len(jpgs)} jpgs)"
        )
    i0, i1 = names.index(start_fn), names.index(end_fn)
    if i0 > i1:
        i0, i1 = i1, i0
    return jpgs[i0 : i1 + 1]


def list_ego_stream_frames(folder: Path, duration_seconds: int | None) -> list[Path]:
    """Ego RVS: fps≈1 jpgs `000001.jpg`… in `folder`; optional cap by QA `duration` seconds."""

    def frame_key(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            return 10**12

    imgs = sorted(folder.glob("*.jpg"), key=frame_key)
    if not imgs:
        imgs = sorted(folder.glob("*.png"), key=frame_key)
    if duration_seconds is not None and duration_seconds > 0:
        imgs = imgs[: int(duration_seconds)]
    return imgs


def pixel_values_from_images(
    images: list[Image.Image],
    *,
    input_size: int,
    max_num: int,
) -> tuple[torch.Tensor, list[int]]:
    transform = build_transform(input_size=input_size)
    pixel_values_list: list[torch.Tensor] = []
    num_patches_list: list[int] = []
    for img in images:
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)
    out = torch.cat(pixel_values_list, dim=0)
    return out, num_patches_list


def load_vstream_frames_pixel_values(
    frame_paths: list[Path],
    *,
    num_segments: int,
    input_size: int,
    max_num: int,
) -> tuple[torch.Tensor, list[int]]:
    idx = uniform_sample_indices(len(frame_paths), num_segments)
    chosen = [frame_paths[i] for i in idx]
    pil_images = [Image.open(p).convert("RGB") for p in chosen]
    return pixel_values_from_images(pil_images, input_size=input_size, max_num=max_num)


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video(video_path: Path, bound=None, input_size=448, max_num=1, num_segments=32):
    if VideoReader is None:
        sys.exit("decord is required for --video mode: pip install decord")
    vr = VideoReader(str(video_path), ctx=decord_cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[int(frame_index)].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list


def write_token_tsv(path: Path, tokenizer, text: str, *, add_special_tokens: bool = False) -> int:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    rows = ["# idx\ttoken_id\tsubword_piece\tdecode_single_id"]
    for i, tid in enumerate(ids):
        subtokens = tokenizer.convert_ids_to_tokens([tid])
        sub = subtokens[0] if subtokens else ""
        frag = tokenizer.decode([tid])
        safe_sub = sub.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
        safe_frag = frag.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
        rows.append(f"{i}\t{tid}\t{safe_sub}\t{safe_frag}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(ids)


def write_plain(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def hf_id_piece_lines(tokenizer, text: str, *, add_special_tokens: bool = False) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    lines = []
    for tid in ids:
        subtokens = tokenizer.convert_ids_to_tokens([tid])
        piece = subtokens[0] if subtokens else ""
        piece_esc = piece.replace("\t", " ").replace("\r", "\\r").replace("\n", "\\n")
        lines.append(f"{tid}\t{piece_esc}")
    return lines


def build_foundation_token_io_document(
    tokenizer,
    *,
    question: str,
    num_patches_list: list[int],
    vision_slots_per_patch: int,
    response1: str,
) -> str:
    lines_out: list[str] = []
    lines_out.append("User:")
    lines_out.append(question)
    lines_out.append("")
    lines_out.append("Assistant:")
    lines_out.append(response1)
    lines_out.append("")

    lines_out.append("--- HF_TOKENIZER (full question string; id\\tsubword_piece) ---")
    lines_out.extend(hf_id_piece_lines(tokenizer, question))
    lines_out.append("")
    lines_out.append("")

    total_vis = sum(n * vision_slots_per_patch for n in num_patches_list)
    lines_out.append("--- PREFILL_STYLE (per-frame synthetic vision rows) ---")
    lines_out.append(
        f"# sum(num_patches_list)*{vision_slots_per_patch} = {total_vis} placeholders total "
        f"(denominator below is per sampled frame chunk)"
    )
    lines_out.append("--- PREFILL ---")

    for fi, n_patch in enumerate(num_patches_list):
        chunk = f"Frame{fi + 1}: <image>\n"
        lines_out.append(f"## FRAME {fi + 1} TEXT")
        lines_out.extend(hf_id_piece_lines(tokenizer, chunk))
        n_slots = n_patch * vision_slots_per_patch
        lines_out.append(f"## FRAME {fi + 1} IMAGE n_placeholder_tokens={n_slots}")
        for j in range(n_slots):
            lines_out.append(f"-1\t<VISION_KV_SLOT {j + 1}/{n_slots}>")

    tail_start = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
    tail_q = question[len(tail_start) :] if question.startswith(tail_start) else ""
    if tail_q:
        lines_out.append("## TAIL_QUESTION TEXT")
        lines_out.extend(hf_id_piece_lines(tokenizer, tail_q))
    lines_out.append("")
    lines_out.append("")

    lines_out.append("--- DECODE (assistant; tokenizer.encode on reply text) ---")
    lines_out.extend(hf_id_piece_lines(tokenizer, response1))
    lines_out.append("")
    return "\n".join(lines_out)


def load_model(args: argparse.Namespace):
    path = args.model
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def resolve_movienet_frames_root(vstream_root: Path) -> Path:
    rt = vstream_root / VSTREAM_REALTIME / "movienet_frames"
    return rt


def resolve_ego_frames_root(vstream_root: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    # After `7z x ego4d_frames_online.partaa`, clips live under ego4d_frames/<video_id>/
    rt = vstream_root / VSTREAM_REALTIME
    for candidate in (
        rt / "ego4d_frames",
        rt / "ego4d_frames_online",
        rt / "frames",
    ):
        if candidate.is_dir():
            return candidate
    return rt / "ego4d_frames"


def parse_args():
    p = argparse.ArgumentParser(description="InternVL VStream-QA streaming (RVS) + optional mp4 fallback")
    p.add_argument(
        "--download-vstream-only",
        action="store_true",
        help="Download IVGSZ/VStream-QA into --vstream-root and exit",
    )
    p.add_argument(
        "--vstream-root",
        type=Path,
        default=VSTREAM_DEFAULT_ROOT,
        help="Local clone of Hugging Face dataset IVGSZ/VStream-QA",
    )
    p.add_argument(
        "--vstream-source",
        choices=("movienet", "ego4d", "none"),
        default="movienet",
        help="RVS subset: movienet vs ego4d frame folders under vstream-realtime",
    )
    p.add_argument(
        "--ego-frames-root",
        type=Path,
        default=None,
        help="Parent of clip folders (each named video_id e.g. 000000/) with fps≈1 jpgs",
    )
    p.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="First row index in test_qa_*.json",
    )
    p.add_argument("--sample-limit", type=int, default=1, help="Max QA items to run (streaming bench)")
    p.add_argument("--video", type=Path, default=None, help="If set, run single-file demo instead of VStream")
    p.add_argument("--model", type=str, default="OpenGVLab/InternVL3-1B")
    p.add_argument("--num-segments", type=int, default=8, help="Uniform subsample count over frame list")
    p.add_argument("--max-num", type=int, default=1, help="InternVL dynamic_preprocess max_num tiles per frame")
    p.add_argument("--input-size", type=int, default=448)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    p.add_argument("--question", type=str, default=None, help="Only for --video mode (tail after Frame lines)")
    p.add_argument("--results-root", type=Path, default=RESULTS_PARENT)
    p.add_argument("--vision-slots-per-patch", type=int, default=256)
    return p.parse_args()


def run_single_video(args, tokenizer, model) -> None:
    vid = args.video or DEFAULT_VIDEO
    if not vid.is_file():
        sys.exit(f"missing video file: {vid}")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root.resolve() / f"{vid.stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"results dir: {run_dir}", flush=True)
    pixel_values, num_patches_list = load_video(
        vid,
        input_size=args.input_size,
        max_num=args.max_num,
        num_segments=args.num_segments,
    )
    pv_dtype = torch.bfloat16 if args.bf16 else torch.float16
    if torch.cuda.is_available():
        pixel_values = pixel_values.to(dtype=pv_dtype, device="cuda")

    generation_config = {
        k: v
        for k, v in dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
        ).items()
        if v is not None
    }

    video_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
    tail_q = args.question or "Briefly describe what happens in this video."
    question = video_prefix + tail_q

    response, _ = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True,
    )
    write_plain(run_dir / "meta.json", json.dumps({"video": str(vid), "mode": "mp4"}, indent=2))
    write_plain(run_dir / "question.txt", question)
    write_plain(run_dir / "response.txt", response)
    write_plain(
        run_dir / "foundation_token_io.txt",
        build_foundation_token_io_document(
            tokenizer,
            question=question,
            num_patches_list=num_patches_list,
            vision_slots_per_patch=args.vision_slots_per_patch,
            response1=response,
        ),
    )
    print(response, flush=True)


def run_vstream_streaming(args, tokenizer, model) -> None:
    root = args.vstream_root.resolve()
    qa_dir = root / VSTREAM_REALTIME
    if args.vstream_source == "movienet":
        qa_path = qa_dir / "test_qa_movienet.json"
        frames_parent = resolve_movienet_frames_root(root)
    else:
        qa_path = qa_dir / "test_qa_ego4d.json"
        frames_parent = resolve_ego_frames_root(root, args.ego_frames_root)

    if not qa_path.is_file():
        sys.exit(
            f"Missing {qa_path}. Download dataset: python {Path(__file__).name} --download-vstream-only"
        )
    if not frames_parent.is_dir():
        sys.exit(
            f"Missing frames directory {frames_parent}.\n"
            "MovieNet: unzip vstream-realtime/movienet_frames_online.zip here.\n"
            "Ego4d: `cd vstream-realtime && 7z x ego4d_frames_online.partaa ego4d_frames/000000/` "
            "(one clip) or extract full archive; clips are ego4d_frames/<video_id>/."
        )

    qa_rows: list[dict] = json.loads(qa_path.read_text(encoding="utf-8"))
    start, limit = args.sample_start, args.sample_limit
    slice_rows = qa_rows[start : start + limit if limit is not None else None]

    pv_dtype = torch.bfloat16 if args.bf16 else torch.float16
    generation_config = {
        k: v
        for k, v in dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
        ).items()
        if v is not None
    }

    for row in slice_rows:
        qid = row.get("id", row.get("video_id", "unknown"))
        video_name = row["video_name"]
        question_text = row["question"]

        if args.vstream_source == "movienet":
            clip_dir = frames_parent / video_name
            frame_paths = list_movienet_frames_in_qa_interval(
                clip_dir, str(row["start_time"]), str(row["end_time"])
            )
        else:
            # HF archive paths: ego4d_frames/<video_id>/ e.g. 000000 (not video_name UUID)
            clip_key = row.get("video_id") or video_name
            clip_dir = frames_parent / str(clip_key)
            frame_paths = list_ego_stream_frames(clip_dir, row.get("duration"))

        if not frame_paths:
            print(f"[skip] no frames clip_dir={clip_dir}", flush=True)
            continue

        pixel_values, num_patches_list = load_vstream_frames_pixel_values(
            frame_paths,
            num_segments=args.num_segments,
            input_size=args.input_size,
            max_num=args.max_num,
        )
        if torch.cuda.is_available():
            pixel_values = pixel_values.to(dtype=pv_dtype, device="cuda")

        video_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
        question = video_prefix + question_text

        response, _ = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = args.results_root.resolve() / f"vstream_{args.vstream_source}_{qid}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "benchmark": "VStream-QA-RVS",
            "source": args.vstream_source,
            "qa_id": qid,
            "video_name": video_name,
            "video_id": row.get("video_id"),
            "frame_dir": str(clip_dir),
            "n_frames_available": len(frame_paths),
            "num_segments_sampled": args.num_segments,
            "answer_gt": row.get("answer"),
            "answer_type": row.get("answer_type"),
            "row": row,
        }
        write_plain(run_dir / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
        write_plain(run_dir / "question.txt", question)
        write_plain(run_dir / "response.txt", response)
        write_plain(
            run_dir / "foundation_token_io.txt",
            build_foundation_token_io_document(
                tokenizer,
                question=question,
                num_patches_list=num_patches_list,
                vision_slots_per_patch=args.vision_slots_per_patch,
                response1=response,
            ),
        )
        print(f"\n=== {qid} (video_id={row.get('video_id')}) ===\nQ: {question_text}\nA: {response}\n", flush=True)


def main():
    args = parse_args()
    _require_transformers_v4()

    if args.download_vstream_only:
        dest = download_vstream_qa(args.vstream_root)
        print(f"IVGSZ/VStream-QA snapshot -> {dest}", flush=True)
        return

    if not torch.cuda.is_available():
        print("CUDA not available — CPU may OOM.", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    print("loading model...", flush=True)
    model = load_model(args)

    if args.video is not None or args.vstream_source == "none":
        run_single_video(args, tokenizer, model)
        return

    run_vstream_streaming(args, tokenizer, model)


if __name__ == "__main__":
    main()
