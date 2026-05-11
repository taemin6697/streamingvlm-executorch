from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from .config import PreparedMedia


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEDIA_MARKER = "<__media__>"
MEDIA_MANIFEST_VERSION = 2


def normalize_image_to_bin(image, output_path: Path, image_size: int = 448) -> None:
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = image.convert("RGB").resize((image_size, image_size), resampling)
    arr = np.asarray(image).astype("float32") / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype("float32").tofile(output_path)


def prepare_warmup_image(image: Path, work_dir: Path) -> tuple[Path, Path]:
    load_image = importlib.import_module("transformers.image_utils").load_image

    warmup_bin = work_dir / "warmup_golden_gate_448.bin"
    warmup_layout = work_dir / image.name
    normalize_image_to_bin(load_image(str(image)), warmup_bin)
    if image.resolve() != warmup_layout.resolve():
        warmup_layout.write_bytes(image.read_bytes())
    return warmup_bin, warmup_layout


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
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


def dynamic_preprocess_tiles(image, *, image_size: int = 448, max_num: int = 1, use_thumbnail: bool = True) -> list:
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(1, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if 1 <= i * j <= max_num
    }
    sorted_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_ratio = find_closest_aspect_ratio(aspect_ratio, sorted_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized_img = image.resize((target_width, target_height), resampling)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), resampling))
    return processed_images


def prepare_image_media(image: Path, work_dir: Path, prompt: str) -> PreparedMedia:
    load_image = importlib.import_module("transformers.image_utils").load_image

    frame_bin = work_dir / "frame_0000.bin"
    layout_image = work_dir / image.name
    normalize_image_to_bin(load_image(str(image)), frame_bin)
    if image.resolve() != layout_image.resolve():
        layout_image.write_bytes(image.read_bytes())
    metadata = {
        "schema_version": MEDIA_MANIFEST_VERSION,
        "source_kind": "image",
        "source": str(image),
        "num_patches_list": [1],
        "frame_indices": [0],
        "frame_bins": [frame_bin.name],
        "layout_images": [layout_image.name],
        "prompt": prompt,
    }
    metadata_path = work_dir / "media_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return PreparedMedia(
        frame_bins=[frame_bin],
        layout_images=[layout_image],
        prompt=prompt,
        metadata_path=metadata_path,
        num_patches_list=[1],
        frame_indices=[0],
        source_kind="image",
    )


def prepare_video_media(video: Path, work_dir: Path, prompt: str, *, num_segments: int, max_num: int) -> PreparedMedia:
    from PIL import Image

    decord = importlib.import_module("decord")
    vr = decord.VideoReader(str(video), ctx=decord.cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    seg_size = float(max_frame) / num_segments
    frame_indices = [int((seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]

    frame_bins: list[Path] = []
    layout_images: list[Path] = []
    num_patches_list: list[int] = []
    frame_records: list[dict[str, object]] = []
    for frame_i, frame_index in enumerate(frame_indices):
        image = Image.fromarray(vr[int(frame_index)].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess_tiles(image, image_size=448, max_num=max_num, use_thumbnail=True)
        num_patches_list.append(len(tiles))
        tile_records: list[dict[str, object]] = []
        for tile_i, tile in enumerate(tiles):
            stem = f"frame_{frame_i:04d}_tile_{tile_i:04d}"
            frame_bin = work_dir / f"{stem}.bin"
            layout_image = work_dir / f"{stem}.png"
            normalize_image_to_bin(tile, frame_bin)
            tile.save(layout_image)
            frame_bins.append(frame_bin)
            layout_images.append(layout_image)
            tile_records.append({"bin": frame_bin.name, "layout_image": layout_image.name})
        frame_records.append(
            {
                "frame": frame_i + 1,
                "video_frame_index": int(frame_index),
                "num_patches": len(tiles),
                "tiles": tile_records,
            }
        )

    prefix_parts: list[str] = []
    for frame_i, n_patches in enumerate(num_patches_list):
        prefix_parts.append(f"Frame {frame_i + 1}: " + (MEDIA_MARKER * n_patches) + "\n")
    media_prompt = "".join(prefix_parts) + prompt

    metadata = {
        "schema_version": MEDIA_MANIFEST_VERSION,
        "source_kind": "video",
        "source": str(video),
        "fps": fps,
        "max_frame": max_frame,
        "num_segments": num_segments,
        "max_num": max_num,
        "frame_indices": [int(i) for i in frame_indices],
        "num_patches_list": num_patches_list,
        "frame_bins": [p.name for p in frame_bins],
        "layout_images": [p.name for p in layout_images],
        "frames": frame_records,
        "prompt": media_prompt,
        "raw_prompt": prompt,
    }
    metadata_path = work_dir / "media_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return PreparedMedia(
        frame_bins=frame_bins,
        layout_images=layout_images,
        prompt=media_prompt,
        metadata_path=metadata_path,
        num_patches_list=num_patches_list,
        frame_indices=[int(i) for i in frame_indices],
        source_kind="video",
    )


def prepare_media(args, work_dir: Path) -> PreparedMedia:
    if args.video is not None:
        return prepare_video_media(
            args.video,
            work_dir,
            args.prompt,
            num_segments=args.num_segments,
            max_num=args.max_num,
        )
    if args.image is None:
        raise SystemExit("media preparation requires --image or --video")
    return prepare_image_media(args.image, work_dir, args.prompt)

