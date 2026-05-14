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
STREAM_MODE_SINGLE_BUFFER = "single_buffer"
STREAM_MODE_SLIDING_WINDOW = "sliding_window"
STREAM_MODE_VISION_PREFILL = "vision_prefill"
STREAM_MODES = {
    STREAM_MODE_SINGLE_BUFFER,
    STREAM_MODE_SLIDING_WINDOW,
    STREAM_MODE_VISION_PREFILL,
}


def normalize_stream_mode(stream_mode: str | None, *, single_buffer: bool = False) -> str:
    if single_buffer:
        return STREAM_MODE_SINGLE_BUFFER
    if stream_mode is None:
        return STREAM_MODE_SINGLE_BUFFER
    normalized = stream_mode.strip().lower().replace("-", "_")
    if normalized not in STREAM_MODES:
        choices = ", ".join(sorted(mode.replace("_", "-") for mode in STREAM_MODES))
        raise ValueError(f"unsupported stream mode: {stream_mode!r}; expected one of: {choices}")
    return normalized


def _evenly_limit_items(items: list, limit: int) -> list:
    if limit <= 0:
        raise ValueError("window_max_frames must be positive")
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[-1]]
    last = len(items) - 1
    indices = [round(i * last / (limit - 1)) for i in range(limit)]
    return [items[i] for i in indices]


def select_recent_window_frames(
    frames: list[dict[str, object]],
    *,
    prompt_time_s: float,
    window_sec: float | None,
    window_max_frames: int,
) -> list[dict[str, object]]:
    if window_max_frames <= 0:
        raise ValueError("window_max_frames must be positive")
    start_s = float("-inf") if window_sec is None else prompt_time_s - window_sec
    selected = [
        frame
        for frame in frames
        if start_s <= float(frame.get("timestamp_s", 0.0)) <= prompt_time_s
    ]
    if not selected:
        selected = [
            frame
            for frame in frames
            if float(frame.get("timestamp_s", 0.0)) <= prompt_time_s
        ][-1:]
    return _evenly_limit_items(selected, window_max_frames)


def build_streaming_video_prompt(frames: list[dict[str, object]], raw_prompt: str) -> str:
    prefix_parts: list[str] = []
    for frame_i, frame in enumerate(frames):
        n_patches = int(frame.get("num_patches", 1) or 1)
        prefix_parts.append(f"Frame{frame_i + 1}: " + (MEDIA_MARKER * n_patches) + "\n")
    return "".join(prefix_parts) + raw_prompt


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
        prefix_parts.append(f"Frame{frame_i + 1}: " + (MEDIA_MARKER * n_patches) + "\n")
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


def prepare_streaming_video_media(
    video: Path,
    work_dir: Path,
    *,
    sampling_fps: float,
    prompt_events: list[dict[str, object]],
    max_num: int,
    max_video_time: float | None = None,
    stream_mode: str | None = None,
    window_sec: float | None = None,
    window_max_frames: int = 8,
    single_buffer: bool = False,
) -> PreparedMedia:
    from PIL import Image

    decord = importlib.import_module("decord")
    vr = decord.VideoReader(str(video), ctx=decord.cpu(0), num_threads=1)
    source_fps = float(vr.get_avg_fps())
    if source_fps <= 0:
        raise SystemExit(f"Could not read FPS from streaming video: {video}")
    frame_count = len(vr)
    duration_s = frame_count / source_fps if frame_count else 0.0
    if frame_count <= 0:
        raise SystemExit(f"Streaming video has no frames: {video}")
    effective_duration_s = min(duration_s, max_video_time) if max_video_time is not None else duration_s
    normalized_stream_mode = normalize_stream_mode(stream_mode, single_buffer=single_buffer)
    use_single_buffer_frames = normalized_stream_mode == STREAM_MODE_SINGLE_BUFFER

    sample_count = int(np.floor(effective_duration_s * sampling_fps)) + 1
    timestamps = [idx / sampling_fps for idx in range(sample_count)]
    frame_indices = sorted({
        min(int(round(ts * source_fps)), frame_count - 1)
        for ts in timestamps
        if ts <= effective_duration_s
    })

    frame_bins: list[Path] = []
    layout_images: list[Path] = []
    num_patches_list: list[int] = []
    frame_records: list[dict[str, object]] = []
    for stream_i, frame_index in enumerate(frame_indices):
        timestamp_s = frame_index / source_fps
        image = Image.fromarray(vr[int(frame_index)].asnumpy()).convert("RGB")
        if use_single_buffer_frames:
            frame_bin = work_dir / f"stream_frame_{stream_i:04d}.bin"
            layout_image = work_dir / f"stream_frame_{stream_i:04d}.png"
            normalize_image_to_bin(image, frame_bin)
            image.save(layout_image)
            frame_bins.append(frame_bin)
            layout_images.append(layout_image)
            num_patches_list.append(1)
            frame_records.append(
                {
                    "stream_frame": stream_i,
                    "timestamp_s": round(timestamp_s, 6),
                    "video_frame_index": int(frame_index),
                    "num_patches": 1,
                    "tiles": [{"bin": frame_bin.name, "layout_image": layout_image.name}],
                }
            )
            continue
        tiles = dynamic_preprocess_tiles(image, image_size=448, max_num=max_num, use_thumbnail=True)
        num_patches_list.append(len(tiles))
        tile_records: list[dict[str, object]] = []
        for tile_i, tile in enumerate(tiles):
            stem = f"stream_frame_{stream_i:04d}_tile_{tile_i:04d}"
            frame_bin = work_dir / f"{stem}.bin"
            layout_image = work_dir / f"{stem}.png"
            normalize_image_to_bin(tile, frame_bin)
            tile.save(layout_image)
            frame_bins.append(frame_bin)
            layout_images.append(layout_image)
            tile_records.append({"bin": frame_bin.name, "layout_image": layout_image.name})
        frame_records.append(
            {
                "stream_frame": stream_i,
                "timestamp_s": round(timestamp_s, 6),
                "video_frame_index": int(frame_index),
                "num_patches": len(tiles),
                "tiles": tile_records,
            }
        )

    metadata = {
        "schema_version": MEDIA_MANIFEST_VERSION,
        "source_kind": "streaming_video",
        "source": str(video),
        "source_fps": source_fps,
        "sampling_fps": sampling_fps,
        "duration_s": duration_s,
        "effective_duration_s": effective_duration_s,
        "max_video_time": max_video_time,
        "frame_count": frame_count,
        "max_num": max_num,
        "stream_mode": normalized_stream_mode,
        "window_sec": window_sec,
        "window_max_frames": window_max_frames,
        "frame_indices": [int(i) for i in frame_indices],
        "num_patches_list": num_patches_list,
        "frame_bins": [p.name for p in frame_bins],
        "layout_images": [p.name for p in layout_images],
        "frames": frame_records,
        "prompt_events": prompt_events,
        "prompt": "",
        "raw_prompt": [event["prompt"] for event in prompt_events],
    }
    metadata_path = work_dir / "media_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return PreparedMedia(
        frame_bins=frame_bins,
        layout_images=layout_images,
        prompt="",
        metadata_path=metadata_path,
        num_patches_list=num_patches_list,
        frame_indices=[int(i) for i in frame_indices],
        source_kind="streaming_video",
    )


def prepare_media(args, work_dir: Path) -> PreparedMedia:
    if getattr(args, "streaming_video", None) is not None:
        return prepare_streaming_video_media(
            args.streaming_video,
            work_dir,
            sampling_fps=args.sampling_fps,
            prompt_events=args.prompt_events,
            max_num=args.max_num,
            max_video_time=getattr(args, "max_video_time", None),
            stream_mode=getattr(args, "stream_mode", None),
            window_sec=getattr(args, "window_sec", None),
            window_max_frames=getattr(args, "window_max_frames", 8),
            single_buffer=getattr(args, "single_buffer", False),
        )
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
