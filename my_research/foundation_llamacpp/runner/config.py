from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MediaMode(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO_FILE = "video_file"
    STREAMING = "streaming"


class BackendMode(str, Enum):
    CPU = "cpu"
    OPENCL = "opencl"
    HYBRID_QNN_OPENCL = "hybrid_qnn_opencl"


@dataclass
class PreparedMedia:
    frame_bins: list[Path]
    layout_images: list[Path]
    prompt: str
    metadata_path: Path
    num_patches_list: list[int]
    frame_indices: list[int]
    source_kind: str


def media_mode_from_args(args) -> MediaMode:
    if getattr(args, "video", None) is not None:
        return MediaMode.VIDEO_FILE
    if getattr(args, "image", None) is not None:
        return MediaMode.IMAGE
    return MediaMode.TEXT


def backend_mode_from_processor(processor: str) -> BackendMode:
    if processor == "cpu":
        return BackendMode.CPU
    if processor == "gpu":
        return BackendMode.OPENCL
    if processor == "hybrid":
        return BackendMode.HYBRID_QNN_OPENCL
    raise ValueError(f"unsupported processor: {processor}")

