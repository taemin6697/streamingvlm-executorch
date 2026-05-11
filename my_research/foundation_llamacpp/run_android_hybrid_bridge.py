#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from my_research.foundation.host.android_timeline_memory_summary import (
    write_memory_usage_summary_from_rows,
)

WORKSPACE = Path(__file__).resolve().parents[2]
FOUNDATION_LLAMA = Path(__file__).resolve().parent
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEDIA_MARKER = "<__media__>"


@dataclass
class PreparedMedia:
    frame_bins: list[Path]
    layout_images: list[Path]
    prompt: str
    metadata_path: Path
    num_patches_list: list[int]
    frame_indices: list[int]
    source_kind: str


def _standalone_completion_mode(args: argparse.Namespace) -> bool:
    """--processor cpu|gpu and no media → text-only llama-completion (no mmproj/vision)."""
    return args.processor in ("cpu", "gpu") and getattr(args, "image", None) is None and getattr(args, "video", None) is None


def _run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture_output)


def _adb(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _normalize_image_to_bin(image, output_path: Path, image_size: int = 448) -> None:
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


def _find_closest_aspect_ratio(
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


def _dynamic_preprocess_tiles(image, *, image_size: int = 448, max_num: int = 1, use_thumbnail: bool = True) -> list:
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
    target_ratio = _find_closest_aspect_ratio(aspect_ratio, sorted_ratios, orig_width, orig_height, image_size)
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


def _prepare_image_media(image: Path, work_dir: Path, prompt: str) -> PreparedMedia:
    load_image = importlib.import_module("transformers.image_utils").load_image

    frame_bin = work_dir / "frame_0000.bin"
    layout_image = work_dir / image.name
    _normalize_image_to_bin(load_image(str(image)), frame_bin)
    if image.resolve() != layout_image.resolve():
        layout_image.write_bytes(image.read_bytes())
    metadata = {
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


def _prepare_video_media(video: Path, work_dir: Path, prompt: str, *, num_segments: int, max_num: int) -> PreparedMedia:
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
        tiles = _dynamic_preprocess_tiles(image, image_size=448, max_num=max_num, use_thumbnail=True)
        num_patches_list.append(len(tiles))
        tile_records: list[dict[str, object]] = []
        for tile_i, tile in enumerate(tiles):
            stem = f"frame_{frame_i:04d}_tile_{tile_i:04d}"
            frame_bin = work_dir / f"{stem}.bin"
            layout_image = work_dir / f"{stem}.png"
            _normalize_image_to_bin(tile, frame_bin)
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


def _prepare_media(args: argparse.Namespace, work_dir: Path) -> PreparedMedia:
    if args.video is not None:
        return _prepare_video_media(
            args.video,
            work_dir,
            args.prompt,
            num_segments=args.num_segments,
            max_num=args.max_num,
        )
    if args.image is None:
        raise SystemExit("media preparation requires --image or --video")
    return _prepare_image_media(args.image, work_dir, args.prompt)


def _load_manifest(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    resolved = dict(data)
    paths = {}
    for key, value in data.get("paths", {}).items():
        if not value:
            continue
        path = Path(value)
        paths[key] = str(path if path.is_absolute() else (base / path).resolve())
    resolved["paths"] = paths
    return resolved


def _push(adb: list[str], local: Path, remote_dir: str) -> None:
    _run(adb + ["push", str(local), f"{remote_dir}/{local.name}"])


def _remote_exists(adb: list[str], remote_path: str) -> bool:
    result = subprocess.run(
        adb + ["shell", f"test -f {shlex.quote(remote_path)}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _remote_read_text(adb: list[str], remote_path: str) -> str | None:
    result = subprocess.run(
        adb + ["shell", f"cat {shlex.quote(remote_path)}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _push_model_cached(adb: list[str], local: Path, remote_dir: str, *, force: bool) -> None:
    remote = f"{remote_dir}/{local.name}"
    if not force and _remote_exists(adb, remote):
        print(f"[push-cache] keep remote model: {remote}")
        return
    _push(adb, local, remote_dir)


def _push_qnn_libs(adb: list[str], remote_dir: str, qnn_sdk: Path, executorch_build_dir: Path, soc_model: str) -> None:
    qnn_lib_dir = qnn_sdk / "lib" / "aarch64-android"
    if qnn_lib_dir.exists():
        for lib in sorted(qnn_lib_dir.glob("libQnn*.so")):
            _push(adb, lib, remote_dir)
    arch = {
        "SM8750": "79",
        "SM8650": "75",
        "SM8550": "73",
        "SM8450": "69",
        "SM8350": "68",
    }.get(soc_model, "73")
    skel = qnn_sdk / "lib" / f"hexagon-v{arch}" / "unsigned" / f"libQnnHtpV{arch}Skel.so"
    if skel.exists():
        _push(adb, skel, remote_dir)
    backend = executorch_build_dir / "backends" / "qualcomm" / "libqnn_executorch_backend.so"
    if backend.exists():
        _push(adb, backend, remote_dir)


def _push_llama_runtime(
    adb: list[str],
    remote_dir: str,
    llama_build_dir: Path,
    opencl_lib: Path | None,
    push_opencl_loader: bool,
) -> None:
    for subdir in (llama_build_dir / "bin", llama_build_dir / "lib"):
        if subdir.exists():
            for path in sorted(subdir.iterdir()):
                if path.name == "libOpenCL.so" and not push_opencl_loader:
                    continue
                if path.is_file() and (
                    path.suffix == ".so"
                    or path.name in {"hybrid_decode", "llama-mtmd-cli", "opencl_phase_mtmd", "llama-completion"}
                ):
                    _push(adb, path, remote_dir)
    for name in ("hybrid_decode", "opencl_phase_mtmd", "llama-completion"):
        path = llama_build_dir / name
        if path.exists():
            _push(adb, path, remote_dir)
    runtime_patterns = ["libc++_shared.so"]
    if push_opencl_loader:
        runtime_patterns.append("libOpenCL.so")
    for pattern in runtime_patterns:
        for path in sorted(llama_build_dir.rglob(pattern)):
            if path.is_file():
                _push(adb, path, remote_dir)
    if push_opencl_loader and opencl_lib and opencl_lib.exists():
        _push(adb, opencl_lib, remote_dir)


def _pull_if_exists(adb: list[str], remote: str, local: Path) -> None:
    result = subprocess.run(adb + ["shell", f"test -f {shlex.quote(remote)}"], check=False)
    if result.returncode == 0:
        local.parent.mkdir(parents=True, exist_ok=True)
        _run(adb + ["pull", remote, str(local)])


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_memory_usage_txt(output_dir: Path, rows: list[dict[str, str]]) -> None:
    write_memory_usage_summary_from_rows(output_dir, rows)


def _match_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else 0.0


def _match_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


def _parse_log_summary(log_text: str) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key, pattern in [
        ("image_slice_encoded_ms", r"image slice encoded in ([0-9.]+) ms"),
        ("image_decoded_ms", r"image decoded(?: \(batch \d+/\d+\))? in ([0-9.]+) ms"),
        ("model_buffer_mib", r"model buffer size =\s+([0-9.]+) MiB"),
        ("context_buffer_mib", r"KV buffer size =\s+([0-9.]+) MiB"),
        ("compute_buffer_mib", r"compute buffer size =\s+([0-9.]+) MiB"),
        ("load_time_ms", r"llama_perf_context_print:\s+load time =\s+([0-9.]+) ms"),
        ("prompt_eval_time_ms", r"llama_perf_context_print:\s+prompt eval time =\s+([0-9.]+) ms"),
        ("decode_eval_time_ms", r"llama_perf_context_print:\s+eval time =\s+([0-9.]+) ms"),
        ("total_time_ms", r"llama_perf_context_print:\s+total time =\s+([0-9.]+) ms"),
    ]:
        matches = re.findall(pattern, log_text)
        if matches:
            summary[key] = float(matches[-1])
    prompt_tokens = _match_int(log_text, r"prompt eval time =\s+[0-9.]+ ms /\s+([0-9]+) tokens")
    decode_runs = _match_int(log_text, r"eval time =\s+[0-9.]+ ms /\s+([0-9]+) runs")
    if prompt_tokens:
        summary["prompt_eval_tokens"] = prompt_tokens
    if decode_runs:
        summary["decode_eval_runs"] = decode_runs
    if summary.get("prompt_eval_time_ms") and prompt_tokens:
        summary["prompt_eval_tok_s"] = prompt_tokens / float(summary["prompt_eval_time_ms"]) * 1000.0
    if summary.get("decode_eval_time_ms") and decode_runs:
        summary["decode_eval_tok_s"] = decode_runs / float(summary["decode_eval_time_ms"]) * 1000.0
    return summary


def _ansi_sub(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _extract_generated_text_text_only(log_text: str) -> str:
    """Best-effort completion output (llama-completion) before llama_perf_context_print."""
    body = _ansi_sub(log_text)
    stop = body.find("llama_perf_context_print:")
    if stop >= 0:
        body = body[:stop]
    body = re.sub(r"(?mi)\s*\[end of text\]\s*\Z", "", body)
    lines_out: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(("ggml_", "llama_", "clip_", "load_backend", "broadcast_", "check_tensor", "alloc_tensor")):
            continue
        if re.match(r"^(main|eval|graph|prompt|encode|decode|sampler|tensor|model|kv|opencl|cpu|gpu|device)\s*:", s, re.I):
            continue
        if s.startswith(("[", "warn", "info", "deprecat", "fatal", "error")):
            continue
        lines_out.append(line.rstrip())
    return "\n".join(lines_out).strip()


def _extract_generated_text_from_log(log_text: str) -> str:
    lines = log_text.splitlines()
    start_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith("image decoded"):
            start_idx = idx + 1
    if start_idx < 0:
        return ""
    out_lines: list[str] = []
    for line in lines[start_idx:]:
        if line.startswith("llama_perf_context_print:"):
            break
        if not line.strip() and not out_lines:
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def _write_fallback_token_io_txt(
    result_dir: Path,
    prompt: str,
    log_text: str,
    *,
    text_only: bool = False,
    image_tokens: int = 256,
) -> None:
    del image_tokens
    token_io = result_dir / "foundation_token_io.txt"
    if token_io.exists():
        return
    if text_only:
        generated = _extract_generated_text_text_only(log_text)
        text = f"<|im_start|>user\n{prompt}\n<|im_start|>assistant\n{generated}\n"
    else:
        generated = _extract_generated_text_from_log(log_text)
        text = f"User: {prompt}\nAssistant: {generated}\n"
    token_io.write_text(text, encoding="utf-8")


def _phase_float(row: dict[str, str], key: str) -> float:
    value = (row.get(key) or "").strip()
    return float(value) if value else 0.0


def _read_phase_rows(path: Path, *, offset_s: float = 0.0) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        header: list[str] | None = None
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("row_type,"):
                header = raw.split(",")
                continue
            if header is None:
                continue
            parts = raw.split(",")
            if len(parts) < len(header):
                parts.extend([""] * (len(header) - len(parts)))
            row = {key: parts[idx] for idx, key in enumerate(header)}
            if row.get("row_type") == "L_DecoderInit":
                row["row_type"] = "L_DecoderRuntimeInit"
            row["elapsed_s_start"] = f"{_phase_float(row, 'elapsed_s_start') + offset_s:.6f}"
            row["elapsed_s_end"] = f"{_phase_float(row, 'elapsed_s_end') + offset_s:.6f}"
            rows.append(row)
    return rows


def _write_phase_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "row_type",
        "elapsed_s_start",
        "elapsed_s_end",
        "rss_kb_start",
        "rss_kb_end",
        "col_a_ms",
        "col_b_ms",
        "total_ms",
        "kv_pos",
        "kv_total",
        "kv_used_pct",
        "kv_estimated_used_kb",
        "kv_total_kb",
        "kv_physical_committed_kb",
        "token_idx",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "row_type": "# L_VisionLoad: QNN vision module load  ImageLoad: input tensor load  "
            "V_Encode: QNN projected vision embedding  EmbeddingFileWrite: .svlmemb write  "
            "L_DecoderRuntimeInit: llama.cpp args/OpenCL runtime init  ExternalEmbeddingRead: .svlmemb read  "
            "L_DecoderLoad: llama.cpp model/mmproj load  "
            "LayoutTokenize: mtmd layout  Prefill: combined text/image prefill  "
            "ImagePrefill: image embedding prefill  T_Prefill: text prompt prefill  D: one generated-token decode"
        })
        writer.writerows(rows)


def _phase_rows_from_artifacts(result_dir: Path) -> list[dict[str, str]]:
    vision_rows = _read_phase_rows(result_dir / "vision_phase_stats.csv")
    # In coordinated-load mode both processes are launched by the same remote
    # script and use process-local clocks that are close enough for timeline
    # plotting. Do not offset decoder phases after vision phases: that would
    # incorrectly make loading appear after QNN encode.
    decoder_rows = _read_phase_rows(result_dir / "decoder_phase_stats.csv")
    return sorted(vision_rows + decoder_rows, key=lambda row: (_phase_float(row, "elapsed_s_start"), _phase_float(row, "elapsed_s_end")))


def _make_phase_row(name: str, start_s: float, duration_ms: float) -> dict[str, str]:
    end_s = start_s + max(duration_ms, 0.0) / 1000.0
    return {
        "row_type": name,
        "elapsed_s_start": f"{start_s:.6f}",
        "elapsed_s_end": f"{end_s:.6f}",
        "rss_kb_start": "",
        "rss_kb_end": "",
        "col_a_ms": f"{duration_ms:.3f}",
        "col_b_ms": "",
        "total_ms": f"{duration_ms:.3f}",
        "kv_pos": "",
        "kv_total": "",
        "kv_used_pct": "",
        "kv_estimated_used_kb": "",
        "kv_total_kb": "",
        "kv_physical_committed_kb": "",
        "token_idx": "0",
    }


def _synthetic_standalone_phase_rows(summary: dict[str, object]) -> list[dict[str, str]]:
    """Build comparable phase rows when upstream llama-mtmd-cli lacks timers."""
    image_encode_ms = float(summary.get("image_slice_encoded_ms", 0.0) or 0.0)
    image_prefill_ms = float(summary.get("image_decoded_ms", 0.0) or 0.0)
    prompt_eval_ms = float(summary.get("prompt_eval_time_ms", 0.0) or 0.0)
    text_prefill_ms = max(prompt_eval_ms - image_prefill_ms, 0.0)
    decode_ms = float(summary.get("decode_eval_time_ms", 0.0) or 0.0)
    phases = [
        ("V_Encode", image_encode_ms),
        ("T_Prefill", text_prefill_ms),
        ("ImagePrefill", image_prefill_ms),
        ("Decode", decode_ms),
    ]
    rows: list[dict[str, str]] = []
    cursor_s = 0.0
    for name, duration_ms in phases:
        if duration_ms <= 0:
            continue
        rows.append(_make_phase_row(name, cursor_s, duration_ms))
        cursor_s += duration_ms / 1000.0
    return rows


def _phase_colors() -> dict[str, str]:
    return {
        "L_VisionLoad": "#8e44ad",
        "ImageLoad": "#74b9ff",
        "V_Encode": "#00b894",
        "EmbeddingFileWrite": "#55efc4",
        "ExternalEmbeddingRead": "#00cec9",
        "L_DecoderRuntimeInit": "#a29bfe",
        "L_DecoderLoad": "#6c5ce7",
        "LayoutTokenize": "#fdcb6e",
        "Prefill": "#2d98da",
        "ImagePrefill": "#0984e3",
        "T_Prefill": "#e17055",
        "D": "#ff7675",
        "Decode": "#d63031",
    }


def _write_png_memory_timeline(
    output_dir: Path,
    rows: list[dict[str, str]],
    phase_rows: list[dict[str, str]] | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    usable = [r for r in rows if r.get("elapsed_s") and r.get("mem_available_kb")]
    if not usable:
        return
    xs = [float(r["elapsed_s"]) for r in usable]
    ys = [float(r["mem_available_kb"]) / 1024.0 for r in usable]
    fig, ax = plt.subplots(figsize=(19, 8), dpi=160)
    ax.plot(
        xs,
        ys,
        label="MemAvailable (MiB)",
        linewidth=2.4,
        marker="o",
        markersize=2.6,
        color="#0984e3",
    )
    colors = _phase_colors()
    phase_count: dict[str, int] = {}
    for phase in phase_rows or []:
        name = phase.get("row_type", "")
        if name == "D":
            continue
        start = _phase_float(phase, "elapsed_s_start")
        end = _phase_float(phase, "elapsed_s_end")
        if end <= start:
            continue
        color = colors.get(name, "#636e72")
        ax.axvspan(start, end, color=color, alpha=0.06)
        ax.axvline(start, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        ax.axvline(end, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        phase_count[name] = phase_count.get(name, 0) + 1
        label = f"{name}{phase_count[name]}" if name in {"T_Prefill", "V_Encode"} else name
        ax.text(
            (start + end) / 2.0,
            0.045,
            label,
            fontsize=8,
            ha="center",
            va="bottom",
            color=color,
            transform=ax.get_xaxis_transform(),
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.75,
            },
        )
    ax.set_title(f"Android Memory Timeline: {output_dir.name}")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.9,
    )
    ax.set_xlim(left=min(xs), right=max(xs))
    fig.tight_layout()
    fig.savefig(output_dir / "memory_timeline_plot.png", bbox_inches="tight")
    plt.close(fig)


def _write_png_phase_duration(output_dir: Path, perf: dict[str, float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    phases = [
        ("Load", perf.get("load_time_ms", 0.0)),
        ("QNN Vision Encoder", perf.get("qnn_vision_encode_ms", 0.0)),
        ("External Image Decode", perf.get("external_image_decode_ms", 0.0)),
        ("Prompt Eval", perf.get("prompt_eval_time_ms", 0.0)),
        ("Token Decode", perf.get("decode_eval_time_ms", 0.0)),
    ]
    phases = [(name, value) for name, value in phases if value > 0]
    if not phases:
        return
    total = sum(value for _, value in phases)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
    bottom = 0.0
    colors = ["#6c5ce7", "#00b894", "#0984e3", "#e17055", "#d63031"]
    for idx, (name, value) in enumerate(phases):
        color = colors[idx % len(colors)]
        ax.bar(["total"], [value], bottom=bottom, color=color, edgecolor="white")
        if value / total >= 0.05:
            ax.text(
                0,
                bottom + value / 2.0,
                f"{name}\n{value / 1000.0:.2f}s",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        bottom += value
    ax.set_title(f"Runtime Breakdown: {output_dir.name}")
    ax.set_ylabel("Elapsed Time (ms)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[idx % len(colors)])
        for idx in range(len(phases))
    ]
    labels = [f"{name}: {value / 1000.0:.2f}s" for name, value in phases]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_duration_stacked_bar.png", bbox_inches="tight")
    plt.close(fig)


def _write_png_phase_duration_from_rows(output_dir: Path, phase_rows: list[dict[str, str]]) -> None:
    if not phase_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    excluded_from_plot = {
        "ImageLoad",
        "L_DecoderLoad",
        "L_DecoderRuntimeInit",
        "L_VisionLoad",
        "LayoutTokenize",
        "ImagePrefill",
        "T_Prefill",
    }
    durations: dict[str, float] = {}
    first_start_s: dict[str, float] = {}
    has_decode_summary = any(row.get("row_type") == "Decode" for row in phase_rows)
    for row in phase_rows:
        name = row.get("row_type", "")
        if name in excluded_from_plot:
            continue
        if name == "D" and has_decode_summary:
            continue
        normalized = "Decode" if name == "D" else name
        start_s = _phase_float(row, "elapsed_s_start")
        duration = max(_phase_float(row, "elapsed_s_end") - start_s, 0.0)
        if duration > 0:
            durations[normalized] = durations.get(normalized, 0.0) + duration
            first_start_s[normalized] = min(first_start_s.get(normalized, start_s), start_s)
    phases = sorted(
        durations.items(),
        key=lambda item: (first_start_s.get(item[0], float("inf")), item[0]),
    )
    if not phases:
        return
    total = sum(value for _, value in phases)
    colors = _phase_colors()
    fig, ax = plt.subplots(figsize=(6.2, 8.8), dpi=160)
    bottom = 0.0
    for name, value in phases:
        color = colors.get(name, "#636e72")
        ax.bar(["total"], [value], bottom=bottom, color=color, edgecolor="white")
        if value / total >= 0.035:
            ax.text(
                0,
                bottom + value / 2.0,
                f"{name}\n{value:.3f}s",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        bottom += value
    ax.set_title(f"Precise Runtime Breakdown: {output_dir.name}")
    ax.set_ylabel("Elapsed Time (s)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors.get(name, "#636e72")) for name, _ in phases]
    labels = [f"{name}: {value:.3f}s ({value / total * 100:.1f}%)" for name, value in phases]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_duration_stacked_bar.png", bbox_inches="tight")
    plt.close(fig)


def _finalize_hybrid_outputs(result_dir: Path) -> None:
    decode_log = result_dir / "hybrid_decode_stdout.txt"
    if not decode_log.exists():
        decode_log = result_dir / "hybrid_decode_opencl_stdout.txt"
    vision_stats_path = result_dir / "vision_output_stats.csv"
    if not vision_stats_path.exists():
        vision_stats_path = result_dir / "hybrid_vision_output_stats_opencl.csv"
    if not decode_log.exists():
        return
    log_text = decode_log.read_text(encoding="utf-8", errors="replace")
    (result_dir / "foundation_output.txt").write_text(log_text, encoding="utf-8")

    raw_vision_stats: dict[str, str] = {}
    if vision_stats_path.exists():
        for row in _read_csv_dicts(vision_stats_path):
            raw_vision_stats[row.get("metric", "")] = row.get("value", "")
    qnn_encode_ms = raw_vision_stats.get("encode_ms") or raw_vision_stats.get("qnn_vision_encode_ms") or 0
    perf: dict[str, float] = {
        "load_time_ms": _match_float(log_text, r"llama_perf_context_print:\s+load time =\s+([0-9.]+) ms"),
        "qnn_vision_encode_ms": float(qnn_encode_ms or 0),
        "external_image_decode_ms": _match_float(log_text, r"image decoded(?: \(batch \d+/\d+\))? in ([0-9.]+) ms"),
        "prompt_eval_time_ms": _match_float(log_text, r"llama_perf_context_print:\s+prompt eval time =\s+([0-9.]+) ms"),
        "decode_eval_time_ms": _match_float(log_text, r"llama_perf_context_print:\s+eval time =\s+([0-9.]+) ms"),
        "total_time_ms": _match_float(log_text, r"llama_perf_context_print:\s+total time =\s+([0-9.]+) ms"),
        "prompt_eval_tokens": _match_int(log_text, r"prompt eval time =\s+[0-9.]+ ms /\s+([0-9]+) tokens"),
        "decode_eval_runs": _match_int(log_text, r"eval time =\s+[0-9.]+ ms /\s+([0-9]+) runs"),
    }
    if perf["prompt_eval_time_ms"] and perf["prompt_eval_tokens"]:
        perf["prompt_eval_tok_s"] = perf["prompt_eval_tokens"] / perf["prompt_eval_time_ms"] * 1000.0
    if perf["decode_eval_time_ms"] and perf["decode_eval_runs"]:
        perf["decode_eval_tok_s"] = perf["decode_eval_runs"] / perf["decode_eval_time_ms"] * 1000.0

    memory_rows = []
    memory_path = result_dir / "android_memory_timeline.csv"
    if memory_path.exists():
        memory_rows = _read_csv_dicts(memory_path)
    wall_s = max([float(row.get("elapsed_s", "0") or 0) for row in memory_rows] or [0.0])
    return_code = ""
    if (result_dir / "foundation_exit_code.txt").exists():
        return_code = (result_dir / "foundation_exit_code.txt").read_text(encoding="utf-8").strip()

    summary_rows: list[dict[str, object]] = [
        {"metric": "backend", "value": "hybrid_qnn_vision_llamacpp_opencl", "unit": ""},
        {"metric": "model_name", "value": result_dir.name, "unit": ""},
        {"metric": "wall_time_s", "value": round(wall_s, 3), "unit": "s"},
        {"metric": "return_code", "value": return_code, "unit": ""},
        {"metric": "vision_output_dims", "value": raw_vision_stats.get("output_dims", ""), "unit": ""},
        {"metric": "vision_input_count", "value": raw_vision_stats.get("input_count", ""), "unit": "frames_or_tiles"},
    ]
    for key, value in perf.items():
        unit = "ms" if key.endswith("_ms") else "tok/s" if key.endswith("_tok_s") else "tokens" if "tokens" in key else "runs" if "runs" in key else ""
        summary_rows.append({"metric": key, "value": value, "unit": unit})
    _write_csv(result_dir / "foundation_summary.csv", summary_rows, ["metric", "value", "unit"])

    phase_rows = _phase_rows_from_artifacts(result_dir)
    if phase_rows:
        _write_phase_csv(result_dir / "foundation_proc.csv", phase_rows)
        plot_phase_rows = _read_phase_rows(result_dir / "foundation_proc.csv")
    else:
        _write_csv(result_dir / "foundation_proc.csv", summary_rows, ["metric", "value", "unit"])
        plot_phase_rows = []
    _write_csv(
        result_dir / "vision_output_stats.csv",
        [
            {"metric": "input_dims", "value": raw_vision_stats.get("input_dims", ""), "unit": ""},
            {"metric": "output_dims", "value": raw_vision_stats.get("output_dims", ""), "unit": ""},
            {"metric": "output_values", "value": raw_vision_stats.get("output_values", ""), "unit": "float32"},
            {"metric": "qnn_vision_encode_ms", "value": perf["qnn_vision_encode_ms"], "unit": "ms"},
            {"metric": "external_image_decode_ms", "value": perf["external_image_decode_ms"], "unit": "ms"},
            {"metric": "prompt_eval_time_ms", "value": perf["prompt_eval_time_ms"], "unit": "ms"},
            {"metric": "decode_eval_time_ms", "value": perf["decode_eval_time_ms"], "unit": "ms"},
            {"metric": "total_time_ms", "value": perf["total_time_ms"], "unit": "ms"},
        ],
        ["metric", "value", "unit"],
    )
    _write_memory_usage_txt(result_dir, memory_rows)
    _write_png_memory_timeline(result_dir, memory_rows, plot_phase_rows)
    if plot_phase_rows:
        _write_png_phase_duration_from_rows(result_dir, plot_phase_rows)
    else:
        _write_png_phase_duration(result_dir, perf)


def _finalize_standalone_outputs(
    result_dir: Path,
    *,
    processor: str,
    return_code: str,
    prompt: str = "Describe this image briefly.",
    text_only: bool = False,
) -> None:
    output_path = result_dir / "foundation_output.txt"
    if not output_path.exists():
        return
    log_text = output_path.read_text(encoding="utf-8", errors="replace")
    _write_fallback_token_io_txt(result_dir, prompt, log_text, text_only=text_only)
    summary = _parse_log_summary(log_text)
    memory_rows = []
    memory_path = result_dir / "android_memory_timeline.csv"
    if memory_path.exists():
        memory_rows = _read_csv_dicts(memory_path)
    wall_s = max([float(row.get("elapsed_s", "0") or 0) for row in memory_rows] or [0.0])

    summary_rows: list[dict[str, object]] = [
        {
            "metric": "backend",
            "value": (
                "llamacpp_cpu_text"
                if processor == "cpu" and text_only
                else "llamacpp_opencl_text"
                if processor == "gpu" and text_only
                else "llamacpp_cpu"
                if processor == "cpu"
                else "llamacpp_opencl"
            ),
            "unit": "",
        },
        {"metric": "model_name", "value": result_dir.name, "unit": ""},
        {"metric": "wall_time_s", "value": round(wall_s, 3), "unit": "s"},
        {"metric": "return_code", "value": return_code, "unit": ""},
    ]
    for key, value in summary.items():
        unit = (
            "ms" if key.endswith("_ms")
            else "MiB" if key.endswith("_mib")
            else "tok/s" if key.endswith("_tok_s")
            else "tokens" if "tokens" in key
            else "runs" if "runs" in key
            else ""
        )
        summary_rows.append({"metric": key, "value": value, "unit": unit})
    _write_csv(result_dir / "foundation_summary.csv", summary_rows, ["metric", "value", "unit"])

    phase_rows = _read_phase_rows(result_dir / "foundation_phase_stats.csv")
    phase_rows_are_synthetic = False
    if not phase_rows:
        phase_rows = _synthetic_standalone_phase_rows(summary)
        phase_rows_are_synthetic = bool(phase_rows)
    if phase_rows:
        _write_phase_csv(result_dir / "foundation_proc.csv", phase_rows)
        plot_phase_rows = _read_phase_rows(result_dir / "foundation_proc.csv")
    else:
        _write_csv(result_dir / "foundation_proc.csv", summary_rows, ["metric", "value", "unit"])
        plot_phase_rows = []
    # Synthetic CPU rows are summary-derived durations, not wall-clock aligned
    # timestamps. Do not overlay them on memory timelines.
    memory_phase_rows = [] if phase_rows_are_synthetic else plot_phase_rows
    _write_memory_usage_txt(result_dir, memory_rows)
    _write_png_memory_timeline(result_dir, memory_rows, memory_phase_rows)
    if plot_phase_rows:
        _write_png_phase_duration_from_rows(result_dir, plot_phase_rows)
    else:
        perf = {key: float(value) for key, value in summary.items() if key.endswith("_ms") and isinstance(value, (int, float))}
        _write_png_phase_duration(result_dir, perf)


def _result_kv_slug_part(cache_type: str | None) -> str:
    """Slugs for result dir names: q8_0 -> 8, q4_0 -> 4, f16 -> 16 → suffix _kv16."""
    t = (cache_type or "f16").strip().lower().replace("-", "_")
    if t in ("fp16", "f16"):
        return "16"
    m = re.fullmatch(r"q([0-9]+)_0", t)
    if m:
        return m.group(1)
    return re.sub(r"[^a-z0-9_]+", "_", t).strip("_") or "unknown"


def _result_kv_suffix(cache_type_k: str | None, cache_type_v: str | None) -> str:
    k = _result_kv_slug_part(cache_type_k)
    v = _result_kv_slug_part(cache_type_v)
    if k == v:
        return f"_kv{k}"
    return f"_kv{k}_{v}"


def _result_model_name(
    model: Path,
    processor: str,
    ctx_size: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    *,
    text_only: bool = False,
) -> str:
    suffix = "opencl" if processor == "gpu" else processor
    kv = _result_kv_suffix(cache_type_k, cache_type_v)
    mid = "_text" if text_only else ""
    return f"{model.stem}_{suffix}_ctx_{ctx_size}{mid}{kv}"


def _find_executable(build_dir: Path, name: str) -> Path:
    for path in (build_dir / name, build_dir / "bin" / name):
        if path.exists():
            return path
    return build_dir / name


def _memory_csv_header() -> str:
    return (
        "sample_idx,elapsed_s,pid,pid_alive,vmrss_kb,vmsize_kb,vmhwm_kb,"
        "smaps_rss_kb,smaps_pss_kb,smaps_private_dirty_kb,smaps_shared_clean_kb,"
        "mem_available_kb,cached_kb,dma_heap_pool_kb,gpu_total_kb,kgsl_shmem_usage_kb"
    )


def _baseline_sampling_shell(remote_memory_csv: str, sample_interval: float, baseline_window_s: float) -> str:
    if baseline_window_s <= 0:
        return ":"
    return f"""baseline_start_uptime=$(awk '{{print $1; exit}}' /proc/uptime 2>/dev/null)
baseline_idx=0
while true; do
  now_uptime=$(awk '{{print $1; exit}}' /proc/uptime 2>/dev/null)
  elapsed_s=$(awk -v now="${{now_uptime:-0}}" -v start="${{baseline_start_uptime:-0}}" -v window="{baseline_window_s}" 'BEGIN {{ e = now - start - window; if (e > 0) e = 0; printf "%.3f", e }}')
  mem_available=$(awk '/^MemAvailable:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  cached=$(awk '/^Cached:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  dma_heap_pool=$(awk '/^DmaHeapPool:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  gpu_total=$(awk '/^GpuTotal:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  kgsl_shmem_usage=$(awk '/^KgslShmemUsage:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  printf '%s,%s,0,0,0,0,0,0,0,0,0,%s,%s,%s,%s,%s\\n' "$baseline_idx" "$elapsed_s" "${{mem_available:-0}}" "${{cached:-0}}" "${{dma_heap_pool:-0}}" "${{gpu_total:-0}}" "${{kgsl_shmem_usage:-0}}" >> {shlex.quote(remote_memory_csv)}
  done_flag=$(awk -v now="${{now_uptime:-0}}" -v start="${{baseline_start_uptime:-0}}" -v window="{baseline_window_s}" 'BEGIN {{ print ((now - start) >= window) ? 1 : 0 }}')
  if [ "$done_flag" = "1" ]; then
    break
  fi
  baseline_idx=$((baseline_idx + 1))
  sleep {sample_interval}
done"""


def _memory_sampling_shell(remote_memory_csv: str, sample_interval: float, live_condition: str, pid_expr: str) -> str:
    return f"""sample_idx=0
start_uptime=$(awk '{{print $1; exit}}' /proc/uptime 2>/dev/null)
while {live_condition}; do
  elapsed_s=$(awk -v start="${{start_uptime:-0}}" '{{ printf "%.3f", $1 - start }}' /proc/uptime 2>/dev/null)
  sample_pid={pid_expr}
  status_file=/proc/"$sample_pid"/status
  smaps_file=/proc/"$sample_pid"/smaps_rollup
  vmrss=$(awk '/^VmRSS:/ {{print $2; exit}}' "$status_file" 2>/dev/null)
  vmsize=$(awk '/^VmSize:/ {{print $2; exit}}' "$status_file" 2>/dev/null)
  vmhwm=$(awk '/^VmHWM:/ {{print $2; exit}}' "$status_file" 2>/dev/null)
  smaps_rss=$(awk '/^Rss:/ {{print $2; exit}}' "$smaps_file" 2>/dev/null)
  smaps_pss=$(awk '/^Pss:/ {{print $2; exit}}' "$smaps_file" 2>/dev/null)
  smaps_private_dirty=$(awk '/^Private_Dirty:/ {{print $2; exit}}' "$smaps_file" 2>/dev/null)
  smaps_shared_clean=$(awk '/^Shared_Clean:/ {{print $2; exit}}' "$smaps_file" 2>/dev/null)
  mem_available=$(awk '/^MemAvailable:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  cached=$(awk '/^Cached:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  dma_heap_pool=$(awk '/^DmaHeapPool:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  gpu_total=$(awk '/^GpuTotal:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  kgsl_shmem_usage=$(awk '/^KgslShmemUsage:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  printf '%s,%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\\n' "$sample_idx" "$elapsed_s" "$sample_pid" "${{vmrss:-0}}" "${{vmsize:-0}}" "${{vmhwm:-0}}" "${{smaps_rss:-0}}" "${{smaps_pss:-0}}" "${{smaps_private_dirty:-0}}" "${{smaps_shared_clean:-0}}" "${{mem_available:-0}}" "${{cached:-0}}" "${{dma_heap_pool:-0}}" "${{gpu_total:-0}}" "${{kgsl_shmem_usage:-0}}" >> {shlex.quote(remote_memory_csv)}
  sample_idx=$((sample_idx + 1))
  sleep {sample_interval}
done"""


def _extend_llama_rope_cli(cmd: list[str], args: argparse.Namespace) -> None:
    """Passthrough llama.cpp RoPE / YaRN flags (common_params)."""
    rs = getattr(args, "rope_scaling", None)
    if rs:
        cmd.extend(["--rope-scaling", rs])
    if getattr(args, "rope_scale", None) is not None:
        cmd.extend(["--rope-scale", str(args.rope_scale)])
    if getattr(args, "rope_freq_base", None) is not None:
        cmd.extend(["--rope-freq-base", str(args.rope_freq_base)])
    if getattr(args, "rope_freq_scale", None) is not None:
        cmd.extend(["--rope-freq-scale", str(args.rope_freq_scale)])
    if getattr(args, "yarn_orig_ctx", None) is not None:
        cmd.extend(["--yarn-orig-ctx", str(args.yarn_orig_ctx)])
    if getattr(args, "yarn_ext_factor", None) is not None:
        cmd.extend(["--yarn-ext-factor", str(args.yarn_ext_factor)])
    if getattr(args, "yarn_attn_factor", None) is not None:
        cmd.extend(["--yarn-attn-factor", str(args.yarn_attn_factor)])
    if getattr(args, "yarn_beta_slow", None) is not None:
        cmd.extend(["--yarn-beta-slow", str(args.yarn_beta_slow)])
    if getattr(args, "yarn_beta_fast", None) is not None:
        cmd.extend(["--yarn-beta-fast", str(args.yarn_beta_fast)])


def _rope_shell_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    rs = getattr(args, "rope_scaling", None)
    if rs:
        parts.append(f"--rope-scaling {shlex.quote(rs)}")
    if getattr(args, "rope_scale", None) is not None:
        parts.append(f"--rope-scale {shlex.quote(str(args.rope_scale))}")
    if getattr(args, "rope_freq_base", None) is not None:
        parts.append(f"--rope-freq-base {shlex.quote(str(args.rope_freq_base))}")
    if getattr(args, "rope_freq_scale", None) is not None:
        parts.append(f"--rope-freq-scale {shlex.quote(str(args.rope_freq_scale))}")
    if getattr(args, "yarn_orig_ctx", None) is not None:
        parts.append(f"--yarn-orig-ctx {shlex.quote(str(args.yarn_orig_ctx))}")
    if getattr(args, "yarn_ext_factor", None) is not None:
        parts.append(f"--yarn-ext-factor {shlex.quote(str(args.yarn_ext_factor))}")
    if getattr(args, "yarn_attn_factor", None) is not None:
        parts.append(f"--yarn-attn-factor {shlex.quote(str(args.yarn_attn_factor))}")
    if getattr(args, "yarn_beta_slow", None) is not None:
        parts.append(f"--yarn-beta-slow {shlex.quote(str(args.yarn_beta_slow))}")
    if getattr(args, "yarn_beta_fast", None) is not None:
        parts.append(f"--yarn-beta-fast {shlex.quote(str(args.yarn_beta_fast))}")
    return (" " + " ".join(parts)) if parts else ""


def _media_image_arg(args: argparse.Namespace) -> str:
    images = getattr(args, "remote_layout_images", None)
    if images:
        return ",".join(images)
    if args.image is None:
        raise SystemExit("--image is required for multimodal standalone mode.")
    return args.image.name


def _build_standalone_command(args: argparse.Namespace, *, use_precise_phases: bool) -> list[str]:
    selected_gpu_layers = 0 if args.processor == "cpu" else args.gpu_layers
    selected_device = args.device
    if selected_device is None and args.processor == "cpu":
        selected_device = "none"
    n_predict = args.force_generation or args.n_predict
    cmd = [
        "./opencl_phase_mtmd" if use_precise_phases else "./llama-mtmd-cli",
        "-m",
        args.model.name,
        "--mmproj",
        args.mmproj.name,
        "--image",
        _media_image_arg(args),
        "-p",
        getattr(args, "remote_prompt", args.prompt),
        "-n",
        str(n_predict),
        "-t",
        str(args.threads),
        "--n-gpu-layers",
        str(selected_gpu_layers),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--temp",
        str(args.temperature),
    ]
    if use_precise_phases:
        cmd.extend(["--phase-stats-path", "foundation_phase_stats.csv"])
        cmd.extend(["--token-io-path", "foundation_token_io.txt"])
        if args.force_generation:
            cmd.append("--force-generation")
    elif args.force_generation:
        cmd.append("--ignore-eos")
    if selected_device:
        cmd.extend(["--device", selected_device])
    if getattr(args, "cache_type_k", None):
        cmd.extend(["--cache-type-k", args.cache_type_k])
    if getattr(args, "cache_type_v", None):
        cmd.extend(["--cache-type-v", args.cache_type_v])
    if getattr(args, "fit", None) is not None:
        cmd.extend(["--fit", args.fit])
    if getattr(args, "flash_attn", None):
        cmd.extend(["--flash-attn", args.flash_attn])
    if getattr(args, "no_kv_offload", False):
        cmd.append("--no-kv-offload")
    if getattr(args, "no_warmup", False):
        cmd.append("--no-warmup")
    _extend_llama_rope_cli(cmd, args)
    return cmd


def _build_text_only_command(args: argparse.Namespace) -> list[str]:
    """Standalone text generation via upstream llama-completion (no mmproj / vision)."""
    selected_gpu_layers = 0 if args.processor == "cpu" else args.gpu_layers
    selected_device = args.device
    if selected_device is None and args.processor == "cpu":
        selected_device = "none"
    n_predict = args.force_generation or args.n_predict
    cmd: list[str] = [
        "./llama-completion",
        "-m",
        args.model.name,
        "-no-cnv",
        "-lv",
        "2",
        "-co",
        "off",
        "--no-display-prompt",
        "-st",
        "-p",
        args.prompt,
        "-n",
        str(n_predict),
        "-t",
        str(args.threads),
        "-ngl",
        str(selected_gpu_layers),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--temp",
        str(args.temperature),
    ]
    if args.force_generation:
        cmd.append("--ignore-eos")
    if selected_device:
        cmd.extend(["--device", selected_device])
    if getattr(args, "cache_type_k", None):
        cmd.extend(["--cache-type-k", args.cache_type_k])
    if getattr(args, "cache_type_v", None):
        cmd.extend(["--cache-type-v", args.cache_type_v])
    if getattr(args, "fit", None) is not None:
        cmd.extend(["--fit", args.fit])
    if getattr(args, "flash_attn", None):
        cmd.extend(["--flash-attn", args.flash_attn])
    if getattr(args, "no_kv_offload", False):
        cmd.append("--no-kv-offload")
    if getattr(args, "no_warmup", False):
        cmd.append("--no-warmup")
    _extend_llama_rope_cli(cmd, args)
    return cmd


def _cache_type_shell_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if getattr(args, "cache_type_k", None):
        parts.append(f"--cache-type-k {shlex.quote(args.cache_type_k)}")
    if getattr(args, "cache_type_v", None):
        parts.append(f"--cache-type-v {shlex.quote(args.cache_type_v)}")
    return (" " + " ".join(parts)) if parts else ""


def _fit_shell_suffix(args: argparse.Namespace) -> str:
    if getattr(args, "fit", None) is None:
        return ""
    return f" --fit {args.fit}"


def _flash_attn_kv_shell_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    fa = getattr(args, "flash_attn", None)
    if fa:
        parts.append(f"--flash-attn {shlex.quote(fa)}")
    if getattr(args, "no_kv_offload", False):
        parts.append("--no-kv-offload")
    if getattr(args, "no_warmup", False):
        parts.append("--no-warmup")
    return (" " + " ".join(parts)) if parts else ""


def _remote_llama_env_exports(args: argparse.Namespace) -> str:
    """Lines emitted after cd ... || exit 1 (LD_LIBRARY_PATH + optional experiment vars)."""
    lines: list[str] = []
    if getattr(args, "disable_attn_kv_rotation", False):
        lines.append("export LLAMA_ATTN_ROT_DISABLE=1")
    dbg_kv = os.environ.get("GGML_OPENCL_DEBUG_KV")
    if dbg_kv:
        lines.append(f"export GGML_OPENCL_DEBUG_KV={shlex.quote(dbg_kv)}")
    lines.append("export LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.")
    return "\n".join(lines)


def _build_hybrid_remote_script(args: argparse.Namespace, *, encoder_pte: Path, media: PreparedMedia) -> str:
    prompt = shlex.quote(media.prompt)
    image_arg = ",".join(path.name for path in media.layout_images)
    image_paths_arg = ",".join(path.name for path in media.frame_bins)
    group_sizes_arg = ",".join(str(n) for n in media.num_patches_list)
    device_arg = f"--device {shlex.quote(args.device)}" if args.device else ""
    remote_memory_csv = f"{args.remote_root}/android_memory_timeline.csv"
    baseline_loop = _baseline_sampling_shell(remote_memory_csv, args.sample_interval, args.baseline_window)
    memory_loop = _memory_sampling_shell(
        remote_memory_csv,
        args.sample_interval,
        'kill -0 "$decoder_pid" 2>/dev/null || kill -0 "$vision_pid" 2>/dev/null',
        '"$decoder_pid"',
    )
    cache_suffix = _cache_type_shell_suffix(args)
    fit_suffix = _fit_shell_suffix(args)
    rope_suffix = _rope_shell_suffix(args)
    flash_kv_suffix = _flash_attn_kv_shell_suffix(args)
    env_exports = _remote_llama_env_exports(args)
    return f"""#!/system/bin/sh
cd {shlex.quote(args.remote_root)} || exit 1
{env_exports}
rm -f android_memory_timeline.csv hybrid_vision_stdout.txt hybrid_decode_stdout.txt \\
  vision_output_stats.csv vision_phase_stats.csv decoder_phase_stats.csv \\
  foundation_token_io.txt foundation_inference_tokens.txt \\
  foundation_exit_code.txt vision_exit_code.txt decoder_exit_code.txt \\
  vision_ready.flag decoder_ready.flag start_encode.flag vision_embedding.svlmemb
printf '%s\\n' '{_memory_csv_header()}' > {shlex.quote(remote_memory_csv)}

{baseline_loop}

./hybrid_decode -m {shlex.quote(args.model.name)} --mmproj {shlex.quote(args.mmproj.name)} \\
  --image {shlex.quote(image_arg)} --external-embedding vision_embedding.svlmemb \\
  --phase-stats-path decoder_phase_stats.csv --ready-path decoder_ready.flag \\
  --wait-for-embedding --wait-timeout-ms 120000 \\
  --token-io-path foundation_token_io.txt {('--force-generation' if args.force_generation else '')} \\
  -p {prompt} -n {args.force_generation or args.n_predict} -c {args.ctx_size} -b {args.batch_size} -ub {args.ubatch_size} \\
  -ngl {args.gpu_layers} {device_arg}{cache_suffix}{fit_suffix}{rope_suffix}{flash_kv_suffix} > hybrid_decode_stdout.txt 2>&1 &
decoder_pid=$!

./hybrid_vision_dump --encoder_path={shlex.quote(encoder_pte.name)} \\
  --image_paths={shlex.quote(image_paths_arg)} --output_path=vision_embedding.svlmemb \\
  --group_sizes={shlex.quote(group_sizes_arg)} \\
  --stats_path=vision_output_stats.csv --phase_stats_path=vision_phase_stats.csv \\
  --ready_path=vision_ready.flag --wait_path=start_encode.flag --wait_timeout_ms=120000 \\
  > hybrid_vision_stdout.txt 2>&1 &
vision_pid=$!

(
  ready_i=0
  while [ "$ready_i" -lt 2400 ]; do
    if [ -f decoder_ready.flag ] && [ -f vision_ready.flag ]; then
      touch start_encode.flag
      exit 0
    fi
    sleep 0.05
    ready_i=$((ready_i + 1))
  done
  echo coordinator_timeout > coordinator_error.txt
  touch start_encode.flag
) &
coordinator_pid=$!

{memory_loop}

wait "$vision_pid"
vision_rc=$?
wait "$decoder_pid"
decoder_rc=$?
wait "$coordinator_pid" 2>/dev/null
echo "$vision_rc" > vision_exit_code.txt
echo "$decoder_rc" > decoder_exit_code.txt
echo "$decoder_rc" > foundation_exit_code.txt
exit "$decoder_rc"
"""


def _build_standalone_remote_script(args: argparse.Namespace, *, use_precise_phases: bool) -> str:
    remote_memory_csv = f"{args.remote_root}/android_memory_timeline.csv"
    if _standalone_completion_mode(args):
        llm_cmd = _build_text_only_command(args)
    else:
        llm_cmd = _build_standalone_command(args, use_precise_phases=use_precise_phases)
    baseline_loop = _baseline_sampling_shell(remote_memory_csv, args.sample_interval, args.baseline_window)
    memory_loop = _memory_sampling_shell(
        remote_memory_csv,
        args.sample_interval,
        'kill -0 "$runner_pid" 2>/dev/null',
        '"$runner_pid"',
    )
    env_exports = _remote_llama_env_exports(args)
    return f"""#!/system/bin/sh
cd {shlex.quote(args.remote_root)} || exit 1
{env_exports}
rm -f foundation_output.txt foundation_exit_code.txt android_memory_timeline.csv foundation_phase_stats.csv foundation_token_io.txt foundation_inference_tokens.txt
printf '%s\\n' '{_memory_csv_header()}' > {shlex.quote(remote_memory_csv)}

{baseline_loop}

( {_shell_join(llm_cmd)} > foundation_output.txt 2>&1; echo $? > foundation_exit_code.txt ) &
runner_pid=$!

{memory_loop}

wait "$runner_pid"
runner_wait_rc=$?
exit_code=$(cat foundation_exit_code.txt 2>/dev/null)
exit "${{exit_code:-$runner_wait_rc}}"
"""


def _write_and_run_remote_script(adb: list[str], remote_root: str, script_text: str) -> subprocess.CompletedProcess[str]:
    remote_script = f"{remote_root}/run_android_processor.sh"
    with tempfile.TemporaryDirectory(prefix="streamingvlm_android_script_") as tmp_script_dir:
        script_path = Path(tmp_script_dir) / "run_android_processor.sh"
        script_path.write_text(script_text, encoding="utf-8")
        _push(adb, script_path, remote_root)
    _run(adb + ["shell", f"chmod +x {shlex.quote(remote_script)}"])
    return subprocess.run(
        adb + ["shell", f"sh {shlex.quote(remote_script)}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _pull_outputs(adb: list[str], remote_root: str, result_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        local = result_dir / name
        local.unlink(missing_ok=True)
        _pull_if_exists(adb, f"{remote_root}/{name}", local)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified llama.cpp / hybrid Android VLM processors.")
    parser.add_argument("--processor", choices=("cpu", "gpu", "hybrid"), required=True)
    parser.add_argument("--serial", default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Foundation QNN manifest with vision_encoder_pte. Hybrid may use this or --vision.",
    )
    parser.add_argument(
        "--vision",
        "--vision-encoder",
        dest="vision_encoder",
        type=Path,
        default=None,
        help="Direct ExecuTorch/QNN vision encoder PTE for --processor hybrid. Overrides --manifest.",
    )
    parser.add_argument("--vision-build-dir", type=Path, default=None, help="CMake build dir containing hybrid_vision_dump. Defaults to --llama-build-dir.")
    parser.add_argument(
        "--executorch-build-dir",
        type=Path,
        default=WORKSPACE / "executorch" / "build-android-unified",
        help="ExecuTorch Android build dir containing QNN runtime libraries.",
    )
    parser.add_argument("--llama-build-dir", "--build-dir", dest="llama_build_dir", type=Path, required=True, help="Android build dir containing llama.cpp and overlay binaries.")
    parser.add_argument("--model", type=Path, required=True, help="Text GGUF model.")
    parser.add_argument(
        "--mmproj",
        type=Path,
        default=None,
        help="Mmproj GGUF for multimodal (--image given). Omitted when --image is not passed (cpu/gpu text-only). Required for --processor hybrid.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image for vision. Omit on cpu/gpu for text-only generation (llama-completion). Mutually exclusive with --video.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Input video for vision. Sampled as multiple image frames; mutually exclusive with --image.",
    )
    parser.add_argument("--num-segments", type=int, default=8, help="Uniform temporal samples for --video.")
    parser.add_argument("--max-num", type=int, default=1, help="Max InternVL dynamic-preprocess tiles per sampled video frame.")
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--n-predict", "--max-new-tokens", dest="n_predict", type=int, default=32)
    parser.add_argument(
        "--force-generation",
        "--force_generation",
        dest="force_generation",
        type=int,
        default=None,
        help="Generate exactly this many tokens by continuing through EOS/EOG where supported.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", "--n-gpu-layers", dest="gpu_layers", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None, help="llama.cpp device, e.g. GPUOpenCL.")
    parser.add_argument(
        "--cache-type-k",
        "--cache_type_k",
        dest="cache_type_k",
        default=None,
        metavar="TYPE",
        help="KV-cache dtype for K (llama.cpp --cache-type-k). Multimodal: opencl_phase_mtmd / llama-mtmd-cli; text-only (no --image): llama-completion.",
    )
    parser.add_argument(
        "--cache-type-v",
        "--cache_type_v",
        dest="cache_type_v",
        default=None,
        metavar="TYPE",
        help="KV-cache dtype for V (llama.cpp --cache-type-v), e.g. q8_0.",
    )
    parser.add_argument(
        "--fit",
        choices=("on", "off"),
        default=None,
        help="llama.cpp memory fit (passthrough --fit). Omit by default; use off to skip common_fit_params (OpenCL SET_ROWS abort workaround).",
    )
    parser.add_argument(
        "--flash-attn",
        "-fa",
        choices=("on", "off", "auto"),
        default=None,
        dest="flash_attn",
        metavar="MODE",
        help="llama.cpp --flash-attn (-fa). Omit to use binary default.",
    )
    parser.add_argument(
        "--no-kv-offload",
        action="store_true",
        dest="no_kv_offload",
        help="Pass --no-kv-offload (-nkvo) to llama (standalone GPU/CPU and hybrid_decode).",
    )
    parser.add_argument(
        "--disable-attn-kv-rotation",
        action="store_true",
        dest="disable_attn_kv_rotation",
        help="On device: export LLAMA_ATTN_ROT_DISABLE=1 before llama (see llama-kv-cache.cpp).",
    )
    parser.set_defaults(no_warmup=True)
    parser.add_argument(
        "--warmup",
        action="store_false",
        dest="no_warmup",
        help="Pass --warmup to llama (enable empty warmup before generation). Off by default for bridge benchmarks.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        dest="no_warmup",
        help="Skip empty warmup (default for this script).",
    )
    parser.add_argument(
        "--rope-scaling",
        choices=("none", "linear", "yarn"),
        default=None,
        dest="rope_scaling",
        help="llama.cpp --rope-scaling (HF rope_scaling.rope_type yarn -> yarn).",
    )
    parser.add_argument(
        "--rope-scale",
        type=float,
        default=None,
        dest="rope_scale",
        metavar="N",
        help="llama.cpp --rope-scale (HF YaRN factor maps here when extending context).",
    )
    parser.add_argument(
        "--rope-freq-base",
        type=float,
        default=None,
        dest="rope_freq_base",
        metavar="N",
        help="llama.cpp --rope-freq-base.",
    )
    parser.add_argument(
        "--rope-freq-scale",
        type=float,
        default=None,
        dest="rope_freq_scale",
        metavar="N",
        help="llama.cpp --rope-freq-scale.",
    )
    parser.add_argument(
        "--yarn-orig-ctx",
        type=int,
        default=None,
        dest="yarn_orig_ctx",
        metavar="N",
        help="llama.cpp --yarn-orig-ctx (HF original_max_position_embeddings).",
    )
    parser.add_argument(
        "--yarn-ext-factor",
        type=float,
        default=None,
        dest="yarn_ext_factor",
        metavar="N",
        help="llama.cpp --yarn-ext-factor.",
    )
    parser.add_argument(
        "--yarn-attn-factor",
        type=float,
        default=None,
        dest="yarn_attn_factor",
        metavar="N",
        help="llama.cpp --yarn-attn-factor.",
    )
    parser.add_argument(
        "--yarn-beta-slow",
        type=float,
        default=None,
        dest="yarn_beta_slow",
        metavar="N",
        help="llama.cpp --yarn-beta-slow.",
    )
    parser.add_argument(
        "--yarn-beta-fast",
        type=float,
        default=None,
        dest="yarn_beta_fast",
        metavar="N",
        help="llama.cpp --yarn-beta-fast.",
    )
    parser.add_argument(
        "--opencl-lib",
        type=Path,
        default=WORKSPACE / "third_party" / "OpenCL-ICD-Loader" / "build-android" / "libOpenCL.so",
    )
    parser.add_argument(
        "--push-opencl-loader",
        action="store_true",
        help="Push local libOpenCL.so. Disabled by default because the system Qualcomm OpenCL loader works better on the tested device.",
    )
    parser.add_argument("--soc-model", default="SM8750")
    parser.add_argument("--remote-root", "--device-workdir", dest="remote_root", default="/data/local/tmp/streamingvlm_vlm")
    parser.add_argument("--results-root", type=Path, default=FOUNDATION_LLAMA / "results" / "log")
    parser.add_argument("--sample-interval", type=float, default=0.05, help="Android /proc/meminfo sampling interval in seconds.")
    parser.add_argument(
        "--baseline-window",
        "--baseline_window",
        dest="baseline_window",
        type=float,
        default=5.0,
        help="Seconds of pre-run MemAvailable samples to average for memory_usage_summary.txt.",
    )
    parser.add_argument("--force-push", action="store_true", help="Clear the remote workdir before pushing.")
    parser.add_argument("--model-push", "--model_push", dest="model_push", action="store_true", help="Force pushing model files even if they already exist on device.")
    args = parser.parse_args()

    args.remote_root = args.remote_root.rstrip("/")
    args.baseline_window = max(args.baseline_window, 0.0)
    args.model = args.model.resolve()
    if args.vision_encoder is not None:
        args.vision_encoder = args.vision_encoder.resolve()
    args.llama_build_dir = args.llama_build_dir.resolve()
    if args.num_segments <= 0:
        raise SystemExit("--num-segments must be positive.")
    if args.max_num <= 0:
        raise SystemExit("--max-num must be positive.")
    if args.image is not None and args.video is not None:
        raise SystemExit("--image and --video are mutually exclusive.")
    if args.vision_build_dir is None:
        args.vision_build_dir = args.llama_build_dir
    else:
        args.vision_build_dir = args.vision_build_dir.resolve()

    if args.processor == "hybrid":
        if args.image is None and args.video is None:
            raise SystemExit("--processor hybrid requires --image or --video.")
        if args.mmproj is None:
            raise SystemExit("--processor hybrid requires --mmproj.")
        args.mmproj = args.mmproj.resolve()
        if args.image is not None:
            args.image = args.image.resolve()
        if args.video is not None:
            args.video = args.video.resolve()
    elif _standalone_completion_mode(args):
        args.mmproj = None
        args.image = None
        args.video = None
    else:
        if args.mmproj is None:
            raise SystemExit(
                "--mmproj is required when --image/--video is set. Omit media on cpu/gpu for text-only (llama-completion)."
            )
        args.mmproj = args.mmproj.resolve()
        if args.image is not None:
            args.image = args.image.resolve()
        if args.video is not None:
            args.video = args.video.resolve()

    exist_paths: list[Path] = [args.llama_build_dir, args.model]
    if args.mmproj is not None:
        exist_paths.append(args.mmproj)
    if args.image is not None:
        exist_paths.append(args.image)
    if args.video is not None:
        exist_paths.append(args.video)
    if args.vision_encoder is not None:
        exist_paths.append(args.vision_encoder)
    for required in exist_paths:
        if not required.exists():
            raise SystemExit(f"Missing required path: {required}")
    if args.processor == "hybrid" and args.manifest is None and args.vision_encoder is None:
        raise SystemExit("--processor hybrid requires either --manifest or --vision.")

    qnn_sdk = os.environ.get("QNN_SDK_ROOT")
    if args.processor == "hybrid" and not qnn_sdk:
        raise SystemExit("QNN_SDK_ROOT is required when --processor hybrid.")

    adb = _adb(args.serial)
    if args.force_push:
        _run(adb + ["shell", "rm", "-rf", args.remote_root])
    _run(adb + ["shell", "mkdir", "-p", args.remote_root])

    result_dir = args.results_root / _result_model_name(
        args.model,
        args.processor,
        args.ctx_size,
        args.cache_type_k,
        args.cache_type_v,
        text_only=_standalone_completion_mode(args),
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    _push_llama_runtime(
        adb,
        args.remote_root,
        args.llama_build_dir,
        args.opencl_lib,
        args.push_opencl_loader,
    )
    _push_model_cached(adb, args.model, args.remote_root, force=args.model_push)
    if args.mmproj is not None:
        _push_model_cached(adb, args.mmproj, args.remote_root, force=args.model_push)
    if args.image is not None:
        _push(adb, args.image, args.remote_root)

    encoder_pte: Path | None = None
    with tempfile.TemporaryDirectory(prefix="streamingvlm_android_inputs_") as tmp:
        media: PreparedMedia | None = None
        if args.processor == "hybrid":
            if args.vision_encoder is not None:
                encoder_pte = args.vision_encoder
            else:
                manifest = _load_manifest(args.manifest.resolve())
                encoder_pte = Path(manifest["paths"]["vision_encoder_pte"])
            vision_bin = _find_executable(args.vision_build_dir, "hybrid_vision_dump")
            decode_bin = _find_executable(args.llama_build_dir, "hybrid_decode")
            if not vision_bin.exists() or not decode_bin.exists():
                raise SystemExit("Missing hybrid_vision_dump or hybrid_decode. Build hybrid_bridge first.")
            media = _prepare_media(args, Path(tmp))
            _push(adb, vision_bin, args.remote_root)
            _push(adb, decode_bin, args.remote_root)
            _push_model_cached(adb, encoder_pte, args.remote_root, force=args.model_push)
            _push(adb, media.metadata_path, args.remote_root)
            for frame_bin in media.frame_bins:
                _push(adb, frame_bin, args.remote_root)
            pushed_layouts: set[Path] = set()
            for layout_image in media.layout_images:
                resolved_layout = layout_image.resolve()
                if args.image is not None and resolved_layout == args.image.resolve():
                    continue
                if resolved_layout in pushed_layouts:
                    continue
                _push(adb, layout_image, args.remote_root)
                pushed_layouts.add(resolved_layout)
            _push_qnn_libs(adb, args.remote_root, Path(qnn_sdk), args.executorch_build_dir, args.soc_model)
            _run(adb + ["shell", f"chmod +x {shlex.quote(args.remote_root)}/hybrid_vision_dump {shlex.quote(args.remote_root)}/hybrid_decode"])
            script_text = _build_hybrid_remote_script(args, encoder_pte=encoder_pte, media=media)
            pull_names = (
                "hybrid_vision_stdout.txt",
                "hybrid_decode_stdout.txt",
                "vision_output_stats.csv",
                "vision_phase_stats.csv",
                "decoder_phase_stats.csv",
                "foundation_token_io.txt",
                "foundation_inference_tokens.txt",
                "vision_embedding.svlmemb",
                "media_manifest.json",
                "foundation_exit_code.txt",
                "vision_exit_code.txt",
                "decoder_exit_code.txt",
                "android_memory_timeline.csv",
            )
        else:
            completion_mode = _standalone_completion_mode(args)
            if not completion_mode:
                media = _prepare_media(args, Path(tmp))
                args.remote_prompt = media.prompt
                args.remote_layout_images = [path.name for path in media.layout_images]
                _push(adb, media.metadata_path, args.remote_root)
                for layout_image in media.layout_images:
                    _push(adb, layout_image, args.remote_root)
            if completion_mode:
                completion_bin = _find_executable(args.llama_build_dir, "llama-completion")
                if not completion_bin.exists():
                    raise SystemExit(
                        "Missing llama-completion. Build llama.cpp with the completion tool target (see tools/completion)."
                    )
                chmod_targets = "llama-completion"
                use_precise_phases = False
            else:
                llama_cli = _find_executable(args.llama_build_dir, "llama-mtmd-cli")
                use_precise_phases = args.processor == "gpu" and _find_executable(args.llama_build_dir, "opencl_phase_mtmd").exists()
                if not use_precise_phases and not llama_cli.exists():
                    raise SystemExit("Missing llama-mtmd-cli. Build llama.cpp first.")
                chmod_targets = "llama-mtmd-cli"
                if use_precise_phases:
                    chmod_targets += " opencl_phase_mtmd"
            _run(adb + ["shell", f"cd {shlex.quote(args.remote_root)} && chmod +x {chmod_targets} 2>/dev/null || true"])
            script_text = _build_standalone_remote_script(args, use_precise_phases=use_precise_phases)
            pull_names = (
                "foundation_output.txt",
                "foundation_exit_code.txt",
                "foundation_phase_stats.csv",
                "foundation_token_io.txt",
                "foundation_inference_tokens.txt",
                "android_memory_timeline.csv",
            )

        run_res = _write_and_run_remote_script(adb, args.remote_root, script_text)

    (result_dir / "host_adb_output.txt").write_text(run_res.stdout, encoding="utf-8")
    _pull_outputs(adb, args.remote_root, result_dir, pull_names)
    if not (result_dir / "foundation_exit_code.txt").exists():
        (result_dir / "foundation_exit_code.txt").write_text(str(run_res.returncode), encoding="utf-8")
    return_code = (result_dir / "foundation_exit_code.txt").read_text(encoding="utf-8").strip()

    if args.processor == "hybrid":
        _finalize_hybrid_outputs(result_dir)
        (result_dir / "hybrid_decode_stdout.txt").unlink(missing_ok=True)
    else:
        _finalize_standalone_outputs(
            result_dir,
            processor=args.processor,
            return_code=return_code,
            prompt=args.prompt,
            text_only=_standalone_completion_mode(args),
        )
    return run_res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
