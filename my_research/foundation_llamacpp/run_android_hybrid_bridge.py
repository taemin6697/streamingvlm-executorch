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
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
FOUNDATION_LLAMA = Path(__file__).resolve().parent


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
    image = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype("float32").tofile(output_path)


def _prepare_inputs(image: Path, work_dir: Path) -> tuple[Path, Path]:
    load_image = importlib.import_module("transformers.image_utils").load_image

    frame_bin = work_dir / "frame_0000.bin"
    layout_image = work_dir / image.name
    _normalize_image_to_bin(load_image(str(image)), frame_bin)
    if image.resolve() != layout_image.resolve():
        layout_image.write_bytes(image.read_bytes())
    return frame_bin, layout_image


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
                if path.is_file() and (path.suffix == ".so" or path.name in {"hybrid_decode", "llama-mtmd-cli"}):
                    _push(adb, path, remote_dir)
    for name in ("hybrid_decode",):
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


def _match_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else 0.0


def _match_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


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
            "LayoutTokenize: mtmd layout  ImagePrefill: image embedding prefill  "
            "T_Prefill: text prompt prefill  D: one generated-token decode"
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
        label = f"{name}{phase_count[name]}" if name in {"T_Prefill"} else name
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


def _finalize_outputs(result_dir: Path) -> None:
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
    _write_png_memory_timeline(result_dir, memory_rows, plot_phase_rows)
    if plot_phase_rows:
        _write_png_phase_duration_from_rows(result_dir, plot_phase_rows)
    else:
        _write_png_phase_duration(result_dir, perf)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run split-process ExecuTorch-QNN vision + llama.cpp decoder bridge on Android.")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--manifest", type=Path, required=True, help="Foundation QNN manifest with vision_encoder_pte.")
    parser.add_argument("--vision-build-dir", type=Path, required=True, help="CMake build dir containing hybrid_vision_dump.")
    parser.add_argument(
        "--executorch-build-dir",
        type=Path,
        default=WORKSPACE / "executorch" / "build-android-unified",
        help="ExecuTorch Android build dir containing QNN runtime libraries.",
    )
    parser.add_argument("--llama-build-dir", type=Path, required=True, help="CMake build dir containing hybrid_decode and llama libs.")
    parser.add_argument("--model", type=Path, required=True, help="Text GGUF model.")
    parser.add_argument("--mmproj", type=Path, required=True, help="llama.cpp mmproj GGUF, used only for token layout.")
    parser.add_argument("--image", type=Path, default=FOUNDATION_LLAMA / "sample_coco_cats_448.jpg")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--device", default=None, help="llama.cpp device, e.g. Adreno(TM) 750.")
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
    parser.add_argument("--remote-root", default="/data/local/tmp/streamingvlm_hybrid_bridge")
    parser.add_argument("--results-root", type=Path, default=FOUNDATION_LLAMA / "results" / "log" / "hybrid_bridge")
    parser.add_argument("--sample-interval", type=float, default=0.05, help="Android /proc/meminfo sampling interval in seconds.")
    parser.add_argument("--force-push", action="store_true")
    args = parser.parse_args()

    qnn_sdk = os.environ.get("QNN_SDK_ROOT")
    if not qnn_sdk:
        raise SystemExit("QNN_SDK_ROOT is required for the ExecuTorch QNN vision process.")

    adb = _adb(args.serial)
    if args.force_push:
        _run(adb + ["shell", "rm", "-rf", args.remote_root])
    _run(adb + ["shell", "mkdir", "-p", args.remote_root])

    manifest = _load_manifest(args.manifest)
    encoder_pte = Path(manifest["paths"]["vision_encoder_pte"])
    executorch_build_dir = args.executorch_build_dir

    vision_bin = args.vision_build_dir / "hybrid_vision_dump"
    if not vision_bin.exists():
        vision_bin = args.vision_build_dir / "bin" / "hybrid_vision_dump"
    decode_bin = args.llama_build_dir / "hybrid_decode"
    if not decode_bin.exists():
        decode_bin = args.llama_build_dir / "bin" / "hybrid_decode"
    if not vision_bin.exists() or not decode_bin.exists():
        raise SystemExit("Missing hybrid_vision_dump or hybrid_decode. Build hybrid_bridge first.")

    with tempfile.TemporaryDirectory(prefix="streamingvlm_hybrid_") as tmp:
        frame_bin, layout_image = _prepare_inputs(args.image, Path(tmp))
        for local in (vision_bin, decode_bin, encoder_pte, args.model, args.mmproj, frame_bin, layout_image):
            _push(adb, local, args.remote_root)
        _push_qnn_libs(adb, args.remote_root, Path(qnn_sdk), executorch_build_dir, args.soc_model)
        _push_llama_runtime(
            adb,
            args.remote_root,
            args.llama_build_dir,
            args.opencl_lib,
            args.push_opencl_loader,
        )
        _run(adb + ["shell", f"chmod +x {shlex.quote(args.remote_root)}/hybrid_vision_dump {shlex.quote(args.remote_root)}/hybrid_decode"])

    result_dir = args.results_root / args.model.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt = shlex.quote(args.prompt)
    device_arg = f"--device {shlex.quote(args.device)}" if args.device else ""
    remote_memory_csv = f"{args.remote_root}/android_memory_timeline.csv"
    remote_script = f"{args.remote_root}/run_hybrid_bridge.sh"
    script_text = f"""#!/system/bin/sh
cd {shlex.quote(args.remote_root)} || exit 1
export LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.
rm -f android_memory_timeline.csv hybrid_vision_stdout.txt hybrid_decode_stdout.txt \\
  vision_output_stats.csv vision_phase_stats.csv decoder_phase_stats.csv \\
  foundation_exit_code.txt vision_exit_code.txt decoder_exit_code.txt \\
  vision_ready.flag decoder_ready.flag start_encode.flag vision_embedding.svlmemb
printf '%s\\n' 'sample_idx,elapsed_s,pid,pid_alive,vmrss_kb,vmsize_kb,vmhwm_kb,smaps_rss_kb,smaps_pss_kb,smaps_private_dirty_kb,smaps_shared_clean_kb,mem_available_kb,cached_kb,dma_heap_pool_kb,gpu_total_kb,kgsl_shmem_usage_kb' > {shlex.quote(remote_memory_csv)}

./hybrid_decode -m {shlex.quote(args.model.name)} --mmproj {shlex.quote(args.mmproj.name)} \\
  --image {shlex.quote(layout_image.name)} --external-embedding vision_embedding.svlmemb \\
  --phase-stats-path decoder_phase_stats.csv --ready-path decoder_ready.flag \\
  --wait-for-embedding --wait-timeout-ms 120000 \\
  -p {prompt} -n {args.n_predict} -c {args.ctx_size} -b {args.batch_size} -ub {args.ubatch_size} \\
  -ngl {args.gpu_layers} {device_arg} > hybrid_decode_stdout.txt 2>&1 &
decoder_pid=$!

./hybrid_vision_dump --encoder_path={shlex.quote(encoder_pte.name)} \\
  --image_path=frame_0000.bin --output_path=vision_embedding.svlmemb \\
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

sample_idx=0
while kill -0 "$decoder_pid" 2>/dev/null || kill -0 "$vision_pid" 2>/dev/null; do
  elapsed_s=$(awk -v i="$sample_idx" 'BEGIN {{ printf "%.3f", i * {args.sample_interval} }}')
  mem_available=$(awk '/^MemAvailable:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  cached=$(awk '/^Cached:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  dma_heap_pool=$(awk '/^DmaHeapPool:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  gpu_total=$(awk '/^GpuTotal:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  kgsl_shmem_usage=$(awk '/^KgslShmemUsage:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  printf '%s,%s,%s,1,0,0,0,0,0,0,0,%s,%s,%s,%s,%s\\n' "$sample_idx" "$elapsed_s" "$decoder_pid" "${{mem_available:-0}}" "${{cached:-0}}" "${{dma_heap_pool:-0}}" "${{gpu_total:-0}}" "${{kgsl_shmem_usage:-0}}" >> {shlex.quote(remote_memory_csv)}
  sample_idx=$((sample_idx + 1))
  sleep {args.sample_interval}
done

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
    with tempfile.TemporaryDirectory(prefix="streamingvlm_hybrid_script_") as tmp_script_dir:
        script_path = Path(tmp_script_dir) / "run_hybrid_bridge.sh"
        script_path.write_text(script_text, encoding="utf-8")
        _push(adb, script_path, args.remote_root)
    _run(adb + ["shell", f"chmod +x {shlex.quote(remote_script)}"])
    run_res = subprocess.run(
        adb + ["shell", f"sh {shlex.quote(remote_script)}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (result_dir / "host_adb_output.txt").write_text(run_res.stdout, encoding="utf-8")
    for name in (
        "hybrid_vision_stdout.txt",
        "hybrid_decode_stdout.txt",
        "vision_output_stats.csv",
        "vision_phase_stats.csv",
        "decoder_phase_stats.csv",
        "vision_embedding.svlmemb",
        "foundation_exit_code.txt",
        "vision_exit_code.txt",
        "decoder_exit_code.txt",
        "android_memory_timeline.csv",
    ):
        _pull_if_exists(adb, f"{args.remote_root}/{name}", result_dir / name)
    if not (result_dir / "foundation_exit_code.txt").exists():
        (result_dir / "foundation_exit_code.txt").write_text(str(run_res.returncode), encoding="utf-8")
    _finalize_outputs(result_dir)
    # Keep the result directory aligned with the other foundation runners:
    # foundation_output.txt is the canonical decoder stdout.
    (result_dir / "hybrid_decode_stdout.txt").unlink(missing_ok=True)
    return run_res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
