#!/usr/bin/env python3
"""Meeting plots: OpenCL vs QNN vs CPU under for_meetting/ (memory + phase times vs seq_len).

Memory Y: actual_memory_used_from_baseline_avg_mib (from memory_usage_summary.txt;
regenerates summaries from android_memory_timeline.csv when missing).

OpenCL prefill (ms): sum(ImagePrefill) + sum(T_Prefill) from foundation_phase_stats.csv (preferred)
  or foundation_proc.csv.

CPU prefill (ms): same rule from foundation_proc.csv (typical CPU sweep has no phase_stats file).

QNN prefill (ms): sum(EmbeddingAndMerging) + sum(T_Prefill) from foundation_proc.csv.

Decode (ms): sum(D.total_ms) if non-zero else aggregate Decode row.

Usage:
  Default (no args): InternVL3-1B OpenCL + QNN + CPU under for_meetting/opencl|qnn|cpu.

  OpenCL-only bundles (children named *_opencl_ctx_<n>):
    python3 plot_internvl3_opencl_vs_qnn_meeting.py \\
      --opencl-bundle for_meetting/2026-05-06/8B/InternVL3-8B \\
      --opencl-bundle for_meetting/2026-05-06/2B/InternVL3-2B

Writes meeting_*_opencl.png inside each bundle directory.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Callable

FOR_MEETING = Path(__file__).resolve().parent
OPENCL_PARENT = FOR_MEETING / "opencl" / "InternVL3-1B"
CPU_PARENT = FOR_MEETING / "cpu"
QNN_PARENT = FOR_MEETING / "qnn" / "InternVL3-1B"

OUT_DIR = FOR_MEETING


def seq_len_opencl_dir(name: str) -> int | None:
    m = re.search(r"_opencl_ctx_(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_ctx_(\d+)$", name)
    return int(m.group(1)) if m else None


def seq_len_cpu_dir(name: str) -> int | None:
    m = re.search(r"_cpu_ctx_(\d+)$", name)
    return int(m.group(1)) if m else None


def seq_len_qnn_dir(name: str) -> int | None:
    m = re.search(r"_qnn_(\d+)k_", name, re.I)
    if m:
        return int(m.group(1)) * 1024
    m = re.search(r"_qnn_(\d+)_", name, re.I)
    return int(m.group(1)) if m else None


def _read_proc_skip_comments(path: Path) -> csv.DictReader:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    data = "\n".join(line for line in lines if not line.startswith("#"))
    from io import StringIO

    return csv.DictReader(StringIO(data))


def parse_llamacpp_mtmd_phases(phase_or_proc: Path) -> tuple[float, float, float]:
    """OpenCL-style rows: V_Encode, ImagePrefill, T_Prefill, D, Decode. Returns (v_encode_ms, prefill_ms, decode_ms)."""
    v_enc = 0.0
    img_prefill = 0.0
    t_prefill = 0.0
    decode_d = 0.0
    decode_agg = 0.0
    if not phase_or_proc.is_file():
        return 0.0, 0.0, 0.0
    for row in _read_proc_skip_comments(phase_or_proc):
        rt = row.get("row_type", "")
        try:
            ms = float(row.get("total_ms") or 0.0)
        except ValueError:
            ms = 0.0
        if rt == "V_Encode":
            v_enc += ms
        elif rt == "ImagePrefill":
            img_prefill += ms
        elif rt == "T_Prefill":
            t_prefill += ms
        elif rt == "D":
            decode_d += ms
        elif rt == "Decode":
            decode_agg += ms
    dec = decode_d if decode_d > 0.0 else decode_agg
    return v_enc, img_prefill + t_prefill, dec


def parse_qnn_phases(proc_csv: Path) -> tuple[float, float, float]:
    """Returns (v_encode_ms, prefill_ms, decode_ms). Prefill = merge + T_Prefill."""
    v_enc = 0.0
    merge_ms = 0.0
    t_prefill = 0.0
    decode_d = 0.0
    decode_agg = 0.0
    if not proc_csv.is_file():
        return 0.0, 0.0, 0.0
    for row in _read_proc_skip_comments(proc_csv):
        rt = row.get("row_type", "")
        try:
            ms = float(row.get("total_ms") or 0.0)
        except ValueError:
            ms = 0.0
        if rt == "V_Encode":
            v_enc += ms
        elif rt == "EmbeddingAndMerging":
            merge_ms += ms
        elif rt == "T_Prefill":
            t_prefill += ms
        elif rt == "D":
            decode_d += ms
        elif rt == "Decode":
            decode_agg += ms
    dec = decode_d if decode_d > 0.0 else decode_agg
    return v_enc, merge_ms + t_prefill, dec


def _collect_with_memory_and_phases(
    parent: Path,
    seq_fn: Callable[[str], int | None],
    phase_paths: Callable[[Path], list[Path]],
    phase_parse: Callable[[Path], tuple[float, float, float]],
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not parent.is_dir():
        return out
    from my_research.foundation.host.android_timeline_memory_summary import (
        parse_memory_usage_summary_txt,
        write_memory_usage_summary_from_timeline_csv,
    )

    for folder in sorted(parent.iterdir()):
        if not folder.is_dir():
            continue
        seq = seq_fn(folder.name)
        if seq is None:
            continue
        summary = folder / "memory_usage_summary.txt"
        timeline = folder / "android_memory_timeline.csv"
        if timeline.is_file():
            if not summary.is_file():
                write_memory_usage_summary_from_timeline_csv(folder)
        if not summary.is_file():
            continue
        sm = parse_memory_usage_summary_txt(summary)
        mib = sm.get("actual_memory_used_from_baseline_avg_mib")
        if mib is None:
            continue
        v_e, pre, dec = 0.0, 0.0, 0.0
        for pth in phase_paths(folder):
            if pth.is_file():
                v_e, pre, dec = phase_parse(pth)
                break
        out[seq] = {
            "memory_mib": float(mib),
            "v_encode_s": v_e / 1000.0,
            "prefill_s": pre / 1000.0,
            "decode_s": dec / 1000.0,
        }
    return out


def collect_opencl() -> dict[int, dict[str, float]]:
    def paths(folder: Path) -> list[Path]:
        return [folder / "foundation_phase_stats.csv", folder / "foundation_proc.csv"]

    return _collect_with_memory_and_phases(
        OPENCL_PARENT, seq_len_opencl_dir, paths, parse_llamacpp_mtmd_phases
    )


def collect_opencl_under(parent: Path) -> dict[int, dict[str, float]]:
    """Scan direct child dirs named *_opencl_ctx_<n> (OpenCL-only meeting bundles)."""

    def paths(folder: Path) -> list[Path]:
        return [folder / "foundation_phase_stats.csv", folder / "foundation_proc.csv"]

    return _collect_with_memory_and_phases(
        parent.resolve(), seq_len_opencl_dir, paths, parse_llamacpp_mtmd_phases
    )


def collect_cpu() -> dict[int, dict[str, float]]:
    def paths(folder: Path) -> list[Path]:
        return [folder / "foundation_phase_stats.csv", folder / "foundation_proc.csv"]

    return _collect_with_memory_and_phases(
        CPU_PARENT, seq_len_cpu_dir, paths, parse_llamacpp_mtmd_phases
    )


def collect_qnn() -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not QNN_PARENT.is_dir():
        return out
    from my_research.foundation.host.android_timeline_memory_summary import (
        parse_memory_usage_summary_txt,
        write_memory_usage_summary_from_timeline_csv,
    )

    for folder in sorted(QNN_PARENT.iterdir()):
        if not folder.is_dir():
            continue
        seq = seq_len_qnn_dir(folder.name)
        if seq is None:
            continue
        summary = folder / "memory_usage_summary.txt"
        timeline = folder / "android_memory_timeline.csv"
        if timeline.is_file():
            if not summary.is_file():
                write_memory_usage_summary_from_timeline_csv(folder)
        if not summary.is_file():
            continue
        sm = parse_memory_usage_summary_txt(summary)
        mib = sm.get("actual_memory_used_from_baseline_avg_mib")
        if mib is None:
            continue
        proc = folder / "foundation_proc.csv"
        v_e, pre, dec = parse_qnn_phases(proc)
        out[seq] = {
            "memory_mib": float(mib),
            "v_encode_s": v_e / 1000.0,
            "prefill_s": pre / 1000.0,
            "decode_s": dec / 1000.0,
        }
    return out


def _plot_backends(
    title: str,
    ylabel: str,
    outfile: Path,
    series_map: dict[str, dict[int, dict[str, float]]],
    key: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    all_seq: set[int] = set()
    for d in series_map.values():
        all_seq |= set(d.keys())
    seq_sorted = sorted(all_seq)
    if len(seq_sorted) < 1:
        raise SystemExit(f"No data for plot: {title}")

    xs = np.arange(len(seq_sorted), dtype=float)
    labels = [str(s) for s in seq_sorted]

    styles = [
        ("OpenCL (llama.cpp)", "o-", "tab:blue"),
        ("QNN (ExecuTorch)", "s-", "tab:orange"),
        ("CPU (llama.cpp)", "^-", "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    for label, fmt, color in styles:
        data = series_map.get(label)
        if not data:
            continue
        ys = [data[s][key] if s in data else np.nan for s in seq_sorted]
        if all(np.isnan(ys)):
            continue
        ax.plot(xs, ys, fmt, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_xlabel("seq_len / ctx (categorical ticks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.35)
    fmt_y = mticker.ScalarFormatter(useOffset=False)
    fmt_y.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt_y)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_opencl_bundle(out_parent: Path, title_stem: str, data: dict[int, dict[str, float]]) -> None:
    """Write four PNGs under out_parent (single OpenCL series)."""
    series_map = {"OpenCL (llama.cpp)": data}
    out_parent.mkdir(parents=True, exist_ok=True)
    _plot_backends(
        f"{title_stem}: system memory usage vs ctx (device)",
        "Memory usage (MiB)\nactual_memory_used_from_baseline_avg_mib",
        out_parent / "meeting_memory_vs_seq_opencl.png",
        series_map,
        "memory_mib",
    )
    _plot_backends(
        f"{title_stem}: vision encode time vs ctx",
        "Time (s)",
        out_parent / "meeting_v_encode_vs_seq_opencl.png",
        series_map,
        "v_encode_s",
    )
    _plot_backends(
        f"{title_stem}: prefill time vs ctx\nOpenCL = ImagePrefill + T_Prefill",
        "Time (s)",
        out_parent / "meeting_prefill_vs_seq_opencl.png",
        series_map,
        "prefill_s",
    )
    _plot_backends(
        f"{title_stem}: decode time vs ctx",
        "Time (s)",
        out_parent / "meeting_decode_vs_seq_opencl.png",
        series_map,
        "decode_s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting plots: OpenCL / QNN / CPU or OpenCL-only bundles.")
    parser.add_argument(
        "--opencl-bundle",
        action="append",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory whose children are *_opencl_ctx_* runs. Writes meeting_*.png inside DIR. Repeatable.",
    )
    args = parser.parse_args()

    if args.opencl_bundle:
        for raw in args.opencl_bundle:
            bundle = raw.resolve()
            data = collect_opencl_under(bundle)
            if not data:
                print(f"warning: no usable runs under {bundle}", file=sys.stderr)
                continue
            plot_opencl_bundle(bundle, bundle.name, data)
            print(f"Wrote OpenCL meeting PNGs under {bundle}")
        return

    opencl = collect_opencl()
    cpu = collect_cpu()
    qnn = collect_qnn()
    if not opencl and not cpu and not qnn:
        raise SystemExit(f"No data under {OPENCL_PARENT}, {CPU_PARENT}, or {QNN_PARENT}")

    series_map = {
        "OpenCL (llama.cpp)": opencl,
        "QNN (ExecuTorch)": qnn,
        "CPU (llama.cpp)": cpu,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_backends(
        "InternVL3-1B: system memory usage vs ctx (device)",
        "Memory usage (MiB)\nactual_memory_used_from_baseline_avg_mib",
        OUT_DIR / "meeting_memory_vs_seq_opencl_qnn_cpu.png",
        series_map,
        "memory_mib",
    )
    _plot_backends(
        "InternVL3-1B: vision encode time vs ctx",
        "Time (s)",
        OUT_DIR / "meeting_v_encode_vs_seq_opencl_qnn_cpu.png",
        series_map,
        "v_encode_s",
    )
    _plot_backends(
        "InternVL3-1B: prefill time vs ctx\n"
        "OpenCL/CPU = ImagePrefill + T_Prefill | QNN = EmbeddingAndMerging + T_Prefill",
        "Time (s)",
        OUT_DIR / "meeting_prefill_vs_seq_opencl_qnn_cpu.png",
        series_map,
        "prefill_s",
    )
    _plot_backends(
        "InternVL3-1B: decode time vs ctx",
        "Time (s)",
        OUT_DIR / "meeting_decode_vs_seq_opencl_qnn_cpu.png",
        series_map,
        "decode_s",
    )
    print(f"Wrote PNGs under {OUT_DIR}")


if __name__ == "__main__":
    main()
