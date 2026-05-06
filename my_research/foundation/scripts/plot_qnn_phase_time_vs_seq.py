#!/usr/bin/env python3
"""Plot QNN phase times vs seq_len from per-run foundation_proc.csv.

Reads the same run folders as plot_qnn_memory_vs_seq.py (_qnn_<512|1k|...>_) and sums:
  V_Encode, T_Prefill, Decode (per-token D rows when non-zero; else aggregate Decode row)."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def seq_len_from_run_dir(name: str) -> int | None:
    """Parse max_seq / artifact token bucket from directory name."""
    m = re.search(r"_qnn_(\d+)k_", name, re.I)
    if m:
        return int(m.group(1)) * 1024
    m = re.search(r"_qnn_(\d+)_", name, re.I)
    if m:
        return int(m.group(1))
    return None


def summarize_qnn_proc_phase_ms(proc_csv: Path) -> tuple[float, float, float] | None:
    """Return (vision_encode_ms, prefill_ms, decode_ms). Matches debug plot_backend decode fallback."""
    vision_ms = 0.0
    prefill_ms = 0.0
    decode_ms = 0.0
    decode_aggregate_ms = 0.0

    try:
        with proc_csv.open(newline="") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                row_type = row.get("row_type", "")
                try:
                    total_ms = float(row.get("total_ms") or 0.0)
                except ValueError:
                    total_ms = 0.0

                if row_type == "V_Encode":
                    vision_ms += total_ms
                elif row_type == "T_Prefill":
                    prefill_ms += total_ms
                elif row_type == "D":
                    decode_ms += total_ms
                elif row_type == "Decode":
                    decode_aggregate_ms += total_ms
    except OSError:
        return None

    if decode_ms == 0.0 and decode_aggregate_ms > 0.0:
        decode_ms = decode_aggregate_ms

    return vision_ms, prefill_ms, decode_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Directory containing per-run subfolders (e.g. .../qnn/InternVL3-1B)",
    )
    parser.add_argument(
        "--glob",
        default="*_qnn_*",
        help="Glob for run folders under parent (default: *_qnn_*)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <parent>/qnn_phase_time_vs_seq.png)",
    )
    args = parser.parse_args()

    parent = args.parent.resolve()
    rows: list[tuple[int, float, float, float]] = []

    for folder in sorted(parent.glob(args.glob)):
        if not folder.is_dir():
            continue
        seq = seq_len_from_run_dir(folder.name)
        if seq is None:
            continue
        proc_path = folder / "foundation_proc.csv"
        if not proc_path.is_file():
            continue
        phases = summarize_qnn_proc_phase_ms(proc_path)
        if phases is None:
            continue
        v_ms, p_ms, d_ms = phases
        rows.append((seq, v_ms, p_ms, d_ms))

    if len(rows) < 2:
        raise SystemExit(
            f"Need at least 2 runs with foundation_proc.csv under {parent}; found {len(rows)}"
        )

    rows.sort(key=lambda r: r[0])
    seqs = [r[0] for r in rows]

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    x_idx = np.arange(len(seqs), dtype=float)
    x_labels = [str(s) for s in seqs]

    def plain_axes(ax: plt.Axes) -> None:
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), layout="constrained", sharex=True)
    series = [
        ("Vision encode (V_Encode)", [r[1] / 1000.0 for r in rows], "tab:blue"),
        ("Text prefill (T_Prefill)", [r[2] / 1000.0 for r in rows], "tab:orange"),
        ("Decode", [r[3] / 1000.0 for r in rows], "tab:green"),
    ]
    for ax, (title, yvals, color) in zip(axes, series):
        ax.plot(x_idx, yvals, "o-", color=color, linewidth=2, markersize=8)
        ax.set_ylabel("Time (s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        plain_axes(ax)

    axes[-1].set_xticks(x_idx)
    axes[-1].set_xticklabels(x_labels)
    axes[-1].set_xlim(x_idx[0] - 0.5, x_idx[-1] + 0.5)
    axes[-1].set_xlabel("Exported seq_len / manifest ctx (categorical)")
    fig.suptitle(f"{parent.name}: QNN phase time vs seq (device)", fontsize=12)

    out = args.output or (parent / "qnn_phase_time_vs_seq.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
