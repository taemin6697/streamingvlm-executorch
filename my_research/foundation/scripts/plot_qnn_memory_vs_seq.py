#!/usr/bin/env python3
"""Plot foundation QNN (or any) seq_len vs memory using memory_usage_summary.txt.

Folders are typically named like internvl3_1b_qnn_512_16a8w or internvl3_1b_qnn_2k_16a8w.
Regenerates memory_usage_summary.txt from android_memory_timeline.csv each run unless --no-write-summary."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from my_research.foundation.host.android_timeline_memory_summary import (
    parse_memory_usage_summary_txt,
    write_memory_usage_summary_from_timeline_csv,
)


def seq_len_from_run_dir(name: str) -> int | None:
    """Parse max_seq / artifact token bucket from directory name."""
    m = re.search(r"_qnn_(\d+)k_", name, re.I)
    if m:
        return int(m.group(1)) * 1024
    m = re.search(r"_qnn_(\d+)_", name, re.I)
    if m:
        return int(m.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Directory containing per-run subfolders (e.g. .../qnn/16a8w)",
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
        help="Output PNG (default: <parent>/qnn_memory_vs_seq.png)",
    )
    parser.add_argument(
        "--no-write-summary",
        action="store_true",
        help="Do not overwrite memory_usage_summary.txt from android_memory_timeline.csv",
    )
    args = parser.parse_args()

    parent = args.parent.resolve()
    Row = tuple[int, float]
    rows: list[Row] = []

    for folder in sorted(parent.glob(args.glob)):
        if not folder.is_dir():
            continue
        seq = seq_len_from_run_dir(folder.name)
        if seq is None:
            continue
        summary_path = folder / "memory_usage_summary.txt"
        timeline_path = folder / "android_memory_timeline.csv"
        if not args.no_write_summary and timeline_path.is_file():
            write_memory_usage_summary_from_timeline_csv(folder)
        if not summary_path.is_file():
            continue
        sm = parse_memory_usage_summary_txt(summary_path)
        drop = sm.get("actual_memory_used_from_baseline_avg_mib")
        if drop is None:
            continue
        rows.append((seq, float(drop)))

    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 runs with memory summaries under {parent}; found {len(rows)}")

    rows.sort(key=lambda r: r[0])
    seqs = [r[0] for r in rows]

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    x_idx = np.arange(len(seqs), dtype=float)
    x_labels = [str(c) for c in seqs]
    usage_mib = [r[1] for r in rows]

    def _plain_axes(ax: plt.Axes, *, include_x: bool = True, include_y: bool = True) -> None:
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        if include_x:
            ax.xaxis.set_major_formatter(fmt)
        if include_y:
            ax.yaxis.set_major_formatter(fmt)

    fig, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_ylabel("Memory usage (MiB)")
    ax1.plot(
        x_idx,
        usage_mib,
        "o-",
        color="tab:blue",
        linewidth=2,
        markersize=8,
        label="System memory usage (first MemAvail − min MemAvail)",
    )
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlim(x_idx[0] - 0.5, x_idx[-1] + 0.5)
    ax1.set_xlabel("Exported seq_len / manifest ctx (categorical)")
    ax1.legend(loc="best")
    ax1.set_title(f"{parent.name}: QNN memory usage vs seq (device)")
    ax1.grid(True, alpha=0.35)
    _plain_axes(ax1, include_x=False)

    out = args.output or (parent / "qnn_memory_vs_seq.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
