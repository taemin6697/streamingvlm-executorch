#!/usr/bin/env python3
"""Plot memory component deltas for a single foundation run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_LOG_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "log"
    / "vulkan"
    / "internvl3_vulkan_1b_8k_fp16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mem_available drop, GPU/KGSL allocation growth, and process RSS "
            "growth on the same timeline."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Run log directory. Default: {DEFAULT_LOG_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "vulkan_8k_memory_components.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def read_memory_timeline(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    pass
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No timeline rows found in {path}")
    return rows


def read_phases(path: Path) -> list[tuple[str, float, float]]:
    phases: list[tuple[str, float, float]] = []
    if not path.exists():
        return phases
    with path.open(newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for row in reader:
            phase = row.get("row_type", "")
            if phase == "D":
                continue
            try:
                start = float(row["elapsed_s_start"])
                end = float(row["elapsed_s_end"])
            except (KeyError, TypeError, ValueError):
                continue
            phases.append((phase, start, end))
    return phases


def series_delta(rows: list[dict[str, float]], column: str) -> tuple[list[float], list[float]]:
    times = [row["elapsed_s"] for row in rows if "elapsed_s" in row and column in row]
    values = [row[column] for row in rows if "elapsed_s" in row and column in row]
    if not values:
        raise ValueError(f"No values found for {column}")
    base = values[0]
    return times, [(value - base) / 1024.0 for value in values]


def series_available_drop(rows: list[dict[str, float]]) -> tuple[list[float], list[float]]:
    times = [
        row["elapsed_s"]
        for row in rows
        if "elapsed_s" in row and "mem_available_kb" in row
    ]
    values = [
        row["mem_available_kb"]
        for row in rows
        if "elapsed_s" in row and "mem_available_kb" in row
    ]
    if not values:
        raise ValueError("No values found for mem_available_kb")
    base = max(values)
    return times, [(base - value) / 1024.0 for value in values]


def plot(log_dir: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = read_memory_timeline(log_dir / "android_memory_timeline.csv")
    phases = read_phases(log_dir / "foundation_proc.csv")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    time, values = series_available_drop(rows)
    ax.plot(time, values, label="Available memory drop", linewidth=2.4)
    for column, label in (
        ("gpu_total_kb", "GPU total growth"),
        ("kgsl_shmem_usage_kb", "KGSL shmem growth"),
        ("self_rss_kb", "Process self RSS growth"),
        ("cached_kb", "Cached memory growth"),
    ):
        try:
            time, values = series_delta(rows, column)
        except ValueError:
            continue
        ax.plot(time, values, label=label, linewidth=1.8)

    y_min, y_max = ax.get_ylim()
    label_y = y_max * 0.94
    for phase, start, end in phases:
        if phase in {"L", "V_Encode", "EmbeddingAndMerging", "T_Prefill", "Decode"}:
            ax.axvspan(start, end, alpha=0.08)
            ax.text(
                (start + end) / 2,
                label_y,
                phase,
                rotation=90,
                ha="center",
                va="top",
                fontsize=8,
            )

    ax.set_title(f"Memory Components Timeline: {log_dir.name}")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Delta / Drop (MiB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def main() -> int:
    args = parse_args()
    plot(args.log_dir, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
