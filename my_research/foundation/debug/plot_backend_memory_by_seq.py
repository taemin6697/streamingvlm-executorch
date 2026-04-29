#!/usr/bin/env python3
"""Plot XNNPACK/Vulkan memory usage by context length from run logs.

Default metric is drop from maximum to minimum `mem_available_kb`, which captures
how much system-available memory decreased during the run.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SEQ_TAG_TO_LEN = {
    "512": 512,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot XNNPACK/Vulkan memory usage up to 8k context length."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Workspace root. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--column",
        default="mem_available_kb",
        help="Memory column from android_memory_timeline.csv. Default: mem_available_kb.",
    )
    parser.add_argument(
        "--metric",
        choices=("max_minus_min", "peak_minus_min", "first_minus_min"),
        default="max_minus_min",
        help=(
            "Memory usage formula. max_minus_min is the default for mem_available_kb "
            "because lower available memory means higher usage. peak_minus_min is "
            "kept as an alias for RSS-style columns."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "backend_memory_by_seq.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(__file__).resolve().parent / "backend_memory_by_seq.csv",
        help="Output summary CSV path.",
    )
    return parser.parse_args()


def seq_len_from_artifact_name(name: str, backend: str) -> int | None:
    match = re.fullmatch(rf"internvl3_{backend}_1b_(512|1k|2k|4k|8k|16k)_fp16", name)
    if not match:
        return None
    tag = match.group(1)
    return SEQ_TAG_TO_LEN.get(tag)


def read_memory_values(csv_path: Path, column: str) -> list[float]:
    values: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not contain column {column!r}")
        for row in reader:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            values.append(float(raw))
    if not values:
        raise ValueError(f"{csv_path} has no numeric values for {column!r}")
    return values


def summarize_backend(root: Path, backend: str, column: str, metric: str) -> list[dict]:
    log_root = root / "my_research" / "foundation" / "results" / "log" / backend
    rows: list[dict] = []
    for csv_path in sorted(log_root.glob(f"internvl3_{backend}_1b_*_fp16/android_memory_timeline.csv")):
        seq_len = seq_len_from_artifact_name(csv_path.parent.name, backend)
        if seq_len is None or seq_len > 8192:
            continue
        values = read_memory_values(csv_path, column)
        min_kb = min(values)
        first_kb = values[0]
        peak_kb = max(values)
        if metric == "first_minus_min":
            used_kb = first_kb - min_kb
        else:
            used_kb = peak_kb - min_kb
        rows.append(
            {
                "backend": backend,
                "seq_len": seq_len,
                "artifact": csv_path.parent.name,
                "column": column,
                "metric": metric,
                "first_mib": first_kb / 1024.0,
                "min_mib": min_kb / 1024.0,
                "peak_mib": peak_kb / 1024.0,
                "used_mib": used_kb / 1024.0,
                "csv_path": str(csv_path),
            }
        )
    return sorted(rows, key=lambda row: row["seq_len"])


def write_summary_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "seq_len",
        "artifact",
        "column",
        "metric",
        "first_mib",
        "min_mib",
        "peak_mib",
        "used_mib",
        "csv_path",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for backend, color in (("xnnpack", "tab:blue"), ("vulkan", "tab:orange")):
        backend_rows = [row for row in rows if row["backend"] == backend]
        if not backend_rows:
            continue
        ax.plot(
            [row["seq_len"] for row in backend_rows],
            [row["used_mib"] for row in backend_rows],
            marker="o",
            linewidth=2,
            label=backend.upper(),
            color=color,
        )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Memory Used (MiB)")
    ax.set_title("Backend Memory Usage by Sequence Length")
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)


def main() -> int:
    args = parse_args()
    rows = []
    for backend in ("xnnpack", "vulkan"):
        rows.extend(summarize_backend(args.root, backend, args.column, args.metric))
    if not rows:
        raise SystemExit("No matching memory logs found.")
    write_summary_csv(rows, args.csv_output)
    plot(rows, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
