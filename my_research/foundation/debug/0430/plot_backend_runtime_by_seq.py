#!/usr/bin/env python3
"""Plot foundation runtime phase breakdown by backend and sequence length."""

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

PHASES = (
    ("load_ms", "Loading", "L"),
    ("vision_ms", "Vision Encode", "V_Encode"),
    ("merge_ms", "Embedding/Merge", "EmbeddingAndMerging"),
    ("prefill_ms", "Text Prefill", "T_Prefill"),
    ("decode_ms", "Decode", "D"),
)


def default_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "my_research" / "foundation" / "results" / "log").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked runtime bars for XNNPACK/Vulkan/QNN runs up to 8k."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_workspace_root(),
        help="Workspace root. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for PNG and summary CSV files.",
    )
    parser.add_argument(
        "--backend",
        choices=("xnnpack", "vulkan", "qnn", "all"),
        default="all",
        help="Backend to plot. Default: all.",
    )
    return parser.parse_args()


def seq_len_from_artifact_name(name: str, backend: str) -> int | None:
    if backend == "qnn":
        match = re.fullmatch(r"internvl3_1b_qnn_(512|1k|2k|4k|8k|16k)_.+", name)
    else:
        match = re.fullmatch(rf"internvl3_{backend}_1b_(512|1k|2k|4k|8k|16k)_.+", name)
    if not match:
        return None
    return SEQ_TAG_TO_LEN.get(match.group(1))


def summarize_proc_csv(csv_path: Path, backend: str, seq_len: int) -> dict[str, object]:
    phase_ms = {key: 0.0 for key, _, _ in PHASES}
    decode_aggregate_ms = 0.0
    elapsed_s = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    artifact = csv_path.parent.name

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for row in reader:
            row_type = row.get("row_type", "")
            try:
                total_ms = float(row.get("total_ms") or 0.0)
            except ValueError:
                total_ms = 0.0

            for key, _, phase_name in PHASES:
                if row_type == phase_name:
                    phase_ms[key] += total_ms
                    if row_type == "T_Prefill":
                        try:
                            prefill_tokens = max(prefill_tokens, int(float(row.get("kv_pos") or 0)))
                        except ValueError:
                            pass
                    if row_type == "D":
                        decode_tokens += 1
                    break
            if row_type == "Decode":
                decode_aggregate_ms += total_ms

            try:
                elapsed_s = max(elapsed_s, float(row.get("elapsed_s_end") or 0.0))
            except ValueError:
                pass

    if phase_ms["decode_ms"] == 0.0 and decode_aggregate_ms > 0.0:
        phase_ms["decode_ms"] = decode_aggregate_ms

    total_ms = sum(phase_ms.values())
    return {
        "backend": backend,
        "seq_len": seq_len,
        "artifact": artifact,
        **phase_ms,
        "measured_total_ms": total_ms,
        "elapsed_s": elapsed_s,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "proc_csv": str(csv_path),
    }


def summarize_backend(root: Path, backend: str) -> list[dict[str, object]]:
    log_root = root / "my_research" / "foundation" / "results" / "log" / backend
    rows: list[dict[str, object]] = []
    artifact_glob = "internvl3_1b_qnn_*/foundation_proc.csv"
    if backend != "qnn":
        artifact_glob = f"internvl3_{backend}_1b_*/foundation_proc.csv"
    for proc_csv in sorted(log_root.glob(artifact_glob)):
        seq_len = seq_len_from_artifact_name(proc_csv.parent.name, backend)
        if seq_len is None or seq_len > 8192:
            continue
        rows.append(summarize_proc_csv(proc_csv, backend, seq_len))
    return sorted(rows, key=lambda row: int(row["seq_len"]))


def write_summary_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "seq_len",
        "artifact",
        "load_ms",
        "vision_ms",
        "merge_ms",
        "prefill_ms",
        "decode_ms",
        "measured_total_ms",
        "elapsed_s",
        "prefill_tokens",
        "decode_tokens",
        "proc_csv",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_backend(rows: list[dict[str, object]], backend: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    backend_rows = [row for row in rows if row["backend"] == backend]
    if not backend_rows:
        raise ValueError(f"No rows found for backend {backend}")

    seq_labels = [str(row["seq_len"]) for row in backend_rows]
    x = list(range(len(backend_rows)))
    bottoms = [0.0] * len(backend_rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    colors = {
        "load_ms": "tab:gray",
        "vision_ms": "tab:blue",
        "merge_ms": "tab:purple",
        "prefill_ms": "tab:orange",
        "decode_ms": "tab:green",
    }
    for key, label, _ in PHASES:
        values = [float(row[key]) / 1000.0 for row in backend_rows]
        ax.bar(x, values, bottom=bottoms, label=label, color=colors[key])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for idx, total_s in enumerate(bottoms):
        ax.text(idx, total_s, f"{total_s:.1f}s", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{backend.upper()} Runtime Phase Breakdown by Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Runtime (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    backends = ("xnnpack", "vulkan", "qnn") if args.backend == "all" else (args.backend,)
    rows: list[dict[str, object]] = []
    for backend in backends:
        rows.extend(summarize_backend(args.root, backend))
    if not rows:
        raise SystemExit("No matching runtime logs found.")

    write_summary_csv(rows, args.output_dir / "backend_runtime_by_seq.csv")
    for backend in backends:
        plot_backend(rows, backend, args.output_dir / f"{backend}_runtime_by_seq.png")

    print(f"Wrote {args.output_dir / 'backend_runtime_by_seq.csv'}")
    for backend in backends:
        print(f"Wrote {args.output_dir / f'{backend}_runtime_by_seq.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
