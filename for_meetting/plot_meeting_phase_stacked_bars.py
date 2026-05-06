#!/usr/bin/env python3
"""Stacked bar chart: Vision encode / Prefill / Decode vs seq (Loading omitted).

Prefill aggregates:
  - QNN: EmbeddingAndMerging + T_Prefill
  - llama (OpenCL/CPU): ImagePrefill + T_Prefill

Reads each run folder under --bundle:
  1) foundation_phase_stats.csv if present, else foundation_proc.csv
  2) Optional stats.csv (same columns as proc) if added later

Seq from folder name: *_qnn_<n>k_* / *_qnn_<n>_* / *_opencl_ctx_<n> / *_cpu_ctx_<n> / *_ctx_<n>.

Output default: <bundle>/runtime_phase_stacked_bar.png
"""

from __future__ import annotations

import argparse
import csv
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def seq_from_folder(name: str) -> int | None:
    m = re.search(r"_qnn_(\d+)k_", name, re.I)
    if m:
        return int(m.group(1)) * 1024
    m = re.search(r"_qnn_(\d+)_", name, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"_opencl_ctx_(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_cpu_ctx_(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_ctx_(\d+)$", name)
    return int(m.group(1)) if m else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
    return list(csv.DictReader(StringIO(data)))


def _pick_phase_csv(folder: Path) -> Path | None:
    for name in ("stats.csv", "foundation_phase_stats.csv", "foundation_proc.csv"):
        p = folder / name
        if p.is_file():
            return p
    return None


def extract_phase_ms(rows: list[dict[str, str]]) -> tuple[float, float, float, float]:
    """Returns loading_ms, vision_ms, prefill_ms, decode_ms.

    prefill_ms combines merge + text prefill per backend (see module docstring).
    """
    has_llama_load = any(str(r.get("row_type", "")).startswith("L_Decoder") for r in rows)
    has_qnn_l = any(r.get("row_type") == "L" for r in rows)
    has_merge = any(r.get("row_type") == "EmbeddingAndMerging" for r in rows)
    is_qnn_style = has_merge or (has_qnn_l and not has_llama_load)

    loading = vision = prefill = 0.0
    decode_d = decode_agg = 0.0

    llama_loading_types = frozenset(
        {"ImageLoad", "LayoutTokenize", "EmbeddingFileWrite", "ExternalEmbeddingRead"}
    )

    for row in rows:
        rt = (row.get("row_type") or "").strip()
        try:
            ms = float(row.get("total_ms") or 0.0)
        except ValueError:
            ms = 0.0

        if is_qnn_style:
            if rt == "L":
                loading += ms
            elif rt == "V_Encode":
                vision += ms
            elif rt == "EmbeddingAndMerging":
                prefill += ms
            elif rt == "T_Prefill":
                prefill += ms
            elif rt == "D":
                decode_d += ms
            elif rt == "Decode":
                decode_agg += ms
            continue

        if rt.startswith("L_") or rt in llama_loading_types:
            loading += ms
        elif rt == "V_Encode":
            vision += ms
        elif rt == "ImagePrefill":
            prefill += ms
        elif rt == "T_Prefill":
            prefill += ms
        elif rt == "D":
            decode_d += ms
        elif rt == "Decode":
            decode_agg += ms

    dec = decode_d if decode_d > 0.0 else decode_agg
    return loading, vision, prefill, dec


def infer_backend_label(parent: Path) -> str:
    ps = str(parent.resolve()).replace("\\", "/")
    if "/qnn/" in ps:
        return "QNN"
    if "/cpu/" in ps:
        return "CPU"
    if "/opencl/" in ps:
        return "OpenCL"
    for ch in sorted(parent.iterdir()):
        if not ch.is_dir():
            continue
        ln = ch.name.lower()
        if "_opencl_" in ln:
            return "OpenCL"
        if "_cpu_" in ln:
            return "CPU"
        if "_qnn_" in ln:
            return "QNN"
    return ""


def plot_bundle(bundle: Path, outfile: Path | None) -> Path:
    bundle = bundle.resolve()
    rows_by_seq: dict[int, tuple[float, float, float, float]] = {}

    for folder in sorted(bundle.iterdir()):
        if not folder.is_dir():
            continue
        seq = seq_from_folder(folder.name)
        if seq is None:
            continue
        csv_path = _pick_phase_csv(folder)
        if csv_path is None:
            continue
        try:
            parsed = extract_phase_ms(_read_csv_rows(csv_path))
        except OSError:
            continue
        rows_by_seq[seq] = parsed

    if len(rows_by_seq) < 1:
        raise SystemExit(f"No runs with phase CSV under {bundle}")

    seqs = sorted(rows_by_seq.keys())
    n = len(seqs)
    x = np.arange(n, dtype=float)

    # seconds, bottom-to-top; index 0 = Loading (parsed but not plotted)
    labels_phase = [
        "Vision encode",
        "Prefill",
        "Decode",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    stacks = []
    for i in range(1, 4):
        stacks.append(np.array([rows_by_seq[s][i] / 1000.0 for s in seqs], dtype=float))

    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
    bottom = np.zeros(n)
    for label, vals, color in zip(labels_phase, stacks, colors):
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.72)
        bottom += vals

    totals = bottom
    for i, t in enumerate(totals):
        if t <= 0:
            continue
        ax.text(float(i), t, f"{t:.1f}s", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seqs])
    ax.set_xlim(x[0] - 0.65, x[-1] + 0.65)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Runtime (s)")
    ax.grid(True, axis="y", alpha=0.35)

    fmt_y = mticker.ScalarFormatter(useOffset=False)
    fmt_y.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt_y)

    model = bundle.name
    be = infer_backend_label(bundle)
    subtitle = f"{model}" + (f" · {be}" if be else "")
    ax.set_title(
        f"{subtitle}\nRuntime phase breakdown by sequence length (excluding loading)",
        fontsize=12,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.92,
    )

    out = outfile or (bundle / "runtime_phase_stacked_bar.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        action="append",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing per-seq run subfolders. Repeatable.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Single output path (only valid with exactly one --bundle)",
    )
    args = parser.parse_args()

    if args.output is not None and len(args.bundle) != 1:
        raise SystemExit("--output requires exactly one --bundle")

    for b in args.bundle:
        out = plot_bundle(b, args.output if len(args.bundle) == 1 else None)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
