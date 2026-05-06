#!/usr/bin/env python3
"""Plot OpenCL ctx sweep vs memory (default: system + foundation breakdown, categorical ctx)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_memory_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, val = line.split(": ", 1)
        key = key.strip()
        try:
            out[key] = float(val.split()[0])
        except (ValueError, IndexError):
            continue
    return out


def parse_kv_mib(foundation_output: Path) -> float | None:
    text = foundation_output.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"llama_kv_cache:\s+OpenCL KV buffer size\s*=\s*([\d.]+)\s*MiB",
        text,
    )
    if not m:
        return None
    return float(m.group(1))


def parse_common_memory_breakdown(foundation_output: Path) -> dict[str, float] | None:
    """Parse common_memory_breakdown_print GPUOpenCL line (MiB)."""
    text = foundation_output.read_text(encoding="utf-8", errors="replace")
    gpu_pat = re.compile(
        r"\(\s*([\d.]+)\s*=\s*([\d.]+)\s*\+\s*([\d.]+)\s*\+\s*([\d.]+)\s*\)"
    )
    for line in text.splitlines():
        if "common_memory_breakdown_print:" not in line or "GPUOpenCL" not in line:
            continue
        m = gpu_pat.search(line)
        if not m:
            continue
        return {
            "gpu_self_mib": float(m.group(1)),
            "gpu_model_mib": float(m.group(2)),
            "gpu_context_mib": float(m.group(3)),
            "gpu_compute_mib": float(m.group(4)),
        }
    return None


def ctx_from_dir(name: str) -> int | None:
    m = re.search(r"_ctx_(\d+)$", name)
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Directory containing result folders named *opencl_ctx_<N>",
    )
    parser.add_argument(
        "--glob",
        default="*_opencl_ctx_*",
        help="Glob under parent for run folders (default: *_opencl_ctx_*)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <parent>/opencl_memory_vs_ctx.png)",
    )
    parser.add_argument(
        "--plot-style",
        choices=("usage", "dual", "avail-min"),
        default="usage",
        help=(
            "usage: single Y — system usage + GPUOpenCL breakdown "
            "(common_memory_breakdown_print) when foundation_output.txt exists (default). "
            "dual: twin Y — system usage + OpenCL KV buffer from log. "
            "avail-min: single Y — runtime_min_mem_available_mib."
        ),
    )
    args = parser.parse_args()

    parent = args.parent.resolve()
    Row = tuple[int, float, float | None, dict[str, float] | None]
    rows: list[Row] = []
    for folder in sorted(parent.glob(args.glob)):
        if not folder.is_dir():
            continue
        ctx = ctx_from_dir(folder.name)
        if ctx is None:
            continue
        summary = folder / "memory_usage_summary.txt"
        fout = folder / "foundation_output.txt"
        if not summary.exists():
            continue
        sm = parse_memory_summary(summary)
        breakdown = parse_common_memory_breakdown(fout) if fout.exists() else None

        if args.plot_style == "avail-min":
            v = sm.get("runtime_min_mem_available_mib")
            if v is None:
                continue
            rows.append((ctx, v, None, breakdown))
        elif args.plot_style == "dual":
            if not fout.exists():
                continue
            drop = sm.get("actual_memory_used_from_baseline_avg_mib")
            kv = parse_kv_mib(fout)
            if drop is None or kv is None:
                continue
            rows.append((ctx, drop, kv, breakdown))
        else:
            drop = sm.get("actual_memory_used_from_baseline_avg_mib")
            if drop is None:
                continue
            rows.append((ctx, drop, None, breakdown))

    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 valid runs under {parent}; found {len(rows)}")

    rows.sort(key=lambda r: r[0])
    ctxs = [r[0] for r in rows]

    import math

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    # Categorical X: discrete ctx bucket labels, even spacing (not numeric scale).
    x_idx = np.arange(len(ctxs), dtype=float)
    x_labels = [str(c) for c in ctxs]

    def _plain_axes(ax: plt.Axes, *, include_x: bool = True, include_y: bool = True) -> None:
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        if include_x:
            ax.xaxis.set_major_formatter(fmt)
        if include_y:
            ax.yaxis.set_major_formatter(fmt)

    def _apply_categorical_x(ax: plt.Axes) -> None:
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(x_idx[0] - 0.5, x_idx[-1] + 0.5)
        ax.set_xlabel("Context size (tokens, categorical)")

    fig, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")

    if args.plot_style == "avail-min":
        mem_mib = [r[1] for r in rows]
        ax1.set_ylabel("Memory usage (MiB)")
        ax1.plot(
            x_idx,
            mem_mib,
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=8,
            label="MemAvailable (min during run)",
        )
        _apply_categorical_x(ax1)
        ax1.legend(loc="best")
        ax1.set_title(f"{parent.name}: memory usage vs ctx (OpenCL)")
        ax1.grid(True, alpha=0.35)
        _plain_axes(ax1, include_x=False)
    elif args.plot_style == "dual":
        drop_mib = [r[1] for r in rows]
        kv_mib = [r[2] for r in rows]
        assert all(v is not None for v in kv_mib)
        ax1.set_ylabel("Memory usage (MiB)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        (ln1,) = ax1.plot(
            x_idx,
            drop_mib,
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=8,
            label="System memory usage",
        )
        ax1.grid(True, alpha=0.35)
        ax2 = ax1.twinx()
        ax2.set_ylabel("KV memory (MiB)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        (ln2,) = ax2.plot(
            x_idx,
            kv_mib,
            "s-",
            color="tab:orange",
            linewidth=2,
            markersize=7,
            label="OpenCL KV cache",
        )
        _apply_categorical_x(ax1)
        ax1.legend(handles=[ln1, ln2], loc="upper left")
        ax1.set_title(f"{parent.name}: memory usage vs ctx (OpenCL, dual)")
        _plain_axes(ax1, include_x=False)
        _plain_axes(ax2, include_x=False)
    else:
        usage_mib = [r[1] for r in rows]

        def _series(key: str) -> list[float]:
            return [
                float(r[3][key]) if r[3] and key in r[3] else float("nan")
                for r in rows
            ]

        ax1.set_ylabel("Memory usage (MiB)")
        ax1.plot(
            x_idx,
            usage_mib,
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=8,
            label="System memory usage",
        )
        gpu_self = _series("gpu_self_mib")
        if not all(math.isnan(v) for v in gpu_self):
            ax1.plot(
                x_idx,
                gpu_self,
                "^-",
                color="tab:orange",
                linewidth=2,
                markersize=7,
                label="GPUOpenCL self (breakdown)",
            )
        _apply_categorical_x(ax1)
        ax1.legend(loc="best")
        ax1.set_title(f"{parent.name}: memory usage vs ctx (OpenCL)")
        ax1.grid(True, alpha=0.35)
        _plain_axes(ax1, include_x=False)

    out = args.output or (parent / "opencl_memory_vs_ctx.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
