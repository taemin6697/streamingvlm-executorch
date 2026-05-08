#!/usr/bin/env python3
"""After KV folder layout like InternVL3-1B-Q8_kv{8,16}/:
- Per-model: kv16 vs kv8 memory vs ctx on one chart.
- Per-KV: all *_kv8 (or *_kv16) bundles on one chart.

Uses actual_memory_used_from_baseline_avg_mib from memory_usage_summary.txt."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from my_research.foundation.host.android_timeline_memory_summary import (
    parse_memory_usage_summary_txt,
    write_memory_usage_summary_from_timeline_csv,
)


def ctx_from_dir(name: str) -> int | None:
    m = re.search(r"_ctx_(\d+)", name)
    return int(m.group(1)) if m else None


def collect_ctx_memory(bundle_dir: Path, *, refresh: bool) -> dict[int, float]:
    """Map context size -> system memory usage MiB for one bundle (e.g. .../InternVL3-1B-Q8_kv16)."""
    out: dict[int, float] = {}
    for folder in sorted(bundle_dir.glob("*_opencl_ctx_*")):
        if not folder.is_dir():
            continue
        ctx = ctx_from_dir(folder.name)
        if ctx is None:
            continue
        summary = folder / "memory_usage_summary.txt"
        tl = folder / "android_memory_timeline.csv"
        if refresh and tl.is_file():
            write_memory_usage_summary_from_timeline_csv(folder)
        if not summary.is_file():
            continue
        sm = parse_memory_usage_summary_txt(summary)
        mib = sm.get("actual_memory_used_from_baseline_avg_mib")
        if mib is None:
            continue
        out[ctx] = float(mib)
    return out


def _plot_lines(
    title: str,
    out_path: Path,
    xs: list[int],
    series: list[tuple[str, dict[int, float]]],
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    x_idx = np.arange(len(xs), dtype=float)
    labels = [str(c) for c in xs]

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    cmap = plt.cm.tab10.colors
    for i, (label, data) in enumerate(series):
        ys = [data.get(x, float("nan")) for x in xs]
        ax.plot(
            x_idx,
            ys,
            "o-",
            color=cmap[i % len(cmap)],
            linewidth=2,
            markersize=7,
            label=label,
        )
    fmt = mticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels)
    ax.set_xlim(x_idx[0] - 0.5, x_idx[-1] + 0.5)
    ax.set_xlabel("Context size (tokens, categorical)")
    ax.set_ylabel("Memory usage (MiB) — baseline avg MemAvail − min MemAvail during run")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_model_kv_pair(
    log_root: Path,
    model_stem: str,
    *,
    refresh: bool,
) -> None:
    """model_stem e.g. InternVL3-1B-Q8 (folders InternVL3-1B-Q8_kv8 / _kv16)."""
    d8 = log_root / f"{model_stem}_kv8"
    d16 = log_root / f"{model_stem}_kv16"
    if not d16.is_dir() or not d8.is_dir():
        raise SystemExit(f"Need both {d16} and {d8} as directories")
    m8 = collect_ctx_memory(d8, refresh=refresh)
    m16 = collect_ctx_memory(d16, refresh=refresh)
    if len(m8) < 1 or len(m16) < 1:
        raise SystemExit(f"Insufficient data for {model_stem}: kv8={len(m8)} kv16={len(m16)}")
    xs = sorted(set(m8) | set(m16))
    out = log_root / f"{model_stem}_kv16_vs_kv8_memory_by_ctx.png"
    _plot_lines(
        f"{model_stem}: OpenCL KV8 vs KV16 memory vs ctx",
        out,
        xs,
        [("KV8", m8), ("KV16", m16)],
    )


def plot_one_kv_all_models(log_root: Path, kv: int, *, refresh: bool) -> None:
    suffix = f"_kv{kv}"
    bundles = sorted(p for p in log_root.glob(f"*{suffix}") if p.is_dir())
    bundles = [
        b
        for b in bundles
        if not b.name.startswith("old")
        and list(b.glob("*_opencl_ctx_*"))
    ]
    if len(bundles) < 1:
        raise SystemExit(f"No *{suffix} bundle dirs under {log_root}")

    series: list[tuple[str, dict[int, float]]] = []
    for b in bundles:
        label = b.name[: -len(suffix)]
        data = collect_ctx_memory(b, refresh=refresh)
        if len(data) >= 1:
            series.append((label, data))
    if len(series) < 1:
        raise SystemExit(f"No data for KV{kv}")

    all_ctx: set[int] = set()
    for _, data in series:
        all_ctx |= set(data.keys())
    xs = sorted(all_ctx)
    out = log_root / f"all_models_kv{kv}_memory_by_ctx.png"
    _plot_lines(
        f"OpenCL KV{kv}: memory vs ctx (all bundles under log root)",
        out,
        xs,
        series,
    )


def default_model_stems() -> list[str]:
    return [
        "InternVL3-1B-Q8",
        "InternVL3-2B-Q8",
        "InternVL3-8B-Q4_K_M",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-root",
        type=Path,
        default=Path("my_research/foundation_llamacpp/results/log"),
        help="Directory containing InternVL3-*-Q8_kv8 style folders",
    )
    ap.add_argument(
        "--refresh-summaries",
        action="store_true",
        help="Rewrite memory_usage_summary.txt from android_memory_timeline.csv per run",
    )
    ap.add_argument(
        "--model-stems",
        nargs="*",
        default=None,
        help="Stems for kv8/kv16 pair plots (default: 1B-Q8, 2B-Q8, 8B-Q4_K_M)",
    )
    ap.add_argument(
        "--skip-pairs",
        action="store_true",
        help="Only emit all_models_kv{8,16} charts",
    )
    ap.add_argument(
        "--skip-by-kv",
        action="store_true",
        help="Only emit per-model kv16 vs kv8 charts",
    )
    args = ap.parse_args()
    log_root = args.log_root.resolve()
    stems = args.model_stems if args.model_stems else default_model_stems()

    if not args.skip_pairs:
        for stem in stems:
            d8 = log_root / f"{stem}_kv8"
            d16 = log_root / f"{stem}_kv16"
            if not d8.is_dir() or not d16.is_dir():
                print(f"Skip pair {stem}: missing {d8.name} or {d16.name}")
                continue
            plot_model_kv_pair(log_root, stem, refresh=args.refresh_summaries)

    if not args.skip_by_kv:
        for kv in (8, 16):
            try:
                plot_one_kv_all_models(log_root, kv, refresh=args.refresh_summaries)
            except SystemExit as e:
                print(f"KV{kv} combined plot: {e}")


if __name__ == "__main__":
    main()
