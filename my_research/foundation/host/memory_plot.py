from __future__ import annotations

import csv
import math
from pathlib import Path


def _float_or_nan(value: str | None) -> float:
    value = (value or "").strip()
    return float(value) if value else math.nan


def _generate_phase_duration_bar(
    out_dir: Path,
    proc_rows: list[tuple[str, float, float, dict[str, str]]],
    phase_colors: dict[str, str],
) -> Path | None:
    if not proc_rows:
        return None

    output_png = out_dir / "phase_duration_stacked_bar.png"
    has_decode_summary = any(row_type == "Decode" for row_type, _, _, _ in proc_rows)
    phase_order = [
        "L",
        "L_VisionLoad",
        "L_EmbeddingLoad",
        "L_DecoderLoad",
        "V_Encode",
        "EmbeddingAndMerging",
        "T_Prefill",
        "Decode",
    ]
    durations: dict[str, float] = {}

    for row_type, start, end, _ in proc_rows:
        if row_type == "D" and has_decode_summary:
            continue
        normalized = "Decode" if row_type == "D" else row_type
        duration = max(end - start, 0.0)
        durations[normalized] = durations.get(normalized, 0.0) + duration

    ordered_phases = [phase for phase in phase_order if durations.get(phase, 0.0) > 0]
    ordered_phases.extend(
        phase
        for phase in sorted(durations)
        if phase not in ordered_phases and durations[phase] > 0
    )
    if not ordered_phases:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - best effort plotting
        print(f"[foundation] warning: failed to create phase bar plot: {exc}")
        return None

    total = sum(durations[phase] for phase in ordered_phases)
    fig, ax = plt.subplots(figsize=(5.5, 9), dpi=160)
    bottom = 0.0
    for phase in ordered_phases:
        duration = durations[phase]
        color = phase_colors.get(phase, "#636e72")
        ax.bar(["total"], [duration], bottom=bottom, color=color, edgecolor="white")
        if total > 0 and duration / total >= 0.035:
            ax.text(
                0,
                bottom + duration / 2.0,
                f"{phase}\n{duration:.3f}s",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        bottom += duration

    ax.set_title(f"Total Runtime Breakdown ({total:.3f}s)")
    ax.set_ylabel("Elapsed Time (s)")
    ax.set_ylim(0, max(total, 1e-9))
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=phase_colors.get(phase, "#636e72"))
        for phase in ordered_phases
    ]
    labels = [
        f"{phase}: {durations[phase]:.3f}s ({durations[phase] / total * 100:.1f}%)"
        for phase in ordered_phases
    ]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return output_png


def generate_memory_timeline_plot(out_dir: Path) -> Path | None:
    timeline_csv = out_dir / "android_memory_timeline.csv"
    proc_csv = out_dir / "foundation_proc.csv"
    output_png = out_dir / "memory_timeline_plot.png"

    if not timeline_csv.exists() or not proc_csv.exists():
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - best effort plotting
        print(f"[foundation] warning: failed to create memory plot: {exc}")
        return None

    timeline_rows: list[dict[str, float | str]] = []
    with timeline_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timeline_rows.append(
                {
                    "elapsed_s": _float_or_nan(row.get("elapsed_s")),
                    "phase": (row.get("phase") or "").strip(),
                    "mem_available_mb": _float_or_nan(row.get("mem_available_kb"))
                    / 1024.0,
                    "smaps_rss_mb": _float_or_nan(row.get("smaps_rss_kb"))
                    / 1024.0,
                    "self_rss_mb": _float_or_nan(row.get("self_rss_kb")) / 1024.0,
                    "kv_physical_committed_mb": _float_or_nan(
                        row.get("kv_physical_committed_kb")
                    )
                    / 1024.0,
                    "dma_heap_pool_mb": _float_or_nan(row.get("dma_heap_pool_kb"))
                    / 1024.0,
                }
            )

    proc_rows: list[tuple[str, float, float, dict[str, str]]] = []
    with proc_csv.open(encoding="utf-8") as f:
        header = None
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("row_type,"):
                header = raw.split(",")
                continue
            parts = raw.split(",")
            if len(parts) < 3:
                continue
            extra = {}
            if header and len(header) == len(parts):
                for i, h in enumerate(header):
                    extra[h.strip()] = parts[i].strip() if i < len(parts) else ""
            proc_rows.append((parts[0], float(parts[1]), float(parts[2]), extra))

    if not timeline_rows:
        return None

    phase_colors = {
        "L": "#6c5ce7",
        "L_VisionLoad": "#8e44ad",
        "L_EmbeddingLoad": "#9b59b6",
        "L_DecoderLoad": "#a569bd",
        "V_Encode": "#00b894",
        "EmbeddingAndMerging": "#fdcb6e",
        "T_Prefill": "#e17055",
        "Decode": "#d63031",
        "D": "#ff7675",
    }

    fig, ax = plt.subplots(figsize=(19, 8), dpi=160)

    xs = [float(r["elapsed_s"]) for r in timeline_rows]
    mem_available = [float(r["mem_available_mb"]) for r in timeline_rows]
    smaps_rss = [float(r["smaps_rss_mb"]) for r in timeline_rows]
    self_rss = [float(r["self_rss_mb"]) for r in timeline_rows]
    kv_physical_committed = [
        float(r["kv_physical_committed_mb"]) for r in timeline_rows
    ]
    dma_heap_pool = [float(r["dma_heap_pool_mb"]) for r in timeline_rows]

    ax.plot(
        xs,
        mem_available,
        color="#0984e3",
        linewidth=2.4,
        marker="o",
        markersize=2.6,
        label="MemAvailable (MB)",
    )
    if any(not math.isnan(v) for v in smaps_rss):
        ax.plot(
            xs,
            smaps_rss,
            color="#b2bec3",
            linewidth=1.1,
            alpha=0.9,
            label="smaps RSS (MB)",
        )
    if any(not math.isnan(v) for v in self_rss):
        ax.plot(
            xs,
            self_rss,
            color="#636e72",
            linewidth=1.1,
            alpha=0.75,
            label="self RSS (MB)",
        )
    if any(not math.isnan(v) and v > 0 for v in kv_physical_committed):
        ax.plot(
            xs,
            kv_physical_committed,
            color="#00cec9",
            linewidth=1.4,
            alpha=0.9,
            label="KV physical committed (MB)",
        )
    if any(not math.isnan(v) and v > 0 for v in dma_heap_pool):
        ax.plot(
            xs,
            dma_heap_pool,
            color="#e67e22",
            linewidth=1.2,
            alpha=0.85,
            label="DmaHeapPool (MB)",
        )

    label_y_axes = 0.045
    phase_count: dict[str, int] = {}
    for row_type, start, end, _ in proc_rows:
        if row_type not in phase_colors:
            continue
        phase_count[row_type] = phase_count.get(row_type, 0) + 1
        label = f"{row_type}{phase_count[row_type]}" if row_type == "V_Encode" else row_type
        color = phase_colors[row_type]
        ax.axvline(start, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        ax.axvline(end, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        ax.axvspan(start, end, color=color, alpha=0.06)
        ax.text(
            (start + end) / 2.0,
            label_y_axes,
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

    ax.axvline(0.0, color="#2d3436", linestyle="-", linewidth=1.2, alpha=0.9)
    ax.set_title("Android Memory Timeline vs Foundation Events")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Memory (MB)")
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
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    phase_bar_path = _generate_phase_duration_bar(out_dir, proc_rows, phase_colors)
    if phase_bar_path:
        print(f"[foundation] phase duration plot: {phase_bar_path}")

    return output_png
