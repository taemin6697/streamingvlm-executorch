"""Derive memory_usage_summary.txt from android_memory_timeline.csv (foundation + llama runners)."""

from __future__ import annotations

import csv
from pathlib import Path


def parse_memory_usage_summary_txt(path: Path) -> dict[str, float]:
    """Same key convention as foundation_llamacpp/scripts/plot_opencl_ctx_memory_series.py."""
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


def _memory_summary_text_from_rows(rows: list[dict[str, str]]) -> str | None:
    usable: list[tuple[int, float, int, int]] = []
    for idx, row in enumerate(rows):
        raw_mem = (row.get("mem_available_kb") or "").strip()
        if not raw_mem:
            continue
        try:
            mem_available_kb = int(float(raw_mem))
            elapsed_s = float(row.get("elapsed_s") or 0.0)
            pid_alive = int(float(row.get("pid_alive") or 0))
        except ValueError:
            continue
        usable.append((idx, elapsed_s, mem_available_kb, pid_alive))
    if not usable:
        return None

    start_idx, start_elapsed_s, start_kb, _ = usable[0]
    # Llama pre-run baseline: pid_alive==0 and elapsed_s<=0. Exclude postrun rows (pid_alive 0, elapsed>0).
    baseline = [item for item in usable if item[3] == 0 and item[1] <= 0]
    global_min_idx, global_min_elapsed_s, global_min_kb, _ = min(
        usable, key=lambda item: item[2]
    )

    # With baseline: min MemAvail during pid_alive==1 workload. Without: global min MemAvail (full CSV).
    runtime_only = [item for item in usable if item[3] == 1]
    if baseline:
        calc_runtime = runtime_only or usable
        min_idx, min_elapsed_s, min_kb, _ = min(calc_runtime, key=lambda item: item[2])
        memory_usage_method = (
            "baseline_avg_mem_available_kb - runtime_min_mem_available_kb (pid_alive==1)"
        )
    else:
        min_idx, min_elapsed_s, min_kb = global_min_idx, global_min_elapsed_s, global_min_kb
        memory_usage_method = (
            "first_mem_available_kb - global_min_mem_available_kb (no baseline rows)"
        )

    used_from_start_kb = max(start_kb - min_kb, 0)
    if baseline:
        baseline_avg_kb = sum(item[2] for item in baseline) / len(baseline)
        baseline_min_idx, baseline_min_elapsed_s, baseline_min_kb, _ = min(
            baseline, key=lambda item: item[2]
        )
        baseline_max_idx, baseline_max_elapsed_s, baseline_max_kb, _ = max(
            baseline, key=lambda item: item[2]
        )
    else:
        baseline_avg_kb = float(start_kb)
        baseline_min_idx, baseline_min_elapsed_s, baseline_min_kb = start_idx, start_elapsed_s, start_kb
        baseline_max_idx, baseline_max_elapsed_s, baseline_max_kb = start_idx, start_elapsed_s, start_kb

    if baseline:
        used_from_baseline_avg_kb = max(baseline_avg_kb - min_kb, 0.0)
    else:
        used_from_baseline_avg_kb = max(float(start_kb) - float(global_min_kb), 0.0)

    used_from_baseline_max_kb = max(float(baseline_max_kb - min_kb), 0.0)
    return "\n".join(
        [
            f"memory_usage_method: {memory_usage_method}",
            f"baseline_sample_count: {len(baseline)}",
            f"baseline_avg_mem_available_kb: {baseline_avg_kb:.3f}",
            f"baseline_avg_mem_available_mib: {baseline_avg_kb / 1024.0:.3f}",
            f"baseline_min_sample_idx: {baseline_min_idx}",
            f"baseline_min_elapsed_s: {baseline_min_elapsed_s:.3f}",
            f"baseline_min_mem_available_kb: {baseline_min_kb}",
            f"baseline_min_mem_available_mib: {baseline_min_kb / 1024.0:.3f}",
            f"baseline_max_sample_idx: {baseline_max_idx}",
            f"baseline_max_elapsed_s: {baseline_max_elapsed_s:.3f}",
            f"baseline_max_mem_available_kb: {baseline_max_kb}",
            f"baseline_max_mem_available_mib: {baseline_max_kb / 1024.0:.3f}",
            f"start_sample_idx: {start_idx}",
            f"start_elapsed_s: {start_elapsed_s:.3f}",
            f"start_mem_available_kb: {start_kb}",
            f"start_mem_available_mib: {start_kb / 1024.0:.3f}",
            f"runtime_min_sample_idx: {min_idx}",
            f"runtime_min_elapsed_s: {min_elapsed_s:.3f}",
            f"runtime_min_mem_available_kb: {min_kb}",
            f"runtime_min_mem_available_mib: {min_kb / 1024.0:.3f}",
            f"actual_memory_used_from_baseline_avg_kb: {used_from_baseline_avg_kb:.3f}",
            f"actual_memory_used_from_baseline_avg_mib: {used_from_baseline_avg_kb / 1024.0:.3f}",
            f"actual_memory_used_from_baseline_max_kb: {used_from_baseline_max_kb:.3f}",
            f"actual_memory_used_from_baseline_max_mib: {used_from_baseline_max_kb / 1024.0:.3f}",
            f"legacy_start_minus_runtime_min_kb: {used_from_start_kb}",
            f"legacy_start_minus_runtime_min_mib: {used_from_start_kb / 1024.0:.3f}",
            "",
        ]
    )


def write_memory_usage_summary_from_rows(output_dir: Path, rows: list[dict[str, str]]) -> bool:
    text = _memory_summary_text_from_rows(rows)
    if not text:
        return False
    (output_dir / "memory_usage_summary.txt").write_text(text, encoding="utf-8")
    return True


def write_memory_usage_summary_from_timeline_csv(run_dir: Path) -> bool:
    csv_path = run_dir / "android_memory_timeline.csv"
    if not csv_path.is_file():
        return False
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return write_memory_usage_summary_from_rows(run_dir, rows)
