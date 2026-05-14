#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from my_research.foundation.host.android_timeline_memory_summary import (
    write_memory_usage_summary_from_rows,
)
from my_research.foundation_llamacpp.runner.config import (
    PreparedMedia,
    backend_mode_from_processor,
    media_mode_from_args,
)
from my_research.foundation_llamacpp.runner.media import normalize_stream_mode
from my_research.foundation_llamacpp.runner.media import prepare_media
from my_research.foundation_llamacpp.runner.media import prepare_warmup_image
from my_research.foundation_llamacpp.runner.artifacts import (
    HYBRID_PULL_ARTIFACTS,
    HYBRID_STREAMING_PULL_ARTIFACTS,
    STANDALONE_PULL_ARTIFACTS,
)
from my_research.foundation_llamacpp.runner.remote import (
    adb_cmd,
    pull_if_exists,
    push,
    remote_exists,
    run,
    shell_join,
)

WORKSPACE = Path(__file__).resolve().parents[3]
FOUNDATION_LLAMA = Path(__file__).resolve().parents[1]
DEFAULT_WARMUP_IMAGE = FOUNDATION_LLAMA / "sample_images" / "golden_gate_bridge_448.jpg"


def _standalone_completion_mode(args: argparse.Namespace) -> bool:
    """--processor cpu|gpu and no media → text-only llama-completion (no mmproj/vision)."""
    return (
        args.processor in ("cpu", "gpu")
        and getattr(args, "image", None) is None
        and getattr(args, "video", None) is None
        and getattr(args, "streaming_video", None) is None
    )


def _run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return run(cmd, check=check, capture_output=capture_output)


def _adb(serial: str | None) -> list[str]:
    return adb_cmd(serial)


def _prepare_media(args: argparse.Namespace, work_dir: Path) -> PreparedMedia:
    return prepare_media(args, work_dir)


def _load_manifest(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    resolved = dict(data)
    paths = {}
    for key, value in data.get("paths", {}).items():
        if not value:
            continue
        path = Path(value)
        paths[key] = str(path if path.is_absolute() else (base / path).resolve())
    resolved["paths"] = paths
    return resolved


def _parse_json_list_arg(value: str, *, name: str) -> list[object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be a JSON list, e.g. '[5.0, 10.0]': {exc}") from exc
    if not isinstance(parsed, list):
        raise SystemExit(f"{name} must be a JSON list.")
    return parsed


def _parse_streaming_prompt_events(times_text: str | None, prompts_text: str) -> list[dict[str, object]]:
    if times_text is None:
        raise SystemExit("--streaming-video requires --time '[...]'.")
    times_raw = _parse_json_list_arg(times_text, name="--time")
    prompts_raw = _parse_json_list_arg(prompts_text, name="--prompt")
    if len(times_raw) != len(prompts_raw):
        raise SystemExit("--time and --prompt must have the same length in --streaming-video mode.")
    events: list[dict[str, object]] = []
    for idx, (time_value, prompt_value) in enumerate(zip(times_raw, prompts_raw)):
        if not isinstance(time_value, (int, float)):
            raise SystemExit(f"--time[{idx}] must be a number.")
        if float(time_value) < 0:
            raise SystemExit(f"--time[{idx}] must be non-negative.")
        if not isinstance(prompt_value, str):
            raise SystemExit(f"--prompt[{idx}] must be a string.")
        events.append({"time": float(time_value), "prompt": prompt_value})
    return sorted(events, key=lambda item: float(item["time"]))


def _push(adb: list[str], local: Path, remote_dir: str) -> None:
    push(adb, local, remote_dir)


def _remote_exists(adb: list[str], remote_path: str) -> bool:
    return remote_exists(adb, remote_path)


def _remote_read_text(adb: list[str], remote_path: str) -> str | None:
    result = subprocess.run(
        adb + ["shell", f"cat {shlex.quote(remote_path)}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _shell_join(parts: list[str]) -> str:
    return shell_join(parts)


def _push_model_cached(adb: list[str], local: Path, remote_dir: str, *, force: bool) -> None:
    remote = f"{remote_dir}/{local.name}"
    if not force and _remote_exists(adb, remote):
        print(f"[push-cache] keep remote model: {remote}")
        return
    _push(adb, local, remote_dir)


def _push_qnn_libs(adb: list[str], remote_dir: str, qnn_sdk: Path, executorch_build_dir: Path, soc_model: str) -> None:
    qnn_lib_dir = qnn_sdk / "lib" / "aarch64-android"
    if qnn_lib_dir.exists():
        for lib in sorted(qnn_lib_dir.glob("libQnn*.so")):
            _push(adb, lib, remote_dir)
    arch = {
        "SM8750": "79",
        "SM8650": "75",
        "SM8550": "73",
        "SM8450": "69",
        "SM8350": "68",
    }.get(soc_model, "73")
    skel = qnn_sdk / "lib" / f"hexagon-v{arch}" / "unsigned" / f"libQnnHtpV{arch}Skel.so"
    if skel.exists():
        _push(adb, skel, remote_dir)
    backend = executorch_build_dir / "backends" / "qualcomm" / "libqnn_executorch_backend.so"
    if backend.exists():
        _push(adb, backend, remote_dir)


def _push_llama_runtime(
    adb: list[str],
    remote_dir: str,
    llama_build_dir: Path,
    opencl_lib: Path | None,
    push_opencl_loader: bool,
) -> None:
    for subdir in (llama_build_dir / "bin", llama_build_dir / "lib"):
        if subdir.exists():
            for path in sorted(subdir.iterdir()):
                if path.name == "libOpenCL.so" and not push_opencl_loader:
                    continue
                if path.is_file() and (
                    path.suffix == ".so"
                    or path.name in {
                        "hybrid_decode",
                        "hybrid_streaming_decode",
                        "opencl_streaming_decode",
                        "llama-mtmd-cli",
                        "opencl_phase_mtmd",
                        "llama-completion",
                    }
                ):
                    _push(adb, path, remote_dir)
    for name in ("hybrid_decode", "hybrid_streaming_decode", "opencl_streaming_decode", "opencl_phase_mtmd", "llama-completion"):
        path = llama_build_dir / name
        if path.exists():
            _push(adb, path, remote_dir)
    runtime_patterns = ["libc++_shared.so"]
    if push_opencl_loader:
        runtime_patterns.append("libOpenCL.so")
    for pattern in runtime_patterns:
        for path in sorted(llama_build_dir.rglob(pattern)):
            if path.is_file():
                _push(adb, path, remote_dir)
    if push_opencl_loader and opencl_lib and opencl_lib.exists():
        _push(adb, opencl_lib, remote_dir)


def _pull_if_exists(adb: list[str], remote: str, local: Path) -> None:
    pull_if_exists(adb, remote, local)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_memory_usage_txt(output_dir: Path, rows: list[dict[str, str]]) -> None:
    write_memory_usage_summary_from_rows(output_dir, rows)


def _match_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else 0.0


def _match_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


def _parse_log_summary(log_text: str) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key, pattern in [
        ("image_slice_encoded_ms", r"image slice encoded in ([0-9.]+) ms"),
        ("image_decoded_ms", r"image decoded(?: \(batch \d+/\d+\))? in ([0-9.]+) ms"),
        ("model_buffer_mib", r"model buffer size =\s+([0-9.]+) MiB"),
        ("context_buffer_mib", r"KV buffer size =\s+([0-9.]+) MiB"),
        ("compute_buffer_mib", r"compute buffer size =\s+([0-9.]+) MiB"),
        ("load_time_ms", r"llama_perf_context_print:\s+load time =\s+([0-9.]+) ms"),
        ("prompt_eval_time_ms", r"llama_perf_context_print:\s+prompt eval time =\s+([0-9.]+) ms"),
        ("decode_eval_time_ms", r"llama_perf_context_print:\s+eval time =\s+([0-9.]+) ms"),
        ("total_time_ms", r"llama_perf_context_print:\s+total time =\s+([0-9.]+) ms"),
    ]:
        matches = re.findall(pattern, log_text)
        if matches:
            summary[key] = float(matches[-1])
    prompt_tokens = _match_int(log_text, r"prompt eval time =\s+[0-9.]+ ms /\s+([0-9]+) tokens")
    decode_runs = _match_int(log_text, r"eval time =\s+[0-9.]+ ms /\s+([0-9]+) runs")
    if prompt_tokens:
        summary["prompt_eval_tokens"] = prompt_tokens
    if decode_runs:
        summary["decode_eval_runs"] = decode_runs
    if summary.get("prompt_eval_time_ms") and prompt_tokens:
        summary["prompt_eval_tok_s"] = prompt_tokens / float(summary["prompt_eval_time_ms"]) * 1000.0
    if summary.get("decode_eval_time_ms") and decode_runs:
        summary["decode_eval_tok_s"] = decode_runs / float(summary["decode_eval_time_ms"]) * 1000.0
    return summary


def _ansi_sub(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _extract_generated_text_text_only(log_text: str) -> str:
    """Best-effort completion output (llama-completion) before llama_perf_context_print."""
    body = _ansi_sub(log_text)
    stop = body.find("llama_perf_context_print:")
    if stop >= 0:
        body = body[:stop]
    body = re.sub(r"(?mi)\s*\[end of text\]\s*\Z", "", body)
    lines_out: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(("ggml_", "llama_", "clip_", "load_backend", "broadcast_", "check_tensor", "alloc_tensor")):
            continue
        if re.match(r"^(main|eval|graph|prompt|encode|decode|sampler|tensor|model|kv|opencl|cpu|gpu|device)\s*:", s, re.I):
            continue
        if s.startswith(("[", "warn", "info", "deprecat", "fatal", "error")):
            continue
        lines_out.append(line.rstrip())
    return "\n".join(lines_out).strip()


def _extract_generated_text_from_log(log_text: str) -> str:
    lines = log_text.splitlines()
    start_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith("image decoded"):
            start_idx = idx + 1
    if start_idx < 0:
        return ""
    out_lines: list[str] = []
    for line in lines[start_idx:]:
        if line.startswith("llama_perf_context_print:"):
            break
        if not line.strip() and not out_lines:
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def _write_fallback_token_io_txt(
    result_dir: Path,
    prompt: str,
    log_text: str,
    *,
    text_only: bool = False,
    image_tokens: int = 256,
) -> None:
    del image_tokens
    token_io = result_dir / "foundation_token_io.txt"
    if token_io.exists():
        return
    if text_only:
        generated = _extract_generated_text_text_only(log_text)
        text = f"<|im_start|>user\n{prompt}\n<|im_start|>assistant\n{generated}\n"
    else:
        generated = _extract_generated_text_from_log(log_text)
        text = f"User: {prompt}\nAssistant: {generated}\n"
    token_io.write_text(text, encoding="utf-8")


def _phase_float(row: dict[str, str], key: str) -> float:
    value = (row.get(key) or "").strip()
    return float(value) if value else 0.0


def _read_phase_rows(path: Path, *, offset_s: float = 0.0) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        header: list[str] | None = None
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("row_type,"):
                header = raw.split(",")
                continue
            if header is None:
                continue
            parts = raw.split(",")
            if len(parts) < len(header):
                parts.extend([""] * (len(header) - len(parts)))
            row = {key: parts[idx] for idx, key in enumerate(header)}
            if row.get("row_type") == "L_DecoderInit":
                row["row_type"] = "L_DecoderRuntimeInit"
            row["elapsed_s_start"] = f"{_phase_float(row, 'elapsed_s_start') + offset_s:.6f}"
            row["elapsed_s_end"] = f"{_phase_float(row, 'elapsed_s_end') + offset_s:.6f}"
            rows.append(row)
    return rows


def _phase_clock_origin_ms(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            match = re.match(r"#\s*clock_origin_ms:\s*(-?\d+)", line)
            if match:
                return int(match.group(1))
    return None


def _write_phase_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "row_type",
        "elapsed_s_start",
        "elapsed_s_end",
        "rss_kb_start",
        "rss_kb_end",
        "col_a_ms",
        "col_b_ms",
        "total_ms",
        "kv_pos",
        "kv_total",
        "kv_used_pct",
        "kv_estimated_used_kb",
        "kv_total_kb",
        "kv_physical_committed_kb",
        "token_idx",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "row_type": "# L_VisionLoad: QNN vision module load  ImageLoad: input image/tensor load  "
            "V_Encode: vision tower encode (QNN hybrid or llama.cpp OpenCL)  EmbeddingFileWrite: .svlmemb write  "
            "L_DecoderRuntimeInit: llama.cpp args/OpenCL runtime init  ExternalEmbeddingRead: .svlmemb read  "
            "L_DecoderLoad: llama.cpp model/mmproj load  "
            "LayoutTokenize: mtmd layout  Prefill: combined text/image prefill split to exclude DynamicKVGrow  "
            "ImagePrefill: image embedding prefill  T_Prefill: text prompt prefill  "
            "DynamicKVGrow: KV cache capacity grow  D: one generated-token decode"
        })
        writer.writerows(rows)


def _parse_stdout_uptime_s(line: str) -> float | None:
    match = re.search(r"\b[IE]\s+(\d+):(\d+):(\d+)\.(\d+)", line)
    if not match:
        return None
    hh, mm, ss, frac = match.groups()
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + float(f"0.{frac}")


def _dynamic_kv_rows_from_stdout(stdout_path: Path, *, clock_origin_ms: int | None = None) -> list[dict[str, str]]:
    if not stdout_path.exists():
        return []
    lines = stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()

    load_origin: float | None = None
    for line in lines:
        if "load_tensors:" in line:
            load_origin = _parse_stdout_uptime_s(line)
            if load_origin is not None:
                break
    if load_origin is None:
        load_origin = 0.0

    rows: list[dict[str, str]] = []
    for idx, line in enumerate(lines):
        grow = re.search(r"grow_to: growing dynamic KV cache: old = (\d+), new = (\d+), logical = (\d+)(?:, clock_ms = (-?\d+))?", line)
        if not grow:
            continue

        old_cells, new_cells, logical_cells = (int(grow.group(1)), int(grow.group(2)), int(grow.group(3)))
        start_clock_ms = int(grow.group(4)) if grow.group(4) is not None else None
        duration_ms = 0.0
        new_mib = 0.0
        end_clock_ms: int | None = None
        retry_start_clock_ms: int | None = None
        retry_end_clock_ms: int | None = None
        for follow in lines[idx + 1 : idx + 32]:
            size = re.search(r"reset_capacity: size =\s*([0-9.]+) MiB", follow)
            if size:
                new_mib = float(size.group(1))
            done = re.search(r"dynamic KV grow completed in\s*([0-9.]+) ms(?:, clock_ms = (-?\d+))?", follow)
            if done:
                duration_ms = float(done.group(1))
                end_clock_ms = int(done.group(2)) if done.group(2) is not None else None
            retry_window = re.search(
                r"dynamic KV grow retry window: old = (\d+), new = (\d+), logical = (\d+), clock_start_ms = (-?\d+), clock_end_ms = (-?\d+)",
                follow,
            )
            if retry_window:
                retry_start_clock_ms = int(retry_window.group(4))
                retry_end_clock_ms = int(retry_window.group(5))
                break

        old_mib = new_mib * old_cells / new_cells if new_cells and new_mib else 0.0
        if clock_origin_ms is not None and start_clock_ms is not None:
            start = ((retry_start_clock_ms if retry_start_clock_ms is not None else start_clock_ms) - clock_origin_ms) / 1000.0
            if retry_end_clock_ms is not None:
                end = (retry_end_clock_ms - clock_origin_ms) / 1000.0
                duration_ms = (end - start) * 1000.0
            else:
                end = (end_clock_ms - clock_origin_ms) / 1000.0 if end_clock_ms is not None else start + duration_ms / 1000.0
        else:
            start_abs = _parse_stdout_uptime_s(line)
            if start_abs is None:
                for prev in reversed(lines[max(0, idx - 8) : idx]):
                    start_abs = _parse_stdout_uptime_s(prev)
                    if start_abs is not None:
                        break
            if start_abs is None:
                continue
            start = start_abs - load_origin
            end = start + duration_ms / 1000.0
        detail = f"{old_cells}->{new_cells}/{logical_cells} cells; {old_mib:.2f}->{new_mib:.2f} MiB"
        rows.append({
            "row_type": "DynamicKVGrow",
            "elapsed_s_start": f"{start:.6f}",
            "elapsed_s_end": f"{end:.6f}",
            "rss_kb_start": "",
            "rss_kb_end": "",
            "col_a_ms": f"{duration_ms:.0f}",
            "col_b_ms": "",
            "total_ms": f"{duration_ms:.0f}",
            "kv_pos": str(old_cells),
            "kv_total": str(new_cells),
            "kv_used_pct": "",
            "kv_estimated_used_kb": str(int(round(old_mib * 1024))) if old_mib else "",
            "kv_total_kb": str(int(round(new_mib * 1024))) if new_mib else "",
            "kv_physical_committed_kb": str(int(round(new_mib * 1024))) if new_mib else "",
            "token_idx": detail,
        })
    return rows


def _split_phase_around_intervals(row: dict[str, str], intervals: list[tuple[float, float]]) -> list[dict[str, str]]:
    start = _phase_float(row, "elapsed_s_start")
    end = _phase_float(row, "elapsed_s_end")
    if end <= start:
        return [row]

    segments = [(start, end)]
    for cut_start, cut_end in intervals:
        if cut_end <= cut_start:
            continue
        next_segments: list[tuple[float, float]] = []
        for seg_start, seg_end in segments:
            if cut_end <= seg_start or cut_start >= seg_end:
                next_segments.append((seg_start, seg_end))
                continue
            if seg_start < cut_start:
                next_segments.append((seg_start, cut_start))
            if cut_end < seg_end:
                next_segments.append((cut_end, seg_end))
        segments = next_segments

    if not segments:
        return []

    out: list[dict[str, str]] = []
    for seg_start, seg_end in segments:
        new_row = dict(row)
        duration_ms = max((seg_end - seg_start) * 1000.0, 0.0)
        new_row["elapsed_s_start"] = f"{seg_start:.6f}"
        new_row["elapsed_s_end"] = f"{seg_end:.6f}"
        new_row["col_a_ms"] = f"{duration_ms:.0f}"
        new_row["total_ms"] = f"{duration_ms:.0f}"
        detail = new_row.get("token_idx", "")
        suffix = "split_excluding_dynamic_kv_grow"
        new_row["token_idx"] = f"{detail}; {suffix}" if detail else suffix
        out.append(new_row)
    return out


def _clip_phase_retry_start(row: dict[str, str], intervals: list[tuple[float, float]]) -> dict[str, str]:
    start = _phase_float(row, "elapsed_s_start")
    end = _phase_float(row, "elapsed_s_end")
    if end <= start:
        return row

    new_start = start
    for cut_start, cut_end in intervals:
        if cut_start < new_start < cut_end < end:
            new_start = cut_end
        elif new_start <= cut_start < cut_end < end and row.get("row_type") in {"ImagePrefill", "T_Prefill", "Mmproj"}:
            new_start = cut_end

    if new_start <= start:
        return row

    new_row = dict(row)
    duration_ms = max((end - new_start) * 1000.0, 0.0)
    new_row["elapsed_s_start"] = f"{new_start:.6f}"
    new_row["col_a_ms"] = f"{duration_ms:.0f}"
    new_row["total_ms"] = f"{duration_ms:.0f}"
    detail = new_row.get("token_idx", "")
    suffix = "retry_start_after_dynamic_kv_grow"
    new_row["token_idx"] = f"{detail}; {suffix}" if detail else suffix
    return new_row


def _separate_dynamic_kv_grow_overlaps(phase_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grow_rows = [row for row in phase_rows if row.get("row_type") == "DynamicKVGrow"]
    if not grow_rows:
        return phase_rows

    intervals = sorted(
        (_phase_float(row, "elapsed_s_start"), _phase_float(row, "elapsed_s_end"))
        for row in grow_rows
    )
    # Split the aggregate Prefill wrapper, and clip fine-grained rows that
    # partially overlap grow so the visible duration starts from the retry.
    split_phase_names = {"Prefill"}
    retry_clip_phase_names = {"ImagePrefill", "T_Prefill", "Mmproj"}
    separated: list[dict[str, str]] = []
    for row in phase_rows:
        if row.get("row_type") in split_phase_names:
            separated.extend(_split_phase_around_intervals(row, intervals))
        elif row.get("row_type") in retry_clip_phase_names:
            separated.append(_clip_phase_retry_start(row, intervals))
        else:
            separated.append(row)
    return sorted(separated, key=lambda row: (_phase_float(row, "elapsed_s_start"), _phase_float(row, "elapsed_s_end"), row.get("row_type", "")))


def _phase_rows_from_artifacts(result_dir: Path) -> list[dict[str, str]]:
    vision_rows = _read_phase_rows(result_dir / "vision_phase_stats.csv")
    # In coordinated-load mode both processes are launched by the same remote
    # script and use process-local clocks that are close enough for timeline
    # plotting. Do not offset decoder phases after vision phases: that would
    # incorrectly make loading appear after QNN encode.
    decoder_rows = _read_phase_rows(result_dir / "decoder_phase_stats.csv")
    return sorted(vision_rows + decoder_rows, key=lambda row: (_phase_float(row, "elapsed_s_start"), _phase_float(row, "elapsed_s_end")))


def _make_phase_row(name: str, start_s: float, duration_ms: float) -> dict[str, str]:
    end_s = start_s + max(duration_ms, 0.0) / 1000.0
    return {
        "row_type": name,
        "elapsed_s_start": f"{start_s:.6f}",
        "elapsed_s_end": f"{end_s:.6f}",
        "rss_kb_start": "",
        "rss_kb_end": "",
        "col_a_ms": f"{duration_ms:.3f}",
        "col_b_ms": "",
        "total_ms": f"{duration_ms:.3f}",
        "kv_pos": "",
        "kv_total": "",
        "kv_used_pct": "",
        "kv_estimated_used_kb": "",
        "kv_total_kb": "",
        "kv_physical_committed_kb": "",
        "token_idx": "0",
    }


def _synthetic_standalone_phase_rows(summary: dict[str, object]) -> list[dict[str, str]]:
    """Build comparable phase rows when upstream llama-mtmd-cli lacks timers."""
    image_encode_ms = float(summary.get("image_slice_encoded_ms", 0.0) or 0.0)
    image_prefill_ms = float(summary.get("image_decoded_ms", 0.0) or 0.0)
    prompt_eval_ms = float(summary.get("prompt_eval_time_ms", 0.0) or 0.0)
    text_prefill_ms = max(prompt_eval_ms - image_prefill_ms, 0.0)
    decode_ms = float(summary.get("decode_eval_time_ms", 0.0) or 0.0)
    phases = [
        ("V_Encode", image_encode_ms),
        ("T_Prefill", text_prefill_ms),
        ("ImagePrefill", image_prefill_ms),
        ("Decode", decode_ms),
    ]
    rows: list[dict[str, str]] = []
    cursor_s = 0.0
    for name, duration_ms in phases:
        if duration_ms <= 0:
            continue
        rows.append(_make_phase_row(name, cursor_s, duration_ms))
        cursor_s += duration_ms / 1000.0
    return rows


def _phase_colors() -> dict[str, str]:
    return {
        "L_VisionLoad": "#8e44ad",
        "ImageLoad": "#74b9ff",
        "V_Encode": "#00b894",
        "EmbeddingFileWrite": "#55efc4",
        "ExternalEmbeddingRead": "#00cec9",
        "L_DecoderRuntimeInit": "#a29bfe",
        "L_DecoderLoad": "#6c5ce7",
        "SingleBufferUpdate": "#636e72",
        "LayoutTokenize": "#fdcb6e",
        "Prefill": "#2d98da",
        "I_Prefill": "#0984e3",
        "ImagePrefill": "#0984e3",
        "T_Prefill": "#e17055",
        "Mmproj": "#00cec9",
        "DynamicKVGrow": "#2d3436",
        "D": "#ff7675",
        "Decode": "#d63031",
    }


def _write_png_memory_timeline(
    output_dir: Path,
    rows: list[dict[str, str]],
    phase_rows: list[dict[str, str]] | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    usable = [r for r in rows if r.get("elapsed_s") and r.get("mem_available_kb")]
    if not usable:
        return
    xs = [float(r["elapsed_s"]) for r in usable]
    ys = [float(r["mem_available_kb"]) / 1024.0 for r in usable]
    fig, ax = plt.subplots(figsize=(19, 8), dpi=160)
    ax.plot(
        xs,
        ys,
        label="MemAvailable (MiB)",
        linewidth=2.4,
        marker="o",
        markersize=2.6,
        color="#0984e3",
    )
    colors = _phase_colors()
    phase_count: dict[str, int] = {}
    for phase in phase_rows or []:
        name = phase.get("row_type", "")
        if name == "D":
            continue
        start = _phase_float(phase, "elapsed_s_start")
        end = _phase_float(phase, "elapsed_s_end")
        if end <= start:
            continue
        color = colors.get(name, "#636e72")
        ax.axvspan(start, end, color=color, alpha=0.06)
        ax.axvline(start, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        ax.axvline(end, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        phase_count[name] = phase_count.get(name, 0) + 1
        label = f"{name}{phase_count[name]}" if name in {"T_Prefill", "V_Encode"} else name
        ax.text(
            (start + end) / 2.0,
            0.045,
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
    ax.set_title(f"Android Memory Timeline: {output_dir.name}")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Memory (MiB)")
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
    fig.savefig(output_dir / "memory_timeline_plot.png", bbox_inches="tight")
    plt.close(fig)


def _write_png_memory_timeline_decode_window(
    output_dir: Path,
    rows: list[dict[str, str]],
    phase_rows: list[dict[str, str]] | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    phase_rows = phase_rows or []
    vision_starts = [_phase_float(row, "elapsed_s_start") for row in phase_rows if row.get("row_type") == "V_Encode"]
    decode_ends = [_phase_float(row, "elapsed_s_end") for row in phase_rows if row.get("row_type") in {"D", "Decode"}]
    if not vision_starts or not decode_ends:
        return
    window_start = min(vision_starts)
    window_end = max(decode_ends)
    if window_end <= window_start:
        return

    usable: list[dict[str, str]] = []
    for row in rows:
        if not row.get("elapsed_s") or not row.get("mem_available_kb"):
            continue
        try:
            elapsed = float(row["elapsed_s"])
        except ValueError:
            continue
        if window_start <= elapsed <= window_end:
            usable.append(row)
    if not usable:
        return

    xs = [float(r["elapsed_s"]) for r in usable]
    mem_available = [float(r["mem_available_kb"]) / 1024.0 for r in usable]
    kgsl = [float(r.get("kgsl_shmem_usage_kb") or 0) / 1024.0 for r in usable]

    fig, ax = plt.subplots(figsize=(19, 8), dpi=160)
    ax.plot(xs, mem_available, label="MemAvailable (MiB)", linewidth=2.4, marker="o", markersize=3.0, color="#0984e3")
    if any(value > 0 for value in kgsl):
        ax.plot(xs, kgsl, label="KgslShmemUsage (MiB)", linewidth=2.0, marker="s", markersize=2.5, color="#6c5ce7")

    colors = _phase_colors()
    phase_labels: dict[str, object] = {}
    for phase in phase_rows:
        name = phase.get("row_type", "")
        if name not in {"V_Encode", "ImagePrefill", "T_Prefill", "Mmproj", "DynamicKVGrow"}:
            continue
        start = _phase_float(phase, "elapsed_s_start")
        end = _phase_float(phase, "elapsed_s_end")
        if end <= start or end < window_start or start > window_end:
            continue
        color = colors.get(name, "#636e72")
        phase_alpha = 0.14 if name == "DynamicKVGrow" else 0.06
        span = ax.axvspan(start, end, color=color, alpha=phase_alpha)
        if name not in phase_labels:
            phase_labels[name] = span
        ax.axvline(start, color=color, linestyle="--", linewidth=1.1, alpha=0.85)
        if name == "DynamicKVGrow":
            detail = phase.get("token_idx", "")
            label = f"KV grow\n{detail}" if detail else "KV grow"
            ax.text(
                start,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=8,
                color=color,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.85},
            )

    ax.set_title(f"Memory Timeline From Vision Encode To Decode End: {output_dir.name}")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_xlim(left=window_start, right=window_end)
    ax.grid(True, linestyle=":", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    for name in ["V_Encode", "ImagePrefill", "T_Prefill", "Mmproj", "DynamicKVGrow"]:
        if name in phase_labels:
            handles.append(phase_labels[name])
            labels.append(name)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_timeline_decode_window.png", bbox_inches="tight")
    plt.close(fig)


def _write_png_phase_duration(output_dir: Path, perf: dict[str, float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    phases = [
        ("Load", perf.get("load_time_ms", 0.0)),
        ("QNN Vision Encoder", perf.get("qnn_vision_encode_ms", 0.0)),
        ("External Image Decode", perf.get("external_image_decode_ms", 0.0)),
        ("Prompt Eval", perf.get("prompt_eval_time_ms", 0.0)),
        ("Token Decode", perf.get("decode_eval_time_ms", 0.0)),
    ]
    phases = [(name, value) for name, value in phases if value > 0]
    if not phases:
        return
    total = sum(value for _, value in phases)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
    bottom = 0.0
    colors = ["#6c5ce7", "#00b894", "#0984e3", "#e17055", "#d63031"]
    for idx, (name, value) in enumerate(phases):
        color = colors[idx % len(colors)]
        ax.bar(["total"], [value], bottom=bottom, color=color, edgecolor="white")
        if value / total >= 0.05:
            ax.text(
                0,
                bottom + value / 2.0,
                f"{name}\n{value / 1000.0:.2f}s",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        bottom += value
    ax.set_title(f"Runtime Breakdown: {output_dir.name}")
    ax.set_ylabel("Elapsed Time (ms)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[idx % len(colors)])
        for idx in range(len(phases))
    ]
    labels = [f"{name}: {value / 1000.0:.2f}s" for name, value in phases]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_duration_stacked_bar.png", bbox_inches="tight")
    plt.close(fig)


def _write_png_phase_duration_from_rows(output_dir: Path, phase_rows: list[dict[str, str]]) -> None:
    if not phase_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    excluded_from_plot = {
        "EmbeddingFileWrite",
        "ExternalEmbeddingRead",
        "ImageLoad",
        "L_DecoderLoad",
        "L_DecoderRuntimeInit",
        "L_VisionLoad",
        "LayoutTokenize",
    }
    durations: dict[str, float] = {}
    first_start_s: dict[str, float] = {}
    has_decode_summary = any(row.get("row_type") == "Decode" for row in phase_rows)
    has_detailed_prefill = any(row.get("row_type") in {"ImagePrefill", "T_Prefill", "Mmproj"} for row in phase_rows)
    for row in phase_rows:
        name = row.get("row_type", "")
        if name in excluded_from_plot:
            continue
        if name == "Prefill" and has_detailed_prefill:
            continue
        if name == "D" and has_decode_summary:
            continue
        if name == "ImagePrefill":
            normalized = "I_Prefill"
        else:
            normalized = "Decode" if name == "D" else name
        start_s = _phase_float(row, "elapsed_s_start")
        duration = max(_phase_float(row, "elapsed_s_end") - start_s, 0.0)
        if duration > 0:
            durations[normalized] = durations.get(normalized, 0.0) + duration
            first_start_s[normalized] = min(first_start_s.get(normalized, start_s), start_s)
    phases = sorted(
        durations.items(),
        key=lambda item: (first_start_s.get(item[0], float("inf")), item[0]),
    )
    if not phases:
        return
    total = sum(value for _, value in phases)
    colors = _phase_colors()
    fig, ax = plt.subplots(figsize=(6.2, 8.8), dpi=160)
    bottom = 0.0
    for name, value in phases:
        color = colors.get(name, "#636e72")
        ax.bar(["total"], [value], bottom=bottom, color=color, edgecolor="white")
        if value / total >= 0.035:
            ax.text(
                0,
                bottom + value / 2.0,
                f"{name}\n{value:.3f}s",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        bottom += value
    ax.set_title(f"Precise Runtime Breakdown: {output_dir.name}")
    ax.set_ylabel("Elapsed Time (s)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors.get(name, "#636e72")) for name, _ in phases]
    labels = [f"{name}: {value:.3f}s ({value / total * 100:.1f}%)" for name, value in phases]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_duration_stacked_bar.png", bbox_inches="tight")
    plt.close(fig)


def _streaming_timeline_phase_name(name: str) -> str | None:
    aliases = {
        "VisionPrefillV_Encode": "V_Encode",
        "VisionPrefillMmproj": "Mmproj",
        "VisionPrefillImagePrefill": "ImagePrefill",
        "VisionPrefillT_Prefill": "T_Prefill",
    }
    name = aliases.get(name, name)
    if name in {"SingleBufferUpdate", "V_Encode", "Mmproj", "ImagePrefill", "T_Prefill", "D"}:
        return name
    return None


def _streaming_timeline_origin(
    stream_origin_video: float,
    prompt_markers: list[float],
    phases: list[tuple[str, float, float, int]],
) -> float:
    return stream_origin_video


def _write_png_streaming_phase_timeline(output_dir: Path, phase_rows: list[dict[str, str]]) -> None:
    if not phase_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return

    stream_origin_elapsed = 0.0
    stream_origin_video = 0.0
    events_path = output_dir / "stream_events.csv"
    if events_path.exists():
        for event_row in _read_csv_dicts(events_path):
            if event_row.get("event") not in {"StreamFrameEnqueue", "SingleBufferUpdate"}:
                continue
            try:
                stream_origin_elapsed = float(event_row.get("elapsed_s_start", "0") or 0)
                stream_origin_video = float(event_row.get("video_time_s", "0") or 0)
                break
            except ValueError:
                continue

    def to_stream_time(elapsed_s: float) -> float:
        return elapsed_s - stream_origin_elapsed + stream_origin_video

    phases: list[tuple[str, float, float, int]] = []
    prompt_idx = -1
    prompt_markers: list[float] = []
    for row in phase_rows:
        name = row.get("row_type", "")
        start = _phase_float(row, "elapsed_s_start")
        end = _phase_float(row, "elapsed_s_end")
        if name == "StreamPromptPrefill":
            prompt_idx += 1
            prompt_markers.append(to_stream_time(start))
            continue
        display_name = _streaming_timeline_phase_name(name)
        if display_name is None:
            continue
        if end <= start:
            if display_name != "SingleBufferUpdate":
                continue
            end = start + 0.015
        phases.append((display_name, to_stream_time(start), to_stream_time(end), max(prompt_idx, 0)))
    if not phases:
        return

    timeline_origin = _streaming_timeline_origin(stream_origin_video, prompt_markers, phases)
    decode_ends = [end for name, _, end, _ in phases if name == "D" and end >= timeline_origin]
    timeline_end = max(decode_ends) if decode_ends else max(end for _, _, end, _ in phases)
    phases = [
        (name, start, end, idx)
        for name, start, end, idx in phases
        if end >= timeline_origin and start <= timeline_end and not name.startswith("L_Decoder")
    ]
    prompt_markers = [
        marker
        for marker in prompt_markers
        if marker >= timeline_origin and marker <= timeline_end
    ]
    if not phases:
        return

    visible = [
        "SingleBufferUpdate",
        "V_Encode",
        "Mmproj",
        "ImagePrefill",
        "T_Prefill",
        "D",
    ]
    y_for = {name: idx for idx, name in enumerate(visible)}
    colors = _phase_colors()
    fig_height = max(4.8, 0.48 * len(visible) + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_height), dpi=160)
    for name, start, end, idx in phases:
        y = y_for.get(name)
        if y is None:
            continue
        color = colors.get(name, "#636e72")
        if name == "SingleBufferUpdate":
            ax.vlines(
                start,
                y - 0.32,
                y + 0.32,
                color=color,
                linewidth=1.4,
                alpha=0.85,
            )
            continue
        alpha = 0.38 if name == "SingleBufferUpdate" else 0.82
        ax.barh(
            y,
            end - start,
            left=start,
            height=0.55,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=alpha,
        )
        duration_ms = (end - start) * 1000.0
        if name != "D" and duration_ms >= 20.0:
            label = f"{duration_ms:.0f}ms"
            if name in {"V_Encode", "ImagePrefill", "T_Prefill"}:
                label = f"P{idx} {label}"
            if name == "DynamicKVGrow":
                label = f"KV +{duration_ms:.0f}ms"
            ax.text(
                start + (end - start) / 2.0,
                y,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    for idx, marker in enumerate(prompt_markers):
        ax.axvline(marker, color="#2d3436", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.text(
            marker,
            1.015,
            f"Prompt {idx} @ {marker:.1f}s",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#2d3436",
        )

    ax.set_yticks(list(y_for.values()))
    ax.set_yticklabels(visible)
    ax.invert_yaxis()
    ax.set_xlabel("Stream Time (s)")
    ax.set_title(f"Streaming Prompt Timeline: {output_dir.name}")
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    xmax = max(timeline_end, timeline_origin + 0.1)
    ax.set_xlim(left=timeline_origin, right=max(xmax * 1.02, xmax + 0.1))
    legend_names = [name for name in visible if any(phase[0] == name for phase in phases)]
    handles = [Patch(facecolor=colors.get(name, "#636e72"), label=name) for name in legend_names]
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "streaming_phase_timeline.png", bbox_inches="tight")
    plt.close(fig)


def _finalize_hybrid_outputs(result_dir: Path) -> None:
    decode_log = result_dir / "hybrid_decode_stdout.txt"
    if not decode_log.exists():
        decode_log = result_dir / "hybrid_decode_opencl_stdout.txt"
    vision_stats_path = result_dir / "vision_output_stats.csv"
    if not vision_stats_path.exists():
        vision_stats_path = result_dir / "hybrid_vision_output_stats_opencl.csv"
    if not decode_log.exists():
        return
    log_text = decode_log.read_text(encoding="utf-8", errors="replace")
    (result_dir / "foundation_output.txt").write_text(log_text, encoding="utf-8")

    raw_vision_stats: dict[str, str] = {}
    if vision_stats_path.exists():
        for row in _read_csv_dicts(vision_stats_path):
            raw_vision_stats[row.get("metric", "")] = row.get("value", "")
    qnn_encode_ms = raw_vision_stats.get("encode_ms") or raw_vision_stats.get("qnn_vision_encode_ms") or 0
    perf: dict[str, float] = {
        "load_time_ms": _match_float(log_text, r"llama_perf_context_print:\s+load time =\s+([0-9.]+) ms"),
        "qnn_vision_encode_ms": float(qnn_encode_ms or 0),
        "external_image_decode_ms": _match_float(log_text, r"image decoded(?: \(batch \d+/\d+\))? in ([0-9.]+) ms"),
        "prompt_eval_time_ms": _match_float(log_text, r"llama_perf_context_print:\s+prompt eval time =\s+([0-9.]+) ms"),
        "decode_eval_time_ms": _match_float(log_text, r"llama_perf_context_print:\s+eval time =\s+([0-9.]+) ms"),
        "total_time_ms": _match_float(log_text, r"llama_perf_context_print:\s+total time =\s+([0-9.]+) ms"),
        "prompt_eval_tokens": _match_int(log_text, r"prompt eval time =\s+[0-9.]+ ms /\s+([0-9]+) tokens"),
        "decode_eval_runs": _match_int(log_text, r"eval time =\s+[0-9.]+ ms /\s+([0-9]+) runs"),
    }
    if perf["prompt_eval_time_ms"] and perf["prompt_eval_tokens"]:
        perf["prompt_eval_tok_s"] = perf["prompt_eval_tokens"] / perf["prompt_eval_time_ms"] * 1000.0
    if perf["decode_eval_time_ms"] and perf["decode_eval_runs"]:
        perf["decode_eval_tok_s"] = perf["decode_eval_runs"] / perf["decode_eval_time_ms"] * 1000.0

    memory_rows = []
    memory_path = result_dir / "android_memory_timeline.csv"
    if memory_path.exists():
        memory_rows = _read_csv_dicts(memory_path)
    wall_s = max([float(row.get("elapsed_s", "0") or 0) for row in memory_rows] or [0.0])
    return_code = ""
    if (result_dir / "foundation_exit_code.txt").exists():
        return_code = (result_dir / "foundation_exit_code.txt").read_text(encoding="utf-8").strip()

    summary_rows: list[dict[str, object]] = [
        {"metric": "backend", "value": "hybrid_qnn_vision_llamacpp_opencl", "unit": ""},
        {"metric": "model_name", "value": result_dir.name, "unit": ""},
        {"metric": "wall_time_s", "value": round(wall_s, 3), "unit": "s"},
        {"metric": "return_code", "value": return_code, "unit": ""},
        {"metric": "vision_output_dims", "value": raw_vision_stats.get("output_dims", ""), "unit": ""},
        {"metric": "vision_input_count", "value": raw_vision_stats.get("input_count", ""), "unit": "frames_or_tiles"},
    ]
    for key, value in perf.items():
        unit = "ms" if key.endswith("_ms") else "tok/s" if key.endswith("_tok_s") else "tokens" if "tokens" in key else "runs" if "runs" in key else ""
        summary_rows.append({"metric": key, "value": value, "unit": unit})
    _write_csv(result_dir / "foundation_summary.csv", summary_rows, ["metric", "value", "unit"])

    phase_rows = _phase_rows_from_artifacts(result_dir)
    if phase_rows:
        phase_rows.extend(_dynamic_kv_rows_from_stdout(stdout_path))
        phase_rows.sort(key=lambda row: (_phase_float(row, "elapsed_s_start"), _phase_float(row, "elapsed_s_end")))
        _write_phase_csv(result_dir / "foundation_proc.csv", phase_rows)
        plot_phase_rows = _read_phase_rows(result_dir / "foundation_proc.csv")
    else:
        _write_csv(result_dir / "foundation_proc.csv", summary_rows, ["metric", "value", "unit"])
        plot_phase_rows = []
    _write_csv(
        result_dir / "vision_output_stats.csv",
        [
            {"metric": "input_dims", "value": raw_vision_stats.get("input_dims", ""), "unit": ""},
            {"metric": "output_dims", "value": raw_vision_stats.get("output_dims", ""), "unit": ""},
            {"metric": "output_values", "value": raw_vision_stats.get("output_values", ""), "unit": "float32"},
            {"metric": "input_count", "value": raw_vision_stats.get("input_count", ""), "unit": "frames_or_tiles"},
            {"metric": "qnn_vision_encode_ms", "value": perf["qnn_vision_encode_ms"], "unit": "ms"},
            {"metric": "external_image_decode_ms", "value": perf["external_image_decode_ms"], "unit": "ms"},
            {"metric": "prompt_eval_time_ms", "value": perf["prompt_eval_time_ms"], "unit": "ms"},
            {"metric": "decode_eval_time_ms", "value": perf["decode_eval_time_ms"], "unit": "ms"},
            {"metric": "total_time_ms", "value": perf["total_time_ms"], "unit": "ms"},
        ],
        ["metric", "value", "unit"],
    )
    _write_memory_usage_txt(result_dir, memory_rows)
    _write_png_memory_timeline(result_dir, memory_rows, plot_phase_rows)
    if plot_phase_rows:
        _write_png_phase_duration_from_rows(result_dir, plot_phase_rows)
    else:
        _write_png_phase_duration(result_dir, perf)


def _finalize_hybrid_streaming_outputs(result_dir: Path) -> None:
    stdout_path = result_dir / "hybrid_streaming_stdout.txt"
    if not stdout_path.exists():
        stdout_path = result_dir / "opencl_streaming_stdout.txt"
    if stdout_path.exists() and not (result_dir / "foundation_output.txt").exists():
        (result_dir / "foundation_output.txt").write_text(
            stdout_path.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

    memory_rows = []
    memory_path = result_dir / "android_memory_timeline.csv"
    if memory_path.exists():
        memory_rows = _read_csv_dicts(memory_path)
    wall_s = max([float(row.get("elapsed_s", "0") or 0) for row in memory_rows] or [0.0])
    return_code = ""
    if (result_dir / "foundation_exit_code.txt").exists():
        return_code = (result_dir / "foundation_exit_code.txt").read_text(encoding="utf-8").strip()

    event_rows = _read_csv_dicts(result_dir / "stream_events.csv") if (result_dir / "stream_events.csv").exists() else []
    n_frames = sum(1 for row in event_rows if row.get("event") == "StreamFrameEnqueue")
    n_prompts = sum(1 for row in event_rows if row.get("event") == "StreamPromptPrefill")
    backend = "hybrid_qnn_vision_streaming_sim" if (result_dir / "hybrid_streaming_stdout.txt").exists() else "opencl_streaming_sim"
    summary_rows: list[dict[str, object]] = [
        {"metric": "backend", "value": backend, "unit": ""},
        {"metric": "model_name", "value": result_dir.name, "unit": ""},
        {"metric": "wall_time_s", "value": round(wall_s, 3), "unit": "s"},
        {"metric": "return_code", "value": return_code, "unit": ""},
        {"metric": "stream_frame_count", "value": n_frames, "unit": "frames"},
        {"metric": "stream_prompt_count", "value": n_prompts, "unit": "prompts"},
    ]
    _write_csv(result_dir / "foundation_summary.csv", summary_rows, ["metric", "value", "unit"])

    phase_rows = _read_phase_rows(result_dir / "streaming_phase_stats.csv")
    if phase_rows:
        phase_clock_origin_ms = _phase_clock_origin_ms(result_dir / "streaming_phase_stats.csv")
        phase_rows.extend(_dynamic_kv_rows_from_stdout(stdout_path, clock_origin_ms=phase_clock_origin_ms))
        phase_rows = _separate_dynamic_kv_grow_overlaps(phase_rows)
        phase_rows.sort(key=lambda row: (_phase_float(row, "elapsed_s_start"), _phase_float(row, "elapsed_s_end")))
        _write_phase_csv(result_dir / "foundation_proc.csv", phase_rows)
        plot_phase_rows = _read_phase_rows(result_dir / "foundation_proc.csv")
    else:
        _write_csv(result_dir / "foundation_proc.csv", summary_rows, ["metric", "value", "unit"])
        plot_phase_rows = []
    _write_memory_usage_txt(result_dir, memory_rows)
    _write_png_memory_timeline(result_dir, memory_rows, plot_phase_rows)
    _write_png_memory_timeline_decode_window(result_dir, memory_rows, plot_phase_rows)
    if plot_phase_rows:
        _write_png_phase_duration_from_rows(result_dir, plot_phase_rows)
        _write_png_streaming_phase_timeline(result_dir, plot_phase_rows)


def _finalize_standalone_outputs(
    result_dir: Path,
    *,
    processor: str,
    return_code: str,
    prompt: str = "Describe this image briefly.",
    text_only: bool = False,
) -> None:
    output_path = result_dir / "foundation_output.txt"
    if not output_path.exists():
        return
    log_text = output_path.read_text(encoding="utf-8", errors="replace")
    _write_fallback_token_io_txt(result_dir, prompt, log_text, text_only=text_only)
    summary = _parse_log_summary(log_text)
    memory_rows = []
    memory_path = result_dir / "android_memory_timeline.csv"
    if memory_path.exists():
        memory_rows = _read_csv_dicts(memory_path)
    wall_s = max([float(row.get("elapsed_s", "0") or 0) for row in memory_rows] or [0.0])

    summary_rows: list[dict[str, object]] = [
        {
            "metric": "backend",
            "value": (
                "llamacpp_cpu_text"
                if processor == "cpu" and text_only
                else "llamacpp_opencl_text"
                if processor == "gpu" and text_only
                else "llamacpp_cpu"
                if processor == "cpu"
                else "llamacpp_opencl"
            ),
            "unit": "",
        },
        {"metric": "model_name", "value": result_dir.name, "unit": ""},
        {"metric": "wall_time_s", "value": round(wall_s, 3), "unit": "s"},
        {"metric": "return_code", "value": return_code, "unit": ""},
    ]
    for key, value in summary.items():
        unit = (
            "ms" if key.endswith("_ms")
            else "MiB" if key.endswith("_mib")
            else "tok/s" if key.endswith("_tok_s")
            else "tokens" if "tokens" in key
            else "runs" if "runs" in key
            else ""
        )
        summary_rows.append({"metric": key, "value": value, "unit": unit})
    _write_csv(result_dir / "foundation_summary.csv", summary_rows, ["metric", "value", "unit"])

    phase_rows = _read_phase_rows(result_dir / "foundation_phase_stats.csv")
    phase_rows_are_synthetic = False
    if not phase_rows:
        phase_rows = _synthetic_standalone_phase_rows(summary)
        phase_rows_are_synthetic = bool(phase_rows)
    if phase_rows:
        _write_phase_csv(result_dir / "foundation_proc.csv", phase_rows)
        plot_phase_rows = _read_phase_rows(result_dir / "foundation_proc.csv")
    else:
        _write_csv(result_dir / "foundation_proc.csv", summary_rows, ["metric", "value", "unit"])
        plot_phase_rows = []
    # Synthetic CPU rows are summary-derived durations, not wall-clock aligned
    # timestamps. Do not overlay them on memory timelines.
    memory_phase_rows = [] if phase_rows_are_synthetic else plot_phase_rows
    _write_memory_usage_txt(result_dir, memory_rows)
    _write_png_memory_timeline(result_dir, memory_rows, memory_phase_rows)
    if plot_phase_rows:
        _write_png_phase_duration_from_rows(result_dir, plot_phase_rows)
    else:
        perf = {key: float(value) for key, value in summary.items() if key.endswith("_ms") and isinstance(value, (int, float))}
        _write_png_phase_duration(result_dir, perf)


def _result_kv_slug_part(cache_type: str | None) -> str:
    """Slugs for result dir names: q8_0 -> 8, q4_0 -> 4, f16 -> 16 → suffix _kv16."""
    t = (cache_type or "f16").strip().lower().replace("-", "_")
    if t in ("fp16", "f16"):
        return "16"
    m = re.fullmatch(r"q([0-9]+)_0", t)
    if m:
        return m.group(1)
    return re.sub(r"[^a-z0-9_]+", "_", t).strip("_") or "unknown"


def _result_kv_suffix(cache_type_k: str | None, cache_type_v: str | None) -> str:
    k = _result_kv_slug_part(cache_type_k)
    v = _result_kv_slug_part(cache_type_v)
    if k == v:
        return f"_kv{k}"
    return f"_kv{k}_{v}"


def _result_model_name(
    model: Path,
    processor: str,
    ctx_size: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    *,
    text_only: bool = False,
    streaming: bool = False,
    stream_mode: str | None = None,
    dynamic_kv_cache: bool = False,
    paged_kv_cache: bool = False,
) -> str:
    suffix = "opencl" if processor == "gpu" else processor
    kv = _result_kv_suffix(cache_type_k, cache_type_v)
    if streaming:
        mode_suffix = "" if stream_mode in (None, "single_buffer") else f"_{stream_mode}"
        mid = f"_streaming{mode_suffix}"
    else:
        mid = "_text" if text_only else ""
    dynamic = "_dynamic" if dynamic_kv_cache else ""
    paged = "_paged" if paged_kv_cache else ""
    return f"{model.stem}_{suffix}_ctx_{ctx_size}{mid}{kv}{dynamic}{paged}"


def _find_executable(build_dir: Path, name: str) -> Path:
    for path in (build_dir / name, build_dir / "bin" / name):
        if path.exists():
            return path
    return build_dir / name


def _memory_csv_header() -> str:
    return (
        "sample_idx,elapsed_s,pid,pid_alive,vmrss_kb,vmsize_kb,vmhwm_kb,"
        "smaps_rss_kb,smaps_pss_kb,smaps_private_dirty_kb,smaps_shared_clean_kb,"
        "mem_available_kb,cached_kb,dma_heap_pool_kb,gpu_total_kb,kgsl_shmem_usage_kb"
    )


def _baseline_sampling_shell(remote_memory_csv: str, sample_interval: float, baseline_window_s: float) -> str:
    if baseline_window_s <= 0:
        return ":"
    return f"""baseline_start_uptime=$(awk '{{print $1; exit}}' /proc/uptime 2>/dev/null)
baseline_idx=0
while true; do
  now_uptime=$(awk '{{print $1; exit}}' /proc/uptime 2>/dev/null)
  elapsed_s=$(awk -v now="${{now_uptime:-0}}" -v start="${{baseline_start_uptime:-0}}" -v window="{baseline_window_s}" 'BEGIN {{ e = now - start - window; if (e > 0) e = 0; printf "%.3f", e }}')
  mem_available=$(awk '/^MemAvailable:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  cached=$(awk '/^Cached:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  dma_heap_pool=$(awk '/^DmaHeapPool:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  gpu_total=$(awk '/^GpuTotal:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  kgsl_shmem_usage=$(awk '/^KgslShmemUsage:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  printf '%s,%s,0,0,0,0,0,0,0,0,0,%s,%s,%s,%s,%s\\n' "$baseline_idx" "$elapsed_s" "${{mem_available:-0}}" "${{cached:-0}}" "${{dma_heap_pool:-0}}" "${{gpu_total:-0}}" "${{kgsl_shmem_usage:-0}}" >> {shlex.quote(remote_memory_csv)}
  done_flag=$(awk -v now="${{now_uptime:-0}}" -v start="${{baseline_start_uptime:-0}}" -v window="{baseline_window_s}" 'BEGIN {{ print ((now - start) >= window) ? 1 : 0 }}')
  if [ "$done_flag" = "1" ]; then
    break
  fi
  baseline_idx=$((baseline_idx + 1))
  sleep {sample_interval}
done"""


def _memory_sampling_shell(remote_memory_csv: str, sample_interval: float, live_condition: str, pid_expr: str) -> str:
    return f"""sample_idx=0
start_uptime=$(awk '{{print $1; exit}}' /proc/uptime 2>/dev/null)
while {live_condition}; do
  elapsed_s=$(awk -v start="${{start_uptime:-0}}" '{{ printf "%.3f", $1 - start }}' /proc/uptime 2>/dev/null)
  sample_pid={pid_expr}
  status_file=/proc/"$sample_pid"/status
  smaps_file=/proc/"$sample_pid"/smaps_rollup
  vmrss=$(awk '/^VmRSS:/ {{print $2; exit}}' "$status_file" 2>/dev/null)
  vmsize=$(awk '/^VmSize:/ {{print $2; exit}}' "$status_file" 2>/dev/null)
  vmhwm=$(awk '/^VmHWM:/ {{print $2; exit}}' "$status_file" 2>/dev/null)
  smaps_rss=$(awk '/^Rss:/ {{print $2; exit}}' "$smaps_file" 2>/dev/null)
  smaps_pss=$(awk '/^Pss:/ {{print $2; exit}}' "$smaps_file" 2>/dev/null)
  smaps_private_dirty=$(awk '/^Private_Dirty:/ {{print $2; exit}}' "$smaps_file" 2>/dev/null)
  smaps_shared_clean=$(awk '/^Shared_Clean:/ {{print $2; exit}}' "$smaps_file" 2>/dev/null)
  mem_available=$(awk '/^MemAvailable:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  cached=$(awk '/^Cached:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  dma_heap_pool=$(awk '/^DmaHeapPool:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  gpu_total=$(awk '/^GpuTotal:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  kgsl_shmem_usage=$(awk '/^KgslShmemUsage:/ {{print $2; exit}}' /proc/meminfo 2>/dev/null)
  printf '%s,%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\\n' "$sample_idx" "$elapsed_s" "$sample_pid" "${{vmrss:-0}}" "${{vmsize:-0}}" "${{vmhwm:-0}}" "${{smaps_rss:-0}}" "${{smaps_pss:-0}}" "${{smaps_private_dirty:-0}}" "${{smaps_shared_clean:-0}}" "${{mem_available:-0}}" "${{cached:-0}}" "${{dma_heap_pool:-0}}" "${{gpu_total:-0}}" "${{kgsl_shmem_usage:-0}}" >> {shlex.quote(remote_memory_csv)}
  sample_idx=$((sample_idx + 1))
  sleep {sample_interval}
done"""


def _extend_llama_rope_cli(cmd: list[str], args: argparse.Namespace) -> None:
    """Passthrough llama.cpp RoPE / YaRN flags (common_params)."""
    rs = getattr(args, "rope_scaling", None)
    if rs:
        cmd.extend(["--rope-scaling", rs])
    if getattr(args, "rope_scale", None) is not None:
        cmd.extend(["--rope-scale", str(args.rope_scale)])
    if getattr(args, "rope_freq_base", None) is not None:
        cmd.extend(["--rope-freq-base", str(args.rope_freq_base)])
    if getattr(args, "rope_freq_scale", None) is not None:
        cmd.extend(["--rope-freq-scale", str(args.rope_freq_scale)])
    if getattr(args, "yarn_orig_ctx", None) is not None:
        cmd.extend(["--yarn-orig-ctx", str(args.yarn_orig_ctx)])
    if getattr(args, "yarn_ext_factor", None) is not None:
        cmd.extend(["--yarn-ext-factor", str(args.yarn_ext_factor)])
    if getattr(args, "yarn_attn_factor", None) is not None:
        cmd.extend(["--yarn-attn-factor", str(args.yarn_attn_factor)])
    if getattr(args, "yarn_beta_slow", None) is not None:
        cmd.extend(["--yarn-beta-slow", str(args.yarn_beta_slow)])
    if getattr(args, "yarn_beta_fast", None) is not None:
        cmd.extend(["--yarn-beta-fast", str(args.yarn_beta_fast)])


def _rope_shell_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    rs = getattr(args, "rope_scaling", None)
    if rs:
        parts.append(f"--rope-scaling {shlex.quote(rs)}")
    if getattr(args, "rope_scale", None) is not None:
        parts.append(f"--rope-scale {shlex.quote(str(args.rope_scale))}")
    if getattr(args, "rope_freq_base", None) is not None:
        parts.append(f"--rope-freq-base {shlex.quote(str(args.rope_freq_base))}")
    if getattr(args, "rope_freq_scale", None) is not None:
        parts.append(f"--rope-freq-scale {shlex.quote(str(args.rope_freq_scale))}")
    if getattr(args, "yarn_orig_ctx", None) is not None:
        parts.append(f"--yarn-orig-ctx {shlex.quote(str(args.yarn_orig_ctx))}")
    if getattr(args, "yarn_ext_factor", None) is not None:
        parts.append(f"--yarn-ext-factor {shlex.quote(str(args.yarn_ext_factor))}")
    if getattr(args, "yarn_attn_factor", None) is not None:
        parts.append(f"--yarn-attn-factor {shlex.quote(str(args.yarn_attn_factor))}")
    if getattr(args, "yarn_beta_slow", None) is not None:
        parts.append(f"--yarn-beta-slow {shlex.quote(str(args.yarn_beta_slow))}")
    if getattr(args, "yarn_beta_fast", None) is not None:
        parts.append(f"--yarn-beta-fast {shlex.quote(str(args.yarn_beta_fast))}")
    return (" " + " ".join(parts)) if parts else ""


def _media_image_arg(args: argparse.Namespace) -> str:
    images = getattr(args, "remote_layout_images", None)
    if images:
        return ",".join(images)
    if args.image is None:
        raise SystemExit("--image is required for multimodal standalone mode.")
    return args.image.name


def _build_standalone_command(args: argparse.Namespace, *, use_precise_phases: bool) -> list[str]:
    selected_gpu_layers = 0 if args.processor == "cpu" else args.gpu_layers
    selected_device = args.device
    if selected_device is None and args.processor == "cpu":
        selected_device = "none"
    n_predict = args.force_generation or args.n_predict
    cmd = [
        "./opencl_phase_mtmd" if use_precise_phases else "./llama-mtmd-cli",
        "-m",
        args.model.name,
        "--mmproj",
        args.mmproj.name,
        "--image",
        _media_image_arg(args),
        "-p",
        getattr(args, "remote_prompt", args.prompt),
        "-n",
        str(n_predict),
        "-t",
        str(args.threads),
        "--n-gpu-layers",
        str(selected_gpu_layers),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--temp",
        str(args.temperature),
    ]
    if use_precise_phases:
        cmd.extend(["--phase-stats-path", "foundation_phase_stats.csv"])
        cmd.extend(["--token-io-path", "foundation_token_io.txt"])
        if getattr(args, "remote_warmup_layout_image", ""):
            cmd.extend(["--warmup-image", args.remote_warmup_layout_image])
        if args.force_generation:
            cmd.append("--force-generation")
    elif args.force_generation:
        cmd.append("--ignore-eos")
    if selected_device:
        cmd.extend(["--device", selected_device])
    if getattr(args, "cache_type_k", None):
        cmd.extend(["--cache-type-k", args.cache_type_k])
    if getattr(args, "cache_type_v", None):
        cmd.extend(["--cache-type-v", args.cache_type_v])
    if getattr(args, "fit", None) is not None:
        cmd.extend(["--fit", args.fit])
    if getattr(args, "flash_attn", None):
        cmd.extend(["--flash-attn", args.flash_attn])
    if getattr(args, "no_kv_offload", False):
        cmd.append("--no-kv-offload")
    if getattr(args, "no_warmup", False):
        cmd.append("--no-warmup")
    if getattr(args, "paged_kv_cache", False):
        cmd.extend(["--paged-kv-cache", "--kv-page-size", str(args.kv_page_size)])
    _extend_llama_rope_cli(cmd, args)
    return cmd


def _build_text_only_command(args: argparse.Namespace) -> list[str]:
    """Standalone text generation via upstream llama-completion (no mmproj / vision)."""
    selected_gpu_layers = 0 if args.processor == "cpu" else args.gpu_layers
    selected_device = args.device
    if selected_device is None and args.processor == "cpu":
        selected_device = "none"
    n_predict = args.force_generation or args.n_predict
    cmd: list[str] = [
        "./llama-completion",
        "-m",
        args.model.name,
        "-no-cnv",
        "-lv",
        "2",
        "-co",
        "off",
        "--no-display-prompt",
        "-st",
        "-p",
        args.prompt,
        "-n",
        str(n_predict),
        "-t",
        str(args.threads),
        "-ngl",
        str(selected_gpu_layers),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--temp",
        str(args.temperature),
    ]
    if args.force_generation:
        cmd.append("--ignore-eos")
    if selected_device:
        cmd.extend(["--device", selected_device])
    if getattr(args, "cache_type_k", None):
        cmd.extend(["--cache-type-k", args.cache_type_k])
    if getattr(args, "cache_type_v", None):
        cmd.extend(["--cache-type-v", args.cache_type_v])
    if getattr(args, "fit", None) is not None:
        cmd.extend(["--fit", args.fit])
    if getattr(args, "flash_attn", None):
        cmd.extend(["--flash-attn", args.flash_attn])
    if getattr(args, "no_kv_offload", False):
        cmd.append("--no-kv-offload")
    if getattr(args, "no_warmup", False):
        cmd.append("--no-warmup")
    if getattr(args, "paged_kv_cache", False):
        cmd.extend(["--paged-kv-cache", "--kv-page-size", str(args.kv_page_size)])
    _extend_llama_rope_cli(cmd, args)
    return cmd


def _cache_type_shell_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if getattr(args, "cache_type_k", None):
        parts.append(f"--cache-type-k {shlex.quote(args.cache_type_k)}")
    if getattr(args, "cache_type_v", None):
        parts.append(f"--cache-type-v {shlex.quote(args.cache_type_v)}")
    return (" " + " ".join(parts)) if parts else ""


def _fit_shell_suffix(args: argparse.Namespace) -> str:
    if getattr(args, "fit", None) is None:
        return ""
    return f" --fit {args.fit}"


def _flash_attn_kv_shell_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    fa = getattr(args, "flash_attn", None)
    if fa:
        parts.append(f"--flash-attn {shlex.quote(fa)}")
    if getattr(args, "no_kv_offload", False):
        parts.append("--no-kv-offload")
    if getattr(args, "no_warmup", False):
        parts.append("--no-warmup")
    return (" " + " ".join(parts)) if parts else ""


def _ctx_dynamic_kv_shell_suffix(args: argparse.Namespace) -> str:
    if getattr(args, "paged_kv_cache", False):
        return f" --paged-kv-cache --kv-page-size {args.kv_page_size}"
    if getattr(args, "dynamic_kv_cache", False):
        parts = ["--dynamic-kv-cache"]
        if getattr(args, "kv_init_size", None):
            parts.extend(["--kv-init-size", str(args.kv_init_size)])
        if getattr(args, "kv_grow_step", None):
            parts.extend(["--kv-grow-step", str(args.kv_grow_step)])
        return " " + " ".join(parts)
    return f" -c {args.ctx_size}"


def _remote_llama_env_exports(args: argparse.Namespace) -> str:
    """Lines emitted after cd ... || exit 1 (LD_LIBRARY_PATH + optional experiment vars)."""
    lines: list[str] = []
    if getattr(args, "disable_attn_kv_rotation", False):
        lines.append("export LLAMA_ATTN_ROT_DISABLE=1")
    dbg_kv = os.environ.get("GGML_OPENCL_DEBUG_KV")
    if dbg_kv:
        lines.append(f"export GGML_OPENCL_DEBUG_KV={shlex.quote(dbg_kv)}")
    lines.append("export LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.")
    return "\n".join(lines)


def _build_hybrid_remote_script(args: argparse.Namespace, *, encoder_pte: Path, media: PreparedMedia) -> str:
    prompt = shlex.quote(media.prompt)
    image_arg = ",".join(path.name for path in media.layout_images)
    image_paths_arg = ",".join(path.name for path in media.frame_bins)
    warmup_image_arg = f"--warmup_image_path={shlex.quote(args.remote_warmup_bin)}" if getattr(args, "remote_warmup_bin", "") else ""
    warmup_embedding_arg = f"--warmup-embedding {shlex.quote(args.remote_warmup_embedding)}" if getattr(args, "remote_warmup_embedding", "") else ""
    group_sizes_arg = ",".join(str(n) for n in media.num_patches_list)
    device_arg = f"--device {shlex.quote(args.device)}" if args.device else ""
    remote_memory_csv = f"{args.remote_root}/android_memory_timeline.csv"
    baseline_loop = _baseline_sampling_shell(remote_memory_csv, args.sample_interval, args.baseline_window)
    memory_loop = _memory_sampling_shell(
        remote_memory_csv,
        args.sample_interval,
        'kill -0 "$decoder_pid" 2>/dev/null || kill -0 "$vision_pid" 2>/dev/null',
        '"$decoder_pid"',
    )
    cache_suffix = _cache_type_shell_suffix(args)
    fit_suffix = _fit_shell_suffix(args)
    rope_suffix = _rope_shell_suffix(args)
    flash_kv_suffix = _flash_attn_kv_shell_suffix(args)
    ctx_dynamic_kv_suffix = _ctx_dynamic_kv_shell_suffix(args)
    env_exports = _remote_llama_env_exports(args)
    return f"""#!/system/bin/sh
cd {shlex.quote(args.remote_root)} || exit 1
{env_exports}
rm -f android_memory_timeline.csv hybrid_vision_stdout.txt hybrid_decode_stdout.txt \\
  vision_output_stats.csv vision_phase_stats.csv decoder_phase_stats.csv \\
  foundation_token_io.txt foundation_inference_tokens.txt \\
  foundation_exit_code.txt vision_exit_code.txt decoder_exit_code.txt \\
  vision_ready.flag decoder_ready.flag start_encode.flag vision_embedding.svlmemb warmup_vision_embedding.svlmemb
printf '%s\\n' '{_memory_csv_header()}' > {shlex.quote(remote_memory_csv)}

{baseline_loop}

./hybrid_decode -m {shlex.quote(args.model.name)} --mmproj {shlex.quote(args.mmproj.name)} \\
  --image {shlex.quote(image_arg)} --external-embedding vision_embedding.svlmemb \\
  {warmup_embedding_arg} \\
  --phase-stats-path decoder_phase_stats.csv --ready-path decoder_ready.flag \\
  --wait-for-embedding --wait-timeout-ms 120000 \\
  --token-io-path foundation_token_io.txt {('--force-generation' if args.force_generation else '')} \\
  -p {prompt} -n {args.force_generation or args.n_predict}{ctx_dynamic_kv_suffix} -b {args.batch_size} -ub {args.ubatch_size} \\
  -ngl {args.gpu_layers} {device_arg}{cache_suffix}{fit_suffix}{rope_suffix}{flash_kv_suffix} > hybrid_decode_stdout.txt 2>&1 &
decoder_pid=$!

./hybrid_vision_dump --encoder_path={shlex.quote(encoder_pte.name)} \\
  --image_paths={shlex.quote(image_paths_arg)} --output_path=vision_embedding.svlmemb \\
  --warmup_output_path=warmup_vision_embedding.svlmemb \\
  {warmup_image_arg} \\
  --group_sizes={shlex.quote(group_sizes_arg)} \\
  --stats_path=vision_output_stats.csv --phase_stats_path=vision_phase_stats.csv \\
  --ready_path=vision_ready.flag --wait_path=start_encode.flag --wait_timeout_ms=120000 \\
  > hybrid_vision_stdout.txt 2>&1 &
vision_pid=$!

(
  ready_i=0
  while [ "$ready_i" -lt 2400 ]; do
    if [ -f decoder_ready.flag ] && [ -f vision_ready.flag ]; then
      touch start_encode.flag
      exit 0
    fi
    sleep 0.05
    ready_i=$((ready_i + 1))
  done
  echo coordinator_timeout > coordinator_error.txt
  touch start_encode.flag
) &
coordinator_pid=$!

{memory_loop}

wait "$vision_pid"
vision_rc=$?
wait "$decoder_pid"
decoder_rc=$?
wait "$coordinator_pid" 2>/dev/null
echo "$vision_rc" > vision_exit_code.txt
echo "$decoder_rc" > decoder_exit_code.txt
echo "$decoder_rc" > foundation_exit_code.txt
exit "$decoder_rc"
"""


def _build_hybrid_streaming_remote_script(args: argparse.Namespace) -> str:
    remote_memory_csv = f"{args.remote_root}/android_memory_timeline.csv"
    baseline_loop = _baseline_sampling_shell(remote_memory_csv, args.sample_interval, args.baseline_window)
    memory_loop = _memory_sampling_shell(
        remote_memory_csv,
        args.sample_interval,
        'kill -0 "$runner_pid" 2>/dev/null',
        '"$runner_pid"',
    )
    env_exports = _remote_llama_env_exports(args)
    device_arg = f"--device {shlex.quote(args.device)}" if args.device else ""
    cache_suffix = _cache_type_shell_suffix(args)
    fit_suffix = _fit_shell_suffix(args)
    rope_suffix = _rope_shell_suffix(args)
    flash_kv_suffix = _flash_attn_kv_shell_suffix(args)
    ctx_dynamic_kv_suffix = _ctx_dynamic_kv_shell_suffix(args)
    force_arg = "--force-generation" if args.force_generation else ""
    single_buffer_arg = "--single-buffer" if args.single_buffer else ""
    stream_mode_arg = f"--stream-mode {shlex.quote(args.stream_mode)}"
    window_sec_arg = f"--window-sec {args.window_sec}" if args.window_sec is not None else ""
    window_max_frames_arg = f"--window-max-frames {args.window_max_frames}"
    is_hybrid = args.processor == "hybrid"
    runner_bin = "hybrid_streaming_decode" if is_hybrid else "opencl_streaming_decode"
    stdout_name = "hybrid_streaming_stdout.txt" if is_hybrid else "opencl_streaming_stdout.txt"
    encoder_arg = f"--encoder-path {shlex.quote(args.remote_encoder_pte)}" if is_hybrid else ""
    warmup_image_arg = (
        f"--warmup-image-path {shlex.quote(args.remote_warmup_bin)}"
        if is_hybrid and getattr(args, "remote_warmup_bin", "")
        else ""
    )
    return f"""#!/system/bin/sh
cd {shlex.quote(args.remote_root)} || exit 1
{env_exports}
rm -f android_memory_timeline.csv hybrid_streaming_stdout.txt opencl_streaming_stdout.txt stream_events.csv streaming_phase_stats.csv \\
  foundation_output.txt foundation_token_io.txt foundation_inference_tokens.txt foundation_exit_code.txt \\
  stream_response_*.txt stream_token_io_*.txt stream_inference_tokens_*.txt stream_prompt_phase_*.csv
printf '%s\\n' '{_memory_csv_header()}' > {shlex.quote(remote_memory_csv)}

{baseline_loop}

./{runner_bin} {single_buffer_arg} --runner ./opencl_phase_mtmd \\
  {encoder_arg} {warmup_image_arg} \\
  {stream_mode_arg} {window_sec_arg} {window_max_frames_arg} \\
  -m {shlex.quote(args.model.name)} --mmproj {shlex.quote(args.mmproj.name)} \\
  --stream-manifest media_manifest.json \\
  --stream-events-path stream_events.csv --phase-stats-path streaming_phase_stats.csv \\
  --output foundation_output.txt --token-io-path foundation_token_io.txt \\
  -n {args.force_generation or args.n_predict}{ctx_dynamic_kv_suffix} -b {args.batch_size} -ub {args.ubatch_size} \\
  -ngl {args.gpu_layers} -t {args.threads} --temp {args.temperature} \\
  {device_arg}{cache_suffix}{fit_suffix}{rope_suffix}{flash_kv_suffix} {force_arg} > {stdout_name} 2>&1 &
runner_pid=$!

{memory_loop}

wait "$runner_pid"
runner_rc=$?
echo "$runner_rc" > foundation_exit_code.txt
exit "$runner_rc"
"""


def _build_standalone_remote_script(args: argparse.Namespace, *, use_precise_phases: bool) -> str:
    remote_memory_csv = f"{args.remote_root}/android_memory_timeline.csv"
    if _standalone_completion_mode(args):
        llm_cmd = _build_text_only_command(args)
    else:
        llm_cmd = _build_standalone_command(args, use_precise_phases=use_precise_phases)
    baseline_loop = _baseline_sampling_shell(remote_memory_csv, args.sample_interval, args.baseline_window)
    memory_loop = _memory_sampling_shell(
        remote_memory_csv,
        args.sample_interval,
        'kill -0 "$runner_pid" 2>/dev/null',
        '"$runner_pid"',
    )
    env_exports = _remote_llama_env_exports(args)
    return f"""#!/system/bin/sh
cd {shlex.quote(args.remote_root)} || exit 1
{env_exports}
rm -f foundation_output.txt foundation_exit_code.txt android_memory_timeline.csv foundation_phase_stats.csv foundation_token_io.txt foundation_inference_tokens.txt
printf '%s\\n' '{_memory_csv_header()}' > {shlex.quote(remote_memory_csv)}

{baseline_loop}

( {_shell_join(llm_cmd)} > foundation_output.txt 2>&1; echo $? > foundation_exit_code.txt ) &
runner_pid=$!

{memory_loop}

wait "$runner_pid"
runner_wait_rc=$?
exit_code=$(cat foundation_exit_code.txt 2>/dev/null)
exit "${{exit_code:-$runner_wait_rc}}"
"""


def _write_and_run_remote_script(adb: list[str], remote_root: str, script_text: str) -> subprocess.CompletedProcess[str]:
    remote_script = f"{remote_root}/run_android_processor.sh"
    with tempfile.TemporaryDirectory(prefix="streamingvlm_android_script_") as tmp_script_dir:
        script_path = Path(tmp_script_dir) / "run_android_processor.sh"
        script_path.write_text(script_text, encoding="utf-8")
        _push(adb, script_path, remote_root)
    _run(adb + ["shell", f"chmod +x {shlex.quote(remote_script)}"])
    return subprocess.run(
        adb + ["shell", f"sh {shlex.quote(remote_script)}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _pull_outputs(adb: list[str], remote_root: str, result_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        if "*" in name or "?" in name or "[" in name:
            listing = subprocess.run(
                adb + ["shell", f"for f in {shlex.quote(remote_root)}/{name}; do [ -f \"$f\" ] && basename \"$f\"; done"],
                check=False,
                text=True,
                capture_output=True,
            )
            if listing.returncode != 0:
                continue
            for remote_name in listing.stdout.splitlines():
                remote_name = remote_name.strip()
                if not remote_name:
                    continue
                local = result_dir / remote_name
                local.unlink(missing_ok=True)
                _pull_if_exists(adb, f"{remote_root}/{remote_name}", local)
            continue
        local = result_dir / name
        local.unlink(missing_ok=True)
        _pull_if_exists(adb, f"{remote_root}/{name}", local)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified llama.cpp / hybrid Android VLM processors.")
    parser.add_argument("--processor", choices=("cpu", "gpu", "hybrid"), required=True)
    parser.add_argument("--serial", default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Foundation QNN manifest with vision_encoder_pte. Hybrid may use this or --vision.",
    )
    parser.add_argument(
        "--vision",
        "--vision-encoder",
        dest="vision_encoder",
        type=Path,
        default=None,
        help="Direct ExecuTorch/QNN vision encoder PTE for --processor hybrid. Overrides --manifest.",
    )
    parser.add_argument("--vision-build-dir", type=Path, default=None, help="CMake build dir containing hybrid_vision_dump. Defaults to --llama-build-dir.")
    parser.add_argument(
        "--executorch-build-dir",
        type=Path,
        default=WORKSPACE / "executorch" / "build-android-unified",
        help="ExecuTorch Android build dir containing QNN runtime libraries.",
    )
    parser.add_argument("--llama-build-dir", "--build-dir", dest="llama_build_dir", type=Path, required=True, help="Android build dir containing llama.cpp and overlay binaries.")
    parser.add_argument("--model", type=Path, required=True, help="Text GGUF model.")
    parser.add_argument(
        "--mmproj",
        type=Path,
        default=None,
        help="Mmproj GGUF for multimodal (--image given). Omitted when --image is not passed (cpu/gpu text-only). Required for --processor hybrid.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image for vision. Omit on cpu/gpu for text-only generation (llama-completion). Mutually exclusive with --video.",
    )
    parser.add_argument(
        "--warmup-image",
        type=Path,
        default=DEFAULT_WARMUP_IMAGE,
        help="Fixed image used for bridge-local vision/mmproj warmup before measured image or video inference.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Input video for vision. Sampled as multiple image frames; mutually exclusive with --image.",
    )
    parser.add_argument(
        "--streaming-video",
        "--streaming_video",
        dest="streaming_video",
        type=Path,
        default=None,
        help="Input video for streaming simulation. Sampled by --sampling-fps; mutually exclusive with --image/--video.",
    )
    parser.add_argument(
        "--single-buffer",
        "--single_buffer",
        dest="single_buffer",
        action="store_true",
        help="Streaming mode: keep only the latest sampled frame as an image buffer and answer prompt events with that frame.",
    )
    parser.add_argument(
        "--stream-mode",
        "--stream_mode",
        dest="stream_mode",
        default=None,
        choices=("single-buffer", "sliding-window", "vision-prefill"),
        help="Streaming strategy for --streaming-video. Defaults to single-buffer; --single-buffer remains an alias.",
    )
    parser.add_argument("--num-segments", type=int, default=8, help="Uniform temporal samples for --video.")
    parser.add_argument("--sampling-fps", "--sampling_fps", dest="sampling_fps", type=float, default=None, help="Frame sampling FPS for --streaming-video.")
    parser.add_argument("--max-video-time", "--max_video_time", dest="max_video_time", type=float, default=None, help="Optional maximum streaming-video duration to sample, in seconds.")
    parser.add_argument("--window-sec", "--window_sec", dest="window_sec", type=float, default=None, help="Prompt-time lookback window in seconds for sliding-window streaming. vision-prefill ignores this and caches full history.")
    parser.add_argument("--window-max-frames", "--window_max_frames", dest="window_max_frames", type=int, default=8, help="Maximum sampled frames used by one sliding-window prompt. vision-prefill ignores this and caches full history.")
    parser.add_argument("--time", default=None, help="JSON list of prompt timestamps for --streaming-video, e.g. '[5.0, 10.0]'.")
    parser.add_argument("--max-num", type=int, default=1, help="Max InternVL dynamic-preprocess tiles per sampled video frame.")
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--n-predict", "--max-new-tokens", dest="n_predict", type=int, default=32)
    parser.add_argument(
        "--force-generation",
        "--force_generation",
        dest="force_generation",
        type=int,
        default=None,
        help="Generate exactly this many tokens by continuing through EOS/EOG where supported.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument(
        "--dynamic-kv-cache",
        action="store_true",
        dest="dynamic_kv_cache",
        help="Project prototype: use model max context as logical limit and grow physical KV on demand.",
    )
    parser.add_argument("--kv-init-size", type=int, default=1024, help="Initial physical KV capacity for --dynamic-kv-cache.")
    parser.add_argument("--kv-grow-step", type=int, default=1024, help="Physical KV grow step for --dynamic-kv-cache.")
    parser.add_argument("--paged-kv-cache", action="store_true", dest="paged_kv_cache", help="Project prototype: enable experimental paged KV cache metadata/page-table mode.")
    parser.add_argument("--kv-page-size", type=int, default=256, help="Page size in cells for --paged-kv-cache.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", "--n-gpu-layers", dest="gpu_layers", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None, help="llama.cpp device, e.g. GPUOpenCL.")
    parser.add_argument(
        "--cache-type-k",
        "--cache_type_k",
        dest="cache_type_k",
        default=None,
        metavar="TYPE",
        help="KV-cache dtype for K (llama.cpp --cache-type-k). Multimodal: opencl_phase_mtmd / llama-mtmd-cli; text-only (no --image): llama-completion.",
    )
    parser.add_argument(
        "--cache-type-v",
        "--cache_type_v",
        dest="cache_type_v",
        default=None,
        metavar="TYPE",
        help="KV-cache dtype for V (llama.cpp --cache-type-v), e.g. q8_0.",
    )
    parser.add_argument(
        "--fit",
        choices=("on", "off"),
        default=None,
        help="llama.cpp memory fit (passthrough --fit). Omit by default; use off to skip common_fit_params (OpenCL SET_ROWS abort workaround).",
    )
    parser.add_argument(
        "--flash-attn",
        "-fa",
        choices=("on", "off", "auto"),
        default=None,
        dest="flash_attn",
        metavar="MODE",
        help="llama.cpp --flash-attn (-fa). Omit to use binary default.",
    )
    parser.add_argument(
        "--no-kv-offload",
        action="store_true",
        dest="no_kv_offload",
        help="Pass --no-kv-offload (-nkvo) to llama (standalone GPU/CPU and hybrid_decode).",
    )
    parser.add_argument(
        "--disable-attn-kv-rotation",
        action="store_true",
        dest="disable_attn_kv_rotation",
        help="On device: export LLAMA_ATTN_ROT_DISABLE=1 before llama (see llama-kv-cache.cpp).",
    )
    parser.set_defaults(no_warmup=True)
    parser.add_argument(
        "--warmup",
        action="store_false",
        dest="no_warmup",
        help="Pass --warmup to llama (enable empty warmup before generation). Off by default for bridge benchmarks.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        dest="no_warmup",
        help="Skip empty warmup (default for this script).",
    )
    parser.add_argument(
        "--rope-scaling",
        choices=("none", "linear", "yarn"),
        default=None,
        dest="rope_scaling",
        help="llama.cpp --rope-scaling (HF rope_scaling.rope_type yarn -> yarn).",
    )
    parser.add_argument(
        "--rope-scale",
        type=float,
        default=None,
        dest="rope_scale",
        metavar="N",
        help="llama.cpp --rope-scale (HF YaRN factor maps here when extending context).",
    )
    parser.add_argument(
        "--rope-freq-base",
        type=float,
        default=None,
        dest="rope_freq_base",
        metavar="N",
        help="llama.cpp --rope-freq-base.",
    )
    parser.add_argument(
        "--rope-freq-scale",
        type=float,
        default=None,
        dest="rope_freq_scale",
        metavar="N",
        help="llama.cpp --rope-freq-scale.",
    )
    parser.add_argument(
        "--yarn-orig-ctx",
        type=int,
        default=None,
        dest="yarn_orig_ctx",
        metavar="N",
        help="llama.cpp --yarn-orig-ctx (HF original_max_position_embeddings).",
    )
    parser.add_argument(
        "--yarn-ext-factor",
        type=float,
        default=None,
        dest="yarn_ext_factor",
        metavar="N",
        help="llama.cpp --yarn-ext-factor.",
    )
    parser.add_argument(
        "--yarn-attn-factor",
        type=float,
        default=None,
        dest="yarn_attn_factor",
        metavar="N",
        help="llama.cpp --yarn-attn-factor.",
    )
    parser.add_argument(
        "--yarn-beta-slow",
        type=float,
        default=None,
        dest="yarn_beta_slow",
        metavar="N",
        help="llama.cpp --yarn-beta-slow.",
    )
    parser.add_argument(
        "--yarn-beta-fast",
        type=float,
        default=None,
        dest="yarn_beta_fast",
        metavar="N",
        help="llama.cpp --yarn-beta-fast.",
    )
    parser.add_argument(
        "--opencl-lib",
        type=Path,
        default=WORKSPACE / "third_party" / "OpenCL-ICD-Loader" / "build-android" / "libOpenCL.so",
    )
    parser.add_argument(
        "--push-opencl-loader",
        action="store_true",
        help="Push local libOpenCL.so. Disabled by default because the system Qualcomm OpenCL loader works better on the tested device.",
    )
    parser.add_argument("--soc-model", default="SM8750")
    parser.add_argument("--remote-root", "--device-workdir", dest="remote_root", default="/data/local/tmp/streamingvlm_vlm")
    parser.add_argument("--results-root", type=Path, default=FOUNDATION_LLAMA / "results" / "log")
    parser.add_argument("--sample-interval", type=float, default=0.05, help="Android /proc/meminfo sampling interval in seconds.")
    parser.add_argument(
        "--baseline-window",
        "--baseline_window",
        dest="baseline_window",
        type=float,
        default=5.0,
        help="Seconds of pre-run MemAvailable samples to average for memory_usage_summary.txt.",
    )
    parser.add_argument("--force-push", action="store_true", help="Clear the remote workdir before pushing.")
    parser.add_argument("--model-push", "--model_push", dest="model_push", action="store_true", help="Force pushing model files even if they already exist on device.")
    args = parser.parse_args()

    args.remote_root = args.remote_root.rstrip("/")
    args.baseline_window = max(args.baseline_window, 0.0)
    args.model = args.model.resolve()
    args.warmup_image = args.warmup_image.resolve()
    if args.vision_encoder is not None:
        args.vision_encoder = args.vision_encoder.resolve()
    args.llama_build_dir = args.llama_build_dir.resolve()
    if args.num_segments <= 0:
        raise SystemExit("--num-segments must be positive.")
    if args.max_num <= 0:
        raise SystemExit("--max-num must be positive.")
    if args.max_video_time is not None and args.max_video_time <= 0:
        raise SystemExit("--max-video-time must be positive when provided.")
    if args.window_sec is not None and args.window_sec <= 0:
        raise SystemExit("--window-sec must be positive when provided.")
    if args.window_max_frames <= 0:
        raise SystemExit("--window-max-frames must be positive.")
    selected_media_count = sum(x is not None for x in (args.image, args.video, args.streaming_video))
    if selected_media_count > 1:
        raise SystemExit("--image, --video, and --streaming-video are mutually exclusive.")
    if args.streaming_video is not None:
        if args.sampling_fps is None or args.sampling_fps <= 0:
            raise SystemExit("--streaming-video requires positive --sampling-fps.")
        try:
            args.stream_mode = normalize_stream_mode(args.stream_mode, single_buffer=args.single_buffer)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        args.single_buffer = args.stream_mode == "single_buffer"
        args.prompt_events = _parse_streaming_prompt_events(args.time, args.prompt)
    else:
        args.prompt_events = []
        args.stream_mode = normalize_stream_mode(args.stream_mode, single_buffer=args.single_buffer)
    args.media_mode = media_mode_from_args(args)
    args.backend_mode = backend_mode_from_processor(args.processor)
    if args.vision_build_dir is None:
        args.vision_build_dir = args.llama_build_dir
    else:
        args.vision_build_dir = args.vision_build_dir.resolve()

    if args.processor == "hybrid":
        if args.image is None and args.video is None and args.streaming_video is None:
            raise SystemExit("--processor hybrid requires --image, --video, or --streaming-video.")
        if args.mmproj is None:
            raise SystemExit("--processor hybrid requires --mmproj.")
        args.mmproj = args.mmproj.resolve()
        if args.image is not None:
            args.image = args.image.resolve()
        if args.video is not None:
            args.video = args.video.resolve()
        if args.streaming_video is not None:
            args.streaming_video = args.streaming_video.resolve()
    elif _standalone_completion_mode(args):
        args.mmproj = None
        args.image = None
        args.video = None
    else:
        if args.mmproj is None:
            raise SystemExit(
                "--mmproj is required when --image/--video is set. Omit media on cpu/gpu for text-only (llama-completion)."
            )
        args.mmproj = args.mmproj.resolve()
        if args.image is not None:
            args.image = args.image.resolve()
        if args.video is not None:
            args.video = args.video.resolve()
        if args.streaming_video is not None and args.processor != "gpu":
            raise SystemExit("--streaming-video is currently supported only with --processor gpu or --processor hybrid.")

    exist_paths: list[Path] = [args.llama_build_dir, args.model]
    if args.mmproj is not None:
        exist_paths.append(args.mmproj)
    if args.image is not None:
        exist_paths.append(args.image)
    if args.video is not None:
        exist_paths.append(args.video)
    if args.streaming_video is not None:
        exist_paths.append(args.streaming_video)
    if args.mmproj is not None:
        exist_paths.append(args.warmup_image)
    if args.vision_encoder is not None:
        exist_paths.append(args.vision_encoder)
    for required in exist_paths:
        if not required.exists():
            raise SystemExit(f"Missing required path: {required}")
    if (
        args.processor == "hybrid"
        and args.manifest is None
        and args.vision_encoder is None
    ):
        raise SystemExit("--processor hybrid requires either --manifest or --vision.")

    qnn_sdk = os.environ.get("QNN_SDK_ROOT")
    if args.processor == "hybrid" and not qnn_sdk:
        raise SystemExit("QNN_SDK_ROOT is required when --processor hybrid.")

    adb = _adb(args.serial)
    if args.force_push:
        _run(adb + ["shell", "rm", "-rf", args.remote_root])
    _run(adb + ["shell", "mkdir", "-p", args.remote_root])

    result_dir = args.results_root / _result_model_name(
        args.model,
        args.processor,
        args.ctx_size,
        args.cache_type_k,
        args.cache_type_v,
        text_only=_standalone_completion_mode(args),
        streaming=args.streaming_video is not None,
        stream_mode=getattr(args, "stream_mode", None),
        dynamic_kv_cache=getattr(args, "dynamic_kv_cache", False),
        paged_kv_cache=getattr(args, "paged_kv_cache", False),
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    _push_llama_runtime(
        adb,
        args.remote_root,
        args.llama_build_dir,
        args.opencl_lib,
        args.push_opencl_loader,
    )
    _push_model_cached(adb, args.model, args.remote_root, force=args.model_push)
    if args.mmproj is not None:
        _push_model_cached(adb, args.mmproj, args.remote_root, force=args.model_push)
    if args.image is not None:
        _push(adb, args.image, args.remote_root)

    encoder_pte: Path | None = None
    with tempfile.TemporaryDirectory(prefix="streamingvlm_android_inputs_") as tmp:
        media: PreparedMedia | None = None
        if args.processor == "hybrid":
            streaming_mode = args.streaming_video is not None
            if args.vision_encoder is not None:
                encoder_pte = args.vision_encoder
            else:
                manifest = _load_manifest(args.manifest.resolve())
                encoder_pte = Path(manifest["paths"]["vision_encoder_pte"])
            args.remote_encoder_pte = encoder_pte.name
            vision_bin = _find_executable(args.vision_build_dir, "hybrid_vision_dump")
            decode_bin = _find_executable(args.llama_build_dir, "hybrid_decode")
            streaming_bin = _find_executable(args.vision_build_dir, "hybrid_streaming_decode")
            if streaming_mode:
                if not streaming_bin.exists():
                    raise SystemExit("Missing hybrid_streaming_decode. Build hybrid_bridge first.")
                if not vision_bin.exists():
                    raise SystemExit("Missing hybrid_vision_dump. Build hybrid_bridge with ExecuTorch/QNN support first.")
            elif not vision_bin.exists() or not decode_bin.exists():
                raise SystemExit("Missing hybrid_vision_dump or hybrid_decode. Build hybrid_bridge first.")
            media = _prepare_media(args, Path(tmp))
            warmup_bin, warmup_layout = prepare_warmup_image(args.warmup_image, Path(tmp))
            args.remote_warmup_bin = warmup_bin.name
            args.remote_warmup_layout_image = warmup_layout.name
            args.remote_warmup_embedding = "warmup_vision_embedding.svlmemb"
            if streaming_mode:
                _push(adb, streaming_bin, args.remote_root)
            else:
                _push(adb, vision_bin, args.remote_root)
                _push(adb, decode_bin, args.remote_root)
            if encoder_pte is not None:
                _push_model_cached(adb, encoder_pte, args.remote_root, force=args.model_push)
            _push(adb, media.metadata_path, args.remote_root)
            _push(adb, warmup_bin, args.remote_root)
            if not streaming_mode:
                _push(adb, warmup_layout, args.remote_root)
            for frame_bin in media.frame_bins:
                _push(adb, frame_bin, args.remote_root)
            pushed_layouts: set[Path] = set()
            for layout_image in media.layout_images:
                resolved_layout = layout_image.resolve()
                if args.image is not None and resolved_layout == args.image.resolve():
                    continue
                if resolved_layout in pushed_layouts:
                    continue
                _push(adb, layout_image, args.remote_root)
                pushed_layouts.add(resolved_layout)
            _push_qnn_libs(adb, args.remote_root, Path(qnn_sdk), args.executorch_build_dir, args.soc_model)
            if streaming_mode:
                _run(adb + ["shell", f"chmod +x {shlex.quote(args.remote_root)}/hybrid_streaming_decode {shlex.quote(args.remote_root)}/opencl_phase_mtmd 2>/dev/null || true"])
                script_text = _build_hybrid_streaming_remote_script(args)
                pull_names = HYBRID_STREAMING_PULL_ARTIFACTS
            else:
                _run(adb + ["shell", f"chmod +x {shlex.quote(args.remote_root)}/hybrid_vision_dump {shlex.quote(args.remote_root)}/hybrid_decode"])
                script_text = _build_hybrid_remote_script(args, encoder_pte=encoder_pte, media=media)
                pull_names = HYBRID_PULL_ARTIFACTS
        else:
            completion_mode = _standalone_completion_mode(args)
            streaming_mode = args.streaming_video is not None
            if not completion_mode:
                media = _prepare_media(args, Path(tmp))
                _, warmup_layout = prepare_warmup_image(args.warmup_image, Path(tmp))
                args.remote_warmup_layout_image = warmup_layout.name
                args.remote_prompt = media.prompt
                args.remote_layout_images = [path.name for path in media.layout_images]
                _push(adb, media.metadata_path, args.remote_root)
                _push(adb, warmup_layout, args.remote_root)
                for layout_image in media.layout_images:
                    _push(adb, layout_image, args.remote_root)
            if completion_mode:
                completion_bin = _find_executable(args.llama_build_dir, "llama-completion")
                if not completion_bin.exists():
                    raise SystemExit(
                        "Missing llama-completion. Build llama.cpp with the completion tool target (see tools/completion)."
                    )
                chmod_targets = "llama-completion"
                use_precise_phases = False
            elif streaming_mode:
                if args.processor != "gpu":
                    raise SystemExit("--streaming-video is supported only with --processor gpu or --processor hybrid.")
                streaming_bin = _find_executable(args.llama_build_dir, "opencl_streaming_decode")
                if not streaming_bin.exists():
                    raise SystemExit("Missing opencl_streaming_decode. Build hybrid_bridge first.")
                _push(adb, streaming_bin, args.remote_root)
                chmod_targets = "opencl_streaming_decode opencl_phase_mtmd"
                script_text = _build_hybrid_streaming_remote_script(args)
                pull_names = HYBRID_STREAMING_PULL_ARTIFACTS
                _run(adb + ["shell", f"cd {shlex.quote(args.remote_root)} && chmod +x {chmod_targets} 2>/dev/null || true"])
                run_res = _write_and_run_remote_script(adb, args.remote_root, script_text)
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "host_adb_output.txt").write_text(run_res.stdout, encoding="utf-8")
                _pull_outputs(adb, args.remote_root, result_dir, pull_names)
                if not (result_dir / "foundation_exit_code.txt").exists():
                    (result_dir / "foundation_exit_code.txt").write_text(str(run_res.returncode), encoding="utf-8")
                _finalize_hybrid_streaming_outputs(result_dir)
                return run_res.returncode
            else:
                llama_cli = _find_executable(args.llama_build_dir, "llama-mtmd-cli")
                use_precise_phases = args.processor == "gpu" and _find_executable(args.llama_build_dir, "opencl_phase_mtmd").exists()
                if not use_precise_phases and not llama_cli.exists():
                    raise SystemExit("Missing llama-mtmd-cli. Build llama.cpp first.")
                chmod_targets = "llama-mtmd-cli"
                if use_precise_phases:
                    chmod_targets += " opencl_phase_mtmd"
            _run(adb + ["shell", f"cd {shlex.quote(args.remote_root)} && chmod +x {chmod_targets} 2>/dev/null || true"])
            script_text = _build_standalone_remote_script(args, use_precise_phases=use_precise_phases)
            pull_names = STANDALONE_PULL_ARTIFACTS

        run_res = _write_and_run_remote_script(adb, args.remote_root, script_text)

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "host_adb_output.txt").write_text(run_res.stdout, encoding="utf-8")
    _pull_outputs(adb, args.remote_root, result_dir, pull_names)
    if not (result_dir / "foundation_exit_code.txt").exists():
        (result_dir / "foundation_exit_code.txt").write_text(str(run_res.returncode), encoding="utf-8")
    return_code = (result_dir / "foundation_exit_code.txt").read_text(encoding="utf-8").strip()

    if args.processor == "hybrid":
        if args.streaming_video is not None:
            _finalize_hybrid_streaming_outputs(result_dir)
        else:
            _finalize_hybrid_outputs(result_dir)
        (result_dir / "hybrid_decode_stdout.txt").unlink(missing_ok=True)
    else:
        _finalize_standalone_outputs(
            result_dir,
            processor=args.processor,
            return_code=return_code,
            prompt=args.prompt,
            text_only=_standalone_completion_mode(args),
        )
    return run_res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
