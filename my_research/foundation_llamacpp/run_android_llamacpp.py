#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import time
from pathlib import Path


def _run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _adb_base(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _remote_exists(adb: list[str], remote_path: str) -> bool:
    result = subprocess.run(
        adb + ["shell", f"test -f {shlex.quote(remote_path)}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


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


def _parse_proc_kb_value(text: str, key: str) -> int:
    for line in text.splitlines():
        if not line.startswith(f"{key}:"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
    return 0


def _remote_memory_snapshot(adb: list[str], pid: str) -> dict[str, int | str]:
    result = subprocess.run(
        adb
        + [
            "shell",
            (
                "cat "
                f"/proc/{shlex.quote(pid)}/status "
                f"/proc/{shlex.quote(pid)}/smaps_rollup "
                "/proc/meminfo 2>/dev/null"
            ),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0 or not result.stdout:
        return {"pid_alive": 0, "vmrss_kb": 0, "vmsize_kb": 0, "vmhwm_kb": 0}

    text = result.stdout
    values: dict[str, int | str] = {
        "pid_alive": 1,
        "vmrss_kb": _parse_proc_kb_value(text, "VmRSS"),
        "vmsize_kb": _parse_proc_kb_value(text, "VmSize"),
        "vmhwm_kb": _parse_proc_kb_value(text, "VmHWM"),
        "smaps_rss_kb": _parse_proc_kb_value(text, "Rss"),
        "smaps_pss_kb": _parse_proc_kb_value(text, "Pss"),
        "smaps_private_dirty_kb": _parse_proc_kb_value(text, "Private_Dirty"),
        "smaps_shared_clean_kb": _parse_proc_kb_value(text, "Shared_Clean"),
        "mem_available_kb": _parse_proc_kb_value(text, "MemAvailable"),
        "cached_kb": _parse_proc_kb_value(text, "Cached"),
        "dma_heap_pool_kb": _parse_proc_kb_value(text, "DmaHeapPool"),
        "gpu_total_kb": _parse_proc_kb_value(text, "GpuTotal"),
        "kgsl_shmem_usage_kb": _parse_proc_kb_value(text, "KgslShmemUsage"),
    }
    return values


def _push_runtime_files(adb: list[str], build_dir: Path, remote_root: str) -> None:
    bin_dir = build_dir / "bin"
    lib_dirs = [build_dir / "lib", bin_dir]
    # Do not push the local OpenCL ICD loader by default. On the tested Qualcomm
    # device, the system OpenCL loader discovers Adreno correctly while the local
    # loader can make ggml_opencl report "platform IDs not available".
    extra_libs: list[Path] = []

    _run(adb + ["shell", f"mkdir -p {shlex.quote(remote_root)}"])
    for path in sorted(bin_dir.iterdir()):
        if path.is_file() and path.name == "llama-mtmd-cli":
            _run(adb + ["push", str(path), f"{remote_root}/{path.name}"])
    for path in (build_dir / "opencl_phase_mtmd", bin_dir / "opencl_phase_mtmd"):
        if path.exists() and path.is_file():
            _run(adb + ["push", str(path), f"{remote_root}/{path.name}"])
            break
    for lib_dir in lib_dirs:
        if not lib_dir.exists():
            continue
        for path in sorted(lib_dir.iterdir()):
            if path.is_file() and path.suffix == ".so":
                _run(adb + ["push", str(path), f"{remote_root}/{path.name}"])
    for pattern in ("libggml-htp-v*.so", "libc++_shared.so"):
        for path in sorted(build_dir.rglob(pattern)):
            if path.is_file():
                _run(adb + ["push", str(path), f"{remote_root}/{path.name}"])
    for path in extra_libs:
        if path.exists():
            _run(adb + ["push", str(path), f"{remote_root}/{path.name}"])
    _run(
        adb
        + [
            "shell",
            f"chmod +x {shlex.quote(remote_root)}/llama-mtmd-cli {shlex.quote(remote_root)}/opencl_phase_mtmd 2>/dev/null || true",
        ]
    )


def _result_dir(results_root: Path, backend: str, model_name: str) -> Path:
    out_dir = results_root / backend / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _phase_float(row: dict[str, str], key: str) -> float:
    value = (row.get(key) or "").strip()
    return float(value) if value else 0.0


def _read_phase_rows(path: Path) -> list[dict[str, str]]:
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
            row["elapsed_s_start"] = f"{_phase_float(row, 'elapsed_s_start'):.6f}"
            row["elapsed_s_end"] = f"{_phase_float(row, 'elapsed_s_end'):.6f}"
            rows.append(row)
    return sorted(rows, key=lambda row: (_phase_float(row, "elapsed_s_start"), _phase_float(row, "elapsed_s_end")))


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
            "row_type": "# L_DecoderRuntimeInit: llama.cpp args/OpenCL runtime init  "
            "L_DecoderLoad: llama.cpp model/mmproj load  ImageLoad: input image load  "
            "LayoutTokenize: mtmd layout  V_Encode: OpenCL vision encode/projector  "
            "ImagePrefill: image embedding prefill  T_Prefill: text prompt prefill  "
            "D: one generated-token decode"
        })
        writer.writerows(rows)


def _phase_colors() -> dict[str, str]:
    return {
        "L_DecoderRuntimeInit": "#a29bfe",
        "L_DecoderLoad": "#6c5ce7",
        "ImageLoad": "#74b9ff",
        "LayoutTokenize": "#fdcb6e",
        "V_Encode": "#00b894",
        "ImagePrefill": "#0984e3",
        "T_Prefill": "#e17055",
        "EmbeddingFileWrite": "#55efc4",
        "ExternalEmbeddingRead": "#00cec9",
        "D": "#ff7675",
        "Decode": "#d63031",
    }


def _parse_log_summary(log_text: str) -> dict[str, object]:
    patterns: list[tuple[str, str]] = [
        ("image_slice_encoded_ms", r"image slice encoded in ([0-9.]+) ms"),
        ("image_decoded_ms", r"image decoded(?: \(batch \d+/\d+\))? in ([0-9.]+) ms"),
        ("prompt_eval_tokens", r"prompt eval: ([0-9]+) tokens, ([0-9.]+) tok/s"),
        ("decode_eval_runs", r"decode eval: ([0-9]+) runs, ([0-9.]+) tok/s"),
        ("total_time_ms", r"total time: ([0-9.]+) ms"),
    ]
    summary: dict[str, object] = {}
    for key, pattern in patterns:
        match = re.search(pattern, log_text)
        if not match:
            continue
        if key == "prompt_eval_tokens":
            summary[key] = int(match.group(1))
            summary["prompt_eval_tok_s"] = float(match.group(2))
        elif key == "decode_eval_runs":
            summary[key] = int(match.group(1))
            summary["decode_eval_tok_s"] = float(match.group(2))
        else:
            value = match.group(1)
            summary[key] = float(value) if "." in value else int(value)

    for key, pattern in [
        ("model_buffer_mib", r"model buffer size =\s+([0-9.]+) MiB"),
        ("context_buffer_mib", r"KV buffer size =\s+([0-9.]+) MiB"),
        ("compute_buffer_mib", r"compute buffer size =\s+([0-9.]+) MiB"),
    ]:
        matches = re.findall(pattern, log_text)
        if matches:
            summary[key] = float(matches[-1])
    return summary


def _parse_perf_summary(log_text: str) -> dict[str, float]:
    patterns = {
        "load_time_ms": r"llama_perf_context_print:\s+load time =\s+([0-9.]+) ms",
        "prompt_eval_time_ms": r"llama_perf_context_print:\s+prompt eval time =\s+([0-9.]+) ms",
        "decode_eval_time_ms": r"llama_perf_context_print:\s+eval time =\s+([0-9.]+) ms",
        "total_time_ms": r"llama_perf_context_print:\s+total time =\s+([0-9.]+) ms",
        "image_slice_encoded_ms": r"image slice encoded in ([0-9.]+) ms",
        "image_decoded_ms": r"image decoded(?: \(batch \d+/\d+\))? in ([0-9.]+) ms",
    }
    perf: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if match:
            perf[key] = float(match.group(1))
    return perf


def _write_png_memory_timeline(
    output_dir: Path,
    memory_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rows = [
        row
        for row in memory_rows
        if row.get("elapsed_s") is not None and row.get("mem_available_kb") not in ("", None)
    ]
    output_png = output_dir / "memory_timeline_plot.png"

    if not rows:
        fallback_available = float(summary.get("mem_available_mib", 0.0) or 0.0)
        if fallback_available <= 0:
            return None

        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
        ax.bar(["memory"], [fallback_available], color="#0984e3", edgecolor="white")
        ax.text(
            0,
            fallback_available / 2.0,
            f"MemAvailable\n{fallback_available:.1f} MiB",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

        ax.set_title(f"Memory Timeline: {output_dir.name}")
        ax.set_ylabel("Memory (MiB)")
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_png, bbox_inches="tight")
        plt.close(fig)
        return output_png

    xs = [float(row["elapsed_s"]) for row in rows]
    mem_available = [
        float(row.get("mem_available_kb", 0) or 0) / 1024.0 for row in rows
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.plot(xs, mem_available, label="MemAvailable (MiB)", linewidth=2.2, color="#0984e3")

    ax.set_title(f"Android Memory Timeline: {output_dir.name}")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return output_png


def _write_png_phase_duration(output_dir: Path, perf: dict[str, float]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    phases = [
        ("Load", perf.get("load_time_ms", 0.0)),
        ("Image Encode", perf.get("image_slice_encoded_ms", 0.0)),
        ("Image Decode", perf.get("image_decoded_ms", 0.0)),
        ("Prompt Eval", perf.get("prompt_eval_time_ms", 0.0)),
        ("Token Decode", perf.get("decode_eval_time_ms", 0.0)),
    ]
    phases = [(name, value) for name, value in phases if value > 0]
    if not phases:
        return None

    total = sum(value for _, value in phases)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    bottom = 0.0
    colors = ["#6c5ce7", "#00b894", "#0984e3", "#e17055", "#d63031"]
    for idx, (name, value) in enumerate(phases):
        color = colors[idx % len(colors)]
        ax.bar(["total"], [value], bottom=bottom, color=color, edgecolor="white")
        if value / total >= 0.06:
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
    ax.set_ylim(0, max(total, 1.0))
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[idx % len(colors)])
        for idx in range(len(phases))
    ]
    labels = [f"{name}: {value / 1000.0:.2f}s" for name, value in phases]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()
    output_png = output_dir / "phase_duration_stacked_bar.png"
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return output_png


def _write_png_phase_duration_from_rows(output_dir: Path, phase_rows: list[dict[str, str]]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    excluded_from_plot = {
        "ImageLoad",
        "L_DecoderLoad",
        "L_DecoderRuntimeInit",
        "LayoutTokenize",
    }
    durations: dict[str, float] = {}
    first_start_s: dict[str, float] = {}
    has_decode_summary = any(row.get("row_type") == "Decode" for row in phase_rows)
    for row in phase_rows:
        name = row.get("row_type", "")
        if name in excluded_from_plot:
            continue
        if name == "D" and has_decode_summary:
            continue
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
        return None

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
    output_png = output_dir / "phase_duration_stacked_bar.png"
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return output_png


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run llama.cpp Android VLM smoke tests and store results in the foundation layout."
    )
    parser.add_argument("--backend", choices=("cpu", "vulkan", "opencl", "hexagon"), required=True)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Result folder name. Defaults to the GGUF stem, e.g. InternVL3-1B-Instruct-Q8_0.",
    )
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="GGUF model file")
    parser.add_argument("--mmproj", required=True, type=Path, help="Multimodal projector GGUF file")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="Layers to offload. Defaults to 0 for CPU and 99 for accelerator backends.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="llama.cpp --device value. Defaults to none for CPU and HTP0 for Hexagon.",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=None,
        help="Optional llama.cpp context size, e.g. 8192 for Snapdragon examples.",
    )
    parser.add_argument(
        "--ubatch-size",
        type=int,
        default=None,
        help="Optional llama.cpp ubatch size, e.g. 256 for Snapdragon examples.",
    )
    parser.add_argument(
        "--hexagon-ndev",
        type=int,
        default=None,
        help="Optional GGML_HEXAGON_NDEV session count for HTP multi-device runs.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--serial", default=None, help="adb device serial")
    parser.add_argument(
        "--device-workdir",
        default="/data/local/tmp/llama-vlm",
        help="Remote working directory on the Android device.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspace/streamingvlm/my_research/foundation_llamacpp/results/log"),
        help="Local root for result folders.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.05,
        help="Seconds between remote memory samples while the model runs.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    adb = _adb_base(args.serial)
    build_dir: Path = args.build_dir.resolve()
    bin_dir = build_dir / "bin"
    model = args.model.resolve()
    mmproj = args.mmproj.resolve()
    image = args.image.resolve()
    model_name = args.model_name or model.stem

    for required in (build_dir, bin_dir, model, mmproj, image):
        if not required.exists():
            raise SystemExit(f"Missing required path: {required}")

    remote_root = args.device_workdir.rstrip("/")
    remote_model = f"{remote_root}/{model.name}"
    remote_mmproj = f"{remote_root}/{mmproj.name}"
    remote_image = f"{remote_root}/{image.name}"
    remote_output = f"{remote_root}/foundation_output.txt"
    remote_exit_code = f"{remote_root}/foundation_exit_code.txt"
    remote_memory_csv = f"{remote_root}/android_memory_timeline.csv"
    remote_phase_csv = f"{remote_root}/foundation_phase_stats.csv"

    _push_runtime_files(adb, build_dir, remote_root)
    _run(adb + ["push", str(model), remote_model])
    _run(adb + ["push", str(mmproj), remote_mmproj])
    _run(adb + ["push", str(image), remote_image])

    runner_cmd = [
        f"cd {shlex.quote(remote_root)}",
        "export LD_LIBRARY_PATH=.",
        "export ADSP_LIBRARY_PATH=.",
        f"rm -f {shlex.quote(remote_output)} {shlex.quote(remote_exit_code)}",
    ]
    selected_n_gpu_layers = args.n_gpu_layers
    if selected_n_gpu_layers is None:
        selected_n_gpu_layers = 0 if args.backend == "cpu" else 99

    precise_opencl_bin = build_dir / "opencl_phase_mtmd"
    if not precise_opencl_bin.exists():
        precise_opencl_bin = bin_dir / "opencl_phase_mtmd"
    use_precise_phases = args.backend == "opencl" and precise_opencl_bin.exists()

    llm_cmd = [
        "./opencl_phase_mtmd" if use_precise_phases else "./llama-mtmd-cli",
        "-m",
        remote_model,
        "--mmproj",
        remote_mmproj,
        "--image",
        remote_image,
        "-p",
        args.prompt,
        "-n",
        str(args.max_new_tokens),
        "-t",
        str(args.threads),
        "--n-gpu-layers",
        str(selected_n_gpu_layers),
    ]
    if use_precise_phases:
        llm_cmd.extend(["--phase-stats-path", remote_phase_csv])
    selected_device = args.device
    if selected_device is None:
        if args.backend == "cpu":
            selected_device = "none"
        elif args.backend == "hexagon":
            selected_device = "HTP0"
    if selected_device:
        llm_cmd.extend(["--device", selected_device])
    if args.backend == "hexagon":
        llm_cmd.append("--no-mmap")
        llm_cmd.extend(["-fa", "on"])
    if args.ctx_size is not None:
        llm_cmd.extend(["--ctx-size", str(args.ctx_size)])
    if args.ubatch_size is not None:
        llm_cmd.extend(["--ubatch-size", str(args.ubatch_size)])
    if args.temperature is not None:
        llm_cmd.extend(["--temp", str(args.temperature)])

    sample_interval = f"{args.sample_interval:.3f}"
    shell_prefix = " && ".join(
        runner_cmd
        + [
            f"export GGML_HEXAGON_EXPERIMENTAL=1" if args.backend == "hexagon" else ":",
            f"export GGML_HEXAGON_NDEV={args.hexagon_ndev}" if args.hexagon_ndev is not None else ":",
            f"rm -f {shlex.quote(remote_phase_csv)}",
            f"printf '%s\\n' 'sample_idx,elapsed_s,pid,pid_alive,vmrss_kb,vmsize_kb,vmhwm_kb,smaps_rss_kb,smaps_pss_kb,smaps_private_dirty_kb,smaps_shared_clean_kb,mem_available_kb,cached_kb,dma_heap_pool_kb,gpu_total_kb,kgsl_shmem_usage_kb' > {shlex.quote(remote_memory_csv)}",
            f"( {_shell_join(llm_cmd)} > {shlex.quote(remote_output)} 2>&1; echo $? > {shlex.quote(remote_exit_code)} ) &",
        ]
    )
    shell_loop = "while kill -0 \"$runner_pid\" 2>/dev/null; do " + " ; ".join(
        [
            f"elapsed_s=$(awk -v i=\"$sample_idx\" -v s=\"{sample_interval}\" 'BEGIN {{ printf \"%.3f\", i * s }}')",
            "status_file=/proc/\"$runner_pid\"/status",
            "smaps_file=/proc/\"$runner_pid\"/smaps_rollup",
            "vmrss=$(awk '/^VmRSS:/ {print $2; exit}' \"$status_file\" 2>/dev/null)",
            "vmsize=$(awk '/^VmSize:/ {print $2; exit}' \"$status_file\" 2>/dev/null)",
            "vmhwm=$(awk '/^VmHWM:/ {print $2; exit}' \"$status_file\" 2>/dev/null)",
            "smaps_rss=$(awk '/^Rss:/ {print $2; exit}' \"$smaps_file\" 2>/dev/null)",
            "smaps_pss=$(awk '/^Pss:/ {print $2; exit}' \"$smaps_file\" 2>/dev/null)",
            "smaps_private_dirty=$(awk '/^Private_Dirty:/ {print $2; exit}' \"$smaps_file\" 2>/dev/null)",
            "smaps_shared_clean=$(awk '/^Shared_Clean:/ {print $2; exit}' \"$smaps_file\" 2>/dev/null)",
            "mem_available=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null)",
            "cached=$(awk '/^Cached:/ {print $2; exit}' /proc/meminfo 2>/dev/null)",
            "dma_heap_pool=$(awk '/^DmaHeapPool:/ {print $2; exit}' /proc/meminfo 2>/dev/null)",
            "gpu_total=$(awk '/^GpuTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null)",
            "kgsl_shmem_usage=$(awk '/^KgslShmemUsage:/ {print $2; exit}' /proc/meminfo 2>/dev/null)",
            f"printf '%s,%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\\n' \"$sample_idx\" \"$elapsed_s\" \"$runner_pid\" \"${{vmrss:-0}}\" \"${{vmsize:-0}}\" \"${{vmhwm:-0}}\" \"${{smaps_rss:-0}}\" \"${{smaps_pss:-0}}\" \"${{smaps_private_dirty:-0}}\" \"${{smaps_shared_clean:-0}}\" \"${{mem_available:-0}}\" \"${{cached:-0}}\" \"${{dma_heap_pool:-0}}\" \"${{gpu_total:-0}}\" \"${{kgsl_shmem_usage:-0}}\" >> {shlex.quote(remote_memory_csv)}",
            "sample_idx=$((sample_idx + 1))",
            f"sleep {sample_interval}",
        ]
    ) + "; done ; wait \"$runner_pid\""
    shell_script = shell_prefix + " runner_pid=$! ; sample_idx=0 ; " + shell_loop
    _run(adb + ["shell", shell_script])

    log_text = ""
    exit_code_text = _remote_read_text(adb, remote_exit_code)
    if _remote_exists(adb, remote_output):
        pulled_log = _run(adb + ["shell", f"cat {shlex.quote(remote_output)}"], capture_output=True)
        log_text = pulled_log.stdout
    else:
        log_text = ""

    output_dir = _result_dir(args.results_root, args.backend, model_name)
    (output_dir / "foundation_output.txt").write_text(log_text, encoding="utf-8")

    if exit_code_text is not None:
        (output_dir / "foundation_exit_code.txt").write_text(exit_code_text.strip() + "\n", encoding="utf-8")

    _run(adb + ["pull", remote_memory_csv, str(output_dir / "android_memory_timeline.csv")])
    phase_rows: list[dict[str, str]] = []
    if _remote_exists(adb, remote_phase_csv):
        _run(adb + ["pull", remote_phase_csv, str(output_dir / "foundation_phase_stats.csv")])
        phase_rows = _read_phase_rows(output_dir / "foundation_phase_stats.csv")
    memory_rows = _read_csv_dicts(output_dir / "android_memory_timeline.csv")
    duration_s = max(
        [float(row.get("elapsed_s", "0") or 0.0) for row in memory_rows] or [0.0]
    )
    remote_pid = next((row.get("pid", "") for row in memory_rows if row.get("pid")), "")

    summary = _parse_log_summary(log_text)
    perf = _parse_perf_summary(log_text)
    summary.update(perf)
    proc_rows = [
        {"metric": "backend", "value": f"llamacpp_{args.backend}", "unit": ""},
        {"metric": "model_name", "value": model_name, "unit": ""},
        {"metric": "wall_time_s", "value": round(duration_s, 3), "unit": "s"},
        {"metric": "remote_pid", "value": remote_pid, "unit": ""},
        {"metric": "return_code", "value": (exit_code_text or "").strip(), "unit": ""},
    ]
    for key, value in summary.items():
        unit = "ms" if key.endswith("_ms") else "MiB" if key.endswith("_mib") else ""
        proc_rows.append({"metric": key, "value": value, "unit": unit})
    if phase_rows:
        _write_csv(output_dir / "foundation_summary.csv", proc_rows, ["metric", "value", "unit"])
        _write_phase_csv(output_dir / "foundation_proc.csv", phase_rows)
    else:
        _write_csv(output_dir / "foundation_proc.csv", proc_rows, ["metric", "value", "unit"])

    vision_rows = []
    for metric in ("image_slice_encoded_ms", "image_decoded_ms", "prompt_eval_tokens", "prompt_eval_tok_s", "decode_eval_runs", "decode_eval_tok_s", "total_time_ms"):
        if metric in summary:
            unit = "ms" if metric.endswith("_ms") else "tok/s" if metric.endswith("_tok_s") else "tokens" if "tokens" in metric else "runs" if "runs" in metric else ""
            vision_rows.append({"metric": metric, "value": summary[metric], "unit": unit})
    if vision_rows:
        _write_csv(output_dir / "vision_output_stats.csv", vision_rows, ["metric", "value", "unit"])

    _write_png_memory_timeline(output_dir, memory_rows, summary)
    if phase_rows:
        _write_png_phase_duration_from_rows(output_dir, _read_phase_rows(output_dir / "foundation_proc.csv"))
    else:
        _write_png_phase_duration(output_dir, perf)

    print(f"[llamacpp] result dir: {output_dir}")
    print(f"[llamacpp] wall time: {duration_s:.3f}s")
    if exit_code_text is not None:
        print(f"[llamacpp] exit code: {exit_code_text.strip()}")
    if log_text:
        print(log_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
