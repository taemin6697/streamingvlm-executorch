from pathlib import Path

from my_research.foundation_llamacpp.runner import cli as runner_cli


ROOT = Path(__file__).resolve().parents[3]
KV_CPP = ROOT / "llama.cpp/src/llama-kv-cache.cpp"
KV_CELLS_H = ROOT / "llama.cpp/src/llama-kv-cells.h"
OPENCL_H = ROOT / "llama.cpp/ggml/include/ggml-opencl.h"
OPENCL_CPP = ROOT / "llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp"
RUNNER_CLI = ROOT / "my_research/foundation_llamacpp/runner/cli.py"


def test_dynamic_kv_grow_has_device_to_device_copy_path():
    kv_cpp = KV_CPP.read_text()
    opencl_h = OPENCL_H.read_text()
    opencl_cpp = OPENCL_CPP.read_text()

    assert "#include \"ggml-opencl.h\"" in kv_cpp
    assert "ggml_backend_opencl_tensor_copy_bytes" in opencl_h
    assert "clEnqueueCopyBuffer" in opencl_cpp
    assert "copy_existing_data_from" in kv_cpp
    assert "device-to-device" in kv_cpp


def test_dynamic_kv_grow_preserves_cell_metadata_without_state_snapshot():
    kv_cpp = KV_CPP.read_text()
    kv_cells_h = KV_CELLS_H.read_text()

    reset_body = kv_cpp[
        kv_cpp.index("bool llama_kv_cache::reset_capacity"):
        kv_cpp.index("void llama_kv_cache::clear")
    ]

    assert "void grow_to(uint32_t n)" in kv_cells_h
    assert "const uint32_t old_size = copy_existing ? get_size() : 0;" in reset_body
    assert "old_v_cells" in reset_body
    assert "v_cells[s].grow_to(kv_size)" in reset_body
    assert "state_write(snapshot)" not in reset_body
    assert "state_read(reader)" not in reset_body


def test_streaming_timeline_shows_only_aggregate_dynamic_kv_grow_lane():
    runner_cli_source = RUNNER_CLI.read_text()

    phase_name_body = runner_cli_source[
        runner_cli_source.index("def _streaming_timeline_phase_name"):
        runner_cli_source.index("def _streaming_timeline_origin")
    ]
    timeline_body = runner_cli_source[
        runner_cli_source.index("def _write_png_streaming_phase_timeline"):
        runner_cli_source.index("def _finalize_hybrid_outputs")
    ]

    assert "\"DynamicKVGrow\"" in phase_name_body
    assert "\"DynamicKVGrow\"," in timeline_body
    for phase in runner_cli.DYNAMIC_KV_GROW_BREAKDOWN_PHASES.values():
        assert runner_cli._streaming_timeline_phase_name(phase) is None
        assert f'"{phase}",' not in timeline_body


def test_dynamic_kv_grow_logs_breakdown_stages():
    kv_cpp = KV_CPP.read_text()
    context_cpp = (ROOT / "llama.cpp/src/llama-context.cpp").read_text()

    assert "dynamic KV grow breakdown alloc" in kv_cpp
    assert "dynamic KV grow breakdown metadata" in kv_cpp
    assert "dynamic KV grow breakdown copy" in kv_cpp
    assert "dynamic KV grow breakdown scheduler_reserve" in context_cpp


def test_dynamic_kv_stdout_parser_emits_breakdown_rows(tmp_path):
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(
        "\n".join(
            [
                "llama_kv_cache::grow_to: growing dynamic KV cache: old = 1024, new = 16384, logical = 32768, clock_ms = 100000",
                "llama_kv_cache::reset_capacity: dynamic KV grow breakdown alloc clock_start_ms = 100010, clock_end_ms = 100120",
                "llama_kv_cache::reset_capacity: dynamic KV grow breakdown metadata clock_start_ms = 100120, clock_end_ms = 100130",
                "llama_kv_cache::reset_capacity: dynamic KV grow breakdown copy clock_start_ms = 100130, clock_end_ms = 100300",
                "llama_kv_cache::reset_capacity: size = 448.00 MiB",
                "llama_kv_cache::grow_to: dynamic KV grow completed in 305.000 ms, clock_ms = 100305",
                "llama_context::decode: dynamic KV grow breakdown scheduler_reserve clock_start_ms = 100305, clock_end_ms = 100394",
                "llama_context::decode: dynamic KV grow retry window: old = 1024, new = 16384, logical = 32768, clock_start_ms = 100000, clock_end_ms = 100394",
            ]
        ),
        encoding="utf-8",
    )

    rows = runner_cli._dynamic_kv_rows_from_stdout(stdout, clock_origin_ms=99000)
    by_name = {row["row_type"]: row for row in rows}

    assert set(by_name) >= {
        "DynamicKVGrow",
        "DynamicKVGrowAlloc",
        "DynamicKVGrowMetadata",
        "DynamicKVGrowCopy",
        "DynamicKVGrowSchedulerReserve",
    }
    assert by_name["DynamicKVGrow"]["elapsed_s_start"] == "1.000000"
    assert by_name["DynamicKVGrow"]["elapsed_s_end"] == "1.394000"
    assert by_name["DynamicKVGrowAlloc"]["elapsed_s_start"] == "1.010000"
    assert by_name["DynamicKVGrowAlloc"]["elapsed_s_end"] == "1.120000"
    assert by_name["DynamicKVGrowCopy"]["col_a_ms"] == "170"
    assert by_name["DynamicKVGrowSchedulerReserve"]["col_a_ms"] == "89"


def test_dynamic_kv_breakdown_has_dedicated_stacked_bar(tmp_path):
    rows = [
        {
            "row_type": "DynamicKVGrow",
            "elapsed_s_start": "1.000000",
            "elapsed_s_end": "1.394000",
            "col_a_ms": "394",
            "token_idx": "1024->16384/32768 cells; 28.00->448.00 MiB",
        },
        {"row_type": "DynamicKVGrowAlloc", "elapsed_s_start": "1.010000", "elapsed_s_end": "1.120000", "col_a_ms": "110", "token_idx": "stage=alloc"},
        {"row_type": "DynamicKVGrowMetadata", "elapsed_s_start": "1.120000", "elapsed_s_end": "1.130000", "col_a_ms": "10", "token_idx": "stage=metadata"},
        {"row_type": "DynamicKVGrowCopy", "elapsed_s_start": "1.130000", "elapsed_s_end": "1.300000", "col_a_ms": "170", "token_idx": "stage=copy"},
        {"row_type": "DynamicKVGrowSchedulerReserve", "elapsed_s_start": "1.305000", "elapsed_s_end": "1.394000", "col_a_ms": "89", "token_idx": "stage=scheduler_reserve"},
    ]

    runner_cli._write_png_dynamic_kv_grow_breakdown(tmp_path, rows)

    assert (tmp_path / "dynamic_kv_grow_breakdown_stacked_bar.png").exists()
