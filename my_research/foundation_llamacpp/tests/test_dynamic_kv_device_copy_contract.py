from pathlib import Path


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


def test_streaming_timeline_shows_dynamic_kv_grow_lane():
    runner_cli = RUNNER_CLI.read_text()

    phase_name_body = runner_cli[
        runner_cli.index("def _streaming_timeline_phase_name"):
        runner_cli.index("def _streaming_timeline_origin")
    ]
    timeline_body = runner_cli[
        runner_cli.index("def _write_png_streaming_phase_timeline"):
        runner_cli.index("def _finalize_hybrid_outputs")
    ]

    assert "\"DynamicKVGrow\"" in phase_name_body
    assert "\"DynamicKVGrow\"," in timeline_body
