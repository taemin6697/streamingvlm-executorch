from pathlib import Path

from my_research.foundation_llamacpp.runner import cli as runner_cli


ROOT = Path(__file__).resolve().parents[3]
LLAMA_H = ROOT / "llama.cpp/include/llama.h"
COMMON_H = ROOT / "llama.cpp/common/common.h"
COMMON_ARG = ROOT / "llama.cpp/common/arg.cpp"
COMMON_CPP = ROOT / "llama.cpp/common/common.cpp"
CPARAMS_H = ROOT / "llama.cpp/src/llama-cparams.h"
CONTEXT_CPP = ROOT / "llama.cpp/src/llama-context.cpp"
MODEL_CPP = ROOT / "llama.cpp/src/llama-model.cpp"
KV_H = ROOT / "llama.cpp/src/llama-kv-cache.h"
KV_CPP = ROOT / "llama.cpp/src/llama-kv-cache.cpp"
KV_CELLS_H = ROOT / "llama.cpp/src/llama-kv-cells.h"
GRAPH_H = ROOT / "llama.cpp/src/llama-graph.h"
GRAPH_CPP = ROOT / "llama.cpp/src/llama-graph.cpp"
GGML_H = ROOT / "llama.cpp/ggml/include/ggml.h"
GGML_C = ROOT / "llama.cpp/ggml/src/ggml.c"
OPENCL_CPP = ROOT / "llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp"
OPENCL_FA_F16 = ROOT / "llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f16.cl"
OPENCL_FA_F32 = ROOT / "llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32.cl"
OPENCL_FA_F32_F16 = ROOT / "llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl"
RUNNER_CLI = ROOT / "my_research/foundation_llamacpp/runner/cli.py"
STREAMING_CPP = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp"
README = ROOT / "my_research/foundation_llamacpp/docs/README.md"
ARCHIVE = ROOT / "my_research/foundation_llamacpp/docs/archive/paged_kv_opencl_prototype.md"


def test_paged_kv_flags_are_plumbed():
    llama_h = LLAMA_H.read_text()
    common_h = COMMON_H.read_text()
    common_arg = COMMON_ARG.read_text()
    common_cpp = COMMON_CPP.read_text()
    cparams_h = CPARAMS_H.read_text()
    context_cpp = CONTEXT_CPP.read_text()
    runner_cli = RUNNER_CLI.read_text()
    streaming_cpp = STREAMING_CPP.read_text()

    assert "paged_kv_cache" in llama_h
    assert "kv_page_size" in llama_h
    assert "paged_kv_cache" in common_h
    assert "kv_page_size" in common_h
    assert "--paged-kv-cache" in common_arg
    assert "--kv-page-size" in common_arg
    assert "cparams.paged_kv_cache" in common_cpp
    assert "cparams.kv_page_size" in common_cpp
    assert "paged_kv_cache" in cparams_h
    assert "kv_page_size" in cparams_h
    assert "paged KV cache requires --kv-init-size and --kv-grow-step" in context_cpp
    assert "cparams.dynamic_kv_cache || cparams.paged_kv_cache" in context_cpp
    assert "--paged-kv-cache" in runner_cli
    assert "--kv-page-size" in runner_cli
    assert "kv_init_size" in runner_cli
    assert "kv_grow_step" in runner_cli
    assert 'parts = ["-c", str(args.ctx_size), "--paged-kv-cache", "--kv-page-size", str(args.kv_page_size)]' in runner_cli
    assert "--paged-kv-cache" in streaming_cpp
    assert "--kv-page-size" in streaming_cpp
    assert "args.paged_kv_cache || args.dynamic_kv_cache" in streaming_cpp


def test_paged_kv_cache_has_page_table_structures():
    kv_h = KV_H.read_text()
    kv_cpp = KV_CPP.read_text()
    model_cpp = MODEL_CPP.read_text()

    assert "PagedKVBlockTable" in kv_h
    assert "paged_block_table" in kv_h
    assert "allocate_paged_kv_page" in kv_h
    assert "logical_pos_to_page_offset" in kv_h
    assert "get_kv_page_size" in kv_h
    assert "is_paged_kv" in kv_h
    assert "allocate_paged_kv_page" in kv_cpp
    assert "logical_pos_to_page_offset" in kv_cpp
    assert "logical_pos_to_physical_pos" in kv_cpp
    assert "LLAMA_LOG_INFO(\"%s: paged KV allocated page" in kv_cpp
    assert "cparams.paged_kv_cache" in model_cpp
    assert "cparams.kv_page_size" in model_cpp
    assert "(cparams.dynamic_kv_cache || cparams.paged_kv_cache)" in model_cpp


def test_paged_kv_graph_exposes_block_table_input():
    graph_h = GRAPH_H.read_text()
    graph_cpp = GRAPH_CPP.read_text()
    kv_h = KV_H.read_text()
    kv_cpp = KV_CPP.read_text()

    assert "self_kv_page_table" in graph_h
    assert "build_input_kv_page_table" in kv_h
    assert "set_input_kv_page_table" in kv_h
    assert "build_input_kv_page_table" in kv_cpp
    assert "set_input_kv_page_table" in kv_cpp
    assert "attn_inp_kv_page_table" in kv_cpp
    assert "self_kv_page_table" in graph_cpp
    assert "set_input_kv_page_table" in graph_cpp


def test_paged_kv_write_indices_are_page_mapped():
    kv_cpp = KV_CPP.read_text()

    assert "logical_pos_to_physical_pos" in kv_cpp
    assert "data[s*sinfo.size() + i] = offs + logical_pos_to_physical_pos(sinfo.idxs[s][i]);" in kv_cpp
    assert "const uint32_t physical_row = logical_pos_to_physical_pos(row_idx);" in kv_cpp


def test_paged_kv_grow_is_metadata_only_after_reserved_backing_allocation():
    kv_cpp = KV_CPP.read_text()
    kv_cells_h = KV_CELLS_H.read_text()

    assert "void grow_to(uint32_t n)" in kv_cells_h
    assert "const uint32_t backing_kv_size = paged_kv_cache ? logical_kv_size : kv_size;" in kv_cpp
    paged_grow_start = kv_cpp.index("if (paged_kv_cache) {", kv_cpp.index("bool llama_kv_cache::grow_to"))
    dynamic_grow_start = kv_cpp.index('LLAMA_LOG_INFO("%s: growing dynamic KV cache', paged_grow_start)
    paged_grow = kv_cpp[paged_grow_start:dynamic_grow_start]

    assert "v_cells[s].grow_to(new_size);" in paged_grow
    assert "paged KV grow metadata-only" in paged_grow
    assert "reset_capacity" not in paged_grow


def test_paged_kv_stdout_metadata_grow_is_reported_as_dynamic_grow_row(tmp_path):
    stdout_path = tmp_path / "hybrid_streaming_stdout.txt"
    stdout_path.write_text(
        "\n".join(
            [
                "reset_capacity: size =  896.00 MiB (  1024 active /  32768 backing /  32768 logical cells,  28 layers,  1/1 seqs), K (f16):  448.00 MiB, V (f16):  448.00 MiB",
                "grow_to: paged KV grow metadata-only: old = 1024, new = 16384, backing = 32768, logical = 32768, pages = 4 -> 64, page_size = 256, elapsed = 0.201 ms, clock_start_ms = 858918956, clock_end_ms = 858918957",
                "sched_reserve: reserve took 112.16 ms, sched copies = 1",
                "decode: dynamic KV grow retry window: old = 1024, new = 16384, logical = 32768, clock_start_ms = 858918956, clock_end_ms = 858919069",
            ]
        ),
        encoding="utf-8",
    )

    rows = runner_cli._dynamic_kv_rows_from_stdout(stdout_path, clock_origin_ms=858900000)

    assert len(rows) == 1
    assert rows[0]["row_type"] == "DynamicKVGrow"
    assert rows[0]["total_ms"] == "113"
    assert rows[0]["kv_pos"] == "1024"
    assert rows[0]["kv_total"] == "16384"
    assert rows[0]["kv_estimated_used_kb"] == str(28 * 1024)
    assert rows[0]["kv_total_kb"] == str(448 * 1024)
    assert rows[0]["kv_physical_committed_kb"] == str(896 * 1024)
    assert "paged metadata-only" in rows[0]["token_idx"]


def test_paged_kv_flash_attention_consumes_page_table():
    context_cpp = CONTEXT_CPP.read_text()
    graph_cpp = GRAPH_CPP.read_text()
    ggml_h = GGML_H.read_text()
    ggml_c = GGML_C.read_text()
    opencl_cpp = OPENCL_CPP.read_text()
    kernels = "\n".join(
        path.read_text()
        for path in [OPENCL_FA_F16, OPENCL_FA_F32, OPENCL_FA_F32_F16]
    )

    assert "paged KV metadata/page-table mode is wired" not in context_cpp
    assert "paged attention is not implemented yet" not in context_cpp
    assert "TODO_PAGED_KV_ATTENTION" not in graph_cpp
    assert "ggml_flash_attn_ext_set_page_table" in ggml_h
    assert "a->src[5] = page_table" in ggml_c
    assert "const ggml_tensor * page_table = dst->src[5]" in opencl_cpp
    assert "kv_page_size" in opencl_cpp
    assert "physical_k_row_idx" in kernels
    assert "physical_v_row_idx" in kernels
    assert "paged_kv_row" in kernels


def test_paged_kv_future_true_compression_requires_shrink_repack_docs():
    readme = README.read_text()
    archive = ARCHIVE.read_text()

    assert "true KV compression" in archive
    assert "shrink/repack" in archive
    assert "device-to-device" in archive
    assert "capacity pages" in archive
    assert "--paged-kv-cache" in readme
    assert "--kv-page-size" in readme
