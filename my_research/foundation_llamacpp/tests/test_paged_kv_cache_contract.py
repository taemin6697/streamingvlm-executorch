from pathlib import Path


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
GRAPH_H = ROOT / "llama.cpp/src/llama-graph.h"
GRAPH_CPP = ROOT / "llama.cpp/src/llama-graph.cpp"
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
    assert "paged KV and dynamic contiguous KV cannot both be enabled" in context_cpp
    assert "--paged-kv-cache" in runner_cli
    assert "--kv-page-size" in runner_cli
    assert "--paged-kv-cache" in streaming_cpp
    assert "--kv-page-size" in streaming_cpp


def test_paged_kv_cache_has_page_table_structures():
    kv_h = KV_H.read_text()
    kv_cpp = KV_CPP.read_text()
    model_cpp = MODEL_CPP.read_text()

    assert "PagedKVBlockTable" in kv_h
    assert "paged_block_table" in kv_h
    assert "allocate_paged_kv_page" in kv_h
    assert "logical_pos_to_page_offset" in kv_h
    assert "is_paged_kv" in kv_h
    assert "allocate_paged_kv_page" in kv_cpp
    assert "logical_pos_to_page_offset" in kv_cpp
    assert "LLAMA_LOG_INFO(\"%s: paged KV allocated page" in kv_cpp
    assert "cparams.paged_kv_cache" in model_cpp
    assert "cparams.kv_page_size" in model_cpp


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


def test_paged_kv_mode_is_guarded_until_attention_reads_page_table():
    context_cpp = CONTEXT_CPP.read_text()
    graph_cpp = GRAPH_CPP.read_text()

    assert "paged KV metadata/page-table mode is wired" in context_cpp
    assert "paged attention is not implemented yet" in context_cpp
    assert "TODO_PAGED_KV_ATTENTION" in graph_cpp


def test_paged_kv_future_true_compression_requires_shrink_repack_docs():
    readme = README.read_text()
    archive = ARCHIVE.read_text()

    assert "true KV compression" in archive
    assert "shrink/repack" in archive
    assert "device-to-device" in archive
    assert "capacity pages" in archive
    assert "--paged-kv-cache" in readme
    assert "--kv-page-size" in readme
