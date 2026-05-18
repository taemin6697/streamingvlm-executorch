from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CLIP_CPP = ROOT / "llama.cpp/tools/mtmd/clip.cpp"
CLIP_H = ROOT / "llama.cpp/tools/mtmd/clip.h"
MTMD_CPP = ROOT / "llama.cpp/tools/mtmd/mtmd.cpp"
HYBRID_DECODE_CPP = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/hybrid_decode.cpp"


def test_mtmd_projects_qwen25_premerger_features_through_qwen_merger():
    clip_cpp = CLIP_CPP.read_text()
    clip_h = CLIP_H.read_text()
    mtmd_cpp = MTMD_CPP.read_text()

    assert "clip_graph_qwen2vl_projector_only" in clip_cpp
    assert "clip_project_qwen2vl_features" in clip_h
    assert "clip_project_qwen2vl_features" in clip_cpp
    assert "PROJECTOR_TYPE_QWEN25VL" in mtmd_cpp
    assert "PROJECTOR_TYPE_QWEN2VL" in mtmd_cpp
    assert "clip_project_qwen2vl_features" in mtmd_cpp


def test_qwen25_premerger_projection_preserves_llamacpp_window_ordering():
    clip_cpp = CLIP_CPP.read_text()
    start = clip_cpp.index("struct clip_graph_qwen2vl_projector_only")
    end = clip_cpp.index("struct clip_graph_internvl_preprojector_only")
    projector_only = clip_cpp[start:end]

    assert "model.post_ln_w" in projector_only
    assert "build_norm(embeddings, model.post_ln_w, model.post_ln_b" in projector_only
    assert 'ggml_set_name(inv_window_idx, "inv_window_idx")' in projector_only
    assert 'ggml_set_name(window_idx, "window_idx")' in projector_only
    assert "ggml_get_rows(ctx0, embeddings, inv_window_idx)" in projector_only
    assert "ggml_get_rows(ctx0, embeddings, window_idx)" in projector_only

    project_fn = clip_cpp[clip_cpp.index("bool clip_project_qwen2vl_features"):]
    assert 'ggml_graph_get_tensor(gf, "inv_window_idx")' in project_fn
    assert 'ggml_graph_get_tensor(gf, "window_idx")' in project_fn


def test_hybrid_embedding_cursor_distinguishes_feature_tokens_from_image_tokens():
    source = HYBRID_DECODE_CPP.read_text()

    assert "struct embedding_slice" in source
    assert "next_slice_for_chunk" in source
    assert "feature_tokens" in source
    assert "mtmd_project_features(" in source
    assert "static_cast<int32_t>(slice.feature_tokens)" in source
