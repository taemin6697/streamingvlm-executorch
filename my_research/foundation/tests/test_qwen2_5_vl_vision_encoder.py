import pytest
import torch

from my_research.foundation.models.qwen2_5_vl.vision_encoder.model import (
    Qwen2_5_VLVisionTower,
    make_exportable_fixed_grid_visual,
    resolve_image_export_shape,
    visual_weight_files_from_index,
)


def test_resolve_square_image_shape_matches_qwen_token_count():
    shape = resolve_image_export_shape(
        image_size=(448, 448),
        image_tokens=None,
        patch_size=14,
        spatial_merge_size=2,
    )

    assert shape.image_height == 448
    assert shape.image_width == 448
    assert shape.grid_thw == (1, 32, 32)
    assert shape.num_tokens == 1024


def test_resolve_rectangular_image_shape_matches_qwen_token_count():
    shape = resolve_image_export_shape(
        image_size=(392, 560),
        image_tokens=None,
        patch_size=14,
        spatial_merge_size=2,
    )

    assert shape.grid_thw == (1, 28, 40)
    assert shape.num_tokens == 1120


def test_resolve_image_tokens_infers_square_export_shape():
    shape = resolve_image_export_shape(
        image_size=None,
        image_tokens=256,
        patch_size=14,
        spatial_merge_size=2,
    )

    assert (shape.image_height, shape.image_width) == (224, 224)
    assert shape.grid_thw == (1, 16, 16)
    assert shape.num_tokens == 256


def test_resolve_image_tokens_infers_near_square_rectangular_shape():
    shape = resolve_image_export_shape(
        image_size=None,
        image_tokens=280,
        patch_size=14,
        spatial_merge_size=2,
    )

    assert (shape.image_height, shape.image_width) == (196, 280)
    assert shape.grid_thw == (1, 14, 20)
    assert shape.num_tokens == 280


def test_resolve_image_shape_rejects_mismatched_expected_tokens():
    with pytest.raises(ValueError, match="Expected 128 image tokens, but 448x448 produces 1024"):
        resolve_image_export_shape(
            image_size=(448, 448),
            image_tokens=128,
            patch_size=14,
            spatial_merge_size=2,
        )


def test_visual_weight_files_from_index_selects_only_visual_shards():
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "visual.blocks.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
            "visual.merger.mlp.2.weight": "model-00001-of-00002.safetensors",
            "model.layers.30.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        }
    }

    assert visual_weight_files_from_index(index) == ["model-00001-of-00002.safetensors"]


def test_qwen_vision_tower_patchifies_chw_input_and_checks_output_tokens():
    class FakeVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received_pixel_values_shape = None
            self.received_grid_thw = None

        def forward(self, pixel_values, grid_thw):
            self.received_pixel_values_shape = tuple(pixel_values.shape)
            self.received_grid_thw = tuple(int(v) for v in grid_thw[0].tolist())
            num_tokens = int(grid_thw[0].prod().item())
            return torch.zeros(num_tokens, 17, dtype=pixel_values.dtype)

    fake_visual = FakeVisual()
    wrapper = Qwen2_5_VLVisionTower(
        fake_visual,
        image_height=392,
        image_width=560,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=17,
        expected_tokens=1120,
    )

    output = wrapper(torch.randn(1, 3, 392, 560))

    assert output.shape == (1, 1120, 17)
    assert fake_visual.received_grid_thw == (1, 28, 40)
    assert fake_visual.received_pixel_values_shape == (1, 3, 392, 560)


def test_qwen_vision_tower_rejects_unexpected_output_token_count():
    class BadVisual(torch.nn.Module):
        def forward(self, hidden_states, grid_thw):
            return torch.zeros(279, 17, dtype=hidden_states.dtype)

    wrapper = Qwen2_5_VLVisionTower(
        BadVisual(),
        image_height=392,
        image_width=560,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=17,
        expected_tokens=1120,
    )

    with pytest.raises(RuntimeError, match="Qwen2.5-VL vision output token mismatch"):
        wrapper(torch.randn(1, 3, 392, 560))


def test_fixed_grid_qwen_visual_is_torch_exportable():
    pytest.importorskip("transformers")
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    config = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        fullatt_block_indexes=[0],
        window_size=112,
    )
    config._attn_implementation = "eager"
    visual = Qwen2_5_VisionTransformerPretrainedModel(config).eval()
    fixed_visual = make_exportable_fixed_grid_visual(visual, grid_thw=(1, 2, 2))
    wrapper = Qwen2_5_VLVisionTower(
        fixed_visual,
        image_height=28,
        image_width=28,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=32,
        expected_tokens=4,
    ).eval()

    example_inputs = wrapper.get_example_inputs()
    with torch.no_grad():
        assert wrapper(*example_inputs).shape == (1, 4, 32)
        exported = torch.export.export(wrapper, example_inputs, strict=False)

    assert exported.graph_module is not None


def test_fixed_grid_qwen_visual_matches_hf_visual_before_patch_merger():
    pytest.importorskip("transformers")
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    config = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        fullatt_block_indexes=[0],
        window_size=112,
    )
    config._attn_implementation = "eager"
    visual = Qwen2_5_VisionTransformerPretrainedModel(config).eval()
    fixed_visual = make_exportable_fixed_grid_visual(visual, grid_thw=(1, 2, 2))
    wrapper = Qwen2_5_VLVisionTower(
        fixed_visual,
        image_height=28,
        image_width=28,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=32,
        expected_tokens=4,
    ).eval()
    pixel_values = torch.randn(1, 3, 28, 28)
    flattened_patches = wrapper._patchify(pixel_values)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)

    with torch.no_grad():
        hf_output = _run_hf_visual_without_patch_merger(visual, flattened_patches, grid_thw)
        fixed_output = fixed_visual(pixel_values)

    torch.testing.assert_close(fixed_output, hf_output, atol=1e-5, rtol=1e-5)


def _run_hf_visual_without_patch_merger(visual, hidden_states, grid_thw):
    hidden_states = visual.patch_embed(hidden_states)
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=torch.int32,
    )
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(visual.blocks):
        cu_seqlens_now = cu_seqlens if layer_num in visual.fullatt_block_indexes else cu_window_seqlens
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
        )

    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    hidden_states = hidden_states[reverse_indices, :, :]
    return hidden_states.reshape(seq_len, -1)
