import pytest
import torch

from my_research.test_python.internvl3_partial_prefill import (
    aggregate_visual_attention,
    build_attention_island_mask,
    cc_background_cluster_count,
    cls_patch_attention,
    downsample_grid_average,
    island_background_merge_pool_selection,
    island_context_pool_selection,
    object_topk_residual_cluster_pool_selection,
    parse_ratios,
    patch_attention_rollout,
    patch_incoming_attention,
    percent_reduction,
    select_topk_indices,
    selection_source_coverage,
    square_grid_size,
    visual_token_scores_from_vit_patch_scores,
    vit_attention_rollout,
)


def test_parse_ratios_accepts_percent_values():
    assert parse_ratios("10,30,100") == [10, 30, 100]


def test_select_topk_indices_preserves_original_order_after_ranking():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])

    selected = select_topk_indices(scores, 50)

    assert selected.tolist() == [1, 3]


def test_aggregate_visual_attention_uses_text_queries_only():
    attention = torch.zeros(1, 1, 5, 5)
    attention[0, 0, 3, 1] = 0.7
    attention[0, 0, 4, 2] = 0.3

    scores = aggregate_visual_attention(
        attentions=(attention,),
        visual_positions=torch.tensor([1, 2]),
        text_positions=torch.tensor([3, 4]),
    )

    assert torch.allclose(scores, torch.tensor([0.35, 0.15]))


def test_square_grid_size_returns_side_for_square_token_count():
    assert square_grid_size(256) == 16


def test_square_grid_size_rejects_non_square_token_count():
    with pytest.raises(ValueError):
        square_grid_size(250)


def test_percent_reduction_reports_savings_against_baseline():
    assert percent_reduction(75.0, 100.0) == pytest.approx(25.0)


def test_cls_patch_attention_averages_heads_for_one_layer():
    attention = torch.zeros(1, 2, 5, 5)
    attention[0, 0, 0, 1:] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    attention[0, 1, 0, 1:] = torch.tensor([0.5, 0.4, 0.3, 0.2])

    scores = cls_patch_attention((attention,), layer_index=-1)

    assert torch.allclose(scores, torch.tensor([0.3, 0.3, 0.3, 0.3]))


def test_vit_attention_rollout_returns_cls_to_patch_scores():
    attention = torch.eye(5).reshape(1, 1, 5, 5)
    attention[0, 0, 0, 1:] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    attention[0, 0, 0] = attention[0, 0, 0] / attention[0, 0, 0].sum()

    scores = vit_attention_rollout((attention,))

    assert scores.shape == (4,)
    assert scores.argmax().item() == 0


def test_downsample_grid_average_reduces_by_integer_factor():
    grid = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    reduced = downsample_grid_average(grid, 2)

    assert torch.allclose(reduced, torch.tensor([[2.5, 4.5], [10.5, 12.5]]))


def test_patch_incoming_attention_excludes_cls_and_averages_queries():
    layer0 = torch.zeros(1, 1, 4, 4)
    layer1 = torch.zeros(1, 1, 4, 4)
    # Patch-to-patch block only is rows/cols 1:.
    layer0[0, 0, 1:, 1:] = torch.tensor(
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4],
            [0.5, 0.2, 0.3],
        ]
    )
    layer1[0, 0, 1:, 1:] = torch.tensor(
        [
            [0.2, 0.2, 0.6],
            [0.4, 0.3, 0.3],
            [0.6, 0.2, 0.2],
        ]
    )

    scores = patch_incoming_attention((layer0, layer1))

    expected = torch.tensor(
        [
            (0.1 + 0.3 + 0.5 + 0.2 + 0.4 + 0.6) / 6,
            (0.2 + 0.3 + 0.2 + 0.2 + 0.3 + 0.2) / 6,
            (0.7 + 0.4 + 0.3 + 0.6 + 0.3 + 0.2) / 6,
        ]
    )
    assert torch.allclose(scores, expected)


def test_patch_attention_rollout_excludes_cls_and_scores_source_patch_flow():
    attention = torch.zeros(1, 1, 4, 4)
    # CLS row/column should be ignored. Inside the 3x3 patch block, every patch
    # points at the middle patch before residual rollout normalization.
    attention[0, 0, 1:, 1:] = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    attention[0, 0, 0, :] = 1.0
    attention[0, 0, :, 0] = 1.0

    scores = patch_attention_rollout((attention,))

    assert scores.shape == (3,)
    assert scores.argmax().item() == 1
    assert torch.allclose(scores, torch.tensor([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]))


def test_visual_token_scores_from_vit_patch_scores_downsamples_to_visual_grid():
    patch_scores = torch.arange(16, dtype=torch.float32)

    visual_scores = visual_token_scores_from_vit_patch_scores(patch_scores, target_token_count=4)

    assert torch.allclose(visual_scores, torch.tensor([2.5, 4.5, 10.5, 12.5]))


def test_attention_island_mask_separates_high_attention_region():
    scores = torch.tensor(
        [
            0.9,
            0.8,
            0.1,
            0.1,
            0.8,
            0.9,
            0.1,
            0.1,
            0.1,
            0.1,
            0.2,
            0.2,
            0.1,
            0.1,
            0.2,
            0.2,
        ]
    )

    mask = build_attention_island_mask(scores, grid_side=4, smoothing_sigma=0.0)

    assert mask.shape == (4, 4)
    assert mask[:2, :2].all()
    assert not mask[2:, :2].any()


def test_island_context_pool_selection_keeps_object_topk_and_pools_context_cells():
    features = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    scores = torch.tensor(
        [
            0.90,
            0.80,
            0.70,
            0.60,
            0.10,
            0.20,
            0.30,
            0.40,
            0.10,
            0.20,
            0.30,
            0.40,
            0.10,
            0.20,
            0.30,
            0.40,
        ]
    )
    island_mask = torch.zeros(4, 4, dtype=torch.bool)
    island_mask[0, :] = True

    selection = island_context_pool_selection(
        features,
        scores,
        ratio_percent=50,
        grid_side=4,
        background_ratio_percent=25,
        background_pool_grid=2,
        island_mask=island_mask,
    )

    assert selection.object_indices.tolist() == [0, 1, 2, 3]
    assert len(selection.background_summaries) == 4
    assert selection.features.flatten().tolist() == pytest.approx(
        [0.0, 4.5, 1.0, 2.0, 6.5, 3.0, 10.5, 12.5]
    )


def test_island_background_merge_pool_selection_partitions_background_tokens():
    features = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    scores = torch.tensor(
        [
            0.90,
            0.80,
            0.70,
            0.60,
            0.10,
            0.20,
            0.30,
            0.40,
            0.10,
            0.20,
            0.30,
            0.40,
            0.10,
            0.20,
            0.30,
            0.40,
        ]
    )
    island_mask = torch.zeros(4, 4, dtype=torch.bool)
    island_mask[0, :] = True

    selection = island_background_merge_pool_selection(
        features,
        scores,
        ratio_percent=50,
        grid_side=4,
        background_ratio_percent=25,
        island_mask=island_mask,
    )

    assert selection.object_indices.tolist() == [0, 1, 2, 3]
    assert len(selection.background_summaries) == 4

    source_sets = [set(summary["source_indices"]) for summary in selection.background_summaries]
    assert set().union(*source_sets) == set(range(4, 16))
    assert sum(len(source) for source in source_sets) == 12
    for summary in selection.background_summaries:
        expected = features[summary["source_indices"]].mean().item()
        feature_index = next(
            idx for idx, item in enumerate(selection.background_summaries) if item is summary
        )
        assert selection.background_summaries[feature_index]["score"] == pytest.approx(
            scores[summary["source_indices"]].mean().item()
        )
        assert expected >= 4.0


def test_object_topk_residual_cluster_pool_selection_covers_all_non_object_tokens():
    features = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    scores = torch.tensor(
        [
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
            0.10,
            0.20,
            0.30,
            0.40,
            0.10,
            0.20,
            0.30,
            0.40,
        ]
    )
    island_mask = torch.zeros(4, 4, dtype=torch.bool)
    island_mask[:2, :] = True

    selection = object_topk_residual_cluster_pool_selection(
        features,
        scores,
        ratio_percent=50,
        grid_side=4,
        background_ratio_percent=25,
        island_mask=island_mask,
    )

    assert selection.object_indices.tolist() == [0, 1, 2, 3]
    assert len(selection.background_summaries) == 4

    source_sets = [set(summary["source_indices"]) for summary in selection.background_summaries]
    assert set().union(*source_sets) == set(range(4, 16))
    assert sum(len(source) for source in source_sets) == 12
    assert all(not (source & set(selection.object_indices.tolist())) for source in source_sets)


def test_object_topk_residual_cluster_pool_selection_accepts_fixed_background_count():
    features = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    scores = torch.tensor(
        [
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
            0.10,
            0.20,
            0.30,
            0.40,
            0.10,
            0.20,
            0.30,
            0.40,
        ]
    )
    island_mask = torch.zeros(4, 4, dtype=torch.bool)
    island_mask[:2, :] = True

    selection = object_topk_residual_cluster_pool_selection(
        features,
        scores,
        ratio_percent=50,
        grid_side=4,
        background_ratio_percent=25,
        background_fixed_clusters=2,
        island_mask=island_mask,
    )

    assert selection.object_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert len(selection.background_summaries) == 2
    source_sets = [set(summary["source_indices"]) for summary in selection.background_summaries]
    assert set().union(*source_sets) == set(range(16)) - set(selection.object_indices.tolist())


def test_cc_background_cluster_count_uses_component_count_and_component_size():
    residual_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )

    count = cc_background_cluster_count(
        residual_mask,
        k_total=10,
        target_tokens_per_cluster=3,
        min_clusters=1,
        max_clusters=16,
        max_fraction=1.0,
    )

    assert count == 4


def test_selection_source_coverage_reports_missing_and_overlap_tokens():
    coverage = selection_source_coverage(
        object_indices=torch.tensor([0, 1, 2]),
        background_summaries=[
            {"source_indices": [2, 3]},
            {"source_indices": [4]},
        ],
        total_tokens=6,
    )

    assert coverage == {
        "visual_source_object_count": 3,
        "visual_source_background_count": 3,
        "visual_source_coverage_count": 5,
        "visual_source_overlap_count": 1,
        "visual_source_missing_count": 1,
    }
