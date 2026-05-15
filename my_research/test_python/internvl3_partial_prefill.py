#!/usr/bin/env python3
"""
InternVL3 partial vision prefill experiment.

Default ranking uses InternVL's image encoder self-attention with CLS-excluded
patch-to-patch attention rollout, then downsamples the 32x32 patch rollout map
to the 16x16 IMG_CONTEXT grid used by InternVL3-1B. The script also keeps
alternative ranking sources for comparison.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image, ImageDraw, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PARENT = REPO_ROOT / "my_research/test_python/results"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

IMPORTANCE_SOURCES = ("llm-attention", "vit-rollout", "vit-patch-incoming", "vit-patch-rollout")
DEFAULT_IMPORTANCE_SOURCE = "vit-patch-rollout"

SELECTION_POLICY_TOPK = "topk"
SELECTION_POLICY_ISLAND_CONTEXT_POOL = "island-context-pool"
SELECTION_POLICY_ISLAND_BACKGROUND_MERGE_POOL = "island-background-merge-pool"
SELECTION_POLICY_OBJECT_TOPK_RESIDUAL_CLUSTER_POOL = "object-topk-residual-cluster-pool"
SELECTION_POLICY_OBJECT_ALL_BACKGROUND_KMEANS = "object-all-background-kmeans"
SELECTION_POLICY_OBJECT_BUDGET_BACKGROUND_KMEANS = "object-budget-background-kmeans"
POOLED_SELECTION_POLICIES = (
    SELECTION_POLICY_ISLAND_CONTEXT_POOL,
    SELECTION_POLICY_ISLAND_BACKGROUND_MERGE_POOL,
    SELECTION_POLICY_OBJECT_TOPK_RESIDUAL_CLUSTER_POOL,
    SELECTION_POLICY_OBJECT_ALL_BACKGROUND_KMEANS,
    SELECTION_POLICY_OBJECT_BUDGET_BACKGROUND_KMEANS,
)
SELECTION_POLICIES = (SELECTION_POLICY_TOPK, *POOLED_SELECTION_POLICIES)


@dataclass
class IslandContextSelection:
    features: torch.Tensor
    object_indices: torch.Tensor
    background_summaries: list[dict[str, Any]]
    island_mask: torch.Tensor


@dataclass
class RatioSelection:
    features: torch.Tensor
    object_indices: torch.Tensor
    background_summaries: list[dict[str, Any]]
    details: dict[str, Any]


@dataclass(frozen=True)
class ObjectBackgroundBudget:
    total_budget: int
    object_budget: int
    initial_background_budget: int
    object_count: int
    background_count: int


@dataclass
class ImportanceScores:
    scores: torch.Tensor
    score_basis: str
    vit_last_scores: torch.Tensor | None = None
    vit_rollout_scores: torch.Tensor | None = None
    vit_patch_scores: torch.Tensor | None = None
    vit_patch_rollout_scores: torch.Tensor | None = None


def _require_transformers_v4() -> None:
    major_str = transformers.__version__.split(".")[0]
    try:
        major = int(major_str)
    except ValueError:
        return
    if major >= 5:
        sys.exit(
            f"Incompatible transformers {transformers.__version__}. "
            "OpenGVLab InternVL remote code expects transformers 4.x.\n"
            "Fix: pip install 'transformers>=4.45,<5'"
        )


def parse_ratios(text: str) -> list[int]:
    ratios: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0 or value > 100:
            raise ValueError(f"ratio must be in 1..100, got {value}")
        ratios.append(value)
    if not ratios:
        raise ValueError("at least one ratio is required")
    return ratios


def select_topk_indices(scores: torch.Tensor, ratio_percent: int) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("scores must not be empty")
    if ratio_percent <= 0 or ratio_percent > 100:
        raise ValueError(f"ratio_percent must be in 1..100, got {ratio_percent}")

    k = max(1, math.ceil(scores.numel() * ratio_percent / 100.0))
    k = min(k, scores.numel())
    if k == scores.numel():
        return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)

    ranked = torch.topk(scores, k=k, largest=True, sorted=False).indices
    return torch.sort(ranked).values


def _normalize_tensor(scores: torch.Tensor) -> torch.Tensor:
    values = scores.detach().float().cpu()
    lo = values.min()
    hi = values.max()
    if float(hi - lo) <= 1e-12:
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo)


def _gaussian_kernel2d(sigma: float, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones((1, 1), dtype=dtype)
    radius = max(1, int(math.ceil(sigma * 3.0)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    return kernel_1d[:, None] @ kernel_1d[None, :]


def _smooth_grid(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return grid
    kernel = _gaussian_kernel2d(sigma, dtype=grid.dtype)
    pad = kernel.shape[0] // 2
    x = grid.reshape(1, 1, grid.shape[0], grid.shape[1])
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(x, kernel.reshape(1, 1, *kernel.shape)).reshape_as(grid)


def otsu_threshold(values: torch.Tensor, *, bins: int = 64) -> float:
    flat = values.detach().float().cpu().flatten()
    if flat.numel() == 0:
        raise ValueError("values must not be empty")
    lo = float(flat.min())
    hi = float(flat.max())
    if hi <= lo:
        return lo
    hist = torch.histc(flat, bins=bins, min=lo, max=hi)
    bin_centers = torch.linspace(lo, hi, bins)
    weight_total = hist.sum().clamp_min(1e-12)
    weight_bg = torch.cumsum(hist, dim=0)
    weight_fg = weight_total - weight_bg
    mean_bg = torch.cumsum(hist * bin_centers, dim=0) / weight_bg.clamp_min(1e-12)
    mean_total = (hist * bin_centers).sum()
    mean_fg = (mean_total - torch.cumsum(hist * bin_centers, dim=0)) / weight_fg.clamp_min(1e-12)
    between = weight_bg * weight_fg * (mean_bg - mean_fg).pow(2)
    between = torch.where((weight_bg > 0) & (weight_fg > 0), between, torch.zeros_like(between))
    return float(bin_centers[int(between.argmax().item())])


def _connected_components(mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    mask_np = mask.detach().cpu().bool().numpy()
    h, w = mask_np.shape
    seen = np.zeros_like(mask_np, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or not mask_np[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and mask_np[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            components.append(comp)
    return components


def _remove_small_components(mask: torch.Tensor, min_size: int) -> torch.Tensor:
    if min_size <= 1:
        return mask
    out = torch.zeros_like(mask, dtype=torch.bool)
    for comp in _connected_components(mask):
        if len(comp) >= min_size:
            for y, x in comp:
                out[y, x] = True
    return out


def _fill_small_holes(mask: torch.Tensor, max_hole_size: int) -> torch.Tensor:
    if max_hole_size <= 0:
        return mask
    inverse = ~mask
    out = mask.clone()
    h, w = mask.shape
    for comp in _connected_components(inverse):
        touches_border = any(y == 0 or x == 0 or y == h - 1 or x == w - 1 for y, x in comp)
        if not touches_border and len(comp) <= max_hole_size:
            for y, x in comp:
                out[y, x] = True
    return out


def build_attention_island_mask(
    scores: torch.Tensor,
    *,
    grid_side: int,
    smoothing_sigma: float = 0.85,
    min_component_size: int = 2,
    max_hole_size: int = 2,
) -> torch.Tensor:
    values = _normalize_tensor(scores).reshape(grid_side, grid_side)
    landscape = _smooth_grid(values, smoothing_sigma)
    landscape = _normalize_tensor(landscape).reshape(grid_side, grid_side)
    threshold = otsu_threshold(landscape)
    mask = landscape >= threshold
    mask = _remove_small_components(mask, min_component_size)
    mask = _fill_small_holes(mask, max_hole_size)
    return mask


def _selection_counts(
    total_tokens: int,
    ratio_percent: int,
    background_ratio_percent: int,
    background_fixed_clusters: int | None = None,
) -> tuple[int, int, int]:
    k_total = max(1, math.ceil(total_tokens * ratio_percent / 100.0))
    k_total = min(k_total, total_tokens)
    if ratio_percent >= 100:
        return k_total, k_total, 0
    if background_fixed_clusters is None:
        k_bg = max(0, math.ceil(total_tokens * background_ratio_percent / 100.0))
    else:
        k_bg = max(0, int(background_fixed_clusters))
    k_bg = min(k_bg, max(0, k_total - 1))
    return k_total, k_total - k_bg, k_bg


def object_background_budget_counts(
    *,
    total_tokens: int,
    ratio_percent: int,
    object_budget_percent: int,
    object_candidate_count: int,
) -> ObjectBackgroundBudget:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if ratio_percent <= 0 or ratio_percent > 100:
        raise ValueError(f"ratio_percent must be in 1..100, got {ratio_percent}")
    if object_budget_percent <= 0 or object_budget_percent >= 100:
        raise ValueError("object_budget_percent must be in 1..99")

    total_budget = max(1, min(total_tokens, math.ceil(total_tokens * ratio_percent / 100.0)))
    background_budget_percent = 100 - int(object_budget_percent)
    initial_background_budget = max(0, math.ceil(total_budget * background_budget_percent / 100.0))
    initial_background_budget = min(initial_background_budget, max(0, total_budget - 1))
    object_budget = max(1, total_budget - initial_background_budget)

    candidates = max(0, min(int(object_candidate_count), int(total_tokens)))
    object_count = min(object_budget, candidates if candidates > 0 else total_tokens)
    background_count = min(max(0, total_tokens - object_count), max(0, total_budget - object_count))
    return ObjectBackgroundBudget(
        total_budget=int(total_budget),
        object_budget=int(object_budget),
        initial_background_budget=int(initial_background_budget),
        object_count=int(object_count),
        background_count=int(background_count),
    )


def _coarse_cell_center_index(
    *,
    grid_side: int,
    background_pool_grid: int,
    coarse_row: int,
    coarse_col: int,
) -> int:
    row_start = coarse_row * grid_side // background_pool_grid
    row_end = (coarse_row + 1) * grid_side // background_pool_grid
    col_start = coarse_col * grid_side // background_pool_grid
    col_end = (coarse_col + 1) * grid_side // background_pool_grid
    center_row = (row_start + row_end - 1) // 2
    center_col = (col_start + col_end - 1) // 2
    return center_row * grid_side + center_col


def _region_from_indices(indices: list[int], scores: torch.Tensor, grid_side: int) -> dict[str, Any]:
    rows = [idx // grid_side for idx in indices]
    cols = [idx % grid_side for idx in indices]
    return {
        "indices": sorted(int(idx) for idx in indices),
        "score_sum": float(scores[indices].sum().item()),
        "row_sum": float(sum(rows)),
        "col_sum": float(sum(cols)),
        "min_row": int(min(rows)),
        "max_row": int(max(rows)),
        "min_col": int(min(cols)),
        "max_col": int(max(cols)),
    }


def _merge_region_dicts(a: dict[str, Any], b: dict[str, Any], scores: torch.Tensor, grid_side: int) -> dict[str, Any]:
    return _region_from_indices([*a["indices"], *b["indices"]], scores, grid_side)


def _regions_are_adjacent(a: dict[str, Any], b: dict[str, Any], grid_side: int) -> bool:
    b_cells = {(idx // grid_side, idx % grid_side) for idx in b["indices"]}
    for idx in a["indices"]:
        row = idx // grid_side
        col = idx % grid_side
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (nr, nc) in b_cells:
                return True
    return False


def _region_center(region: dict[str, Any]) -> tuple[float, float]:
    size = max(1, len(region["indices"]))
    return float(region["row_sum"]) / size, float(region["col_sum"]) / size


def _region_mean_score(region: dict[str, Any]) -> float:
    return float(region["score_sum"]) / max(1, len(region["indices"]))


def _merge_cost(a: dict[str, Any], b: dict[str, Any], *, adjacent: bool, grid_side: int) -> tuple[float, int, int]:
    ar, ac = _region_center(a)
    br, bc = _region_center(b)
    center_distance = abs(ar - br) + abs(ac - bc)
    score_distance = abs(_region_mean_score(a) - _region_mean_score(b))
    min_row = min(a["min_row"], b["min_row"])
    max_row = max(a["max_row"], b["max_row"])
    min_col = min(a["min_col"], b["min_col"])
    max_col = max(a["max_col"], b["max_col"])
    merged_size = len(a["indices"]) + len(b["indices"])
    bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
    compactness_penalty = (bbox_area - merged_size) / max(1, merged_size)
    disconnected_penalty = 100.0 if not adjacent else 0.0
    cost = disconnected_penalty + center_distance + 4.0 * score_distance + 0.35 * compactness_penalty
    return cost, min(a["indices"]), min(b["indices"])


def merge_background_regions(
    context_mask: torch.Tensor,
    scores: torch.Tensor,
    *,
    grid_side: int,
    target_region_count: int,
) -> list[dict[str, Any]]:
    context_mask = context_mask.detach().cpu().bool().reshape(grid_side, grid_side)
    scores = scores.detach().float().cpu().flatten()
    context_indices = context_mask.flatten().nonzero(as_tuple=False).flatten().tolist()
    if target_region_count <= 0 or not context_indices:
        return []
    regions = [_region_from_indices([int(idx)], scores, grid_side) for idx in context_indices]
    target_region_count = min(target_region_count, len(regions))

    while len(regions) > target_region_count:
        best: tuple[float, int, int] | None = None
        best_pair: tuple[int, int] | None = None
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                adjacent = _regions_are_adjacent(regions[i], regions[j], grid_side)
                cost = _merge_cost(regions[i], regions[j], adjacent=adjacent, grid_side=grid_side)
                if best is None or cost < best:
                    best = cost
                    best_pair = (i, j)
        assert best_pair is not None
        i, j = best_pair
        merged = _merge_region_dicts(regions[i], regions[j], scores, grid_side)
        regions = [region for idx, region in enumerate(regions) if idx not in (i, j)]
        regions.append(merged)

    return sorted(regions, key=lambda region: (region["min_row"], region["min_col"], min(region["indices"])))


def _kmeans_labels(
    points: torch.Tensor,
    k: int,
    *,
    max_iterations: int = 30,
) -> tuple[torch.Tensor, float]:
    points = points.detach().float().cpu()
    n_points = int(points.shape[0])
    if n_points == 0:
        return torch.empty(0, dtype=torch.long), 0.0
    k = max(1, min(int(k), n_points))
    if k == n_points:
        return torch.arange(n_points, dtype=torch.long), 0.0

    initial = torch.linspace(0, n_points - 1, steps=k).round().long()
    centers = points.index_select(0, initial).clone()
    labels = torch.full((n_points,), -1, dtype=torch.long)
    for _ in range(max(1, max_iterations)):
        distances = torch.cdist(points, centers).pow(2)
        next_labels = distances.argmin(dim=1)
        if torch.equal(next_labels, labels):
            break
        labels = next_labels
        next_centers = centers.clone()
        for cluster_id in range(k):
            cluster_points = points[labels == cluster_id]
            if cluster_points.numel() > 0:
                next_centers[cluster_id] = cluster_points.mean(dim=0)
        centers = next_centers

    inertia = float(torch.cdist(points, centers).pow(2).min(dim=1).values.sum().item())
    return labels, inertia


def _choose_k_by_elbow(
    points: torch.Tensor,
    *,
    k_min: int,
    k_max: int,
    elbow_threshold: float,
    max_iterations: int = 30,
) -> int:
    n_points = int(points.shape[0])
    if n_points <= 0:
        return 0
    lower = max(1, min(int(k_min), n_points))
    upper = max(lower, min(int(k_max), n_points))
    previous_inertia: float | None = None
    previous_k = lower
    for k in range(lower, upper + 1):
        _, inertia = _kmeans_labels(points, k, max_iterations=max_iterations)
        if previous_inertia is not None:
            if previous_inertia <= 1e-12:
                return previous_k
            improvement = (previous_inertia - inertia) / previous_inertia
            if improvement < elbow_threshold:
                return previous_k
        previous_inertia = inertia
        previous_k = k
    return upper


def _background_cluster_features(
    background_indices: torch.Tensor,
    scores: torch.Tensor,
    *,
    grid_side: int,
    spatial_weight: float,
    score_weight: float,
) -> torch.Tensor:
    idx = background_indices.detach().cpu().long()
    rows = (idx // grid_side).float() / max(1, grid_side - 1)
    cols = (idx % grid_side).float() / max(1, grid_side - 1)
    score_values = _normalize_tensor(scores).flatten().index_select(0, idx)
    return torch.stack(
        [
            rows * float(spatial_weight),
            cols * float(spatial_weight),
            score_values * float(score_weight),
        ],
        dim=1,
    )


def split_disconnected_label_grid(label_grid: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    labels = label_grid.detach().cpu().long()
    valid = valid_mask.detach().cpu().bool()
    if labels.shape != valid.shape:
        raise ValueError("label_grid and valid_mask must have the same shape")

    h, w = labels.shape
    visited = torch.zeros((h, w), dtype=torch.bool)
    out = torch.full((h, w), -1, dtype=torch.long)
    next_label = 0
    for row in range(h):
        for col in range(w):
            if visited[row, col] or not bool(valid[row, col]):
                continue
            old_label = int(labels[row, col].item())
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                cur_row, cur_col = stack.pop()
                out[cur_row, cur_col] = next_label
                for next_row, next_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if not (0 <= next_row < h and 0 <= next_col < w):
                        continue
                    if visited[next_row, next_col] or not bool(valid[next_row, next_col]):
                        continue
                    if int(labels[next_row, next_col].item()) != old_label:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            next_label += 1
    return out


def _regions_from_label_grid(label_grid: torch.Tensor, scores: torch.Tensor, grid_side: int) -> list[dict[str, Any]]:
    labels = label_grid.detach().cpu().long().reshape(grid_side, grid_side)
    regions: list[dict[str, Any]] = []
    for label in sorted(int(x) for x in labels.unique().tolist() if int(x) >= 0):
        indices = labels.flatten().eq(label).nonzero(as_tuple=False).flatten().tolist()
        if indices:
            region = _region_from_indices([int(idx) for idx in indices], scores, grid_side)
            region["region_id"] = len(regions)
            regions.append(region)
    return regions


def _merge_small_regions(
    regions: list[dict[str, Any]],
    scores: torch.Tensor,
    *,
    grid_side: int,
    min_region_size: int,
) -> list[dict[str, Any]]:
    if min_region_size <= 1:
        return regions
    regions = [dict(region) for region in regions]
    while len(regions) > 1:
        small_indices = [idx for idx, region in enumerate(regions) if len(region["indices"]) < min_region_size]
        if not small_indices:
            break
        i = min(small_indices, key=lambda idx: (len(regions[idx]["indices"]), min(regions[idx]["indices"])))
        best: tuple[float, int, int] | None = None
        best_j: int | None = None
        for j, region in enumerate(regions):
            if i == j:
                continue
            adjacent = _regions_are_adjacent(regions[i], region, grid_side)
            cost = _merge_cost(regions[i], region, adjacent=adjacent, grid_side=grid_side)
            if best is None or cost < best:
                best = cost
                best_j = j
        assert best_j is not None
        merged = _merge_region_dicts(regions[i], regions[best_j], scores, grid_side)
        regions = [region for idx, region in enumerate(regions) if idx not in (i, best_j)]
        regions.append(merged)

    return sorted(regions, key=lambda region: (region["min_row"], region["min_col"], min(region["indices"])))


def _merge_regions_to_target(
    regions: list[dict[str, Any]],
    scores: torch.Tensor,
    *,
    grid_side: int,
    target_region_count: int,
) -> list[dict[str, Any]]:
    if target_region_count <= 0:
        return []
    regions = [dict(region) for region in regions]
    target_region_count = min(target_region_count, len(regions))
    while len(regions) > target_region_count:
        best: tuple[float, int, int] | None = None
        best_pair: tuple[int, int] | None = None
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                adjacent = _regions_are_adjacent(regions[i], regions[j], grid_side)
                cost = _merge_cost(regions[i], regions[j], adjacent=adjacent, grid_side=grid_side)
                if best is None or cost < best:
                    best = cost
                    best_pair = (i, j)
        assert best_pair is not None
        i, j = best_pair
        merged = _merge_region_dicts(regions[i], regions[j], scores, grid_side)
        regions = [region for idx, region in enumerate(regions) if idx not in (i, j)]
        regions.append(merged)
    return sorted(regions, key=lambda region: (region["min_row"], region["min_col"], min(region["indices"])))


def _split_region_dict(region: dict[str, Any], scores: torch.Tensor, grid_side: int) -> list[dict[str, Any]]:
    indices = [int(idx) for idx in region["indices"]]
    if len(indices) <= 1:
        return [region]
    row_span = int(region["max_row"]) - int(region["min_row"])
    col_span = int(region["max_col"]) - int(region["min_col"])
    if row_span >= col_span:
        indices = sorted(indices, key=lambda idx: (idx // grid_side, idx % grid_side))
    else:
        indices = sorted(indices, key=lambda idx: (idx % grid_side, idx // grid_side))
    mid = max(1, len(indices) // 2)
    return [
        _region_from_indices(indices[:mid], scores, grid_side),
        _region_from_indices(indices[mid:], scores, grid_side),
    ]


def _split_regions_to_target(
    regions: list[dict[str, Any]],
    scores: torch.Tensor,
    *,
    grid_side: int,
    target_region_count: int,
) -> list[dict[str, Any]]:
    regions = [dict(region) for region in regions]
    while len(regions) < target_region_count:
        candidates = [idx for idx, region in enumerate(regions) if len(region["indices"]) > 1]
        if not candidates:
            break
        split_idx = max(candidates, key=lambda idx: (len(regions[idx]["indices"]), -min(regions[idx]["indices"])))
        split_regions = _split_region_dict(regions[split_idx], scores, grid_side)
        regions = [region for idx, region in enumerate(regions) if idx != split_idx]
        regions.extend(split_regions)
    return sorted(regions, key=lambda region: (region["min_row"], region["min_col"], min(region["indices"])))


def connected_kmeans_background_regions(
    context_mask: torch.Tensor,
    scores: torch.Tensor,
    *,
    grid_side: int,
    k_min: int,
    k_max: int,
    elbow_threshold: float,
    min_region_size: int,
    spatial_weight: float,
    score_weight: float,
    max_iterations: int = 30,
) -> tuple[list[dict[str, Any]], int]:
    context_mask = context_mask.detach().cpu().bool().reshape(grid_side, grid_side)
    scores = scores.detach().float().cpu().flatten()
    background_indices = context_mask.flatten().nonzero(as_tuple=False).flatten()
    if background_indices.numel() == 0:
        return [], 0

    points = _background_cluster_features(
        background_indices,
        scores,
        grid_side=grid_side,
        spatial_weight=spatial_weight,
        score_weight=score_weight,
    )
    selected_k = _choose_k_by_elbow(
        points,
        k_min=k_min,
        k_max=k_max,
        elbow_threshold=elbow_threshold,
        max_iterations=max_iterations,
    )
    labels, _ = _kmeans_labels(points, selected_k, max_iterations=max_iterations)
    label_grid = torch.full((grid_side * grid_side,), -1, dtype=torch.long)
    label_grid[background_indices] = labels
    label_grid = label_grid.reshape(grid_side, grid_side)
    connected_labels = split_disconnected_label_grid(label_grid, context_mask)
    regions = _regions_from_label_grid(connected_labels, scores, grid_side)
    regions = _merge_small_regions(regions, scores, grid_side=grid_side, min_region_size=min_region_size)
    return regions, selected_k


def connected_kmeans_background_regions_for_target(
    context_mask: torch.Tensor,
    scores: torch.Tensor,
    *,
    grid_side: int,
    target_region_count: int,
    min_region_size: int,
    spatial_weight: float,
    score_weight: float,
    max_iterations: int = 30,
) -> tuple[list[dict[str, Any]], int]:
    context_mask = context_mask.detach().cpu().bool().reshape(grid_side, grid_side)
    scores = scores.detach().float().cpu().flatten()
    background_indices = context_mask.flatten().nonzero(as_tuple=False).flatten()
    if target_region_count <= 0 or background_indices.numel() == 0:
        return [], 0

    selected_k = min(int(target_region_count), int(background_indices.numel()))
    if selected_k == int(background_indices.numel()):
        regions = [
            _region_from_indices([int(idx)], scores, grid_side)
            for idx in background_indices.tolist()
        ]
        return regions, selected_k

    points = _background_cluster_features(
        background_indices,
        scores,
        grid_side=grid_side,
        spatial_weight=spatial_weight,
        score_weight=score_weight,
    )
    labels, _ = _kmeans_labels(points, selected_k, max_iterations=max_iterations)
    label_grid = torch.full((grid_side * grid_side,), -1, dtype=torch.long)
    label_grid[background_indices] = labels
    label_grid = label_grid.reshape(grid_side, grid_side)
    connected_labels = split_disconnected_label_grid(label_grid, context_mask)
    regions = _regions_from_label_grid(connected_labels, scores, grid_side)
    if len(regions) > selected_k:
        regions = _merge_regions_to_target(
            regions,
            scores,
            grid_side=grid_side,
            target_region_count=selected_k,
        )
    elif len(regions) < selected_k:
        regions = _split_regions_to_target(
            regions,
            scores,
            grid_side=grid_side,
            target_region_count=selected_k,
        )
    return regions, selected_k


def cc_background_cluster_count(
    residual_mask: torch.Tensor,
    *,
    k_total: int,
    target_tokens_per_cluster: int,
    min_clusters: int,
    max_clusters: int,
    max_fraction: float,
) -> int:
    residual_mask = residual_mask.detach().cpu().bool()
    residual_count = int(residual_mask.sum().item())
    if residual_count <= 0 or k_total <= 1:
        return 0
    if target_tokens_per_cluster <= 0:
        raise ValueError("target_tokens_per_cluster must be positive")

    components = _connected_components(residual_mask)
    raw_count = sum(max(1, math.ceil(len(component) / target_tokens_per_cluster)) for component in components)
    max_by_fraction = max(1, int(math.floor(k_total * max_fraction)))
    upper = min(max_clusters, max_by_fraction, k_total - 1, residual_count)
    lower = min(max(0, min_clusters), upper)
    return max(lower, min(raw_count, upper))


def _region_pseudo_index(region: dict[str, Any], grid_side: int) -> int:
    row, col = _region_center(region)
    row_i = max(0, min(grid_side - 1, int(round(row))))
    col_i = max(0, min(grid_side - 1, int(round(col))))
    return row_i * grid_side + col_i


def _assemble_selection_features(
    flat_features: torch.Tensor,
    object_indices: torch.Tensor,
    summary_candidates: list[dict[str, Any]],
) -> torch.Tensor:
    entries: list[tuple[int, torch.Tensor, str]] = []
    for idx in object_indices.tolist():
        entries.append((int(idx), flat_features[int(idx)], "object"))
    for item in summary_candidates:
        entries.append((int(item["pseudo_index"]), item["feature"], "background"))
    entries.sort(key=lambda item: (item[0], 0 if item[2] == "object" else 1))
    return torch.stack([entry[1] for entry in entries], dim=0) if entries else flat_features[:0]


def _background_summary_candidates_from_regions(
    flat_features: torch.Tensor,
    scores: torch.Tensor,
    regions: list[dict[str, Any]],
    *,
    grid_side: int,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary_candidates: list[dict[str, Any]] = []
    extra_fields = extra_fields or {}
    for region_id, region in enumerate(regions):
        source_indices = [int(idx) for idx in region["indices"]]
        source_tensor = torch.tensor(source_indices, dtype=torch.long, device=flat_features.device)
        cpu_source_tensor = torch.tensor(source_indices, dtype=torch.long)
        summary = {
            "feature": flat_features.index_select(0, source_tensor).mean(dim=0),
            "score": float(scores.index_select(0, cpu_source_tensor).mean().item()),
            "pseudo_index": _region_pseudo_index(region, grid_side),
            "region_id": region_id,
            "source_indices": source_indices,
            "min_row": int(region["min_row"]),
            "max_row": int(region["max_row"]),
            "min_col": int(region["min_col"]),
            "max_col": int(region["max_col"]),
        }
        summary.update(extra_fields)
        summary_candidates.append(summary)
    return summary_candidates


def _background_summary_metadata(
    summary_candidates: list[dict[str, Any]],
    *,
    extra_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    base_keys = (
        "score",
        "pseudo_index",
        "region_id",
        "source_indices",
        "min_row",
        "max_row",
        "min_col",
        "max_col",
    )
    summaries: list[dict[str, Any]] = []
    for item in summary_candidates:
        summary: dict[str, Any] = {}
        for key in (*base_keys, *extra_keys):
            if key not in item:
                continue
            value = item[key]
            if key == "source_indices":
                summary[key] = [int(x) for x in value]
            elif key == "score":
                summary[key] = float(value)
            else:
                summary[key] = int(value)
        summaries.append(summary)
    return summaries


def selection_source_coverage(
    object_indices: torch.Tensor,
    background_summaries: list[dict[str, Any]],
    total_tokens: int,
) -> dict[str, int]:
    object_set = {int(idx) for idx in object_indices.detach().cpu().tolist()}
    background_set: set[int] = set()
    for summary in background_summaries:
        background_set.update(int(idx) for idx in summary.get("source_indices", []))

    covered = object_set | background_set
    return {
        "visual_source_object_count": len(object_set),
        "visual_source_background_count": len(background_set),
        "visual_source_coverage_count": len(covered),
        "visual_source_overlap_count": len(object_set & background_set),
        "visual_source_missing_count": max(0, int(total_tokens) - len(covered)),
    }


def island_context_pool_selection(
    flat_features: torch.Tensor,
    scores: torch.Tensor,
    *,
    ratio_percent: int,
    grid_side: int,
    background_ratio_percent: int,
    background_pool_grid: int,
    background_fixed_clusters: int | None = None,
    island_mask: torch.Tensor | None = None,
) -> IslandContextSelection:
    if flat_features.ndim != 2:
        raise ValueError(f"flat_features must be [tokens, hidden], got {tuple(flat_features.shape)}")
    scores = scores.detach().float().cpu().flatten()
    total_tokens = int(scores.numel())
    if flat_features.shape[0] != total_tokens:
        raise ValueError("flat_features and scores must have the same token count")
    if grid_side * grid_side != total_tokens:
        raise ValueError(f"grid_side {grid_side} does not match token count {total_tokens}")
    if background_pool_grid <= 0 or grid_side % background_pool_grid != 0:
        raise ValueError("background_pool_grid must evenly divide grid_side")

    k_total, k_obj, k_bg = _selection_counts(
        total_tokens,
        ratio_percent,
        background_ratio_percent,
        background_fixed_clusters,
    )
    if k_total == total_tokens and k_bg == 0:
        all_indices = torch.arange(total_tokens, dtype=torch.long)
        return IslandContextSelection(
            features=flat_features,
            object_indices=all_indices,
            background_summaries=[],
            island_mask=torch.ones((grid_side, grid_side), dtype=torch.bool),
        )

    if island_mask is None:
        island_mask = build_attention_island_mask(scores, grid_side=grid_side)
    else:
        island_mask = island_mask.detach().cpu().bool()
    if tuple(island_mask.shape) != (grid_side, grid_side):
        raise ValueError(f"island_mask must be {(grid_side, grid_side)}, got {tuple(island_mask.shape)}")

    flat_island = island_mask.flatten()
    object_candidates = flat_island.nonzero(as_tuple=False).flatten()
    if object_candidates.numel() == 0:
        object_candidates = torch.arange(total_tokens, dtype=torch.long)
    object_take = min(k_obj, int(object_candidates.numel()))
    object_scores = scores.index_select(0, object_candidates)
    selected_local = torch.topk(object_scores, k=object_take, largest=True, sorted=False).indices
    object_indices = torch.sort(object_candidates.index_select(0, selected_local)).values

    context_mask = ~flat_island
    summary_candidates: list[dict[str, Any]] = []
    for coarse_row in range(background_pool_grid):
        for coarse_col in range(background_pool_grid):
            row_start = coarse_row * grid_side // background_pool_grid
            row_end = (coarse_row + 1) * grid_side // background_pool_grid
            col_start = coarse_col * grid_side // background_pool_grid
            col_end = (coarse_col + 1) * grid_side // background_pool_grid
            source_indices: list[int] = []
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    idx = row * grid_side + col
                    if bool(context_mask[idx]):
                        source_indices.append(idx)
            if not source_indices:
                continue
            source_tensor = torch.tensor(source_indices, dtype=torch.long, device=flat_features.device)
            cpu_source_tensor = torch.tensor(source_indices, dtype=torch.long)
            summary_feature = flat_features.index_select(0, source_tensor).mean(dim=0)
            summary_score = float(scores.index_select(0, cpu_source_tensor).mean().item())
            pseudo_index = _coarse_cell_center_index(
                grid_side=grid_side,
                background_pool_grid=background_pool_grid,
                coarse_row=coarse_row,
                coarse_col=coarse_col,
            )
            summary_candidates.append(
                {
                    "feature": summary_feature,
                    "score": summary_score,
                    "pseudo_index": pseudo_index,
                    "coarse_row": coarse_row,
                    "coarse_col": coarse_col,
                    "source_indices": source_indices,
                }
            )

    summary_candidates = sorted(summary_candidates, key=lambda item: item["score"], reverse=True)[:k_bg]
    entries: list[tuple[int, torch.Tensor, str, dict[str, Any] | None]] = []
    for idx in object_indices.tolist():
        entries.append((idx, flat_features[idx], "object", None))
    for item in summary_candidates:
        entries.append((int(item["pseudo_index"]), item["feature"], "background", item))
    entries.sort(key=lambda item: (item[0], 0 if item[2] == "object" else 1))
    features = torch.stack([entry[1] for entry in entries], dim=0) if entries else flat_features[:0]

    background_summaries: list[dict[str, Any]] = []
    for item in summary_candidates:
        background_summaries.append(
            {
                "score": float(item["score"]),
                "pseudo_index": int(item["pseudo_index"]),
                "coarse_row": int(item["coarse_row"]),
                "coarse_col": int(item["coarse_col"]),
                "source_indices": [int(x) for x in item["source_indices"]],
            }
        )

    return IslandContextSelection(
        features=features,
        object_indices=object_indices,
        background_summaries=background_summaries,
        island_mask=island_mask,
    )


def island_background_merge_pool_selection(
    flat_features: torch.Tensor,
    scores: torch.Tensor,
    *,
    ratio_percent: int,
    grid_side: int,
    background_ratio_percent: int,
    background_fixed_clusters: int | None = None,
    island_mask: torch.Tensor | None = None,
) -> IslandContextSelection:
    if flat_features.ndim != 2:
        raise ValueError(f"flat_features must be [tokens, hidden], got {tuple(flat_features.shape)}")
    scores = scores.detach().float().cpu().flatten()
    total_tokens = int(scores.numel())
    if flat_features.shape[0] != total_tokens:
        raise ValueError("flat_features and scores must have the same token count")
    if grid_side * grid_side != total_tokens:
        raise ValueError(f"grid_side {grid_side} does not match token count {total_tokens}")

    k_total, k_obj, k_bg = _selection_counts(
        total_tokens,
        ratio_percent,
        background_ratio_percent,
        background_fixed_clusters,
    )
    if k_total == total_tokens and k_bg == 0:
        all_indices = torch.arange(total_tokens, dtype=torch.long)
        return IslandContextSelection(
            features=flat_features,
            object_indices=all_indices,
            background_summaries=[],
            island_mask=torch.ones((grid_side, grid_side), dtype=torch.bool),
        )

    if island_mask is None:
        island_mask = build_attention_island_mask(scores, grid_side=grid_side)
    else:
        island_mask = island_mask.detach().cpu().bool()
    if tuple(island_mask.shape) != (grid_side, grid_side):
        raise ValueError(f"island_mask must be {(grid_side, grid_side)}, got {tuple(island_mask.shape)}")

    flat_island = island_mask.flatten()
    object_candidates = flat_island.nonzero(as_tuple=False).flatten()
    if object_candidates.numel() == 0:
        object_candidates = torch.arange(total_tokens, dtype=torch.long)
    object_take = min(k_obj, int(object_candidates.numel()))
    object_scores = scores.index_select(0, object_candidates)
    selected_local = torch.topk(object_scores, k=object_take, largest=True, sorted=False).indices
    object_indices = torch.sort(object_candidates.index_select(0, selected_local)).values

    context_mask = (~flat_island).reshape(grid_side, grid_side)
    target_background_regions = min(int(context_mask.sum().item()), max(0, k_total - int(object_indices.numel())))
    merged_regions = merge_background_regions(
        context_mask,
        scores,
        grid_side=grid_side,
        target_region_count=target_background_regions,
    )

    summary_candidates = _background_summary_candidates_from_regions(
        flat_features,
        scores,
        merged_regions,
        grid_side=grid_side,
    )
    features = _assemble_selection_features(flat_features, object_indices, summary_candidates)
    background_summaries = _background_summary_metadata(summary_candidates)

    return IslandContextSelection(
        features=features,
        object_indices=object_indices,
        background_summaries=background_summaries,
        island_mask=island_mask,
    )


def object_topk_residual_cluster_pool_selection(
    flat_features: torch.Tensor,
    scores: torch.Tensor,
    *,
    ratio_percent: int,
    grid_side: int,
    background_ratio_percent: int,
    background_fixed_clusters: int | None = None,
    background_count_policy: str = "fixed",
    background_target_tokens_per_cluster: int = 12,
    background_min_clusters: int = 4,
    background_max_clusters: int = 16,
    background_max_fraction: float = 0.25,
    island_mask: torch.Tensor | None = None,
) -> IslandContextSelection:
    if flat_features.ndim != 2:
        raise ValueError(f"flat_features must be [tokens, hidden], got {tuple(flat_features.shape)}")
    scores = scores.detach().float().cpu().flatten()
    total_tokens = int(scores.numel())
    if flat_features.shape[0] != total_tokens:
        raise ValueError("flat_features and scores must have the same token count")
    if grid_side * grid_side != total_tokens:
        raise ValueError(f"grid_side {grid_side} does not match token count {total_tokens}")

    k_total, k_obj, k_bg = _selection_counts(
        total_tokens,
        ratio_percent,
        background_ratio_percent,
        background_fixed_clusters if background_count_policy == "fixed" else None,
    )
    if k_total == total_tokens and k_bg == 0:
        all_indices = torch.arange(total_tokens, dtype=torch.long)
        return IslandContextSelection(
            features=flat_features,
            object_indices=all_indices,
            background_summaries=[],
            island_mask=torch.ones((grid_side, grid_side), dtype=torch.bool),
        )

    if island_mask is None:
        island_mask = build_attention_island_mask(scores, grid_side=grid_side)
    else:
        island_mask = island_mask.detach().cpu().bool()
    if tuple(island_mask.shape) != (grid_side, grid_side):
        raise ValueError(f"island_mask must be {(grid_side, grid_side)}, got {tuple(island_mask.shape)}")

    flat_island = island_mask.flatten()
    object_candidates = flat_island.nonzero(as_tuple=False).flatten()
    if object_candidates.numel() == 0:
        object_candidates = torch.arange(total_tokens, dtype=torch.long)

    def select_object_indices(background_count: int) -> torch.Tensor:
        object_budget = max(1, min(k_total - background_count, int(object_candidates.numel())))
        object_scores = scores.index_select(0, object_candidates)
        selected_local = torch.topk(object_scores, k=object_budget, largest=True, sorted=False).indices
        return torch.sort(object_candidates.index_select(0, selected_local)).values

    object_indices = select_object_indices(k_bg)
    if background_count_policy == "cc":
        for _ in range(3):
            residual_mask_for_count = torch.ones(total_tokens, dtype=torch.bool)
            residual_mask_for_count[object_indices.cpu()] = False
            next_k_bg = cc_background_cluster_count(
                residual_mask_for_count.reshape(grid_side, grid_side),
                k_total=k_total,
                target_tokens_per_cluster=background_target_tokens_per_cluster,
                min_clusters=background_min_clusters,
                max_clusters=background_max_clusters,
                max_fraction=background_max_fraction,
            )
            if next_k_bg == k_bg:
                break
            k_bg = next_k_bg
            object_indices = select_object_indices(k_bg)
    elif background_count_policy != "fixed":
        raise ValueError(f"unknown background_count_policy: {background_count_policy}")

    residual_mask = torch.ones(total_tokens, dtype=torch.bool)
    residual_mask[object_indices.cpu()] = False
    target_background_regions = min(
        int(residual_mask.sum().item()),
        max(0, k_total - int(object_indices.numel())),
    )
    merged_regions = merge_background_regions(
        residual_mask.reshape(grid_side, grid_side),
        scores,
        grid_side=grid_side,
        target_region_count=target_background_regions,
    )

    summary_candidates = _background_summary_candidates_from_regions(
        flat_features,
        scores,
        merged_regions,
        grid_side=grid_side,
    )
    features = _assemble_selection_features(flat_features, object_indices, summary_candidates)
    background_summaries = _background_summary_metadata(summary_candidates)

    return IslandContextSelection(
        features=features,
        object_indices=object_indices,
        background_summaries=background_summaries,
        island_mask=island_mask,
    )


def object_all_background_kmeans_selection(
    flat_features: torch.Tensor,
    scores: torch.Tensor,
    *,
    grid_side: int,
    background_k_min: int = 4,
    background_k_max: int = 24,
    background_k_elbow_threshold: float = 0.08,
    background_min_region_size: int = 2,
    background_spatial_weight: float = 1.0,
    background_score_weight: float = 0.35,
    island_mask: torch.Tensor | None = None,
) -> IslandContextSelection:
    if flat_features.ndim != 2:
        raise ValueError(f"flat_features must be [tokens, hidden], got {tuple(flat_features.shape)}")
    scores = scores.detach().float().cpu().flatten()
    total_tokens = int(scores.numel())
    if flat_features.shape[0] != total_tokens:
        raise ValueError("flat_features and scores must have the same token count")
    if grid_side * grid_side != total_tokens:
        raise ValueError(f"grid_side {grid_side} does not match token count {total_tokens}")

    if island_mask is None:
        island_mask = build_attention_island_mask(scores, grid_side=grid_side)
    else:
        island_mask = island_mask.detach().cpu().bool()
    if tuple(island_mask.shape) != (grid_side, grid_side):
        raise ValueError(f"island_mask must be {(grid_side, grid_side)}, got {tuple(island_mask.shape)}")

    flat_island = island_mask.flatten()
    object_indices = torch.sort(flat_island.nonzero(as_tuple=False).flatten()).values
    if object_indices.numel() == total_tokens:
        return IslandContextSelection(
            features=flat_features,
            object_indices=object_indices,
            background_summaries=[],
            island_mask=island_mask,
        )

    context_mask = (~flat_island).reshape(grid_side, grid_side)
    merged_regions, selected_k = connected_kmeans_background_regions(
        context_mask,
        scores,
        grid_side=grid_side,
        k_min=background_k_min,
        k_max=background_k_max,
        elbow_threshold=background_k_elbow_threshold,
        min_region_size=background_min_region_size,
        spatial_weight=background_spatial_weight,
        score_weight=background_score_weight,
    )

    summary_candidates = _background_summary_candidates_from_regions(
        flat_features,
        scores,
        merged_regions,
        grid_side=grid_side,
        extra_fields={"kmeans_raw_k": int(selected_k)},
    )
    features = _assemble_selection_features(flat_features, object_indices, summary_candidates)
    background_summaries = _background_summary_metadata(
        summary_candidates,
        extra_keys=("kmeans_raw_k",),
    )

    return IslandContextSelection(
        features=features,
        object_indices=object_indices,
        background_summaries=background_summaries,
        island_mask=island_mask,
    )


def object_budget_background_kmeans_selection(
    flat_features: torch.Tensor,
    scores: torch.Tensor,
    *,
    ratio_percent: int,
    grid_side: int,
    object_budget_percent: int = 90,
    background_min_region_size: int = 2,
    background_spatial_weight: float = 1.0,
    background_score_weight: float = 0.35,
    island_mask: torch.Tensor | None = None,
) -> IslandContextSelection:
    if flat_features.ndim != 2:
        raise ValueError(f"flat_features must be [tokens, hidden], got {tuple(flat_features.shape)}")
    scores = scores.detach().float().cpu().flatten()
    total_tokens = int(scores.numel())
    if flat_features.shape[0] != total_tokens:
        raise ValueError("flat_features and scores must have the same token count")
    if grid_side * grid_side != total_tokens:
        raise ValueError(f"grid_side {grid_side} does not match token count {total_tokens}")

    if island_mask is None:
        island_mask = build_attention_island_mask(scores, grid_side=grid_side)
    else:
        island_mask = island_mask.detach().cpu().bool()
    if tuple(island_mask.shape) != (grid_side, grid_side):
        raise ValueError(f"island_mask must be {(grid_side, grid_side)}, got {tuple(island_mask.shape)}")

    flat_island = island_mask.flatten()
    object_candidates = flat_island.nonzero(as_tuple=False).flatten()
    if object_candidates.numel() == 0:
        object_candidates = torch.arange(total_tokens, dtype=torch.long)

    budget = object_background_budget_counts(
        total_tokens=total_tokens,
        ratio_percent=ratio_percent,
        object_budget_percent=object_budget_percent,
        object_candidate_count=int(object_candidates.numel()),
    )
    background_budget_percent = 100 - int(object_budget_percent)

    object_take = budget.object_count
    object_scores = scores.index_select(0, object_candidates)
    selected_local = torch.topk(object_scores, k=object_take, largest=True, sorted=False).indices
    object_indices = torch.sort(object_candidates.index_select(0, selected_local)).values

    residual_mask = torch.ones(total_tokens, dtype=torch.bool)
    residual_mask[object_indices.cpu()] = False
    merged_regions, selected_k = connected_kmeans_background_regions_for_target(
        residual_mask.reshape(grid_side, grid_side),
        scores,
        grid_side=grid_side,
        target_region_count=budget.background_count,
        min_region_size=background_min_region_size,
        spatial_weight=background_spatial_weight,
        score_weight=background_score_weight,
    )

    summary_candidates = _background_summary_candidates_from_regions(
        flat_features,
        scores,
        merged_regions,
        grid_side=grid_side,
        extra_fields={
            "kmeans_target_k": int(selected_k),
            "total_budget": int(budget.total_budget),
            "object_budget": int(budget.object_budget),
            "initial_background_budget": int(budget.initial_background_budget),
            "effective_background_budget": int(budget.background_count),
            "object_budget_percent": int(object_budget_percent),
            "background_budget_percent": int(background_budget_percent),
        },
    )
    features = _assemble_selection_features(flat_features, object_indices, summary_candidates)
    background_summaries = _background_summary_metadata(
        summary_candidates,
        extra_keys=(
            "kmeans_target_k",
            "total_budget",
            "object_budget",
            "initial_background_budget",
            "effective_background_budget",
            "object_budget_percent",
            "background_budget_percent",
        ),
    )

    return IslandContextSelection(
        features=features,
        object_indices=object_indices,
        background_summaries=background_summaries,
        island_mask=island_mask,
    )


def uses_pooled_selection_policy(policy: str) -> bool:
    return policy in POOLED_SELECTION_POLICIES


def _background_fixed_clusters_for_args(args: argparse.Namespace) -> int | None:
    if args.background_count_policy != "fixed":
        return None
    return args.background_fixed_clusters


def has_single_square_tile(*, grid_side: int, num_patches_list: list[int]) -> bool:
    return bool(grid_side and len(num_patches_list) == 1 and num_patches_list[0] == 1)


def _require_single_square_tile(
    policy: str,
    *,
    grid_side: int,
    num_patches_list: list[int],
) -> None:
    if not has_single_square_tile(grid_side=grid_side, num_patches_list=num_patches_list):
        raise ValueError(f"{policy} currently requires one square image tile")


def _selection_details(
    args: argparse.Namespace,
    *,
    object_indices: torch.Tensor,
    background_summaries: list[dict[str, Any]],
    island_mask: torch.Tensor | None,
    total_tokens: int,
) -> dict[str, Any]:
    background_fixed_clusters = _background_fixed_clusters_for_args(args)
    details: dict[str, Any] = {
        "policy": args.selection_policy,
        "object_original_token_count": int(object_indices.numel()),
        "background_summary_token_count": len(background_summaries),
        **selection_source_coverage(object_indices, background_summaries, total_tokens),
    }

    if uses_pooled_selection_policy(args.selection_policy):
        details.update(
            {
                "background_pool_grid": int(args.background_pool_grid),
                "background_ratio_percent": int(args.background_ratio_percent),
                "background_fixed_clusters": (
                    None if background_fixed_clusters is None else int(background_fixed_clusters)
                ),
                "background_count_policy": args.background_count_policy,
                "background_target_tokens_per_cluster": int(args.background_target_tokens_per_cluster),
                "background_min_clusters": int(args.background_min_clusters),
                "background_max_clusters": int(args.background_max_clusters),
                "background_max_fraction": float(args.background_max_fraction),
                "background_k_min": int(args.background_k_min),
                "background_k_max": int(args.background_k_max),
                "background_k_elbow_threshold": float(args.background_k_elbow_threshold),
                "background_min_region_size": int(args.background_min_region_size),
                "background_spatial_weight": float(args.background_spatial_weight),
                "background_score_weight": float(args.background_score_weight),
                "object_budget_percent": int(args.object_budget_percent),
                "background_budget_percent": int(100 - args.object_budget_percent),
                "background_summaries": background_summaries,
            }
        )
        if island_mask is not None:
            details["island_mask"] = island_mask.int().cpu().tolist()

    return details


def select_visual_features_for_ratio(
    args: argparse.Namespace,
    flat_features: torch.Tensor,
    scores: torch.Tensor,
    *,
    ratio_percent: int,
    grid_side: int,
    num_patches_list: list[int],
) -> RatioSelection:
    total_tokens = int(scores.numel())
    background_summaries: list[dict[str, Any]] = []
    island_mask: torch.Tensor | None = None

    if uses_pooled_selection_policy(args.selection_policy):
        _require_single_square_tile(args.selection_policy, grid_side=grid_side, num_patches_list=num_patches_list)
        background_fixed_clusters = _background_fixed_clusters_for_args(args)
        if args.selection_policy == SELECTION_POLICY_OBJECT_TOPK_RESIDUAL_CLUSTER_POOL:
            selection = object_topk_residual_cluster_pool_selection(
                flat_features,
                scores,
                ratio_percent=ratio_percent,
                grid_side=grid_side,
                background_ratio_percent=args.background_ratio_percent,
                background_fixed_clusters=background_fixed_clusters,
                background_count_policy=args.background_count_policy,
                background_target_tokens_per_cluster=args.background_target_tokens_per_cluster,
                background_min_clusters=args.background_min_clusters,
                background_max_clusters=args.background_max_clusters,
                background_max_fraction=args.background_max_fraction,
            )
        elif args.selection_policy == SELECTION_POLICY_OBJECT_ALL_BACKGROUND_KMEANS:
            selection = object_all_background_kmeans_selection(
                flat_features,
                scores,
                grid_side=grid_side,
                background_k_min=args.background_k_min,
                background_k_max=args.background_k_max,
                background_k_elbow_threshold=args.background_k_elbow_threshold,
                background_min_region_size=args.background_min_region_size,
                background_spatial_weight=args.background_spatial_weight,
                background_score_weight=args.background_score_weight,
            )
        elif args.selection_policy == SELECTION_POLICY_OBJECT_BUDGET_BACKGROUND_KMEANS:
            selection = object_budget_background_kmeans_selection(
                flat_features,
                scores,
                ratio_percent=ratio_percent,
                grid_side=grid_side,
                object_budget_percent=args.object_budget_percent,
                background_min_region_size=args.background_min_region_size,
                background_spatial_weight=args.background_spatial_weight,
                background_score_weight=args.background_score_weight,
            )
        elif args.selection_policy == SELECTION_POLICY_ISLAND_BACKGROUND_MERGE_POOL:
            selection = island_background_merge_pool_selection(
                flat_features,
                scores,
                ratio_percent=ratio_percent,
                grid_side=grid_side,
                background_ratio_percent=args.background_ratio_percent,
                background_fixed_clusters=background_fixed_clusters,
            )
        elif args.selection_policy == SELECTION_POLICY_ISLAND_CONTEXT_POOL:
            selection = island_context_pool_selection(
                flat_features,
                scores,
                ratio_percent=ratio_percent,
                grid_side=grid_side,
                background_ratio_percent=args.background_ratio_percent,
                background_pool_grid=args.background_pool_grid,
                background_fixed_clusters=background_fixed_clusters,
            )
        else:
            raise ValueError(f"unknown selection policy: {args.selection_policy}")

        selected_indices = selection.object_indices
        selected_features = selection.features
        background_summaries = selection.background_summaries
        island_mask = selection.island_mask
    else:
        selected_indices = select_topk_indices(scores, ratio_percent)
        selected_features = flat_features.index_select(0, selected_indices.to(flat_features.device))

    return RatioSelection(
        features=selected_features,
        object_indices=selected_indices,
        background_summaries=background_summaries,
        details=_selection_details(
            args,
            object_indices=selected_indices,
            background_summaries=background_summaries,
            island_mask=island_mask,
            total_tokens=total_tokens,
        ),
    )


def aggregate_visual_attention(
    attentions: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
    visual_positions: torch.Tensor,
    text_positions: torch.Tensor,
) -> torch.Tensor:
    if visual_positions.numel() == 0:
        raise ValueError("visual_positions must not be empty")
    if text_positions.numel() == 0:
        raise ValueError("text_positions must not be empty")

    per_layer: list[torch.Tensor] = []
    for attention in attentions:
        if attention is None:
            continue
        if attention.ndim != 4:
            raise ValueError(f"attention tensors must be [batch, heads, query, key], got {tuple(attention.shape)}")
        attn = attention.detach().float()
        vpos = visual_positions.to(device=attn.device, dtype=torch.long)
        tpos = text_positions.to(device=attn.device, dtype=torch.long)
        query_to_visual = attn[0].index_select(1, tpos).index_select(2, vpos)
        per_layer.append(query_to_visual.mean(dim=(0, 1)).cpu())

    if not per_layer:
        raise ValueError("no attention tensors were returned; use an eager attention implementation")
    return torch.stack(per_layer, dim=0).mean(dim=0)


def cls_patch_attention(
    attentions: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    layer_index: int = -1,
) -> torch.Tensor:
    if not attentions:
        raise ValueError("attentions must not be empty")
    attention = attentions[layer_index].detach().float()
    if attention.ndim != 4:
        raise ValueError(f"attention tensors must be [batch, heads, query, key], got {tuple(attention.shape)}")
    return attention[0, :, 0, 1:].mean(dim=0).cpu()


def vit_attention_rollout(attentions: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    if not attentions:
        raise ValueError("attentions must not be empty")
    first = attentions[0]
    n_tokens = first.shape[-1]
    joint = torch.eye(n_tokens, dtype=torch.float32)
    for attention in attentions:
        attn = attention.detach().float()[0].mean(dim=0).cpu()
        aug = attn + torch.eye(n_tokens, dtype=torch.float32)
        aug = aug / aug.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        joint = aug @ joint
    return joint[0, 1:]


def patch_incoming_attention(attentions: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    if not attentions:
        raise ValueError("attentions must not be empty")
    per_layer: list[torch.Tensor] = []
    for attention in attentions:
        attn = attention.detach().float()
        if attn.ndim != 4:
            raise ValueError(f"attention tensors must be [batch, heads, query, key], got {tuple(attn.shape)}")
        patch_to_patch = attn[0, :, 1:, 1:]
        per_layer.append(patch_to_patch.mean(dim=(0, 1)).cpu())
    return torch.stack(per_layer, dim=0).mean(dim=0)


def patch_attention_rollout(attentions: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    if not attentions:
        raise ValueError("attentions must not be empty")
    first = attentions[0]
    if first.ndim != 4:
        raise ValueError(f"attention tensors must be [batch, heads, query, key], got {tuple(first.shape)}")
    n_tokens = int(first.shape[-1])
    if n_tokens < 2:
        raise ValueError("patch attention rollout needs at least one CLS token and one patch token")
    n_patches = n_tokens - 1
    joint = torch.eye(n_patches, dtype=torch.float32)
    identity = torch.eye(n_patches, dtype=torch.float32)
    for attention in attentions:
        attn = attention.detach().float()
        if attn.ndim != 4:
            raise ValueError(f"attention tensors must be [batch, heads, query, key], got {tuple(attn.shape)}")
        if int(attn.shape[-1]) != n_tokens or int(attn.shape[-2]) != n_tokens:
            raise ValueError("all attention tensors must use the same query/key token count")
        patch_to_patch = attn[0, :, 1:, 1:].mean(dim=0).cpu()
        aug = patch_to_patch + identity
        aug = aug / aug.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        joint = aug @ joint
    return joint.mean(dim=0)


def downsample_grid_average(grid: torch.Tensor, target_side: int) -> torch.Tensor:
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError(f"grid must be square 2-D, got shape {tuple(grid.shape)}")
    side = int(grid.shape[0])
    if target_side <= 0 or side % target_side != 0:
        raise ValueError(f"target_side {target_side} must divide grid side {side}")
    factor = side // target_side
    return grid.reshape(target_side, factor, target_side, factor).mean(dim=(1, 3))


def visual_token_scores_from_vit_patch_scores(
    patch_scores: torch.Tensor,
    *,
    target_token_count: int,
) -> torch.Tensor:
    patch_scores = patch_scores.detach().float().cpu().flatten()
    patch_side = square_grid_size(int(patch_scores.numel()))
    target_side = square_grid_size(int(target_token_count))
    if patch_side == target_side:
        return patch_scores
    downsampled = downsample_grid_average(patch_scores.reshape(patch_side, patch_side), target_side)
    return downsampled.flatten()


def square_grid_size(token_count: int) -> int:
    side = int(math.isqrt(token_count))
    if side * side != token_count:
        raise ValueError(f"visual token count {token_count} does not form a square grid")
    return side


def percent_reduction(value: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - value) * 100.0 / baseline


def build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image_pixel_values(
    image_path: Path,
    *,
    input_size: int,
    max_num: int,
) -> tuple[torch.Tensor, list[int], Image.Image]:
    image = Image.open(image_path).convert("RGB")
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    transform = build_transform(input_size)
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values, [len(tiles)], image


def get_device(model) -> torch.device:
    return next(model.parameters()).device


def build_expanded_query(
    model,
    tokenizer,
    question: str,
    *,
    visual_token_count: int,
) -> tuple[str, int | None, str]:
    parent_pkg = model.__class__.__module__.rsplit(".", 1)[0]
    conv = importlib.import_module(parent_pkg + ".conversation")
    template = conv.get_conv_template(model.template)
    template.system_message = model.system_message

    q = question if "<image>" in question else "<image>\n" + question
    template.append_message(template.roles[0], q)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * visual_token_count + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)

    sep = template.sep.strip()
    eos_token_id = tokenizer.convert_tokens_to_ids(sep) if sep else None
    if eos_token_id == tokenizer.unk_token_id:
        eos_token_id = None
    return query, eos_token_id, sep


def prepare_inputs_with_visual_embeds(
    model,
    tokenizer,
    query: str,
    visual_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = get_device(model)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    batch, seq_len, hidden = input_embeds.shape
    flat_embeds = input_embeds.reshape(batch * seq_len, hidden)
    flat_ids = input_ids.reshape(batch * seq_len)
    selected = flat_ids == model.img_context_token_id
    selected_count = int(selected.sum().item())

    features = visual_features.reshape(-1, hidden).to(device=device, dtype=flat_embeds.dtype)
    if selected_count != features.shape[0]:
        raise ValueError(
            f"prompt has {selected_count} IMG_CONTEXT tokens but visual_features has {features.shape[0]} rows"
        )
    flat_embeds[selected] = features
    input_embeds = flat_embeds.reshape(batch, seq_len, hidden)

    visual_positions = selected.reshape(batch, seq_len)[0].nonzero(as_tuple=False).flatten().detach().cpu()
    active_positions = attention_mask[0].bool().detach().cpu()
    visual_mask = selected.reshape(batch, seq_len)[0].detach().cpu()
    text_positions = (active_positions & ~visual_mask).nonzero(as_tuple=False).flatten()
    return input_ids, attention_mask, input_embeds, visual_positions, text_positions


def normalize_scores(scores: torch.Tensor) -> np.ndarray:
    arr = scores.detach().float().cpu().numpy()
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def write_heatmap_overlay(image: Image.Image, scores: torch.Tensor, grid_side: int, path: Path) -> None:
    norm = normalize_scores(scores).reshape(grid_side, grid_side)
    heat = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize(image.size, Image.Resampling.BICUBIC)
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay.putalpha(heat.point(lambda p: int(p * 0.70)))
    out = Image.alpha_composite(image.convert("RGBA"), overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(path)


def write_side_by_side_attention_panel(
    image: Image.Image,
    scores: torch.Tensor,
    grid_side: int,
    path: Path,
    *,
    title: str,
) -> None:
    norm = normalize_scores(scores).reshape(grid_side, grid_side)
    src = image.convert("RGB").resize((320, 320), Image.Resampling.BICUBIC)
    left = src.filter(ImageFilter.GaussianBlur(radius=0.6))

    x = np.clip(norm, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    heat_rgb = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
    heat = Image.fromarray(heat_rgb, mode="RGB").resize((320, 320), Image.Resampling.BICUBIC)
    right = Image.blend(src, heat, alpha=0.70)

    title_h = 48
    border = 3
    canvas = Image.new("RGB", (320 * 2 + border * 3, title_h + 320 + border), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    text_w = draw.textbbox((0, 0), title, font=font)[2]
    draw.text(((canvas.width - text_w) // 2, 6), title, fill=(0, 130, 0), font=font)

    y0 = title_h
    draw.rectangle([0, y0, canvas.width - 1, y0 + 320 + border - 1], outline="black", width=border)
    canvas.paste(left, (border, y0 + border))
    canvas.paste(right, (border * 2 + 320, y0 + border))
    for dx in range(border):
        x_mid = border + 320 + dx
        draw.line((x_mid, y0, x_mid, y0 + 320 + border), fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_vit_attention_side_by_side_panel(
    image: Image.Image,
    patch_scores: torch.Tensor,
    patch_side: int,
    path: Path,
    *,
    title: str,
) -> None:
    write_side_by_side_attention_panel(image, patch_scores, patch_side, path, title=title)


def _capturing_naive_attn(self, x):
    batch, seq_len, channels = x.shape
    qkv = self.qkv(x).reshape(
        batch, seq_len, 3, self.num_heads, channels // self.num_heads
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if self.qk_normalization:
        bsz, n_heads, n_tokens, head_dim = q.shape
        q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(
            bsz, n_tokens, n_heads, head_dim
        ).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(
            bsz, n_tokens, n_heads, head_dim
        ).transpose(1, 2)

    attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    self._svlm_last_attn = attn.detach().float().cpu()
    attn = self.attn_drop(attn)

    out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, channels)
    out = self.proj(out)
    out = self.proj_drop(out)
    return out


def capture_vit_attentions(model, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    patched = []
    for layer in model.vision_model.encoder.layers:
        attn = layer.attn
        patched.append((attn, attn._naive_attn, attn.use_flash_attn))
        attn.use_flash_attn = False
        attn._naive_attn = MethodType(_capturing_naive_attn, attn)
        if hasattr(attn, "_svlm_last_attn"):
            delattr(attn, "_svlm_last_attn")
    try:
        with torch.no_grad():
            model.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        captured = []
        for layer in model.vision_model.encoder.layers:
            attn = layer.attn
            if not hasattr(attn, "_svlm_last_attn"):
                raise RuntimeError("failed to capture ViT attention from one vision layer")
            captured.append(attn._svlm_last_attn)
        return tuple(captured)
    finally:
        for attn, original_naive, original_use_flash in patched:
            attn._naive_attn = original_naive
            attn.use_flash_attn = original_use_flash


def compute_importance_scores(
    args: argparse.Namespace,
    model,
    pixel_values: torch.Tensor,
    reference,
    visual_positions: torch.Tensor,
    text_positions: torch.Tensor,
    *,
    grid_side: int,
    num_patches_list: list[int],
) -> ImportanceScores:
    llm_scores = aggregate_visual_attention(reference.attentions, visual_positions, text_positions)
    score_basis = "mean attention from non-visual text positions to IMG_CONTEXT positions"
    scores = llm_scores

    if not has_single_square_tile(grid_side=grid_side, num_patches_list=num_patches_list):
        return ImportanceScores(scores=scores, score_basis=score_basis)

    vit_attentions = capture_vit_attentions(model, pixel_values)
    vit_last_scores = cls_patch_attention(vit_attentions, layer_index=-1)
    vit_rollout_scores = vit_attention_rollout(vit_attentions)
    vit_patch_scores = patch_incoming_attention(vit_attentions)
    vit_patch_rollout_scores = patch_attention_rollout(vit_attentions)

    if args.importance_source == "vit-rollout":
        scores = visual_token_scores_from_vit_patch_scores(
            vit_rollout_scores,
            target_token_count=model.num_image_token,
        )
        score_basis = "ViT CLS attention rollout downsampled from patch grid to IMG_CONTEXT grid"
    elif args.importance_source == "vit-patch-incoming":
        scores = visual_token_scores_from_vit_patch_scores(
            vit_patch_scores,
            target_token_count=model.num_image_token,
        )
        score_basis = "ViT CLS-excluded patch-to-patch incoming attention downsampled to IMG_CONTEXT grid"
    elif args.importance_source == "vit-patch-rollout":
        scores = visual_token_scores_from_vit_patch_scores(
            vit_patch_rollout_scores,
            target_token_count=model.num_image_token,
        )
        score_basis = "ViT CLS-excluded patch-to-patch attention rollout downsampled to IMG_CONTEXT grid"

    return ImportanceScores(
        scores=scores,
        score_basis=score_basis,
        vit_last_scores=vit_last_scores,
        vit_rollout_scores=vit_rollout_scores,
        vit_patch_scores=vit_patch_scores,
        vit_patch_rollout_scores=vit_patch_rollout_scores,
    )


def write_selection_overlay(
    image: Image.Image,
    selected_indices: torch.Tensor,
    token_count: int,
    grid_side: int,
    path: Path,
) -> None:
    mask = np.zeros(token_count, dtype=np.uint8)
    mask[selected_indices.detach().cpu().numpy()] = 255
    mask = mask.reshape(grid_side, grid_side)
    alpha = Image.fromarray(mask, mode="L").resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", image.size, (30, 144, 255, 0))
    overlay.putalpha(alpha.point(lambda p: 130 if p else 0))
    out = Image.alpha_composite(image.convert("RGBA"), overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(path)


def write_island_context_selection_overlay(
    image: Image.Image,
    object_indices: torch.Tensor,
    background_summaries: list[dict[str, Any]],
    *,
    token_count: int,
    grid_side: int,
    background_pool_grid: int,
    path: Path,
) -> None:
    if grid_side * grid_side != token_count:
        raise ValueError("grid_side must match token_count")
    cell_w = image.width / grid_side
    cell_h = image.height / grid_side
    out = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    labels = np.full((grid_side, grid_side), -1, dtype=np.int32)
    for summary_idx, summary in enumerate(background_summaries):
        source_indices = summary.get("source_indices")
        if source_indices:
            for idx in source_indices:
                row = int(idx) // grid_side
                col = int(idx) % grid_side
                labels[row, col] = summary_idx
            continue

        coarse_row = int(summary["coarse_row"])
        coarse_col = int(summary["coarse_col"])
        row_start = coarse_row * grid_side // background_pool_grid
        row_end = (coarse_row + 1) * grid_side // background_pool_grid
        col_start = coarse_col * grid_side // background_pool_grid
        col_end = (coarse_col + 1) * grid_side // background_pool_grid
        labels[row_start:row_end, col_start:col_end] = summary_idx

    for idx in object_indices.detach().cpu().tolist():
        row = int(idx) // grid_side
        col = int(idx) % grid_side
        labels[row, col] = -2

    for row in range(grid_side):
        for col in range(grid_side):
            label = int(labels[row, col])
            if label == -1:
                continue
            x0 = round(col * cell_w)
            y0 = round(row * cell_h)
            x1 = round((col + 1) * cell_w)
            y1 = round((row + 1) * cell_h)
            fill = (255, 45, 35, 150) if label == -2 else (30, 144, 255, 95)
            draw.rectangle([x0, y0, x1, y1], fill=fill)

    for idx in range(grid_side + 1):
        x = round(idx * cell_w)
        y = round(idx * cell_h)
        draw.line([(x, 0), (x, image.height)], fill=(255, 255, 255, 45), width=1)
        draw.line([(0, y), (image.width, y)], fill=(255, 255, 255, 45), width=1)

    for row in range(grid_side):
        for col in range(grid_side):
            label = int(labels[row, col])
            if label == -1:
                continue
            x0 = round(col * cell_w)
            y0 = round(row * cell_h)
            x1 = round((col + 1) * cell_w)
            y1 = round((row + 1) * cell_h)
            color = (255, 0, 0, 245) if label == -2 else (0, 90, 255, 245)
            width = 3
            top = int(labels[row - 1, col]) if row > 0 else None
            bottom = int(labels[row + 1, col]) if row < grid_side - 1 else None
            left = int(labels[row, col - 1]) if col > 0 else None
            right = int(labels[row, col + 1]) if col < grid_side - 1 else None
            if top != label:
                draw.line([(x0, y0), (x1, y0)], fill=color, width=width)
            if bottom != label:
                draw.line([(x0, y1), (x1, y1)], fill=color, width=width)
            if left != label:
                draw.line([(x0, y0), (x0, y1)], fill=color, width=width)
            if right != label:
                draw.line([(x1, y0), (x1, y1)], fill=color, width=width)

    out = Image.alpha_composite(out, overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(path)


def write_selection_visualization(
    image: Image.Image,
    object_indices: torch.Tensor,
    background_summaries: list[dict[str, Any]],
    *,
    selection_policy: str,
    token_count: int,
    grid_side: int,
    background_pool_grid: int,
    path: Path,
) -> None:
    if uses_pooled_selection_policy(selection_policy):
        write_island_context_selection_overlay(
            image,
            object_indices,
            background_summaries,
            token_count=token_count,
            grid_side=grid_side,
            background_pool_grid=background_pool_grid,
            path=path,
        )
        return

    write_selection_overlay(
        image,
        object_indices,
        token_count=token_count,
        grid_side=grid_side,
        path=path,
    )


def write_attention_artifacts(
    image: Image.Image,
    run_dir: Path,
    scores: torch.Tensor,
    grid_side: int,
    *,
    vit_last_scores: torch.Tensor | None = None,
    vit_rollout_scores: torch.Tensor | None = None,
    vit_patch_scores: torch.Tensor | None = None,
    vit_patch_rollout_scores: torch.Tensor | None = None,
) -> str:
    heatmap_name = "attention_heatmap_full.png"
    write_heatmap_overlay(image, scores, grid_side, run_dir / heatmap_name)
    write_side_by_side_attention_panel(
        image,
        scores,
        grid_side,
        run_dir / "attention_100_side_by_side_style.png",
        title="100% visual tokens / pred: full",
    )

    if (
        vit_last_scores is None
        or vit_rollout_scores is None
        or vit_patch_scores is None
        or vit_patch_rollout_scores is None
    ):
        return heatmap_name

    vit_patch_side = square_grid_size(int(vit_last_scores.numel()))
    raw_artifacts = (
        ("vit_last_layer_cls_attention", vit_last_scores, "ViT last-layer CLS attention"),
        ("vit_rollout_cls_attention", vit_rollout_scores, "ViT CLS attention rollout"),
        ("vit_patch_incoming_attention", vit_patch_scores, "ViT patch incoming attention"),
        ("vit_patch_rollout_attention", vit_patch_rollout_scores, "ViT patch attention rollout"),
    )
    for stem, artifact_scores, title in raw_artifacts:
        np.savetxt(
            run_dir / f"{stem}_grid.tsv",
            artifact_scores.reshape(vit_patch_side, vit_patch_side).numpy(),
            fmt="%.9f",
            delimiter="\t",
        )
        write_vit_attention_side_by_side_panel(
            image,
            artifact_scores,
            vit_patch_side,
            run_dir / f"{stem}_side_by_side.png",
            title=title,
        )

    if vit_patch_side % grid_side == 0:
        downsampled_artifacts = (
            ("vit_last_layer_cls_attention", vit_last_scores, "ViT CLS attention downsampled to 16x16"),
            ("vit_patch_incoming_attention", vit_patch_scores, "ViT patch incoming attention 16x16"),
            ("vit_patch_rollout_attention", vit_patch_rollout_scores, "ViT patch attention rollout 16x16"),
        )
        for stem, artifact_scores, title in downsampled_artifacts:
            downsampled = downsample_grid_average(artifact_scores.reshape(vit_patch_side, vit_patch_side), grid_side)
            np.savetxt(
                run_dir / f"{stem}_downsampled_16_grid.tsv",
                downsampled.numpy(),
                fmt="%.9f",
                delimiter="\t",
            )
            write_vit_attention_side_by_side_panel(
                image,
                downsampled.flatten(),
                grid_side,
                run_dir / f"{stem}_downsampled_16_side_by_side.png",
                title=title,
            )

    return heatmap_name


def write_summary_html(run_dir: Path, records: list[dict[str, Any]], heatmap_name: str | None) -> None:
    rows = []
    for rec in records:
        ratio = rec["ratio_percent"]
        rows.append(
            "<tr>"
            f"<td>{ratio}%</td>"
            f"<td>{rec['retained_visual_tokens']}</td>"
            f"<td>{rec.get('object_original_token_count', '')}</td>"
            f"<td>{rec.get('background_summary_token_count', '')}</td>"
            f"<td>{rec.get('visual_source_coverage_count', '')}/{rec.get('total_visual_tokens', '')}</td>"
            f"<td>{rec.get('visual_source_missing_count', '')}</td>"
            f"<td>{rec['prefill_ms']:.2f}</td>"
            f"<td>{rec['prefill_reduction_percent']:.2f}%</td>"
            f"<td>{rec['latency_ms']:.2f}</td>"
            f"<td><pre>{rec['response']}</pre></td>"
            f"<td><img src='selected_tokens_{ratio}.png' width='240'></td>"
            "</tr>"
        )
    heatmap = f"<h2>Attention Heatmap</h2><img src='{heatmap_name}' width='420'>" if heatmap_name else ""
    side_by_side = (
        "<h2>100% Attention Side-by-Side</h2>"
        "<img src='attention_100_side_by_side_style.png' width='650'>"
        if (run_dir / "attention_100_side_by_side_style.png").is_file()
        else ""
    )
    vit_last = (
        "<h2>ViT Internal Attention: Last-Layer CLS to Patches</h2>"
        "<img src='vit_last_layer_cls_attention_side_by_side.png' width='650'>"
        if (run_dir / "vit_last_layer_cls_attention_side_by_side.png").is_file()
        else ""
    )
    vit_rollout = (
        "<h2>ViT Internal Attention Rollout: CLS to Patches</h2>"
        "<img src='vit_rollout_cls_attention_side_by_side.png' width='650'>"
        if (run_dir / "vit_rollout_cls_attention_side_by_side.png").is_file()
        else ""
    )
    vit_patch = (
        "<h2>ViT Internal Attention: Patch-to-Patch Incoming Mean</h2>"
        "<img src='vit_patch_incoming_attention_side_by_side.png' width='650'>"
        if (run_dir / "vit_patch_incoming_attention_side_by_side.png").is_file()
        else ""
    )
    vit_patch_rollout = (
        "<h2>ViT Internal Attention Rollout: Patch-to-Patch Source Flow</h2>"
        "<img src='vit_patch_rollout_attention_side_by_side.png' width='650'>"
        if (run_dir / "vit_patch_rollout_attention_side_by_side.png").is_file()
        else ""
    )
    vit_patch_down = (
        "<h2>ViT Patch Incoming Attention Downsampled to 16x16</h2>"
        "<img src='vit_patch_incoming_attention_downsampled_16_side_by_side.png' width='650'>"
        if (run_dir / "vit_patch_incoming_attention_downsampled_16_side_by_side.png").is_file()
        else ""
    )
    vit_patch_rollout_down = (
        "<h2>ViT Patch Rollout Attention Downsampled to 16x16</h2>"
        "<img src='vit_patch_rollout_attention_downsampled_16_side_by_side.png' width='650'>"
        if (run_dir / "vit_patch_rollout_attention_downsampled_16_side_by_side.png").is_file()
        else ""
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>InternVL3 Partial Prefill</title>"
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:8px;vertical-align:top}pre{white-space:pre-wrap}</style>"
        "</head><body><h1>InternVL3 Partial Vision Prefill</h1>"
        f"{heatmap}"
        f"{side_by_side}"
        f"{vit_last}"
        f"{vit_rollout}"
        f"{vit_patch}"
        f"{vit_patch_rollout}"
        f"{vit_patch_down}"
        f"{vit_patch_rollout_down}"
        "<h2>Ratio Results</h2>"
        "<table><tr><th>Ratio</th><th>Visual Tokens</th><th>Object</th><th>Background</th>"
        "<th>Source Coverage</th><th>Missing</th><th>Prefill ms</th>"
        "<th>Prefill Reduction</th><th>Generate ms</th><th>Response</th><th>Mask</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>"
    )
    (run_dir / "summary.html").write_text(html, encoding="utf-8")


def decode_response(tokenizer, output_ids: torch.Tensor, sep: str) -> str:
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    if sep:
        text = text.split(sep)[0]
    return text.strip()


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_prefill_forward(
    model,
    *,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool,
):
    device = input_embeds.device
    synchronize_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
    synchronize_if_cuda(device)
    return outputs, (time.perf_counter() - start) * 1000.0


def run_generation_from_inputs(
    model,
    tokenizer,
    *,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    eos_token_id: int | None,
    sep: str,
) -> tuple[str, float]:
    device = input_embeds.device
    kwargs: dict[str, Any] = {
        "inputs_embeds": input_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if do_sample:
        kwargs["temperature"] = temperature

    synchronize_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.language_model.generate(**kwargs)
    synchronize_if_cuda(device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return decode_response(tokenizer, output_ids, sep), latency_ms


def write_outputs(
    run_dir: Path,
    records: list[dict[str, Any]],
    selected_by_ratio: dict[int, list[int]],
    scores: torch.Tensor,
    selection_details_by_ratio: dict[int, dict[str, Any]] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "selected_tokens.json").write_text(
        json.dumps(
            {
                "scores": [float(x) for x in scores.detach().cpu().tolist()],
                "selected_indices_by_ratio": selected_by_ratio,
                "selection_details_by_ratio": selection_details_by_ratio or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    tsv_lines = [
        "ratio_percent\tretained_visual_tokens\tobject_original_token_count\t"
        "background_summary_token_count\tvisual_source_coverage_count\tvisual_source_missing_count\t"
        "prefill_ms\tprefill_reduction_percent\tgeneration_latency_ms\tresponse"
    ]
    for rec in records:
        (run_dir / f"response_{rec['ratio_percent']}.txt").write_text(rec["response"], encoding="utf-8")
        safe_response = rec["response"].replace("\t", " ").replace("\n", "\\n")
        tsv_lines.append(
            f"{rec['ratio_percent']}\t{rec['retained_visual_tokens']}\t"
            f"{rec['object_original_token_count']}\t{rec['background_summary_token_count']}\t"
            f"{rec['visual_source_coverage_count']}\t{rec['visual_source_missing_count']}\t"
            f"{rec['prefill_ms']:.3f}\t{rec['prefill_reduction_percent']:.3f}\t"
            f"{rec['latency_ms']:.3f}\t{safe_response}"
        )
    (run_dir / "summary.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")


def load_model_and_tokenizer(args):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def run_experiment(args: argparse.Namespace) -> Path:
    _require_transformers_v4()
    ratios = parse_ratios(args.ratios)
    if args.image is None:
        raise SystemExit("--image is required unless --dry-run is set")
    if not args.image.is_file():
        raise SystemExit(f"missing image file: {args.image}")
    if not torch.cuda.is_available():
        print("CUDA not available; CPU runs may be very slow or run out of memory.", file=sys.stderr)

    tokenizer, model = load_model_and_tokenizer(args)
    device = get_device(model)

    pixel_values, num_patches_list, source_image = load_image_pixel_values(
        args.image,
        input_size=args.input_size,
        max_num=args.max_num,
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    with torch.no_grad():
        visual_features = model.extract_feature(pixel_values)
    full_visual_count = int(visual_features.reshape(-1, visual_features.shape[-1]).shape[0])
    query, eos_token_id, sep = build_expanded_query(
        model,
        tokenizer,
        args.question,
        visual_token_count=full_visual_count,
    )
    _, attention_mask, input_embeds, visual_positions, text_positions = prepare_inputs_with_visual_embeds(
        model, tokenizer, query, visual_features
    )

    reference, attention_prefill_ms = run_prefill_forward(
        model,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    _, baseline_prefill_ms = run_prefill_forward(
        model,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        output_attentions=False,
    )
    try:
        grid_side = square_grid_size(model.num_image_token)
    except ValueError as exc:
        grid_side = 0
        print(f"warning: visualization disabled: {exc}", file=sys.stderr)

    importance = compute_importance_scores(
        args,
        model,
        pixel_values,
        reference,
        visual_positions,
        text_positions,
        grid_side=grid_side,
        num_patches_list=num_patches_list,
    )
    scores = importance.scores

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root.resolve() / f"partial_prefill_{args.image.stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    heatmap_name: str | None = None
    if has_single_square_tile(grid_side=grid_side, num_patches_list=num_patches_list):
        heatmap_name = write_attention_artifacts(
            source_image,
            run_dir,
            scores,
            grid_side,
            vit_last_scores=importance.vit_last_scores,
            vit_rollout_scores=importance.vit_rollout_scores,
            vit_patch_scores=importance.vit_patch_scores,
            vit_patch_rollout_scores=importance.vit_patch_rollout_scores,
        )

    records: list[dict[str, Any]] = []
    selected_by_ratio: dict[int, list[int]] = {}
    selection_details_by_ratio: dict[int, dict[str, Any]] = {}
    flat_features = visual_features.reshape(-1, visual_features.shape[-1])
    for ratio in ratios:
        selection = select_visual_features_for_ratio(
            args,
            flat_features,
            scores,
            ratio_percent=ratio,
            grid_side=grid_side,
            num_patches_list=num_patches_list,
        )
        selected_indices = selection.object_indices
        selected_features = selection.features
        background_summaries = selection.background_summaries
        selected_by_ratio[ratio] = [int(x) for x in selected_indices.detach().cpu().tolist()]
        selection_details_by_ratio[ratio] = selection.details
        ratio_query, ratio_eos_token_id, ratio_sep = build_expanded_query(
            model,
            tokenizer,
            args.question,
            visual_token_count=int(selected_features.shape[0]),
        )
        _, ratio_attention_mask, ratio_input_embeds, _, _ = prepare_inputs_with_visual_embeds(
            model, tokenizer, ratio_query, selected_features
        )
        _, prefill_ms = run_prefill_forward(
            model,
            input_embeds=ratio_input_embeds,
            attention_mask=ratio_attention_mask,
            output_attentions=False,
        )
        response, latency_ms = run_generation_from_inputs(
            model,
            tokenizer,
            input_embeds=ratio_input_embeds,
            attention_mask=ratio_attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            eos_token_id=ratio_eos_token_id if ratio_eos_token_id is not None else eos_token_id,
            sep=ratio_sep or sep,
        )
        if has_single_square_tile(grid_side=grid_side, num_patches_list=num_patches_list):
            write_selection_visualization(
                source_image,
                selected_indices,
                background_summaries,
                selection_policy=args.selection_policy,
                token_count=model.num_image_token,
                grid_side=grid_side,
                background_pool_grid=args.background_pool_grid,
                path=run_dir / f"selected_tokens_{ratio}.png",
            )
        records.append(
            {
                "ratio_percent": ratio,
                "retained_visual_tokens": int(selected_features.shape[0]),
                "object_original_token_count": int(selected_indices.numel()),
                "background_summary_token_count": len(background_summaries),
                "visual_source_coverage_count": selection.details["visual_source_coverage_count"],
                "visual_source_overlap_count": selection.details["visual_source_overlap_count"],
                "visual_source_missing_count": selection.details["visual_source_missing_count"],
                "total_visual_tokens": full_visual_count,
                "attention_prefill_ms": attention_prefill_ms,
                "baseline_prefill_ms": baseline_prefill_ms,
                "prefill_ms": prefill_ms,
                "prefill_reduction_percent": 0.0,
                "generation_latency_ms": latency_ms,
                "latency_ms": latency_ms,
                "response": response,
            }
        )

    comparison_prefill_baseline_ms = next(
        (rec["prefill_ms"] for rec in records if rec["ratio_percent"] == 100),
        baseline_prefill_ms,
    )
    for rec in records:
        rec["comparison_prefill_baseline_ms"] = comparison_prefill_baseline_ms
        rec["prefill_reduction_percent"] = percent_reduction(
            rec["prefill_ms"], comparison_prefill_baseline_ms
        )
        print(
            f"[{rec['ratio_percent']:>3}%] retained={rec['retained_visual_tokens']}/{full_visual_count} "
            f"prefill_ms={rec['prefill_ms']:.2f} "
            f"prefill_reduction_vs_100={rec['prefill_reduction_percent']:.2f}% "
            f"generation_ms={rec['latency_ms']:.2f} response={rec['response']}",
            flush=True,
        )

    meta = {
        "model": args.model,
        "image": str(args.image.resolve()),
        "question": args.question,
        "ratios": ratios,
        "input_size": args.input_size,
        "max_num": args.max_num,
        "importance_source": args.importance_source,
        "num_patches_list": num_patches_list,
        "num_image_token": int(model.num_image_token),
        "full_visual_count": full_visual_count,
        "selection_policy": args.selection_policy,
        "background_ratio_percent": args.background_ratio_percent,
        "background_fixed_clusters": args.background_fixed_clusters,
        "background_count_policy": args.background_count_policy,
        "background_target_tokens_per_cluster": args.background_target_tokens_per_cluster,
        "background_min_clusters": args.background_min_clusters,
        "background_max_clusters": args.background_max_clusters,
        "background_max_fraction": args.background_max_fraction,
        "background_pool_grid": args.background_pool_grid,
        "background_k_min": args.background_k_min,
        "background_k_max": args.background_k_max,
        "background_k_elbow_threshold": args.background_k_elbow_threshold,
        "background_min_region_size": args.background_min_region_size,
        "background_spatial_weight": args.background_spatial_weight,
        "background_score_weight": args.background_score_weight,
        "object_budget_percent": args.object_budget_percent,
        "background_budget_percent": 100 - args.object_budget_percent,
        "attention_prefill_ms": attention_prefill_ms,
        "baseline_prefill_ms": baseline_prefill_ms,
        "comparison_prefill_baseline_ms": comparison_prefill_baseline_ms,
        "score_basis": importance.score_basis,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_outputs(run_dir, records, selected_by_ratio, scores, selection_details_by_ratio)
    write_summary_html(run_dir, records, heatmap_name)
    print(f"results dir: {run_dir}", flush=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL3 partial vision prefill benchmark")
    parser.add_argument("--image", type=Path, default=None, help="Input image path")
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--question", type=str, default="Describe this image briefly.")
    parser.add_argument("--ratios", type=str, default="10,30,50,70,80,100")
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-num", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--results-root", type=Path, default=RESULTS_PARENT)
    parser.add_argument(
        "--importance-source",
        choices=IMPORTANCE_SOURCES,
        default=DEFAULT_IMPORTANCE_SOURCE,
        help=(
            "Source used to rank visual tokens for partial prefill. "
            "Default/recommended: vit-patch-rollout."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default=SELECTION_POLICY_TOPK,
        help=(
            "How to turn importance scores into visual embeddings. "
            "topk keeps original tokens only; island-context-pool keeps high-attention island tokens "
            "and replaces context-field tokens with coarse pooled background summaries; "
            "island-background-merge-pool clusters only non-island background tokens into merged summary regions; "
            "object-topk-residual-cluster-pool clusters every non-object token so the visualization has no gaps; "
            "object-budget-background-kmeans treats each ratio as the total visual-token budget and splits it "
            "between object top-k and k-means background summaries."
        ),
    )
    parser.add_argument(
        "--background-ratio-percent",
        type=int,
        default=5,
        help="Fixed percent of the full visual-token budget reserved for background summaries.",
    )
    parser.add_argument(
        "--background-fixed-clusters",
        type=int,
        default=None,
        help=(
            "For --background-count-policy fixed, use this exact number of background summary tokens. "
            "Overrides --background-ratio-percent."
        ),
    )
    parser.add_argument(
        "--background-count-policy",
        choices=("fixed", "cc"),
        default="fixed",
        help=(
            "How to choose the number of background summary tokens. "
            "fixed uses --background-ratio-percent; cc uses residual connected components and component sizes."
        ),
    )
    parser.add_argument(
        "--background-target-tokens-per-cluster",
        type=int,
        default=12,
        help="For --background-count-policy cc, target residual source tokens represented by one summary token.",
    )
    parser.add_argument(
        "--background-min-clusters",
        type=int,
        default=4,
        help="For --background-count-policy cc, minimum background summary tokens when residual tokens exist.",
    )
    parser.add_argument(
        "--background-max-clusters",
        type=int,
        default=16,
        help="For --background-count-policy cc, hard cap on background summary tokens.",
    )
    parser.add_argument(
        "--background-max-fraction",
        type=float,
        default=0.25,
        help="For --background-count-policy cc, max fraction of the total retained budget used for summaries.",
    )
    parser.add_argument(
        "--background-pool-grid",
        type=int,
        default=4,
        help="Coarse grid side used to pool context/background tokens for island-context-pool.",
    )
    parser.add_argument(
        "--background-k-min",
        type=int,
        default=4,
        help="For object-all-background-kmeans, minimum k considered by elbow auto-k.",
    )
    parser.add_argument(
        "--background-k-max",
        type=int,
        default=24,
        help="For object-all-background-kmeans, maximum k considered by elbow auto-k.",
    )
    parser.add_argument(
        "--background-k-elbow-threshold",
        type=float,
        default=0.08,
        help="For object-all-background-kmeans, stop when inertia improvement falls below this value.",
    )
    parser.add_argument(
        "--background-min-region-size",
        type=int,
        default=2,
        help=(
            "For object-all-background-kmeans and object-budget-background-kmeans, merge connected "
            "background regions smaller than this size."
        ),
    )
    parser.add_argument(
        "--background-spatial-weight",
        type=float,
        default=1.0,
        help=(
            "For object-all-background-kmeans and object-budget-background-kmeans, spatial coordinate "
            "weight used in k-means features."
        ),
    )
    parser.add_argument(
        "--background-score-weight",
        type=float,
        default=0.35,
        help=(
            "For object-all-background-kmeans and object-budget-background-kmeans, attention score "
            "weight used in k-means features."
        ),
    )
    parser.add_argument(
        "--object-budget-percent",
        type=int,
        default=90,
        help=(
            "For object-budget-background-kmeans, percent of each total ratio budget reserved for "
            "original object tokens. The rest is reserved for pooled background summaries."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned ratios without loading the model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios)
    if args.dry_run:
        print("InternVL3 partial vision prefill dry run")
        print(f"model: {args.model}")
        print(f"image: {args.image if args.image is not None else '(not required for dry-run)'}")
        print(f"question: {args.question}")
        print(f"importance source: {args.importance_source}")
        print(f"selection policy: {args.selection_policy}")
        print(f"ratios: {', '.join(str(r) + '%' for r in ratios)}")
        print(f"object budget percent: {args.object_budget_percent}")
        if args.background_fixed_clusters is not None:
            print(f"background fixed clusters: {args.background_fixed_clusters}")
        print("No model was loaded.")
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
