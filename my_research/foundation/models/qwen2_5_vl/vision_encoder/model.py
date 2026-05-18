"""Qwen2.5-VL vision tower export helpers.

Qwen2.5-VL accepts dynamically resized images in the full HuggingFace pipeline,
but QNN exports are compiled for a concrete input shape. This module fixes one
export resolution, embeds the matching `image_grid_thw`, and keeps the public
PTE input compatible with the existing hybrid bridge's single CHW image tensor.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class ImageExportShape:
    image_height: int
    image_width: int
    grid_thw: tuple[int, int, int]
    num_tokens: int


def _closest_token_factor_pair(num_tokens: int, *, divisor: int = 1) -> tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError("--image-tokens must be a positive integer.")
    if divisor <= 0:
        raise ValueError("divisor must be positive.")
    root = math.isqrt(num_tokens)
    for h_tokens in range(root, 0, -1):
        if num_tokens % h_tokens != 0:
            continue
        w_tokens = num_tokens // h_tokens
        if h_tokens % divisor == 0 and w_tokens % divisor == 0:
            return h_tokens, w_tokens
    raise ValueError(
        f"Unable to factor image token count {num_tokens} into grid dimensions "
        f"divisible by spatial_merge_size {divisor}."
    )


def resolve_image_export_shape(
    *,
    image_size: Optional[tuple[int, int]],
    image_tokens: Optional[int],
    patch_size: int,
    spatial_merge_size: int,
) -> ImageExportShape:
    """Resolve a fixed Qwen2.5-VL image export shape and token count.

    `image_size` is `(height, width)` after preprocessing. `image_tokens`
    counts pure pre-merger patch tokens, so it equals `grid_t * grid_h *
    grid_w`. If `image_size` is omitted, the closest-to-square patch grid is
    inferred from `image_tokens`.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive.")

    if image_size is None:
        if image_tokens is None:
            image_height = 448
            image_width = 448
        else:
            grid_h, grid_w = _closest_token_factor_pair(image_tokens, divisor=spatial_merge_size)
            image_height = grid_h * patch_size
            image_width = grid_w * patch_size
    else:
        image_height, image_width = image_size

    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_size dimensions must be positive.")
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError(
            f"Qwen2.5-VL export image size must be divisible by patch_size "
            f"({patch_size}); got {image_height}x{image_width}."
        )

    grid_h = image_height // patch_size
    grid_w = image_width // patch_size
    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
        raise ValueError(
            f"Qwen2.5-VL grid {grid_h}x{grid_w} dimensions must be divisible by "
            f"spatial_merge_size {spatial_merge_size}."
        )
    num_tokens = grid_h * grid_w
    if image_tokens is not None and num_tokens != image_tokens:
        raise ValueError(
            f"Expected {image_tokens} image tokens, but {image_height}x{image_width} produces {num_tokens}."
        )
    return ImageExportShape(
        image_height=image_height,
        image_width=image_width,
        grid_thw=(1, grid_h, grid_w),
        num_tokens=num_tokens,
    )


def _load_state_dict(path: Union[str, Path]) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        state_dict = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(("visual.", "model.visual.", "vision_tower.")):
                    state_dict[key] = handle.get_tensor(key)
        if state_dict:
            return state_dict
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


def visual_weight_files_from_index(index: dict) -> list[str]:
    weight_map = index.get("weight_map", {})
    return sorted({filename for key, filename in weight_map.items() if key.startswith("visual.")})


def _extract_visual_state_dict(full_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    visual_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith("visual."):
            visual_state_dict[key[len("visual.") :]] = value
        elif key.startswith("model.visual."):
            visual_state_dict[key[len("model.visual.") :]] = value
        elif key.startswith("vision_tower."):
            visual_state_dict[key[len("vision_tower.") :]] = value
    return visual_state_dict or dict(full_state_dict)


def _load_visual_state_dict_from_pretrained(model_path: str) -> dict[str, torch.Tensor]:
    """Load only Qwen2.5-VL `visual.*` tensors from a local or HF checkpoint."""
    from safetensors import safe_open

    path = Path(model_path)
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            filenames = visual_weight_files_from_index(index)
            if not filenames:
                raise ValueError(f"No visual.* weights listed in {index_path}.")
            shard_paths = [path / filename for filename in filenames]
        else:
            shard_path = path / "model.safetensors"
            if not shard_path.exists():
                raise ValueError(
                    f"Local Qwen2.5-VL checkpoint must contain model.safetensors "
                    f"or model.safetensors.index.json: {path}"
                )
            shard_paths = [shard_path]
    else:
        from huggingface_hub import hf_hub_download

        index_path = Path(hf_hub_download(model_path, "model.safetensors.index.json"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = visual_weight_files_from_index(index)
        if not filenames:
            raise ValueError(f"No visual.* weights listed in {model_path}/model.safetensors.index.json.")
        shard_paths = [Path(hf_hub_download(model_path, filename)) for filename in filenames]

    state_dict = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("visual."):
                    state_dict[key[len("visual.") :]] = handle.get_tensor(key)
    if not state_dict:
        raise ValueError(f"No visual.* tensors found in Qwen2.5-VL checkpoint: {model_path}")
    return state_dict


class Qwen2_5_VLVisionTower(torch.nn.Module):
    """Run Qwen2.5-VL visual embeddings from one fixed-size CHW image tensor."""

    def __init__(
        self,
        visual: torch.nn.Module,
        *,
        image_height: int,
        image_width: int,
        patch_size: int,
        temporal_patch_size: int,
        spatial_merge_size: int,
        out_hidden_size: int,
        expected_tokens: int,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.spatial_merge_size = int(spatial_merge_size)
        self.out_hidden_size = int(out_hidden_size)
        self.expected_tokens = int(expected_tokens)

        shape = resolve_image_export_shape(
            image_size=(self.image_height, self.image_width),
            image_tokens=self.expected_tokens,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
        )
        self.grid_t, self.grid_h, self.grid_w = shape.grid_thw
        self.register_buffer(
            "image_grid_thw",
            torch.tensor([shape.grid_thw], dtype=torch.long),
            persistent=False,
        )

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        return (
            torch.randn(
                (1, 3, self.image_height, self.image_width),
                dtype=torch.float32,
            ),
        )

    def _patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError("Qwen2.5-VL vision export expects input shape [1, 3, H, W].")
        batch, channels, height, width = pixel_values.shape
        if batch != 1 or channels != 3 or height != self.image_height or width != self.image_width:
            raise ValueError(
                f"Qwen2.5-VL vision export expects [1, 3, {self.image_height}, {self.image_width}], "
                f"got {tuple(pixel_values.shape)}."
            )

        frames = pixel_values.repeat(self.temporal_patch_size, 1, 1, 1)
        patches = frames.reshape(
            self.grid_t,
            self.temporal_patch_size,
            channels,
            self.grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            self.grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        return patches.reshape(
            self.grid_t * self.grid_h * self.grid_w,
            channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError("Qwen2.5-VL vision export expects input shape [1, 3, H, W].")
        batch, channels, height, width = pixel_values.shape
        if batch != 1 or channels != 3 or height != self.image_height or width != self.image_width:
            raise ValueError(
                f"Qwen2.5-VL vision export expects [1, 3, {self.image_height}, {self.image_width}], "
                f"got {tuple(pixel_values.shape)}."
            )
        image_embeds = self.visual(pixel_values, grid_thw=self.image_grid_thw.to(pixel_values.device))
        if image_embeds.shape[0] != self.expected_tokens:
            raise RuntimeError(
                "Qwen2.5-VL vision output token mismatch: "
                f"expected {self.expected_tokens}, got {image_embeds.shape[0]}."
            )
        if image_embeds.shape[-1] != self.out_hidden_size:
            raise RuntimeError(
                "Qwen2.5-VL vision output hidden-size mismatch: "
                f"expected {self.out_hidden_size}, got {image_embeds.shape[-1]}."
            )
        return image_embeds.unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class FixedGridQwen2_5_Visual(torch.nn.Module):
    """Exportable fixed-grid Qwen2.5-VL visual tower before patch merger."""

    def __init__(self, visual: torch.nn.Module, *, grid_thw: tuple[int, int, int]) -> None:
        super().__init__()
        self.blocks = visual.blocks
        self.grid_t, self.grid_h, self.grid_w = (int(v) for v in grid_thw)
        self.seq_len = self.grid_t * self.grid_h * self.grid_w
        self.spatial_merge_size = int(visual.spatial_merge_size)
        self.spatial_merge_unit = int(visual.spatial_merge_unit)
        self.fullatt_block_indexes = set(int(v) for v in visual.fullatt_block_indexes)
        self.hidden_size = int(visual.config.hidden_size)
        self.patch_size = int(visual.config.patch_size)
        self.in_channels = int(visual.config.in_channels)
        self.patch_proj = torch.nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        with torch.no_grad():
            folded_weight = visual.patch_embed.proj.weight.detach().sum(dim=2)
            self.patch_proj.weight.copy_(folded_weight)
        self.patch_proj.weight.requires_grad_(False)

        grid = torch.tensor([grid_thw], dtype=torch.long)
        with torch.no_grad():
            rotary_pos_emb = visual.rot_pos_emb(grid).detach()
            window_index, cu_window_seqlens = visual.get_window_index(grid)
            cu_window_seqlens_tensor = torch.unique_consecutive(
                torch.tensor(cu_window_seqlens, dtype=torch.int32)
            )
            window_lengths = (
                cu_window_seqlens_tensor[1:] - cu_window_seqlens_tensor[:-1]
            ).tolist()
            rotary_pos_emb = rotary_pos_emb.reshape(
                self.seq_len // self.spatial_merge_unit,
                self.spatial_merge_unit,
                -1,
            )
            rotary_pos_emb = rotary_pos_emb[window_index.to(torch.long), :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(self.seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        self.full_lengths = (self.seq_len,)
        self.window_lengths = tuple(int(v) for v in window_lengths if int(v) > 0)
        self.register_buffer("window_index", window_index.to(torch.long), persistent=False)
        self.register_buffer(
            "reverse_indices",
            torch.argsort(window_index.to(torch.long)),
            persistent=False,
        )
        self.register_buffer("position_cos", emb.cos(), persistent=False)
        self.register_buffer("position_sin", emb.sin(), persistent=False)

    def _patch_embed_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_proj(pixel_values.to(dtype=self.patch_proj.weight.dtype))
        hidden_states = hidden_states.squeeze(0).permute(1, 2, 0).contiguous()
        hidden_states = hidden_states.reshape(
            self.grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.hidden_size,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        return hidden_states.reshape(self.seq_len, self.hidden_size)

    def _fixed_attention(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        lengths: tuple[int, ...],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            attn.qkv(hidden_states)
            .reshape(seq_length, 3, attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            cos,
            sin,
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        outputs = []
        start = 0
        for length in lengths:
            end = start + length
            q = query_states[:, :, start:end, :]
            k = key_states[:, :, start:end, :]
            v = value_states[:, :, start:end, :]
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            outputs.append(torch.matmul(attn_weights, v).transpose(1, 2).contiguous())
            start = end

        attn_output = torch.cat(outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return attn.proj(attn_output)

    def _run_block(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
        lengths: tuple[int, ...],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self._fixed_attention(
            block.attn,
            block.norm1(hidden_states),
            lengths,
            position_embeddings,
        )
        return hidden_states + block.mlp(block.norm2(hidden_states))

    def forward(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        del grid_thw
        hidden_states = self._patch_embed_pixels(pixel_values)
        hidden_states = hidden_states.reshape(
            self.seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            self.hidden_size,
        )
        hidden_states = hidden_states[self.window_index, :, :]
        hidden_states = hidden_states.reshape(self.seq_len, self.hidden_size)
        position_embeddings = (self.position_cos, self.position_sin)

        for layer_num, block in enumerate(self.blocks):
            lengths = self.full_lengths if layer_num in self.fullatt_block_indexes else self.window_lengths
            hidden_states = self._run_block(block, hidden_states, lengths, position_embeddings)

        hidden_states = hidden_states.reshape(
            self.seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            self.hidden_size,
        )
        hidden_states = hidden_states[self.reverse_indices, :, :]
        return hidden_states.reshape(self.seq_len, self.hidden_size)


def make_exportable_fixed_grid_visual(
    visual: torch.nn.Module,
    *,
    grid_thw: tuple[int, int, int],
) -> FixedGridQwen2_5_Visual:
    return FixedGridQwen2_5_Visual(visual, grid_thw=grid_thw).eval()


def load_vision_tower(
    model_path: Union[str, Path],
    *,
    image_size: Optional[tuple[int, int]] = None,
    image_tokens: Optional[int] = None,
    encoder_weights: Optional[Union[str, Path]] = None,
    trust_remote_code: bool = True,
    attn_implementation: str = "eager",
) -> Qwen2_5_VLVisionTower:
    """Load Qwen2.5-VL visual tower for one fixed export shape."""
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    model_path = str(model_path)
    load_kwargs = {"trust_remote_code": trust_remote_code}
    config = AutoConfig.from_pretrained(model_path, **load_kwargs)
    vision_config = config.vision_config
    vision_config._attn_implementation = attn_implementation

    shape = resolve_image_export_shape(
        image_size=image_size,
        image_tokens=image_tokens,
        patch_size=vision_config.patch_size,
        spatial_merge_size=vision_config.spatial_merge_size,
    )

    visual = Qwen2_5_VisionTransformerPretrainedModel(vision_config).eval()
    if encoder_weights:
        visual_sd = _extract_visual_state_dict(_load_state_dict(encoder_weights))
    else:
        visual_sd = _load_visual_state_dict_from_pretrained(model_path)

    missing, unexpected = visual.load_state_dict(visual_sd, strict=False)
    if len(missing) == len(visual.state_dict()):
        source = encoder_weights or model_path
        raise ValueError(f"No Qwen2.5-VL visual weights found in: {source}.")
    if unexpected:
        print(f"[qwen2.5-vl] Ignoring unexpected visual weight keys: {len(unexpected)}")

    return Qwen2_5_VLVisionTower(
        make_exportable_fixed_grid_visual(visual, grid_thw=shape.grid_thw),
        image_height=shape.image_height,
        image_width=shape.image_width,
        patch_size=vision_config.patch_size,
        temporal_patch_size=vision_config.temporal_patch_size,
        spatial_merge_size=vision_config.spatial_merge_size,
        out_hidden_size=vision_config.hidden_size,
        expected_tokens=shape.num_tokens,
    ).eval()


@torch.no_grad()
def describe_vision_tower_output(model: Qwen2_5_VLVisionTower) -> dict[str, int | list[int]]:
    example_inputs = model.get_example_inputs()
    output = model(*example_inputs)
    shape = [int(v) for v in output.shape]
    expected_tokens = int(model.expected_tokens)
    actual_tokens = shape[1] if len(shape) >= 2 else 0
    if actual_tokens != expected_tokens:
        raise RuntimeError(
            f"Qwen2.5-VL token count check failed: expected {expected_tokens}, got {actual_tokens}."
        )
    return {
        "output_shape": shape,
        "num_tokens": actual_tokens,
        "expected_tokens": expected_tokens,
        "vision_hidden_size": shape[2] if len(shape) >= 3 else 0,
        "image_grid_thw": [int(v) for v in model.image_grid_thw[0].tolist()],
        "image_height": int(model.image_height),
        "image_width": int(model.image_width),
    }
