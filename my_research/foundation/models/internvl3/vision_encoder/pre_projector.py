# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""InternVL3 vision tower export helpers that stop before multi_modal_projector.

The existing `vision_encoder.model.load_vision_encoder()` returns Qualcomm's
InternVL3VisionEncoder, which includes both `vision_tower` and
`multi_modal_projector`. This module keeps that path untouched and exposes a
separate wrapper for experiments that want projector-free visual features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from my_research.foundation.models.internvl3.vision_encoder.model import (
    InternVL3Encoder,
    _extract_vision_encoder_state_dict,
    replace_vision_attention_for_vulkan,
)


def _vision_tower_only_state_dict(full_state_dict: dict) -> dict:
    return _extract_vision_encoder_state_dict(
        full_state_dict,
        encoder_prefixes=("vision_tower",),
    )


class InternVL3VisionPreProjector(torch.nn.Module):
    """Run InternVL3 up to pixel-shuffled visual tokens, before projector.

    The forward logic mirrors Qualcomm's
    `InternVL3VisionEncoder.forward()` through the reshape + pixel_shuffle step
    and intentionally omits `multi_modal_projector(...)`.
    """

    def __init__(self, fused_encoder: torch.nn.Module):
        super().__init__()
        self.vision_tower = fused_encoder.vision_tower
        self.config = fused_encoder.config
        self.img_resized_h = fused_encoder.img_resized_h
        self.img_resized_w = fused_encoder.img_resized_w
        self._pixel_shuffle = fused_encoder.pixel_shuffle

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        return (
            torch.randn(
                (1, 3, self.img_resized_h, self.img_resized_w),
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "default",
    ) -> torch.Tensor:
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if vision_feature_layer == -1:
            vision_features = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        else:
            vision_features = self.vision_tower(
                pixel_values=pixel_values,
                output_hidden_states=True,
            ).hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :]

        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]
        vision_features = vision_features.reshape(
            batch_size,
            feature_size,
            feature_size,
            -1,
        )
        vision_features = self._pixel_shuffle(
            vision_features,
            scale_factor=self.config.downsample_ratio,
        )
        return vision_features.reshape(batch_size, -1, vision_features.shape[-1])


def load_vision_pre_projector(
    model_path: Union[str, Path],
    encoder_weights: Optional[Union[str, Path]] = None,
    trust_remote_code: bool = True,
    vulkan_friendly_attention: bool = False,
) -> InternVL3VisionPreProjector:
    """Load InternVL3 vision tower up to pre-projector visual tokens.

    `encoder_weights`, when provided, may be the same extracted full vision
    encoder state used by `load_vision_encoder`; projector weights are simply
    ignored by the wrapper after loading.
    """
    from transformers import AutoConfig, AutoModel

    model_path = str(model_path)
    load_kwargs = {"trust_remote_code": trust_remote_code}
    config = AutoConfig.from_pretrained(model_path, **load_kwargs)

    fused_encoder = InternVL3Encoder().create_encoder(config).eval()
    if encoder_weights:
        enc_path = Path(encoder_weights)
        if enc_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            encoder_sd = _vision_tower_only_state_dict(load_file(str(enc_path)))
        else:
            encoder_sd = _vision_tower_only_state_dict(
                torch.load(str(enc_path), map_location="cpu", weights_only=True)
            )
        if not encoder_sd:
            raise ValueError(
                f"No vision_tower weights found in encoder_weights: {enc_path}. "
                "Pass a full HF checkpoint or a vision-encoder checkpoint, not a decoder-only checkpoint."
            )
        fused_encoder.load_state_dict(encoder_sd, strict=False)
    else:
        auto_model = AutoModel.from_pretrained(model_path, config=config, **load_kwargs)
        auto_model = auto_model.eval()
        encoder_sd = _vision_tower_only_state_dict(auto_model.state_dict())
        if encoder_sd:
            fused_encoder.load_state_dict(encoder_sd, strict=False)
        else:
            fused_encoder.load_state_dict(auto_model.state_dict(), strict=False)

    if vulkan_friendly_attention:
        fused_encoder = replace_vision_attention_for_vulkan(fused_encoder)
    return InternVL3VisionPreProjector(fused_encoder).eval()


@torch.no_grad()
def describe_pre_projector_output(model: torch.nn.Module) -> dict[str, int | list[int]]:
    """Run one example input and summarize the pre-projector output shape."""
    example_inputs = model.get_example_inputs()
    output = model(*example_inputs)
    shape = [int(v) for v in output.shape]
    return {
        "output_shape": shape,
        "num_tokens": shape[1] if len(shape) >= 2 else 0,
        "vision_hidden_size": shape[2] if len(shape) >= 3 else 0,
    }
