# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
InternVL3 Vision Encoder (InternViT + MLP projector) for XNNPACK export.

Full InternVL3 체크포인트에서 비전 인코더만 로드합니다.
"""

from pathlib import Path
from typing import Optional, Union
import importlib.util
import os

import torch

from transformers import AutoConfig, AutoModel


def _executorch_root() -> Path:
    project_root = Path(__file__).resolve().parents[5]
    project_executorch = project_root / "executorch"
    if (
        project_executorch
        / "examples/qualcomm/oss_scripts/llama/model/vision_encoder.py"
    ).exists():
        return project_executorch

    env_root = os.environ.get("EXECUTORCH_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if (root / "examples/qualcomm/oss_scripts/llama/model/vision_encoder.py").exists():
            return root

    import executorch

    candidates = []
    package_file = getattr(executorch, "__file__", None)
    if package_file:
        candidates.append(Path(package_file).resolve().parents[1])
    for package_path in getattr(executorch, "__path__", []):
        candidates.append(Path(package_path).resolve().parent)

    for root in candidates:
        if (root / "examples/qualcomm/oss_scripts/llama/model/vision_encoder.py").exists():
            return root

    raise ImportError(
        "Could not locate ExecuTorch root. Set EXECUTORCH_ROOT=/path/to/executorch."
    )


def _load_qualcomm_vision_encoder_class():
    executorch_root = _executorch_root()
    vision_encoder_path = (
        executorch_root
        / "examples/qualcomm/oss_scripts/llama/model/vision_encoder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_foundation_qualcomm_vision_encoder",
        vision_encoder_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {vision_encoder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InternVL3VisionEncoder


class InternVL3Encoder:
    img_resized_h = 448
    img_resized_w = 448

    def create_encoder(self, config):
        encoder_class = _load_qualcomm_vision_encoder_class()
        return encoder_class(
            config,
            img_resized_h=self.img_resized_h,
            img_resized_w=self.img_resized_w,
        )


def _extract_vision_encoder_state_dict(
    full_state_dict: dict,
    encoder_prefixes: tuple = ("vision_tower", "multi_modal_projector"),
    full_model_prefix: str = "model.",
) -> dict:
    """
    Full InternVL state_dict에서 비전 인코더 가중치만 추출.

    HuggingFace InternVLChatModel은 보통 "model.vision_tower.*", "model.multi_modal_projector.*"
    형태의 키를 사용합니다. encoder는 "vision_tower.*", "multi_modal_projector.*"를 기대합니다.
    """
    extracted = {}
    for k, v in full_state_dict.items():
        # "model.vision_tower.xxx" -> "vision_tower.xxx"
        if k.startswith(full_model_prefix):
            subkey = k[len(full_model_prefix) :]
            if any(subkey.startswith(p) for p in encoder_prefixes):
                extracted[subkey] = v
        # 이미 "vision_tower.xxx" 형태면 그대로 사용
        elif any(k.startswith(p) for p in encoder_prefixes):
            extracted[k] = v
    return extracted


def load_vision_encoder(
    model_path: Union[str, Path],
    encoder_weights: Optional[Union[str, Path]] = None,
    trust_remote_code: bool = True,
) -> torch.nn.Module:
    """
    InternVL3 전체 모델에서 비전 인코더만 로드.

    Args:
        model_path: HuggingFace 모델 ID (예: "OpenGVLab/InternVL3-1B-hf") 또는 로컬 경로.
            config 로드에 사용 (encoder_weights 없을 때 가중치도 여기서 로드).
        encoder_weights: (선택) 미리 추출한 비전 인코더 .safetensors 경로.
            지정 시 model_path는 config만 로드하고, 가중치는 여기서 로드.
        trust_remote_code: transformers trust_remote_code 옵션

    Returns:
        InternVL3VisionEncoder (eval 모드)
    """
    model_path = str(model_path)
    load_kwargs = {"trust_remote_code": trust_remote_code}

    if Path(model_path).exists():
        config = AutoConfig.from_pretrained(model_path, **load_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_path, **load_kwargs)

    config_obj = InternVL3Encoder()
    encoder = config_obj.create_encoder(config)
    encoder = encoder.eval()

    if encoder_weights:
        from safetensors.torch import load_file

        enc_path = Path(encoder_weights)
        if enc_path.suffix == ".safetensors":
            encoder_sd = load_file(str(enc_path))
        else:
            encoder_sd = torch.load(
                str(enc_path), map_location="cpu", weights_only=True
            )
        encoder.load_state_dict(encoder_sd, strict=True)
    else:
        if Path(model_path).exists():
            auto_model = AutoModel.from_pretrained(
                model_path, config=config, **load_kwargs
            )
        else:
            auto_model = AutoModel.from_pretrained(model_path, **load_kwargs)
        auto_model = auto_model.eval()
        full_sd = auto_model.state_dict()
        encoder_sd = _extract_vision_encoder_state_dict(full_sd)
        if encoder_sd:
            encoder.load_state_dict(encoder_sd, strict=False)
        else:
            encoder.load_state_dict(full_sd, strict=False)

    return encoder


def extract_vision_encoder_weights(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    trust_remote_code: bool = True,
) -> None:
    """
    Full InternVL3 체크포인트에서 비전 인코더 가중치만 추출하여 저장.

    추출된 .safetensors는 나중에 load_vision_encoder에서 --encoder_weights 경로로
    직접 로드할 때 사용할 수 있습니다 (전체 모델 로드 없이 빠른 로드).

    Args:
        model_path: HuggingFace 모델 ID 또는 로컬 경로
        output_path: 저장 경로 (예: vision_encoder_only.safetensors)
        trust_remote_code: transformers trust_remote_code
    """
    from safetensors.torch import load_file, save_file

    model_path = str(model_path)
    output_path = Path(output_path)
    load_kwargs = {"trust_remote_code": trust_remote_code}

    if Path(model_path).is_dir():
        # safetensors 또는 bin 로드
        model_dir = Path(model_path)
        st_files = sorted(model_dir.glob("*.safetensors"))
        if st_files:
            full_sd = {}
            for f in st_files:
                full_sd.update(load_file(str(f)))
        else:
            ckpt = list(model_dir.glob("pytorch_model*.bin")) or list(
                model_dir.glob("*.bin")
            )
            if ckpt:
                full_sd = torch.load(
                    str(ckpt[0]), map_location="cpu", weights_only=True
                )
            else:
                auto_model = AutoModel.from_pretrained(model_path, **load_kwargs)
                full_sd = auto_model.state_dict()
    else:
        auto_model = AutoModel.from_pretrained(model_path, **load_kwargs)
        full_sd = auto_model.state_dict()

    encoder_sd = _extract_vision_encoder_state_dict(full_sd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_str = str(output_path)
    if not out_str.endswith(".safetensors"):
        out_str = str(output_path) + ".safetensors"
    save_file(encoder_sd, out_str)
    print(f"[extract] Saved {len(encoder_sd)} keys to {out_str}")
