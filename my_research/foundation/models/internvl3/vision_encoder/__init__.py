# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .model import extract_vision_encoder_weights, load_vision_encoder

__all__ = [
    "load_vision_encoder",
    "extract_vision_encoder_weights",
]
