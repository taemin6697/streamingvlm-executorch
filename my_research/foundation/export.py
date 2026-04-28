# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Foundation export orchestrator. Dispatches to native backend exporters."""

from __future__ import annotations

import argparse
from pathlib import Path


def export_with_backend(args: argparse.Namespace) -> int:
    """Run foundation-native export for the given backend."""
    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.backend in {"xnnpack", "vulkan"}:
        from my_research.foundation.exporters.xnnpack import export_xnnpack

        return export_xnnpack(args)

    if args.backend == "qnn":
        from my_research.foundation.exporters.qnn import export_qnn

        return export_qnn(args)

    raise SystemExit(f"지원하지 않는 backend: {args.backend}")
