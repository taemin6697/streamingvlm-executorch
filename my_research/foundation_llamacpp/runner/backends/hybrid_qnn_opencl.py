from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HybridQnnOpenClBackend:
    """QNN vision + OpenCL llama.cpp decoder backend marker.

    The current runtime intentionally remains a two-process coordinated flow:
    `hybrid_vision_dump` produces `.svlmemb`, while `hybrid_decode` consumes it.
    """

    encoder_pte: Path

