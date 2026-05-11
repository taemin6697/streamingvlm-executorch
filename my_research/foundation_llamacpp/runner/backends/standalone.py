from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandaloneBackend:
    """CPU/OpenCL standalone llama.cpp backend marker.

    The detailed command construction still lives in the compatibility runner
    during this incremental refactor.
    """

    processor: str
    use_precise_phases: bool

