from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_tensor(path: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} floats, expected {expected}")
    return values.reshape(shape)


def _stats(values: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(values.shape),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "l2_norm": float(np.linalg.norm(values.reshape(-1))),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "first_16": [float(x) for x in values.reshape(-1)[:16]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare dumped Android vision outputs.")
    parser.add_argument("--reference", required=True, help="Reference float32 .bin path")
    parser.add_argument("--candidate", required=True, help="Candidate float32 .bin path")
    parser.add_argument("--shape", default="1,256,896")
    parser.add_argument(
        "--output",
        default="/workspace/streamingvlm/my_research/foundation/debug/vision_compare/results/android_vision_dump_diff.json",
    )
    args = parser.parse_args()

    shape = tuple(int(part) for part in args.shape.split(","))
    reference = _read_tensor(args.reference, shape)
    candidate = _read_tensor(args.candidate, shape)
    diff = reference - candidate
    ref_flat = reference.reshape(-1).astype(np.float64)
    cand_flat = candidate.reshape(-1).astype(np.float64)
    cosine = float(
        np.dot(ref_flat, cand_flat)
        / (np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat))
    )

    summary = {
        "reference": args.reference,
        "candidate": args.candidate,
        "reference_stats": _stats(reference),
        "candidate_stats": _stats(candidate),
        "diff": {
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "l2_norm": float(np.linalg.norm(diff.reshape(-1))),
            "cosine": cosine,
            "first_16": [float(x) for x in diff.reshape(-1)[:16]],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
