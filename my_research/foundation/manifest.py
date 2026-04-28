from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


FOUNDATION_MANIFEST_FILENAME = "manifest.json"
FOUNDATION_SCHEMA_VERSION = 1


@dataclass
class FoundationManifest:
    schema_version: int
    backend: str
    model_family: str
    variant: str
    runner_type: str
    paths: Dict[str, str]
    export: Dict[str, Any] = field(default_factory=dict)
    quant: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolve_paths(self, base_dir: Path) -> "FoundationManifest":
        resolved = {}
        for key, value in self.paths.items():
            if value is None:
                continue
            path = Path(value)
            resolved[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
        return FoundationManifest(
            schema_version=self.schema_version,
            backend=self.backend,
            model_family=self.model_family,
            variant=self.variant,
            runner_type=self.runner_type,
            paths=resolved,
            export=dict(self.export),
            quant=dict(self.quant),
            runtime=dict(self.runtime),
            metadata=dict(self.metadata),
        )


def _relpath(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_manifest(
    *,
    artifact_root: Path,
    backend: str,
    variant: str,
    model_family: str = "internvl3",
    runner_type: str = "multimodal_split",
    vision_encoder_pte: Optional[Path] = None,
    text_embedding_pte: Optional[Path] = None,
    text_decoder_pte: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    export: Optional[Dict[str, Any]] = None,
    quant: Optional[Dict[str, Any]] = None,
    runtime: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FoundationManifest:
    artifact_root = artifact_root.resolve()
    paths = {
        "artifact_root": str(artifact_root),
        "vision_encoder_pte": _relpath(vision_encoder_pte, artifact_root),
        "text_embedding_pte": _relpath(text_embedding_pte, artifact_root),
        "text_decoder_pte": _relpath(text_decoder_pte, artifact_root),
        "tokenizer_path": _relpath(tokenizer_path, artifact_root),
    }
    return FoundationManifest(
        schema_version=FOUNDATION_SCHEMA_VERSION,
        backend=backend,
        model_family=model_family,
        variant=variant,
        runner_type=runner_type,
        paths=paths,
        export=export or {},
        quant=quant or {},
        runtime=runtime or {},
        metadata=metadata or {},
    )


def write_manifest(manifest: FoundationManifest, output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def load_manifest(path: Path, *, resolve_paths: bool = True) -> FoundationManifest:
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = FoundationManifest(
        schema_version=data["schema_version"],
        backend=data["backend"],
        model_family=data["model_family"],
        variant=data["variant"],
        runner_type=data["runner_type"],
        paths=data["paths"],
        export=data.get("export", {}),
        quant=data.get("quant", {}),
        runtime=data.get("runtime", {}),
        metadata=data.get("metadata", {}),
    )
    if resolve_paths:
        return manifest.resolve_paths(path.parent)
    return manifest
