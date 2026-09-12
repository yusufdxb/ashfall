"""Fixtures versus scientific data, kept apart by manifest rather than by convention.

Two kinds of trajectory data live under ``data/``:

``data/fixtures/``
    Hand-authored or generated trajectories used to exercise schema, wiring
    and regression behaviour. They are test fixtures. They are never counted
    as validation of a detector, an intervention or a learning result, and a
    manifest that declares them cannot be passed where scientific data is
    required.

``data/scientific/``
    Trajectories produced by physics (simulator rollouts) or by the robot,
    each set with a manifest that names its provenance: policy, environment
    configuration, simulator version, seeds, intervention, and the hash of
    every file. A scientific manifest without provenance is refused.

The separation is enforced by :func:`assert_scientific`, which every
scientific consumer (harvest, detector evaluation, H0 calibration) calls.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ashfall.provenance import content_hash, file_hash, write_artifact

DATASET_KINDS = ("fixture", "scientific")
DATASET_SOURCES = ("synthetic_generator", "simulation", "hardware")
MANIFEST_NAME = "dataset.json"

#: Provenance keys a scientific dataset must carry.
SCIENTIFIC_PROVENANCE_KEYS = (
    "policy_id",
    "env_config_hash",
    "simulator_version",
    "seeds",
    "ashfall_sha",
    "phoenix_sha",
)


class NotScientificData(ValueError):
    """Raised when fixture data is offered where physics or hardware data is required."""


@dataclass(frozen=True)
class DatasetManifest:
    kind: str
    source: str
    files: Mapping[str, str]
    description: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    intervention: Mapping[str, Any] | None = None
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.kind not in DATASET_KINDS:
            raise ValueError(f"unknown dataset kind {self.kind!r}")
        if self.source not in DATASET_SOURCES:
            raise ValueError(f"unknown dataset source {self.source!r}")
        if self.kind == "scientific" and self.source == "synthetic_generator":
            raise NotScientificData("a synthetic generator cannot produce scientific data")
        if self.kind == "fixture" and self.source != "synthetic_generator":
            raise ValueError("physics or hardware data is scientific data, not a fixture")
        if not self.files:
            raise ValueError("a dataset manifest needs at least one file")
        for name, digest in self.files.items():
            if len(digest) != 64:
                raise ValueError(f"{name}: file hash must be SHA256")
        if not self.description.strip():
            raise ValueError("description is required")
        if self.kind == "scientific":
            missing = [k for k in SCIENTIFIC_PROVENANCE_KEYS if self.provenance.get(k) is None]
            if missing:
                raise NotScientificData(f"scientific dataset lacks provenance {missing}")

    @property
    def dataset_id(self) -> str:
        return "ds_" + content_hash(asdict(self))

    @property
    def is_scientific(self) -> bool:
        return self.kind == "scientific"

    def to_dict(self) -> dict:
        return {**asdict(self), "dataset_id": self.dataset_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetManifest":
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        result = cls(**values)
        if data.get("dataset_id") not in (None, result.dataset_id):
            raise ValueError("dataset manifest content identity mismatch")
        return result

    def save(self, directory: str | Path) -> Path:
        return write_artifact(Path(directory) / MANIFEST_NAME, self.to_dict())

    @classmethod
    def load(cls, directory: str | Path) -> "DatasetManifest":
        path = Path(directory) / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"{directory} has no {MANIFEST_NAME}; unclassified trajectory data cannot be used"
            )
        return cls.from_dict(json.loads(path.read_text()))

    def verify(self, directory: str | Path) -> None:
        """Every listed file must exist with its recorded hash."""
        for name, digest in self.files.items():
            path = Path(directory) / name
            if not path.is_file():
                raise FileNotFoundError(f"{path} listed in manifest is missing")
            if file_hash(path) != digest:
                raise ValueError(f"{path} does not match its manifest hash")


def hash_files(directory: str | Path, patterns: Iterable[str] = ("*.parquet",)) -> dict[str, str]:
    directory = Path(directory)
    files = sorted({p for pattern in patterns for p in directory.glob(pattern)})
    return {p.name: file_hash(p) for p in files}


def fixture_manifest(directory: str | Path, description: str, **provenance) -> DatasetManifest:
    return DatasetManifest(
        "fixture", "synthetic_generator", hash_files(directory), description, dict(provenance)
    )


def scientific_manifest(
    directory: str | Path,
    *,
    source: str,
    description: str,
    provenance: Mapping[str, Any],
    intervention: Mapping[str, Any] | None = None,
    patterns: Iterable[str] = ("*.parquet", "*.json"),
) -> DatasetManifest:
    files = {k: v for k, v in hash_files(directory, patterns).items() if k != MANIFEST_NAME}
    return DatasetManifest("scientific", source, files, description, dict(provenance), intervention)


def assert_scientific(manifest: DatasetManifest, *, purpose: str) -> DatasetManifest:
    """Refuse fixtures wherever a scientific claim would be made from the data."""
    if not manifest.is_scientific:
        raise NotScientificData(
            f"{purpose} requires simulator or hardware data; this dataset is a "
            f"{manifest.kind} from {manifest.source} and cannot support a scientific claim"
        )
    return manifest


__all__ = [
    "DATASET_KINDS",
    "DATASET_SOURCES",
    "MANIFEST_NAME",
    "SCIENTIFIC_PROVENANCE_KEYS",
    "DatasetManifest",
    "NotScientificData",
    "assert_scientific",
    "fixture_manifest",
    "hash_files",
    "scientific_manifest",
]
