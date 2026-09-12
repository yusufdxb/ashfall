"""Independently labelled episodes for detector evaluation.

The one rule this module exists to enforce: the ground truth must not come
from the detector being evaluated. An episode whose label source is
``"detector"``, or whose evidence says it was labelled by a threshold, is
refused at construction. Labels come from simulator ground truth (termination
terms, intervention identity, physical criteria), from human review, or, for
regression fixtures only, from the generating recipe.

A dataset carries a :class:`ashfall.datasets.DatasetManifest`. When that
manifest is a fixture, every report computed from the dataset is a
``fixture_regression`` and may not be quoted as validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ashfall.datasets import DatasetManifest
from ashfall.ontology import PHENOTYPE_NAMES, OnsetWindow, PhenotypeObservation
from ashfall.provenance import content_hash
from ashfall.taxonomy.phenotype_detector import PLATFORMS

CATEGORIES: tuple[str, ...] = (
    "positive",
    "negative",
    "near_miss",
    "confusable",
    "recovery",
    "multi_failure",
)

#: Label sources an evaluation set may use. ``detector`` is deliberately absent.
GROUND_TRUTH_SOURCES: tuple[str, ...] = (
    "simulator_ground_truth",
    "human_review",
    "synthetic_fixture",
)

#: Evidence markers that reveal a label was produced by the detector under test.
_THRESHOLD_MARKERS = ("detector_event_frame", "detector_id", "threshold_config_version")
_THRESHOLD_VALUES = ("detector", "threshold", "detector_threshold", "failure_detector")


class LabelProvenanceError(ValueError):
    """Raised when ground truth traces back to the detector being evaluated."""


@dataclass(frozen=True, eq=False)
class Telemetry:
    """Per-step signals of one episode. Arrays are validated, finite and equal in length."""

    dt_s: float
    pitch_rad: np.ndarray
    roll_rad: np.ndarray
    base_height_m: np.ndarray
    cmd_lin_vel: np.ndarray
    actual_lin_vel: np.ndarray
    joint_vel: np.ndarray | None = None
    contact_forces: np.ndarray | None = None

    def __post_init__(self):
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be positive and finite")
        for name in ("pitch_rad", "roll_rad", "base_height_m"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float).ravel())
        for name, width in (("cmd_lin_vel", 2), ("actual_lin_vel", 2)):
            arr = np.atleast_2d(np.asarray(getattr(self, name), dtype=float))
            if arr.shape[1] < width:
                raise ValueError(f"{name} needs at least {width} columns")
            object.__setattr__(self, name, arr[:, :width])
        for name, width in (("joint_vel", 12), ("contact_forces", 4)):
            value = getattr(self, name)
            if value is not None:
                arr = np.atleast_2d(np.asarray(value, dtype=float))
                if arr.shape[1] != width:
                    raise ValueError(f"{name} must have {width} columns")
                object.__setattr__(self, name, arr)
        n = len(self.pitch_rad)
        if n == 0:
            raise ValueError("telemetry is empty")
        for name in self.channels:
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"{name} length {arr.shape[0]} differs from {n}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} has non-finite values")

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "pitch_rad",
                "roll_rad",
                "base_height_m",
                "cmd_lin_vel",
                "actual_lin_vel",
                "joint_vel",
                "contact_forces",
            )
            if getattr(self, name) is not None
        )

    @property
    def n_steps(self) -> int:
        return int(len(self.pitch_rad))

    @property
    def duration_s(self) -> float:
        return self.n_steps * self.dt_s

    def as_mapping(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.channels}

    def to_dict(self) -> dict:
        data: dict[str, Any] = {"dt_s": self.dt_s}
        for name in self.channels:
            data[name] = np.asarray(getattr(self, name)).tolist()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Telemetry":
        return cls(
            dt_s=float(data["dt_s"]),
            pitch_rad=np.asarray(data["pitch_rad"]),
            roll_rad=np.asarray(data["roll_rad"]),
            base_height_m=np.asarray(data["base_height_m"]),
            cmd_lin_vel=np.asarray(data["cmd_lin_vel"]),
            actual_lin_vel=np.asarray(data["actual_lin_vel"]),
            joint_vel=None if data.get("joint_vel") is None else np.asarray(data["joint_vel"]),
            contact_forces=(
                None if data.get("contact_forces") is None else np.asarray(data["contact_forces"])
            ),
        )


def _refuse_threshold_provenance(evidence: Mapping[str, Any], where: str) -> None:
    for key in _THRESHOLD_MARKERS:
        if key in evidence:
            raise LabelProvenanceError(
                f"{where}: evidence carries {key!r}; ground truth produced by the detector under "
                "test cannot evaluate it"
            )
    for key in ("labelled_by", "label_source", "method", "source"):
        value = evidence.get(key)
        if isinstance(value, str) and value.lower() in _THRESHOLD_VALUES:
            raise LabelProvenanceError(f"{where}: evidence says the label came from {value!r}")


@dataclass(frozen=True, eq=False)
class LabeledEpisode:
    """One episode with independently produced phenotype windows."""

    episode_id: str
    telemetry: Telemetry
    truth: tuple[PhenotypeObservation, ...]
    category: str
    label_source: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode_id:
            raise ValueError("episode_id is required")
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}; expected {CATEGORIES}")
        if self.label_source == "detector":
            raise LabelProvenanceError(
                "an evaluation episode cannot be labelled by the detector it evaluates"
            )
        if self.label_source not in GROUND_TRUTH_SOURCES:
            raise ValueError(f"unknown label source {self.label_source!r}")
        if not isinstance(self.telemetry, Telemetry):
            raise ValueError("telemetry must be a Telemetry record")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be a mapping")
        _refuse_threshold_provenance(self.evidence, self.episode_id)
        object.__setattr__(self, "truth", tuple(self.truth))
        n = self.telemetry.n_steps
        for observation in self.truth:
            if not isinstance(observation, PhenotypeObservation):
                raise ValueError("truth must hold PhenotypeObservation records")
            if observation.window.label_source != self.label_source:
                raise LabelProvenanceError(
                    f"{self.episode_id}: window label source {observation.window.label_source!r} "
                    f"differs from episode label source {self.label_source!r}"
                )
            _refuse_threshold_provenance(
                observation.evidence, f"{self.episode_id}/{observation.phenotype}"
            )
            last = observation.window.end
            if observation.window.established_start >= n or (last is not None and last >= n):
                raise ValueError(f"{self.episode_id}: truth window outside the episode")
        self._check_category()

    def _check_category(self) -> None:
        phenotypes = [o.phenotype for o in self.truth]
        if self.category in ("negative", "near_miss") and self.truth:
            raise ValueError(f"{self.episode_id}: a {self.category} episode carries no phenotype")
        if self.category == "positive" and not self.truth:
            raise ValueError(f"{self.episode_id}: a positive episode needs a phenotype window")
        if self.category == "multi_failure" and len(set(phenotypes)) < 2:
            raise ValueError(f"{self.episode_id}: multi_failure needs two distinct phenotypes")
        if self.category == "recovery":
            by_name: dict[str, list[OnsetWindow]] = {}
            for o in self.truth:
                by_name.setdefault(o.phenotype, []).append(o.window)
            recurrent = [
                ws
                for ws in by_name.values()
                if len(ws) >= 2 and all(w.end is not None for w in ws[:-1])
            ]
            if not recurrent:
                raise ValueError(
                    f"{self.episode_id}: recovery needs the same phenotype twice with the first "
                    "window closed (a failure that occurs, resolves, and occurs again)"
                )
            for ws in recurrent:
                ordered = sorted(ws, key=lambda w: w.transition_start)
                for earlier, later in zip(ordered, ordered[1:]):
                    if earlier.end is None or earlier.end >= later.transition_start:
                        raise ValueError(f"{self.episode_id}: recovery windows overlap")

    @property
    def truth_phenotypes(self) -> tuple[str, ...]:
        return tuple(sorted({o.phenotype for o in self.truth}))

    @property
    def primary_truth(self) -> str:
        if not self.truth:
            return "none"
        return min(self.truth, key=lambda o: (o.window.transition_start, o.phenotype)).phenotype

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "category": self.category,
            "label_source": self.label_source,
            "evidence": dict(self.evidence),
            "truth": [o.to_dict() for o in self.truth],
            "telemetry": self.telemetry.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabeledEpisode":
        return cls(
            episode_id=data["episode_id"],
            telemetry=Telemetry.from_dict(data["telemetry"]),
            truth=tuple(PhenotypeObservation.from_dict(o) for o in data["truth"]),
            category=data["category"],
            label_source=data["label_source"],
            evidence=data.get("evidence", {}),
        )


@dataclass(frozen=True, eq=False)
class DetectorEvalDataset:
    """Episodes plus the manifest that says what kind of evidence they are."""

    episodes: tuple[LabeledEpisode, ...]
    manifest: DatasetManifest
    platform: str
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.schema_version != "1.0":
            raise ValueError("unsupported detector evaluation dataset schema")
        object.__setattr__(self, "episodes", tuple(self.episodes))
        if not self.episodes:
            raise ValueError("an evaluation dataset needs at least one episode")
        if self.platform not in PLATFORMS:
            raise ValueError(f"unknown platform {self.platform!r}")
        ids = [e.episode_id for e in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate episode ids")
        sources = {e.label_source for e in self.episodes}
        if len(sources) != 1:
            raise ValueError("report fixture, simulation and hardware labels separately")
        source = next(iter(sources))
        if self.manifest.is_scientific and source == "synthetic_fixture":
            raise ValueError("fixture-labelled episodes cannot sit under a scientific manifest")
        if not self.manifest.is_scientific and source != "synthetic_fixture":
            raise ValueError("a fixture manifest can only hold synthetic_fixture labels")
        if self.platform == "synthetic_fixture" and source != "synthetic_fixture":
            raise ValueError("platform synthetic_fixture is for fixture labels only")

    @property
    def label_source(self) -> str:
        return self.episodes[0].label_source

    @property
    def report_kind(self) -> str:
        return "independent_evaluation" if self.manifest.is_scientific else "fixture_regression"

    @property
    def validation_claim_allowed(self) -> bool:
        return self.manifest.is_scientific

    @property
    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {c: 0 for c in CATEGORIES}
        for episode in self.episodes:
            counts[episode.category] += 1
        return counts

    @property
    def dataset_id(self) -> str:
        return "deval_" + content_hash(self.to_dict())

    def header(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "platform": self.platform,
            "manifest": self.manifest.to_dict(),
        }

    def to_dict(self) -> dict:
        return {**self.header(), "episodes": [e.to_dict() for e in self.episodes]}

    def episodes_jsonl(self) -> str:
        """The episode lines alone; what a manifest's file hash refers to."""
        return "".join(
            json.dumps(e.to_dict(), sort_keys=True, allow_nan=False) + "\n" for e in self.episodes
        )

    def episodes_sha256(self) -> str:
        return hashlib.sha256(self.episodes_jsonl().encode()).hexdigest()

    def to_jsonl(self) -> str:
        return (
            json.dumps(self.header(), sort_keys=True, allow_nan=False)
            + "\n"
            + self.episodes_jsonl()
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload = self.to_jsonl()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite evaluation data: {path}")
        path.write_text(payload)
        return path

    @classmethod
    def from_jsonl(cls, text: str) -> "DetectorEvalDataset":
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("empty evaluation dataset")
        header = json.loads(lines[0])
        return cls(
            tuple(LabeledEpisode.from_dict(json.loads(line)) for line in lines[1:]),
            DatasetManifest.from_dict(header["manifest"]),
            header["platform"],
            header.get("schema_version", "1.0"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DetectorEvalDataset":
        return cls.from_jsonl(Path(path).read_text())

    def file_sha256(self) -> str:
        return hashlib.sha256(self.to_jsonl().encode()).hexdigest()


def phenotypes_in(episodes: Iterable[LabeledEpisode]) -> tuple[str, ...]:
    seen = {o.phenotype for e in episodes for o in e.truth}
    return tuple(name for name in PHENOTYPE_NAMES if name in seen)


__all__ = [
    "CATEGORIES",
    "GROUND_TRUTH_SOURCES",
    "DetectorEvalDataset",
    "LabelProvenanceError",
    "LabeledEpisode",
    "Telemetry",
    "phenotypes_in",
]
