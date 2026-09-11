"""Frozen, content-addressed simulator scenarios and leakage checks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .provenance import content_hash, write_artifact

SPLITS = {"train", "validation", "held_out", "nominal"}


def seed_state_id(capsule_id, seed_row):
    return content_hash({"capsule_id": capsule_id, "seed_row": seed_row})


@dataclass(frozen=True)
class Scenario:
    capsule_id: str
    parameters: tuple[tuple[str, float], ...]
    scenario_seed: int
    split: str
    reproduction_id: str
    initial_state_id: str
    family: str = "slip"

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError("Unknown scenario split")
        if not self.capsule_id or not self.initial_state_id or not self.reproduction_id:
            raise ValueError("Scenario requires capsule, state, reproduction identities")
        if isinstance(self.scenario_seed, bool) or not isinstance(self.scenario_seed, int):
            raise ValueError("scenario_seed must be an integer")
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("Duplicate parameter names")
        if not all(math.isfinite(v) for _, v in self.parameters):
            raise ValueError("Non-finite scenario parameter")
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))

    @property
    def parameter_sample_id(self):
        return content_hash(dict(self.parameters))

    @property
    def scenario_id(self):
        # Split is not part of physical identity: relabeling a held-out case cannot evade checks.
        return content_hash({k: v for k, v in asdict(self).items() if k != "split"})

    def to_dict(self):
        return {
            **asdict(self),
            "scenario_id": self.scenario_id,
            "parameter_sample_id": self.parameter_sample_id,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        identity = data.pop("scenario_id")
        parameter_id = data.pop("parameter_sample_id")
        data["parameters"] = tuple(tuple(p) for p in data["parameters"])
        result = cls(**data)
        if result.scenario_id != identity or result.parameter_sample_id != parameter_id:
            raise ValueError("Scenario content identity mismatch")
        return result


@dataclass(frozen=True)
class ScenarioManifest:
    scenarios: tuple[Scenario, ...]
    schema_version: str = "2.0.0"

    def __post_init__(self):
        if self.schema_version != "2.0.0" or not self.scenarios:
            raise ValueError("Unsupported or empty scenario manifest")
        ids = [s.scenario_id for s in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scenario or split leakage")
        # Even different random seeds must not clone identical physical parameter/state cases
        # across train and held-out. Validation and nominal are frozen too.
        groups = {}
        for s in self.scenarios:
            key = (s.capsule_id, s.initial_state_id, s.parameters)
            if key in groups and groups[key] != s.split:
                raise ValueError("Physical scenario duplicated across splits")
            groups[key] = s.split

    def select(self, split):
        return tuple(s for s in self.scenarios if s.split == split)

    def assert_training(self, scenario: Scenario):
        allowed = {s.scenario_id for s in self.select("train")}
        if scenario.split != "train" or scenario.scenario_id not in allowed:
            raise ValueError("Held-out, validation or unknown scenario cannot enter training")

    @property
    def manifest_hash(self):
        return content_hash(self.to_dict())

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "scenarios": [s.to_dict() for s in sorted(self.scenarios, key=lambda x: x.scenario_id)],
        }

    def save(self, path):
        return write_artifact(path, self.to_dict())

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(tuple(Scenario.from_dict(s) for s in data["scenarios"]), data["schema_version"])
