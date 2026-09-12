"""The four research arms of the Phase-II comparison, under one budget.

    A  standard: the baseline continues training with no failure-specific
       machinery. It receives the same fresh-transition and update budget as
       every other arm, so "no repair" is a real arm, not the untouched
       checkpoint.
    B  uniform failure replay: resets are seeded from the capsule pool with
       no conditioning; every capsule is equally likely.
    C  intervention-conditioned replay: resets are stratified by the physical
       cause that produced the capsule (intervention kind and intensity).
    D  phenotype-conditioned repair: resets are stratified by the observed
       failure phenotype.

Arms C and D are what the study is about: whether conditioning on the cause or
on the effect buys anything over B at equal compute. The names are
configurable, the kinds and their constraints are not. A phenotype-conditioned
arm can only name phenotypes retained for the first confirmatory study; an
intervention-conditioned arm can only name intervention kinds; neither may use
a name from the other vocabulary, which is the conflation this redesign
removed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ashfall.ontology import (
    FIRST_STUDY_PHENOTYPES,
    INTERVENTION_KINDS,
    PHENOTYPE_NAMES,
    PHENOTYPES,
)
from ashfall.protocol.budget import ComputeBudget
from ashfall.provenance import content_hash, write_artifact

ARM_KINDS: tuple[str, ...] = (
    "standard",
    "uniform_replay",
    "intervention_conditioned",
    "phenotype_conditioned",
)

#: Derived, never chosen: what each kind conditions its replay on.
CONDITIONING: Mapping[str, str] = {
    "standard": "none",
    "uniform_replay": "none",
    "intervention_conditioned": "intervention_parameters",
    "phenotype_conditioned": "phenotype",
}

REPLAY_SOURCES: Mapping[str, str | None] = {
    "standard": None,
    "uniform_replay": "capsules",
    "intervention_conditioned": "capsules",
    "phenotype_conditioned": "capsules",
}

#: Seed-row strategies a replay arm may use. ``first`` (row 0) is the Phase-I
#: defect and is not in this list on purpose.
SEED_ROW_STRATEGIES: tuple[str, ...] = (
    "phenotype_window_fraction",
    "transition_start",
    "precursor_start",
)

#: Strata an arm's sampler may balance on.
STRATA: tuple[str, ...] = ("phenotype", "intervention_kind", "intensity_bin", "command_bucket")


@dataclass(frozen=True)
class SamplerConfig:
    """How an arm draws its resets.

    ``replay_fraction`` is the fraction of environment resets seeded from a
    replayed capsule state; ``strata`` is what the draw is balanced over;
    ``seed_row_strategy`` names how the seed row inside the onset window is
    chosen, and ``window_fraction`` where in the transition-to-established
    window it lands. Row 0 is not a strategy.
    """

    replay_fraction: float = 0.0
    strata: tuple[str, ...] = ()
    seed_row_strategy: str = "phenotype_window_fraction"
    window_fraction: float = 0.5

    def __post_init__(self):
        if not isinstance(self.replay_fraction, (int, float)) or isinstance(
            self.replay_fraction, bool
        ):
            raise ValueError("replay_fraction must be a number")
        if not math.isfinite(self.replay_fraction) or not 0 <= self.replay_fraction <= 1:
            raise ValueError("replay_fraction must lie in [0, 1]")
        object.__setattr__(self, "strata", tuple(self.strata))
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("duplicate stratum")
        unknown = [s for s in self.strata if s not in STRATA]
        if unknown:
            raise ValueError(f"unknown strata {unknown}; expected a subset of {STRATA}")
        if self.seed_row_strategy == "first":
            raise ValueError(
                "seed_row_strategy 'first' seeds row 0, which is a nominal state in every "
                "recording this repository has shipped; that is the Phase-I defect"
            )
        if self.seed_row_strategy not in SEED_ROW_STRATEGIES:
            raise ValueError(
                f"unknown seed_row_strategy {self.seed_row_strategy!r}; "
                f"expected {SEED_ROW_STRATEGIES}"
            )
        if not math.isfinite(self.window_fraction) or not 0 <= self.window_fraction < 1:
            raise ValueError("window_fraction must lie in [0, 1)")
        object.__setattr__(self, "replay_fraction", float(self.replay_fraction))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str
    description: str
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    phenotypes: tuple[str, ...] = ()
    intervention_kinds: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("arm needs a name")
        if self.kind not in ARM_KINDS:
            raise ValueError(f"unknown arm kind {self.kind!r}; expected {ARM_KINDS}")
        if not self.description.strip():
            raise ValueError("arm needs a description")
        if not isinstance(self.sampler, SamplerConfig):
            raise ValueError("sampler must be a SamplerConfig")
        object.__setattr__(self, "phenotypes", tuple(self.phenotypes))
        object.__setattr__(self, "intervention_kinds", tuple(self.intervention_kinds))
        for name in ("phenotypes", "intervention_kinds"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate entry in {name}")
        # Conflation checks run on every arm before the per-kind rules, so a
        # phenotype name in intervention_kinds is refused even on an arm that
        # should not carry either list.
        for value in self.intervention_kinds:
            if value in PHENOTYPE_NAMES:
                raise ValueError(
                    f"{value!r} is a phenotype; an intervention-conditioned arm conditions "
                    "on causes"
                )
            if value not in INTERVENTION_KINDS or value == "none":
                raise ValueError(f"unknown or empty intervention kind {value!r}")
        for value in self.phenotypes:
            if value in INTERVENTION_KINDS:
                raise ValueError(
                    f"{value!r} is an intervention kind; a phenotype-conditioned arm conditions "
                    "on observed effects"
                )
            if value not in PHENOTYPE_NAMES:
                raise ValueError(f"unknown phenotype {value!r}")
            if value not in FIRST_STUDY_PHENOTYPES:
                reason = PHENOTYPES[value].exclusion_reason
                raise ValueError(
                    f"phenotype {value!r} is excluded from the first confirmatory study: {reason}"
                )
        if self.kind == "standard":
            if self.sampler.replay_fraction != 0.0:
                raise ValueError("the standard arm replays nothing")
            if self.phenotypes or self.intervention_kinds:
                raise ValueError("the standard arm conditions on nothing")
        else:
            if self.sampler.replay_fraction <= 0.0:
                raise ValueError(f"a {self.kind} arm needs a positive replay_fraction")
        if self.kind == "uniform_replay":
            if self.phenotypes or self.intervention_kinds or self.sampler.strata:
                raise ValueError("uniform replay is unconditioned: no strata, phenotypes or kinds")
        if self.kind == "intervention_conditioned":
            if not self.intervention_kinds:
                raise ValueError("an intervention-conditioned arm must name its intervention kinds")
            if self.phenotypes:
                raise ValueError("an intervention-conditioned arm does not condition on phenotypes")
            if "intervention_kind" not in self.sampler.strata:
                raise ValueError(
                    "an intervention-conditioned sampler must stratify on intervention_kind"
                )
        if self.kind == "phenotype_conditioned":
            if not self.phenotypes:
                raise ValueError("a phenotype-conditioned arm must name its phenotypes")
            if self.intervention_kinds:
                raise ValueError("a phenotype-conditioned arm does not condition on interventions")
            if "phenotype" not in self.sampler.strata:
                raise ValueError("a phenotype-conditioned sampler must stratify on phenotype")

    @property
    def conditioning(self) -> str:
        return CONDITIONING[self.kind]

    @property
    def replay_source(self) -> str | None:
        return REPLAY_SOURCES[self.kind]

    @property
    def replays(self) -> bool:
        return self.replay_source is not None

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "conditioning": self.conditioning,
            "replay_source": self.replay_source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArmSpec":
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        values["sampler"] = SamplerConfig(**values.get("sampler", {}))
        return cls(**values)


@dataclass(frozen=True)
class ArmSet:
    """The arms of one comparison and the single budget they all receive."""

    arms: tuple[ArmSpec, ...]
    budget: ComputeBudget
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.schema_version != "1.0":
            raise ValueError("unsupported arm set schema")
        object.__setattr__(self, "arms", tuple(self.arms))
        if len(self.arms) < 2:
            raise ValueError("a comparison needs at least two arms")
        if not all(isinstance(a, ArmSpec) for a in self.arms):
            raise ValueError("arms must be ArmSpec records")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        standard = [a for a in self.arms if a.kind == "standard"]
        if len(standard) != 1:
            raise ValueError("exactly one standard arm is required")
        if not isinstance(self.budget, ComputeBudget):
            raise ValueError("budget must be a ComputeBudget shared by every arm")
        if any(a.replays for a in self.arms) and self.budget.replay_transitions == 0:
            raise ValueError("replay arms declared but the shared budget grants zero replay")
        if not any(a.replays for a in self.arms) and self.budget.replay_transitions != 0:
            raise ValueError("no arm replays but the budget grants replay transitions")

    @property
    def standard_arm(self) -> ArmSpec:
        return next(a for a in self.arms if a.kind == "standard")

    @property
    def replay_arm_names(self) -> frozenset[str]:
        return frozenset(a.name for a in self.arms if a.replays)

    def by_name(self, name: str) -> ArmSpec:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(name)

    @property
    def arm_set_id(self) -> str:
        return "arms_" + content_hash(self.to_dict(include_id=False))

    def to_dict(self, include_id: bool = True) -> dict:
        data = {
            "schema_version": self.schema_version,
            "arms": [a.to_dict() for a in self.arms],
            "budget": self.budget.to_dict(),
        }
        if include_id:
            data["arm_set_id"] = self.arm_set_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArmSet":
        result = cls(
            tuple(ArmSpec.from_dict(a) for a in data["arms"]),
            ComputeBudget.from_dict(data["budget"]),
            data.get("schema_version", "1.0"),
        )
        if data.get("arm_set_id") not in (None, result.arm_set_id):
            raise ValueError("arm set content identity mismatch")
        return result

    def save(self, path: str | Path) -> Path:
        return write_artifact(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ArmSet":
        return cls.from_dict(json.loads(Path(path).read_text()))


def default_arm_set(
    budget: ComputeBudget,
    *,
    phenotypes: Sequence[str] = FIRST_STUDY_PHENOTYPES,
    intervention_kinds: Sequence[str] = ("friction_reduction",),
    replay_fraction: float = 0.5,
) -> ArmSet:
    """Arms A to D as the preregistration describes them.

    ``replay_fraction`` is a protocol input, not a tuned value.
    """
    return ArmSet(
        (
            ArmSpec(
                "A_standard",
                "standard",
                "Continue training the baseline with no failure-specific machinery, at the "
                "same fresh-transition and update budget as every other arm.",
            ),
            ArmSpec(
                "B_uniform_replay",
                "uniform_replay",
                "Seed a fixed fraction of resets from the capsule pool, every capsule equally "
                "likely, no conditioning.",
                SamplerConfig(replay_fraction=replay_fraction),
            ),
            ArmSpec(
                "C_intervention_conditioned",
                "intervention_conditioned",
                "Seed the same fraction of resets, stratified by the physical intervention "
                "that produced each capsule and its intensity bin.",
                SamplerConfig(
                    replay_fraction=replay_fraction, strata=("intervention_kind", "intensity_bin")
                ),
                intervention_kinds=tuple(intervention_kinds),
            ),
            ArmSpec(
                "D_phenotype_conditioned",
                "phenotype_conditioned",
                "Seed the same fraction of resets, stratified by the observed failure phenotype.",
                SamplerConfig(replay_fraction=replay_fraction, strata=("phenotype",)),
                phenotypes=tuple(phenotypes),
            ),
        ),
        budget,
    )


__all__ = [
    "ARM_KINDS",
    "CONDITIONING",
    "REPLAY_SOURCES",
    "SEED_ROW_STRATEGIES",
    "STRATA",
    "ArmSet",
    "ArmSpec",
    "SamplerConfig",
    "default_arm_set",
]
