"""Equal compute, enforced rather than declared.

The v2 protocol named five arms and one shared ``training_steps_per_arm``
integer that nothing read. This module makes the budget a record every arm
must match: each training run emits a :class:`TrainingArtifact` with the eight
measured quantities and its initial and final checkpoint hashes, and
:func:`assert_equal_compute` refuses a comparison whose artifacts fall outside
the budget's declared tolerances. A comparison that was never checked cannot
be reported as equal-compute.

Two quantities need their meaning stated once, here:

``fresh_env_transitions``
    Every simulator transition PPO consumed. This is the compute that must be
    equal across arms.
``replay_transitions``
    The transitions that began in an episode seeded from a replayed capsule
    state (or drawn from stored data, for a buffer-based arm). It is a subset
    of the fresh transitions for a reset-seeded curriculum, not an addition to
    them. It is the dose the treatment arms received and must match across
    the arms that replay; the standard arm receives none.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ashfall.provenance import content_hash

#: The eight quantities every training artifact records, in this order.
BUDGET_FIELDS: tuple[str, ...] = (
    "fresh_env_transitions",
    "replay_transitions",
    "episodes",
    "optimizer_updates",
    "rollout_length",
    "num_envs",
    "wall_clock_s",
    "gpu_time_s",
)

INTEGER_FIELDS: tuple[str, ...] = (
    "fresh_env_transitions",
    "replay_transitions",
    "episodes",
    "optimizer_updates",
    "rollout_length",
    "num_envs",
)

#: Fields whose tolerance may never be ``recorded``: an equal-compute claim
#: rests on these and they must be enforced.
ENFORCED_MINIMUM: tuple[str, ...] = (
    "fresh_env_transitions",
    "optimizer_updates",
    "num_envs",
    "rollout_length",
)

TOLERANCE_KINDS: tuple[str, ...] = ("exact", "absolute", "relative", "recorded")


class ProtocolViolation(ValueError):
    """A comparison that does not satisfy the preregistered compute protocol."""


def _hex64(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


@dataclass(frozen=True)
class Tolerance:
    """How far an artifact may sit from the budget on one field.

    ``exact``      the values must be equal;
    ``absolute``   ``|observed - budget| <= value``;
    ``relative``   ``|observed - budget| <= value * budget``;
    ``recorded``   the field is recorded and reported, not enforced. Only
                   allowed on fields outside :data:`ENFORCED_MINIMUM`.
    """

    kind: str = "exact"
    value: float = 0.0

    def __post_init__(self):
        if self.kind not in TOLERANCE_KINDS:
            raise ValueError(f"unknown tolerance kind {self.kind!r}; expected {TOLERANCE_KINDS}")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("tolerance value must be finite and nonnegative")
        if self.kind in ("exact", "recorded") and self.value != 0.0:
            raise ValueError(f"{self.kind} tolerance carries no value")
        if self.kind in ("absolute", "relative") and self.value == 0.0:
            raise ValueError(f"{self.kind} tolerance with value 0 is 'exact'; say so")

    def allows(self, observed: float, expected: float) -> bool:
        if self.kind == "recorded":
            return True
        if self.kind == "exact":
            return observed == expected
        if self.kind == "absolute":
            return abs(observed - expected) <= self.value
        return abs(observed - expected) <= self.value * abs(expected)

    def to_dict(self) -> dict:
        return asdict(self)


def default_tolerances() -> dict[str, Tolerance]:
    """Discrete compute quantities exact; behaviour-dependent ones recorded.

    ``episodes`` is an outcome of how often episodes terminate, which a
    treatment changes by design, so it is recorded rather than enforced by
    default. Wall-clock and GPU time vary with machine load; a protocol that
    wants them bounded declares a relative tolerance explicitly.
    """
    return {
        "fresh_env_transitions": Tolerance("exact"),
        "replay_transitions": Tolerance("exact"),
        "episodes": Tolerance("recorded"),
        "optimizer_updates": Tolerance("exact"),
        "rollout_length": Tolerance("exact"),
        "num_envs": Tolerance("exact"),
        "wall_clock_s": Tolerance("recorded"),
        "gpu_time_s": Tolerance("recorded"),
    }


@dataclass(frozen=True)
class ComputeBudget:
    """The budget every arm must receive, with a tolerance per field."""

    fresh_env_transitions: int
    replay_transitions: int
    episodes: int
    optimizer_updates: int
    rollout_length: int
    num_envs: int
    wall_clock_s: float
    gpu_time_s: float
    tolerances: Mapping[str, Tolerance] = field(default_factory=default_tolerances)

    def __post_init__(self):
        for name in INTEGER_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"budget {name} must be a nonnegative integer")
        for name in ("wall_clock_s", "gpu_time_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"budget {name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"budget {name} must be finite and nonnegative")
        if self.fresh_env_transitions == 0 or self.optimizer_updates == 0:
            raise ValueError("a training budget needs positive transitions and updates")
        if self.num_envs == 0 or self.rollout_length == 0:
            raise ValueError("a training budget needs positive num_envs and rollout_length")
        tolerances = {}
        for name, tolerance in dict(self.tolerances).items():
            if name not in BUDGET_FIELDS:
                raise ValueError(f"tolerance for unknown budget field {name!r}")
            if isinstance(tolerance, Mapping):
                tolerance = Tolerance(**tolerance)
            if not isinstance(tolerance, Tolerance):
                raise ValueError(f"tolerance for {name} must be a Tolerance")
            tolerances[name] = tolerance
        missing = [name for name in BUDGET_FIELDS if name not in tolerances]
        if missing:
            raise ValueError(f"every budget field needs a tolerance; missing {missing}")
        unenforced = [n for n in ENFORCED_MINIMUM if tolerances[n].kind == "recorded"]
        if unenforced:
            raise ValueError(
                f"{unenforced} cannot be merely recorded; an equal-compute claim rests on them"
            )
        object.__setattr__(self, "tolerances", tolerances)

    @property
    def budget_id(self) -> str:
        return "budget_" + content_hash(self.to_dict(include_id=False))

    def expected(self, name: str) -> float:
        return getattr(self, name)

    def to_dict(self, include_id: bool = True) -> dict:
        data = {name: getattr(self, name) for name in BUDGET_FIELDS}
        data["tolerances"] = {k: v.to_dict() for k, v in self.tolerances.items()}
        if include_id:
            data["budget_id"] = self.budget_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComputeBudget":
        values = {k: v for k, v in data.items() if k in BUDGET_FIELDS}
        values["tolerances"] = {k: Tolerance(**v) for k, v in data["tolerances"].items()}
        result = cls(**values)
        if data.get("budget_id") not in (None, result.budget_id):
            raise ValueError("budget content identity mismatch")
        return result


@dataclass(frozen=True)
class TrainingArtifact:
    """What one training run consumed and produced. Every field is required."""

    arm: str
    training_seed: int
    fresh_env_transitions: int
    replay_transitions: int
    episodes: int
    optimizer_updates: int
    rollout_length: int
    num_envs: int
    wall_clock_s: float
    gpu_time_s: float
    initial_checkpoint_sha256: str
    final_checkpoint_sha256: str
    config_hashes: Mapping[str, str]
    artifact_id: str = ""

    def __post_init__(self):
        if not isinstance(self.arm, str) or not self.arm.strip():
            raise ValueError("artifact needs an arm name")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValueError("training_seed must be a nonnegative integer")
        for name in INTEGER_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"artifact {name} must be a nonnegative integer")
        for name in ("wall_clock_s", "gpu_time_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"artifact {name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"artifact {name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        _hex64(self.initial_checkpoint_sha256, "initial_checkpoint_sha256")
        _hex64(self.final_checkpoint_sha256, "final_checkpoint_sha256")
        if (
            self.optimizer_updates > 0
            and self.final_checkpoint_sha256 == self.initial_checkpoint_sha256
        ):
            raise ValueError(
                "final checkpoint equals the initial one after optimizer updates; the run "
                "did not train, or the wrong file was hashed"
            )
        if (
            self.optimizer_updates == 0
            and self.final_checkpoint_sha256 != self.initial_checkpoint_sha256
        ):
            raise ValueError("a run with zero optimizer updates cannot change the checkpoint")
        if self.replay_transitions > self.fresh_env_transitions:
            raise ValueError("replay transitions are a subset of the fresh transitions")
        if not isinstance(self.config_hashes, Mapping) or not self.config_hashes:
            raise ValueError("config_hashes must name at least one configuration")
        hashes = {str(k): _hex64(v, f"config_hashes[{k!r}]") for k, v in self.config_hashes.items()}
        object.__setattr__(self, "config_hashes", dict(sorted(hashes.items())))
        expected = self.content_id()
        if self.artifact_id and self.artifact_id != expected:
            raise ValueError("artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected)

    def content_id(self) -> str:
        return "train_" + content_hash(self.to_dict(include_id=False))

    def measured(self, name: str) -> float:
        if name not in BUDGET_FIELDS:
            raise KeyError(name)
        return getattr(self, name)

    def to_dict(self, include_id: bool = True) -> dict:
        data = {
            "arm": self.arm,
            "training_seed": self.training_seed,
            **{name: getattr(self, name) for name in BUDGET_FIELDS},
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "final_checkpoint_sha256": self.final_checkpoint_sha256,
            "config_hashes": dict(self.config_hashes),
        }
        if include_id:
            data["artifact_id"] = self.artifact_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingArtifact":
        required = set(cls.__dataclass_fields__) - {"artifact_id"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"training artifact is missing fields {missing}")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def artifact_from_rsl_rl(
    *,
    arm: str,
    training_seed: int,
    iterations: int,
    num_envs: int,
    num_steps_per_env: int,
    num_learning_epochs: int,
    num_mini_batches: int,
    episodes: int,
    replay_transitions: int,
    wall_clock_s: float,
    gpu_time_s: float,
    initial_checkpoint_sha256: str,
    final_checkpoint_sha256: str,
    config_hashes: Mapping[str, str],
) -> TrainingArtifact:
    """Derive the compute quantities from rsl_rl's runner parameters.

    ``OnPolicyRunner.learn`` collects ``num_steps_per_env`` transitions from
    each of ``num_envs`` environments per iteration and performs
    ``num_learning_epochs * num_mini_batches`` gradient updates per
    iteration. Recording those products here, from the numbers the runner was
    actually configured with, means no caller has to reconstruct them later.
    """
    for name, value in (
        ("iterations", iterations),
        ("num_envs", num_envs),
        ("num_steps_per_env", num_steps_per_env),
        ("num_learning_epochs", num_learning_epochs),
        ("num_mini_batches", num_mini_batches),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return TrainingArtifact(
        arm=arm,
        training_seed=training_seed,
        fresh_env_transitions=iterations * num_envs * num_steps_per_env,
        replay_transitions=replay_transitions,
        episodes=episodes,
        optimizer_updates=iterations * num_learning_epochs * num_mini_batches,
        rollout_length=num_steps_per_env,
        num_envs=num_envs,
        wall_clock_s=wall_clock_s,
        gpu_time_s=gpu_time_s,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        final_checkpoint_sha256=final_checkpoint_sha256,
        config_hashes=config_hashes,
    )


@dataclass(frozen=True)
class Violation:
    arm: str
    training_seed: int | None
    field: str
    observed: Any
    expected: Any
    tolerance: str

    def __str__(self) -> str:
        seed = "" if self.training_seed is None else f" seed {self.training_seed}"
        return (
            f"arm {self.arm!r}{seed}: {self.field} observed {self.observed!r}, "
            f"expected {self.expected!r} ({self.tolerance})"
        )


@dataclass(frozen=True)
class ComplianceReport:
    budget_id: str
    artifact_ids: tuple[str, ...]
    violations: tuple[Violation, ...]
    checked_fields: tuple[str, ...]
    recorded_only_fields: tuple[str, ...]

    @property
    def compliant(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {
            "budget_id": self.budget_id,
            "artifact_ids": list(self.artifact_ids),
            "compliant": self.compliant,
            "violations": [asdict(v) for v in self.violations],
            "checked_fields": list(self.checked_fields),
            "recorded_only_fields": list(self.recorded_only_fields),
        }


def compare_arms(
    artifacts: Sequence[TrainingArtifact],
    budget: ComputeBudget,
    *,
    replay_arms: Iterable[str] = (),
    require_shared_initial_checkpoint: bool = True,
    require_paired_seeds: bool = True,
) -> ComplianceReport:
    """Check every artifact against the budget; never raises on a violation.

    ``replay_arms`` names the arms that are supposed to receive the replay
    dose; every other arm must report zero replay transitions.
    """
    artifacts = list(artifacts)
    if not artifacts:
        raise ValueError("no training artifacts to compare")
    if len({a.artifact_id for a in artifacts}) != len(artifacts):
        raise ValueError("duplicate training artifact")
    replay = set(replay_arms)
    violations: list[Violation] = []
    for artifact in artifacts:
        for name in BUDGET_FIELDS:
            tolerance = budget.tolerances[name]
            expected = budget.expected(name)
            if name == "replay_transitions" and artifact.arm not in replay:
                expected = 0
            observed = artifact.measured(name)
            if not tolerance.allows(observed, expected):
                violations.append(
                    Violation(
                        artifact.arm,
                        artifact.training_seed,
                        name,
                        observed,
                        expected,
                        f"{tolerance.kind}"
                        + (
                            f" {tolerance.value}"
                            if tolerance.kind in ("absolute", "relative")
                            else ""
                        ),
                    )
                )
    if require_shared_initial_checkpoint:
        initial = {a.initial_checkpoint_sha256 for a in artifacts}
        if len(initial) != 1:
            for artifact in artifacts:
                violations.append(
                    Violation(
                        artifact.arm,
                        artifact.training_seed,
                        "initial_checkpoint_sha256",
                        artifact.initial_checkpoint_sha256[:12],
                        "one shared baseline",
                        "exact",
                    )
                )
    keys = {(a.arm, a.training_seed) for a in artifacts}
    if len(keys) != len(artifacts):
        raise ValueError("two artifacts share an (arm, training_seed) cell")
    if require_paired_seeds:
        seeds_by_arm: dict[str, set[int]] = {}
        for artifact in artifacts:
            seeds_by_arm.setdefault(artifact.arm, set()).add(artifact.training_seed)
        union = set().union(*seeds_by_arm.values())
        for arm, seeds in sorted(seeds_by_arm.items()):
            for seed in sorted(union - seeds):
                violations.append(
                    Violation(arm, seed, "training_seed", "absent", "paired across arms", "exact")
                )
    checked = tuple(n for n in BUDGET_FIELDS if budget.tolerances[n].kind != "recorded")
    recorded = tuple(n for n in BUDGET_FIELDS if budget.tolerances[n].kind == "recorded")
    return ComplianceReport(
        budget.budget_id,
        tuple(a.artifact_id for a in artifacts),
        tuple(violations),
        checked,
        recorded,
    )


def assert_equal_compute(
    artifacts: Sequence[TrainingArtifact], budget: ComputeBudget, **kwargs
) -> ComplianceReport:
    """Raise :class:`ProtocolViolation` unless every artifact matches the budget."""
    report = compare_arms(artifacts, budget, **kwargs)
    if not report.compliant:
        raise ProtocolViolation(
            "compute protocol violated; the comparison cannot be reported as equal-compute: "
            + "; ".join(str(v) for v in report.violations)
        )
    return report


__all__ = [
    "BUDGET_FIELDS",
    "ENFORCED_MINIMUM",
    "INTEGER_FIELDS",
    "TOLERANCE_KINDS",
    "ComplianceReport",
    "ComputeBudget",
    "ProtocolViolation",
    "Tolerance",
    "TrainingArtifact",
    "Violation",
    "artifact_from_rsl_rl",
    "assert_equal_compute",
    "compare_arms",
    "default_tolerances",
]
