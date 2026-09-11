"""Baseline-policy counterexample reproduction, separate from randomization."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from scipy.stats import qmc

from .delivery import assert_delivers
from .provenance import content_hash, write_artifact


@dataclass(frozen=True)
class FailureDescriptor:
    mode: str | None
    time_to_failure_s: float | None
    termination_reason: str | None = None
    attitude_rms_rad: float | None = None
    velocity_error_mps: float | None = None
    contact_signature: tuple[float, ...] | None = None

    def __post_init__(self):
        for value in (self.time_to_failure_s, self.attitude_rms_rad, self.velocity_error_mps):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Descriptor values must be finite and nonnegative")
        if self.contact_signature is not None:
            if not self.contact_signature or not all(
                math.isfinite(v) for v in self.contact_signature
            ):
                raise ValueError("Contact signature must be nonempty and finite")
            object.__setattr__(self, "contact_signature", tuple(self.contact_signature))
        if self.mode is not None and self.time_to_failure_s is None:
            raise ValueError("Failure event requires time to onset")


@dataclass(frozen=True)
class ReproductionConfig:
    baseline_policy_id: str
    target: FailureDescriptor
    seed_row: int
    seeds: tuple[int, ...] = (101, 103, 107)
    minimum_match_fraction: float = 2 / 3
    minimum_similarity: float = 0.75
    time_tolerance_s: float = 0.5
    attitude_tolerance_rad: float = 0.3
    velocity_tolerance_mps: float = 0.3
    require_termination_match: bool = False

    def __post_init__(self):
        if not self.baseline_policy_id or self.target.mode is None or self.seed_row < 0:
            raise ValueError("Baseline, target failure and valid seed row required")
        if type(self.seed_row) is not int:
            raise ValueError("seed_row must be an integer")
        if any(type(seed) is not int for seed in self.seeds):
            raise ValueError("Reproduction seeds must be integers")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Distinct reproduction seeds required")
        if not 0 < self.minimum_match_fraction <= 1 or not 0 < self.minimum_similarity <= 1:
            raise ValueError("Invalid reproduction acceptance threshold")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.time_tolerance_s,
                self.attitude_tolerance_rad,
                self.velocity_tolerance_mps,
            )
        ):
            raise ValueError("Similarity tolerances must be positive")


class FailureSimilarity:
    """Mode is mandatory; absent required target channels cannot earn credit.

    Uses bounded absolute differences in preregistered summary channels, not a
    claim of trajectory equivalence. Type-specific configs select the channels.
    """

    def __init__(self, config: ReproductionConfig):
        self.config = config

    def score(self, observed: FailureDescriptor) -> float:
        c, target = self.config, self.config.target
        if observed.mode != target.mode:
            return 0.0
        if c.require_termination_match and observed.termination_reason != target.termination_reason:
            return 0.0
        terms = []
        for name, tolerance in [
            ("time_to_failure_s", c.time_tolerance_s),
            ("attitude_rms_rad", c.attitude_tolerance_rad),
            ("velocity_error_mps", c.velocity_tolerance_mps),
        ]:
            expected, actual = getattr(target, name), getattr(observed, name)
            if expected is not None:
                if actual is None:
                    return 0.0
                terms.append(max(0.0, 1 - abs(expected - actual) / tolerance))
        if target.contact_signature is not None:
            if observed.contact_signature is None or len(target.contact_signature) != len(
                observed.contact_signature
            ):
                return 0.0
            terms.append(
                max(
                    0.0,
                    1
                    - float(
                        np.mean(
                            np.abs(np.array(target.contact_signature) - observed.contact_signature)
                        )
                    ),
                )
            )
        return float(np.mean(terms)) if terms else 0.0


@dataclass(frozen=True)
class ReproductionCandidate:
    capsule_id: str
    parameters: tuple[tuple[str, float], ...]
    seed_row: int

    def __post_init__(self):
        if not self.capsule_id or type(self.seed_row) is not int or self.seed_row < 0:
            raise ValueError("Candidate needs capsule identity and nonnegative integer seed row")
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("Duplicate candidate parameter")
        if any(not name or not math.isfinite(value) for name, value in self.parameters):
            raise ValueError("Candidate parameters must be named and finite")
        object.__setattr__(self, "parameters", tuple(sorted(tuple(p) for p in self.parameters)))

    @property
    def candidate_id(self):
        return content_hash(asdict(self))


@dataclass(frozen=True)
class ReproductionResult:
    candidate: ReproductionCandidate
    config: ReproductionConfig
    observations: tuple[FailureDescriptor, ...]
    scores: tuple[float, ...]
    evidence_kind: str
    backend_id: str

    def __post_init__(self):
        if self.candidate.seed_row != self.config.seed_row:
            raise ValueError("Candidate seed row differs from reproduction config")
        if len(self.observations) != len(self.config.seeds) or len(self.scores) != len(
            self.config.seeds
        ):
            raise ValueError("Every reproduction seed requires one observation and score")
        if not self.backend_id or self.evidence_kind not in {"mock", "simulation"}:
            raise ValueError("Reproduction requires backend identity and explicit evidence kind")
        expected = tuple(FailureSimilarity(self.config).score(o) for o in self.observations)
        if any(not math.isfinite(score) for score in self.scores) or tuple(self.scores) != expected:
            raise ValueError("Reproduction scores do not match observed descriptors")
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "scores", tuple(self.scores))

    @property
    def status(self):
        matched = sum(s >= self.config.minimum_similarity for s in self.scores)
        return (
            "REPRODUCED"
            if matched / len(self.config.seeds) >= self.config.minimum_match_fraction
            else "UNREPRODUCED"
        )

    @property
    def reproduction_id(self):
        return content_hash(asdict(self))

    def assert_eligible(self, *, allow_mock=False):
        if self.status != "REPRODUCED":
            raise ValueError("UNREPRODUCED capsules cannot enter repair")
        if self.evidence_kind != "simulation" and not (allow_mock and self.evidence_kind == "mock"):
            raise ValueError(
                "Simulator reproduction evidence required; mock is software evidence only"
            )

    def to_dict(self):
        return {**asdict(self), "status": self.status, "reproduction_id": self.reproduction_id}

    def save(self, path):
        return write_artifact(path, self.to_dict())


class ReproductionBackend(Protocol):
    evidence_kind: str
    backend_id: str

    def replay(
        self, capsule, candidate: ReproductionCandidate, policy_id: str, seed: int
    ) -> FailureDescriptor: ...


class ReproductionGate:
    def __init__(self, config: ReproductionConfig):
        self.config = config

    def run(self, capsule, candidate: ReproductionCandidate, backend: ReproductionBackend):
        if candidate.capsule_id != capsule.capsule_id or candidate.seed_row != self.config.seed_row:
            raise ValueError("Capsule or seed row differs from reproduction protocol")
        if capsule.failure_mode != self.config.target.mode:
            raise ValueError("Reproduction target mode differs from original capsule failure")
        if capsule.policy_id != self.config.baseline_policy_id:
            raise ValueError("Reproduction must use the capsule baseline policy identity")
        if not capsule.pre_failure_start_index <= candidate.seed_row < capsule.failure_onset_index:
            # Strictly before onset. The upper bound was inclusive, so seeding
            # AT onset was gate-legal, which makes reproducing the failure
            # close to tautological: the seeded state already is the failure.
            raise ValueError("Reproduction seed must lie in recorded pre-onset window")
        frame = capsule.frames[candidate.seed_row]
        if frame.missing_reset_fields:
            # The gate never checked this, so an unresettable capsule could
            # reach REPRODUCED and pass assert_eligible while the real
            # simulator would refuse it.
            raise ValueError(
                f"Reproduction seed row lacks reset state: {frame.missing_reset_fields}"
            )
        # Refuse an undeliverable mode and an undelivered seed state before any
        # score is computed. Without this the gate scores whatever the
        # simulator did, with no diagnostic that the treatment never arrived,
        # which is exactly how the Phase-I result came to be uninformative.
        assert_delivers(capsule, candidate.seed_row)
        observations = tuple(
            backend.replay(capsule, candidate, self.config.baseline_policy_id, s)
            for s in self.config.seeds
        )
        similarity = FailureSimilarity(self.config)
        return ReproductionResult(
            candidate,
            self.config,
            observations,
            tuple(similarity.score(o) for o in observations),
            backend.evidence_kind,
            backend.backend_id,
        )

    def search(
        self, capsule, backend, bounds: dict[str, tuple[float, float]], *, count=16, search_seed=0
    ):
        """Unknown parameters are search dimensions, never inferred facts.

        All attempts are returned for persistence, including unsuccessful attempts.
        """
        parameters = sample_parameters(bounds, count, search_seed)
        return [
            self.run(
                capsule,
                ReproductionCandidate(capsule.capsule_id, tuple(p.items()), self.config.seed_row),
                backend,
            )
            for p in parameters
        ]


def sample_parameters(bounds, count, seed):
    if count <= 0 or not bounds:
        raise ValueError("Nonempty bounds and positive count required")
    names = sorted(bounds)
    low, high = np.array([bounds[k] for k in names], dtype=float).T
    if not np.all(np.isfinite([low, high])) or np.any(low >= high):
        raise ValueError("Search bounds must be finite and increasing")
    samples = qmc.scale(qmc.Halton(len(names), scramble=True, seed=seed).random(count), low, high)
    return [dict(zip(names, map(float, row))) for row in samples]


def load_reproduction(path):
    """Recompute gate decisions on load and reject changed recorded scores/identity."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    candidate = data["candidate"]
    candidate["parameters"] = tuple(tuple(v) for v in candidate["parameters"])
    config = data["config"]
    config["target"] = FailureDescriptor(**config["target"])
    config["seeds"] = tuple(config["seeds"])
    config = ReproductionConfig(**config)
    observations = tuple(FailureDescriptor(**o) for o in data["observations"])
    if len(observations) != len(config.seeds):
        raise ValueError("Reproduction replicate evidence incomplete")
    scores = tuple(FailureSimilarity(config).score(o) for o in observations)
    if list(scores) != data["scores"]:
        raise ValueError("Reproduction scores do not match recorded observations")
    result = ReproductionResult(
        ReproductionCandidate(**candidate),
        config,
        observations,
        scores,
        data["evidence_kind"],
        data["backend_id"],
    )
    if data["status"] != result.status or data["reproduction_id"] != result.reproduction_id:
        raise ValueError("Reproduction identity or verdict changed")
    return result
