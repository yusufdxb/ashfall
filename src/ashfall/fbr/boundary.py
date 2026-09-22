"""Failure boundaries and the local training distributions built around them.

A boundary is estimated per seed state: from one restored pre-failure state,
the current policy is replayed at several friction values with independent
simulator seeds, and the friction at which its failure probability crosses
0.5 is located with the isotonic estimator in :mod:`ashfall.frontier`. Only an
identified crossing (a point or a plateau) becomes a boundary; a censored or
unidentifiable frontier is refused, because a curriculum centred on a guess is
not a failure-boundary curriculum.

The same sampler class serves two arms. Arm D (FBR) builds it from seed states
taken out of a deployment failure capsule; arm C builds it from seed states
taken out of the baseline's own successful rollouts. Everything else (the
neighbourhood widths, the nominal fraction, the boundary estimator, the
budget) is shared, so the two arms differ only in where the seeds came from.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from ashfall.frontier import FrontierEstimator, FrontierPoint, SeverityMetric
from ashfall.provenance import content_hash

SEED_SOURCES: tuple[str, ...] = (
    "deployment_failure",
    "hardware_failure",
    "nominal_rollout",
    "simulator_search",
)

#: Severity on the friction axis: 1 - mu, larger is harder.
FRICTION_SEVERITY = SeverityMetric(
    family="slip", parameter="dynamic_friction", units="dimensionless", direction=-1, offset=1.0
)


class BoundaryNotIdentified(ValueError):
    """No boundary from this seed; ``censoring`` names why (a frontier censoring kind)."""

    def __init__(self, seed_id: str, censoring: str):
        self.seed_id, self.censoring = seed_id, censoring
        super().__init__(
            f"no failure boundary identified from seed {seed_id}: {censoring}; widen the "
            "declared friction support or drop the seed"
        )


@dataclass(frozen=True)
class BoundaryEstimate:
    seed_id: str
    seed_source: str
    mu_boundary: float
    mu_interval: tuple[float, float]
    censoring: str
    raw: tuple[tuple[float, int, int], ...]  # (mu, failures, replicates)

    def __post_init__(self):
        if self.seed_source not in SEED_SOURCES:
            raise ValueError(f"seed_source must be one of {SEED_SOURCES}")
        lo, hi = self.mu_interval
        if not (math.isfinite(self.mu_boundary) and lo <= self.mu_boundary <= hi):
            raise ValueError("mu_boundary must lie in its interval")

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_boundary(
    seed_id: str,
    outcomes: Mapping[float, Sequence[bool]],
    *,
    seed_source: str,
    target: float = 0.5,
) -> BoundaryEstimate:
    """Locate the friction where failure probability crosses ``target`` from one seed.

    ``outcomes`` maps a friction value to replicate failure booleans. Raises
    when the crossing is censored by the search support or unidentifiable.
    """
    if len(outcomes) < 3:
        raise ValueError("a boundary needs at least three friction levels")
    points = []
    for mu, failures in outcomes.items():
        values = [bool(v) for v in failures]
        if not values or not 0.0 < float(mu) <= 2.0:
            raise ValueError("each friction level needs replicate outcomes and mu in (0, 2]")
        points.append(
            FrontierPoint(
                scenario_id=f"{seed_id}@mu={float(mu):.6f}",
                severity=1.0 - float(mu),
                failures=sum(values),
                replicates=len(values),
                probability=sum(values) / len(values),
                uncertainty=0.0,
            )
        )
    margin = FrontierEstimator(FRICTION_SEVERITY, target).estimate(points)
    if margin.censoring == "interpolated_crossing":
        assert margin.threshold is not None
        mu_b = 1.0 - margin.threshold
        interval = (mu_b, mu_b)
    elif margin.censoring == "identified_interval":
        assert margin.threshold_interval is not None
        lo, hi = margin.threshold_interval
        interval = (1.0 - hi, 1.0 - lo)
        mu_b = 0.5 * (interval[0] + interval[1])
    else:
        raise BoundaryNotIdentified(seed_id, margin.censoring)
    raw = tuple(sorted((float(mu), sum(map(bool, f)), len(f)) for mu, f in outcomes.items()))
    return BoundaryEstimate(seed_id, seed_source, mu_b, interval, margin.censoring, raw)


@dataclass(frozen=True)
class LocalNeighborhood:
    """How far FBR samples around a boundary. Predeclared, identical for arms C and D."""

    mu_half_width: float = 0.05
    mu_min: float = 0.02
    mu_max: float = 1.2
    state_noise_scale: float = 1.0  # multiplies backend-declared per-channel noise
    command_jitter_mps: float = 0.05

    def __post_init__(self):
        values = (self.mu_half_width, self.mu_min, self.mu_max, self.command_jitter_mps)
        if not all(math.isfinite(v) and v >= 0 for v in values) or self.mu_min >= self.mu_max:
            raise ValueError("neighbourhood widths must be nonnegative and mu_min < mu_max")
        if not math.isfinite(self.state_noise_scale) or self.state_noise_scale < 0:
            raise ValueError("state_noise_scale must be nonnegative")


@dataclass(frozen=True)
class ReplayDraw:
    """One training reset: nominal (baseline distribution) or a boundary replay."""

    kind: str  # "nominal", "broad" or "boundary"
    seed_id: str | None = None
    mu: float | None = None
    command_offset_mps: float = 0.0
    noise_seed: int = 0

    def __post_init__(self):
        if self.kind not in ("nominal", "broad", "boundary"):
            raise ValueError(f"unknown draw kind {self.kind!r}")
        if (self.kind == "boundary") != (self.seed_id is not None):
            raise ValueError("exactly the boundary draws carry a seed state")
        if self.kind in ("broad", "boundary") and (self.mu is None or not math.isfinite(self.mu)):
            raise ValueError("broad and boundary draws carry a friction value")


class BoundarySampler:
    """Arms C and D: boundary replays around seed states, mixed with nominal resets."""

    def __init__(
        self,
        boundaries: Sequence[BoundaryEstimate],
        neighborhood: LocalNeighborhood,
        *,
        nominal_fraction: float,
        rng_seed: int,
        one_sided: bool = False,
        base_mu_range: tuple[float, float] | None = None,
    ):
        """``one_sided`` samples friction on ``[mu_min, mu_hi + w]`` (the boundary and
        everything harder) instead of a band around the boundary. ``base_mu_range``
        makes the non-replay draws broad-DR draws on that range instead of nominal
        resets, so the arm is "broad DR plus replays"."""
        if not boundaries:
            raise ValueError("a boundary sampler needs at least one identified boundary")
        sources = {b.seed_source for b in boundaries}
        if len(sources) != 1:
            raise ValueError("one sampler, one seed source; mixing sources confounds the arms")
        if not 0.0 <= nominal_fraction < 1.0:
            raise ValueError("nominal_fraction must lie in [0, 1)")
        self.boundaries = tuple(boundaries)
        self.neighborhood = neighborhood
        self.nominal_fraction = float(nominal_fraction)
        self.seed_source = sources.pop()
        self.one_sided = bool(one_sided)
        if base_mu_range is not None:
            lo, hi = (float(v) for v in base_mu_range)
            if not 0.0 < lo < hi:
                raise ValueError("base_mu_range must satisfy 0 < lo < hi")
            base_mu_range = (lo, hi)
        self.base_mu_range = base_mu_range
        self.rng = np.random.default_rng(rng_seed)

    def sample(self, n: int) -> list[ReplayDraw]:
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        nb = self.neighborhood
        draws = []
        for _ in range(n):
            noise_seed = int(self.rng.integers(2**31 - 1))
            if self.rng.random() < self.nominal_fraction:
                if self.base_mu_range is None:
                    draws.append(ReplayDraw("nominal", noise_seed=noise_seed))
                else:
                    mu_base = float(self.rng.uniform(*self.base_mu_range))
                    draws.append(ReplayDraw("broad", mu=mu_base, noise_seed=noise_seed))
                continue
            b = self.boundaries[int(self.rng.integers(len(self.boundaries)))]
            lo = (
                nb.mu_min if self.one_sided else max(nb.mu_min, b.mu_interval[0] - nb.mu_half_width)
            )
            hi = min(nb.mu_max, b.mu_interval[1] + nb.mu_half_width)
            mu = float(self.rng.uniform(lo, hi)) if hi > lo else lo
            offset = float(self.rng.uniform(-nb.command_jitter_mps, nb.command_jitter_mps))
            draws.append(ReplayDraw("boundary", b.seed_id, mu, offset, noise_seed))
        return draws

    def describe(self) -> dict:
        return {
            "sampler": "boundary",
            "seed_source": self.seed_source,
            "nominal_fraction": self.nominal_fraction,
            "one_sided": self.one_sided,
            "base_mu_range": self.base_mu_range,
            "neighborhood": asdict(self.neighborhood),
            "boundaries": [b.to_dict() for b in self.boundaries],
            "sampler_id": content_hash(
                {
                    "nominal_fraction": self.nominal_fraction,
                    "one_sided": self.one_sided,
                    "base_mu_range": self.base_mu_range,
                    "neighborhood": asdict(self.neighborhood),
                    "boundaries": [b.to_dict() for b in self.boundaries],
                }
            ),
        }


class BroadSampler:
    """Arm B: friction drawn uniformly over a broad declared range, nominal resets."""

    def __init__(self, mu_range: tuple[float, float], *, rng_seed: int):
        lo, hi = (float(v) for v in mu_range)
        if not (0.0 < lo < hi and math.isfinite(hi)):
            raise ValueError("broad range must satisfy 0 < lo < hi")
        self.mu_range = (lo, hi)
        self.rng = np.random.default_rng(rng_seed)

    def sample(self, n: int) -> list[ReplayDraw]:
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        return [
            ReplayDraw(
                "broad",
                mu=float(self.rng.uniform(*self.mu_range)),
                noise_seed=int(self.rng.integers(2**31 - 1)),
            )
            for _ in range(n)
        ]

    def describe(self) -> dict:
        return {"sampler": "broad", "mu_range": self.mu_range}


def boundary_mass(probabilities: Sequence[float]) -> float:
    """Mean of sqrt(p(1-p)) over training contexts: the informativeness diagnostic.

    For a binary outcome R with success probability p under the current
    policy, E[(R - p)^2] = p(1-p), and by Cauchy-Schwarz the score-function
    gradient of p is bounded by sqrt(p(1-p)) times the root-mean-square score
    norm. Contexts the policy always solves or always fails contribute nothing
    to the gradient of the success probability. This is a diagnostic of a
    training distribution, not a guarantee about shaped-reward PPO.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) == 0 or np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("probabilities must be a nonempty vector in [0, 1]")
    return float(np.mean(np.sqrt(p * (1.0 - p))))
