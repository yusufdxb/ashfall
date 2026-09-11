"""Empirical failure frontiers; thresholds are bounded by observed support."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .basin import BasinObservation
from .scenarios import ScenarioManifest


@dataclass(frozen=True)
class SeverityMetric:
    """An explicit one-dimensional axis, larger means more challenging.

    Slip can use `1-friction` (offset=1,direction=-1), push magnitude +1,
    torque loss `1-torque_scale`, obstacle height +1. Other varying dimensions
    are nuisance parameters and require stratification for a causal axis.
    """

    family: str
    parameter: str
    units: str
    direction: int = 1
    offset: float = 0.0

    def __post_init__(self):
        if (
            self.direction not in (-1, 1)
            or not self.family
            or not self.units
            or not self.parameter
            or not math.isfinite(self.offset)
        ):
            raise ValueError("Declare failure family, units and severity direction")

    def __call__(self, scenario):
        if scenario.family != self.family:
            raise ValueError("Severity metric belongs to a different failure family")
        return self.offset + self.direction * dict(scenario.parameters)[self.parameter]


@dataclass(frozen=True)
class FrontierPoint:
    scenario_id: str
    severity: float
    failures: int
    replicates: int
    probability: float
    uncertainty: float

    def __post_init__(self):
        if not self.scenario_id or not math.isfinite(self.severity):
            raise ValueError("Frontier point requires identity and finite severity")
        if (
            type(self.replicates) is not int
            or type(self.failures) is not int
            or self.replicates <= 0
            or not 0 <= self.failures <= self.replicates
        ):
            raise ValueError("Frontier outcomes must be valid integer binomial counts")
        if not math.isfinite(self.probability) or not math.isclose(
            self.probability, self.failures / self.replicates
        ):
            raise ValueError("Frontier probability must equal observed failures/replicates")
        if not math.isfinite(self.uncertainty) or not 0 <= self.uncertainty <= 1:
            raise ValueError("Frontier uncertainty must be a bounded interval width")


@dataclass(frozen=True)
class RobustnessMargin:
    quantile: float
    threshold: float | None
    censoring: str
    support: tuple[float, float]
    axis: dict
    monotonic_assumption: bool = True
    scenario_ids: tuple[str, ...] = ()

    def to_dict(self):
        return asdict(self)


class FrontierEstimator:
    def __init__(self, metric: SeverityMetric, target_probability=0.5):
        if not 0 < target_probability < 1:
            raise ValueError("Target failure probability must lie inside (0,1)")
        self.metric, self.target_probability = metric, target_probability

    def points(
        self,
        manifest: ScenarioManifest,
        observations: list[BasinObservation],
        *,
        splits=("train", "validation"),
    ):
        scenarios = {s.scenario_id: s for s in manifest.scenarios}
        policies = {o.policy_id for o in observations}
        if len(policies) != 1:
            raise ValueError("Frontier needs one policy at a time")
        result, seen = [], set()
        nuisance_support = set()
        for o in observations:
            if o.scenario_id in seen or o.scenario_id not in scenarios:
                raise ValueError("Duplicate or unknown frontier observation")
            seen.add(o.scenario_id)
            scenario = scenarios[o.scenario_id]
            if scenario.split not in splits:
                raise ValueError("Evaluation observations cannot tune the training frontier")
            nuisance_support.add(
                tuple(
                    (name, value)
                    for name, value in scenario.parameters
                    if name != self.metric.parameter
                )
            )
            if len(nuisance_support) > 1:
                raise ValueError(
                    "R50 requires fixed nuisance parameters; stratify multidimensional basin first"
                )
            lo, hi = o.probability_interval
            result.append(
                FrontierPoint(
                    o.scenario_id,
                    self.metric(scenario),
                    sum(o.failures),
                    len(o.failures),
                    o.failure_probability,
                    hi - lo,
                )
            )
        return sorted(result, key=lambda p: (p.severity, p.scenario_id))

    def estimate(self, points, quantile=None):
        """Weighted isotonic binomial proportions (PAVA), linear interpolation.

        Assumes monotonically increasing failure probability on the declared axis.
        Does not extrapolate beyond samples or manufacture thresholds without a
        crossing. The fitted value is descriptive; replicate uncertainty is separate.
        """
        q = self.target_probability if quantile is None else quantile
        if not points or not 0 < q < 1:
            raise ValueError("Need observations and an interior quantile")
        groups = {}
        if len({p.scenario_id for p in points}) != len(points):
            raise ValueError("Duplicate frontier scenario cannot count as independent evidence")
        for p in points:
            if p.replicates <= 0 or not 0 <= p.failures <= p.replicates:
                raise ValueError("Invalid binomial observation")
            f, n = groups.get(p.severity, (0, 0))
            groups[p.severity] = (f + p.failures, n + p.replicates)
        xs = np.array(sorted(groups), dtype=float)
        blocks = []
        for i, x in enumerate(xs):
            f, n = groups[x]
            blocks.append([i, i, f, n])
            while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
                b = blocks.pop()
                a = blocks.pop()
                blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
        fitted = np.empty(len(xs))
        for first, last, f, n in blocks:
            fitted[first : last + 1] = f / n
        support = (float(xs[0]), float(xs[-1]))
        if fitted[0] > q:
            threshold, censoring = None, "below_support"
        elif fitted[-1] < q:
            threshold, censoring = None, "above_support"
        elif np.all(fitted == q):
            threshold, censoring = None, "unidentified_plateau"
        else:
            idx = int(np.flatnonzero(fitted >= q)[0])
            threshold = (
                float(xs[idx])
                if idx == 0
                else float(np.interp(q, fitted[idx - 1 : idx + 1], xs[idx - 1 : idx + 1]))
            )
            censoring = "observed_crossing"
        return RobustnessMargin(
            q,
            threshold,
            censoring,
            support,
            asdict(self.metric),
            scenario_ids=tuple(sorted(p.scenario_id for p in points)),
        )

    def prioritize(self, points):
        """Multidimensional cases can use uncertainty/boundary weights without R50."""
        weights = np.array(
            [
                max(0.01, 1 - 2 * abs(p.probability - self.target_probability)) + p.uncertainty
                for p in points
            ]
        )
        if not len(weights):
            raise ValueError("No frontier points")
        return weights / weights.sum()


def frontier_shift(baseline: RobustnessMargin, repaired: RobustnessMargin):
    if (
        baseline.axis != repaired.axis
        or baseline.quantile != repaired.quantile
        or baseline.support != repaired.support
        or baseline.scenario_ids != repaired.scenario_ids
    ):
        raise ValueError("Compare identical severity definitions and scenario support")
    if baseline.threshold is None or repaired.threshold is None:
        return None
    return repaired.threshold - baseline.threshold
