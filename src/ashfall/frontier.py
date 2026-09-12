"""Empirical failure frontiers; a threshold is reported only where the data identify one.

The estimator fits a weighted isotonic (pool-adjacent-violators) curve of
failure probability against one declared severity axis and asks where that
curve crosses a target quantile. Four answers are possible and each is named:

``interpolated_crossing``
    The fitted curve is strictly below the quantile at one observed severity
    and strictly above it at the next. The threshold is a linear interpolation
    between those two observed severities. It is a fitted value, never an
    observed one, which is why the earlier name ``observed_crossing`` is gone.
``identified_interval``
    The fitted curve equals the quantile on a plateau spanning several observed
    severities. Under the monotone assumption the crossing lies somewhere in
    that plateau; the data identify an interval, not a point, and the estimate
    reports the interval with ``threshold`` left as None.
``below_support`` / ``above_support``
    The fitted curve is already at or past the quantile at the smallest
    severity, or never reaches it at the largest. Censored, not extrapolated.
``unidentifiable``
    There is no monotone signal: every observation pools into one isotonic
    block, so the fitted curve is flat. No crossing exists in the fit and none
    is manufactured.

Monotonicity is an assumption. Whether the raw proportions honour it is
reported (``monotone_violations``) rather than asserted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .basin import BasinObservation
from .scenarios import ScenarioManifest

CENSORING_KINDS = (
    "interpolated_crossing",
    "identified_interval",
    "below_support",
    "above_support",
    "unidentifiable",
)


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
    """The quantile crossing of a fitted frontier, or the reason there is none.

    ``threshold`` is set only for ``interpolated_crossing``. For
    ``identified_interval`` the data bound the crossing to
    ``threshold_interval`` and no point is reported. ``monotonic_assumption``
    is set from the data: True when the raw proportions are non-decreasing in
    severity, so the isotonic fit changed nothing; False when the fit had to
    pool violations (``monotone_violations`` counts them). It describes whether
    the assumption is contradicted by these observations; it does not establish
    it.
    """

    quantile: float
    threshold: float | None
    censoring: str
    support: tuple[float, float]
    axis: dict
    monotonic_assumption: bool = True
    scenario_ids: tuple[str, ...] = ()
    threshold_interval: tuple[float, float] | None = None
    monotone_violations: int = 0
    raw_probabilities: tuple[tuple[float, float], ...] = ()
    fitted_probabilities: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        if self.censoring not in CENSORING_KINDS:
            raise ValueError(f"unknown censoring {self.censoring!r}; expected {CENSORING_KINDS}")
        if (self.threshold is not None) != (self.censoring == "interpolated_crossing"):
            raise ValueError("a point threshold is reported only for an interpolated crossing")
        if (self.threshold_interval is not None) != (self.censoring == "identified_interval"):
            raise ValueError("a threshold interval is reported only for an identified plateau")
        if self.threshold_interval is not None:
            lo, hi = self.threshold_interval
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError("threshold_interval must be finite and ordered")
        if type(self.monotone_violations) is not int or self.monotone_violations < 0:
            raise ValueError("monotone_violations must be a nonnegative integer")
        if self.monotonic_assumption != (self.monotone_violations == 0):
            raise ValueError("monotonic_assumption must be derived from monotone_violations")

    @property
    def identified(self) -> bool:
        """True when the data locate the crossing, as a point or an interval."""
        return self.censoring in ("interpolated_crossing", "identified_interval")

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
        """Weighted isotonic binomial proportions (PAVA), then the quantile crossing.

        Assumes failure probability is non-decreasing on the declared axis.
        Equal-severity observations are pooled by replicate count before the
        fit. The crossing is reported as a linear interpolation only when the
        fitted curve is strictly below the quantile at one observed severity
        and strictly above it at the next; a plateau at the quantile is
        returned as the identified interval; a fit that never crosses is
        censored at the support boundary; a fit with no monotone signal (one
        isotonic block) is unidentifiable. Nothing is extrapolated. Replicate
        uncertainty is reported per point and is not propagated into the
        crossing.
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
        raw = np.array([groups[x][0] / groups[x][1] for x in xs], dtype=float)
        violations = int(np.sum(np.diff(raw) < 0)) if len(raw) > 1 else 0
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
        threshold: float | None = None
        interval: tuple[float, float] | None = None
        flat = bool(np.ptp(fitted) == 0)
        if (len(blocks) == 1 and violations > 0) or (flat and math.isclose(fitted[0], q)):
            # No monotone signal. Either the violations pooled every observation
            # into one block, so the flat level is an average over a sample that
            # contradicts the assumption, or the curve sits on the quantile
            # everywhere and carries no location at all. Nothing is invented.
            censoring = "unidentifiable"
        elif fitted[0] >= q:
            censoring = "below_support"
        elif fitted[-1] < q:
            censoring = "above_support"
        else:
            idx = int(np.flatnonzero(fitted >= q)[0])
            if math.isclose(fitted[idx], q):
                # The fit sits exactly on the quantile from idx through the end
                # of its block. Under monotonicity the crossing is anywhere in
                # that plateau: an identified set, reported as such.
                last = idx
                while last + 1 < len(fitted) and math.isclose(fitted[last + 1], q):
                    last += 1
                interval = (float(xs[idx]), float(xs[last]))
                censoring = "identified_interval"
            else:
                threshold = float(np.interp(q, fitted[idx - 1 : idx + 1], xs[idx - 1 : idx + 1]))
                censoring = "interpolated_crossing"
        return RobustnessMargin(
            q,
            threshold,
            censoring,
            support,
            asdict(self.metric),
            monotonic_assumption=violations == 0,
            scenario_ids=tuple(sorted(p.scenario_id for p in points)),
            threshold_interval=interval,
            monotone_violations=violations,
            raw_probabilities=tuple((float(x), float(p)) for x, p in zip(xs, raw)),
            fitted_probabilities=tuple((float(x), float(p)) for x, p in zip(xs, fitted)),
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


def frontier_shift_with_reason(
    baseline: RobustnessMargin, repaired: RobustnessMargin
) -> tuple[float | None, str]:
    """Difference of two interpolated crossings, or None and the reason there is none.

    A definitional mismatch (axis, quantile, support or scenario set) is an
    error, because the two margins are not comparable at all. A censored or
    interval-valued side is not an error, but it yields no scalar shift: an
    interval minus a point is an interval, and a censored margin is a bound.
    """
    if (
        baseline.axis != repaired.axis
        or baseline.quantile != repaired.quantile
        or baseline.support != repaired.support
        or baseline.scenario_ids != repaired.scenario_ids
    ):
        raise ValueError("Compare identical severity definitions and scenario support")
    if (
        baseline.censoring != "interpolated_crossing"
        or repaired.censoring != "interpolated_crossing"
    ):
        return None, (
            "no scalar shift: baseline is "
            f"{baseline.censoring}, repaired is {repaired.censoring}; only two interpolated "
            "crossings subtract to a point"
        )
    assert baseline.threshold is not None and repaired.threshold is not None
    return repaired.threshold - baseline.threshold, "difference of interpolated crossings"


def frontier_shift(baseline: RobustnessMargin, repaired: RobustnessMargin):
    """Scalar frontier shift, or None when either side is not an interpolated crossing."""
    return frontier_shift_with_reason(baseline, repaired)[0]
