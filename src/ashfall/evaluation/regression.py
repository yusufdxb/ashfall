"""Explicit target improvement and nominal degradation budgets.

A budget is met only when the interval bound that could violate it stays
inside it. A degenerate interval (every matched scenario delta identical, as
happens whenever both policies succeed everywhere) has no width and therefore
cannot show that. For binary metrics the gate falls back to the exact
conservative bound in :mod:`ashfall.evaluation.paired`; for continuous metrics
it fails closed with a named reason.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ashfall.evaluation.paired import paired_binary_effect_bounds, paired_comparison

#: Metrics whose per-episode values are 0/1 and admit an exact bound.
BINARY_METRICS = ("success", "intervention_required")


@dataclass(frozen=True)
class RegressionBudget:
    """Preregistered nominal-degradation margins.

    With ``require_interval_within_budget`` the bound that could violate a
    margin must lie inside it. A margin of exactly zero can then never be
    established from a finite sample, because every valid upper bound on a
    rate increase is strictly positive; a study that wants interval gating
    must preregister a positive margin, and a study that keeps a zero margin
    is gating on the point estimate whether it says so or not. The defaults
    below are protocol inputs, not empirically justified values.
    """

    nominal_success_drop: float = 0.02
    tracking_error_increase: float = 0.02
    intervention_rate_increase: float = 0.0
    minimum_target_improvement: float = 0.05
    require_interval_within_budget: bool = True
    required_nominal_groups: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (
            self.nominal_success_drop,
            self.tracking_error_increase,
            self.intervention_rate_increase,
            self.minimum_target_improvement,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("budgets must be finite and nonnegative")


@dataclass(frozen=True)
class RepairVerdict:
    target_improved: bool
    regression_passed: bool
    accepted: bool
    reasons: tuple[str, ...]


def _budget_bound(effect, metric, direction, left, right, budget):
    """The value compared against the budget, and how it was obtained."""
    if not budget.require_interval_within_budget:
        return effect.candidate_minus_baseline, "point_estimate"
    if not effect.degenerate:
        return (effect.ci_low if direction == "lower" else effect.ci_high), effect.interval_method
    if metric in BINARY_METRICS:
        lower, upper = paired_binary_effect_bounds(
            left, right, metric, confidence=effect.confidence
        )
        return (lower if direction == "lower" else upper), "exact_binary_bound"
    raise ValueError(
        "degenerate interval cannot establish budget: every matched scenario delta is "
        f"identical for {metric}, so the bootstrap interval has no width and is not evidence"
    )


def regression_verdict(
    target_improvement: float,
    baseline_nominal: Sequence[Mapping],
    candidate_nominal: Sequence[Mapping],
    budget: RegressionBudget | None = None,
) -> RepairVerdict:
    """Accept only positive target effect and complete matched nominal evidence.

    ``target_improvement`` is oriented so larger is better and must come from
    a frozen counterexample evaluation (e.g. R50 increase or recurrence drop).
    The budget must be frozen before evaluating the candidate. With
    ``require_interval_within_budget`` the bound that could violate each
    budget must lie inside it; a degenerate zero-width interval is never
    accepted as that bound.
    """
    budget = budget or RegressionBudget()
    target = (
        math.isfinite(target_improvement)
        and target_improvement > 0
        and target_improvement >= budget.minimum_target_improvement
    )
    reasons = [] if target else ["target improvement below prespecified minimum"]
    passed = True
    limits = (
        ("success", -budget.nominal_success_drop, "lower"),
        ("tracking_error", budget.tracking_error_increase, "upper"),
        ("intervention_required", budget.intervention_rate_increase, "upper"),
    )
    baseline_groups = {r.get("nominal_group", "all") for r in baseline_nominal}
    candidate_groups = {r.get("nominal_group", "all") for r in candidate_nominal}
    if (
        baseline_groups != candidate_groups
        or not set(budget.required_nominal_groups) <= baseline_groups
    ):
        passed = False
        reasons.append("nominal suite groups missing or mismatched")
    groups = sorted(baseline_groups | candidate_groups) or ["all"]
    for group in groups:
        left = [r for r in baseline_nominal if r.get("nominal_group", "all") == group]
        right = [r for r in candidate_nominal if r.get("nominal_group", "all") == group]
        for metric, limit, direction in limits:
            try:
                effect = paired_comparison(left, right, metric)
                value, method = _budget_bound(effect, metric, direction, left, right, budget)
                violated = value < limit if direction == "lower" else value > limit
                if violated:
                    passed = False
                    reasons.append(
                        f"nominal {group}/{metric} exceeds degradation budget ({method})"
                    )
            except (ValueError, TypeError) as exc:
                passed = False
                reasons.append(f"nominal {group}/{metric}: {exc}")
    return RepairVerdict(target, passed, target and passed, tuple(reasons))
