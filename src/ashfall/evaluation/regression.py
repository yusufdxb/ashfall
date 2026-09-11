"""Explicit target improvement and nominal degradation budgets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ashfall.evaluation.paired import paired_comparison


@dataclass(frozen=True)
class RegressionBudget:
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


def regression_verdict(
    target_improvement: float,
    baseline_nominal: Sequence[Mapping],
    candidate_nominal: Sequence[Mapping],
    budget: RegressionBudget | None = None,
) -> RepairVerdict:
    """Accept only positive target effect and complete matched nominal evidence.

    ``target_improvement`` is oriented so larger is better and must come from
    a frozen counterexample evaluation (e.g. R50 increase or recurrence drop).
    The budget must be frozen before evaluating the candidate.
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
                value = effect.candidate_minus_baseline
                if budget.require_interval_within_budget:
                    value = effect.ci_low if direction == "lower" else effect.ci_high
                violated = value < limit if direction == "lower" else value > limit
                if violated:
                    passed = False
                    reasons.append(f"nominal {group}/{metric} exceeds degradation budget")
            except (ValueError, TypeError) as exc:
                passed = False
                reasons.append(f"nominal {group}/{metric}: {exc}")
    return RepairVerdict(target, passed, target and passed, tuple(reasons))
