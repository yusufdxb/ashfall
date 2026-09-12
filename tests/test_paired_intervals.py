"""Degenerate cluster bootstraps are flagged and never read as proof."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ashfall.evaluation.paired import (
    PairedEffect,
    exact_binary_bounds,
    exact_binary_effect_bounds,
    paired_binary_effect_bounds,
    paired_comparison,
)
from ashfall.evaluation.regression import RegressionBudget, regression_verdict


def records(values, *, metric="success", policy="baseline", clusters=None):
    out = []
    for i, value in enumerate(values):
        scenario = f"s{i if clusters is None else clusters[i]}"
        record = dict(
            scenario_id=scenario,
            scenario_seed=int(scenario[1:]),
            evaluation_seed=i,
            parameter_sample_id=f"p{scenario}",
            policy_id=policy,
            training_seed=1,
            success=True,
            tracking_error=0.1,
            intervention_required=False,
            environment_parameters={"f": 0.5},
        )
        record[metric] = value
        out.append(record)
    return out


class TestDegeneracy:
    @pytest.mark.guard
    def test_identical_cluster_deltas_are_flagged_not_certain(self):
        base = records([True] * 6)
        cand = records([True] * 6, policy="cand")
        effect = paired_comparison(base, cand)
        assert effect.candidate_minus_baseline == 0.0
        assert effect.ci_low == effect.ci_high == 0.0
        assert effect.degenerate
        assert effect.interval_method == "degenerate_all_clusters_identical"
        assert effect.n_bootstrap == 0

    def test_varying_deltas_use_the_cluster_bootstrap(self):
        base = records([0.10, 0.12, 0.11, 0.13, 0.10], metric="tracking_error")
        cand = records([0.11, 0.11, 0.12, 0.12, 0.11], metric="tracking_error", policy="c")
        effect = paired_comparison(base, cand, "tracking_error")
        assert not effect.degenerate and effect.n_bootstrap == 2000
        assert effect.ci_low <= effect.candidate_minus_baseline <= effect.ci_high
        assert effect.ci_low < effect.ci_high

    def test_single_cluster_is_degenerate(self):
        base = records([True, False], clusters=[0, 0])
        cand = records([True, True], clusters=[0, 0], policy="c")
        assert paired_comparison(base, cand).degenerate

    def test_flags_must_agree(self):
        with pytest.raises(ValueError, match="degenerate"):
            PairedEffect("success", 1, 1, 0.0, 0.0, 0.0, 0.95, degenerate=True)
        with pytest.raises(ValueError, match="unknown interval method"):
            PairedEffect("success", 1, 1, 0.0, 0.0, 0.0, 0.95, interval_method="wishful")


class TestExactBounds:
    def test_clopper_pearson_known_values(self):
        lo, hi = exact_binary_bounds(0, 10)
        assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-4)
        lo, hi = exact_binary_bounds(10, 10)
        assert lo == pytest.approx(0.6915, abs=1e-4) and hi == 1.0
        lo, hi = exact_binary_bounds(5, 10)
        assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)
        with pytest.raises(ValueError):
            exact_binary_bounds(11, 10)
        with pytest.raises(ValueError):
            exact_binary_bounds(1, 10, confidence=1.0)

    def test_ceiling_bounds_are_not_zero_width(self):
        n = 32
        lower, upper = exact_binary_effect_bounds([[1.0]] * n, [[1.0]] * n)
        assert lower < 0 < upper
        assert lower == pytest.approx(0.025 ** (1 / n) - 1.0)
        assert upper == pytest.approx(1.0 - 0.025 ** (1 / n))

    @given(st.integers(min_value=2, max_value=400))
    @settings(max_examples=40, deadline=None)
    def test_more_clusters_tighten_the_ceiling_bound(self, n):
        lower_n, _ = exact_binary_effect_bounds([[1.0]] * n, [[1.0]] * n)
        lower_more, _ = exact_binary_effect_bounds([[1.0]] * (n + 1), [[1.0]] * (n + 1))
        assert -1.0 <= lower_n <= lower_more <= 0.0

    def test_bounds_bracket_the_point_estimate(self):
        base = [[1, 1], [1, 0], [0, 0], [1, 1], [1, 1]]
        cand = [[1, 1], [1, 1], [1, 0], [1, 1], [0, 1]]
        lower, upper = exact_binary_effect_bounds(base, cand)
        point = sum(sum(c) / len(c) for c in cand) / 5 - sum(sum(b) / len(b) for b in base) / 5
        assert lower <= point <= upper

    def test_binary_values_required(self):
        with pytest.raises(ValueError, match="0/1"):
            exact_binary_effect_bounds([[0.5]], [[1.0]])
        with pytest.raises(ValueError, match="equal positive length"):
            exact_binary_effect_bounds([[1.0]], [])

    def test_paired_records_route(self):
        base = records([True] * 4)
        cand = records([True] * 4, policy="c")
        lower, upper = paired_binary_effect_bounds(base, cand)
        assert lower == pytest.approx(0.025 ** (1 / 4) - 1.0)


class TestRegressionGate:
    @pytest.mark.guard
    def test_zero_width_interval_at_ceiling_is_refused(self):
        """Five identical all-success clusters used to pass every budget."""
        base = records([True] * 5)
        cand = records([True] * 5, policy="c")
        verdict = regression_verdict(0.2, base, cand)
        assert verdict.target_improved and not verdict.regression_passed and not verdict.accepted
        reasons = " ".join(verdict.reasons)
        assert "success exceeds degradation budget (exact_binary_bound)" in reasons
        assert "degenerate interval cannot establish budget" in reasons  # tracking_error

    def test_point_estimate_gating_is_an_explicit_choice(self):
        base = records([True] * 5)
        cand = records([True] * 5, policy="c")
        budget = RegressionBudget(require_interval_within_budget=False)
        assert regression_verdict(0.2, base, cand, budget).accepted

    def test_exact_bound_can_be_met_with_enough_clusters(self):
        n = 200
        base = records([True] * n)
        cand = records([True] * n, policy="c")
        for i, (b, c) in enumerate(zip(base, cand)):
            b["tracking_error"] = 0.10 + 0.002 * ((i % 5) - 2)
            c["tracking_error"] = b["tracking_error"] + (0.001 if i % 2 else -0.001)
        budget = RegressionBudget(intervention_rate_increase=0.02)
        verdict = regression_verdict(0.2, base, cand, budget)
        assert verdict.regression_passed, verdict.reasons
        # A zero intervention margin cannot be established by any interval.
        assert not regression_verdict(0.2, base, cand).regression_passed

    def test_continuous_degenerate_fails_closed(self):
        base = records([True] * 5)
        cand = records([True] * 5, policy="c")
        for b, c in zip(base, cand):
            b["tracking_error"] = c["tracking_error"] = 0.05
        verdict = regression_verdict(
            0.2, base, cand, RegressionBudget(intervention_rate_increase=1)
        )
        assert any("degenerate interval cannot establish budget" in r for r in verdict.reasons)
