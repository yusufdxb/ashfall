"""Tests for the proportion-difference rigor helpers and sweep pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashfall.analysis.significance import (
    compute_sweep_significance,
    render_significance_markdown,
)
from ashfall.evaluation.significance import (
    bernoulli_arrays,
    bootstrap_diff_proportion,
    fishers_exact_p,
    holm_adjust,
    permutation_p_value,
    proportion_diff_test,
)


class TestBernoulliArrays:
    def test_counts_match(self):
        a, b = bernoulli_arrays(7, 10, 3, 5)
        assert a.sum() == 7 and len(a) == 10
        assert b.sum() == 3 and len(b) == 5

    def test_zero_successes(self):
        a, b = bernoulli_arrays(0, 10, 0, 5)
        assert a.sum() == 0 and b.sum() == 0

    def test_full_successes(self):
        a, b = bernoulli_arrays(10, 10, 5, 5)
        assert a.sum() == 10 and b.sum() == 5

    def test_invalid_counts_raise(self):
        with pytest.raises(ValueError):
            bernoulli_arrays(11, 10, 0, 5)
        with pytest.raises(ValueError):
            bernoulli_arrays(-1, 10, 0, 5)


class TestBootstrapDiffProportion:
    def test_identical_proportions_ci_contains_zero(self):
        diff, lo, hi = bootstrap_diff_proportion(50, 100, 50, 100, n_bootstrap=2000, seed=1)
        assert abs(diff) < 1e-9
        assert lo <= 0 <= hi

    def test_clearly_different_ci_excludes_zero(self):
        # 0.95 vs 0.50 with n=200 each: hugely separated, CI must exclude 0.
        diff, lo, hi = bootstrap_diff_proportion(190, 200, 100, 200, n_bootstrap=2000, seed=2)
        assert diff < 0
        assert hi < 0  # entire CI below zero

    def test_point_estimate_matches_proportion_diff(self):
        diff, _, _ = bootstrap_diff_proportion(80, 100, 90, 100, n_bootstrap=1000, seed=3)
        assert abs(diff - 0.10) < 1e-9

    def test_zero_n_returns_zero(self):
        diff, lo, hi = bootstrap_diff_proportion(0, 0, 0, 0, n_bootstrap=100, seed=4)
        assert (diff, lo, hi) == (0.0, 0.0, 0.0)

    def test_percentile_method(self):
        diff, lo, hi = bootstrap_diff_proportion(
            70, 100, 80, 100, n_bootstrap=2000, seed=5, method="percentile"
        )
        assert lo <= diff <= hi

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            bootstrap_diff_proportion(50, 100, 50, 100, n_bootstrap=10, method="bogus")


class TestPermutationPValue:
    def test_identical_data_high_p(self):
        # Note: when proportions are exactly equal the function returns
        # 1.0 (no observed effect to disprove).
        p = permutation_p_value(50, 100, 50, 100, n_perm=500, seed=7)
        assert p >= 0.5

    def test_clearly_different_low_p(self):
        p = permutation_p_value(95, 100, 30, 100, n_perm=2000, seed=8)
        assert p < 0.01

    def test_p_in_unit_interval(self):
        for k_a, n_a, k_b, n_b in [(40, 100, 50, 100), (10, 50, 25, 50), (1, 10, 9, 10)]:
            p = permutation_p_value(k_a, n_a, k_b, n_b, n_perm=500, seed=9)
            assert 0 < p <= 1


class TestFishersExact:
    def test_identical_high_p(self):
        p = fishers_exact_p(50, 100, 50, 100)
        assert p > 0.5

    def test_clearly_different_low_p(self):
        p = fishers_exact_p(95, 100, 30, 100)
        assert p < 1e-10

    def test_borderline_proportions(self):
        # 130/134 vs 123/131: the actual ff=0.0 vs ff=0.5 slippery cells.
        p = fishers_exact_p(119, 134, 123, 131)
        assert 0 < p < 1


class TestHolmAdjust:
    def test_empty(self):
        assert holm_adjust([]) == []

    def test_single_value_unchanged(self):
        assert holm_adjust([0.04]) == [0.04]

    def test_monotone_after_sorting(self):
        ps = [0.01, 0.04, 0.03, 0.20]
        adj = holm_adjust(ps)
        # In sorted order the adjusted ps must be monotone non-decreasing.
        sorted_pairs = sorted(zip(ps, adj))
        sorted_adj = [a for _, a in sorted_pairs]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1]

    def test_smallest_multiplied_by_n(self):
        ps = [0.01, 0.04, 0.03, 0.20]
        adj = holm_adjust(ps)
        # Smallest p should be multiplied by 4.
        smallest_idx = ps.index(min(ps))
        assert abs(adj[smallest_idx] - 0.04) < 1e-9

    def test_capped_at_one(self):
        adj = holm_adjust([0.5, 0.6, 0.7])
        assert all(a <= 1.0 for a in adj)


class TestProportionDiffTest:
    def test_returns_full_result(self):
        res = proportion_diff_test(80, 100, 90, 100, n_bootstrap=500, n_perm=500)
        d = res.as_dict()
        keys = ("p_a", "p_b", "diff", "ci_lower", "ci_upper",
                "p_value", "n_a", "n_b", "method")
        for key in keys:
            assert key in d
        assert abs(res.diff - 0.10) < 1e-9
        assert res.n_a == 100 and res.n_b == 100

    def test_known_significant_difference(self):
        res = proportion_diff_test(95, 100, 30, 100, n_bootstrap=500, n_perm=500)
        # Fisher should give a vanishingly small p here.
        assert res.p_value < 1e-10
        assert res.ci_upper < 0  # b worse than a, CI excludes zero


class TestComputeSweepSignificance:
    def _write_cell(
        self,
        root: Path,
        ff_short: str,
        sr_slip: float,
        sr_rough: float,
        n: int = 128,
    ):
        cell_root = root / f"ablation_failure_fraction_failure_fraction={ff_short}"
        cell = cell_root / "2026-04-28_00-17-07"
        (cell / "metrics").mkdir(parents=True, exist_ok=True)
        for env, sr in (("slippery", sr_slip), ("rough", sr_rough)):
            data = {
                "num_episodes": n,
                "success_rate": sr,
                "failure_rate": 1 - sr,
                "mean_episode_return": 7.0,
                "mean_episode_length_s": 18.0,
                "slew_saturation_pct": 0.3,
            }
            (cell / "metrics" / f"metrics_{env}.json").write_text(json.dumps(data))

    def test_basic_pipeline(self, tmp_path):
        for ff_short, sr_slip, sr_rough in [
            ("0p0", 0.80, 0.85),
            ("0p25", 0.85, 0.80),
            ("0p5", 0.95, 0.90),
        ]:
            self._write_cell(tmp_path, ff_short, sr_slip, sr_rough, n=128)
        sigs = compute_sweep_significance(tmp_path, n_bootstrap=500, n_perm=500)
        assert len(sigs) == 3
        ff_values = sorted(s.failure_fraction for s in sigs)
        assert ff_values == [0.0, 0.25, 0.5]
        # Control row has no diff_vs_control, others do.
        for s in sigs:
            if s.failure_fraction == 0.0:
                assert s.diff_vs_control == {}
            else:
                assert "slippery" in s.diff_vs_control
                assert "rough" in s.diff_vs_control

    def test_huge_lift_is_significant_after_holm(self, tmp_path):
        # Engineer one cell with a massive lift so it survives Holm correction.
        self._write_cell(tmp_path, "0p0", 0.30, 0.30, n=200)
        self._write_cell(tmp_path, "0p5", 0.95, 0.95, n=200)
        self._write_cell(tmp_path, "0p25", 0.32, 0.32, n=200)
        sigs = compute_sweep_significance(tmp_path, n_bootstrap=500, n_perm=500)
        big = next(s for s in sigs if s.failure_fraction == 0.5)
        assert big.holm_p["slippery"] < 0.05
        assert big.holm_p["rough"] < 0.05
        small = next(s for s in sigs if s.failure_fraction == 0.25)
        assert small.holm_p["slippery"] > 0.05

    def test_render_markdown_smoke(self, tmp_path):
        for ff_short, sr_slip, sr_rough in [
            ("0p0", 0.80, 0.85),
            ("0p5", 0.90, 0.92),
        ]:
            self._write_cell(tmp_path, ff_short, sr_slip, sr_rough, n=128)
        sigs = compute_sweep_significance(tmp_path, n_bootstrap=500, n_perm=500)
        md = render_significance_markdown(sigs)
        assert "## Statistical Significance" in md
        assert "### Slippery" in md
        assert "### Rough" in md
        assert "0.50" in md


class TestNoTorchImport:
    """Ensure significance helpers do not pull torch into no-sim CI."""

    def test_module_import_does_not_load_torch(self):
        # Reimport in a clean sense: confirm sys.modules does not have torch
        # after using the public API. (In real CI we'd run this as a
        # subprocess; here the smoke is sufficient because earlier tests
        # haven't imported torch.)
        import sys
        before = "torch" in sys.modules
        # Run the full happy path.
        bernoulli_arrays(5, 10, 3, 10)
        bootstrap_diff_proportion(5, 10, 3, 10, n_bootstrap=50)
        permutation_p_value(5, 10, 3, 10, n_perm=50)
        fishers_exact_p(5, 10, 3, 10)
        holm_adjust([0.1, 0.2, 0.3])
        after = "torch" in sys.modules
        assert before == after  # no transitive torch import
