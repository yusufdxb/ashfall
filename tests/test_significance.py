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
    bca_acceleration,
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
        keys = ("p_a", "p_b", "diff", "ci_lower", "ci_upper", "p_value", "n_a", "n_b", "method")
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
        # Must run in a subprocess. In-process the assertion is vacuous,
        # because tests/test_phoenix_backend.py imports torch at collection
        # time, so `before` and `after` are both True and nothing is checked.
        import subprocess
        import sys

        source = (
            "import sys\n"
            "from ashfall.evaluation.significance import (bernoulli_arrays,\n"
            "    bootstrap_diff_proportion, permutation_p_value, fishers_exact_p,\n"
            "    holm_adjust)\n"
            "bernoulli_arrays(5, 10, 3, 10)\n"
            "bootstrap_diff_proportion(5, 10, 3, 10, n_bootstrap=50)\n"
            "permutation_p_value(5, 10, 3, 10, n_perm=50)\n"
            "fishers_exact_p(5, 10, 3, 10)\n"
            "holm_adjust([0.1, 0.2, 0.3])\n"
            "assert 'torch' not in sys.modules, 'significance pulled in torch'\n"
        )
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


class TestBcaAccelerationSign:
    """The acceleration term is an odd power of the influence values, so an
    inverted sign silently moves an interval across zero. It was inverted."""

    @pytest.mark.parametrize(
        "k_a,n_a,k_b,n_b",
        [
            (118, 130, 124, 128),  # near ceiling, the regime this repo runs in
            (65, 130, 70, 128),  # symmetric
            (10, 130, 120, 128),  # strongly skewed the other way
        ],
    )
    def test_acceleration_matches_scipy_in_sign_and_magnitude(self, k_a, n_a, k_b, n_b):
        """Compare against scipy's own BCa acceleration, not against ourselves.

        scipy._resampling._bca_interval returns (alpha_1, alpha_2, a_hat) and
        uses U_ji = (n - 1) * (theta_dot - theta_i), the Efron and Tibshirani
        convention. Our pooled-influence shortcut should agree to a few
        significant figures and, critically, in sign.
        """
        import numpy as np
        from scipy.stats._resampling import _bca_interval

        a, b = bernoulli_arrays(k_a, n_a, k_b, n_b)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ours = bca_acceleration(a, b)  # the module's own function, not a copy

        def statistic(x, y, axis=-1):
            return y.mean(axis=axis) - x.mean(axis=axis)

        rng = np.random.default_rng(0)
        theta_hat_b = np.array(
            [
                statistic(a[rng.integers(0, n_a, n_a)], b[rng.integers(0, n_b, n_b)])
                for _ in range(2000)
            ]
        )
        # scipy >= 1.16 added a required array-namespace argument to this private
        # helper; older scipy (the only one Python 3.10 resolves) does not take it.
        import inspect

        kwargs = dict(axis=-1, alpha=0.025, theta_hat_b=theta_hat_b, batch=None)
        if "xp" in inspect.signature(_bca_interval).parameters:
            from scipy._lib._array_api import array_namespace

            kwargs["xp"] = array_namespace(a, b)
        *_, a_hat = _bca_interval((a, b), statistic, **kwargs)
        a_hat = float(np.ravel(a_hat)[0])

        assert np.sign(ours) == np.sign(
            a_hat
        ), f"acceleration sign disagrees with scipy: ours={ours}, scipy={a_hat}"
        assert ours == pytest.approx(a_hat, rel=0.05)

    def test_near_ceiling_interval_is_not_dragged_across_zero(self):
        # With the sign inverted this configuration produced ci_low < 0 while
        # the correct BCa excludes zero. Guard the direction, not the digits.
        _point, low, high = bootstrap_diff_proportion(118, 130, 124, 128, n_bootstrap=4000, seed=0)
        assert low < high
        # Measured: the inverted sign gives [-0.000601, +0.115264] (includes
        # zero); the correct sign gives [+0.006731, +0.122716] (excludes it).
        # The inferential difference is whether zero is inside, so guard that.
        assert low > 0.0, (
            "acceleration sign regression: the near-ceiling interval now "
            f"includes zero (low={low})"
        )


class TestNormPpfFallback:
    """The scipy-free fallback is only reachable if scipy becomes optional,
    but its lower tail returned -134 for q=0.005 before this guard existed."""

    @pytest.mark.parametrize(
        "q", [1e-4, 1e-3, 0.005, 0.01, 0.02, 0.024, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999]
    )
    def test_fallback_matches_scipy_across_both_tails(self, q, monkeypatch):
        from scipy.stats import norm

        from ashfall.evaluation import significance as sig

        monkeypatch.setattr(sig, "_HAS_SCIPY", False)
        assert sig._norm_ppf(q) == pytest.approx(float(norm.ppf(q)), abs=1e-6)
