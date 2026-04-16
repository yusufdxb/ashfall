"""Tests for evaluation harness and metrics."""

import numpy as np
import pytest

from ashfall.evaluation.harness import bootstrap_ci, compare_conditions
from ashfall.evaluation.metrics import FailureAnalyzer
from ashfall.experiment.schema import ExperimentResult


class TestCompareConditions:
    def setup_method(self):
        self.baseline = ExperimentResult(
            name="baseline",
            condition="baseline",
            seed=42,
            training_iters=500,
            wall_time_s=100.0,
            metrics={
                "rough": {"success_rate": 1.0, "mean_episode_return": 18.95},
                "slippery": {"success_rate": 0.906, "mean_episode_return": 15.90},
            },
            failure_counts={},
            checkpoint_path="",
        )
        self.adapted = ExperimentResult(
            name="adapted",
            condition="adapted",
            seed=42,
            training_iters=200,
            wall_time_s=50.0,
            metrics={
                "rough": {"success_rate": 0.969, "mean_episode_return": 17.56},
                "slippery": {"success_rate": 1.0, "mean_episode_return": 16.64},
            },
            failure_counts={},
            checkpoint_path="",
        )

    def test_compare_produces_comparisons(self):
        report = compare_conditions(
            {"baseline": self.baseline, "adapted": self.adapted}
        )
        assert len(report.comparisons) > 0

    def test_adapted_improves_on_slippery(self):
        report = compare_conditions(
            {"baseline": self.baseline, "adapted": self.adapted},
            metrics=["success_rate"],
            envs=["slippery"],
        )
        slippery_sr = [
            c for c in report.comparisons if c.metric == "success_rate" and c.env == "slippery"
        ]
        assert len(slippery_sr) == 1
        assert slippery_sr[0].delta > 0  # adapted > baseline

    def test_baseline_key_required(self):
        with pytest.raises(KeyError):
            compare_conditions(
                {"adapted": self.adapted}, baseline_key="baseline"
            )

    def test_report_save_and_load(self, tmp_path):
        report = compare_conditions(
            {"baseline": self.baseline, "adapted": self.adapted}
        )
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()

        import json

        with open(path) as f:
            data = json.load(f)
        assert "comparisons" in data
        assert "summary" in data


class TestBootstrapCI:
    def test_identical_distributions(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_diff, ci_lo, ci_hi = bootstrap_ci(a, b)
        assert abs(mean_diff) < 0.01
        assert ci_lo <= 0 <= ci_hi

    def test_clearly_different_distributions(self):
        a = np.array([1.0, 2.0, 3.0] * 20)
        b = np.array([10.0, 11.0, 12.0] * 20)
        mean_diff, ci_lo, ci_hi = bootstrap_ci(a, b)
        assert mean_diff > 5.0
        assert ci_lo > 0  # clearly positive


class TestFailureAnalyzer:
    def test_no_failures_in_stable_trajectory(self):
        analyzer = FailureAnalyzer()
        for i in range(100):
            analyzer.step(
                timestamp_s=i * 0.02,
                pitch_rad=0.0,
                roll_rad=0.0,
                base_height_m=0.30,
                cmd_lin_vel=np.array([0.5, 0.0]),
                actual_lin_vel=np.array([0.48, 0.01]),
                joint_vel=np.zeros(12),
                contact_forces=np.array([50, 50, 50, 50]),
                done=(i == 99),
            )
        metrics = analyzer.compute()
        assert metrics.total_failures == 0
        assert metrics.total_episodes == 1

    def test_collapse_counted_as_intervention(self):
        analyzer = FailureAnalyzer()
        analyzer.step(
            timestamp_s=0.0,
            pitch_rad=0.0,
            roll_rad=0.0,
            base_height_m=0.05,
            cmd_lin_vel=np.array([0.0, 0.0]),
            actual_lin_vel=np.array([0.0, 0.0]),
            done=True,
        )
        metrics = analyzer.compute()
        assert metrics.total_failures >= 1
        assert metrics.intervention_count >= 1

    def test_multiple_episodes_tracked(self):
        analyzer = FailureAnalyzer()
        for ep in range(3):
            for step in range(10):
                analyzer.step(
                    timestamp_s=ep * 10 + step * 0.02,
                    pitch_rad=0.0,
                    roll_rad=0.0,
                    base_height_m=0.30,
                    cmd_lin_vel=np.array([0.0, 0.0]),
                    actual_lin_vel=np.array([0.0, 0.0]),
                    done=(step == 9),
                )
        metrics = analyzer.compute()
        assert metrics.total_episodes == 3
        assert metrics.total_steps == 30
