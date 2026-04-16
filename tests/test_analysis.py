"""Tests for analysis pipeline (tables, report)."""



from ashfall.analysis.tables import (
    ablation_table,
    comparison_table,
    failure_taxonomy_table,
)
from ashfall.experiment.schema import ExperimentResult


class TestComparisonTable:
    def test_produces_markdown(self):
        results = {
            "baseline": ExperimentResult(
                name="baseline",
                condition="baseline",
                seed=42,
                training_iters=500,
                wall_time_s=100.0,
                metrics={"rough": {"success_rate": 1.0, "mean_episode_return": 18.95}},
                failure_counts={},
                checkpoint_path="",
            ),
            "adapted": ExperimentResult(
                name="adapted",
                condition="adapted",
                seed=42,
                training_iters=200,
                wall_time_s=50.0,
                metrics={"rough": {"success_rate": 0.969, "mean_episode_return": 17.56}},
                failure_counts={},
                checkpoint_path="",
            ),
        }
        table = comparison_table(results)
        assert "| Condition |" in table
        assert "baseline" in table
        assert "adapted" in table
        assert "100.0%" in table


class TestFailureTaxonomyTable:
    def test_produces_markdown(self):
        table = failure_taxonomy_table()
        assert "| Mode |" in table
        assert "Body Collapse" in table
        assert "Attitude Loss" in table
        assert "Foot Slip" in table
        assert "Stumble" in table
        assert "Contact Loss" in table
        assert "Command Mismatch" in table


class TestAblationTable:
    def test_produces_rows(self):
        sweep_results = [
            {
                "ablation_values": {"failure_fraction": 0.0},
                "metrics": {"slippery": {"success_rate": 0.90}},
            },
            {
                "ablation_values": {"failure_fraction": 0.25},
                "metrics": {"slippery": {"success_rate": 0.95}},
            },
            {
                "ablation_values": {"failure_fraction": 0.5},
                "metrics": {"slippery": {"success_rate": 0.97}},
            },
        ]
        table = ablation_table(sweep_results, "failure_fraction")
        assert "failure_fraction" in table
        assert "90.0%" in table
