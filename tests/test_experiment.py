"""Tests for experiment schema, runner, and sweep."""

import tempfile
from pathlib import Path

import yaml

from ashfall.experiment.runner import ExperimentRunner, load_experiment_config
from ashfall.experiment.schema import (
    AblationAxis,
    Condition,
    EvalSpec,
    ExperimentConfig,
    ExperimentResult,
)
from ashfall.experiment.sweep import (
    ABLATION_NUM_FAILURES,
    SweepConfig,
    generate_sweep,
)


class TestExperimentSchema:
    def test_condition_enum(self):
        assert Condition("baseline") == Condition.BASELINE
        assert Condition("adapted") == Condition.ADAPTED

    def test_experiment_config_defaults(self):
        cfg = ExperimentConfig(name="test", condition=Condition.BASELINE)
        assert cfg.training.num_envs == 4096
        assert cfg.curriculum.failure_fraction == 0.0
        assert cfg.evaluation.num_episodes == 64

    def test_experiment_result_accessors(self):
        result = ExperimentResult(
            name="test",
            condition="baseline",
            seed=42,
            training_iters=500,
            wall_time_s=100.0,
            metrics={
                "rough": {"success_rate": 0.95, "mean_episode_return": 18.5},
                "slippery": {"success_rate": 0.80, "mean_episode_return": 15.0},
            },
            failure_counts={"slip": 3, "attitude": 1},
            checkpoint_path="/tmp/ckpt",
        )
        assert result.success_rate("rough") == 0.95
        assert result.mean_return("slippery") == 15.0
        assert result.success_rate("nonexistent") == 0.0


class TestExperimentRunner:
    def test_prepare_run_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                ashfall_root=Path(tmpdir),
                phoenix_root=Path("/tmp/fake-phoenix"),
            )
            cfg = ExperimentConfig(
                name="test_baseline",
                condition=Condition.BASELINE,
            )
            run_dir = runner.prepare_run(cfg)
            assert run_dir.exists()
            assert (run_dir / "config.yaml").exists()
            assert (run_dir / "commands.sh").exists()
            assert (run_dir / "status.json").exists()
            assert (run_dir / "metrics").is_dir()

    def test_commands_contain_train_for_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                ashfall_root=Path(tmpdir),
                phoenix_root=Path("/tmp/fake-phoenix"),
            )
            cfg = ExperimentConfig(
                name="test_baseline",
                condition=Condition.BASELINE,
            )
            run_dir = runner.prepare_run(cfg)
            commands = (run_dir / "commands.sh").read_text()
            assert "ppo_runner" in commands
            assert "train_baseline" in commands

    def test_commands_contain_adapt_for_adapted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                ashfall_root=Path(tmpdir),
                phoenix_root=Path("/tmp/fake-phoenix"),
            )
            cfg = ExperimentConfig(
                name="test_adapted",
                condition=Condition.ADAPTED,
            )
            run_dir = runner.prepare_run(cfg)
            commands = (run_dir / "commands.sh").read_text()
            assert "fine_tune" in commands
            assert "adapt_with_failures" in commands

    def test_eval_phases_for_all_envs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                ashfall_root=Path(tmpdir),
                phoenix_root=Path("/tmp/fake-phoenix"),
            )
            cfg = ExperimentConfig(
                name="test",
                condition=Condition.BASELINE,
                evaluation=EvalSpec(eval_envs=["rough", "slippery"]),
            )
            run_dir = runner.prepare_run(cfg)
            commands = (run_dir / "commands.sh").read_text()
            assert "eval_rough" in commands
            assert "eval_slippery" in commands


class TestLoadExperimentConfig:
    def test_load_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "test",
                    "condition": "baseline",
                    "training": {"num_envs": 2048},
                },
                f,
            )
            f.flush()
            cfg = load_experiment_config(f.name)
            assert cfg.name == "test"
            assert cfg.condition == Condition.BASELINE
            assert cfg.training.num_envs == 2048


class TestSweep:
    def test_generate_sweep_single_axis(self):
        base = ExperimentConfig(name="base", condition=Condition.ADAPTED)
        sweep = SweepConfig(
            base_experiment=base,
            axes=[ABLATION_NUM_FAILURES],
        )
        configs = generate_sweep(sweep)
        assert len(configs) == len(ABLATION_NUM_FAILURES.values)
        # Each config has a unique name
        names = [c.name for c in configs]
        assert len(set(names)) == len(names)

    def test_generate_sweep_no_axes(self):
        base = ExperimentConfig(name="base", condition=Condition.ADAPTED)
        sweep = SweepConfig(base_experiment=base, axes=[])
        configs = generate_sweep(sweep)
        assert len(configs) == 1
        assert configs[0].name == "base"

    def test_sweep_cell_count(self):
        base = ExperimentConfig(name="base", condition=Condition.ADAPTED)
        axis1 = AblationAxis(name="a", param_path="training.seed", values=[1, 2, 3])
        axis2 = AblationAxis(name="b", param_path="training.num_envs", values=[1024, 2048])
        sweep = SweepConfig(base_experiment=base, axes=[axis1, axis2])
        assert sweep.num_cells == 6
        configs = generate_sweep(sweep)
        assert len(configs) == 6
