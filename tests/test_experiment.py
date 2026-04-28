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
            tmp = Path(tmpdir)
            phoenix_root = tmp / "fake-phoenix"
            (phoenix_root / "configs" / "train").mkdir(parents=True)
            (phoenix_root / "configs" / "train" / "adaptation.yaml").write_text(
                "run: {name: phoenix-adapt}\n"
                "resume: {path: checkpoints/phoenix-base/latest.pt}\n"
                "env: {config: configs/env/slippery.yaml}\n"
                "curriculum: {failure_sample_fraction: 0.0, trajectory_dir: data/failures}\n"
            )
            runner = ExperimentRunner(
                ashfall_root=tmp / "ashfall",
                phoenix_root=phoenix_root,
            )
            cfg = ExperimentConfig(
                name="test_adapted",
                condition=Condition.ADAPTED,
            )
            cfg.training.adapt_config = "configs/train/adaptation.yaml"
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

    def test_sweep_injects_distinct_failure_fractions(self, tmp_path):
        """Regression: each sweep cell must produce its own adapt YAML
        whose ``failure_sample_fraction`` matches that cell's value.

        Previously the runner passed a static ``--config configs/train/adaptation.yaml``
        for every cell, so a 6-cell ``failure_fraction`` sweep collapsed
        into 6 identical training runs.
        """
        import shutil

        # Stub Phoenix repo with a minimal adaptation template the runner
        # can read and override.
        phoenix_root = tmp_path / "fake-phoenix"
        (phoenix_root / "configs" / "train").mkdir(parents=True)
        template = phoenix_root / "configs" / "train" / "adaptation.yaml"
        template.write_text(
            "run:\n  name: phoenix-adapt\n"
            "resume:\n  path: checkpoints/phoenix-base/latest.pt\n"
            "env:\n  config: configs/env/slippery.yaml\n"
            "curriculum:\n  failure_sample_fraction: 0.0\n"
            "  trajectory_dir: data/failures\n"
        )

        ashfall_root = tmp_path / "ashfall"
        (ashfall_root / "data" / "failures").mkdir(parents=True)

        base = ExperimentConfig(
            name="ff_sweep",
            condition=Condition.ADAPTED,
        )
        base.training.adapt_config = "configs/train/adaptation.yaml"

        axis = AblationAxis(
            name="failure_fraction",
            param_path="curriculum.failure_fraction",
            values=[0.0, 0.25, 0.75],
        )
        sweep = SweepConfig(base_experiment=base, axes=[axis])
        configs = generate_sweep(sweep)
        assert len(configs) == 3

        runner = ExperimentRunner(
            ashfall_root=ashfall_root,
            phoenix_root=phoenix_root,
        )

        observed = []
        for cell_cfg in configs:
            run_dir = runner.prepare_run(cell_cfg)
            commands = (run_dir / "commands.sh").read_text()
            # Each cell must reference its own generated adapt YAML.
            assert "_generated/adapt_" in commands, (
                "Runner is still using the static template — failure_fraction "
                "is being ignored."
            )
            override_path = run_dir / "adapt_override.yaml"
            assert override_path.exists()
            with open(override_path) as f:
                override = yaml.safe_load(f)
            observed.append(override["curriculum"]["failure_sample_fraction"])

        assert observed == [0.0, 0.25, 0.75], (
            f"Expected per-cell failure fractions [0.0, 0.25, 0.75]; got {observed}"
        )
        # Three distinct values prove each cell got its own override.
        assert len(set(observed)) == 3

        shutil.rmtree(phoenix_root, ignore_errors=True)
        shutil.rmtree(ashfall_root, ignore_errors=True)

    def test_sweep_cell_count(self):
        base = ExperimentConfig(name="base", condition=Condition.ADAPTED)
        axis1 = AblationAxis(name="a", param_path="training.seed", values=[1, 2, 3])
        axis2 = AblationAxis(name="b", param_path="training.num_envs", values=[1024, 2048])
        sweep = SweepConfig(base_experiment=base, axes=[axis1, axis2])
        assert sweep.num_cells == 6
        configs = generate_sweep(sweep)
        assert len(configs) == 6
