"""Experiment configuration schema.

An Ashfall experiment defines:
- A *condition* (baseline, adapted, control_random, control_noreplay)
- Training parameters (inherited from Phoenix configs)
- Failure curriculum settings
- Evaluation parameters
- Ablation axes (optional)

Experiments are serialized as YAML and produce structured JSON results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Condition(str, Enum):
    """Experimental condition for comparison."""

    BASELINE = "baseline"
    ADAPTED = "adapted"
    CONTROL_RANDOM = "control_random"
    CONTROL_NOREPLAY = "control_noreplay"


@dataclass
class TrainingSpec:
    env_config: str = "configs/env/rough.yaml"
    train_config: str | None = None  # path to Phoenix train YAML
    adapt_config: str | None = None  # path to Phoenix adaptation YAML
    num_envs: int = 4096
    max_iterations: int = 500
    device: str = "cuda:0"
    seed: int = 42


@dataclass
class CurriculumSpec:
    failure_dir: str = "data/failures"
    failure_fraction: float = 0.0
    failure_modes: list[str] = field(default_factory=list)  # empty = all modes
    num_variations: int = 16


@dataclass
class EvalSpec:
    num_episodes: int = 64
    num_envs: int = 32
    eval_envs: list[str] = field(default_factory=lambda: ["rough", "slippery", "flat"])
    record_video: bool = True


@dataclass
class AblationAxis:
    name: str
    param_path: str  # dot-separated path into the config
    values: list[Any] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Top-level experiment definition."""

    name: str
    condition: Condition
    description: str = ""
    training: TrainingSpec = field(default_factory=TrainingSpec)
    curriculum: CurriculumSpec = field(default_factory=CurriculumSpec)
    evaluation: EvalSpec = field(default_factory=EvalSpec)
    ablations: list[AblationAxis] = field(default_factory=list)
    phoenix_root: str = "../../workspace/go2-phoenix"  # relative path to Phoenix repo
    tags: list[str] = field(default_factory=list)

    @property
    def phoenix_path(self) -> Path:
        return Path(self.phoenix_root).expanduser().resolve()


@dataclass
class ExperimentResult:
    """Structured output from a single experiment run."""

    name: str
    condition: str
    seed: int
    training_iters: int
    wall_time_s: float
    metrics: dict[str, dict[str, float]]  # env_name -> metric_name -> value
    failure_counts: dict[str, int]  # failure_mode -> count
    checkpoint_path: str
    ablation_values: dict[str, Any] = field(default_factory=dict)

    def success_rate(self, env: str = "rough") -> float:
        return self.metrics.get(env, {}).get("success_rate", 0.0)

    def mean_return(self, env: str = "rough") -> float:
        return self.metrics.get(env, {}).get("mean_episode_return", 0.0)
