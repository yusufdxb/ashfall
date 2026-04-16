"""Ablation sweep runner.

Generates a matrix of experiment configs by varying one or more axes:
- Number of failure trajectories used in curriculum
- Types of failure included (single-mode vs all-mode)
- Domain randomization range
- Training iterations for adaptation

Each cell in the sweep matrix is a standalone ExperimentConfig that can be
prepared and run independently.
"""

from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass
from typing import Any

from ashfall.experiment.schema import AblationAxis, ExperimentConfig

logger = logging.getLogger("ashfall.experiment.sweep")


@dataclass
class SweepConfig:
    """Defines a sweep over ablation axes."""

    base_experiment: ExperimentConfig
    axes: list[AblationAxis]
    name_template: str = "{base}_{axis}_{value}"

    @property
    def num_cells(self) -> int:
        if not self.axes:
            return 1
        return len(list(itertools.product(*(a.values for a in self.axes))))


def _set_nested(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute via dot-separated path."""
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def generate_sweep(sweep: SweepConfig) -> list[ExperimentConfig]:
    """Expand a sweep config into individual experiment configs."""
    if not sweep.axes:
        return [sweep.base_experiment]

    value_lists = [a.values for a in sweep.axes]
    configs = []

    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(sweep.base_experiment)
        name_parts = []
        ablation_values = {}

        for axis, value in zip(sweep.axes, combo):
            _set_nested(cfg, axis.param_path, value)
            short_val = str(value).replace(".", "p").replace("-", "n")
            name_parts.append(f"{axis.name}={short_val}")
            ablation_values[axis.name] = value

        cfg.name = f"{sweep.base_experiment.name}_{'_'.join(name_parts)}"
        cfg.tags = list(sweep.base_experiment.tags) + ["ablation"]
        configs.append(cfg)

    logger.info("Generated %d sweep cells from %d axes", len(configs), len(sweep.axes))
    return configs


# Pre-defined ablation axes for common sweeps
ABLATION_NUM_FAILURES = AblationAxis(
    name="n_failures",
    param_path="curriculum.failure_fraction",
    values=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
)

ABLATION_FAILURE_MODES = AblationAxis(
    name="failure_modes",
    param_path="curriculum.failure_modes",
    values=[
        ["slip"],
        ["attitude"],
        ["collapse"],
        ["slip", "attitude"],
        ["slip", "attitude", "collapse"],
        ["slip", "attitude", "collapse", "stumble", "contact_loss", "command_mismatch"],
    ],
)

ABLATION_TRAIN_ITERS = AblationAxis(
    name="adapt_iters",
    param_path="training.max_iterations",
    values=[50, 100, 200, 400],
)
