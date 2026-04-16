"""Evaluation harness — structured comparison of experiment conditions.

Wraps Phoenix's evaluation infrastructure and adds:
- Multi-condition comparison (baseline vs adapted vs controls)
- Statistical significance testing (bootstrap CI, paired t-test)
- Failure recurrence analysis
- Aggregate metric computation
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ashfall.experiment.schema import ExperimentResult

logger = logging.getLogger("ashfall.evaluation.harness")


@dataclass
class MetricComparison:
    """Statistical comparison of a single metric between two conditions."""

    metric: str
    env: str
    condition_a: str
    condition_b: str
    value_a: float
    value_b: float
    delta: float
    relative_delta_pct: float
    significant: bool = False
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass
class ComparisonReport:
    """Full comparison report across conditions and environments."""

    comparisons: list[MetricComparison] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "comparisons": [asdict(c) for c in self.comparisons],
            "summary": self.summary,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compare_conditions(
    results: dict[str, ExperimentResult],
    baseline_key: str = "baseline",
    metrics: list[str] | None = None,
    envs: list[str] | None = None,
) -> ComparisonReport:
    """Compare all conditions against the baseline.

    Parameters
    ----------
    results : dict mapping condition name to ExperimentResult
    baseline_key : which result to use as the reference
    metrics : which metrics to compare (default: success_rate, mean_episode_return)
    envs : which environments to compare across (default: all available)
    """
    if baseline_key not in results:
        raise KeyError(f"Baseline key '{baseline_key}' not found in results")

    if metrics is None:
        metrics = [
            "success_rate",
            "failure_rate",
            "mean_episode_return",
            "mean_episode_length_s",
            "mean_lin_vel_error",
            "mean_ang_vel_error",
        ]

    baseline = results[baseline_key]
    if envs is None:
        envs = list(baseline.metrics.keys())

    report = ComparisonReport()

    for condition_name, result in results.items():
        if condition_name == baseline_key:
            continue

        for env in envs:
            base_metrics = baseline.metrics.get(env, {})
            cond_metrics = result.metrics.get(env, {})

            for metric in metrics:
                val_a = base_metrics.get(metric, 0.0)
                val_b = cond_metrics.get(metric, 0.0)
                delta = val_b - val_a
                rel_delta = (delta / val_a * 100) if val_a != 0 else 0.0

                report.comparisons.append(
                    MetricComparison(
                        metric=metric,
                        env=env,
                        condition_a=baseline_key,
                        condition_b=condition_name,
                        value_a=val_a,
                        value_b=val_b,
                        delta=delta,
                        relative_delta_pct=rel_delta,
                    )
                )

    # Summary
    adapted_results = {k: v for k, v in results.items() if k != baseline_key}
    for name, res in adapted_results.items():
        for env in envs:
            base_sr = baseline.metrics.get(env, {}).get("success_rate", 0.0)
            cond_sr = res.metrics.get(env, {}).get("success_rate", 0.0)
            if cond_sr > base_sr:
                report.summary[f"{name}_{env}"] = (
                    f"{name} improves on {env}: "
                    f"success {base_sr:.1%} -> {cond_sr:.1%} "
                    f"(+{(cond_sr - base_sr):.1%})"
                )
            elif cond_sr < base_sr:
                report.summary[f"{name}_{env}"] = (
                    f"{name} regresses on {env}: "
                    f"success {base_sr:.1%} -> {cond_sr:.1%} "
                    f"({(cond_sr - base_sr):.1%})"
                )

    return report


def bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the difference of means.

    Returns (mean_diff, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n_a = len(values_a)
    n_b = len(values_b)
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample_a = values_a[rng.integers(0, n_a, size=n_a)]
        sample_b = values_b[rng.integers(0, n_b, size=n_b)]
        diffs[i] = np.mean(sample_b) - np.mean(sample_a)

    mean_diff = float(np.mean(values_b) - np.mean(values_a))
    ci_lo = float(np.percentile(diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return mean_diff, ci_lo, ci_hi
