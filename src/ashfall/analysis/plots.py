"""Visualization pipeline for Ashfall experiment results.

Generates publication-quality plots:
- Condition comparison bar charts
- Failure mode distribution
- Ablation sweep heatmaps
- Training curves (from TensorBoard logs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("ashfall.analysis.plots")


def _import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    return plt


def plot_condition_comparison(
    results: dict[str, dict[str, dict[str, float]]],
    metric: str = "success_rate",
    output_path: str | Path = "results/comparison.png",
    title: str | None = None,
) -> Path:
    """Bar chart comparing a metric across conditions and environments.

    Parameters
    ----------
    results : {condition_name: {env_name: {metric_name: value}}}
    """
    plt = _import_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = list(results.keys())
    envs = sorted(set(e for cond in results.values() for e in cond.keys()))
    n_envs = len(envs)
    n_conds = len(conditions)

    x = np.arange(n_envs)
    width = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=(max(8, n_envs * 2), 5))

    colors = _physx_style.cmap_cycle(max(n_conds, 3))
    for i, cond in enumerate(conditions):
        values = [results[cond].get(env, {}).get(metric, 0.0) for env in envs]
        bars = ax.bar(x + i * width, values, width, label=cond, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Environment")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by Condition")
    ax.set_xticks(x + width * (n_conds - 1) / 2)
    ax.set_xticklabels(envs)
    ax.legend()
    upper = max(ax.get_ylim()[1] * 1.15, 0.1)
    if "rate" in metric:
        upper = min(upper, 1.05)
    ax.set_ylim(0, upper)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot to %s", output_path)
    return output_path


def plot_failure_distribution(
    failure_counts: dict[str, dict[str, int]],
    output_path: str | Path = "results/failure_distribution.png",
) -> Path:
    """Stacked bar chart of failure mode counts per condition."""
    plt = _import_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = list(failure_counts.keys())
    all_modes = sorted(set(m for fc in failure_counts.values() for m in fc.keys()))

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.5), 5))

    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    colors = _physx_style.cmap_cycle(max(len(all_modes), 1))

    for i, mode in enumerate(all_modes):
        values = [failure_counts[c].get(mode, 0) for c in conditions]
        ax.bar(x, values, bottom=bottoms, label=mode, color=colors[i])
        bottoms += np.array(values)

    ax.set_xlabel("Condition")
    ax.set_ylabel("Failure Count")
    ax.set_title("Failure Distribution by Mode and Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved failure distribution plot to %s", output_path)
    return output_path


def plot_ablation_sweep(
    sweep_results: list[dict[str, Any]],
    x_axis: str,
    y_metric: str = "success_rate",
    env: str = "slippery",
    output_path: str | Path = "results/ablation_sweep.png",
) -> Path:
    """Line plot of a metric across ablation axis values."""
    plt = _import_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_vals = [r["ablation_values"].get(x_axis, 0) for r in sweep_results]
    y_vals = [r["metrics"].get(env, {}).get(y_metric, 0.0) for r in sweep_results]

    # Sort by x
    pairs = sorted(zip(x_vals, y_vals))
    x_sorted = [p[0] for p in pairs]
    y_sorted = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_sorted, y_sorted, "o-", linewidth=2, markersize=8, color=_physx_style.COLORS["physx"])
    ax.fill_between(x_sorted, y_sorted, alpha=0.1, color=_physx_style.COLORS["physx"])

    for xi, yi in zip(x_sorted, y_sorted):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel(x_axis.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.set_title(f"Ablation: {y_metric} vs {x_axis} on {env}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation plot to %s", output_path)
    return output_path
