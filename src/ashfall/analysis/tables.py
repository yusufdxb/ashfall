"""Table generation for experiment results.

Produces markdown and plain-text tables for:
- Condition comparison matrix
- Failure taxonomy summary
- Ablation results
"""

from __future__ import annotations

from typing import Any

from ashfall.experiment.schema import ExperimentResult
from ashfall.taxonomy.schema import taxonomy_table_rows


def _format_val(v: float, metric: str) -> str:
    if "rate" in metric:
        return f"{v:.1%}"
    if "error" in metric:
        return f"{v:.3f}"
    return f"{v:.2f}"


def comparison_table(
    results: dict[str, ExperimentResult],
    metrics: list[str] | None = None,
    envs: list[str] | None = None,
) -> str:
    """Generate a markdown table comparing conditions.

    Returns a string like:
    | Condition | Env | Success Rate | Mean Return | Failure Rate |
    |-----------|-----|-------------|-------------|--------------|
    """
    if metrics is None:
        metrics = ["success_rate", "mean_episode_return", "failure_rate"]

    all_envs: set[str] = set()
    for r in results.values():
        all_envs.update(r.metrics.keys())
    if envs is None:
        envs = sorted(all_envs)

    header_metrics = [m.replace("_", " ").title() for m in metrics]
    header = "| Condition | Env | " + " | ".join(header_metrics) + " |"
    sep = "|" + "|".join("-" * (len(h) + 2) for h in ["Condition", "Env"] + header_metrics) + "|"

    rows = [header, sep]
    for cond_name, result in sorted(results.items()):
        for env in envs:
            env_metrics = result.metrics.get(env, {})
            vals = [_format_val(env_metrics.get(m, 0.0), m) for m in metrics]
            rows.append(f"| {cond_name} | {env} | " + " | ".join(vals) + " |")

    return "\n".join(rows)


def failure_taxonomy_table() -> str:
    """Generate a markdown table of the failure taxonomy."""
    rows = taxonomy_table_rows()
    header = "| Mode | Severity | Detection | Sim Replay Strategy |"
    sep = "|------|----------|-----------|---------------------|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['Mode']} | {r['Severity']} | {r['Detection']} | {r['Sim Replay']} |")
    return "\n".join(lines)


def ablation_table(
    sweep_results: list[dict[str, Any]],
    axis_name: str,
    metric: str = "success_rate",
    env: str = "slippery",
) -> str:
    """Generate a markdown table for ablation sweep results."""
    header = f"| {axis_name} | {metric.replace('_', ' ').title()} | Delta vs Base |"
    sep = "|" + "|".join("---" for _ in range(3)) + "|"
    lines = [header, sep]

    sorted_results = sorted(
        sweep_results, key=lambda r: r["ablation_values"].get(axis_name, 0)
    )

    base_val = None
    for r in sorted_results:
        val = r["metrics"].get(env, {}).get(metric, 0.0)
        axis_val = r["ablation_values"].get(axis_name, "?")
        if base_val is None:
            base_val = val
            delta = "-"
        else:
            d = val - base_val
            delta = f"+{d:.2f}" if d >= 0 else f"{d:.2f}"
        lines.append(f"| {axis_val} | {_format_val(val, metric)} | {delta} |")

    return "\n".join(lines)
