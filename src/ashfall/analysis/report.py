"""Auto-generate analysis report from experiment results.

Reads all results under ``results/`` and produces a structured markdown
report with tables, plots, and narrative summaries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from ashfall.analysis.tables import comparison_table, failure_taxonomy_table
from ashfall.experiment.schema import ExperimentResult

logger = logging.getLogger("ashfall.analysis.report")


def _load_results(results_dir: Path) -> dict[str, ExperimentResult]:
    """Load all experiment results from a results directory."""
    results = {}
    for config_file in sorted(results_dir.rglob("config.yaml")):
        run_dir = config_file.parent
        metrics_dir = run_dir / "metrics"
        if not metrics_dir.exists():
            continue

        import yaml

        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        metrics: dict[str, dict[str, float]] = {}
        for mf in sorted(metrics_dir.glob("metrics_*.json")):
            env_name = mf.stem.replace("metrics_", "")
            with open(mf) as f:
                metrics[env_name] = json.load(f)

        if not metrics:
            continue

        name = cfg.get("name", run_dir.parent.name)
        results[name] = ExperimentResult(
            name=name,
            condition=cfg.get("condition", "unknown"),
            seed=cfg.get("training", {}).get("seed", 0),
            training_iters=cfg.get("training", {}).get("max_iterations", 0),
            wall_time_s=0.0,
            metrics=metrics,
            failure_counts={},
            checkpoint_path="",
        )

    return results


def generate_report(
    results_dir: str | Path,
    output_path: str | Path = "results/REPORT.md",
    include_plots: bool = True,
) -> Path:
    """Generate a full analysis report.

    Reads experiment results and produces a markdown document with:
    - Experiment summary
    - Failure taxonomy table
    - Condition comparison table
    - Key findings
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_dir)

    sections = []

    # Header
    sections.append("# Ashfall Experiment Report")
    sections.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sections.append(f"\nExperiments found: {len(results)}")

    # Failure taxonomy
    sections.append("\n## Failure Taxonomy")
    sections.append("")
    sections.append(
        "Ashfall classifies quadruped locomotion failures into 6 modes, "
        "ordered by severity:"
    )
    sections.append("")
    sections.append(failure_taxonomy_table())

    # Results comparison
    if results:
        sections.append("\n## Condition Comparison")
        sections.append("")
        sections.append(comparison_table(results))

        # Key findings
        sections.append("\n## Key Findings")
        sections.append("")

        baseline = results.get("baseline")
        adapted = results.get("adapted")
        if baseline and adapted:
            for env in sorted(
                set(baseline.metrics.keys()) & set(adapted.metrics.keys())
            ):
                base_sr = baseline.metrics[env].get("success_rate", 0.0)
                adapt_sr = adapted.metrics[env].get("success_rate", 0.0)
                delta = adapt_sr - base_sr
                direction = "improves" if delta > 0 else "regresses" if delta < 0 else "unchanged"
                sections.append(
                    f"- **{env}**: Adapted {direction} "
                    f"({base_sr:.1%} -> {adapt_sr:.1%}, delta={delta:+.1%})"
                )
    else:
        sections.append("\n## Results")
        sections.append("")
        sections.append(
            "No experiment results found. Run experiments first with "
            "`./scripts/run_experiment.sh`."
        )

    # Plots reference
    if include_plots:
        sections.append("\n## Plots")
        sections.append("")
        plot_files = sorted(results_dir.glob("*.png"))
        if plot_files:
            for pf in plot_files:
                sections.append(f"![{pf.stem}]({pf.name})")
        else:
            sections.append(
                "No plots generated yet. Run `python -m ashfall.analysis.plots` "
                "after experiments complete."
            )

    # Limitations
    sections.append("\n## Limitations")
    sections.append("")
    sections.append(
        "- Synthetic failure trajectories are physics-approximate, not sim-grade\n"
        "- Real-hardware failure collection requires a lab session with GO2\n"
        "- Bootstrap CIs require per-episode metric arrays (not yet collected)\n"
        "- Training curves require TensorBoard log parsing (planned)"
    )

    report = "\n".join(sections)
    output_path.write_text(report)
    logger.info("Report written to %s (%d chars)", output_path, len(report))
    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    generate_report(results)
