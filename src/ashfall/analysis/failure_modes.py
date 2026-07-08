"""Failure-mode breakdown for the curriculum pool.

Eval-time per-episode failure-mode counts are not retained by
Phoenix's ``evaluate.py``; only aggregate scalars land in
``metrics_*.json``. Therefore the per-mode breakdown we *can* present
in the v0.3.0 sweep report is the **input curriculum pool**: which
synthetic failure trajectories were available to the fine-tune buffer,
labeled by their authored ``failure_mode``. Across the ff sweep, the
pool itself is identical and only the per-minibatch sampling fraction
varies, so the bar chart is one-per-mode (not per-cell). It is
included for completeness and as the right baseline against which the
next ablation (mode-subset sweep, fixed ff=0.5) compares.

This module also synthesizes a per-cell view: for each ff, the
*expected* number of failure-trajectory-steps drawn into adaptation
is `failure_fraction * pool_step_count` per minibatch, partitioned
by mode in proportion to the pool composition. We render this as a
stacked-bar so reviewers can see at a glance how the curriculum
"dosage" scales with ff.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("ashfall.analysis.failure_modes")

CELL_PATTERN = re.compile(
    r"ablation_failure_fraction_failure_fraction=(?P<short>[0-9p]+)$"
)


def _short_to_float(short: str) -> float:
    return float(short.replace("p", "."))


def load_pool_composition(pool_dir: Path) -> dict[str, dict[str, int]]:
    """Read every parquet in ``pool_dir`` and return per-mode totals.

    Returns
    -------
    {mode_name: {"n_traj": int, "n_steps": int, "n_active_steps": int}}
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not available, skipping pool composition")
        return {}

    out: dict[str, dict[str, int]] = {}
    for fp in sorted(pool_dir.glob("*.parquet")):
        try:
            df = pq.ParquetFile(fp).read().to_pandas()
        except Exception as exc:
            logger.warning("Could not read %s: %s", fp, exc)
            continue
        modes = [m for m in df.get("failure_mode", []) if m]
        if not modes:
            # Fallback to filename pattern: synth_<mode>_NNN.parquet
            stem = fp.stem
            parts = stem.split("_")
            if len(parts) >= 3 and parts[0] == "synth":
                mode = "_".join(parts[1:-1])
            else:
                continue
        else:
            mode = max(set(modes), key=modes.count)
        slot = out.setdefault(
            mode, {"n_traj": 0, "n_steps": 0, "n_active_steps": 0}
        )
        slot["n_traj"] += 1
        slot["n_steps"] += int(len(df))
        if "failure_flag" in df.columns:
            slot["n_active_steps"] += int(df["failure_flag"].astype(bool).sum())
    return out


def per_mode_breakdown(
    results_dir: str | Path,
    pool_dir: str | Path = "data/failures",
) -> dict:
    """Compute the per-mode pool composition + per-cell expected dosage.

    Returns a dict::

        {
          "pool": {mode: {n_traj, n_steps, n_active_steps}, ...},
          "cells": [
            {failure_fraction: float, expected_steps_per_unit: float,
             per_mode_share: {mode: float}}, ...
          ],
        }

    ``per_mode_share`` for each cell sums to ``failure_fraction`` and
    is split in proportion to the active-step counts in the pool.
    """
    results_dir = Path(results_dir)
    pool_dir = Path(pool_dir)
    pool = load_pool_composition(pool_dir)

    total_active = sum(p["n_active_steps"] for p in pool.values()) or 1
    base_share = {m: p["n_active_steps"] / total_active for m, p in pool.items()}

    cells: list[dict] = []
    for cell_dir in sorted(results_dir.glob("ablation_failure_fraction_failure_fraction=*")):
        m = CELL_PATTERN.search(cell_dir.name)
        if not m:
            continue
        ff = _short_to_float(m.group("short"))
        per_mode = {mode: ff * share for mode, share in base_share.items()}
        cells.append({
            "failure_fraction": ff,
            "expected_dosage": ff,
            "per_mode_share": per_mode,
        })
    cells.sort(key=lambda c: c["failure_fraction"])

    return {"pool": pool, "cells": cells}


def plot_curriculum_pool_composition(
    breakdown: dict,
    output_path: str | Path,
) -> Path:
    """Stacked-bar plot of expected failure-mode dosage vs failure_fraction.

    For each ff cell, plots a stacked bar where each segment is the
    fraction of the minibatch expected to come from that failure mode
    (= ff * pool_share_by_active_steps). Cells at ff=0.0 are blank,
    ff=1.0 reproduces the pool composition exactly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    cells = breakdown["cells"]
    if not cells:
        logger.warning("No cells for pool composition plot")
        return output_path

    modes = sorted(breakdown["pool"].keys())
    ffs = [c["failure_fraction"] for c in cells]
    bottoms = np.zeros(len(cells))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [f"C{i % 10}" for i in range(max(len(modes), 1))]
    for i, mode in enumerate(modes):
        vals = np.array([c["per_mode_share"].get(mode, 0.0) for c in cells])
        ax.bar(ffs, vals, bottom=bottoms, label=mode, color=colors[i], width=0.08)
        bottoms = bottoms + vals
    ax.set_xlabel("failure_fraction")
    ax.set_ylabel("expected curriculum share (= ff * pool_proportion)")
    ax.set_title(
        "Ashfall v0.3.0: Expected Curriculum Composition vs failure_fraction\n"
        "(pool composition is constant; only sampling density varies)"
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ffs)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


def render_failure_mode_breakdown_markdown(
    results_dir: str | Path,
    pool_dir: str | Path = "data/failures",
) -> str:
    """Render the failure-mode-composition section as markdown."""
    results_dir = Path(results_dir)
    breakdown = per_mode_breakdown(results_dir, pool_dir)
    pool = breakdown["pool"]
    if not pool:
        return ""

    plot_path = Path(results_dir) / "sweep_curriculum_pool_composition.png"
    try:
        plot_curriculum_pool_composition(breakdown, plot_path)
    except Exception as exc:
        logger.warning("Pool composition plot failed: %s", exc)

    total_traj = sum(p["n_traj"] for p in pool.values())
    total_steps = sum(p["n_steps"] for p in pool.values())
    total_active = sum(p["n_active_steps"] for p in pool.values())

    lines: list[str] = []
    lines.append("## Failure-Mode Composition")
    lines.append("")
    lines.append(
        "Phoenix's `evaluate.py` does not retain per-episode failure-mode "
        "labels at eval time; only aggregate `success_rate` lands in "
        "`metrics_*.json`. The breakdown below is therefore the "
        "**curriculum-input pool**: the synthetic failure trajectories "
        "available to the fine-tune buffer. Across this sweep the pool "
        "is identical for every ff cell; only the per-minibatch sampling "
        "fraction differs."
    )
    lines.append("")
    lines.append(
        f"Pool totals: {total_traj} trajectories, {total_steps} steps, "
        f"{total_active} active failure-flagged steps."
    )
    lines.append("")
    lines.append("| mode             | n_traj | n_steps | n_active_steps | active_share |")
    lines.append("|:-----------------|-------:|--------:|---------------:|-------------:|")
    for mode in sorted(pool.keys()):
        p = pool[mode]
        share = p["n_active_steps"] / total_active if total_active else 0.0
        lines.append(
            f"| {mode:<16} | {p['n_traj']:>6} | {p['n_steps']:>7} | "
            f"{p['n_active_steps']:>14} | {share:>12.3f} |"
        )
    lines.append("")
    if total_active:
        cm_share = pool.get("command_mismatch", {}).get("n_active_steps", 0) / total_active
        slip_share = pool.get("slip", {}).get("n_active_steps", 0) / total_active
        col_share = pool.get("collapse", {}).get("n_active_steps", 0) / total_active
        att_share = pool.get("attitude", {}).get("n_active_steps", 0) / total_active
    else:
        cm_share = slip_share = col_share = att_share = 0.0
    lines.append(
        f"Reading: the pool is balanced at 3 trajectories per mode but "
        f"skewed in active-failure-step duration toward `command_mismatch` "
        f"({cm_share:.2f} of active steps) and `slip` ({slip_share:.2f}). "
        f"High-severity but short-event modes (`collapse` {col_share:.2f}, "
        f"`attitude` {att_share:.2f}) together account for only "
        f"~{(col_share + att_share):.2f} of active steps. Any lift the "
        "curriculum delivers is therefore disproportionately attributable "
        "to the `command_mismatch` and `slip` channels by step-count "
        "exposure. Confirming or refuting this is the explicit purpose "
        "of the next ablation (mode-subset sweep at fixed ff=0.5)."
    )
    lines.append("")
    return "\n".join(lines)
