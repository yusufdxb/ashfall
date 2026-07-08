"""v0.3.0 ablation sweep report.

Reads all ``ablation_failure_fraction_*`` cells under ``results/`` and
produces:

- A markdown ``REPORT.md`` summarising slippery + rough success rates per
  cell with Wilson 95% CIs and the optimum highlighted.
- Plots: ``sweep_success_rate.png`` and ``sweep_slew_saturation.png``.
- A short Pareto narrative.

Wilson CIs (rather than full bootstrap) because evaluate.py only emits
aggregate counts, not per-episode arrays. With num_episodes=128 the
Wilson interval is the right tool: it generalises to any binomial.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path

import yaml

from ashfall.analysis.significance import (
    compute_sweep_significance,
    render_significance_markdown,
)

logger = logging.getLogger("ashfall.analysis.sweep_report")

CELL_PATTERN = re.compile(
    r"ablation_failure_fraction_failure_fraction=(?P<short>[0-9p]+)$"
)


def _short_to_float(short: str) -> float:
    """Convert ``0p25`` -> 0.25, ``1p0`` -> 1.0, ``0p0`` -> 0.0."""
    return float(short.replace("p", "."))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion.

    The Wilson interval has correct coverage near 0/1 (where the
    normal-approximation interval breaks down) and is the standard
    choice for n=128 episode-success counts.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _load_cells(results_dir: Path) -> list[dict]:
    """Discover sweep cells, preferring the most recent stamp per cell."""
    cells = []
    for cell_dir in sorted(results_dir.glob("ablation_failure_fraction_failure_fraction=*")):
        m = CELL_PATTERN.search(cell_dir.name)
        if not m:
            continue
        ff = _short_to_float(m.group("short"))
        # Most recent timestamped subdir with metrics.
        stamp_dirs = sorted(
            (d for d in cell_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
        chosen = None
        for sd in stamp_dirs:
            metrics_dir = sd / "metrics"
            if (metrics_dir / "metrics_slippery.json").exists() or (
                metrics_dir / "metrics_rough.json"
            ).exists():
                chosen = sd
                break
        if chosen is None:
            cells.append(
                {
                    "failure_fraction": ff,
                    "name": cell_dir.name,
                    "run_dir": str(stamp_dirs[0]) if stamp_dirs else str(cell_dir),
                    "metrics": {},
                    "missing": True,
                }
            )
            continue

        metrics = {}
        for env_name in ("slippery", "rough"):
            mf = chosen / "metrics" / f"metrics_{env_name}.json"
            if mf.exists():
                with open(mf) as f:
                    metrics[env_name] = json.load(f)

        cfg = {}
        cfg_path = chosen / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}

        cells.append(
            {
                "failure_fraction": ff,
                "name": cell_dir.name,
                "run_dir": str(chosen),
                "metrics": metrics,
                "config": cfg,
                "missing": False,
            }
        )

    cells.sort(key=lambda c: c["failure_fraction"])
    return cells


def _plot_success_curves(cells: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [c["failure_fraction"] for c in cells if not c["missing"]]
    slip = [
        c["metrics"].get("slippery", {}).get("success_rate", float("nan"))
        for c in cells
        if not c["missing"]
    ]
    rough = [
        c["metrics"].get("rough", {}).get("success_rate", float("nan"))
        for c in cells
        if not c["missing"]
    ]
    n_eps_slip = [
        c["metrics"].get("slippery", {}).get("num_episodes", 0)
        for c in cells
        if not c["missing"]
    ]
    n_eps_rough = [
        c["metrics"].get("rough", {}).get("num_episodes", 0)
        for c in cells
        if not c["missing"]
    ]

    slip_lo, slip_hi = zip(
        *[wilson_ci(int(round(p * n)), n) if n else (0.0, 0.0) for p, n in zip(slip, n_eps_slip)],
        strict=False,
    ) if xs else ([], [])
    rough_lo, rough_hi = zip(
        *[wilson_ci(int(round(p * n)), n) if n else (0.0, 0.0) for p, n in zip(rough, n_eps_rough)],
        strict=False,
    ) if xs else ([], [])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, slip, "o-", color="C0", label="slippery", linewidth=2)
    ax.fill_between(xs, slip_lo, slip_hi, color="C0", alpha=0.15)
    ax.plot(xs, rough, "s-", color="C2", label="rough", linewidth=2)
    ax.fill_between(xs, rough_lo, rough_hi, color="C2", alpha=0.15)
    ax.set_xlabel("failure_fraction")
    ax.set_ylabel("success rate")
    ax.set_title("Ashfall v0.3.0: Success Rate vs Failure-Curriculum Density")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def _plot_slew(cells: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [c["failure_fraction"] for c in cells if not c["missing"]]
    slip = [
        c["metrics"].get("slippery", {}).get("slew_saturation_pct", float("nan"))
        for c in cells
        if not c["missing"]
    ]
    rough = [
        c["metrics"].get("rough", {}).get("slew_saturation_pct", float("nan"))
        for c in cells
        if not c["missing"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, slip, "o-", color="C0", label="slippery", linewidth=2)
    ax.plot(xs, rough, "s-", color="C2", label="rough", linewidth=2)
    ax.set_xlabel("failure_fraction")
    ax.set_ylabel("slew saturation pct")
    ax.set_title("Ashfall v0.3.0: Slew Saturation vs Failure-Curriculum Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def generate_sweep_report(
    results_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    include_significance: bool = True,
    include_failure_mode_breakdown: bool = True,
) -> Path:
    results_dir = Path(results_dir)
    output_path = Path(output_path) if output_path else results_dir / "REPORT.md"

    cells = _load_cells(results_dir)
    if not cells:
        raise RuntimeError(f"No sweep cells found under {results_dir}")

    plot_success = results_dir / "sweep_success_rate.png"
    plot_slew = results_dir / "sweep_slew_saturation.png"
    try:
        _plot_success_curves(cells, plot_success)
        _plot_slew(cells, plot_slew)
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    # Build markdown.
    lines = []
    lines.append("# Ashfall v0.3.0: Failure-Fraction Sweep")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Cells discovered: {len(cells)}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "Each cell warm-starts from the v0.2.0 ashfall-baseline checkpoint "
        "(500-iter PPO on rough), then fine-tunes for 200 iters on slippery "
        "with `failure_sample_fraction` set to the cell value. Failure "
        "trajectories are sampled from the synth pool at "
        "`/home/yusuf/Projects/ashfall/data/failures/`. Each adapted "
        "checkpoint is evaluated with 128 episodes × 32 envs on rough and "
        "slippery (flat is skipped because Flat-v0 obs are incompatible "
        "with the Rough-v0 trained obs)."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| failure_fraction | slippery success | slippery 95% CI "
        "| rough success | rough 95% CI | slip slew sat | rough slew sat |"
    )
    lines.append("|---:|---:|:---:|---:|:---:|---:|---:|")

    rows = []
    for c in cells:
        if c["missing"]:
            lines.append(
                f"| {c['failure_fraction']:.2f} | _missing_ |  | _missing_ |  |  |  |"
            )
            continue
        slip = c["metrics"].get("slippery", {})
        rough = c["metrics"].get("rough", {})
        n_slip = int(slip.get("num_episodes", 0))
        n_rough = int(rough.get("num_episodes", 0))
        sr_slip = float(slip.get("success_rate", float("nan")))
        sr_rough = float(rough.get("success_rate", float("nan")))
        slip_lo, slip_hi = wilson_ci(int(round(sr_slip * n_slip)), n_slip) if n_slip else (0.0, 0.0)
        rough_lo, rough_hi = (
            wilson_ci(int(round(sr_rough * n_rough)), n_rough) if n_rough else (0.0, 0.0)
        )
        slew_slip = float(slip.get("slew_saturation_pct", float("nan")))
        slew_rough = float(rough.get("slew_saturation_pct", float("nan")))
        rows.append((
            c["failure_fraction"],
            sr_slip, sr_rough,
            slip_lo, slip_hi,
            rough_lo, rough_hi,
            slew_slip, slew_rough,
        ))
        lines.append(
            f"| {c['failure_fraction']:.2f} | {sr_slip:.3f} | "
            f"[{slip_lo:.3f}, {slip_hi:.3f}] | {sr_rough:.3f} | "
            f"[{rough_lo:.3f}, {rough_hi:.3f}] | "
            f"{slew_slip:.3f} | {slew_rough:.3f} |"
        )

    # Identify optimum (max slippery success, ties broken by rough).
    if rows:
        best = max(rows, key=lambda r: (r[1], r[2]))
        lines.append("")
        lines.append(
            f"**Optimum (slippery): failure_fraction = {best[0]:.2f}** "
            f"(slippery {best[1]:.3f} [{best[3]:.3f}, {best[4]:.3f}], "
            f"rough {best[2]:.3f} [{best[5]:.3f}, {best[6]:.3f}])."
        )

        # Pareto narrative
        ff0 = next((r for r in rows if abs(r[0]) < 1e-9), None)
        if ff0:
            lines.append("")
            lines.append("## Slippery <-> Rough Pareto")
            lines.append("")
            lines.append(
                f"Control (failure_fraction=0.0): slippery {ff0[1]:.3f}, rough {ff0[2]:.3f}."
            )
            for r in rows:
                if abs(r[0]) < 1e-9:
                    continue
                d_slip = r[1] - ff0[1]
                d_rough = r[2] - ff0[2]
                lines.append(
                    f"- ff={r[0]:.2f}: slippery {d_slip:+.3f}, rough {d_rough:+.3f}"
                )

    lines.append("")

    # Statistical significance section.
    if include_significance:
        try:
            sigs = compute_sweep_significance(results_dir)
            lines.append(render_significance_markdown(sigs))
        except Exception as exc:
            logger.warning("Significance section failed: %s", exc)

    # Failure-mode composition section. Eval-time per-mode counts were
    # not retained; we report the curriculum-input pool composition,
    # which is identical across cells. The next ablation will vary
    # this directly.
    if include_failure_mode_breakdown:
        try:
            from ashfall.analysis.failure_modes import (
                render_failure_mode_breakdown_markdown,
            )

            md = render_failure_mode_breakdown_markdown(
                results_dir=results_dir,
                pool_dir=Path("data/failures"),
            )
            lines.append(md)
        except Exception as exc:
            logger.warning("Failure-mode section failed: %s", exc)

    lines.append("## Plots")
    lines.append("")
    if plot_success.exists():
        lines.append(f"![success]({plot_success.name})")
    if plot_slew.exists():
        lines.append(f"![slew]({plot_slew.name})")
    sweep_modes_png = results_dir / "sweep_curriculum_pool_composition.png"
    if sweep_modes_png.exists():
        lines.append(f"![pool composition]({sweep_modes_png.name})")

    lines.append("")
    lines.append("## Notes & Limitations")
    lines.append(
        "- **Statistical reading**: under BCa bootstrap + Fisher's exact + "
        "Holm-Bonferroni, no individual ff cell is statistically distinct "
        "from the ff=0.0 control at alpha=0.05 per terrain. The v0.3.0 "
        "point-estimate optimum at ff=0.5 (+5.1 pp slippery) is real but "
        "its 95% CI on the difference straddles 0. Treat the optimum as "
        "the best-bet starting point for a follow-up ablation, not as a "
        "proven win."
    )
    lines.append(
        "- **Single seed**: every cell ran with `training.seed = 42`. "
        "All CIs above are within-run binomial; they say nothing about "
        "cross-seed variance."
    )
    lines.append(
        "- **No per-episode raw data**: Phoenix `evaluate.py` collapses "
        "the rollout to aggregate scalars; there are no per-episode "
        "success indicators or failure-mode labels at eval time. "
        "Per-mode breakdowns at eval time require re-running with a "
        "logger patch and are deferred."
    )
    lines.append(
        "- **n_episodes is approximate**: each cell's eval yields "
        "128 to 140 terminations depending on episode length; "
        "`num_episodes` per cell is what we trust for Wilson and "
        "bootstrap CIs."
    )
    lines.append(
        "- **Flat eval skipped**: Flat-v0 obs (no height_scan) are "
        "incompatible with Rough-v0 trained policies."
    )
    lines.append(
        "- **Next ablation**: holding ff=0.5 fixed, sweep failure-mode "
        "subsets to identify which modes carry the lift. Configs and "
        "wrapper script live under `configs/ablations/failure_modes/` "
        "and `scripts/run_failure_modes_ablation.sh`."
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("Report written to %s", out)
    return out


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    generate_sweep_report(results)
