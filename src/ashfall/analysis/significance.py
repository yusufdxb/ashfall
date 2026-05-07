"""Sweep-level statistical-significance pipeline.

Walks an Ashfall failure-fraction sweep results directory, loads each
cell's aggregate ``metrics_*.json``, and runs paired bootstrap CI +
permutation / Fisher's exact tests for every non-control cell against
the ff=0.0 control. Holm-Bonferroni adjusts the resulting family of
p-values per terrain.

The output is consumed by :mod:`ashfall.analysis.sweep_report` to
emit a "Statistical Significance" section in REPORT.md.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ashfall.evaluation.significance import (
    ProportionDiffResult,
    holm_adjust,
    proportion_diff_test,
)

logger = logging.getLogger("ashfall.analysis.significance")

CELL_PATTERN = re.compile(
    r"ablation_failure_fraction_failure_fraction=(?P<short>[0-9p]+)$"
)


def _short_to_float(short: str) -> float:
    return float(short.replace("p", "."))


@dataclass
class CellSignificance:
    """Per-cell rigour summary across both terrains."""

    failure_fraction: float
    n_episodes: dict[str, int] = field(default_factory=dict)
    success_rate: dict[str, float] = field(default_factory=dict)
    diff_vs_control: dict[str, ProportionDiffResult] = field(default_factory=dict)
    holm_p: dict[str, float] = field(default_factory=dict)


def _load_cells_for_significance(results_dir: Path) -> list[dict]:
    cells: list[dict] = []
    for cell_dir in sorted(results_dir.glob("ablation_failure_fraction_failure_fraction=*")):
        m = CELL_PATTERN.search(cell_dir.name)
        if not m:
            continue
        ff = _short_to_float(m.group("short"))
        stamp_dirs = sorted(
            (d for d in cell_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
        chosen = None
        for sd in stamp_dirs:
            mdir = sd / "metrics"
            if (mdir / "metrics_slippery.json").exists() or (mdir / "metrics_rough.json").exists():
                chosen = sd
                break
        if chosen is None:
            continue

        per_env = {}
        for env_name in ("slippery", "rough"):
            mf = chosen / "metrics" / f"metrics_{env_name}.json"
            if not mf.exists():
                continue
            with open(mf) as f:
                d = json.load(f)
            n = int(d.get("num_episodes", 0))
            sr = float(d.get("success_rate", 0.0))
            k = int(round(sr * n))
            per_env[env_name] = {"n": n, "k": k, "sr": sr}

        cells.append({
            "failure_fraction": ff,
            "name": cell_dir.name,
            "per_env": per_env,
        })

    cells.sort(key=lambda c: c["failure_fraction"])
    return cells


def compute_sweep_significance(
    results_dir: str | Path,
    *,
    control_ff: float = 0.0,
    n_bootstrap: int = 10_000,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[CellSignificance]:
    """Compute the per-cell rigour table.

    For every (ff != control_ff, env) pair, runs ``proportion_diff_test``
    against the control cell on the same terrain. Holm-Bonferroni
    adjusts the family of p-values within each terrain.
    """
    results_dir = Path(results_dir)
    cells = _load_cells_for_significance(results_dir)
    if not cells:
        raise RuntimeError(f"No sweep cells found under {results_dir}")

    control = next((c for c in cells if abs(c["failure_fraction"] - control_ff) < 1e-9), None)
    if control is None:
        raise RuntimeError(f"Control cell ff={control_ff} not found")

    out: list[CellSignificance] = []
    raw_p_by_env: dict[str, list[tuple[int, float]]] = {"slippery": [], "rough": []}
    cell_idx_to_out_idx: dict[tuple[int, str], int] = {}

    for cell in cells:
        sig = CellSignificance(failure_fraction=cell["failure_fraction"])
        for env_name in ("slippery", "rough"):
            env_data = cell["per_env"].get(env_name)
            if env_data is None:
                continue
            sig.n_episodes[env_name] = env_data["n"]
            sig.success_rate[env_name] = env_data["sr"]
            if cell is control:
                continue
            ctrl_env = control["per_env"].get(env_name)
            if ctrl_env is None:
                continue
            res = proportion_diff_test(
                k_a=ctrl_env["k"], n_a=ctrl_env["n"],
                k_b=env_data["k"], n_b=env_data["n"],
                n_bootstrap=n_bootstrap, n_perm=n_perm,
                alpha=alpha, seed=seed,
            )
            sig.diff_vs_control[env_name] = res
            raw_p_by_env[env_name].append((len(out), res.p_value))
        out.append(sig)
        if cell is not control:
            for env_name in ("slippery", "rough"):
                if env_name in sig.diff_vs_control:
                    cell_idx_to_out_idx[(len(out) - 1, env_name)] = len(out) - 1

    # Holm-Bonferroni within each terrain.
    for env_name, items in raw_p_by_env.items():
        if not items:
            continue
        ps = [p for _, p in items]
        adj = holm_adjust(ps)
        for (idx, _), q in zip(items, adj):
            out[idx].holm_p[env_name] = q

    return out


def render_significance_markdown(sigs: list[CellSignificance]) -> str:
    """Render the significance table as a markdown section."""
    lines: list[str] = []
    lines.append("## Statistical Significance")
    lines.append("")
    lines.append(
        "Each non-control cell tested against the ff=0.0 control on the "
        "same terrain. Method: BCa bootstrap (10k resamples) for the 95% "
        "CI on the success-rate difference; Fisher's exact two-sided "
        "test for the p-value; Holm-Bonferroni step-down adjustment "
        "across the 5 ff comparisons within each terrain."
    )
    lines.append("")
    for env_name, label in (("slippery", "Slippery"), ("rough", "Rough")):
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            "| ff   | n   | success | delta vs ff=0.0 | 95% CI"
            "            | p (Fisher) | p (Holm) | sig |"
        )
        lines.append(
            "|-----:|----:|--------:|----------------:|:------------------|-----------:|---------:|:----|"
        )
        for s in sigs:
            n = s.n_episodes.get(env_name, 0)
            sr = s.success_rate.get(env_name, float("nan"))
            res = s.diff_vs_control.get(env_name)
            holm = s.holm_p.get(env_name)
            if res is None:
                lines.append(
                    f"| {s.failure_fraction:.2f} | {n} | {sr:.3f} | _control_ |  |  |  |  |"
                )
                continue
            star = ""
            if holm is not None:
                if holm < 0.001:
                    star = "***"
                elif holm < 0.01:
                    star = "**"
                elif holm < 0.05:
                    star = "*"
                else:
                    star = "n.s."
            lines.append(
                f"| {s.failure_fraction:.2f} | {n} | {sr:.3f} | "
                f"{res.diff:+.3f} | "
                f"[{res.ci_lower:+.3f}, {res.ci_upper:+.3f}] | "
                f"{res.p_value:.4f} | "
                f"{(holm if holm is not None else float('nan')):.4f} | {star} |"
            )
        lines.append("")
    lines.append(
        "Stars: `***` Holm-adjusted p<0.001, `**` p<0.01, `*` p<0.05, "
        "`n.s.` not significant after multiple-comparison adjustment."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "results"
    sigs = compute_sweep_significance(target)
    print(render_significance_markdown(sigs))
