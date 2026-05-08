"""Mode-subset Stage-1 analysis (2026-05-08 failure-modes pilot).

Walks an Ashfall mode-subset pilot results directory layout::

    results/
      failure_modes_pilot_2026-05-08_failure_modes=<subset>_seed=<s>/
        <stamp>/metrics/metrics_{rough,slippery}.json

and produces, for each subset, a per-seed paired delta against the
existing n=3 ff=0.0 baseline cells from
``multiseed_pilot_2026-05-07_failure_fraction=0p0_seed=*``.

The Stage-1 verdict is purely directional: a subset is shortlisted as
a "Stage-2 candidate" iff all 3 paired slippery deltas are strictly
positive. n=3 caps the exact sign-flip p-value floor at 2/8 = 0.25, so
no subset can clear alpha=0.05 here. Honest triage only.

Pure numpy. No torch import. Safe to call from no-sim CI.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ashfall.analysis.multiseed import (
    CellRaw,
    _exact_sign_flip_p,
    _t_975,
)

# Mode-subset cell name pattern. Anchored on the 2026-05-08 sweep prefix.
SUBSET_CELL_PATTERN = re.compile(
    r"failure_modes_pilot_2026-05-08_failure_modes=(?P<subset>[A-Za-z_]+)_seed=(?P<seed>\d+)$"
)

# Ordered subset list controls the report row order. Mirrors the labels
# in src/ashfall/experiment/sweep.py::_FAILURE_MODES_LABELS.
SUBSET_ORDER: tuple[str, ...] = (
    "all_modes",
    "slip_only",
    "command_mismatch_only",
    "slip_plus_cm",
    "severe_only",
    "severe_plus_slip",
)

PILOT_SEEDS: tuple[int, ...] = (42, 123, 7)


@dataclass
class SubsetCellRaw:
    """Per-cell raw row for a mode-subset run."""

    subset: str
    seed: int
    terrain: str
    n_episodes: int
    success_rate: float

    @property
    def k(self) -> int:
        return int(round(self.success_rate * self.n_episodes))


@dataclass
class SubsetComparisonResult:
    """Paired-by-seed comparison of one subset to the ff=0.0 baseline."""

    subset: str
    terrain: str
    n_seeds: int
    seeds: list[int]
    subset_success: list[float]  # success_rate per seed for the subset
    baseline_success: list[float]  # success_rate per seed for the baseline
    deltas: list[float]
    mean_delta: float
    std_delta: float
    sem_delta: float
    ci95_lo: float
    ci95_hi: float
    permutation_p_two_sided: float
    n_positive: int  # count of strictly-positive per-seed deltas
    n_negative: int  # count of strictly-negative per-seed deltas

    @property
    def all_positive(self) -> bool:
        return self.n_seeds > 0 and self.n_positive == self.n_seeds

    @property
    def all_negative(self) -> bool:
        return self.n_seeds > 0 and self.n_negative == self.n_seeds


@dataclass
class Stage1Report:
    """Aggregate Stage-1 verdict over all subsets."""

    subset_rows: list[SubsetCellRaw]
    baseline_rows: list[CellRaw]
    comparisons: list[SubsetComparisonResult]
    stage2_candidates: list[str] = field(default_factory=list)
    clear_losers: list[str] = field(default_factory=list)


def load_subset_cells(
    results_dir: Path,
    *,
    sweep_prefix: str = "failure_modes_pilot_2026-05-08",
) -> list[SubsetCellRaw]:
    """Load all mode-subset cells under ``results_dir``."""
    rows: list[SubsetCellRaw] = []
    glob_pattern = f"{sweep_prefix}_failure_modes=*_seed=*"
    for cell_dir in sorted(Path(results_dir).glob(glob_pattern)):
        m = SUBSET_CELL_PATTERN.search(cell_dir.name)
        if not m:
            continue
        subset = m.group("subset")
        seed = int(m.group("seed"))
        stamp_dirs = sorted(
            (d for d in cell_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
        if not stamp_dirs:
            continue
        chosen = None
        for sd in stamp_dirs:
            mdir = sd / "metrics"
            if (
                (mdir / "metrics_slippery.json").exists()
                and (mdir / "metrics_rough.json").exists()
            ):
                chosen = sd
                break
        if chosen is None:
            continue
        for terrain in ("rough", "slippery"):
            mf = chosen / "metrics" / f"metrics_{terrain}.json"
            if not mf.exists():
                continue
            with open(mf) as f:
                d = json.load(f)
            rows.append(
                SubsetCellRaw(
                    subset=subset,
                    seed=seed,
                    terrain=terrain,
                    n_episodes=int(d.get("num_episodes", 0)),
                    success_rate=float(d.get("success_rate", 0.0)),
                )
            )
    return rows


def load_baseline_cells(
    baseline_dir: Path,
    *,
    pilot_prefix: str = "multiseed_pilot_2026-05-07",
    seeds: tuple[int, ...] = PILOT_SEEDS,
) -> list[CellRaw]:
    """Load the existing n=3 ff=0.0 pilot cells for paired comparison."""
    from ashfall.analysis.multiseed import load_cells

    rows = load_cells(Path(baseline_dir), pilot_prefix=pilot_prefix)
    # Restrict to ff=0.0 + the configured seeds. The pilot directory also
    # contains ff=0.5 cells; we only want the ff=0.0 baseline here.
    return [r for r in rows if r.ff == 0.0 and r.seed in seeds]


def compare_subset_to_baseline(
    subset_rows: list[SubsetCellRaw],
    baseline_rows: list[CellRaw],
    *,
    subset: str,
    terrain: str,
    seeds: tuple[int, ...] = PILOT_SEEDS,
) -> SubsetComparisonResult:
    """Per-seed paired delta of one subset vs the ff=0.0 baseline.

    Pairs cells by seed; only seeds present at BOTH the subset and the
    baseline on the requested terrain contribute. Returns the mean
    paired delta, t-CI95, and exact sign-flip permutation p (n<=20
    enumerable). Per-seed positive/negative counts drive the Stage-1
    candidate / loser triage.
    """
    by_seed_subset = {
        r.seed: r
        for r in subset_rows
        if r.subset == subset and r.terrain == terrain and r.seed in seeds
    }
    by_seed_baseline = {
        r.seed: r
        for r in baseline_rows
        if r.terrain == terrain and r.seed in seeds and r.ff == 0.0
    }
    common = sorted(set(by_seed_subset) & set(by_seed_baseline))
    sub_sr = [by_seed_subset[s].success_rate for s in common]
    base_sr = [by_seed_baseline[s].success_rate for s in common]
    deltas = [a - b for a, b in zip(sub_sr, base_sr)]
    n = len(deltas)
    if n == 0:
        return SubsetComparisonResult(
            subset=subset,
            terrain=terrain,
            n_seeds=0,
            seeds=[],
            subset_success=[],
            baseline_success=[],
            deltas=[],
            mean_delta=0.0,
            std_delta=0.0,
            sem_delta=0.0,
            ci95_lo=0.0,
            ci95_hi=0.0,
            permutation_p_two_sided=1.0,
            n_positive=0,
            n_negative=0,
        )
    arr = np.array(deltas, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    half_width = _t_975(n) * sem
    return SubsetComparisonResult(
        subset=subset,
        terrain=terrain,
        n_seeds=n,
        seeds=common,
        subset_success=sub_sr,
        baseline_success=base_sr,
        deltas=deltas,
        mean_delta=mean,
        std_delta=std,
        sem_delta=sem,
        ci95_lo=mean - half_width,
        ci95_hi=mean + half_width,
        permutation_p_two_sided=_exact_sign_flip_p(deltas),
        n_positive=int(sum(1 for d in deltas if d > 0)),
        n_negative=int(sum(1 for d in deltas if d < 0)),
    )


def run_stage1_analysis(
    subset_rows: list[SubsetCellRaw],
    baseline_rows: list[CellRaw],
    *,
    subsets: tuple[str, ...] = SUBSET_ORDER,
    seeds: tuple[int, ...] = PILOT_SEEDS,
) -> Stage1Report:
    """Stage-1 triage across all subsets and both terrains.

    Stage-2 candidate: 3/3 strictly-positive slippery deltas.
    Clear loser: 3/3 strictly-positive negation, i.e. 0/3 positive
    AND all 3 deltas strictly negative.
    """
    comparisons: list[SubsetComparisonResult] = []
    for subset in subsets:
        for terrain in ("slippery", "rough"):
            comparisons.append(
                compare_subset_to_baseline(
                    subset_rows,
                    baseline_rows,
                    subset=subset,
                    terrain=terrain,
                    seeds=seeds,
                )
            )

    candidates: list[str] = []
    losers: list[str] = []
    for subset in subsets:
        slip_cmp = next(
            (c for c in comparisons if c.subset == subset and c.terrain == "slippery"),
            None,
        )
        if slip_cmp is None:
            continue
        if slip_cmp.all_positive:
            candidates.append(subset)
        elif slip_cmp.all_negative:
            losers.append(subset)

    return Stage1Report(
        subset_rows=list(subset_rows),
        baseline_rows=list(baseline_rows),
        comparisons=comparisons,
        stage2_candidates=candidates,
        clear_losers=losers,
    )


def _format_per_seed_signs(cmp: SubsetComparisonResult) -> str:
    """Render per-seed deltas as a compact ``+0.0123 / -0.0050 / +0.0210`` string."""
    return " / ".join(f"{d:+.4f}" for d in cmp.deltas) if cmp.deltas else "(none)"


def render_stage1_markdown(
    report: Stage1Report,
    *,
    title: str = "Mode-subset Stage-1 verdict (2026-05-08)",
) -> str:
    """Render the Stage-1 report as markdown."""
    lines: list[str] = [f"# {title}", ""]

    n_slip = max(
        (c.n_seeds for c in report.comparisons if c.terrain == "slippery"),
        default=0,
    )
    p_floor = (2.0 / (2 ** n_slip)) if n_slip > 0 else 1.0
    lines.append(
        f"Stage-1 triage: 6 mode subsets x {n_slip} pilot seeds at fixed ff=0.5, "
        f"paired against the existing n={n_slip} ff=0.0 multiseed_pilot baseline."
    )
    lines.append(
        f"Exact sign-flip p-floor at n={n_slip} is 2/{2**n_slip} = {p_floor:.4f}; "
        "alpha=0.05 cannot be reached here. Stage-1 is directional triage only."
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(
        "**Stage-2 candidates (3/3 positive slippery sign):** "
        + (", ".join(f"`{s}`" for s in report.stage2_candidates) or "_none_")
    )
    lines.append("")
    lines.append(
        "**Clear losers (3/3 negative slippery sign):** "
        + (", ".join(f"`{s}`" for s in report.clear_losers) or "_none_")
    )
    lines.append("")

    lines.append("## 18-cell raw success rates")
    lines.append("")
    lines.append("| subset                  | seed | terrain  |   n |   k | success |")
    lines.append("|:------------------------|-----:|:---------|----:|----:|--------:|")
    for r in sorted(report.subset_rows, key=lambda x: (x.subset, x.terrain, x.seed)):
        lines.append(
            f"| {r.subset:<22} | {r.seed:>4} | {r.terrain:<8} | "
            f"{r.n_episodes:>3} | {r.k:>3} | {r.success_rate:.4f} |"
        )
    lines.append("")

    lines.append("## Baseline cells (ff=0.0, multiseed_pilot 2026-05-07)")
    lines.append("")
    lines.append("| seed | terrain  |   n |   k | success |")
    lines.append("|-----:|:---------|----:|----:|--------:|")
    for r in sorted(report.baseline_rows, key=lambda x: (x.terrain, x.seed)):
        lines.append(
            f"| {r.seed:>4} | {r.terrain:<8} | "
            f"{r.n_episodes:>3} | {r.k:>3} | {r.success_rate:.4f} |"
        )
    lines.append("")

    for terrain in ("slippery", "rough"):
        lines.append(f"## Per-subset paired delta vs ff=0.0 baseline ({terrain})")
        lines.append("")
        lines.append(
            "| subset                  | n | per-seed deltas               | "
            "mean   | SE     | 95% CI            | exact perm p | signs (+/-) |"
        )
        lines.append(
            "|:------------------------|--:|:------------------------------|"
            "-------:|-------:|:------------------|-------------:|:------------|"
        )
        for cmp in report.comparisons:
            if cmp.terrain != terrain:
                continue
            signs = f"{cmp.n_positive}/{cmp.n_seeds} pos, {cmp.n_negative}/{cmp.n_seeds} neg"
            lines.append(
                f"| {cmp.subset:<22} | {cmp.n_seeds} | "
                f"{_format_per_seed_signs(cmp):<29} | "
                f"{cmp.mean_delta:+.4f} | {cmp.sem_delta:.4f} | "
                f"[{cmp.ci95_lo:+.4f}, {cmp.ci95_hi:+.4f}] | "
                f"{cmp.permutation_p_two_sided:.4f} | {signs} |"
            )
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- n=3 sign-flip p-floor is 0.25; alpha=0.05 cannot be cleared at "
        "this seed budget. Stage-1 is directional only."
    )
    lines.append(
        "- 3/3 positive slippery sign DOES NOT mean the subset works; it "
        "means the subset survives Stage-1 triage and warrants n>=5 scaling."
    )
    lines.append(
        "- The ff=0.0 baseline cells are not re-run in this sweep; we reuse "
        "the existing 2026-05-07 multiseed_pilot ff=0.0 cells. This holds "
        "all training inputs except the curriculum constant across the "
        "paired comparison."
    )
    lines.append("")
    return "\n".join(lines)


def run(
    results_dir: str | Path,
    *,
    baseline_dir: str | Path | None = None,
) -> Stage1Report:
    """One-shot pipeline for the 2026-05-08 mode-subset pilot."""
    results_path = Path(results_dir)
    base_path = Path(baseline_dir) if baseline_dir is not None else results_path
    subset_rows = load_subset_cells(results_path)
    baseline_rows = load_baseline_cells(base_path)
    return run_stage1_analysis(subset_rows, baseline_rows)


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "results"
    report = run(target)
    print(render_stage1_markdown(report))
