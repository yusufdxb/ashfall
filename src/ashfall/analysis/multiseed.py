"""Multi-seed pilot analysis for the 2026-05-07 ff-validation pilot.

Walks an Ashfall multi-seed pilot results directory layout::

    results/
      multiseed_pilot_2026-05-07_failure_fraction=<ff>_seed=<s>/
        <stamp>/metrics/metrics_{rough,slippery}.json

and produces:

- A per-cell raw table (ff, seed, terrain, n, success_rate).
- Per-ff cross-seed mean and standard error per terrain.
- Per-terrain paired-by-seed delta (ff=0.5 - ff=0.0) with the
  exact one-sided sign-flip permutation p-value (n=3 -> 2**3 = 8
  permutations, asymptotic tests do not apply).

The output is consumed by ``notes/2026-05-07-multiseed-verdict.md``
and the ``docs/methodology/2026-05-07-ff-sweep-rigor.md`` update.

Pure numpy. No torch import. Safe to call from no-sim CI.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

CELL_PATTERN = re.compile(
    r"multiseed_pilot_2026-05-07_failure_fraction=(?P<ff_short>[0-9p]+)_seed=(?P<seed>\d+)$"
)


def _short_to_float(short: str) -> float:
    return float(short.replace("p", "."))


@dataclass
class CellRaw:
    ff: float
    seed: int
    terrain: str
    n_episodes: int
    success_rate: float

    @property
    def k(self) -> int:
        return int(round(self.success_rate * self.n_episodes))


@dataclass
class PerFFSummary:
    """Cross-seed mean / SE for a single (ff, terrain) cell-group."""

    ff: float
    terrain: str
    seeds: list[int] = field(default_factory=list)
    success_rates: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    sem: float = 0.0  # std / sqrt(n)


@dataclass
class PairedDeltaResult:
    """Paired-by-seed delta (ff_b - ff_a) test on a terrain."""

    terrain: str
    ff_a: float
    ff_b: float
    n_seeds: int
    deltas: list[float]  # one per seed
    mean_delta: float
    std_delta: float
    sem_delta: float
    ci95_lo: float
    ci95_hi: float
    permutation_p_two_sided: float


def load_cells(
    results_dir: Path,
    pilot_prefix: str = "multiseed_pilot_2026-05-07",
) -> list[CellRaw]:
    """Load all (ff, seed, terrain) cells under ``results_dir``."""
    rows: list[CellRaw] = []
    glob_pattern = f"{pilot_prefix}_failure_fraction=*_seed=*"
    for cell_dir in sorted(Path(results_dir).glob(glob_pattern)):
        m = CELL_PATTERN.search(cell_dir.name)
        if not m:
            continue
        ff = _short_to_float(m.group("ff_short"))
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
                CellRaw(
                    ff=ff,
                    seed=seed,
                    terrain=terrain,
                    n_episodes=int(d.get("num_episodes", 0)),
                    success_rate=float(d.get("success_rate", 0.0)),
                )
            )
    return rows


def per_ff_summary(rows: list[CellRaw]) -> list[PerFFSummary]:
    """Cross-seed mean and SE per (ff, terrain) group."""
    by_key: dict[tuple[float, str], list[CellRaw]] = {}
    for r in rows:
        by_key.setdefault((r.ff, r.terrain), []).append(r)

    out: list[PerFFSummary] = []
    for (ff, terrain), group in sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        srs = [r.success_rate for r in group]
        seeds = [r.seed for r in group]
        n = len(srs)
        if n == 0:
            continue
        mean = float(np.mean(srs))
        # Sample std (ddof=1) for unbiased estimator with small n.
        std = float(np.std(srs, ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else 0.0
        out.append(
            PerFFSummary(
                ff=ff,
                terrain=terrain,
                seeds=seeds,
                success_rates=srs,
                mean=mean,
                std=std,
                sem=sem,
            )
        )
    return out


def _exact_sign_flip_p(deltas: list[float]) -> float:
    """Two-sided exact sign-flip permutation p-value.

    With n_seeds paired deltas, the exact null permutation set under
    H0 (no effect) is the 2**n_seeds choices of sign for each delta.
    We compute the test statistic as |mean(deltas)| and count how
    many sign-flip resamples produce a statistic >= the observed one.
    Add-one smoothing avoids p=0.
    """
    n = len(deltas)
    if n == 0:
        return 1.0
    obs = abs(float(np.mean(deltas)))
    if obs == 0.0:
        return 1.0
    arr = np.array(deltas, dtype=np.float64)
    hits = 0
    total = 0
    for signs in product((-1, 1), repeat=n):
        total += 1
        flipped = arr * np.array(signs, dtype=np.float64)
        if abs(float(np.mean(flipped))) >= obs - 1e-12:
            hits += 1
    return hits / total


def paired_delta(
    rows: list[CellRaw],
    *,
    ff_a: float = 0.0,
    ff_b: float = 0.5,
    terrain: str,
) -> PairedDeltaResult:
    """Paired-by-seed (ff_b - ff_a) on a single terrain.

    Pairs cells by seed and returns the mean delta plus exact
    sign-flip permutation p-value over 2**n_seeds permutations.
    """
    by_seed_a = {r.seed: r for r in rows if r.terrain == terrain and r.ff == ff_a}
    by_seed_b = {r.seed: r for r in rows if r.terrain == terrain and r.ff == ff_b}
    common = sorted(set(by_seed_a) & set(by_seed_b))
    deltas = [by_seed_b[s].success_rate - by_seed_a[s].success_rate for s in common]
    n = len(deltas)
    if n == 0:
        return PairedDeltaResult(
            terrain=terrain,
            ff_a=ff_a,
            ff_b=ff_b,
            n_seeds=0,
            deltas=[],
            mean_delta=0.0,
            std_delta=0.0,
            sem_delta=0.0,
            ci95_lo=0.0,
            ci95_hi=0.0,
            permutation_p_two_sided=1.0,
        )
    arr = np.array(deltas, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    # 95% CI: with n=3 we use a t-multiplier (df=2 -> ~4.303).
    # Hardcode the 2.5/97.5 quantiles for small df to avoid scipy
    # circular dependence in this pure-numpy module.
    t_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 1.96)
    half_width = t_975 * sem
    ci_lo = mean - half_width
    ci_hi = mean + half_width
    p_two = _exact_sign_flip_p(deltas)
    return PairedDeltaResult(
        terrain=terrain,
        ff_a=ff_a,
        ff_b=ff_b,
        n_seeds=n,
        deltas=deltas,
        mean_delta=mean,
        std_delta=std,
        sem_delta=sem,
        ci95_lo=ci_lo,
        ci95_hi=ci_hi,
        permutation_p_two_sided=p_two,
    )


def render_markdown(
    rows: list[CellRaw],
    summaries: list[PerFFSummary],
    deltas: list[PairedDeltaResult],
) -> str:
    """Render the multi-seed verdict tables as markdown."""
    lines: list[str] = []
    lines.append("## Per-cell raw success rates")
    lines.append("")
    lines.append("| ff   | seed | terrain  |   n |   k | success |")
    lines.append("|-----:|-----:|:---------|----:|----:|--------:|")
    for r in sorted(rows, key=lambda x: (x.terrain, x.ff, x.seed)):
        lines.append(
            f"| {r.ff:.2f} | {r.seed:>4} | {r.terrain:<8} | "
            f"{r.n_episodes:>3} | {r.k:>3} | {r.success_rate:.4f} |"
        )
    lines.append("")

    lines.append("## Per-ff cross-seed summary (mean +/- SE)")
    lines.append("")
    lines.append("| terrain  | ff   |  n_seeds | mean   | std    | SE     | per-seed         |")
    lines.append("|:---------|-----:|---------:|-------:|-------:|-------:|:-----------------|")
    for s in summaries:
        per_seed = ", ".join(
            f"{seed}={sr:.3f}" for seed, sr in zip(s.seeds, s.success_rates)
        )
        lines.append(
            f"| {s.terrain:<8} | {s.ff:.2f} | {len(s.seeds):>8} | "
            f"{s.mean:.4f} | {s.std:.4f} | {s.sem:.4f} | {per_seed} |"
        )
    lines.append("")

    lines.append("## Paired-by-seed delta (ff=0.5 - ff=0.0)")
    lines.append("")
    lines.append(
        "| terrain  | n_seeds | deltas (per seed)              | mean  | SE    | "
        "95% CI            | exact perm p (two-sided) |"
    )
    lines.append(
        "|:---------|--------:|:-------------------------------|------:|------:|"
        ":------------------|-------------------------:|"
    )
    for d in deltas:
        deltas_str = ", ".join(f"{x:+.4f}" for x in d.deltas)
        lines.append(
            f"| {d.terrain:<8} | {d.n_seeds:>7} | {deltas_str} | "
            f"{d.mean_delta:+.4f} | {d.sem_delta:.4f} | "
            f"[{d.ci95_lo:+.4f}, {d.ci95_hi:+.4f}] | {d.permutation_p_two_sided:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(results_dir: str | Path) -> tuple[
    list[CellRaw], list[PerFFSummary], list[PairedDeltaResult]
]:
    """One-shot pipeline: load, summarize, paired-delta on each terrain."""
    rows = load_cells(Path(results_dir))
    summaries = per_ff_summary(rows)
    deltas = [
        paired_delta(rows, ff_a=0.0, ff_b=0.5, terrain=t)
        for t in ("slippery", "rough")
    ]
    return rows, summaries, deltas


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "results"
    rows, summaries, deltas = run(target)
    print(render_markdown(rows, summaries, deltas))
