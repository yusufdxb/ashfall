"""Render the 2026-05-07 multi-seed scale (n=7) verdict into notes/.

Combines the n=3 pilot directory with the n=4 scale directory and emits
the n=7 verdict markdown. Runs the paired-by-seed delta + exact
sign-flip permutation test on the combined cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

# pythonpath shim so this works from the ashfall root without an install.
ASHFALL_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ASHFALL_SRC))

from ashfall.analysis.multiseed import (  # noqa: E402
    AnalysisReport,
    combine_pilot_runs,
    paired_delta,
    render_markdown,
    run_combined_analysis,
)


def _verdict_lines(label: str, prior_pp: float, d, p_floor: float) -> list[str]:
    pos = sum(1 for x in d.deltas if x > 0)
    return [
        f"### {label}",
        "",
        f"- Prior single-seed lift: +{prior_pp:.1f} pp.",
        "- n=3 pilot mean delta (subset): see notes/2026-05-07-multiseed-verdict.md.",
        f"- Cross-seed mean delta (n={d.n_seeds}): {d.mean_delta * 100:+.2f} pp "
        f"(SE {d.sem_delta * 100:.2f}, 95% CI "
        f"[{d.ci95_lo * 100:+.2f}, {d.ci95_hi * 100:+.2f}] pp).",
        f"- Per-seed signs: {pos}/{d.n_seeds} positive.",
        f"- Exact two-sided sign-flip p: {d.permutation_p_two_sided:.4f} "
        f"(floor at n={d.n_seeds} is {p_floor:.4f}).",
        f"- CI crosses zero: {'yes' if d.ci95_lo < 0 < d.ci95_hi else 'no'}.",
        f"- Clears alpha=0.05: "
        f"{'yes' if d.permutation_p_two_sided < 0.05 else 'no'}.",
        "",
    ]


def _pilot_subset(report: AnalysisReport, terrain: str, seeds: set[int]):
    """Recompute paired delta on only the pilot seeds for comparison."""
    rows = [r for r in report.rows if r.seed in seeds]
    return paired_delta(rows, ff_a=0.0, ff_b=0.5, terrain=terrain)


def render_n7_verdict(results_dir: Path) -> str:
    combined = combine_pilot_runs(
        (results_dir, "multiseed_pilot_2026-05-07"),
        (results_dir, "multiseed_scale_2026-05-07"),
    )
    if not combined.rows:
        raise RuntimeError(
            f"No pilot or scale cells found under {results_dir}. "
            "Run scripts/run_multiseed_pilot.sh and "
            "scripts/run_multiseed_scale.sh first."
        )
    report = run_combined_analysis(combined)
    body = render_markdown(report.rows, report.per_ff, report.deltas)

    slip = next(d for d in report.deltas if d.terrain == "slippery")
    rough = next(d for d in report.deltas if d.terrain == "rough")
    n_slip = slip.n_seeds
    n_rough = rough.n_seeds
    p_floor_slip = 2.0 / (2 ** n_slip) if n_slip else 1.0
    p_floor_rough = 2.0 / (2 ** n_rough) if n_rough else 1.0

    pilot_seeds = {42, 123, 7}
    slip_pilot = _pilot_subset(report, "slippery", pilot_seeds)
    rough_pilot = _pilot_subset(report, "rough", pilot_seeds)

    parts: list[str] = []
    parts.append("# Multi-seed scale verdict (n=7, 2026-05-07)")
    parts.append("")
    parts.append("Status: COMPLETE")
    parts.append("")
    parts.append("## Mission recap")
    parts.append("")
    parts.append(
        "The 2026-05-07 n=3 pilot found a directionally consistent slippery "
        "lift at ff=0.5 (3/3 positive seeds, mean +2.14 pp) but with a "
        "structural permutation p-floor of 2/8 = 0.25 that prevents "
        "alpha=0.05 significance regardless of effect size. This scale pass "
        "adds 4 new seeds (99, 314, 1729, 2718) at ff in {0.0, 0.5} for an "
        "n=7 paired-by-seed analysis. At n=7 the exact sign-flip "
        "permutation test has 2**7 = 128 permutations, so p-floor = "
        "2/128 = 0.0156 < 0.05; alpha=0.05 IS reachable when the effect is "
        "consistent across seeds."
    )
    parts.append("")
    parts.append(f"## Statistical floor at n={n_slip}")
    parts.append("")
    parts.append(
        f"With n_seeds={n_slip} the exact two-sided sign-flip permutation "
        f"test has 2**{n_slip} = {2**n_slip} permutations and a minimum "
        f"achievable p of 2/{2**n_slip} = {p_floor_slip:.4f}. CIs use the "
        f"t-multiplier at df={n_slip - 1} (t_975 = 2.447 for n=7)."
    )
    parts.append("")
    parts.append(body)
    parts.append("")

    parts.append("## Pilot-only (n=3) vs scaled (n=7) comparison")
    parts.append("")
    parts.append(
        "| terrain  | n  | mean delta | 95% CI                | "
        "exact perm p |"
    )
    parts.append(
        "|:---------|---:|-----------:|:----------------------|"
        "-------------:|"
    )
    for label, d in (
        (f"slippery (pilot, n={slip_pilot.n_seeds})", slip_pilot),
        (f"slippery (scaled, n={slip.n_seeds})", slip),
        (f"rough (pilot, n={rough_pilot.n_seeds})", rough_pilot),
        (f"rough (scaled, n={rough.n_seeds})", rough),
    ):
        parts.append(
            f"| {label:<26} | {d.n_seeds:>2} | "
            f"{d.mean_delta * 100:+.2f} pp | "
            f"[{d.ci95_lo * 100:+.2f}, {d.ci95_hi * 100:+.2f}] pp | "
            f"{d.permutation_p_two_sided:.4f} |"
        )
    parts.append("")

    parts.append("## Verdict")
    parts.append("")
    parts.extend(
        _verdict_lines(
            "Slippery (prior optimum at ff=0.5)", 5.1, slip, p_floor_slip
        )
    )
    parts.extend(
        _verdict_lines(
            "Rough (retention check)", 1.5, rough, p_floor_rough
        )
    )
    parts.append("")
    parts.append("## Honest claim Ashfall can now make")
    parts.append("")

    slip_pos = sum(1 for x in slip.deltas if x > 0)
    rough_pos = sum(1 for x in rough.deltas if x > 0)

    if slip.permutation_p_two_sided < 0.05 and slip.ci95_lo > 0:
        slip_claim = (
            f"the failure curriculum at ff=0.5 yields a positive cross-seed "
            f"mean lift on slippery whose 95% CI excludes zero AND whose "
            f"exact sign-flip permutation p ({slip.permutation_p_two_sided:.4f}) "
            f"clears alpha=0.05 at n={slip.n_seeds}"
        )
    elif slip.permutation_p_two_sided < 0.05:
        slip_claim = (
            f"the failure curriculum at ff=0.5 produces a paired-by-seed "
            f"effect significant by exact sign-flip at "
            f"p={slip.permutation_p_two_sided:.4f} (n={slip.n_seeds}), but "
            f"the t-CI on the mean still crosses zero "
            f"([{slip.ci95_lo * 100:+.2f}, {slip.ci95_hi * 100:+.2f}] pp)"
        )
    elif slip_pos == slip.n_seeds:
        slip_claim = (
            f"the failure curriculum at ff=0.5 yields a positive lift on "
            f"slippery in {slip_pos}/{slip.n_seeds} seeds (perfect sign "
            f"agreement) but the exact permutation p "
            f"({slip.permutation_p_two_sided:.4f}) does NOT clear alpha=0.05; "
            f"effect is directionally seed-stable but the magnitude is too "
            f"small to be detectable at n={slip.n_seeds}"
        )
    elif slip_pos > slip.n_seeds // 2:
        slip_claim = (
            f"the failure curriculum at ff=0.5 lifts slippery success in "
            f"{slip_pos}/{slip.n_seeds} seeds; permutation p "
            f"{slip.permutation_p_two_sided:.4f} does not clear alpha=0.05; "
            f"effect is directionally plausible but unreliable across seeds"
        )
    else:
        slip_claim = (
            f"the failure curriculum at ff=0.5 does NOT reliably lift "
            f"slippery success across seeds (only {slip_pos}/{slip.n_seeds} "
            f"positive); the v0.3.0 single-seed +5.1 pp result does not "
            f"replicate at n={slip.n_seeds}"
        )

    parts.append("- Slippery: " + slip_claim + ".")

    if rough.permutation_p_two_sided < 0.05 and rough.mean_delta < 0:
        rough_claim = (
            f"the curriculum SIGNIFICANTLY DEGRADES rough retention "
            f"(mean {rough.mean_delta * 100:+.2f} pp, exact p="
            f"{rough.permutation_p_two_sided:.4f}, "
            f"{rough_pos}/{rough.n_seeds} positive)"
        )
    elif rough_pos > rough.n_seeds // 2 and rough.ci95_lo > -0.05:
        rough_claim = (
            f"rough retention preserved within noise (mean "
            f"{rough.mean_delta * 100:+.2f} pp, "
            f"95% CI [{rough.ci95_lo * 100:+.2f}, "
            f"{rough.ci95_hi * 100:+.2f}] pp, "
            f"{rough_pos}/{rough.n_seeds} positive)"
        )
    else:
        rough_claim = (
            f"rough retention is NOT preserved on average (mean "
            f"{rough.mean_delta * 100:+.2f} pp, "
            f"95% CI [{rough.ci95_lo * 100:+.2f}, "
            f"{rough.ci95_hi * 100:+.2f}] pp, only "
            f"{rough_pos}/{rough.n_seeds} positive); the curriculum trades "
            f"rough proficiency for slippery training"
        )

    parts.append("- Rough: " + rough_claim + ".")
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    results_dir = Path("/home/yusuf/Projects/ashfall/results")
    out_path = Path(
        "/home/yusuf/Projects/ashfall/notes/2026-05-07-multiseed-scale-verdict.md"
    )
    md = render_n7_verdict(results_dir)
    out_path.write_text(md)
    print(f"wrote {out_path}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
