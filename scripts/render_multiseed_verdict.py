"""Render the 2026-05-07 multi-seed verdict into notes/.

Reads the 6-cell pilot directory and emits the verdict markdown by
plugging the analysis pipeline's tables into the template.
"""

from __future__ import annotations

import sys
from pathlib import Path

# pythonpath shim so this works from the ashfall root without an install.
ASHFALL_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ASHFALL_SRC))

from ashfall.analysis.multiseed import (  # noqa: E402
    paired_delta,
    per_ff_summary,
    load_cells,
    render_markdown,
)


def render_verdict(results_dir: Path) -> str:
    rows = load_cells(results_dir)
    if not rows:
        raise RuntimeError(f"No multi-seed pilot cells found under {results_dir}")
    summaries = per_ff_summary(rows)
    deltas_by_terrain = {
        t: paired_delta(rows, ff_a=0.0, ff_b=0.5, terrain=t)
        for t in ("slippery", "rough")
    }
    body = render_markdown(rows, summaries, list(deltas_by_terrain.values()))

    slip = deltas_by_terrain["slippery"]
    rough = deltas_by_terrain["rough"]

    def _verdict_line(label: str, prior_pp: float, d) -> list[str]:
        signs = [d.deltas[i] > 0 for i in range(len(d.deltas))]
        pos = sum(signs)
        return [
            f"### {label}",
            "",
            f"- Prior single-seed lift: +{prior_pp:.1f} pp.",
            f"- Cross-seed mean delta: {d.mean_delta * 100:+.2f} pp "
            f"(SE {d.sem_delta * 100:.2f}, 95% CI "
            f"[{d.ci95_lo * 100:+.2f}, {d.ci95_hi * 100:+.2f}] pp).",
            f"- Per-seed signs: {pos}/{d.n_seeds} positive.",
            f"- Exact two-sided sign-flip p: {d.permutation_p_two_sided:.3f} "
            f"(floor at n=3 is 0.25).",
            f"- CI crosses zero: {'yes' if d.ci95_lo < 0 < d.ci95_hi else 'no'}.",
            "",
        ]

    parts: list[str] = []
    parts.append("# Multi-seed pilot verdict (2026-05-07)")
    parts.append("")
    parts.append("Status: COMPLETE")
    parts.append("")
    parts.append("## Mission recap")
    parts.append("")
    parts.append(
        "The v0.3.0 single-seed sweep reported a +5.1 pp slippery lift at "
        "ff=0.5 vs ff=0.0 and a +6.1 pp rough lift at ff=0.1. The 2026-05-07 "
        "rigor pass found that neither cell survives Holm-Bonferroni correction "
        "with single-seed data, and cross-seed variance was unmeasured. This "
        "pilot fills that gap with 2 ff values (0.0, 0.5) x 3 seeds (42, 123, 7), "
        "200-iter PPO fine-tune, 128-episode eval on rough + slippery."
    )
    parts.append("")
    parts.append("## Statistical floor with n=3")
    parts.append("")
    parts.append(
        "With n_seeds=3 the exact two-sided sign-flip permutation test has "
        "2**3 = 8 permutations and a minimum achievable p of 2/8 = 0.25. "
        "This is a structural ceiling, not a bug. A 'significant' result at "
        "alpha=0.05 is impossible with n=3; the verdict therefore rests on "
        "effect-size CIs and per-seed pattern, not on a single p-value."
    )
    parts.append("")
    parts.append(body)
    parts.append("")
    parts.append("## Verdict")
    parts.append("")
    parts.extend(_verdict_line("Slippery (prior optimum at ff=0.5)", 5.1, slip))
    parts.extend(_verdict_line("Rough (retention check)", 1.5, rough))
    parts.append("")
    parts.append("## Honest claim Ashfall can now make")
    parts.append("")
    slip_pos = sum(1 for x in slip.deltas if x > 0)
    if slip_pos == slip.n_seeds and slip.ci95_lo > 0:
        slip_claim = (
            "the failure curriculum at ff=0.5 yields a positive cross-seed mean "
            "lift on slippery whose 95% CI excludes zero (n=3)"
        )
    elif slip_pos == slip.n_seeds:
        slip_claim = (
            "the failure curriculum at ff=0.5 yields a positive lift on slippery "
            "in 3/3 seeds, but the 95% CI on the cross-seed mean still crosses "
            "zero at n=3, so the effect is directionally consistent but not "
            "yet rigour-passing"
        )
    elif slip_pos > slip.n_seeds // 2:
        slip_claim = (
            "the failure curriculum at ff=0.5 lifts slippery success in a "
            "majority of seeds but at least one seed regresses, indicating "
            "high cross-seed variance; the v0.3.0 single-seed lift is "
            "directionally plausible but not reliably reproducible"
        )
    else:
        slip_claim = (
            "the failure curriculum at ff=0.5 does NOT reliably lift slippery "
            "success across seeds; the v0.3.0 single-seed +5.1 pp result was a "
            "single-seed artifact and does not replicate"
        )
    parts.append("- Slippery: " + slip_claim + ".")
    parts.append(
        "- Rough: cross-seed mean delta "
        f"{rough.mean_delta * 100:+.2f} pp "
        f"(95% CI [{rough.ci95_lo * 100:+.2f}, {rough.ci95_hi * 100:+.2f}] pp); "
        "the curriculum did "
        + (
            "NOT degrade rough retention beyond noise."
            if rough.ci95_lo > -0.05
            else "appear to degrade rough retention; see CI."
        )
    )
    parts.append("")
    parts.append(
        "Reviewer-defensible one-liner: \"The failure-fraction sweep's "
        "v0.3.0 ff=0.5 slippery optimum is "
        + (
            "directionally seed-stable across 3 seeds (3/3 positive) "
            "but does not clear an n=3 95% CI on the paired-seed mean."
            if slip_pos == slip.n_seeds
            else "NOT seed-stable; the v0.3.0 single-seed lift is an artifact."
        )
        + " Multi-seed scaling (5+ seeds) is the gate before any v0.4.0 "
        "claim that the lift is real.\""
    )
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    results_dir = Path("/home/yusuf/Projects/ashfall/results")
    out_path = Path("/home/yusuf/Projects/ashfall/notes/2026-05-07-multiseed-verdict.md")
    md = render_verdict(results_dir)
    out_path.write_text(md)
    print(f"wrote {out_path}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
