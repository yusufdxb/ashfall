"""Tests for the n>=4 combined multi-seed analysis pipeline.

Covers the 2026-05-07 seed-scaling pass: combining the n=3 pilot
sweep (multiseed_pilot_2026-05-07_*) with the n=4 scale sweep
(multiseed_scale_2026-05-07_*) into a unified n=7 analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from ashfall.analysis.multiseed import (
    _exact_sign_flip_p,
    _t_975,
    combine_pilot_runs,
    load_cells_with_prefix,
    paired_delta,
    render_combined_markdown,
    run_combined_analysis,
)


def _write_metric(dir_path: Path, terrain: str, n: int, sr: float) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"metrics_{terrain}.json").write_text(
        json.dumps({"num_episodes": n, "success_rate": sr})
    )


def _scaffold_layout(
    root: Path,
    prefix: str,
    cells: list[tuple[float, int, dict[str, tuple[int, float]]]],
    stamp: str = "2026-05-07_22-00-00",
) -> None:
    """Write a fake sweep layout under ``root/<prefix>_*/<stamp>/metrics/``."""
    for ff, seed, metrics in cells:
        ff_short = (str(ff)).replace(".", "p")
        cell_dir = (
            root
            / f"{prefix}_failure_fraction={ff_short}_seed={seed}"
            / stamp
            / "metrics"
        )
        for terrain, (n, sr) in metrics.items():
            _write_metric(cell_dir, terrain, n, sr)


# ---------------------------------------------------------------------------
# t-multiplier table coverage.
# ---------------------------------------------------------------------------


def test_t_975_covers_n_up_to_7() -> None:
    # n=7 -> df=6 -> t = 2.447 (the value the n=7 verdict relies on).
    assert _t_975(7) == 2.447
    # n=3 unchanged from pre-existing pipeline.
    assert _t_975(3) == 4.303
    # Unknown df falls back to the normal-approx 1.96.
    assert _t_975(50) == 1.96


# ---------------------------------------------------------------------------
# Sign-flip permutation at n=7.
# ---------------------------------------------------------------------------


def test_sign_flip_p_floor_at_n7_is_two_over_one_twenty_eight() -> None:
    # 7 strictly positive deltas -> only the all-+ original and its
    # all-mirror produce |mean| >= obs. p = 2 / 128 = 0.015625.
    deltas = [0.05, 0.04, 0.03, 0.02, 0.06, 0.025, 0.045]
    p = _exact_sign_flip_p(deltas)
    assert abs(p - (2 / 128)) < 1e-9
    # Sanity: this is below the alpha=0.05 gate that n=3 cannot reach.
    assert p < 0.05


def test_sign_flip_p_at_n7_with_mixed_signs_is_larger() -> None:
    # Two negative seeds out of seven blow up the permutation count.
    # Mean = +0.01; many sign-flip resamples reach |mean| >= 0.01.
    deltas = [+0.05, +0.04, +0.03, +0.02, +0.06, -0.07, -0.06]
    p = _exact_sign_flip_p(deltas)
    # Should be much larger than 2/128; bounded between 0 and 1.
    assert p > (2 / 128)
    assert p <= 1.0


# ---------------------------------------------------------------------------
# Loader and combiner.
# ---------------------------------------------------------------------------


def test_load_cells_with_prefix_picks_only_matching_prefix(tmp_path: Path) -> None:
    _scaffold_layout(
        tmp_path,
        "multiseed_pilot_2026-05-07",
        [(0.0, 42, {"rough": (128, 0.91), "slippery": (130, 0.88)})],
    )
    _scaffold_layout(
        tmp_path,
        "multiseed_scale_2026-05-07",
        [(0.0, 99, {"rough": (128, 0.92), "slippery": (130, 0.89)})],
    )
    pilot_rows = load_cells_with_prefix(tmp_path, "multiseed_pilot_2026-05-07")
    scale_rows = load_cells_with_prefix(tmp_path, "multiseed_scale_2026-05-07")
    assert {r.seed for r in pilot_rows} == {42}
    assert {r.seed for r in scale_rows} == {99}


def test_combine_pilot_runs_unions_seeds(tmp_path: Path) -> None:
    pilot_cells = [
        (0.0, 42, {"slippery": (130, 0.88), "rough": (128, 0.91)}),
        (0.0, 7, {"slippery": (130, 0.88), "rough": (128, 0.97)}),
        (0.5, 42, {"slippery": (130, 0.92), "rough": (128, 0.92)}),
        (0.5, 7, {"slippery": (130, 0.91), "rough": (128, 0.88)}),
    ]
    scale_cells = [
        (0.0, 99, {"slippery": (130, 0.89), "rough": (128, 0.93)}),
        (0.0, 314, {"slippery": (130, 0.90), "rough": (128, 0.94)}),
        (0.5, 99, {"slippery": (130, 0.93), "rough": (128, 0.91)}),
        (0.5, 314, {"slippery": (130, 0.94), "rough": (128, 0.92)}),
    ]
    _scaffold_layout(tmp_path, "multiseed_pilot_2026-05-07", pilot_cells)
    _scaffold_layout(tmp_path, "multiseed_scale_2026-05-07", scale_cells)
    combined = combine_pilot_runs(
        (tmp_path, "multiseed_pilot_2026-05-07"),
        (tmp_path, "multiseed_scale_2026-05-07"),
    )
    assert combined.seeds == sorted([42, 7, 99, 314])
    # 4 seeds x 2 ff x 2 terrains = 16 rows.
    assert len(combined.rows) == 16
    assert combined.sources == [
        "multiseed_pilot_2026-05-07",
        "multiseed_scale_2026-05-07",
    ]


def test_combine_pilot_runs_dedups_on_overlap(tmp_path: Path) -> None:
    # Same (ff, seed, terrain) in both sweeps -> later spec wins.
    _scaffold_layout(
        tmp_path,
        "multiseed_pilot_2026-05-07",
        [(0.0, 42, {"slippery": (130, 0.10), "rough": (130, 0.10)})],
    )
    _scaffold_layout(
        tmp_path,
        "multiseed_scale_2026-05-07",
        [(0.0, 42, {"slippery": (130, 0.99), "rough": (130, 0.99)})],
    )
    combined = combine_pilot_runs(
        (tmp_path, "multiseed_pilot_2026-05-07"),
        (tmp_path, "multiseed_scale_2026-05-07"),
    )
    by_key = {(r.ff, r.seed, r.terrain): r.success_rate for r in combined.rows}
    assert by_key[(0.0, 42, "slippery")] == 0.99
    assert by_key[(0.0, 42, "rough")] == 0.99


# ---------------------------------------------------------------------------
# End-to-end n=7 analysis.
# ---------------------------------------------------------------------------


def test_run_combined_analysis_n7_clears_alpha_05(tmp_path: Path) -> None:
    """n=7 with all-positive deltas -> exact perm p = 2/128 < 0.05."""
    # Build pilot (3 seeds) + scale (4 seeds), all with +0.04 slippery
    # delta and +0.02 rough delta. n=7 paired permutation has 128
    # permutations; only the all-+ and all-- give |mean| >= obs, so
    # p = 2/128 = 0.015625 < 0.05.
    pilot_seeds = [42, 123, 7]
    scale_seeds = [99, 314, 1729, 2718]
    pilot_cells: list[tuple[float, int, dict[str, tuple[int, float]]]] = []
    scale_cells: list[tuple[float, int, dict[str, tuple[int, float]]]] = []
    for s in pilot_seeds:
        pilot_cells.append((0.0, s, {"slippery": (130, 0.85), "rough": (128, 0.90)}))
        pilot_cells.append((0.5, s, {"slippery": (130, 0.89), "rough": (128, 0.92)}))
    for s in scale_seeds:
        scale_cells.append((0.0, s, {"slippery": (130, 0.85), "rough": (128, 0.90)}))
        scale_cells.append((0.5, s, {"slippery": (130, 0.89), "rough": (128, 0.92)}))
    _scaffold_layout(tmp_path, "multiseed_pilot_2026-05-07", pilot_cells)
    _scaffold_layout(tmp_path, "multiseed_scale_2026-05-07", scale_cells)

    combined = combine_pilot_runs(
        (tmp_path, "multiseed_pilot_2026-05-07"),
        (tmp_path, "multiseed_scale_2026-05-07"),
    )
    report = run_combined_analysis(combined)
    assert report.n_seeds_per_terrain["slippery"] == 7
    assert report.n_seeds_per_terrain["rough"] == 7

    slip = next(d for d in report.deltas if d.terrain == "slippery")
    rough = next(d for d in report.deltas if d.terrain == "rough")

    # All seven deltas are exactly +0.04 (slippery) and +0.02 (rough).
    assert all(abs(x - 0.04) < 1e-9 for x in slip.deltas)
    assert all(abs(x - 0.02) < 1e-9 for x in rough.deltas)
    # Permutation p = 2 / 128 = 0.015625 on both terrains.
    assert abs(slip.permutation_p_two_sided - (2 / 128)) < 1e-9
    assert abs(rough.permutation_p_two_sided - (2 / 128)) < 1e-9
    assert slip.permutation_p_two_sided < 0.05
    assert rough.permutation_p_two_sided < 0.05

    # n=7 t-CI uses df=6 (t=2.447). With std=0 here, half-width=0.
    assert abs(slip.ci95_lo - slip.mean_delta) < 1e-9
    assert abs(slip.ci95_hi - slip.mean_delta) < 1e-9


def test_run_combined_analysis_n7_with_one_negative_seed(tmp_path: Path) -> None:
    """One negative seed out of seven raises p above 2/128."""
    pilot_cells = [
        (0.0, 42, {"slippery": (130, 0.85), "rough": (128, 0.90)}),
        (0.0, 123, {"slippery": (130, 0.85), "rough": (128, 0.90)}),
        (0.0, 7, {"slippery": (130, 0.85), "rough": (128, 0.90)}),
        (0.5, 42, {"slippery": (130, 0.89), "rough": (128, 0.92)}),
        (0.5, 123, {"slippery": (130, 0.89), "rough": (128, 0.92)}),
        # seed=7 has a negative slippery delta of -0.04.
        (0.5, 7, {"slippery": (130, 0.81), "rough": (128, 0.92)}),
    ]
    scale_cells = []
    for s in [99, 314, 1729, 2718]:
        scale_cells.append((0.0, s, {"slippery": (130, 0.85), "rough": (128, 0.90)}))
        scale_cells.append((0.5, s, {"slippery": (130, 0.89), "rough": (128, 0.92)}))
    _scaffold_layout(tmp_path, "multiseed_pilot_2026-05-07", pilot_cells)
    _scaffold_layout(tmp_path, "multiseed_scale_2026-05-07", scale_cells)
    combined = combine_pilot_runs(
        (tmp_path, "multiseed_pilot_2026-05-07"),
        (tmp_path, "multiseed_scale_2026-05-07"),
    )
    report = run_combined_analysis(combined)
    slip = next(d for d in report.deltas if d.terrain == "slippery")
    # 6 positives at +0.04, 1 negative at -0.04 -> mean = +0.0286.
    assert abs(slip.mean_delta - (6 * 0.04 - 0.04) / 7) < 1e-9
    # p must be above the floor at n=7 because flipping the one
    # negative seed gives |mean|=0.04 > obs=0.0286.
    assert slip.permutation_p_two_sided > (2 / 128)


# ---------------------------------------------------------------------------
# Markdown rendering.
# ---------------------------------------------------------------------------


def test_render_combined_markdown_contains_n7_floor_string(tmp_path: Path) -> None:
    pilot_cells = []
    scale_cells = []
    for s in [42, 123, 7]:
        pilot_cells.append((0.0, s, {"slippery": (130, 0.85), "rough": (128, 0.90)}))
        pilot_cells.append((0.5, s, {"slippery": (130, 0.89), "rough": (128, 0.92)}))
    for s in [99, 314, 1729, 2718]:
        scale_cells.append((0.0, s, {"slippery": (130, 0.85), "rough": (128, 0.90)}))
        scale_cells.append((0.5, s, {"slippery": (130, 0.89), "rough": (128, 0.92)}))
    _scaffold_layout(tmp_path, "multiseed_pilot_2026-05-07", pilot_cells)
    _scaffold_layout(tmp_path, "multiseed_scale_2026-05-07", scale_cells)
    combined = combine_pilot_runs(
        (tmp_path, "multiseed_pilot_2026-05-07"),
        (tmp_path, "multiseed_scale_2026-05-07"),
    )
    report = run_combined_analysis(combined)
    md = render_combined_markdown(report, title="Test verdict")
    assert "Test verdict" in md
    assert "n_seeds (slippery, paired) = 7" in md
    assert "p-floor = 2/128 = 0.0156" in md


# ---------------------------------------------------------------------------
# Direct paired_delta sanity at n=7 (t-multiplier hand-check).
# ---------------------------------------------------------------------------


def test_paired_delta_n7_uses_t_multiplier_2_447(tmp_path: Path) -> None:
    # Construct synthetic deltas with nontrivial std so the t-CI half
    # width is observable. Values drawn from a known distribution so
    # std and t*sem can be hand-computed.
    deltas_per_seed = {
        42: 0.05,
        123: 0.03,
        7: 0.04,
        99: 0.02,
        314: 0.06,
        1729: 0.025,
        2718: 0.045,
    }
    cells: list[tuple[float, int, dict[str, tuple[int, float]]]] = []
    for s, d in deltas_per_seed.items():
        cells.append((0.0, s, {"slippery": (130, 0.85), "rough": (128, 0.90)}))
        cells.append((0.5, s, {"slippery": (130, 0.85 + d), "rough": (128, 0.90)}))
    _scaffold_layout(tmp_path, "multiseed_pilot_2026-05-07", cells)
    rows = load_cells_with_prefix(tmp_path, "multiseed_pilot_2026-05-07")
    pd = paired_delta(rows, ff_a=0.0, ff_b=0.5, terrain="slippery")
    assert pd.n_seeds == 7
    # CI half-width should be ~ t(df=6) * sem = 2.447 * sem.
    expected_half = 2.447 * pd.sem_delta
    actual_half = (pd.ci95_hi - pd.ci95_lo) / 2.0
    assert abs(actual_half - expected_half) < 1e-9


# ---------------------------------------------------------------------------
# Regression guard on the PUBLISHED n=7 headline.
#
# Every other test in this file runs the pipeline on synthetic fixtures in
# tmp_path. None of them touch the committed results/ tree, so the published
# null verdict (slippery -1.10 pp / rough -1.58 pp, both p > 0.05) had no
# regression protection: a silently mutated result cell or a change to the
# paired-delta / sign-flip math would not fail any test. This test pins the
# headline to the raw metrics on disk so the central scientific claim cannot
# drift unnoticed. The numbers below were reproduced from the committed
# results/ tree on 2026-05-29 and match the vault verdict
# (MULTISEED_SCALE_2026-05-08.md).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "results"


def test_published_n7_headline_reproduces_from_committed_results() -> None:
    pilot = _RESULTS_DIR / "multiseed_pilot_2026-05-07_failure_fraction=0p0_seed=42"
    if not pilot.exists():
        pytest.skip(
            "committed multiseed results not present in this checkout; "
            "headline regression guard is data-dependent"
        )

    combined = combine_pilot_runs(
        (_RESULTS_DIR, "multiseed_pilot_2026-05-07"),
        (_RESULTS_DIR, "multiseed_scale_2026-05-07"),
    )
    report = run_combined_analysis(combined)

    by_terrain = {d.terrain: d for d in report.deltas}
    slip = by_terrain["slippery"]
    rough = by_terrain["rough"]

    # Both terrains are paired across all 7 seeds.
    assert slip.n_seeds == 7
    assert rough.n_seeds == 7

    # Slippery headline: mean -1.10 pp, 95% CI [-5.20, +3.01] pp,
    # 4/7 positive, exact sign-flip p = 0.5625, does NOT clear alpha=0.05.
    assert slip.mean_delta * 100 == pytest.approx(-1.10, abs=0.02)
    assert slip.ci95_lo * 100 == pytest.approx(-5.20, abs=0.05)
    assert slip.ci95_hi * 100 == pytest.approx(+3.01, abs=0.05)
    assert sum(1 for x in slip.deltas if x > 0) == 4
    assert slip.permutation_p_two_sided == pytest.approx(0.5625, abs=1e-4)
    assert slip.permutation_p_two_sided >= 0.05  # null verdict holds

    # Rough headline: mean -1.58 pp, 95% CI [-6.31, +3.15] pp,
    # 2/7 positive, exact sign-flip p = 0.3906, does NOT clear alpha=0.05.
    assert rough.mean_delta * 100 == pytest.approx(-1.58, abs=0.02)
    assert rough.ci95_lo * 100 == pytest.approx(-6.31, abs=0.05)
    assert rough.ci95_hi * 100 == pytest.approx(+3.15, abs=0.05)
    assert sum(1 for x in rough.deltas if x > 0) == 2
    assert rough.permutation_p_two_sided == pytest.approx(0.3906, abs=1e-4)
    assert rough.permutation_p_two_sided >= 0.05  # null verdict holds

    # The single largest negative slippery seed (1729, -8.85 pp) must still
    # be present; its removal would be the easiest way to fake a positive
    # result, so guard it explicitly.
    assert min(slip.deltas) * 100 == pytest.approx(-8.85, abs=0.05)
