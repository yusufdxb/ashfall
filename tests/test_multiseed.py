"""Tests for the 2026-05-07 multi-seed pilot analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from ashfall.analysis.multiseed import (
    CellRaw,
    _exact_sign_flip_p,
    load_cells,
    paired_delta,
    per_ff_summary,
    render_markdown,
    run,
)


def _write_metric(dir_path: Path, terrain: str, n: int, sr: float) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"metrics_{terrain}.json").write_text(
        json.dumps({"num_episodes": n, "success_rate": sr})
    )


def _scaffold_pilot_layout(
    root: Path,
    cells: list[tuple[float, int, dict[str, tuple[int, float]]]],
    stamp: str = "2026-05-07_14-00-00",
) -> None:
    """Create a fake multiseed pilot results tree under ``root``.

    cells: list of (ff, seed, {terrain: (n, sr)}) tuples.
    """
    for ff, seed, metrics in cells:
        ff_short = (str(ff)).replace(".", "p")
        cell_dir = (
            root
            / f"multiseed_pilot_2026-05-07_failure_fraction={ff_short}_seed={seed}"
            / stamp
            / "metrics"
        )
        for terrain, (n, sr) in metrics.items():
            _write_metric(cell_dir, terrain, n, sr)


def test_load_cells_round_trips(tmp_path: Path) -> None:
    cells_in = [
        (0.0, 42, {"rough": (130, 0.91), "slippery": (134, 0.89)}),
        (0.0, 7, {"rough": (128, 0.88), "slippery": (130, 0.86)}),
        (0.5, 42, {"rough": (132, 0.93), "slippery": (131, 0.94)}),
    ]
    _scaffold_pilot_layout(tmp_path, cells_in)
    rows = load_cells(tmp_path)
    # 3 cells x 2 terrains = 6 rows.
    assert len(rows) == 6
    by_key = {(r.ff, r.seed, r.terrain): r for r in rows}
    assert by_key[(0.5, 42, "slippery")].success_rate == 0.94
    assert by_key[(0.0, 7, "rough")].n_episodes == 128
    # k = round(sr * n)
    assert by_key[(0.0, 7, "rough")].k == round(0.88 * 128)


def test_per_ff_summary_mean_se(tmp_path: Path) -> None:
    cells_in = [
        (0.0, 42, {"slippery": (100, 0.80)}),
        (0.0, 123, {"slippery": (100, 0.84)}),
        (0.0, 7, {"slippery": (100, 0.82)}),
    ]
    _scaffold_pilot_layout(tmp_path, cells_in)
    rows = load_cells(tmp_path)
    # Need to fill in rough too for load_cells to pick the dir up.
    # load_cells requires both terrains; add minimal rough metrics.
    cells_in_rough = [
        (0.0, 42, {"rough": (100, 0.80)}),
        (0.0, 123, {"rough": (100, 0.80)}),
        (0.0, 7, {"rough": (100, 0.80)}),
    ]
    _scaffold_pilot_layout(tmp_path, cells_in_rough)
    rows = load_cells(tmp_path)
    summaries = per_ff_summary(rows)
    # 1 ff x 2 terrains.
    assert len(summaries) == 2
    slip = next(s for s in summaries if s.terrain == "slippery")
    assert abs(slip.mean - 0.82) < 1e-9
    # Sample std with ddof=1 of [0.80, 0.84, 0.82] = 0.02 exactly.
    assert abs(slip.std - 0.02) < 1e-9


def test_exact_sign_flip_p_n3() -> None:
    # With n=3 paired deltas, sign-flip enumerates 2**3 = 8 permutations.
    # Statistic = |mean(deltas)|. For deltas = [0.05, 0.03, 0.04],
    # obs = 0.04. Two sign-flips produce |mean| >= obs: the all-+
    # original (mean = +0.04) and its all- mirror (mean = -0.04).
    # So p = 2/8 = 0.25. This is the smallest two-sided p-value
    # achievable with n=3 sign-flip permutations.
    p = _exact_sign_flip_p([0.05, 0.03, 0.04])
    assert abs(p - 2 / 8) < 1e-9


def test_exact_sign_flip_p_n3_minimum_is_one_quarter() -> None:
    # Sanity: even with arbitrarily large effect, two-sided sign-flip
    # p with n=3 cannot drop below 1/4 (paired mirror). This is the
    # statistical-power ceiling of an n=3 paired pilot, and the reason
    # the verdict text needs to discuss CI overlap rather than relying
    # on the p-value alone.
    p = _exact_sign_flip_p([10.0, 10.0, 10.0])
    assert abs(p - 0.25) < 1e-9


def test_exact_sign_flip_p_zero_mean() -> None:
    # Zero observed delta -> p = 1.0 by definition (no signal).
    p = _exact_sign_flip_p([0.0, 0.0, 0.0])
    assert p == 1.0


def test_paired_delta_pipeline(tmp_path: Path) -> None:
    cells_in = [
        (0.0, 42, {"slippery": (130, 0.88), "rough": (128, 0.90)}),
        (0.0, 123, {"slippery": (130, 0.86), "rough": (128, 0.91)}),
        (0.0, 7, {"slippery": (130, 0.84), "rough": (128, 0.89)}),
        (0.5, 42, {"slippery": (130, 0.93), "rough": (128, 0.92)}),
        (0.5, 123, {"slippery": (130, 0.91), "rough": (128, 0.94)}),
        (0.5, 7, {"slippery": (130, 0.89), "rough": (128, 0.91)}),
    ]
    _scaffold_pilot_layout(tmp_path, cells_in)
    rows, summaries, deltas = run(tmp_path)
    assert len(rows) == 12
    # 2 ff x 2 terrain = 4 summaries.
    assert len(summaries) == 4
    # Two terrains x one paired comparison.
    assert len(deltas) == 2
    slip = next(d for d in deltas if d.terrain == "slippery")
    # All three seeds give +0.05 lift -> mean = 0.05. With n=3 sign-flip
    # permutation, the minimum two-sided p is 2/8 = 0.25 (paired mirror).
    assert all(abs(x - 0.05) < 1e-9 for x in slip.deltas)
    assert abs(slip.mean_delta - 0.05) < 1e-9
    assert abs(slip.permutation_p_two_sided - 0.25) < 1e-9
    # Render produces a non-empty markdown body covering all sections.
    md = render_markdown(rows, summaries, deltas)
    assert "Per-cell raw success rates" in md
    assert "Per-ff cross-seed summary" in md
    assert "Paired-by-seed delta" in md


def test_paired_delta_handles_missing_pair(tmp_path: Path) -> None:
    # Only one seed common between ff_a and ff_b -> n_seeds = 1.
    cells_in = [
        (0.0, 42, {"slippery": (130, 0.88), "rough": (128, 0.90)}),
        (0.5, 42, {"slippery": (130, 0.93), "rough": (128, 0.92)}),
        (0.5, 123, {"slippery": (130, 0.91), "rough": (128, 0.94)}),
    ]
    _scaffold_pilot_layout(tmp_path, cells_in)
    rows = load_cells(tmp_path)
    d = paired_delta(rows, ff_a=0.0, ff_b=0.5, terrain="slippery")
    assert d.n_seeds == 1
    assert abs(d.mean_delta - 0.05) < 1e-9


def test_load_cells_skips_directories_missing_metrics(tmp_path: Path) -> None:
    # Build a stamp dir but no metrics_*.json -> cell should be skipped.
    bare = (
        tmp_path
        / "multiseed_pilot_2026-05-07_failure_fraction=0p0_seed=42"
        / "2026-05-07_14-00-00"
        / "metrics"
    )
    bare.mkdir(parents=True)
    rows = load_cells(tmp_path)
    assert rows == []


def test_cellraw_k_round_trip() -> None:
    r = CellRaw(ff=0.5, seed=42, terrain="slippery", n_episodes=131, success_rate=0.9389)
    assert r.k == round(0.9389 * 131)
