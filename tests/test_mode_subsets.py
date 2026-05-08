"""Tests for the 2026-05-08 mode-subset Stage-1 analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from ashfall.analysis.mode_subsets import (
    PILOT_SEEDS,
    SUBSET_ORDER,
    SubsetCellRaw,
    compare_subset_to_baseline,
    load_baseline_cells,
    load_subset_cells,
    render_stage1_markdown,
    run_stage1_analysis,
)
from ashfall.analysis.multiseed import CellRaw


def _write_metric(dir_path: Path, terrain: str, n: int, sr: float) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"metrics_{terrain}.json").write_text(
        json.dumps({"num_episodes": n, "success_rate": sr})
    )


def _scaffold_subset_layout(
    root: Path,
    cells: list[tuple[str, int, dict[str, tuple[int, float]]]],
    stamp: str = "2026-05-08_08-00-00",
) -> None:
    """Create a fake mode-subset pilot results tree under ``root``."""
    for subset, seed, metrics in cells:
        cell_dir = (
            root
            / f"failure_modes_pilot_2026-05-08_failure_modes={subset}_seed={seed}"
            / stamp
            / "metrics"
        )
        for terrain, (n, sr) in metrics.items():
            _write_metric(cell_dir, terrain, n, sr)


def _scaffold_baseline_layout(
    root: Path,
    cells: list[tuple[int, dict[str, tuple[int, float]]]],
    stamp: str = "2026-05-07_14-03-14",
) -> None:
    """Create a fake ff=0.0 multiseed_pilot results tree under ``root``."""
    for seed, metrics in cells:
        cell_dir = (
            root
            / f"multiseed_pilot_2026-05-07_failure_fraction=0p0_seed={seed}"
            / stamp
            / "metrics"
        )
        for terrain, (n, sr) in metrics.items():
            _write_metric(cell_dir, terrain, n, sr)


def test_load_subset_cells_round_trips(tmp_path: Path) -> None:
    """``load_subset_cells`` parses subset, seed, terrain, n, sr correctly."""
    _scaffold_subset_layout(
        tmp_path,
        [
            ("slip_only", 42, {"rough": (128, 0.91), "slippery": (130, 0.93)}),
            ("slip_only", 123, {"rough": (130, 0.88), "slippery": (132, 0.90)}),
            ("severe_only", 42, {"rough": (128, 0.85), "slippery": (132, 0.80)}),
        ],
    )
    rows = load_subset_cells(tmp_path)
    # 3 cells x 2 terrains
    assert len(rows) == 6
    by_key = {(r.subset, r.seed, r.terrain): r for r in rows}
    assert by_key[("slip_only", 42, "slippery")].success_rate == 0.93
    assert by_key[("severe_only", 42, "rough")].n_episodes == 128
    # k = round(sr * n)
    assert by_key[("slip_only", 123, "rough")].k == round(0.88 * 130)


def test_load_baseline_cells_filters_to_ff0(tmp_path: Path) -> None:
    """``load_baseline_cells`` keeps only ff=0.0 entries from multiseed_pilot."""
    # Scaffold both ff=0.0 and ff=0.5 entries; expect only ff=0.0 to come back.
    _scaffold_baseline_layout(
        tmp_path,
        [
            (42, {"rough": (130, 0.91), "slippery": (134, 0.89)}),
            (123, {"rough": (128, 0.88), "slippery": (130, 0.86)}),
        ],
    )
    # Add an ff=0.5 cell at the same layout level.
    cell_dir = (
        tmp_path
        / "multiseed_pilot_2026-05-07_failure_fraction=0p5_seed=42"
        / "2026-05-07_14-03-14"
        / "metrics"
    )
    _write_metric(cell_dir, "rough", 130, 0.95)
    _write_metric(cell_dir, "slippery", 130, 0.94)

    rows = load_baseline_cells(tmp_path)
    assert all(r.ff == 0.0 for r in rows)
    # 2 ff=0.0 seeds x 2 terrains.
    assert len(rows) == 4


def test_compare_subset_to_baseline_pairs_by_seed(tmp_path: Path) -> None:
    """Subset success - baseline success per seed; only matching seeds count."""
    subset_rows = [
        SubsetCellRaw("slip_only", 42, "slippery", 128, 0.93),
        SubsetCellRaw("slip_only", 123, "slippery", 130, 0.90),
        SubsetCellRaw("slip_only", 7, "slippery", 132, 0.88),
    ]
    baseline_rows = [
        CellRaw(0.0, 42, "slippery", 134, 0.89),
        CellRaw(0.0, 123, "slippery", 130, 0.86),
        CellRaw(0.0, 7, "slippery", 132, 0.91),
    ]
    cmp = compare_subset_to_baseline(
        subset_rows,
        baseline_rows,
        subset="slip_only",
        terrain="slippery",
    )
    assert cmp.n_seeds == 3
    assert cmp.seeds == [7, 42, 123]
    # Per-seed deltas (subset - baseline) in the order seeds are sorted.
    expected_deltas = [0.88 - 0.91, 0.93 - 0.89, 0.90 - 0.86]
    for actual, expected in zip(cmp.deltas, expected_deltas):
        assert abs(actual - expected) < 1e-9
    assert cmp.n_positive == 2
    assert cmp.n_negative == 1


def test_one_seed_sign_flip_does_not_qualify_as_candidate(tmp_path: Path) -> None:
    """Synthetic case: 2/3 positive + 1 negative slippery delta is NOT
    a Stage-2 candidate. Stage-2 requires 3/3 strictly positive.
    """
    subset_rows = [
        SubsetCellRaw("slip_only", 42, "slippery", 128, 0.93),  # +0.04 vs 0.89
        SubsetCellRaw("slip_only", 123, "slippery", 130, 0.90),  # +0.04 vs 0.86
        SubsetCellRaw("slip_only", 7, "slippery", 132, 0.88),  # -0.03 vs 0.91 (FLIP)
        # Need rough rows so Stage1 doesn't crash; doesn't affect this assertion.
        SubsetCellRaw("slip_only", 42, "rough", 128, 0.91),
        SubsetCellRaw("slip_only", 123, "rough", 130, 0.91),
        SubsetCellRaw("slip_only", 7, "rough", 132, 0.91),
    ]
    baseline_rows = [
        CellRaw(0.0, 42, "slippery", 134, 0.89),
        CellRaw(0.0, 123, "slippery", 130, 0.86),
        CellRaw(0.0, 7, "slippery", 132, 0.91),
        CellRaw(0.0, 42, "rough", 130, 0.91),
        CellRaw(0.0, 123, "rough", 130, 0.91),
        CellRaw(0.0, 7, "rough", 130, 0.91),
    ]
    report = run_stage1_analysis(
        subset_rows,
        baseline_rows,
        subsets=("slip_only",),
    )
    slip_slippery = next(
        c for c in report.comparisons if c.subset == "slip_only" and c.terrain == "slippery"
    )
    assert slip_slippery.n_positive == 2
    assert slip_slippery.n_negative == 1
    assert not slip_slippery.all_positive
    assert "slip_only" not in report.stage2_candidates
    assert "slip_only" not in report.clear_losers


def test_three_three_positive_qualifies_as_candidate() -> None:
    """3/3 strictly positive slippery deltas marks a subset as Stage-2 candidate."""
    subset_rows = [
        SubsetCellRaw("slip_plus_cm", 42, "slippery", 128, 0.95),
        SubsetCellRaw("slip_plus_cm", 123, "slippery", 130, 0.92),
        SubsetCellRaw("slip_plus_cm", 7, "slippery", 132, 0.95),
        SubsetCellRaw("slip_plus_cm", 42, "rough", 128, 0.91),
        SubsetCellRaw("slip_plus_cm", 123, "rough", 130, 0.91),
        SubsetCellRaw("slip_plus_cm", 7, "rough", 132, 0.91),
    ]
    baseline_rows = [
        CellRaw(0.0, 42, "slippery", 134, 0.89),
        CellRaw(0.0, 123, "slippery", 130, 0.86),
        CellRaw(0.0, 7, "slippery", 132, 0.91),
        CellRaw(0.0, 42, "rough", 130, 0.91),
        CellRaw(0.0, 123, "rough", 130, 0.91),
        CellRaw(0.0, 7, "rough", 130, 0.91),
    ]
    report = run_stage1_analysis(
        subset_rows,
        baseline_rows,
        subsets=("slip_plus_cm",),
    )
    cmp = next(
        c for c in report.comparisons if c.subset == "slip_plus_cm" and c.terrain == "slippery"
    )
    assert cmp.all_positive
    assert "slip_plus_cm" in report.stage2_candidates


def test_three_three_negative_qualifies_as_clear_loser() -> None:
    """3/3 strictly negative slippery deltas marks a subset as a clear loser."""
    subset_rows = [
        SubsetCellRaw("severe_only", 42, "slippery", 128, 0.80),
        SubsetCellRaw("severe_only", 123, "slippery", 130, 0.78),
        SubsetCellRaw("severe_only", 7, "slippery", 132, 0.85),
        SubsetCellRaw("severe_only", 42, "rough", 128, 0.85),
        SubsetCellRaw("severe_only", 123, "rough", 130, 0.85),
        SubsetCellRaw("severe_only", 7, "rough", 132, 0.85),
    ]
    baseline_rows = [
        CellRaw(0.0, 42, "slippery", 134, 0.89),
        CellRaw(0.0, 123, "slippery", 130, 0.86),
        CellRaw(0.0, 7, "slippery", 132, 0.91),
        CellRaw(0.0, 42, "rough", 130, 0.91),
        CellRaw(0.0, 123, "rough", 130, 0.91),
        CellRaw(0.0, 7, "rough", 130, 0.91),
    ]
    report = run_stage1_analysis(
        subset_rows,
        baseline_rows,
        subsets=("severe_only",),
    )
    assert "severe_only" in report.clear_losers
    assert "severe_only" not in report.stage2_candidates


def test_render_stage1_markdown_smoke() -> None:
    """Markdown render emits subset rows, headline, and per-terrain delta tables."""
    subset_rows = [
        SubsetCellRaw("all_modes", 42, "slippery", 128, 0.92),
        SubsetCellRaw("all_modes", 42, "rough", 128, 0.91),
    ]
    baseline_rows = [
        CellRaw(0.0, 42, "slippery", 134, 0.89),
        CellRaw(0.0, 42, "rough", 130, 0.91),
    ]
    report = run_stage1_analysis(
        subset_rows,
        baseline_rows,
        subsets=("all_modes",),
        seeds=(42,),
    )
    md = render_stage1_markdown(report)
    assert "Stage-2 candidates" in md
    assert "Clear losers" in md
    assert "all_modes" in md
    assert "## 18-cell raw success rates" in md
    assert "## Per-subset paired delta vs ff=0.0 baseline (slippery)" in md
    assert "## Per-subset paired delta vs ff=0.0 baseline (rough)" in md


def test_pilot_constants_are_consistent() -> None:
    """SUBSET_ORDER and PILOT_SEEDS match the run-doc spec."""
    assert SUBSET_ORDER == (
        "all_modes",
        "slip_only",
        "command_mismatch_only",
        "slip_plus_cm",
        "severe_only",
        "severe_plus_slip",
    )
    assert PILOT_SEEDS == (42, 123, 7)
