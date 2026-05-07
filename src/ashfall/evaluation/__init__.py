"""Evaluation harness, metrics, and statistical-significance helpers."""

from ashfall.evaluation.significance import (
    ProportionDiffResult,
    bootstrap_diff_proportion,
    fishers_exact_p,
    holm_adjust,
    permutation_p_value,
    proportion_diff_test,
)

__all__ = [
    "ProportionDiffResult",
    "bootstrap_diff_proportion",
    "fishers_exact_p",
    "holm_adjust",
    "permutation_p_value",
    "proportion_diff_test",
]
