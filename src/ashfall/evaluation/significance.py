"""Statistical significance helpers for binomial-outcome experiments.

The Ashfall eval harness emits aggregate counts only: ``num_episodes`` and
``success_rate``. Per-episode trajectories are not retained, so
between-cell tests work on the binomial summary
``(k_a, n_a, k_b, n_b)``. This module provides:

- :func:`bernoulli_arrays` to reconstruct the implied 0/1 success
  indicator vectors from the summary counts.
- :func:`bootstrap_diff_proportion` for a 95% bootstrap CI on the
  difference in success rate (BCa-adjusted).
- :func:`permutation_p_value` for a two-sided permutation test on
  the same difference.
- :func:`fishers_exact_p` as a closed-form check (and the right
  answer for very small n where the bootstrap is noisy).
- :func:`holm_adjust` for Holm-Bonferroni multiple-comparison
  control across a family of comparisons.

All helpers are pure numpy/scipy. No torch import. They are safe to
call from no-sim CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy import stats as _scipy_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover, scipy is a hard project dep
    _HAS_SCIPY = False


@dataclass
class ProportionDiffResult:
    """Outcome of a paired-bootstrap proportion-difference test."""

    p_a: float
    p_b: float
    diff: float  # p_b - p_a
    ci_lower: float
    ci_upper: float
    p_value: float
    n_a: int
    n_b: int
    method: str  # "bootstrap+permutation" | "fisher" | etc.

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "p_a": self.p_a,
            "p_b": self.p_b,
            "diff": self.diff,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "method": self.method,
        }


def bernoulli_arrays(k_a: int, n_a: int, k_b: int, n_b: int) -> tuple[np.ndarray, np.ndarray]:
    """Build 0/1 indicator arrays with the requested success counts.

    Used to feed bootstrap and permutation routines that expect raw
    indicators rather than counts.
    """
    if not (0 <= k_a <= n_a and 0 <= k_b <= n_b):
        raise ValueError(f"Invalid counts: k_a={k_a} n_a={n_a} k_b={k_b} n_b={n_b}")
    a = np.zeros(n_a, dtype=np.int8)
    a[:k_a] = 1
    b = np.zeros(n_b, dtype=np.int8)
    b[:k_b] = 1
    return a, b


def bootstrap_diff_proportion(
    k_a: int,
    n_a: int,
    k_b: int,
    n_b: int,
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    method: str = "bca",
) -> tuple[float, float, float]:
    """Bootstrap CI for ``p_b - p_a`` from binomial summary counts.

    Parameters
    ----------
    k_a, n_a, k_b, n_b : int
        Successes / totals for the two conditions.
    n_bootstrap : int
        Resamples for both arms.
    alpha : float
        Two-sided coverage; 0.05 -> 95% CI.
    seed : int
        RNG seed.
    method : str
        "bca" (default, bias-corrected accelerated) or "percentile".

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    if n_a == 0 or n_b == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    a, b = bernoulli_arrays(k_a, n_a, k_b, n_b)
    point = float(b.mean() - a.mean())

    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sa = a[rng.integers(0, n_a, size=n_a)]
        sb = b[rng.integers(0, n_b, size=n_b)]
        diffs[i] = sb.mean() - sa.mean()

    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)

    if method == "percentile":
        lo = float(np.percentile(diffs, lo_q))
        hi = float(np.percentile(diffs, hi_q))
        return point, lo, hi

    if method != "bca":
        raise ValueError(f"unknown method {method!r}")

    # Bias-correction term z0.
    frac_below = float(np.mean(diffs < point))
    if frac_below in (0.0, 1.0):
        # Distribution is degenerate (e.g. all bootstrap diffs equal),
        # fall back to percentile.
        lo = float(np.percentile(diffs, lo_q))
        hi = float(np.percentile(diffs, hi_q))
        return point, lo, hi
    z0 = _norm_ppf(frac_below)

    # Acceleration: jackknife each arm.
    jacks = []
    for arr in (a, b):
        n = len(arr)
        means = np.empty(n)
        s = arr.sum()
        for i in range(n):
            means[i] = (s - arr[i]) / (n - 1)
        jacks.append(means)
    jack_a, jack_b = jacks
    # Pseudo-jackknife on the difference: for each (i,j) we'd need a 2D
    # grid which is expensive. Standard approach for two-sample BCa is
    # to jackknife the joint pseudovalue (Efron & Tibshirani 1993, p.187).
    # We approximate by stacking jackknife replicates of each arm and
    # using the pooled influence values, which is a standard practical
    # shortcut and recovers the right acceleration sign.
    pseudo = np.concatenate([jack_a.mean() - jack_a, jack_b - jack_b.mean()])
    num = float(np.sum(pseudo**3))
    den = 6.0 * (float(np.sum(pseudo**2)) ** 1.5)
    a_accel = num / den if den > 0 else 0.0

    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)
    a1 = _norm_cdf(z0 + (z0 + z_lo) / (1 - a_accel * (z0 + z_lo)))
    a2 = _norm_cdf(z0 + (z0 + z_hi) / (1 - a_accel * (z0 + z_hi)))
    lo = float(np.percentile(diffs, 100 * a1))
    hi = float(np.percentile(diffs, 100 * a2))
    return point, lo, hi


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    if _HAS_SCIPY:
        return float(_scipy_stats.norm.ppf(q))
    # Beasley-Springer-Moro fallback (good enough for CI quantiles).
    if not 0 < q < 1:
        raise ValueError("q must be in (0,1)")
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    p_low = 0.02425
    p_high = 1 - p_low
    if q < p_low:
        r = math.sqrt(-2 * math.log(q))
        return ((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5] / (
            (((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1
        )
    if q <= p_high:
        r = q - 0.5
        s = r * r
        return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (
            ((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1
        )
    r = math.sqrt(-2 * math.log(1 - q))
    return -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / (
        (((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1
    )


def permutation_p_value(
    k_a: int,
    n_a: int,
    k_b: int,
    n_b: int,
    *,
    n_perm: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test on the difference of proportions.

    Pools the two arms' Bernoulli arrays, reshuffles, and counts how
    often the resampled |diff| matches or exceeds the observed |diff|.
    """
    if n_a == 0 or n_b == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    a, b = bernoulli_arrays(k_a, n_a, k_b, n_b)
    pooled = np.concatenate([a, b])
    obs = abs(b.mean() - a.mean())
    if obs == 0:
        return 1.0
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        sa = pooled[:n_a]
        sb = pooled[n_a:]
        if abs(sb.mean() - sa.mean()) >= obs:
            hits += 1
    # Add-one smoothing so p never hits exactly 0.
    return (hits + 1) / (n_perm + 1)


def fishers_exact_p(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """Two-sided Fisher's exact test on the 2x2 contingency table.

    Closed-form, no resampling. Matches the permutation p-value to
    within Monte-Carlo noise for moderate n.
    """
    if not _HAS_SCIPY:
        return float("nan")
    table = np.array([[k_a, n_a - k_a], [k_b, n_b - k_b]])
    try:
        _, p = _scipy_stats.fisher_exact(table, alternative="two-sided")
    except Exception:
        return float("nan")
    return float(p)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment.

    Returns adjusted p-values in the original input order, monotone
    non-decreasing, capped at 1.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (n - rank) * p_values[idx]
        running = max(running, scaled)
        adj[idx] = min(1.0, running)
    return adj


def proportion_diff_test(
    k_a: int,
    n_a: int,
    k_b: int,
    n_b: int,
    *,
    n_bootstrap: int = 10_000,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ProportionDiffResult:
    """One-shot wrapper: bootstrap CI + permutation p-value, plus Fisher cross-check."""
    p_a = (k_a / n_a) if n_a else 0.0
    p_b = (k_b / n_b) if n_b else 0.0
    diff, lo, hi = bootstrap_diff_proportion(
        k_a, n_a, k_b, n_b,
        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, method="bca",
    )
    perm_p = permutation_p_value(k_a, n_a, k_b, n_b, n_perm=n_perm, seed=seed)
    fisher_p = fishers_exact_p(k_a, n_a, k_b, n_b)
    # Use Fisher as primary when scipy is available (closed-form, exact).
    # Permutation is the bootstrap-consistent backup.
    p_value = fisher_p if not math.isnan(fisher_p) else perm_p
    if not math.isnan(fisher_p):
        method = "fisher_exact+bca_bootstrap"
    else:
        method = "permutation+bca_bootstrap"
    return ProportionDiffResult(
        p_a=p_a, p_b=p_b, diff=diff,
        ci_lower=lo, ci_upper=hi,
        p_value=p_value, n_a=n_a, n_b=n_b,
        method=method,
    )
