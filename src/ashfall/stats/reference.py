"""Independent reference implementations used only to guard the production code.

Nothing here is called by the analysis pipeline. Each function recomputes a
statistic from its textbook definition by a different route than the
production module, so a sign or normalisation error in production is caught
by a test rather than by a reviewer.
"""

from __future__ import annotations

import numpy as np


def reference_two_sample_bca_acceleration(a, b) -> float:
    """BCa acceleration for ``theta = mean(b) - mean(a)`` from the jackknife definition.

    Efron and Tibshirani (1993), eq. 14.15: with jackknife replicates
    ``theta_(i)`` obtained by leaving out one observation at a time, and
    ``U_i = theta_(.) - theta_(i)`` their deviations from the jackknife mean,

        a_hat = sum(U_i ** 3) / (6 * (sum(U_i ** 2)) ** 1.5).

    Here the leave-one-out runs over the POOLED observations: removing ``a[i]``
    recomputes ``mean(b) - mean(a without a[i])`` and removing ``b[j]``
    recomputes ``mean(b without b[j]) - mean(a)``. The statistic is evaluated
    literally on the reduced arrays, never through a closed form, so this
    function shares no algebra with ``ashfall.evaluation.significance``. Its
    sign is fixed by the definition of ``U_i``; the historical production bug
    was exactly that sign.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 2:
        raise ValueError("two one-dimensional samples of at least two observations required")

    def statistic(x, y):
        return float(np.mean(y) - np.mean(x))

    replicates = [statistic(np.delete(a, i), b) for i in range(len(a))]
    replicates += [statistic(a, np.delete(b, j)) for j in range(len(b))]
    theta = np.asarray(replicates)
    influence = theta.mean() - theta
    denominator = 6.0 * float(np.sum(influence**2)) ** 1.5
    return float(np.sum(influence**3)) / denominator if denominator > 0 else 0.0
