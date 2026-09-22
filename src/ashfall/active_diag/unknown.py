"""Is any library mechanism consistent with the evidence? (the UNKNOWN test)

Statistic: under the best library explanation (the MAP mechanism at its MAP value), compute the
six-channel one-step residual ``d^2`` (five state channels in the robust nominal scale plus the
action-consistency channel) on every evidence log, run the FCSI CUSUM on ``d^2 / 6 - 1 - drift``,
and take the largest peak over logs. Each row's ``d^2`` is capped at the 99.9% chi-square point
so that one outlier row (for example at a patch edge) cannot raise the statistic on its own. A
persistent excursion the best known model cannot remove is evidence that the library is
inadequate.

The threshold is **not** chosen here. It is calibrated per method on development cases with
known mechanisms only (the largest statistic seen on them), so on development data the false
UNKNOWN rate is zero by construction; the evaluation measures it on new cases.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ashfall.fcsi.divergence import cusum

from .posterior import action_residuals, rows_of, state_residuals

DIM = 6
Z_CAP = float(chi2.ppf(0.999, DIM))  # per-row cap: persistence needs a run of bad rows


REFINE_POINTS = 41


def refine(cfg, theta, base, w, evidence: list, scale):
    """The best library explanation at a fine parameter resolution (41 points within one coarse
    grid step either side of the MAP value), so the test is not failed by grid quantization."""
    from ashfall.fcsi.mechanisms import BY_NAME

    from .posterior import NONE, log_likelihoods

    if w.top == NONE:
        return base, None
    m = BY_NAME[w.top]
    step = m.span / 12
    vals = np.clip(
        np.linspace(w.map_value[w.top] - step, w.map_value[w.top] + step, REFINE_POINTS),
        m.low,
        m.high,
    )
    hyps = [(w.top, float(v)) for v in vals]
    ll = log_likelihoods(cfg, theta, base, hyps, evidence, scale)
    best = float(vals[int(np.argmax(ll))])
    from .posterior import model_for

    return model_for(base, (w.top, best)), best


def statistic(cfg, theta, model, evidence: list, scale) -> dict:
    peaks = []
    for ev in evidence:
        rows = rows_of(ev.log)
        rs = state_residuals(cfg, [model], ev.log, rows)[0]
        ra = action_residuals(cfg, theta, [model], ev.log, rows, ev.probe_force)[0]
        d2 = np.minimum(scale.d2(rs, ra), Z_CAP)
        peaks.append(float(cusum(d2, dim=DIM).max()) if len(d2) else 0.0)
    return {"peak": float(max(peaks)), "per_log": peaks}
