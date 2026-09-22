"""Expected information gain of a probe about which mechanism is at work.

Monte Carlo estimate in the same currency as the belief update: for each mechanism ``i`` in the
discrimination set (at its MAP parameter), ``EIG_DRAWS`` simulated probe logs (with measurement
noise) are scored by every mechanism in the set with the tempered residual likelihood of
:mod:`ashfall.active_diag.posterior`; the resulting posterior gives

    EIG(u) = sum_i pi_i E_{Y ~ i}[ log pi'(i | Y) ] - sum_i pi_i log pi_i.

``pi`` is the current weight vector restricted to the discrimination set (plausible mechanisms
plus ``none``), mixed with ``FLOOR`` so that a probe separating the leading hypothesis from the
nominal model still scores when one hypothesis dominates (a confirmation probe).

Considered and not used: pairwise Bhattacharyya separation of predicted trajectories. The update
scores one-step reset residuals, not trajectories, so a probe can separate trajectories without
separating likelihoods; the estimate here scores exactly what the update will see. The simulated
logs are reused from the safety check (no extra rollouts).
"""

from __future__ import annotations

import numpy as np

from .posterior import EvidenceLog, log_likelihoods
from .prediction import to_log

EIG_DRAWS = 4
FLOOR = 0.05


def prior(weights: dict, labels: list) -> np.ndarray:
    w = np.array([weights.get(lb, 0.0) for lb in labels])
    w = w / w.sum() if w.sum() > 0 else np.full(len(labels), 1 / len(labels))
    w = (1 - FLOOR) * w + FLOOR / len(labels)
    return w / w.sum()


def expected_information_gain(
    cfg, theta, base, probe, hyps, first_rows, rollouts, scale, temperature, pi, rng, budget=None
) -> float:
    """``hyps[i]``: (mechanism, MAP value) of set member i; ``first_rows[i]``: index of its first
    simulated draw in ``rollouts``."""
    total = 0.0
    for i, r0 in enumerate(first_rows):
        acc = 0.0
        for s in range(EIG_DRAWS):
            lg, u = to_log(cfg, rollouts, r0 + s, probe, rng)
            ll = log_likelihoods(cfg, theta, base, hyps, [EvidenceLog(lg, u)], scale, budget)
            x = np.log(pi) + ll / temperature
            x = x - x.max()
            post = np.exp(x) / np.exp(x).sum()
            acc += np.log(max(post[i], 1e-300))
        total += pi[i] * acc / EIG_DRAWS
    return float(total - np.sum(pi * np.log(pi)))
