"""Scores any calibrated simulator against the hidden truth of a toy instance.

Every method's output is reduced to one :class:`~ashfall.fcsi.toy_world.Physics` (FCSI: its
accepted top hypothesis, or the unpatched nominal model if it abstained; DROPO-like: the
distribution mean) and scored the same way:

* held-out failure prediction: absolute error between the simulator's failure rate and the
  real one on held-out conditions (common random numbers, 256 episodes each), overall and on
  the *discrepant* subset where reality and the nominal model differ by >= 0.15;
* failure reproduction on the target and validation logs (event probability);
* nominal fidelity on held-out nominal logs (one-step residual d^2 per row, and success
  reproduction);
* identification: implied mechanism, and sparsity (parameters moved by > 5% of their span).
"""

from __future__ import annotations

import numpy as np

from ashfall.fcsi import divergence as dv
from ashfall.fcsi import objective as ob
from ashfall.fcsi.mechanisms import LIBRARY
from ashfall.fcsi.toy_world import MAX_DELAY, Physics, failure_rate

DISCREPANT = 0.15


def changed_params(physics: Physics) -> dict:
    vals = dict(physics.values)
    return {m.name: vals[m.name] for m in LIBRARY if m.normalized_change(vals[m.name]) > 0.05}


def failure_rates(cfg, theta, physics: Physics, conditions: list, n=256) -> list:
    return [failure_rate(cfg, theta, physics, c, n=n) for c in conditions]


def score(cfg, theta, physics: Physics, inst: dict, cal: dv.Calibration) -> dict:
    """``inst`` carries real rates, logs and conditions (see ``toy_study.build_instance``)."""
    sim = np.array(failure_rates(cfg, theta, physics, inst["heldout_conditions"]))
    real = np.array(inst["heldout_real_rates"])
    nominal = np.array(inst["heldout_nominal_rates"])
    disc = np.abs(real - nominal) >= DISCREPANT
    ev = inst["evidence"]
    out = {
        "heldout_mae": float(np.mean(np.abs(sim - real))),
        "discrepant_mae": float(np.mean(np.abs(sim - real)[disc])) if disc.any() else None,
        "discrepant_n": int(disc.sum()),
        "heldout_sim_rates": sim.tolist(),
        "target_p_event": float(
            ob.event_probability(cfg, theta, [physics], ev.failure_log, inst["event_row"])[0]
        ),
        "target_condition_rate": failure_rate(cfg, theta, physics, ev.failure_log.cond),
        "changed": changed_params(physics),
        "n_changed": len(changed_params(physics)),
    }
    if ev.validation_log is not None:
        out["validation_p_event"] = float(
            ob.event_probability(
                cfg, theta, [physics], ev.validation_log, inst["validation_event_row"]
            )[0]
        )
    d2 = []
    for lg in inst["heldout_nominal_logs"]:
        rows = np.arange(MAX_DELAY, len(lg) - 1)
        d2.append(2 * ob.nll(cfg, [physics], lg, rows, cal.scale)[0] / len(rows))
    out["nominal_d2_per_row"] = float(np.mean(d2))
    out["nominal_success"] = float(
        ob.nominal_success(cfg, theta, [physics], inst["heldout_nominal_logs"])[0]
    )
    return out
