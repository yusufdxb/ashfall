"""Explicit probe admissibility, checked before any information score is computed.

A probe is admissible only if, under **every** plausible hypothesis (weight >= ``PLAUSIBLE``,
always including the unchanged nominal model), simulated with ``DRAWS`` process-noise draws at
the hypothesis's MAP parameter and at the next most likely grid value, the fraction of draws that
break a bound is at most ``MAX_VIOLATION_RATE``. Bounds (toy): tilt <= 0.35 rad (the fall angle
is 0.6), no fall, cart speed <= 2.0 m/s, probe force <= 8 N (enforced by the probe space).
Safety is a hard filter, never a penalty inside the information objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .prediction import ABORT_TILT, run
from .probe_space import MAX_AMPLITUDE, Probe

PLAUSIBLE = 0.02
DRAWS = 32
MAX_VIOLATION_RATE = 1.0 / 32
MAX_SPEED = 2.0
SAFETY_SEED = 950_000


@dataclass
class SafetyVerdict:
    admissible: bool
    worst_model: str
    worst_rate: float
    per_model: dict

    def to_dict(self) -> dict:
        return asdict(self)


def violations(r) -> np.ndarray:
    return r.fell | (r.max_tilt > ABORT_TILT) | (r.max_speed > MAX_SPEED)


def check_many(cfg, theta, probes: list[Probe], models: dict, budget=None):
    """``models``: label -> Physics. Returns ``{probe: SafetyVerdict}`` and the raw rollouts.

    All (probe, model, draw) rollouts of one probe run in one batch; the rollouts are returned
    so the information step can reuse them (their cost is charged once, here).
    """
    labels = list(models)
    seeds = np.arange(DRAWS) + SAFETY_SEED
    verdicts, rollouts = {}, {}
    for pr in probes:
        if abs(pr.amplitude) > MAX_AMPLITUDE:
            verdicts[pr] = SafetyVerdict(False, "amplitude", 1.0, {})
            continue
        phys = [models[lb] for lb in labels for _ in seeds]
        r = run(cfg, theta, phys, pr, np.tile(seeds, len(labels)))
        if budget is not None:
            budget.add(len(phys) * r.steps, 0)
        rates = violations(r).reshape(len(labels), DRAWS).mean(axis=1)
        worst = int(np.argmax(rates))
        verdicts[pr] = SafetyVerdict(
            bool(rates.max() <= MAX_VIOLATION_RATE + 1e-12),
            labels[worst],
            float(rates[worst]),
            {lb: float(x) for lb, x in zip(labels, rates)},
        )
        rollouts[pr] = (labels, r)
    return verdicts, rollouts
