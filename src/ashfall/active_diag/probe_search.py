"""Rank candidate probes: safety filter first, then expected information gain.

For each candidate probe: (1) simulate it under every plausible hypothesis (MAP value and the
next most likely grid value per plausible mechanism, plus the unchanged nominal model) and drop it
if any is predicted unsafe (:mod:`ashfall.active_diag.safety`); (2) for the admissible ones,
estimate the expected information gain (:mod:`ashfall.active_diag.information`) from the same
rollouts; (3) rank by ``EIG - cost`` (cost is a small tie-breaker preferring gentler probes).
Every simulated step is charged to the method's budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import information as inf
from . import safety as sf
from .posterior import NONE, Weights, model_for
from .probe_space import Probe


@dataclass
class Ranked:
    probe: Probe
    eig: float
    score: float
    safety: dict

    def to_dict(self) -> dict:
        return {
            "probe": self.probe.to_dict(),
            "eig": self.eig,
            "score": self.score,
            "safety": self.safety,
        }


def safety_models(base, w: Weights) -> tuple[dict, list, list]:
    """Label -> Physics for the safety check, and the discrimination set (labels, MAP hyps)."""
    plaus = [m for m in w.plausible(sf.PLAUSIBLE) if m != NONE]
    models = {NONE: base}
    set_labels, set_hyps = [NONE], [(NONE, float("nan"))]
    for m in plaus:
        vals, ww = w.within[m]
        # MAP value first (ties broken as in posterior.weights), then the best other value.
        map_v = w.map_value[m]
        others = [j for j in np.argsort(ww)[::-1] if vals[j] != map_v]
        for v in [map_v] + [vals[j] for j in others[:1]]:
            models[f"{m}@{v:.4g}"] = model_for(base, (m, v))
        set_labels.append(m)
        set_hyps.append((m, w.map_value[m]))
    return models, set_labels, set_hyps


def admissible(cfg, theta, base, w: Weights, probes: list, budget=None):
    models, labels, hyps = safety_models(base, w)
    verdicts, rollouts = sf.check_many(cfg, theta, probes, models, budget)
    ok = [p for p in probes if verdicts[p].admissible]
    return ok, verdicts, rollouts, models, labels, hyps


def rank(cfg, theta, base, w: Weights, probes: list, scale, temperature, rng, budget=None):
    ok, verdicts, rollouts, models, labels, hyps = admissible(cfg, theta, base, w, probes, budget)
    keys = list(models)
    first_rows = []
    for lb, h in zip(labels, hyps):
        key = NONE if lb == NONE else f"{lb}@{h[1]:.4g}"
        first_rows.append(keys.index(key) * sf.DRAWS)
    pi = inf.prior(w.as_dict(), labels)
    out = []
    for p in ok:
        _, r = rollouts[p]
        eig = inf.expected_information_gain(
            cfg, theta, base, p, hyps, first_rows, r, scale, temperature, pi, rng, budget
        )
        out.append(Ranked(p, eig, eig - p.cost, asdict(verdicts[p])))
    out.sort(key=lambda x: -x.score)
    return out, verdicts
