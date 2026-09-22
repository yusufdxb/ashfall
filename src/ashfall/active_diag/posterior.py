"""Weights over simulator-mismatch hypotheses from logged evidence.

Each hypothesis is one library mechanism (or ``none``, the unchanged nominal model) with its
parameter on a fixed grid over the prior range. The per-row score is the one-step reset-based
residual of FCSI (``ashfall.fcsi.divergence``) plus one **action-consistency channel**: the
policy is known, so the command it must have issued can be recomputed from the logged states
under each hypothesis's sensing model (observation delay, tilt bias) and compared with the
logged command. Without that channel sensing mechanisms are invisible to open-loop replay.

Rows are robust: each row's ``d^2`` is capped at ``ROW_CAP`` (an outlier component), because a
single row at a patch edge, where 2 mm of position noise flips on-patch/off-patch, can otherwise
produce ``d^2`` in the thousands under the exact true physics (measured on a development case).

The weights are ``softmax(logmeanexp_theta(LL(m, theta)) / T)``: a tempered, grid-marginalized
Gaussian residual likelihood. Residuals are not independent Gaussians, so these are
**calibrated weights, not posterior probabilities**; ``T`` is fitted on development cases only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ashfall.fcsi import divergence as dv
from ashfall.fcsi.mechanisms import LIBRARY
from ashfall.fcsi.objective import physics_with
from ashfall.fcsi.toy_world import (
    MAX_DELAY,
    Log,
    Physics,
    logged_channels,
    policy_force,
    predict_open_loop,
)

GRID = 13
NONE = "none"
ROW_CAP = 50.0  # per-row d^2 cap: an outlier model, so one bad row cannot decide a diagnosis


def hypothesis_grid(library=LIBRARY, n: int = GRID) -> list[tuple[str, float]]:
    """``(mechanism, value)`` pairs, plus ``(none, nan)``; each grid includes the nominal value."""
    out = [(NONE, float("nan"))]
    for m in library:
        vals = np.unique(np.r_[np.linspace(m.low, m.high, n), m.nominal])
        out += [(m.name, float(v)) for v in vals]
    return out


def model_for(base: Physics, h: tuple[str, float]) -> Physics:
    return base if h[0] == NONE else physics_with(base, **{h[0]: h[1]})


@dataclass
class Scale6:
    """Robust state-residual scale (5 channels) plus the action-residual variance."""

    state: dv.Scale
    action_var: float

    def d2(self, r_state: np.ndarray, r_action: np.ndarray) -> np.ndarray:
        return self.state.d2(r_state) + r_action**2 / self.action_var


def _obs_from_log(log: Log, rows: np.ndarray, delay: float, bias: float, v_cmd: float):
    d = float(np.clip(delay, 0.0, MAX_DELAY - 1e-9))
    k, f = int(np.floor(d)), d - np.floor(d)

    def lag(a):
        r0 = np.clip(rows - k, 0, None)
        r1 = np.clip(rows - k - 1, 0, None)
        return (1 - f) * a[r0] + f * a[r1]

    # the policy acting at row t saw the (delayed) logged state of row t; newest first
    th, thd, v, slip = lag(log.th), lag(log.thd), lag(log.v), lag(log.slip)
    ph = log.ph[rows]
    return np.stack([th + bias, thd, v - v_cmd, slip, np.sin(ph), np.cos(ph)], -1)


def action_residuals(cfg, theta, models: list, log: Log, rows: np.ndarray, probe_force=None):
    """(P, n) logged policy command minus the command each model's sensing implies."""
    policy_cmd = log.action[rows] - (0.0 if probe_force is None else probe_force[rows])
    out = []
    cache = {}
    for m in models:
        key = (m.get("obs_delay"), m.get("theta_bias"))
        if key not in cache:
            obs = _obs_from_log(log, rows, key[0], key[1], log.cond.v_cmd)
            cache[key] = policy_cmd - policy_force(theta, obs, cfg)
        out.append(cache[key])
    return np.array(out)


def state_residuals(cfg, models: list, log: Log, rows: np.ndarray) -> np.ndarray:
    pred = predict_open_loop(cfg, models, log, rows, 1)
    return logged_channels(log, rows + 1)[None] - pred


ENVELOPE_TILT = 0.35


def rows_of(log: Log) -> np.ndarray:
    """Rows inside the calibrated envelope: logged tilt (now and next row) <= ``ENVELOPE_TILT``.

    The residual scale is calibrated on upright nominal logs; in the last moments of a fall even
    the exact true physics leaves one-step residuals hundreds of times larger (measured on a
    development case: CUSUM peak 430 for the truth), so those rows carry no model information.
    """
    rows = np.arange(MAX_DELAY, len(log) - 1)
    ok = (np.abs(log.th[rows]) <= ENVELOPE_TILT) & (np.abs(log.th[rows + 1]) <= ENVELOPE_TILT)
    return rows[ok]


def calibrate_scale(cfg, theta, base: Physics, nominal_logs: list) -> Scale6:
    cal = dv.calibrate(cfg, base, nominal_logs, k=1)
    ra = np.concatenate(
        [action_residuals(cfg, theta, [base], lg, rows_of(lg))[0] for lg in nominal_logs]
    )
    med = np.median(np.abs(ra - np.median(ra)))
    var = (1.4826 * med) ** 2 + 1e-9
    return Scale6(cal.scale, float(var))


@dataclass
class EvidenceLog:
    log: Log
    probe_force: np.ndarray | None = None
    label: str = ""


def log_likelihoods(cfg, theta, base, hyps, evidence: list, scale: Scale6, budget=None):
    """(H,) summed Gaussian residual log-likelihood of each hypothesis over all evidence."""
    models = [model_for(base, h) for h in hyps]
    ll = np.zeros(len(models))
    for ev in evidence:
        rows = rows_of(ev.log)
        rs = state_residuals(cfg, models, ev.log, rows)
        ra = action_residuals(cfg, theta, models, ev.log, rows, ev.probe_force)
        ll -= 0.5 * np.minimum(scale.d2(rs, ra), ROW_CAP).sum(axis=1)
        if budget is not None:
            budget.add(len(models) * len(rows), 0)
    return ll


@dataclass
class Weights:
    mechanisms: list
    weights: np.ndarray
    map_value: dict
    within: dict = field(default_factory=dict)  # mechanism -> (values, normalized weights)

    @property
    def top(self) -> str:
        return self.mechanisms[int(np.argmax(self.weights))]

    @property
    def top_weight(self) -> float:
        return float(np.max(self.weights))

    @property
    def entropy(self) -> float:
        w = self.weights[self.weights > 0]
        return float(-(w * np.log(w)).sum())

    def as_dict(self) -> dict:
        return {m: float(w) for m, w in zip(self.mechanisms, self.weights)}

    def plausible(self, floor: float) -> list:
        return [m for m, w in zip(self.mechanisms, self.weights) if w >= floor]


def weights(hyps, ll: np.ndarray, temperature: float) -> Weights:
    mechs = list(dict.fromkeys(h[0] for h in hyps))
    lse, mapv, within = [], {}, {}
    for m in mechs:
        idx = [i for i, h in enumerate(hyps) if h[0] == m]
        x = ll[idx] / temperature
        mx = x.max()
        lse.append(mx + np.log(np.mean(np.exp(x - mx))))
        mapv[m] = hyps[idx[int(np.argmax(x))]][1]
        w = np.exp(x - mx)
        within[m] = ([hyps[i][1] for i in idx], (w / w.sum()).tolist())
    lse = np.array(lse)
    w = np.exp(lse - lse.max())
    return Weights(mechs, w / w.sum(), mapv, within)


def map_model(base: Physics, w: Weights) -> Physics:
    return model_for(base, (w.top, w.map_value[w.top]))
