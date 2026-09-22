"""Global system-identification baselines, given exactly the evidence FCSI gets.

All three fit every library parameter jointly (10 dimensions) with the same derivative-free
optimizer: a scrambled Halton design of ``HALTON`` points in the normalized prior box, then
bounded Nelder-Mead from the best ``RESTARTS`` points (``NM_EVALS`` evaluations each). None of
them is told which mechanism is failure-critical, and none uses the failure event.

* **G0 whole-trajectory (SimOpt-like)**: closed-loop replay of the frozen policy from each log's
  first row under ``G0_DRAWS`` process-noise draws; mean squared standardized error against the
  whole logged trajectory. This is the "average trajectory error" objective.
* **G0b multiple shooting**: reset-based ``k``-step open-loop prediction error over every row of
  every log (Mahalanobis in the nominal ``k``-step scale). Reset-based, so immune to the
  post-divergence double penalty; the strong classical answer to G0's weakness.
* **G1 DROPO-like**: parameters ~ N(mu, sigma^2) per dimension, ``DROPO_SAMPLES`` common-random
  samples; one-step reset-based predictions give a per-transition Gaussian (sample mean,
  sample variance plus the nominal one-step residual variance); maximize the summed
  log-likelihood of the logged next states over (mu, log sigma). Point estimate = mu.

Fairness rules, fixed before any comparison ran:

* G0b and G1 replay logged commands open-loop, where sensing parameters (observation delay,
  tilt bias) have no effect; they are held at nominal instead of being left wherever the design
  put them. G0 is closed-loop and fits them.
* Every fit adds a weak ridge toward the nominal model, ``RIDGE x loss(nominal) x |du|^2`` in
  normalized units, so parameters the data cannot see stay nominal (standard regularization,
  and it can only help the baselines on sparsity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from ashfall.fcsi import divergence as dv
from ashfall.fcsi import objective as ob
from ashfall.fcsi.mechanisms import LIBRARY
from ashfall.fcsi.toy_world import (
    CHANNELS,
    MAX_DELAY,
    Physics,
    closed_loop,
    logged_channels,
    predict_open_loop,
    state_from_log,
)

HALTON = 256
RESTARTS = 3
NM_EVALS = 250
G0_DRAWS = 8
G0B_K = 5
DROPO_SAMPLES = 10
LOG_SIGMA = (-5.0, -1.0)  # log of sigma as a fraction of the prior span
RIDGE = 0.01
OPEN_LOOP_BLIND = ("obs_delay", "theta_bias")
U0 = np.array([(m.nominal - m.low) / m.span for m in LIBRARY])


@dataclass
class GlobalFit:
    method: str
    values: dict
    sigma: dict | None
    loss: float
    budget: dict

    def physics(self, base: Physics) -> Physics:
        return ob.physics_with(base, **self.values)

    def to_dict(self) -> dict:
        return asdict(self)


def _to_values(u: np.ndarray, frozen=()) -> dict:
    u = np.clip(u, 0.0, 1.0)
    return {
        m.name: (m.nominal if m.name in frozen else float(m.low + ui * m.span))
        for m, ui in zip(LIBRARY, u)
    }


def _ridged(loss_batch, dims: slice = slice(None)):
    """Add RIDGE x loss(nominal) x |u - u0|^2 over the parameter dimensions."""
    ref = float(abs(loss_batch(U0[None] if dims == slice(None) else _nominal_point(dims))[0]))

    def f(U):
        U = np.atleast_2d(U)
        du = U[:, : len(LIBRARY)] - U0[None]
        return loss_batch(U) + RIDGE * max(ref, 1e-9) * np.sum(du**2, axis=1)

    return f


def _nominal_point(dims):
    return np.r_[U0, np.full(len(LIBRARY), LOG_SIGMA[0])][None]


def _optimize(loss_batch, dim: int, seed: int, budget: ob.Budget, lo=None, hi=None):
    """Halton design (batched) then bounded Nelder-Mead restarts; returns (best_u, best_loss)."""
    lo = np.zeros(dim) if lo is None else np.asarray(lo)
    hi = np.ones(dim) if hi is None else np.asarray(hi)
    pts = lo + qmc.Halton(d=dim, scramble=True, seed=seed).random(HALTON) * (hi - lo)
    losses = np.asarray(loss_batch(pts))
    budget.add(0, len(pts))
    order = np.argsort(losses)
    best_u, best_l = pts[order[0]], float(losses[order[0]])
    for i in order[:RESTARTS]:

        def f(u):
            budget.add(0, 1)
            return float(loss_batch(np.clip(u, lo, hi)[None])[0])

        res = minimize(
            f,
            pts[i],
            method="Nelder-Mead",
            bounds=list(zip(lo, hi)),
            options={"maxfev": NM_EVALS, "xatol": 1e-3, "fatol": 1e-6},
        )
        if res.fun < best_l:
            best_u, best_l = np.clip(res.x, lo, hi), float(res.fun)
    return best_u, best_l


def _models(U: np.ndarray, base: Physics, frozen=()) -> list:
    return [ob.physics_with(base, **_to_values(u, frozen)) for u in U]


def fit_g0(cfg, theta, ev: ob.Evidence, *, base=None, seed=0) -> GlobalFit:
    base = base or Physics.of()
    logs = ev.all_logs()
    allch = np.concatenate([logged_channels(lg, np.arange(len(lg))) for lg in logs])
    sd = allch.std(axis=0) + 1e-9
    budget = ob.Budget()
    seeds = np.arange(G0_DRAWS) + 710_000

    def loss_batch(U):
        models = _models(U, base)
        P = len(models)
        tot = np.zeros(P)
        for lg in logs:
            T = len(lg) - MAX_DELAY
            st = state_from_log(lg, np.full(P * G0_DRAWS, MAX_DELAY))
            phys = [m for m in models for _ in range(G0_DRAWS)]
            r = closed_loop(
                cfg, theta, phys, lg.cond, st, np.tile(seeds, P), steps=T, t_offset=MAX_DELAY
            )
            budget.add(P * G0_DRAWS * T, 0)
            sim = np.stack([r.x, r.v, r.th, r.thd, r.slip], -1)  # (P*D, T, 5)
            real = logged_channels(lg, np.arange(MAX_DELAY, len(lg)))[None]
            err = (((sim - real) / sd) ** 2).mean(axis=(1, 2)).reshape(P, G0_DRAWS).mean(1)
            tot += err
        return tot / len(logs)

    u, best = _optimize(_ridged(loss_batch), len(LIBRARY), seed, budget)
    return GlobalFit("G0_whole_trajectory", _to_values(u), None, best, asdict(budget))


def fit_g0b(cfg, ev: ob.Evidence, cal_k: dv.Calibration, *, base=None, seed=0) -> GlobalFit:
    base = base or Physics.of()
    logs = ev.all_logs()
    budget = ob.Budget()

    def loss_batch(U):
        models = _models(U, base, OPEN_LOOP_BLIND)
        tot, n = np.zeros(len(models)), 0
        for lg in logs:
            rows = np.arange(MAX_DELAY, len(lg) - G0B_K)
            tot += 2 * ob.nll(cfg, models, lg, rows, cal_k.scale, k=G0B_K, budget=budget)
            n += len(rows)
        return tot / max(n, 1)

    u, best = _optimize(_ridged(loss_batch), len(LIBRARY), seed, budget)
    return GlobalFit(
        "G0b_multiple_shooting", _to_values(u, OPEN_LOOP_BLIND), None, best, asdict(budget)
    )


def fit_g1(cfg, ev: ob.Evidence, cal1: dv.Calibration, *, base=None, seed=0) -> GlobalFit:
    base = base or Physics.of()
    logs = ev.all_logs()
    budget = ob.Budget()
    D = len(LIBRARY)
    z = np.random.default_rng(seed + 17).standard_normal((DROPO_SAMPLES, D))
    noise_var = np.diag(cal1.scale.second_moment)
    data = []
    for lg in logs:
        rows = np.arange(MAX_DELAY, len(lg) - 1)
        data.append((lg, rows, logged_channels(lg, rows + 1)))

    def loss_one(v):
        mu, log_sig = v[:D], v[D:]
        U = np.clip(mu[None] + np.exp(log_sig)[None] * z, 0.0, 1.0)
        models = _models(U, base, OPEN_LOOP_BLIND)
        ll = 0.0
        for lg, rows, real in data:
            pred = predict_open_loop(cfg, models, lg, rows, 1)  # (S, n, 5)
            budget.add(len(models) * len(rows), 0)
            m = pred.mean(0)
            var = pred.var(0) + noise_var
            ll += -0.5 * np.sum((real - m) ** 2 / var + np.log(var))
        return -ll

    def loss_batch(V):
        return np.array([loss_one(v) for v in V])

    lo = np.r_[np.zeros(D), np.full(D, LOG_SIGMA[0])]
    hi = np.r_[np.ones(D), np.full(D, LOG_SIGMA[1])]
    v, best = _optimize(_ridged(loss_batch, slice(0, D)), 2 * D, seed, budget, lo, hi)
    sig = {
        m.name: (0.0 if m.name in OPEN_LOOP_BLIND else float(np.exp(s) * m.span))
        for m, s in zip(LIBRARY, v[D:])
    }
    return GlobalFit("G1_dropo_like", _to_values(v[:D], OPEN_LOOP_BLIND), sig, best, asdict(budget))


def calibrate_k(cfg, base, nominal_logs, k: int) -> dv.Calibration:
    return dv.calibrate(cfg, base, nominal_logs, k=k)


def implied_mechanism(values: dict) -> tuple[str, int]:
    """Largest normalized change, and how many parameters moved by more than 5% of their span."""
    changes = {m.name: m.normalized_change(values[m.name]) for m in LIBRARY}
    top = max(changes, key=changes.get)
    return top, int(sum(c > 0.05 for c in changes.values()))


__all__ = ["CHANNELS", "fit_g0", "fit_g0b", "fit_g1", "implied_mechanism", "calibrate_k"]
