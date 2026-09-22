"""Scores a candidate simulator against the evidence: residual fit and event reproduction.

Every identifier (FCSI and the global baselines) receives the same :class:`Evidence`: the
nominal real logs, the target failure log, and one validation failure log. What differs is
how they use it.

Two kinds of score:

* **residual negative log-likelihood** of one-step (or k-step) reset-based residuals, in the
  nominal calibration's robust scale, summed over chosen rows (``0.5 * sum d^2``);
* **event probability**: restore the logged state at a start row (with its histories), run the
  frozen policy closed-loop in the candidate simulator under ``M`` process-noise draws, and count
  how often it reproduces the logged outcome: the same phenotype (fall, stall, success) and, for
  a fall, an onset within ``ONSET_TOL_S`` of the logged one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ashfall.fcsi import divergence as dv
from ashfall.fcsi.toy_world import (
    MAX_DELAY,
    Log,
    Physics,
    closed_loop,
    logged_channels,
    predict_open_loop,
    state_from_log,
)

ONSET_TOL_S = 0.3
EVENT_LEAD_S = 0.5
EVENT_DRAWS = 128
NOMINAL_DRAWS = 16
EVENT_NOISE_BASE = 700_000


@dataclass
class Evidence:
    nominal_logs: list
    failure_log: Log
    validation_log: Log | None = None

    def all_logs(self) -> list:
        out = list(self.nominal_logs) + [self.failure_log]
        return out + ([self.validation_log] if self.validation_log is not None else [])


@dataclass
class Budget:
    steps: int = 0
    evaluations: int = 0
    p_nominal_event: float = 0.0  # set by FCSI once, used by its feasibility rule

    def add(self, steps: int, evaluations: int = 1):
        self.steps += int(steps)
        self.evaluations += int(evaluations)


def nll(
    cfg,
    models: list,
    log: Log,
    rows: np.ndarray,
    scale: dv.Scale,
    *,
    k: int = 1,
    budget: Budget | None = None,
) -> np.ndarray:
    """(P,) residual NLL of each candidate over ``rows`` of ``log``."""
    rows = np.asarray(rows)
    rows = rows[(rows >= MAX_DELAY) & (rows + k < len(log))]
    if len(rows) == 0:
        return np.zeros(len(models))
    pred = predict_open_loop(cfg, models, log, rows, k)
    r = logged_channels(log, rows + k)[None] - pred
    if budget is not None:
        budget.add(len(models) * len(rows) * k, 0)
    return 0.5 * scale.d2(r).sum(axis=1)


def outcome_match(log: Log, rollout, start_time: float) -> np.ndarray:
    ph = rollout.phenotype
    if log.outcome == "fall":
        onset = start_time + rollout.fail_time
        return (ph == "fall") & (np.abs(onset - log.fail_time) <= ONSET_TOL_S)
    return ph == log.outcome


def event_probability(
    cfg,
    theta,
    models: list,
    log: Log,
    start_row: int,
    *,
    draws=EVENT_DRAWS,
    budget: Budget | None = None,
) -> np.ndarray:
    """(P,) probability that each candidate reproduces the logged outcome from ``start_row``."""
    P = len(models)
    start_row = int(max(MAX_DELAY, min(start_row, len(log) - 1)))
    st = state_from_log(log, np.full(P * draws, start_row))
    phys = [m for m in models for _ in range(draws)]
    seeds = np.tile(np.arange(draws) + EVENT_NOISE_BASE, P)
    r = closed_loop(cfg, theta, phys, log.cond, st, seeds, t_offset=start_row)
    if budget is not None:
        budget.add(P * draws * r.steps, 0)
    return outcome_match(log, r, float(log.t[start_row])).reshape(P, draws).mean(axis=1)


def nominal_success(
    cfg, theta, models: list, logs: list, *, draws=NOMINAL_DRAWS, budget: Budget | None = None
) -> np.ndarray:
    """(P,) mean probability of reproducing each successful nominal log's success."""
    vals = [
        event_probability(cfg, theta, models, lg, MAX_DELAY, draws=draws, budget=budget)
        for lg in logs
    ]
    return np.mean(vals, axis=0)


def event_start_row(cfg, log: Log, t_div: float | None) -> int:
    """Replay start: ``EVENT_LEAD_S`` before the divergence (or before the failure if none)."""
    anchor = (
        t_div
        if t_div is not None
        else (log.fail_time if np.isfinite(log.fail_time) else float(log.t[-1])) - 1.0
    )
    return int(max(MAX_DELAY, round((anchor - EVENT_LEAD_S) / cfg.dt)))


def physics_with(base: Physics, **values) -> Physics:
    d = dict(base.values)
    d.update(values)
    return Physics(tuple(d.items()), base.hidden)
