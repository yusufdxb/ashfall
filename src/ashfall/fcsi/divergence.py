"""Locate the first persistent, statistically meaningful real/sim divergence.

Residuals are **reset-based**: at every log row the simulator is restored to the logged state
(with logged action and observation histories), the logged commands are replayed open-loop for
``k`` steps, and the prediction is compared with the log ``k`` steps later. Resetting at every
row keeps one early discrepancy from contaminating every later comparison (the reason plain
trajectory error is meaningless after a divergence).

The discrepancy is multivariate: a Mahalanobis distance over all channels, scaled by a
*robust* (trimmed, consistency-corrected) second moment of the nominal model's residuals on
nominal real logs, so ordinary noise and modelling error set the scale, while rare genuine
excursions in the nominal logs (which may contain the very mismatch being sought) do not
inflate it. Persistence comes from a one-sided CUSUM on ``d^2 / D - 1 - drift``; its threshold
gives a declared false-alarm probability per log for chi-square residuals (Monte Carlo, fixed
seed). Alarms on nominal logs are reported as divergences, not suppressed. ``t_div`` is the
start of the excursion that raised the alarm.

One-step residuals (``k = 1``) are the default: longer open-loop windows accumulate the
unknown process noise faster than most mechanisms accumulate signal (measured while building
the toy: a 5-step detector missed latency and braking-friction mismatches entirely).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ashfall.fcsi.toy_world import CHANNELS, MAX_DELAY, Log, logged_channels, predict_open_loop

K_STEPS = 1
DRIFT = 0.5
FALSE_ALARM = 0.01
LOG_STEPS = 175
TRIM = 0.95


def rows_for(log: Log, k: int = K_STEPS) -> np.ndarray:
    return np.arange(MAX_DELAY, len(log) - k)


def residuals(cfg, physics_list, log: Log, *, k: int = K_STEPS, rows=None) -> np.ndarray:
    """(P, n, D) logged minus predicted channels at ``rows + k``."""
    rows = rows_for(log, k) if rows is None else rows
    pred = predict_open_loop(cfg, physics_list, log, rows, k)
    return logged_channels(log, rows + k)[None] - pred


@dataclass
class Scale:
    """Robust second-moment matrix of nominal residuals (trimmed, consistency-corrected)."""

    inv: np.ndarray
    second_moment: np.ndarray
    kept_fraction: float = 1.0

    @staticmethod
    def fit(res: np.ndarray, *, iterations: int = 3) -> "Scale":
        from scipy.stats import chi2

        D = res.shape[1]
        S = res.T @ res / len(res)
        q = chi2.ppf(TRIM, D)
        corr = chi2.cdf(q, D + 2) / TRIM  # E[d2 | d2 <= q] / D for chi-square residuals
        keep = np.ones(len(res), bool)
        for _ in range(iterations):
            d2 = np.einsum("ni,ij,nj->n", res, np.linalg.inv(S), res)
            keep = d2 <= q
            sub = res[keep]
            S = sub.T @ sub / len(sub) / corr
        return Scale(np.linalg.inv(S), S, float(keep.mean()))

    def d2(self, r: np.ndarray) -> np.ndarray:
        return np.einsum("...i,ij,...j->...", r, self.inv, r)


def cusum(d2: np.ndarray, dim: int = len(CHANNELS), drift: float = DRIFT) -> np.ndarray:
    z = d2 / dim - 1.0 - drift
    out = np.zeros(len(z))
    s = 0.0
    for i, v in enumerate(z):
        s = max(0.0, s + v)
        out[i] = s
    return out


def chi2_threshold(
    dim: int = len(CHANNELS),
    *,
    steps: int = LOG_STEPS,
    alpha: float = FALSE_ALARM,
    draws: int = 20_000,
    seed: int = 20260922,
) -> float:
    """CUSUM level exceeded with probability ``alpha`` per log under iid chi-square residuals."""
    rng = np.random.default_rng(seed)
    z = rng.chisquare(dim, (draws, steps)) / dim - 1.0 - DRIFT
    s = np.zeros(draws)
    peak = np.zeros(draws)
    for t in range(steps):
        s = np.maximum(0.0, s + z[:, t])
        peak = np.maximum(peak, s)
    return float(np.quantile(peak, 1 - alpha))


@dataclass
class Calibration:
    scale: Scale
    threshold: float
    nominal_alarms: int
    nominal_logs: int
    k: int = K_STEPS

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "nominal_alarms": self.nominal_alarms,
            "nominal_logs": self.nominal_logs,
            "kept_fraction": self.scale.kept_fraction,
            "k": self.k,
            "second_moment": self.scale.second_moment.tolist(),
        }


def calibrate(cfg, nominal_model, nominal_logs: list[Log], *, k: int = K_STEPS) -> Calibration:
    res = [residuals(cfg, [nominal_model], lg, k=k)[0] for lg in nominal_logs]
    scale = Scale.fit(np.concatenate(res))
    h = chi2_threshold()
    alarms = sum(int(np.any(cusum(scale.d2(r)) > h)) for r in res)
    return Calibration(scale, h, alarms, len(nominal_logs), k)


@dataclass
class DivergenceReport:
    diverged: bool
    t_div: float | None
    alarm_time: float | None
    failure_time: float | None
    match_until: float | None
    channel_ranking: list  # (channel, mean standardized squared residual in the excursion)
    first_single_channel: dict  # channel -> first time |z| > 3 at or after t_div (or None)
    d2: list
    cusum: list

    def to_dict(self) -> dict:
        return asdict(self)

    def text(self) -> str:
        if not self.diverged:
            return "REAL/SIM MATCH over the whole log (no persistent divergence)"
        lines = [
            f"REAL/SIM MATCH  0.00-{self.match_until:.2f} s",
            f"FIRST DIVERGENCE  {self.t_div:.2f} s (alarm {self.alarm_time:.2f} s)",
            f"primary channel: {self.channel_ranking[0][0]}",
        ]
        if len(self.channel_ranking) > 1:
            lines.append(f"secondary: {self.channel_ranking[1][0]}")
        for ch, t in self.first_single_channel.items():
            if t is not None:
                lines.append(f"{ch} alone exceeds 3 sd at {t:.2f} s")
        if self.failure_time is not None:
            lines.append(f"failure onset: {self.failure_time:.2f} s")
        return "\n".join(lines)


def locate(cfg, model, log: Log, cal: Calibration) -> DivergenceReport:
    rows = rows_for(log, cal.k)
    r = residuals(cfg, [model], log, k=cal.k, rows=rows)[0]
    d2 = cal.scale.d2(r)
    cs = cusum(d2)
    alarm = np.flatnonzero(cs > cal.threshold)
    fail_t = None if not np.isfinite(log.fail_time) else float(log.fail_time)
    if len(alarm) == 0:
        return DivergenceReport(False, None, None, fail_t, None, [], {}, d2.tolist(), cs.tolist())
    a = int(alarm[0])
    zeros = np.flatnonzero(cs[: a + 1] == 0.0)
    start = int(zeros[-1] + 1) if len(zeros) else 0
    sd = np.sqrt(np.diag(cal.scale.second_moment))
    z = r / sd
    exc = z[start : a + 1] ** 2
    ranking = sorted(
        ((CHANNELS[i], float(exc[:, i].mean())) for i in range(len(CHANNELS))), key=lambda t: -t[1]
    )
    first = {}
    for i, ch in enumerate(CHANNELS):
        hit = np.flatnonzero(np.abs(z[start:, i]) > 3.0)
        first[ch] = float(log.t[rows[start + hit[0]]]) if len(hit) else None
    t_div = float(log.t[rows[start]])
    return DivergenceReport(
        True,
        t_div,
        float(log.t[rows[a]]),
        fail_t,
        max(0.0, t_div - cfg.dt),
        ranking,
        first,
        d2.tolist(),
        cs.tolist(),
    )
