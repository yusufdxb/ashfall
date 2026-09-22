"""Synchronize a real log and a simulator replay before comparing them.

Nothing here decides whether the simulator is right; it only makes sure two streams are
compared at the same instants, in the same frames, with the uncertainty of that alignment
reported instead of hidden.

Rules:

* timestamps must be strictly increasing; jitter and gaps are measured and reported;
* resampling onto a common grid interpolates only across gaps no longer than ``max_gap_s``;
  anything else is marked missing (NaN) and counted, never filled;
* attitude is compared with a sign-invariant geodesic angle, never component-wise, and the
  quaternion convention is explicit (``"wxyz"`` or ``"xyzw"``);
* command and action streams can be lag-aligned by cross-correlation within a declared bound;
  the chosen lag and the correlation peak are returned so a weak peak is visible;
* contact events are compared as on/off edge times, with unmatched edges reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TimebaseReport:
    n: int
    dt_median: float
    dt_max: float
    jitter_s: float  # std of dt
    gaps: int  # dt > max_gap_s
    monotonic: bool

    def to_dict(self) -> dict:
        return asdict(self)


def check_timebase(t: np.ndarray, *, max_gap_s: float) -> TimebaseReport:
    t = np.asarray(t, float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("need a 1-D timestamp array with at least two samples")
    dt = np.diff(t)
    return TimebaseReport(
        n=len(t),
        dt_median=float(np.median(dt)),
        dt_max=float(dt.max()),
        jitter_s=float(dt.std()),
        gaps=int(np.sum(dt > max_gap_s)),
        monotonic=bool(np.all(dt > 0)),
    )


def resample(t: np.ndarray, y: np.ndarray, grid: np.ndarray, *, max_gap_s: float):
    """Linear interpolation of ``y(t)`` onto ``grid``; NaN outside the data or across gaps.

    Returns ``(values, missing_mask)``. ``y`` may be (T,) or (T, D).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    rep = check_timebase(t, max_gap_s=max_gap_s)
    if not rep.monotonic:
        raise ValueError("timestamps are not strictly increasing")
    y2 = y[:, None] if y.ndim == 1 else y
    out = np.full((len(grid), y2.shape[1]), np.nan)
    j = np.searchsorted(t, grid, side="right") - 1
    inside = (grid >= t[0]) & (grid <= t[-1])
    j = np.clip(j, 0, len(t) - 2)
    t0, t1 = t[j], t[j + 1]
    exact = np.isclose(grid, t[np.clip(np.searchsorted(t, grid), 0, len(t) - 1)])
    ok = inside & (((t1 - t0) <= max_gap_s + 1e-12) | exact)
    w = np.where(t1 > t0, (grid - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
    vals = (1 - w)[:, None] * y2[j] + w[:, None] * y2[j + 1]
    out[ok] = vals[ok]
    missing = ~ok
    return (out[:, 0] if y.ndim == 1 else out), missing


def _to_wxyz(q: np.ndarray, convention: str) -> np.ndarray:
    q = np.asarray(q, float)
    if convention == "wxyz":
        return q
    if convention == "xyzw":
        return q[..., [3, 0, 1, 2]]
    raise ValueError("quaternion convention must be 'wxyz' or 'xyzw'")


def attitude_error(q_a: np.ndarray, q_b: np.ndarray, *, convention: str) -> np.ndarray:
    """Geodesic angle (rad) between orientations; q and -q are the same attitude."""
    a = _to_wxyz(q_a, convention)
    b = _to_wxyz(q_b, convention)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if np.any(np.abs(na - 1) > 1e-3) or np.any(np.abs(nb - 1) > 1e-3):
        raise ValueError("quaternions must be unit norm (check the convention and units)")
    dot = np.abs(np.sum(a * b, axis=-1) / (na * nb))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


@dataclass(frozen=True)
class LagEstimate:
    lag_steps: int
    peak_correlation: float
    second_best: float

    @property
    def ambiguous(self) -> bool:
        return self.peak_correlation - self.second_best < 0.05

    def to_dict(self) -> dict:
        return {**asdict(self), "ambiguous": self.ambiguous}


def estimate_lag(reference: np.ndarray, delayed: np.ndarray, *, max_lag: int) -> LagEstimate:
    """Steps by which ``delayed`` lags ``reference`` (positive = later), by normalized xcorr."""
    a = np.asarray(reference, float)
    b = np.asarray(delayed, float)
    scores = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: len(a) - lag], b[lag : lag + len(a) - lag]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        if n < 3 or x.std() == 0 or y.std() == 0:
            continue
        scores[lag] = float(np.corrcoef(x, y)[0, 1])
    if not scores:
        raise ValueError("streams too short or constant to estimate a lag")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return LagEstimate(ranked[0][0], ranked[0][1], ranked[1][1] if len(ranked) > 1 else -1.0)


def contact_edges(t: np.ndarray, contact: np.ndarray) -> dict:
    c = np.asarray(contact, bool).astype(int)
    d = np.diff(c)
    t = np.asarray(t, float)
    return {"touchdown": t[1:][d == 1], "liftoff": t[1:][d == -1]}


def match_edges(real: np.ndarray, sim: np.ndarray, *, tol_s: float) -> dict:
    """Greedy nearest matching of edge times; unmatched edges are reported, not dropped."""
    real, sim = list(np.sort(real)), list(np.sort(sim))
    pairs, used = [], set()
    for r in real:
        cands = [
            (abs(s - r), i) for i, s in enumerate(sim) if i not in used and abs(s - r) <= tol_s
        ]
        if cands:
            _, i = min(cands)
            used.add(i)
            pairs.append((r, sim[i]))
    offsets = [s - r for r, s in pairs]
    return {
        "matched": len(pairs),
        "unmatched_real": len(real) - len(pairs),
        "unmatched_sim": len(sim) - len(used),
        "mean_offset_s": float(np.mean(offsets)) if offsets else float("nan"),
    }


def validate_toy_log(log, dt: float) -> dict:
    """The toy's contract: one control-rate clock shared by state, action and command."""
    rep = check_timebase(log.t, max_gap_s=1.5 * dt)
    if not rep.monotonic or rep.gaps or abs(rep.dt_median - dt) > 1e-9:
        raise ValueError(f"toy log violates the shared-clock contract: {rep}")
    lengths = {len(getattr(log, k)) for k in ("x", "v", "th", "thd", "ph", "slip", "action")}
    if lengths != {len(log.t)}:
        raise ValueError("toy log streams have different lengths")
    return {"timebase": rep.to_dict(), "missing": 0, "interpolated": 0}
