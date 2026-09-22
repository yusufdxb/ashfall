"""Recoverability of a restored state under a frozen policy, by branched continuation.

``R_pi(s)`` is the probability that the frozen policy ``pi`` completes the traverse
when the simulator is restored to state ``s`` and the remaining uncertainty is
drawn from a declared :class:`ContextDistribution` (patch friction, per-channel
state jitter, excitation noise). It is estimated by ``n`` independent
continuation rollouts, with a Wilson interval.

No training happens here, and the estimator never receives an evaluation
world's hidden friction or any repair outcome: its only inputs are the policy,
the state and the declared distribution. Every simulator step it spends is
returned so callers can charge it to the method that used it.

Evidence kind ``toy_mechanism`` when used with the slip cart-pole.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from ashfall.analysis.sweep_report import wilson_ci
from ashfall.fbr.toy_slip import ToySeed, ToySlipConfig, seed_batch, simulate, traverse_failure

#: Base of the continuation noise streams. Evaluation uses seeds 500_000..500_255
#: (``toy_study_v2.evaluate_world``) and ES training draws from ``[0, 2**31)``
#: with a seeded generator; streams here start far above both small ranges and
#: are hashed per state, so no continuation rollout reuses an evaluation seed.
STREAM_BASE = 3_000_000_000
STREAM_WIDTH = 1_000_000


@dataclass(frozen=True)
class ContextDistribution:
    """The declared uncertainty a continuation rollout samples."""

    name: str
    mu_low: float
    mu_high: float
    state_noise: tuple[tuple[str, float], ...]

    def __post_init__(self):
        if not 0 < self.mu_low <= self.mu_high:
            raise ValueError("need 0 < mu_low <= mu_high")

    @property
    def noise(self) -> dict:
        return dict(self.state_noise)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RecoverabilityEstimate:
    state_id: str
    distribution: str
    n: int
    successes: int
    r: float
    ci_low: float
    ci_high: float
    mean_survival_s: float
    immediate_termination: float
    sim_steps: int

    @property
    def bernoulli_variance(self) -> float:
        return self.r * (1.0 - self.r)

    def to_dict(self) -> dict:
        return asdict(self)


def stream_for(state_id: str, distribution: str, repeat: int = 0) -> int:
    """A deterministic stream index per (state, distribution, repeat)."""
    h = hashlib.sha256(f"{state_id}|{distribution}|{repeat}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def noise_seeds(stream: int, n: int) -> np.ndarray:
    if n > STREAM_WIDTH:
        raise ValueError("n exceeds the stream width")
    return STREAM_BASE + stream * STREAM_WIDTH + np.arange(n, dtype=np.int64)


def estimate(
    cfg: ToySlipConfig,
    theta: np.ndarray,
    seed: ToySeed,
    dist: ContextDistribution,
    *,
    n: int = 512,
    repeat: int = 0,
) -> RecoverabilityEstimate:
    """Restore ``seed`` ``n`` times, roll the frozen policy, count completed traverses."""
    if n < 1:
        raise ValueError("n must be positive")
    stream = stream_for(seed.seed_id, dist.name, repeat)
    rng = np.random.default_rng(STREAM_BASE + stream)
    mus = rng.uniform(dist.mu_low, dist.mu_high, n)
    batch = seed_batch([seed] * n, mus, noise_seeds(stream, n), noise=dist.noise)
    res = simulate(cfg, np.atleast_2d(theta)[:1], batch)
    fail = traverse_failure(res)[0]
    k = int(n - fail.sum())
    lo, hi = wilson_ci(k, n)
    ft = np.minimum(res.fail_time[0], cfg.horizon_s)
    return RecoverabilityEstimate(
        state_id=seed.seed_id,
        distribution=dist.name,
        n=n,
        successes=k,
        r=k / n,
        ci_low=lo,
        ci_high=hi,
        mean_survival_s=float(ft.mean()),
        immediate_termination=float(np.mean(res.fail_time[0] <= 0.1 + 1e-9)),
        sim_steps=n * cfg.steps,
    )
