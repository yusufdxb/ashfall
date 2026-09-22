"""Slip cart-pole: a CPU mechanism study for failure-boundary replay.

This is a toy, not a quadruped. Its evidence kind is ``toy_mechanism`` and
nothing it produces is a GO2 result. It exists to test the argument in
``docs/research/FBR_METHOD.md`` section 4 under a fixed sample budget, with
every arm trained by the same optimizer from the same baseline.

Physics. A pole (point mass ``m`` at length ``l``) balances on a cart of mass
``M`` that must track a commanded speed. A periodic torque on the pole stands
in for gait excitation; its phase is part of the state, as gait phase is part
of a quadruped's joint state. The drive force the policy requests is limited
by traction, ``|F| <= mu(x) (M + m) g``. When the request exceeds traction the
force saturates at the kinetic limit and the cart slips. A patch of low
friction ``mu_patch`` lies on ``[patch_start, patch_start + patch_length)``;
elsewhere ``mu = mu_ground``. The pole falls (failure) when ``|theta| >
fall_angle``. Whether a traverse fails depends jointly on the patch friction,
the excitation phase and speed at entry, and the policy, which is the
structure FBR is meant to exploit: a failure is local in (state, command,
physics), not in physics alone.

Policy. ``F = F_max * tanh((-K s + 10 MLP(obs)) / F_max)`` with ``s = [theta,
theta_dot, v - v_cmd]`` and ``K`` an LQR gain scaled by learned multipliers,
where obs adds the previous step's slip (requested minus delivered force,
normalised) and the excitation phase. A proprioceptive policy that can feel
slip but cannot see friction.

Optimizer. Antithetic evolution strategies with rank shaping and Adam. Every
arm receives exactly the same number of iterations, perturbations and rollout
slots of equal horizon, so the arms differ only in the reset distribution.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from ashfall.fbr.boundary import (
    BoundaryEstimate,
    BoundaryNotIdentified,
    BoundarySampler,
    BroadSampler,
    LocalNeighborhood,
    ReplayDraw,
    boundary_mass,
    estimate_boundary,
)
from ashfall.fbr.criterion import FailureCriterion

GRAVITY = 9.81
OBS_DIM = 6
HIDDEN = 8


@dataclass(frozen=True)
class ToySlipConfig:
    cart_mass: float = 1.0
    pole_mass: float = 0.3
    pole_length: float = 0.6
    dt: float = 0.02
    horizon_s: float = 3.5
    mu_ground: float = 0.8
    kinetic_ratio: float = 0.85
    force_max: float = 30.0
    excitation_amp: float = 11.0  # rad/s^2 on the pole
    excitation_hz: float = 1.5
    fall_angle: float = 0.6
    noise_torque: float = 2.5  # rad/s^2, white, per step
    patch_length: float = 1.0
    tracking_weight: float = 0.5
    # Baseline training distribution (the "nominal" world the policy was trained in).
    nominal_mu: tuple[float, float] = (0.35, 0.8)
    nominal_patch_start: tuple[float, float] = (0.6, 1.2)
    nominal_speed: tuple[float, float] = (0.8, 1.4)
    # Broad DR range used by arm B.
    broad_mu: tuple[float, float] = (0.05, 0.8)

    @property
    def steps(self) -> int:
        return int(round(self.horizon_s / self.dt))


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def param_count() -> int:
    return 3 + OBS_DIM * HIDDEN + HIDDEN + HIDDEN + 1


def lqr_gains(cfg: "ToySlipConfig") -> np.ndarray:
    """Continuous LQR on the linearised cart-pole, state (theta, theta_dot, v - v_cmd).

    Gives the nominal stabilising feedback the policy starts from; the policy's
    first three parameters are relative multipliers on these gains.
    """
    from scipy.linalg import solve_continuous_are

    M, m, ell, g = cfg.cart_mass, cfg.pole_mass, cfg.pole_length, GRAVITY
    A = np.array([[0.0, 1.0, 0.0], [(M + m) * g / (M * ell), 0.0, 0.0], [-m * g / M, 0.0, 0.0]])
    B = np.array([[0.0], [-1.0 / (M * ell)], [1.0 / M]])
    Q = np.diag([10.0, 1.0, 2.0])
    R = np.array([[0.05]])
    P = solve_continuous_are(A, B, Q, R)
    return (np.linalg.solve(R, B.T @ P)).ravel()


def initial_params() -> np.ndarray:
    """LQR feedback (multipliers of one), zero residual network."""
    return np.zeros(param_count())


def _unpack(params: np.ndarray):
    """params: (P, D) -> gains (P,3), W1 (P,6,8), b1 (P,8), w2 (P,8), b2 (P,)."""
    i = 0
    gains = params[:, i : i + 3]
    i += 3
    w1 = params[:, i : i + OBS_DIM * HIDDEN].reshape(-1, OBS_DIM, HIDDEN)
    i += OBS_DIM * HIDDEN
    b1 = params[:, i : i + HIDDEN]
    i += HIDDEN
    w2 = params[:, i : i + HIDDEN]
    i += HIDDEN
    b2 = params[:, i]
    return gains, w1, b1, w2, b2


# --------------------------------------------------------------------------- #
# Batched simulation
# --------------------------------------------------------------------------- #


@dataclass
class Batch:
    """Initial conditions for B rollouts. Arrays of shape (B,)."""

    x: np.ndarray
    v: np.ndarray
    th: np.ndarray
    thd: np.ndarray
    phase: np.ndarray
    v_cmd: np.ndarray
    mu_patch: np.ndarray
    patch_start: np.ndarray
    noise_seed: np.ndarray  # integer per rollout, for common random numbers

    def __len__(self):
        return len(self.x)

    @staticmethod
    def concat(batches: Sequence["Batch"]) -> "Batch":
        return Batch(
            *(np.concatenate([getattr(b, f) for b in batches]) for f in Batch.__dataclass_fields__)
        )

    def take(self, idx) -> "Batch":
        return Batch(*(getattr(self, f)[idx] for f in Batch.__dataclass_fields__))


@dataclass
class Result:
    failed: np.ndarray  # (P, B) bool
    fail_time: np.ndarray  # (P, B) seconds, inf if none
    reward: np.ndarray  # (P, B) mean per-step reward over the horizon
    traversed: np.ndarray  # (P, B) bool, cart passed patch end
    track_rmse: np.ndarray  # (P, B)
    trace: dict | None = None


def _noise(seeds: np.ndarray, steps: int) -> np.ndarray:
    """(B, steps) standard normal, a pure function of each rollout's seed."""
    return np.stack([np.random.default_rng(int(s)).standard_normal(steps) for s in seeds])


def simulate(
    cfg: ToySlipConfig,
    params: np.ndarray,
    batch: Batch,
    *,
    steps: int | None = None,
    record_trace: bool = False,
    friction_override: float | None = None,
) -> Result:
    """Roll P policies over the same B initial conditions (common random numbers).

    ``friction_override`` replaces the patch friction with a constant, used to
    build matched untreated controls (``mu_ground``) for reconstruction checks.
    """
    params = np.atleast_2d(params)
    P, B = params.shape[0], len(batch)
    T = steps or cfg.steps
    gains, w1, b1, w2, b2 = _unpack(params)
    k0 = lqr_gains(cfg)
    gains = (1.0 + gains) * k0[None, :]
    M, m, ell, g = cfg.cart_mass, cfg.pole_mass, cfg.pole_length, GRAVITY
    N = (M + m) * g
    omega = 2 * math.pi * cfg.excitation_hz
    noise = _noise(batch.noise_seed, T)  # (B, T)

    def tile(a):
        return np.broadcast_to(a, (P, B)).copy()

    x, v, th, thd, ph = (tile(getattr(batch, f)) for f in ("x", "v", "th", "thd", "phase"))
    v_cmd, mu_p, p0 = (tile(getattr(batch, f)) for f in ("v_cmd", "mu_patch", "patch_start"))
    if friction_override is not None:
        mu_p = np.full((P, B), float(friction_override))
    slip = np.zeros((P, B))
    alive = np.ones((P, B), bool)
    fail_time = np.full((P, B), np.inf)
    reward = np.zeros((P, B))
    sq_err = np.zeros((P, B))
    counted = np.zeros((P, B))
    trace = {k: [] for k in ("x", "v", "th", "thd", "phase", "slip")} if record_trace else None

    for t in range(T):
        obs = np.stack([th, thd, v - v_cmd, slip, np.sin(ph), np.cos(ph)], axis=-1)  # (P,B,6)
        hidden = np.tanh(np.einsum("pbi,pih->pbh", obs, w1) + b1[:, None, :])
        residual = np.einsum("pbh,ph->pb", hidden, w2) + b2[:, None]
        linear = -(gains[:, 0:1] * th + gains[:, 1:2] * thd + gains[:, 2:3] * (v - v_cmd))
        f_req = cfg.force_max * np.tanh((linear + 10.0 * residual) / cfg.force_max)
        on_patch = (x >= p0) & (x < p0 + cfg.patch_length)
        mu = np.where(on_patch, mu_p, cfg.mu_ground)
        limit = mu * N
        slipping = np.abs(f_req) > limit
        force = np.where(slipping, np.sign(f_req) * cfg.kinetic_ratio * limit, f_req)
        slip = (f_req - force) / N
        tau = cfg.excitation_amp * np.sin(ph) + cfg.noise_torque * noise[None, :, t]
        s, c = np.sin(th), np.cos(th)
        xdd = (force - m * g * s * c + m * ell * thd**2 * s - m * ell * tau * c) / (M + m * s**2)
        thdd = (g * s - xdd * c) / ell + tau
        v = v + xdd * cfg.dt
        x = x + v * cfg.dt
        thd = thd + thdd * cfg.dt
        th = th + thd * cfg.dt
        ph = ph + omega * cfg.dt
        step_reward = 1.0 - cfg.tracking_weight * (v - v_cmd) ** 2 - 0.5 * th**2
        reward += np.where(alive, step_reward, 0.0)
        sq_err += np.where(alive, (v - v_cmd) ** 2, 0.0)
        counted += alive
        fell = alive & (np.abs(th) > cfg.fall_angle)
        fail_time = np.where(fell, (t + 1) * cfg.dt, fail_time)
        alive &= ~fell
        if trace is not None:
            for k, val in (("x", x), ("v", v), ("th", th), ("thd", thd), ("phase", ph)):
                trace[k].append(val.copy())
            trace["slip"].append(slip.copy())
    failed = ~alive
    traversed = x >= p0 + cfg.patch_length
    if trace is not None:
        trace = {k: np.stack(vals, axis=-1) for k, vals in trace.items()}  # (P,B,T)
    return Result(
        failed=failed,
        fail_time=fail_time,
        reward=reward / T,
        traversed=traversed,
        track_rmse=np.sqrt(sq_err / np.maximum(counted, 1)),
        trace=trace,
    )


def traverse_failure(result: Result) -> np.ndarray:
    """The endpoint: a traverse fails when the pole falls or the cart never crosses."""
    return result.failed | ~result.traversed


# --------------------------------------------------------------------------- #
# Contexts
# --------------------------------------------------------------------------- #


def nominal_batch(cfg: ToySlipConfig, rng: np.random.Generator, n: int, *, mu=None) -> Batch:
    """Full traverses from rest under the baseline training distribution."""
    lo, hi = cfg.nominal_mu
    return Batch(
        x=np.zeros(n),
        v=np.zeros(n),
        th=rng.normal(0, 0.02, n),
        thd=rng.normal(0, 0.05, n),
        phase=rng.uniform(0, 2 * math.pi, n),
        v_cmd=rng.uniform(*cfg.nominal_speed, n),
        mu_patch=rng.uniform(lo, hi, n) if mu is None else np.asarray(mu, float),
        patch_start=rng.uniform(*cfg.nominal_patch_start, n),
        noise_seed=rng.integers(0, 2**31 - 1, n),
    )


def condition_batch(
    cfg: ToySlipConfig, *, mu: float, patch_start: float, v_cmd: float, seeds: np.ndarray
) -> Batch:
    """A fixed physical condition, entered at an uncontrolled gait phase.

    Phase and initial wobble are drawn from each rollout's own seed, so two
    policies evaluated on the same seeds meet identical conditions.
    """
    n = len(seeds)
    rngs = [np.random.default_rng(int(s) + 7_919) for s in seeds]
    return Batch(
        x=np.zeros(n),
        v=np.zeros(n),
        th=np.array([r.normal(0, 0.02) for r in rngs]),
        thd=np.array([r.normal(0, 0.05) for r in rngs]),
        phase=np.array([r.uniform(0, 2 * math.pi) for r in rngs]),
        v_cmd=np.full(n, float(v_cmd)),
        mu_patch=np.full(n, float(mu)),
        patch_start=np.full(n, float(patch_start)),
        noise_seed=np.asarray(seeds, dtype=np.int64),
    )


@dataclass(frozen=True)
class ToySeed:
    """A restorable state: the toy's version of a capsule seed frame."""

    seed_id: str
    x: float
    v: float
    th: float
    thd: float
    phase: float
    v_cmd: float
    patch_start: float


#: Per-channel reset noise for boundary replays (the backend-declared scales).
STATE_NOISE = {"th": 0.01, "thd": 0.05, "v": 0.03, "phase": 0.15}


def seed_batch(
    seeds: Sequence[ToySeed],
    mu: np.ndarray,
    noise_seeds: np.ndarray,
    *,
    command_offsets: np.ndarray | None = None,
    noise_scale: float = 1.0,
    noise: dict | None = None,
) -> Batch:
    n = len(seeds)
    scales = STATE_NOISE if noise is None else noise
    rngs = [np.random.default_rng(int(s) + 104_729) for s in noise_seeds]

    def jitter(name, base):
        return np.array(
            [b + noise_scale * scales[name] * r.normal() for b, r in zip(base, rngs)]
        )

    offsets = np.zeros(n) if command_offsets is None else np.asarray(command_offsets, float)
    return Batch(
        x=np.array([s.x for s in seeds]),
        v=jitter("v", [s.v for s in seeds]),
        th=jitter("th", [s.th for s in seeds]),
        thd=jitter("thd", [s.thd for s in seeds]),
        phase=jitter("phase", [s.phase for s in seeds]),
        v_cmd=np.array([s.v_cmd for s in seeds]) + offsets,
        mu_patch=np.asarray(mu, dtype=float),
        patch_start=np.array([s.patch_start for s in seeds]),
        noise_seed=np.asarray(noise_seeds, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Capsule frames (so the toy runs through the real FBR capsule code)
# --------------------------------------------------------------------------- #


def frames_from_trace(result: Result, index: int, batch: Batch, cfg: ToySlipConfig) -> list[dict]:
    """Rows in the capsule schema for rollout ``index`` of policy 0.

    Mapping: pitch = theta (base_quat), pitch rate = theta_dot, forward body
    velocity = cart speed, joint_pos = (sin, cos) of the gait phase,
    base_pos x = cart position. The mapping is declared, not physical.
    """
    tr = result.trace
    if tr is None:
        raise ValueError("simulate with record_trace=True")
    rows = []
    T = tr["x"].shape[-1]
    fail_step = result.fail_time[0, index]
    for t in range(T):
        th = float(tr["th"][0, index, t])
        ph = float(tr["phase"][0, index, t])
        rows.append(
            {
                "timestamp_s": (t + 1) * cfg.dt,
                "base_pos": (float(tr["x"][0, index, t]), 0.0, 0.3),
                "base_quat": (0.0, math.sin(th / 2), 0.0, math.cos(th / 2)),
                "base_lin_vel_body": (float(tr["v"][0, index, t]), 0.0, 0.0),
                "base_ang_vel_body": (0.0, float(tr["thd"][0, index, t]), 0.0),
                "joint_pos": (math.sin(ph), math.cos(ph)),
                "joint_vel": (0.0, 0.0),
                "command_vel": (float(batch.v_cmd[index]), 0.0, 0.0),
                "terminated": bool((t + 1) * cfg.dt >= fail_step - 1e-9),
            }
        )
        if rows[-1]["terminated"]:
            break
    return rows


def seed_from_frame(frame, seed_id: str, patch_start: float) -> ToySeed:
    qx, qy, qz, qw = frame.base_quat
    th = 2.0 * math.atan2(qy, qw)
    s, c = frame.joint_pos
    return ToySeed(
        seed_id=seed_id,
        x=frame.base_pos[0],
        v=frame.base_lin_vel_body[0],
        th=th,
        thd=frame.base_ang_vel_body[1],
        phase=math.atan2(s, c) % (2 * math.pi),
        v_cmd=frame.command_vel[0],
        patch_start=patch_start,
    )


TOY_CRITERION = FailureCriterion(name="toy_pole_fall_v1", max_tilt_rad=0.6, debounce_frames=1)


# --------------------------------------------------------------------------- #
# Evolution strategies
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ESConfig:
    iterations: int = 120
    pairs: int = 16
    contexts: int = 24
    sigma: float = 0.05
    learning_rate: float = 0.02
    refresh_every: int = 30

    @property
    def rollout_slots(self) -> int:
        return self.iterations * 2 * self.pairs * self.contexts


class Adam:
    def __init__(self, dim, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        mh, vh = self.m / (1 - self.b1**self.t), self.v / (1 - self.b2**self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


def _rank_shape(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(values))
    ranks[np.argsort(values)] = np.arange(len(values))
    return ranks / (len(values) - 1) - 0.5


def es_train(
    cfg: ToySlipConfig,
    es: ESConfig,
    theta0: np.ndarray,
    make_batch,
    *,
    seed: int,
    on_refresh=None,
    snapshot_at: Sequence[int] = (),
) -> tuple[np.ndarray, dict]:
    """Maximise mean reward over contexts from ``make_batch(rng, n)``.

    ``on_refresh(theta, iteration)`` is called every ``refresh_every``
    iterations before sampling, so boundary samplers can re-estimate the
    boundary with the current policy. Returns the final parameters and a log.
    """
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    opt = Adam(len(theta), es.learning_rate)
    log: dict = {"iteration": [], "mean_reward": [], "failure_rate": [], "rollout_slots": 0}
    log["snapshots"] = {}
    for it in range(es.iterations):
        if it in snapshot_at:
            log["snapshots"][it] = theta.copy()  # parameters after ``it`` updates
        if on_refresh is not None and it % es.refresh_every == 0:
            on_refresh(theta, it)
        eps = rng.standard_normal((es.pairs, len(theta)))
        population = np.concatenate([theta + es.sigma * eps, theta - es.sigma * eps])
        batch = make_batch(rng, es.contexts)
        result = simulate(cfg, population, batch)
        log["rollout_slots"] += population.shape[0] * len(batch)
        fitness = result.reward.mean(axis=1)
        shaped = _rank_shape(fitness)
        grad = (shaped[: es.pairs] - shaped[es.pairs :]) @ eps / (2 * es.pairs * es.sigma)
        theta = theta + opt.step(grad)
        log["iteration"].append(it)
        log["mean_reward"].append(float(fitness.mean()))
        log["failure_rate"].append(float(traverse_failure(result).mean()))
    return theta, log


# --------------------------------------------------------------------------- #
# Boundary estimation in the toy
# --------------------------------------------------------------------------- #

MU_GRID = tuple(float(v) for v in np.round(np.linspace(0.04, 0.8, 20), 4))


def failure_probabilities(
    cfg: ToySlipConfig,
    theta: np.ndarray,
    seeds: Sequence[ToySeed],
    mu_grid: Sequence[float],
    *,
    replicates: int,
    rng_seed: int,
    replay_steps: int,
) -> np.ndarray:
    """(len(seeds), len(mu_grid)) failure fraction from each restored seed."""
    reps = np.arange(replicates) + 1_000 * (rng_seed + 1)
    rows = []
    for s in seeds:
        mus = np.repeat(np.asarray(mu_grid, float), replicates)
        noise = np.tile(reps, len(mu_grid))
        batch = seed_batch([s] * len(mus), mus, noise)
        res = simulate(cfg, theta, batch, steps=replay_steps)
        rows.append(traverse_failure(res)[0].reshape(len(mu_grid), replicates).mean(axis=1))
    return np.array(rows)


def boundaries_for(
    cfg: ToySlipConfig,
    theta: np.ndarray,
    seeds: Sequence[ToySeed],
    *,
    seed_source: str,
    replicates: int = 8,
    rng_seed: int = 0,
    replay_steps: int = 100,
    clamp_robust: bool = False,
) -> tuple[list[BoundaryEstimate], dict]:
    """Boundary per seed; seeds with no identified crossing are dropped and counted."""
    probs = failure_probabilities(
        cfg,
        theta,
        seeds,
        MU_GRID,
        replicates=replicates,
        rng_seed=rng_seed,
        replay_steps=replay_steps,
    )
    found, dropped = [], {}
    for s, row in zip(seeds, probs):
        outcomes = {
            mu: [True] * int(round(p * replicates))
            + [False] * (replicates - int(round(p * replicates)))
            for mu, p in zip(MU_GRID, row)
        }
        try:
            found.append(estimate_boundary(s.seed_id, outcomes, seed_source=seed_source))
        except BoundaryNotIdentified as exc:
            # The censoring kind comes from the exception, never from parsing its
            # message: seed ids contain colons, and parsing the text once turned
            # every reason into a row number, so the clamp below never ran.
            reason = exc.censoring
            if clamp_robust and reason == "above_support":
                found.append(
                    BoundaryEstimate(
                        s.seed_id,
                        seed_source,
                        MU_GRID[0],
                        (MU_GRID[0], MU_GRID[0]),
                        "clamped_robust",
                        tuple(
                            (mu, int(round(p * replicates)), replicates)
                            for mu, p in zip(MU_GRID, row)
                        ),
                    )
                )
            else:
                dropped[s.seed_id] = reason
    return found, {"dropped": dropped, "probabilities": probs.tolist()}


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #


@dataclass
class ArmRun:
    name: str
    theta: np.ndarray
    log: dict
    sampler_history: list = field(default_factory=list)


def draws_to_batch(
    cfg: ToySlipConfig,
    draws: Sequence[ReplayDraw],
    seeds_by_id: dict,
    rng: np.random.Generator,
    neighborhood: LocalNeighborhood,
    noise: dict | None = None,
) -> Batch:
    nominal_idx = [i for i, d in enumerate(draws) if d.kind in ("nominal", "broad")]
    boundary_idx = [i for i, d in enumerate(draws) if d.kind == "boundary"]
    parts, order = [], []
    if nominal_idx:
        mus = [draws[i].mu for i in nominal_idx]
        b = nominal_batch(cfg, rng, len(nominal_idx))
        b.mu_patch = np.array([b.mu_patch[k] if mu is None else mu for k, mu in enumerate(mus)])
        b.noise_seed = np.array([draws[i].noise_seed for i in nominal_idx])
        parts.append(b)
        order += nominal_idx
    if boundary_idx:
        parts.append(
            seed_batch(
                [seeds_by_id[draws[i].seed_id] for i in boundary_idx],
                np.array([draws[i].mu for i in boundary_idx]),
                np.array([draws[i].noise_seed for i in boundary_idx]),
                command_offsets=np.array([draws[i].command_offset_mps for i in boundary_idx]),
                noise_scale=neighborhood.state_noise_scale,
                noise=noise,
            )
        )
        order += boundary_idx
    merged = Batch.concat(parts)
    return merged.take(np.argsort(order))


def run_arm(
    name: str,
    cfg: ToySlipConfig,
    es: ESConfig,
    theta0: np.ndarray,
    *,
    seed: int,
    neighborhood: LocalNeighborhood,
    nominal_fraction: float,
    seeds: Sequence[ToySeed] = (),
    seed_source: str | None = None,
    band: tuple[float, float] | None = None,
    snapshot_at: Sequence[int] = (),
) -> ArmRun:
    """Train one arm. B broad DR, C/D boundary replay from seeds, E friction band."""
    history: list = []
    seeds_by_id = {s.seed_id: s for s in seeds}
    state: dict = {}

    if name == "B_broad_dr":
        sampler = BroadSampler(cfg.broad_mu, rng_seed=seed)
        history.append(sampler.describe())
        state["sampler"] = sampler
        refresh = None
    elif name == "E_friction_band":
        assert band is not None
        lo, hi = band

        class _Band:
            def __init__(self):
                self.rng = np.random.default_rng(seed)

            def sample(self, n):
                out = []
                for _ in range(n):
                    ns = int(self.rng.integers(2**31 - 1))
                    if self.rng.random() < nominal_fraction:
                        out.append(ReplayDraw("nominal", noise_seed=ns))
                    else:
                        out.append(
                            ReplayDraw("broad", mu=float(self.rng.uniform(lo, hi)), noise_seed=ns)
                        )
                return out

        state["sampler"] = _Band()
        history.append(
            {"sampler": "friction_band", "band": band, "nominal_fraction": nominal_fraction}
        )
        refresh = None
    elif name in ("C_nominal_seed_boundary", "D_fbr"):
        assert seed_source is not None and seeds

        def refresh(theta, it):
            found, info = boundaries_for(
                cfg, theta, seeds, seed_source=seed_source, rng_seed=seed + it, clamp_robust=it > 0
            )
            history.append({"iteration": it, "boundaries": [b.to_dict() for b in found], **info})
            if found:
                state["sampler"] = BoundarySampler(
                    found, neighborhood, nominal_fraction=nominal_fraction, rng_seed=seed + it
                )
            elif "sampler" not in state:
                raise ValueError(f"{name}: no seed has an identified boundary")

    else:
        raise ValueError(f"unknown arm {name!r}")

    def make_batch(rng, n):
        return draws_to_batch(cfg, state["sampler"].sample(n), seeds_by_id, rng, neighborhood)

    theta, log = es_train(
        cfg, es, theta0, make_batch, seed=seed, on_refresh=refresh, snapshot_at=snapshot_at
    )
    return ArmRun(name, theta, log, history)


def informativeness(
    cfg: ToySlipConfig, theta: np.ndarray, batch: Batch, *, replicates: int = 16
) -> dict:
    """Per-context failure probability under ``theta`` and the boundary mass of the batch."""
    n = len(batch)
    idx = np.repeat(np.arange(n), replicates)
    rep = batch.take(idx)
    rep.noise_seed = rep.noise_seed * 31 + np.tile(np.arange(replicates), n)
    p = traverse_failure(simulate(cfg, theta, rep))[0].reshape(n, replicates).mean(axis=1)
    return {
        "boundary_mass": boundary_mass(p),
        "fraction_informative": float(np.mean((p > 0.1) & (p < 0.9))),
        "fraction_always_success": float(np.mean(p == 0.0)),
        "fraction_always_failure": float(np.mean(p == 1.0)),
    }


def config_dict(cfg: ToySlipConfig, es: ESConfig) -> dict:
    return {"physics": asdict(cfg), "es": asdict(es)}
