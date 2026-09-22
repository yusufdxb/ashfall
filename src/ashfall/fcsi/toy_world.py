"""A cart-pole whose physics can be changed one mechanism at a time.

The FCSI ground-truth toy needs two simulators of the same task: a *nominal* model and a
*pseudo-real* world that differs from it in a known, hidden way. Both are this module with
different :class:`Physics`. The task, frozen baseline policy, patch geometry and gait
excitation are the slip cart-pole of ``ashfall.fbr.toy_slip`` (toy v2 baseline
``2bb2a3bf...``); what is new is that every modelling assumption the policy's outcome could
depend on is an explicit, per-rollout parameter.

Mechanism parameters (library, nominal value):

* contact, **on the hazard surface (the patch) only**: ``mu_scale`` (patch friction multiplier,
  1), ``kinetic_ratio`` (sliding friction as a fraction of static, 0.85), ``aniso`` (friction
  multiplier when the drive force points backwards, 1). Ground contact is taken as well
  characterized by nominal operation; the surface that caused the failure is the uncertain one;
* actuation: ``force_scale`` (actuator strength, 1), ``action_delay`` (control steps between
  command and force, fractional, 0), ``force_max`` (actuator saturation in N, 30);
* mechanical: ``pole_mass`` (kg, 0.3), ``cart_damping`` (N s/m, 0);
* sensing: ``obs_delay`` (steps, fractional, 0), ``theta_bias`` (rad added to measured tilt, 0).

Episodes start in steady motion (``START_X`` before the origin, at the commanded speed), as a
walking robot approaches a hazard, not from rest; this differs from ``toy_slip``, whose
from-rest start saturates traction immediately.

Hidden-only mechanisms, deliberately absent from any identification library:

* ``strip``: an unmapped low-friction strip beyond the known patch;
* ``dropout``: phase-locked actuator dropout on the patch.

Evidence kind ``toy_mechanism``. Nothing here is a GO2 result.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ashfall.fbr.toy_slip import GRAVITY, ToySlipConfig, _unpack, lqr_gains

PARAMS = (
    "mu_scale",
    "kinetic_ratio",
    "aniso",
    "force_scale",
    "action_delay",
    "force_max",
    "pole_mass",
    "cart_damping",
    "obs_delay",
    "theta_bias",
)
NOMINAL = {
    "mu_scale": 1.0,
    "kinetic_ratio": 0.85,
    "aniso": 1.0,
    "force_scale": 1.0,
    "action_delay": 0.0,
    "force_max": 30.0,
    "pole_mass": 0.3,
    "cart_damping": 0.0,
    "obs_delay": 0.0,
    "theta_bias": 0.0,
}
MAX_DELAY = 5  # steps of action/observation history kept
CHANNELS = ("x", "v", "th", "thd", "slip")
MEAS_NOISE = {"x": 0.002, "v": 0.02, "th": 0.005, "thd": 0.03, "slip": 0.01}


@dataclass(frozen=True)
class Hidden:
    """Mechanisms only the pseudo-real world may contain."""

    strip: tuple[float, float, float] | None = None  # (offset after patch end, length, mu factor)
    dropout: tuple[float, float] | None = None  # (cos(phase) threshold, force factor) on patch
    # Added for the active-diagnosis study (development-only unknowns); None keeps FCSI unchanged.
    deadzone: float | None = None  # |delivered force| below this (N) becomes zero
    speed_friction: float | None = None  # patch friction x (1 - k |v|), floored at 0.2x

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Physics:
    values: tuple[tuple[str, float], ...] = tuple(NOMINAL.items())
    hidden: Hidden = field(default_factory=Hidden)

    @staticmethod
    def of(hidden: Hidden | None = None, **changes) -> "Physics":
        unknown = set(changes) - set(PARAMS)
        if unknown:
            raise ValueError(f"unknown mechanism parameters {sorted(unknown)}")
        return Physics(tuple({**NOMINAL, **changes}.items()), hidden or Hidden())

    def get(self, name: str) -> float:
        return dict(self.values)[name]

    def changed(self, rel_tol: float = 1e-9) -> dict:
        return {k: v for k, v in self.values if abs(v - NOMINAL[k]) > rel_tol}

    def to_dict(self) -> dict:
        return {"values": dict(self.values), "hidden": self.hidden.to_dict()}


@dataclass(frozen=True)
class Condition:
    mu_patch: float
    patch_start: float
    v_cmd: float

    def to_dict(self) -> dict:
        return asdict(self)


def _param_arrays(physics, B: int) -> dict:
    """Per-rollout parameter arrays from one Physics or a list of B Physics."""
    if isinstance(physics, Physics):
        return {k: np.full(B, v) for k, v in physics.values}
    if len(physics) != B:
        raise ValueError("need one Physics per rollout")
    return {k: np.array([p.get(k) for p in physics]) for k in PARAMS}


def policy_force(theta: np.ndarray, obs: np.ndarray, cfg: ToySlipConfig) -> np.ndarray:
    """The frozen toy policy (same math as ``toy_slip.simulate``), obs shape (B, 6)."""
    gains, w1, b1, w2, b2 = _unpack(np.atleast_2d(theta))
    k = (1.0 + gains[0]) * lqr_gains(cfg)
    hidden = np.tanh(obs @ w1[0] + b1[0])
    residual = hidden @ w2[0] + b2[0]
    linear = -(k[0] * obs[:, 0] + k[1] * obs[:, 1] + k[2] * obs[:, 2])
    return cfg.force_max * np.tanh((linear + 10.0 * residual) / cfg.force_max)


def _delayed(hist: np.ndarray, delay: np.ndarray) -> np.ndarray:
    """hist (..., MAX_DELAY+1) newest first; fractional delay by linear interpolation."""
    d = np.clip(delay, 0.0, MAX_DELAY - 1e-9)
    k = np.floor(d).astype(int)
    f = d - k
    idx = np.arange(hist.shape[0])
    return (1 - f) * hist[idx, k] + f * hist[idx, k + 1]


@dataclass
class StepState:
    """Everything needed to continue a rollout, arrays of shape (B,) or (B, MAX_DELAY+1)."""

    x: np.ndarray
    v: np.ndarray
    th: np.ndarray
    thd: np.ndarray
    ph: np.ndarray
    slip: np.ndarray
    act_hist: np.ndarray  # requested force, newest first (index 0 = previous step)
    obs_hist: np.ndarray  # (B, MAX_DELAY+1, 4): th, thd, v, slip, newest first (0 = now)


def fresh_state(cfg, n, *, x=0.0, v=0.0, th=None, thd=None, ph=None) -> StepState:
    z = np.zeros(n)
    st = StepState(
        x=np.full(n, x) if np.isscalar(x) else np.asarray(x, float),
        v=np.full(n, v) if np.isscalar(v) else np.asarray(v, float),
        th=z.copy() if th is None else np.asarray(th, float),
        thd=z.copy() if thd is None else np.asarray(thd, float),
        ph=z.copy() if ph is None else np.asarray(ph, float),
        slip=z.copy(),
        act_hist=np.zeros((n, MAX_DELAY + 1)),
        obs_hist=np.zeros((n, MAX_DELAY + 1, 4)),
    )
    st.obs_hist[:] = np.stack([st.th, st.thd, st.v, st.slip], -1)[:, None, :]
    return st


def step(
    cfg: ToySlipConfig,
    p: dict,
    hidden: Hidden,
    st: StepState,
    f_req: np.ndarray,
    cond: dict,
    noise: np.ndarray,
) -> StepState:
    """Advance one control step with requested force ``f_req`` (the logged action)."""
    M, g, ell = cfg.cart_mass, GRAVITY, cfg.pole_length
    m = p["pole_mass"]
    act_hist = np.concatenate([f_req[:, None], st.act_hist[:, :-1]], axis=1)
    f_cmd = _delayed(act_hist, p["action_delay"])
    f_act = np.clip(p["force_scale"] * f_cmd, -p["force_max"], p["force_max"])
    p0, plen, mu_p = cond["patch_start"], cfg.patch_length, cond["mu_patch"]
    on_patch = (st.x >= p0) & (st.x < p0 + plen)
    mu = np.where(on_patch, mu_p * p["mu_scale"], cfg.mu_ground)
    if hidden.strip is not None:
        off, length, fac = hidden.strip
        s0 = p0 + plen + off
        mu = np.where((st.x >= s0) & (st.x < s0 + length), mu_p * fac, mu)
    if hidden.dropout is not None:
        thr, fac = hidden.dropout
        f_act = np.where(on_patch & (np.cos(st.ph) > thr), fac * f_act, f_act)
    if hidden.deadzone is not None:
        f_act = np.where(np.abs(f_act) < hidden.deadzone, 0.0, f_act)
    if hidden.speed_friction is not None:
        fac = np.maximum(0.2, 1.0 - hidden.speed_friction * np.abs(st.v))
        mu = np.where(on_patch, mu * fac, mu)
    mu = np.where(on_patch & (f_act < 0), mu * p["aniso"], mu)
    kinetic = np.where(on_patch, p["kinetic_ratio"], NOMINAL["kinetic_ratio"])
    N = (M + m) * g
    limit = mu * N
    slipping = np.abs(f_act) > limit
    force = np.where(slipping, np.sign(f_act) * kinetic * limit, f_act)
    slip = (f_act - force) / N
    force = force - p["cart_damping"] * st.v
    tau = cfg.excitation_amp * np.sin(st.ph) + cfg.noise_torque * noise
    s, c = np.sin(st.th), np.cos(st.th)
    xdd = (force - m * g * s * c + m * ell * st.thd**2 * s - m * ell * tau * c) / (M + m * s**2)
    thdd = (g * s - xdd * c) / ell + tau
    v = st.v + xdd * cfg.dt
    x = st.x + v * cfg.dt
    thd = st.thd + thdd * cfg.dt
    th = st.th + thd * cfg.dt
    ph = st.ph + 2 * math.pi * cfg.excitation_hz * cfg.dt
    obs_now = np.stack([th, thd, v, slip], -1)
    obs_hist = np.concatenate([obs_now[:, None, :], st.obs_hist[:, :-1, :]], axis=1)
    return StepState(x, v, th, thd, ph, slip, act_hist, obs_hist)


def observe(theta_bias: np.ndarray, obs_delay: np.ndarray, st: StepState, v_cmd) -> np.ndarray:
    th, thd, v, slip = (_delayed(st.obs_hist[:, :, i], obs_delay) for i in range(4))
    th = th + theta_bias
    return np.stack([th, thd, v - v_cmd, slip, np.sin(st.ph), np.cos(st.ph)], -1)


@dataclass
class Rollout:
    """Closed-loop result. Per-step arrays (B, T) of the pre-step state and the action."""

    x: np.ndarray
    v: np.ndarray
    th: np.ndarray
    thd: np.ndarray
    ph: np.ndarray
    slip: np.ndarray
    action: np.ndarray
    fell: np.ndarray  # (B,)
    fail_time: np.ndarray  # (B,) s from rollout start, inf if no fall
    traversed: np.ndarray  # (B,)
    steps: int

    @property
    def failed(self) -> np.ndarray:
        return self.fell | ~self.traversed

    @property
    def phenotype(self) -> np.ndarray:
        return np.where(self.fell, "fall", np.where(self.traversed, "success", "stall"))


def closed_loop(
    cfg: ToySlipConfig,
    theta: np.ndarray,
    physics,
    cond: Condition,
    st: StepState,
    noise_seeds: np.ndarray,
    *,
    steps: int | None = None,
    t_offset: int = 0,
) -> Rollout:
    """Run the frozen policy from ``st`` for ``steps`` (default: to the horizon end)."""
    B = len(st.x)
    T = (cfg.steps - t_offset) if steps is None else steps
    p = _param_arrays(physics, B)
    hidden = physics.hidden if isinstance(physics, Physics) else physics[0].hidden
    noise = np.stack(
        [np.random.default_rng(int(s)).standard_normal(cfg.steps) for s in noise_seeds]
    )[:, t_offset : t_offset + T]
    c = {"patch_start": cond.patch_start, "mu_patch": cond.mu_patch}
    rec = {k: np.zeros((B, T)) for k in ("x", "v", "th", "thd", "ph", "slip", "action")}
    alive = np.ones(B, bool)
    fail_time = np.full(B, np.inf)
    for t in range(T):
        obs = observe(p["theta_bias"], p["obs_delay"], st, cond.v_cmd)
        f_req = policy_force(theta, obs, cfg)
        for k, val in (
            ("x", st.x),
            ("v", st.v),
            ("th", st.th),
            ("thd", st.thd),
            ("ph", st.ph),
            ("slip", st.slip),
            ("action", f_req),
        ):
            rec[k][:, t] = val
        st = step(cfg, p, hidden, st, f_req, c, noise[:, t])
        fell = alive & (np.abs(st.th) > cfg.fall_angle)
        fail_time = np.where(fell, (t + 1) * cfg.dt, fail_time)
        alive &= ~fell
    traversed = st.x >= cond.patch_start + cfg.patch_length
    return Rollout(**rec, fell=~alive, fail_time=fail_time, traversed=traversed, steps=T)


START_X = -0.6


def start_state(cfg, cond: Condition, seeds: np.ndarray) -> StepState:
    """Steady approach: at ``START_X``, moving at the command, random wobble and gait phase."""
    rngs = [np.random.default_rng(int(s) + 7_919) for s in seeds]
    th = np.array([r.normal(0, 0.02) for r in rngs])
    thd = np.array([r.normal(0, 0.05) for r in rngs])
    ph = np.array([r.uniform(0, 2 * math.pi) for r in rngs])
    v = np.array([cond.v_cmd + r.normal(0, 0.05) for r in rngs])
    return fresh_state(cfg, len(seeds), x=START_X, v=v, th=th, thd=thd, ph=ph)


def failure_rate(cfg, theta, physics, cond: Condition, *, n=256, base=500_000) -> float:
    seeds = np.arange(n) + base
    r = closed_loop(cfg, theta, physics, cond, start_state(cfg, cond, seeds), seeds)
    return float(r.failed.mean())


# --------------------------------------------------------------------------- #
# Logs
# --------------------------------------------------------------------------- #


@dataclass
class Log:
    """One logged real trajectory: noisy pre-step states and the commanded actions."""

    cond: Condition
    t: np.ndarray  # (T,) seconds
    x: np.ndarray
    v: np.ndarray
    th: np.ndarray
    thd: np.ndarray
    ph: np.ndarray
    slip: np.ndarray
    action: np.ndarray
    outcome: str  # "success", "fall", "stall"
    fail_time: float
    truth_effect_onset: float | None = None  # instrumented, never shown to identifiers

    def __len__(self):
        return len(self.t)

    def channel(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def to_dict(self) -> dict:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        d["cond"] = self.cond.to_dict()
        return d


def record(
    cfg,
    theta,
    physics,
    cond: Condition,
    seed: int,
    rng_meas: np.random.Generator,
    *,
    meas: float = 1.0,
) -> Log:
    """Roll one real episode and log it with measurement noise, truncated at the fall.

    ``meas = 0`` gives the noise-free log used only to instrument ground truth.
    """
    seeds = np.array([seed])
    r = closed_loop(cfg, theta, physics, cond, start_state(cfg, cond, seeds), seeds)
    T = r.steps if not r.fell[0] else int(round(r.fail_time[0] / cfg.dt))
    noisy = {
        k: getattr(r, k)[0, :T] + meas * MEAS_NOISE[k] * rng_meas.standard_normal(T)
        for k in ("x", "v", "th", "thd", "slip")
    }
    return Log(
        cond=cond,
        t=np.arange(T) * cfg.dt,
        ph=r.ph[0, :T].copy(),
        action=r.action[0, :T].copy(),
        outcome=str(r.phenotype[0]),
        fail_time=float(r.fail_time[0]),
        **noisy,
    )


def state_from_log(log: Log, idx: np.ndarray) -> StepState:
    """Reset states at log rows ``idx``, with action and observation histories from the log."""
    idx = np.asarray(idx)
    n = len(idx)
    st = fresh_state(
        ToySlipConfig(),
        n,
        x=log.x[idx],
        v=log.v[idx],
        th=log.th[idx],
        thd=log.thd[idx],
        ph=log.ph[idx],
    )
    st.slip = log.slip[idx].copy()
    for j in range(MAX_DELAY + 1):
        rows = np.clip(idx - j, 0, None)
        st.obs_hist[:, j, :] = np.stack(
            [log.th[rows], log.thd[rows], log.v[rows], log.slip[rows]], -1
        )
        arows = idx - 1 - j
        st.act_hist[:, j] = np.where(arows >= 0, log.action[np.clip(arows, 0, None)], 0.0)
    return st


def predict_open_loop(cfg, physics_list, log: Log, idx: np.ndarray, k: int) -> np.ndarray:
    """Reset at each row in ``idx``, replay the logged actions for ``k`` steps, noise-free.

    ``physics_list`` holds P candidate models. Returns (P, len(idx), 5) predicted channels at
    row ``idx + k``; residuals are ``log[idx + k] - prediction``.
    """
    idx = np.asarray(idx)
    P, n = len(physics_list), len(idx)
    rep_idx = np.tile(idx, P)
    st = state_from_log(log, rep_idx)
    phys = [ph for ph in physics_list for _ in range(n)]
    p = _param_arrays(phys, P * n)
    hidden = physics_list[0].hidden
    c = {"patch_start": log.cond.patch_start, "mu_patch": log.cond.mu_patch}
    zero = np.zeros(P * n)
    for j in range(k):
        st = step(cfg, p, hidden, st, log.action[rep_idx + j], c, zero)
    out = np.stack([st.x, st.v, st.th, st.thd, st.slip], -1)
    return out.reshape(P, n, 5)


def logged_channels(log: Log, rows: np.ndarray) -> np.ndarray:
    return np.stack([log.x[rows], log.v[rows], log.th[rows], log.thd[rows], log.slip[rows]], -1)
