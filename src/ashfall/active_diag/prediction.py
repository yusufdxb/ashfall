"""Run probes: predicted under candidate models, or executed in the (pseudo-)real world.

Both use the FCSI toy physics (``fcsi.toy_world.step``) and the frozen policy; the probe force
is added to the policy's command. Execution in the real world has an abort rule: once the tilt
exceeds ``ABORT_TILT`` the excitation is switched off for the rest of the crossing (the policy
keeps running) and the probe is flagged as a safety violation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ashfall.fcsi.toy_world import (
    MEAS_NOISE,
    Log,
    _param_arrays,
    observe,
    policy_force,
    start_state,
    step,
)

from .probe_space import Probe

ABORT_TILT = 0.35


@dataclass
class ProbeRollouts:
    """Arrays (B, T): pre-step states, total command, probe force; per-rollout safety data."""

    x: np.ndarray
    v: np.ndarray
    th: np.ndarray
    thd: np.ndarray
    ph: np.ndarray
    slip: np.ndarray
    action: np.ndarray
    probe_force: np.ndarray
    fell: np.ndarray
    fail_step: np.ndarray
    max_tilt: np.ndarray
    max_speed: np.ndarray
    aborted: np.ndarray
    steps: int


def run(cfg, theta, physics_list, probe: Probe, seeds, *, abort: bool = False) -> ProbeRollouts:
    """One probe crossing for each (physics, seed) pair; ``physics_list`` has one entry per seed."""
    seeds = np.asarray(seeds)
    B, T = len(seeds), cfg.steps
    cond = probe.condition
    st = start_state(cfg, cond, seeds)
    p = _param_arrays(physics_list, B)
    hidden = physics_list[0].hidden
    c = {"patch_start": cond.patch_start, "mu_patch": cond.mu_patch}
    noise = np.stack([np.random.default_rng(int(s)).standard_normal(T) for s in seeds])
    rec = {k: np.zeros((B, T)) for k in ("x", "v", "th", "thd", "ph", "slip", "action", "u")}
    alive = np.ones(B, bool)
    fail_step = np.full(B, T)
    max_tilt = np.zeros(B)
    max_speed = np.zeros(B)
    aborted = np.zeros(B, bool)
    for t in range(T):
        obs = observe(p["theta_bias"], p["obs_delay"], st, cond.v_cmd)
        u = probe.force(np.full(B, t * cfg.dt), st.x)
        if abort:
            u = np.where(aborted, 0.0, u)
        f = policy_force(theta, obs, cfg) + u
        for k, val in (
            ("x", st.x),
            ("v", st.v),
            ("th", st.th),
            ("thd", st.thd),
            ("ph", st.ph),
            ("slip", st.slip),
            ("action", f),
            ("u", u),
        ):
            rec[k][:, t] = val
        st = step(cfg, p, hidden, st, f, c, noise[:, t])
        tilt = np.abs(st.th)
        max_tilt = np.where(alive, np.maximum(max_tilt, tilt), max_tilt)
        max_speed = np.where(alive, np.maximum(max_speed, np.abs(st.v)), max_speed)
        if abort:
            aborted |= alive & (tilt > ABORT_TILT)
        fell = alive & (tilt > cfg.fall_angle)
        fail_step = np.where(fell, t + 1, fail_step)
        alive &= ~fell
    return ProbeRollouts(
        rec["x"],
        rec["v"],
        rec["th"],
        rec["thd"],
        rec["ph"],
        rec["slip"],
        rec["action"],
        rec["u"],
        ~alive,
        fail_step,
        max_tilt,
        max_speed,
        aborted,
        T,
    )


def to_log(cfg, r: ProbeRollouts, i: int, probe: Probe, rng: np.random.Generator | None):
    """Rollout ``i`` as a logged trajectory (measurement noise if ``rng``), cut at a fall."""
    T = int(r.fail_step[i])
    noisy = {}
    for k in ("x", "v", "th", "thd", "slip"):
        a = getattr(r, k)[i, :T].copy()
        if rng is not None:
            a = a + MEAS_NOISE[k] * rng.standard_normal(T)
        noisy[k] = a
    lg = Log(
        cond=probe.condition,
        t=np.arange(T) * cfg.dt,
        ph=r.ph[i, :T].copy(),
        action=r.action[i, :T].copy(),
        outcome="fall" if r.fell[i] else "success",
        fail_time=float(T * cfg.dt) if r.fell[i] else float("inf"),
        **noisy,
    )
    return lg, r.probe_force[i, :T].copy()


def execute(cfg, theta, real, probe: Probe, seed: int, rng_meas: np.random.Generator):
    """Run a probe in the real world with the abort rule; return the log and safety facts."""
    r = run(cfg, theta, [real], probe, np.array([seed]), abort=True)
    lg, u = to_log(cfg, r, 0, probe, rng_meas)
    return (
        lg,
        u,
        {
            "fell": bool(r.fell[0]),
            "aborted": bool(r.aborted[0]),
            "max_tilt": float(r.max_tilt[0]),
            "max_speed": float(r.max_speed[0]),
            "violation": bool(r.fell[0] or r.max_tilt[0] > ABORT_TILT),
        },
    )
