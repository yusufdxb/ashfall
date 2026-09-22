"""Where along a failed trajectory should repair replay begin? A CPU toy sweep.

Preregistered in ``docs/research/PRECURSOR_SWEEP_PREREGISTRATION.md`` before it was
run. This is a new hypothesis after FBR's negative result
(``docs/results/FBR_TOY_NEGATIVE_RESULT.md``); it names no method.

Run:
    python -m ashfall.fbr.time_sweep --stage 1 --output results/precursor_toy/<commit>
    python -m ashfall.fbr.time_sweep --stage 2 --output results/precursor_toy/<commit>

Stage 2 reads stage 1's ``stage1.json`` and refuses to run unless stage 1 licensed it.

Every replay arm reuses the toy v2 machinery unchanged (sampler, replay share,
friction band from the failure's entry boundary, reset noise, ES optimiser).
Within one failure the start-time arms share the band and the ES random stream,
so only the restored start state differs between them.

Evidence kind ``toy_mechanism``. Not a GO2 result.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

import ashfall.fbr.toy_study as v1
import ashfall.fbr.toy_study_v2 as v2
from ashfall.fbr.acceptance import accept_reconstruction
from ashfall.fbr.boundary import BoundarySampler, boundary_mass
from ashfall.fbr.capsule import extract_capsule, seed_before_position
from ashfall.fbr.toy_slip import (
    MU_GRID,
    TOY_CRITERION,
    ESConfig,
    ToySeed,
    ToySlipConfig,
    _rank_shape,
    boundaries_for,
    condition_batch,
    draws_to_batch,
    frames_from_trace,
    nominal_batch,
    seed_batch,
    seed_from_frame,
    simulate,
    traverse_failure,
)
from ashfall.provenance import content_hash, file_hash

ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "results" / "fbr_toy_v2" / "8f49f02"
BASELINE_SHA = "2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3"

OFFSETS = (2.0, 1.5, 1.0, 0.75, 0.5, 0.25)  # seconds before onset; plus T0
FIXED = tuple(f"d{d:g}" for d in OFFSETS) + ("T0",)
PRE_FAILURE = FIXED[:-1]
MIN_ONSET_S = 2.02
DISCOVERY = dict(v2.WORLDS)
HIDDEN = {
    "W7": {"mu": 0.075, "patch_start": 0.95, "v_cmd": 1.05},
    "W8": {"mu": 0.065, "patch_start": 1.15, "v_cmd": 1.2},
    "W9": {"mu": 0.11, "patch_start": 0.75, "v_cmd": 1.25},
    "W10": {"mu": 0.085, "patch_start": 1.05, "v_cmd": 0.95},
    "W11": {"mu": 0.095, "patch_start": 0.85, "v_cmd": 1.1},
    "W12": {"mu": 0.055, "patch_start": 1.0, "v_cmd": 1.15},
}
STAGE1_SEEDS = v1.TRAINING_SEEDS
STAGE2_SEEDS = (151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211)
SIM_MU = (0.05, 0.8)
SIM_SLOT = {w: i + 1 for i, w in enumerate(list(DISCOVERY) + list(HIDDEN))}
OMNIBUS_PERMUTATIONS = 20_000
RSEL_THRESHOLD = 0.5
RSEL_REPLICATES = 128
G3_MARGIN = -0.03
G5_MIN_WORLDS = 4
BASE_ITERATIONS = v2.BASE_ITERATIONS


# --------------------------------------------------------------------------- #
# Start states
# --------------------------------------------------------------------------- #


def load_baseline() -> tuple[np.ndarray, str]:
    path = V2_DIR / "baseline.npy"
    sha = file_hash(path)
    if sha != BASELINE_SHA:
        raise RuntimeError(f"baseline hash {sha} is not the frozen toy v2 baseline")
    return np.load(path), sha


def _onset_ok(capsule) -> bool:
    frames = capsule.frames
    return frames[capsule.failure_onset_index].timestamp_s - frames[0].timestamp_s >= (
        MIN_ONSET_S - 1e-9
    )


def real_capsule(cfg, theta, policy_sha, world, *, max_trials=400):
    """Toy v2's capsule rule (``toy_study_v2.entry_capsule``) plus the onset condition."""
    rejected, steps = [], 0
    saved = v1.DEPLOYMENT
    v1.DEPLOYMENT = world
    try:
        for trial, capsule in v1.deployment_failures(cfg, theta, policy_sha, max_trials=max_trials):
            if not _onset_ok(capsule):
                rejected.append({"trial": trial, "reason": "onset earlier than MIN_ONSET_S"})
                continue
            try:
                seed = seed_before_position(capsule, world["patch_start"], max_lead_s=v2.MAX_LEAD_S)
            except ValueError as exc:
                rejected.append({"trial": trial, "reason": str(exc)[:120]})
                continue
            toy_seed = seed_from_frame(seed.frame, seed.seed_id, world["patch_start"])
            onset = capsule.failure_onset_index
            observed = [asdict(f) for f in capsule.frames[seed.row : onset + 1]]
            t0 = capsule.frames[seed.row].timestamp_s
            onset_s = capsule.frames[onset].timestamp_s - t0
            horizon = onset - seed.row + int(round(0.6 / cfg.dt))
            reproduced = []
            for mu in MU_GRID:
                pairs = []
                for k in range(6):
                    ctrl = v1._replay_rows(
                        cfg, theta, toy_seed, mu, 70_000 + k, horizon, t0, control=True
                    )
                    trt = v1._replay_rows(
                        cfg, theta, toy_seed, mu, 70_000 + k, horizon, t0, control=False
                    )
                    steps += 2 * horizon
                    pairs.append(
                        (
                            [dict(observed[0])] + ctrl,
                            [dict(observed[0])] + trt,
                            v2._receipt(mu, cfg),
                        )
                    )
                verdict = accept_reconstruction(
                    mu, observed, onset_s, pairs, TOY_CRITERION, v2.ACCEPTANCE
                )
                if verdict.status == "REPRODUCED":
                    reproduced.append(mu)
            if reproduced:
                return (
                    capsule,
                    seed,
                    toy_seed,
                    {
                        "trial": trial,
                        "rejected": rejected,
                        "entry_row": seed.row,
                        "entry_lead_s": seed.lead_s,
                        "reproduced_range": [min(reproduced), max(reproduced)],
                        "hidden_mu_inside": min(reproduced) - 1e-9
                        <= world["mu"]
                        <= max(reproduced) + 1e-9,
                        "reconstruction_steps": steps,
                    },
                )
            rejected.append({"trial": trial, "reason": "UNREPRODUCED"})
    finally:
        v1.DEPLOYMENT = saved
    raise RuntimeError(f"no eligible capsule in world {world}")


def offset_row(capsule, label: str) -> int:
    """Row of the start state named ``label`` (``d<seconds>`` or ``T0``) in a capsule."""
    onset = capsule.failure_onset_index
    if label == "T0":
        return onset - 1
    d = float(label[1:])
    target = capsule.frames[onset].timestamp_s - d
    rows = [i for i in range(onset) if capsule.frames[i].timestamp_s <= target + 1e-9]
    if not rows:
        raise ValueError(f"no frame {d} s before onset")
    return rows[-1]


def _seed_from_row(row: dict, seed_id: str, patch_start: float) -> ToySeed:
    qx, qy, qz, qw = row["base_quat"]
    s, c = row["joint_pos"]
    return ToySeed(
        seed_id=seed_id,
        x=float(row["base_pos"][0]),
        v=float(row["base_lin_vel_body"][0]),
        th=2.0 * math.atan2(qy, qw),
        thd=float(row["base_ang_vel_body"][1]),
        phase=math.atan2(s, c) % (2 * math.pi),
        v_cmd=float(row["command_vel"][0]),
        patch_start=patch_start,
    )


def trajectory_seeds(capsule, patch_start: float, prefix: str, labels) -> dict:
    """ToySeeds at each labelled start state, plus their rows and lead times."""
    out = {}
    onset_t = capsule.frames[capsule.failure_onset_index].timestamp_s
    for label in labels:
        row = 0 if label == "start" else offset_row(capsule, label)
        frame = capsule.frames[row]
        out[label] = {
            "seed": seed_from_frame(frame, f"{prefix}:{label}:{row}", patch_start),
            "row": row,
            "lead_s": onset_t - frame.timestamp_s,
            "x": frame.base_pos[0],
        }
    return out


def success_rows(cfg, theta, world, *, max_trials=400):
    """The first successful baseline traverse of ``world`` (fixed seed stream), noisy rows."""
    rng = np.random.default_rng(2027)
    steps = 0
    for k in range(max_trials):
        batch = condition_batch(cfg, seeds=np.array([190_000 + k]), **world)
        res = simulate(cfg, theta, batch, record_trace=True)
        steps += cfg.steps
        if traverse_failure(res)[0, 0]:
            continue
        rows = v1._noisy_rows(frames_from_trace(res, 0, batch, cfg), rng)
        return rows, {"success_trial": k + 1, "search_steps": steps}
    raise RuntimeError(f"no successful traverse in world {world}")


def matched_seeds(rows, targets: dict, patch_start: float, prefix: str) -> dict:
    xs = np.array([r["base_pos"][0] for r in rows])
    out = {}
    for label, x_target in targets.items():
        i = int(np.argmin(np.abs(xs - x_target)))
        out[label] = {
            "seed": _seed_from_row(rows[i], f"{prefix}:{label}:{i}", patch_start),
            "row": i,
            "x": float(xs[i]),
            "x_target": float(x_target),
        }
    return out


def sim_failure(cfg, theta, policy_sha, slot: int, *, max_rollouts=5_000):
    """A failure found by simulation alone: random worlds, no deployment knowledge."""
    rng = np.random.default_rng(60_000 + slot)
    steps, tried = 0, 0
    for _ in range(max_rollouts):
        tried += 1
        mu = float(rng.uniform(*SIM_MU))
        batch = nominal_batch(cfg, rng, 1, mu=[mu])
        res = simulate(cfg, theta, batch, record_trace=True)
        steps += cfg.steps
        if not res.failed[0, 0]:
            continue
        ps = float(batch.patch_start[0])
        capsule = extract_capsule(
            frames_from_trace(res, 0, batch, cfg),
            criterion=TOY_CRITERION,
            source="toy_simulator_search",
            robot="slip_cartpole",
            policy_sha256=policy_sha,
            history_s=1.0,
            post_s=0.0,
            control_dt=cfg.dt,
            surface_terrain_metadata={
                "patch_start_m": ps,
                "patch_length_m": cfg.patch_length,
                "patch_friction": mu,
            },
        )
        if not _onset_ok(capsule):
            continue
        try:
            entry = seed_before_position(capsule, ps, max_lead_s=v2.MAX_LEAD_S)
        except ValueError:
            continue
        entry_seed = seed_from_frame(entry.frame, entry.seed_id, ps)
        bounds, _ = boundaries_for(
            cfg, theta, [entry_seed], seed_source="simulator_search", replay_steps=v2.REPLAY_STEPS
        )
        steps += len(MU_GRID) * 8 * v2.REPLAY_STEPS
        if not bounds:
            continue
        return (
            capsule,
            entry,
            entry_seed,
            bounds[0],
            {
                "rollouts_tried": tried,
                "search_steps": steps,
                "sim_world": {"mu": mu, "patch_start": ps, "v_cmd": float(batch.v_cmd[0])},
                "entry_row": entry.row,
                "entry_lead_s": entry.lead_s,
            },
        )
    raise RuntimeError(f"no simulator failure found for slot {slot}")


# --------------------------------------------------------------------------- #
# Diagnostics (baseline policy, no training)
# --------------------------------------------------------------------------- #


def recoverability(cfg, theta, seed: ToySeed, band_hi: float, *, mu=None, n=256, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    mus = np.full(n, float(mu)) if mu is not None else rng.uniform(v2.MU_MIN, band_hi, n)
    noise = np.arange(n) + 800_000
    res = simulate(cfg, theta, seed_batch([seed] * n, mus, noise, noise=v2.SEED_NOISE))
    fail = traverse_failure(res)[0]
    ft = np.minimum(res.fail_time[0], cfg.horizon_s)
    return {
        "success": float(1.0 - fail.mean()),
        "survival_s": float(ft.mean()),
        "immediate_termination": float(np.mean(res.fail_time[0] <= 0.1 + 1e-9)),
    }, n * cfg.steps


def replay_sampler(bound, seed: ToySeed, *, rng_seed: int, cfg) -> BoundarySampler:
    b = dataclasses.replace(bound, seed_id=seed.seed_id)
    return BoundarySampler(
        [b],
        v2.NEIGHBORHOOD,
        nominal_fraction=1.0 - v2.REPLAY_SHARE,
        rng_seed=rng_seed,
        one_sided=True,
        base_mu_range=cfg.broad_mu,
    )


def training_signal(cfg, theta, seed: ToySeed, bound, *, contexts=128, reps=16, grads=8):
    """Boundary mass of the replay contexts and the ES gradient at the baseline."""
    rng = np.random.default_rng(99)
    hi = min(v2.NEIGHBORHOOD.mu_max, bound.mu_interval[1] + v2.NEIGHBORHOOD.mu_half_width)
    mus = rng.uniform(v2.MU_MIN, hi, contexts)
    noise = rng.integers(0, 2**31 - 1, contexts)
    batch = seed_batch([seed] * contexts, mus, noise, noise=v2.SEED_NOISE)
    rep = batch.take(np.repeat(np.arange(contexts), reps))
    rep.noise_seed = rep.noise_seed * 31 + np.tile(np.arange(reps), contexts)
    p = traverse_failure(simulate(cfg, theta, rep))[0].reshape(contexts, reps).mean(axis=1)
    es = ESConfig()
    sampler = replay_sampler(bound, seed, rng_seed=0, cfg=cfg)
    by_id = {seed.seed_id: seed}
    grng = np.random.default_rng(1234)
    norms, spreads = [], []
    for _ in range(grads):
        eps = grng.standard_normal((es.pairs, len(theta)))
        pop = np.concatenate([theta + es.sigma * eps, theta - es.sigma * eps])
        b = draws_to_batch(
            cfg, sampler.sample(es.contexts), by_id, grng, v2.NEIGHBORHOOD, noise=v2.SEED_NOISE
        )
        fit = simulate(cfg, pop, b).reward.mean(axis=1)
        shaped = _rank_shape(fit)
        g = (shaped[: es.pairs] - shaped[es.pairs :]) @ eps / (2 * es.pairs * es.sigma)
        norms.append(float(np.linalg.norm(g)))
        spreads.append(float(fit.std()))
    return {
        "boundary_mass": boundary_mass(p),
        "fraction_informative": float(np.mean((p > 0.1) & (p < 0.9))),
        "fraction_always_failure": float(np.mean(p == 1.0)),
        "es_grad_norm": float(np.mean(norms)),
        "fitness_sd": float(np.mean(spreads)),
    }


def recoverability_curve(cfg, theta, capsule, patch_start, band_hi, *, every=5, n=128):
    onset = capsule.failure_onset_index
    onset_t = capsule.frames[onset].timestamp_s
    rows = list(range(0, onset, every))
    if rows[-1] != onset - 1:
        rows.append(onset - 1)
    out = []
    for r in rows:
        s = seed_from_frame(capsule.frames[r], f"curve:{r}", patch_start)
        rec, _ = recoverability(cfg, theta, s, band_hi, n=n)
        out.append({"row": r, "lead_s": onset_t - capsule.frames[r].timestamp_s, **rec})
    return out


# --------------------------------------------------------------------------- #
# Setup: failures, start states, bands, overheads
# --------------------------------------------------------------------------- #


def setup_world(cfg, theta, sha, name, world, labels, *, out_dir: Path, diagnostics=True):
    capsule, entry_pt, entry_seed, info = real_capsule(cfg, theta, sha, world)
    capsule.save(out_dir / f"capsule_{name}.json")
    bounds, binfo = boundaries_for(
        cfg, theta, [entry_seed], seed_source="deployment_failure", replay_steps=v2.REPLAY_STEPS
    )
    if not bounds:
        raise RuntimeError(f"{name}: entry seed has no identified boundary")
    info["boundary_steps"] = len(MU_GRID) * 8 * v2.REPLAY_STEPS
    real_overhead = info["reconstruction_steps"] + info["boundary_steps"]
    band = bounds[0]
    band_hi = min(v2.NEIGHBORHOOD.mu_max, band.mu_interval[1] + v2.NEIGHBORHOOD.mu_half_width)
    ps = world["patch_start"]

    arms: dict = {}
    real = trajectory_seeds(capsule, ps, f"real{name}", [lb for lb in labels if lb != "entry"])
    real["entry"] = {
        "seed": entry_seed,
        "row": entry_pt.row,
        "lead_s": entry_pt.lead_s,
        "x": entry_pt.frame.base_pos[0],
    }
    for label, r in real.items():
        arms[f"real_{label}"] = {
            **r,
            "bound": band,
            "overhead": real_overhead,
            "source": "deployment_failure",
        }

    rows, sinfo = success_rows(cfg, theta, world)
    match = matched_seeds(
        rows,
        {lb: real[lb]["x"] for lb in labels if lb in real and lb not in ("start", "entry")},
        ps,
        f"match{name}",
    )
    for label, m in match.items():
        arms[f"match_{label}"] = {
            **m,
            "bound": dataclasses.replace(band, seed_source="nominal_rollout"),
            "overhead": real_overhead + sinfo["search_steps"],
            "source": "nominal_rollout",
        }

    s_capsule, s_entry, s_entry_seed, s_bound, s_info = sim_failure(cfg, theta, sha, SIM_SLOT[name])
    s_capsule.save(out_dir / f"sim_capsule_{name}.json")
    s_ps = s_info["sim_world"]["patch_start"]
    sim = trajectory_seeds(
        s_capsule, s_ps, f"sim{name}", [lb for lb in labels if lb not in ("start", "entry")]
    )
    if "entry" in labels:
        sim["entry"] = {
            "seed": s_entry_seed,
            "row": s_entry.row,
            "lead_s": s_entry.lead_s,
            "x": s_entry.frame.base_pos[0],
        }
    for label, r in sim.items():
        arms[f"sim_{label}"] = {
            **r,
            "bound": s_bound,
            "overhead": s_info["search_steps"],
            "source": "simulator_search",
        }

    world_info = {
        "real": info,
        "real_capsule_id": capsule.capsule_id,
        "real_onset_s": capsule.frames[capsule.failure_onset_index].timestamp_s,
        "boundary": band.to_dict(),
        "band_hi": band_hi,
        "success": sinfo,
        "sim": {**s_info, "boundary": s_bound.to_dict(), "capsule_id": s_capsule.capsule_id},
        "sim_onset_s": s_capsule.frames[s_capsule.failure_onset_index].timestamp_s,
    }
    if diagnostics:
        diag = {}
        for arm, a in arms.items():
            hi = min(
                v2.NEIGHBORHOOD.mu_max, a["bound"].mu_interval[1] + v2.NEIGHBORHOOD.mu_half_width
            )
            rec_band, _ = recoverability(cfg, theta, a["seed"], hi)
            rec_dep, _ = recoverability(cfg, theta, a["seed"], hi, mu=world["mu"])
            diag[arm] = {
                "recoverability_band": rec_band,
                "recoverability_hidden_mu": rec_dep,
                **training_signal(cfg, theta, a["seed"], a["bound"]),
            }
        world_info["diagnostics"] = diag
        world_info["curve_real"] = recoverability_curve(cfg, theta, capsule, ps, band_hi)
        s_hi = min(v2.NEIGHBORHOOD.mu_max, s_bound.mu_interval[1] + v2.NEIGHBORHOOD.mu_half_width)
        world_info["curve_sim"] = recoverability_curve(cfg, theta, s_capsule, s_ps, s_hi)
    return arms, world_info, capsule


def rsel_arm(cfg, theta, capsule, world, band, overhead):
    """Latest frame whose baseline recoverability at the band is at least the threshold."""
    ps = world["patch_start"]
    hi = min(v2.NEIGHBORHOOD.mu_max, band.mu_interval[1] + v2.NEIGHBORHOOD.mu_half_width)
    onset = capsule.failure_onset_index
    steps = 0
    for r in range(onset - 1, -1, -1):
        s = seed_from_frame(capsule.frames[r], f"rsel:{r}", ps)
        rec, cost = recoverability(cfg, theta, s, hi, n=RSEL_REPLICATES)
        steps += cost
        if rec["success"] >= RSEL_THRESHOLD:
            onset_t = capsule.frames[onset].timestamp_s
            return {
                "seed": s,
                "row": r,
                "lead_s": onset_t - capsule.frames[r].timestamp_s,
                "bound": band,
                "overhead": overhead + steps,
                "source": "deployment_failure",
                "probe_steps": steps,
            }
    raise RuntimeError("no frame reaches the recoverability threshold")


# --------------------------------------------------------------------------- #
# Training jobs (parallel over (seed, world))
# --------------------------------------------------------------------------- #

_JOB: dict = {}


def _nominal(cfg, theta):
    m = v1.evaluate(cfg, theta)
    return {k: m[k] for k in ("nominal_success", "nominal_tracking_rmse")}


def _train_world_job(args):
    seed, wname = args
    cfg, theta0, es = _JOB["cfg"], _JOB["theta0"], _JOB["es"]
    world = _JOB["worlds"][wname]
    out = {}
    for arm, a in _JOB["arms"][wname].items():
        th, log, _ = v2.train_replay_arm(
            cfg,
            es,
            theta0,
            a["seed"],
            [replace_id(a["bound"], a["seed"])],
            seed=seed,
            seed_source=a["source"],
        )
        out[arm] = {
            "eval": v2.evaluate_world(cfg, th, world),
            "nominal": _nominal(cfg, th),
            "steps": log["rollout_slots"] * cfg.steps + a["overhead"],
        }
    return seed, wname, out


def replace_id(bound, seed: ToySeed):
    return dataclasses.replace(bound, seed_id=seed.seed_id)


def _train_global_job(seed):
    cfg, theta0 = _JOB["cfg"], _JOB["theta0"]
    worlds = _JOB["worlds"]
    th_b, log_b = v2.train_broad(cfg, _JOB["es_b"], theta0, seed=seed)
    th_adr, log_adr = v2.train_adr(cfg, _JOB["es_adr"], theta0, seed=seed)
    c_seed, c_bounds = _JOB["c_seed"], _JOB["c_bounds"]
    th_c, log_c, _ = v2.train_replay_arm(
        cfg, _JOB["es"], theta0, c_seed, c_bounds, seed=seed, seed_source="nominal_rollout"
    )
    out = {}
    for arm, th, steps in (
        ("A", theta0, 0),
        ("B", th_b, log_b["rollout_slots"] * cfg.steps),
        ("ADR", th_adr, log_adr["rollout_slots"] * cfg.steps + log_adr["probe_steps"]),
        ("C", th_c, log_c["rollout_slots"] * cfg.steps + _JOB["c_info"]["search_steps"]),
    ):
        out[arm] = {
            "eval": {w: v2.evaluate_world(cfg, th, world) for w, world in worlds.items()},
            "nominal": _nominal(cfg, th),
            "steps": steps,
        }
    return seed, out


def train_all(cfg, theta0, worlds, arms, seeds, *, base_iterations, workers):
    import multiprocessing as mp

    es = ESConfig(iterations=base_iterations)
    steps_per_iteration = 2 * es.pairs * es.contexts * cfg.steps
    c_seed, c_bounds, c_info = v2.nominal_entry_seed(cfg, theta0)
    overhead = max(
        [c_info["search_steps"]] + [a["overhead"] for w in arms.values() for a in w.values()]
    )
    b_iter = base_iterations + math.ceil(overhead / steps_per_iteration)
    adr_iter = base_iterations
    while True:
        probe = math.ceil(adr_iter / v2.ADR["eval_every"]) * v2.ADR["eval_n"] * cfg.steps
        if (adr_iter - base_iterations) * steps_per_iteration + probe >= overhead:
            break
        adr_iter += 1
    _JOB.update(
        cfg=cfg,
        theta0=theta0,
        es=es,
        es_b=ESConfig(iterations=b_iter),
        es_adr=ESConfig(iterations=adr_iter),
        worlds=worlds,
        arms=arms,
        c_seed=c_seed,
        c_bounds=c_bounds,
        c_info=c_info,
    )
    jobs = [(s, w) for s in seeds for w in worlds]
    per: dict = {}
    glob: dict = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        g_async = pool.map_async(_train_global_job, list(seeds))
        for s, w, out in pool.imap_unordered(_train_world_job, jobs):
            per.setdefault(s, {})[w] = out
            print(f"  seed {s} world {w}: {len(out)} arms", flush=True)
        for s, out in g_async.get():
            glob[s] = out
    # Compute accounting, aborting on any violation.
    for s in seeds:
        strong = min(glob[s]["B"]["steps"], glob[s]["ADR"]["steps"])
        worst = max(
            [glob[s]["C"]["steps"]] + [a["steps"] for w in per[s].values() for a in w.values()]
        )
        if worst > strong:
            raise RuntimeError(f"compute accounting violated at seed {s}: {worst} > {strong}")
    budget = {
        "base_iterations": base_iterations,
        "B_iterations": b_iter,
        "ADR_iterations": adr_iter,
        "max_out_of_training_overhead": overhead,
        "B_steps": glob[seeds[0]]["B"]["steps"],
        "ADR_steps": glob[seeds[0]]["ADR"]["steps"],
        "max_replay_arm_steps": max(
            a["steps"] for s in seeds for w in per[s].values() for a in w.values()
        ),
        "C_info": c_info,
    }
    return per, glob, budget


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def holm(pvalues: dict) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def omnibus(matrix: np.ndarray, *, permutations=OMNIBUS_PERMUTATIONS, rng_seed=2026) -> dict:
    """Variance of arm means; null permutes arm labels within each seed."""
    observed = float(np.var(matrix.mean(axis=0)))
    rng = np.random.default_rng(rng_seed)
    n, k = matrix.shape
    hits = 0
    chunk = 2_000
    for start in range(0, permutations, chunk):
        m = min(chunk, permutations - start)
        idx = np.argsort(rng.random((m, n, k)), axis=2)
        perm = np.take_along_axis(np.broadcast_to(matrix, (m, n, k)), idx, axis=2)
        stats = np.var(perm.mean(axis=1), axis=1)
        hits += int(np.sum(stats >= observed - 1e-15))
    return {
        "statistic": observed,
        "p": (hits + 1) / (permutations + 1),
        "permutations": permutations,
    }


def seed_means(per, glob, seeds, worlds, arm, key="held_out_failure"):
    if arm in ("A", "B", "ADR", "C"):
        return {s: float(np.mean([glob[s][arm]["eval"][w][key] for w in worlds])) for s in seeds}
    return {s: float(np.mean([per[s][w][arm]["eval"][key] for w in worlds])) for s in seeds}


def nominal_means(per, glob, seeds, worlds, arm, key):
    if arm in ("A", "B", "ADR", "C"):
        return {s: glob[s][arm]["nominal"][key] for s in seeds}
    return {s: float(np.mean([per[s][w][arm]["nominal"][key] for w in worlds])) for s in seeds}


def effect(base: dict, treat: dict, metric: str, confidence=0.95) -> dict:
    from ashfall.stats import paired_seed_effect

    return paired_seed_effect(base, treat, metric=metric, confidence=confidence).to_dict()


def classify(means: dict, eff: dict, p_omni: float) -> dict:
    """The preregistered curve-class rules, applied in order."""
    t_star = min(FIXED, key=lambda k: means[f"real_{k}"])
    if p_omni >= 0.05:
        return {"class": "A", "t_star": t_star, "reason": "omnibus p >= 0.05"}
    if t_star == "T0":
        return {"class": "D", "t_star": t_star, "reason": "lowest fixed offset is T0"}
    e = eff[f"real_entry-real_{t_star}"]
    if means["real_entry"] < means[f"real_{t_star}"] and e["p_exact_two_sided"] < 0.05:
        return {"class": "E", "t_star": t_star, "reason": "patch entry beats t*"}
    if t_star == PRE_FAILURE[0]:
        return {"class": "B", "t_star": t_star, "reason": "earliest fixed offset is best"}
    vs_t0 = eff[f"real_{t_star}-real_T0"]
    vs_early = eff[f"real_{t_star}-real_{PRE_FAILURE[0]}"]
    if (
        vs_t0["mean"] < 0
        and vs_t0["p_exact_two_sided"] < 0.05
        and vs_early["mean"] < 0
        and vs_early["p_exact_two_sided"] < 0.05
    ):
        return {"class": "C", "t_star": t_star, "reason": "interior optimum resolved"}
    return {"class": "A", "t_star": t_star, "reason": "interior minimum not resolved"}


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def _strip(arms):
    return {
        w: {
            a: {
                k: (v.to_dict() if hasattr(v, "to_dict") else asdict(v) if k == "seed" else v)
                for k, v in d.items()
            }
            for a, d in ws.items()
        }
        for w, ws in arms.items()
    }


def stage1(
    output: Path,
    *,
    seeds=STAGE1_SEEDS,
    worlds=DISCOVERY,
    base_iterations=BASE_ITERATIONS,
    workers=12,
    check_v2=True,
):
    cfg = ToySlipConfig()
    output.mkdir(parents=True, exist_ok=True)
    theta0, sha = load_baseline()
    labels = ["start", "entry", *FIXED]
    arms, info = {}, {}
    for name, world in worlds.items():
        a, wi, capsule = setup_world(cfg, theta0, sha, name, world, labels, out_dir=output)
        if check_v2:
            frozen = file_hash(V2_DIR / f"capsule_{name}.json")
            mine = file_hash(output / f"capsule_{name}.json")
            wi["matches_frozen_v2_capsule"] = frozen == mine
            if frozen != mine:
                raise RuntimeError(f"{name}: capsule differs from the frozen toy v2 capsule")
        arms[name], info[name] = a, wi
        print(
            f"setup {name}: onset {wi['real_onset_s']:.2f} s, band hi {wi['band_hi']:.3f}",
            flush=True,
        )
    per, glob, budget = train_all(
        cfg, theta0, worlds, arms, seeds, base_iterations=base_iterations, workers=workers
    )
    all_arms = ["A", "B", "ADR", "C"] + sorted(next(iter(arms.values())))
    means = {
        a: float(np.mean(list(seed_means(per, glob, seeds, worlds, a).values()))) for a in all_arms
    }
    sm = {a: seed_means(per, glob, seeds, worlds, a) for a in all_arms}
    eff = {}

    def cmp(a, b, key="held_out_failure"):
        eff[f"{a}-{b}"] = effect(sm[b], sm[a], key)

    for k in FIXED + ("start", "entry"):
        for ref in ("real_T0", "real_d2", "real_entry", "B", "ADR", "C", "A"):
            if f"real_{k}" != ref:
                cmp(f"real_{k}", ref)
        cmp(f"real_{k}", f"match_{k}") if f"match_{k}" in sm else None
        cmp(f"real_{k}", f"sim_{k}") if f"sim_{k}" in sm else None
    for k in FIXED:
        cmp("real_entry", f"real_{k}")
    for k in FIXED:
        for ref in FIXED:
            if k != ref and f"real_{k}-real_{ref}" not in eff:
                cmp(f"real_{k}", f"real_{ref}")
    matrix = np.array([[sm[f"real_{k}"][s] for k in FIXED] for s in seeds])
    omni = omnibus(matrix)
    holm_p = holm({k: eff[f"real_{k}-real_T0"]["p_exact_two_sided"] for k in PRE_FAILURE})
    cls = classify(means, eff, omni["p"])
    p_star = min(PRE_FAILURE, key=lambda k: means[f"real_{k}"])
    e_star = eff[f"real_{p_star}-real_T0"]
    licence = bool(e_star["mean"] < 0 and holm_p[p_star] < 0.05)
    nominal = {}
    for a in all_arms:
        for key in ("nominal_success", "nominal_tracking_rmse"):
            nominal[f"{a}:{key}"] = float(
                np.mean(list(nominal_means(per, glob, seeds, worlds, a, key).values()))
            )
    per_world = {
        w: {
            a: float(np.mean([per[s][w][a]["eval"]["held_out_failure"] for s in seeds]))
            for a in arms[w]
        }
        for w in worlds
    }
    for w in worlds:
        for a in ("A", "B", "ADR", "C"):
            per_world[w][a] = float(
                np.mean([glob[s][a]["eval"][w]["held_out_failure"] for s in seeds])
            )
    v2_check = None
    if check_v2:
        v2res = json.loads((V2_DIR / "result.json").read_text())
        v2_check = {
            w: {
                str(s): [
                    per[s][w]["real_entry"]["eval"]["held_out_failure"],
                    v2res["per_seed"]["D_fbr_entry"][str(s)][w]["held_out_failure"],
                ]
                for s in seeds
            }
            for w in worlds
        }
        v2_check["all_equal"] = all(
            abs(x[0] - x[1]) < 1e-12 for w in worlds for x in v2_check[w].values()
        )
    result = {
        "evidence_kind": "toy_mechanism",
        "preregistration": "docs/research/PRECURSOR_SWEEP_PREREGISTRATION.md",
        "baseline_sha256": sha,
        "seeds": list(seeds),
        "worlds": worlds,
        "fixed_offsets": list(FIXED),
        "spec_id": content_hash(
            {
                "seeds": list(seeds),
                "worlds": worlds,
                "fixed": list(FIXED),
                "base_iterations": base_iterations,
            }
        ),
        "setup": info,
        "arms": _strip(arms),
        "budget": budget,
        "means_held_out_failure": means,
        "nominal_means": nominal,
        "per_world_held_out_failure": per_world,
        "comparisons": eff,
        "omnibus": omni,
        "holm_pre_failure_vs_T0": holm_p,
        "curve_class": cls,
        "P_star": p_star,
        "stage2_licensed": licence,
        "v2_entry_reproduction": v2_check,
        "seed_means": {a: {str(s): v for s, v in d.items()} for a, d in sm.items()},
        "raw": {
            "per": {str(s): v for s, v in per.items()},
            "global": {str(s): v for s, v in glob.items()},
        },
    }
    (output / "stage1.json").write_text(
        json.dumps(result, indent=1, sort_keys=True, default=float) + "\n"
    )
    return result


def stage2(
    output: Path, *, seeds=STAGE2_SEEDS, worlds=HIDDEN, base_iterations=BASE_ITERATIONS, workers=12
):
    s1 = json.loads((output / "stage1.json").read_text())
    if not s1["stage2_licensed"]:
        raise SystemExit("stage 1 did not license stage 2: NO-GO, stage 2 is not run")
    p_star = s1["P_star"]
    cfg = ToySlipConfig()
    theta0, sha = load_baseline()
    out2 = output / "stage2"
    out2.mkdir(parents=True, exist_ok=True)
    labels = ["entry", p_star, "T0"]
    arms, info = {}, {}
    for name, world in worlds.items():
        a, wi, capsule = setup_world(
            cfg, theta0, sha, name, world, labels, out_dir=out2, diagnostics=False
        )
        keep = {f"real_{p_star}", "real_T0", "real_entry", f"match_{p_star}", f"sim_{p_star}"}
        a = {k: v for k, v in a.items() if k in keep}
        band = a["real_T0"]["bound"]
        a["real_Rsel"] = rsel_arm(cfg, theta0, capsule, world, band, a["real_T0"]["overhead"])
        wi["rsel"] = {
            "row": a["real_Rsel"]["row"],
            "lead_s": a["real_Rsel"]["lead_s"],
            "probe_steps": a["real_Rsel"]["probe_steps"],
        }
        arms[name], info[name] = a, wi
        print(
            f"setup {name}: onset {wi['real_onset_s']:.2f} s, "
            f"Rsel lead {wi['rsel']['lead_s']:.2f} s",
            flush=True,
        )
    per, glob, budget = train_all(
        cfg, theta0, worlds, arms, seeds, base_iterations=base_iterations, workers=workers
    )
    P = f"real_{p_star}"
    all_arms = ["A", "B", "ADR", "C"] + sorted(next(iter(arms.values())))
    sm = {a: seed_means(per, glob, seeds, worlds, a) for a in all_arms}
    means = {a: float(np.mean(list(v.values()))) for a, v in sm.items()}
    eff = {
        f"{P}-{ref}": effect(sm[ref], sm[P], "held_out_failure")
        for ref in (
            "real_T0",
            "real_entry",
            "C",
            f"match_{p_star}",
            f"sim_{p_star}",
            "B",
            "ADR",
            "A",
            "real_Rsel",
        )
    }
    eff["real_Rsel-real_T0"] = effect(sm["real_T0"], sm["real_Rsel"], "held_out_failure")

    def win(ref):
        e = eff[f"{P}-{ref}"]
        return bool(e["mean"] < 0 and e["p_exact_two_sided"] <= 0.05)

    nom_s = effect(
        nominal_means(per, glob, seeds, worlds, "A", "nominal_success"),
        nominal_means(per, glob, seeds, worlds, P, "nominal_success"),
        "nominal_success",
        confidence=0.90,
    )
    nom_t = effect(
        nominal_means(per, glob, seeds, worlds, "A", "nominal_tracking_rmse"),
        nominal_means(per, glob, seeds, worlds, P, "nominal_tracking_rmse"),
        "nominal_tracking_rmse",
        confidence=0.90,
    )
    per_world = {
        w: {
            **{
                a: float(np.mean([per[s][w][a]["eval"]["held_out_failure"] for s in seeds]))
                for a in arms[w]
            },
            **{
                a: float(np.mean([glob[s][a]["eval"][w]["held_out_failure"] for s in seeds]))
                for a in ("A", "B", "ADR", "C")
            },
        }
        for w in worlds
    }

    def worlds_better(ref):
        return int(sum(per_world[w][P] < per_world[w][ref] for w in worlds))

    better_baseline = min(means["B"], means["ADR"])
    gates = {
        "G1_beats_T0": win("real_T0"),
        "G2_beats_entry_C_match_sim": all(
            win(r) for r in ("real_entry", "C", f"match_{p_star}", f"sim_{p_star}")
        ),
        "G2_detail": {r: win(r) for r in ("real_entry", "C", f"match_{p_star}", f"sim_{p_star}")},
        "G3_beats_B_ADR": bool(win("B") and win("ADR") and means[P] - better_baseline <= G3_MARGIN),
        "G3_detail": {
            "B": win("B"),
            "ADR": win("ADR"),
            "margin_pp": 100 * (means[P] - better_baseline),
        },
        "G4_non_degradation": bool(
            nom_s["ci_low"] > -0.02 and nom_s["sd"] > 0 and nom_t["ci_high"] < 0.05
        ),
        "G4_detail": {"nominal_success": nom_s, "tracking": nom_t},
        "G5_per_world": all(
            worlds_better(r) >= G5_MIN_WORLDS for r in ("real_T0", "B", f"sim_{p_star}")
        ),
        "G5_detail": {r: worlds_better(r) for r in ("real_T0", "B", f"sim_{p_star}")},
    }
    go = all(
        gates[k]
        for k in (
            "G1_beats_T0",
            "G2_beats_entry_C_match_sim",
            "G3_beats_B_ADR",
            "G4_non_degradation",
            "G5_per_world",
        )
    )
    nominal = {
        f"{a}:{k}": float(np.mean(list(nominal_means(per, glob, seeds, worlds, a, k).values())))
        for a in all_arms
        for k in ("nominal_success", "nominal_tracking_rmse")
    }
    result = {
        "evidence_kind": "toy_mechanism",
        "P_star": p_star,
        "seeds": list(seeds),
        "worlds": worlds,
        "setup": info,
        "arms": _strip(arms),
        "budget": budget,
        "means_held_out_failure": means,
        "nominal_means": nominal,
        "per_world_held_out_failure": per_world,
        "comparisons": eff,
        "gates": gates,
        "GO": bool(go),
        "seed_means": {a: {str(s): v for s, v in d.items()} for a, d in sm.items()},
        "raw": {
            "per": {str(s): v for s, v in per.items()},
            "global": {str(s): v for s, v in glob.items()},
        },
    }
    (out2 / "stage2.json").write_text(
        json.dumps(result, indent=1, sort_keys=True, default=float) + "\n"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SWEEP_WORKERS", 12)))
    parser.add_argument("--quick", action="store_true", help="software check: tiny budget")
    args = parser.parse_args(argv)
    kw = {}
    if args.quick:
        kw = {"base_iterations": 3}
        if args.stage == 1:
            kw.update(seeds=STAGE1_SEEDS[:2], worlds={"W1": DISCOVERY["W1"], "W2": DISCOVERY["W2"]})
        else:
            kw.update(seeds=STAGE2_SEEDS[:2], worlds={"W7": HIDDEN["W7"], "W8": HIDDEN["W8"]})
    if args.stage == 1:
        r = stage1(args.output, workers=args.workers, check_v2=not args.quick, **kw)
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "curve_class",
                        "P_star",
                        "stage2_licensed",
                        "omnibus",
                        "holm_pre_failure_vs_T0",
                        "means_held_out_failure",
                    )
                },
                indent=1,
                default=float,
            )
        )
    else:
        if args.quick:
            s1 = json.loads((args.output / "stage1.json").read_text())
            s1["stage2_licensed"] = True  # software check only; never used for a real run
            (args.output / "stage1.json").write_text(json.dumps(s1))
        r = stage2(args.output, workers=args.workers, **kw)
        print(
            json.dumps(
                {k: r[k] for k in ("P_star", "GO", "gates", "means_held_out_failure")},
                indent=1,
                default=float,
            )
        )


if __name__ == "__main__":
    main()
