"""Toy v2: the decision gate before any GO2 compute. Preregistered in
``docs/research/TOY_V2_PREREGISTRATION.md`` before it was run.

Run:  python -m ashfall.fbr.toy_study_v2 --output results/fbr_toy_v2/<commit>

Changes from toy v1, all fixed before running, each answering a review item:

* a real ADR arm on patch friction (widen the lower bound when the policy
  succeeds at it, narrow it when it does not), with its probe rollouts counted;
* the minimal FBR: one seed per capsule at patch entry (the approach, not a
  state already slipping), one boundary estimate at the baseline and no refresh,
  friction on the one-sided band [mu_min, mu_b + w], replay share 0.5 on top of
  broad DR (the arm is "broad DR plus replays"), reset noise equal to the
  capsule's sensor-noise model, no command jitter;
* C seeded the same way from a nominal rollout's patch entry;
* six hidden deployment worlds, one capsule each, twelve training seeds;
* every simulator step outside ES training is counted, and B and ADR receive at
  least as many total simulator steps as C and D.

Evidence kind ``toy_mechanism``. Not a GO2 result.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ashfall.fbr.acceptance import AcceptanceConfig, accept_reconstruction
from ashfall.fbr.boundary import BoundarySampler, BroadSampler, LocalNeighborhood, boundary_mass
from ashfall.fbr.capsule import seed_before_position
from ashfall.fbr.toy_slip import (
    MU_GRID,
    TOY_CRITERION,
    ESConfig,
    ToySeed,
    ToySlipConfig,
    boundaries_for,
    condition_batch,
    config_dict,
    draws_to_batch,
    es_train,
    initial_params,
    nominal_batch,
    seed_from_frame,
    simulate,
    traverse_failure,
)
from ashfall.fbr.toy_study import (
    TRAINING_SEEDS,
    _replay_rows,
    deployment_failures,
    evaluate,
)
from ashfall.gates import InterventionReceipt
from ashfall.ontology import Intervention
from ashfall.provenance import content_hash, file_hash

WORLDS = {
    "W1": {"mu": 0.08, "patch_start": 1.0, "v_cmd": 1.1},
    "W2": {"mu": 0.06, "patch_start": 0.8, "v_cmd": 1.0},
    "W3": {"mu": 0.10, "patch_start": 1.2, "v_cmd": 1.2},
    "W4": {"mu": 0.12, "patch_start": 0.9, "v_cmd": 1.3},
    "W5": {"mu": 0.07, "patch_start": 1.1, "v_cmd": 0.9},
    "W6": {"mu": 0.09, "patch_start": 0.7, "v_cmd": 1.15},
}
REPLAY_SHARE = 0.5
BAND_W = 0.05
MU_MIN = 0.02
MAX_LEAD_S = 2.0
REPLAY_STEPS = 100
SEED_NOISE = {"th": 0.005, "thd": 0.02, "v": 0.02, "phase": 0.0}  # the capsule's sensor noise
ACCEPTANCE = AcceptanceConfig(
    trajectory_margin=0.0,  # weakest form: treated no farther from the recording than control
    onset_tolerance_s=0.3,
    min_pass_fraction=2 / 3,
    scales={"tilt_rad": 0.1, "base_ang_vel_body": 0.5, "base_lin_vel_body": 0.2},
)
ADR = {
    "lo0": 0.35,
    "hi": 0.8,
    "step": 0.03,
    "t_high": 0.8,
    "t_low": 0.5,
    "eval_every": 10,
    "eval_n": 64,
    "lo_min": MU_MIN,
}
BASE_ITERATIONS = 120
ARMS = ("A_no_repair", "B_broad_dr", "ADR", "C_entry", "D_fbr_entry")
NEIGHBORHOOD = LocalNeighborhood(
    mu_half_width=BAND_W, mu_min=MU_MIN, mu_max=0.8, command_jitter_mps=0.0
)


def held_out(world: dict) -> dict:
    mu, ps, v = world["mu"], world["patch_start"], world["v_cmd"]
    return {
        "h1_easier": {"mu": mu + 0.03, "patch_start": ps, "v_cmd": v},
        "h1_harder": {"mu": max(0.03, mu - 0.02), "patch_start": ps, "v_cmd": v},
        "h2_patch_near": {"mu": mu, "patch_start": ps - 0.3, "v_cmd": v},
        "h2_patch_far": {"mu": mu, "patch_start": ps + 0.3, "v_cmd": v},
        "h3_slower": {"mu": mu, "patch_start": ps, "v_cmd": v - 0.25},
        "h3_faster": {"mu": mu, "patch_start": ps, "v_cmd": v + 0.25},
    }


def evaluate_world(cfg, theta, world, *, episodes=256):
    seeds = np.arange(episodes) + 500_000
    out = {}
    for name, cond in {**held_out(world), "deployment_condition": world}.items():
        out[name] = float(
            traverse_failure(simulate(cfg, theta, condition_batch(cfg, seeds=seeds, **cond))).mean()
        )
    out["held_out_failure"] = float(np.mean([out[k] for k in held_out(world)]))
    return out


def _receipt(mu, cfg):
    coefficients = {"static_friction": mu, "dynamic_friction": mu * cfg.kinetic_ratio}
    return InterventionReceipt(
        Intervention("friction_reduction", tuple(coefficients.items())),
        dict(coefficients),
        dict(coefficients),
        "toy_parameter_readback",
    )


def entry_capsule(cfg, theta, policy_sha, world, *, max_trials=400):
    """First deployment failure whose patch-entry seed passes acceptance; counts steps spent."""
    import ashfall.fbr.toy_study as v1

    rejected, steps = [], 0
    saved = v1.DEPLOYMENT
    v1.DEPLOYMENT = world  # deployment_failures reads the module constant
    try:
        for trial, capsule in deployment_failures(cfg, theta, policy_sha, max_trials=max_trials):
            try:
                seed = seed_before_position(capsule, world["patch_start"], max_lead_s=MAX_LEAD_S)
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
                    ctrl = _replay_rows(
                        cfg, theta, toy_seed, mu, 70_000 + k, horizon, t0, control=True
                    )
                    trt = _replay_rows(
                        cfg, theta, toy_seed, mu, 70_000 + k, horizon, t0, control=False
                    )
                    steps += 2 * horizon
                    pairs.append(
                        ([dict(observed[0])] + ctrl, [dict(observed[0])] + trt, _receipt(mu, cfg))
                    )
                verdict = accept_reconstruction(
                    mu, observed, onset_s, pairs, TOY_CRITERION, ACCEPTANCE
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
                        "seed_row": seed.row,
                        "lead_s": seed.lead_s,
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
    raise RuntimeError(f"no reproduced capsule in world {world}")


def nominal_entry_seed(cfg, theta, *, max_rollouts=100):
    """Arm C's seed: patch entry of the first successful nominal rollout with a boundary."""
    rng = np.random.default_rng(4242)
    steps = 0
    for attempt in range(max_rollouts):
        batch = nominal_batch(cfg, rng, 1)
        res = simulate(cfg, theta, batch, record_trace=True)
        steps += cfg.steps
        if traverse_failure(res)[0, 0]:
            continue
        x = res.trace["x"][0, 0]
        before = np.flatnonzero(x < batch.patch_start[0])
        if len(before) == 0:
            continue
        t = int(before[-1])
        seed = ToySeed(
            f"nominal{attempt}:{t}",
            float(x[t]),
            float(res.trace["v"][0, 0, t]),
            float(res.trace["th"][0, 0, t]),
            float(res.trace["thd"][0, 0, t]),
            float(res.trace["phase"][0, 0, t]) % (2 * math.pi),
            float(batch.v_cmd[0]),
            float(batch.patch_start[0]),
        )
        found, _ = boundaries_for(
            cfg,
            theta,
            [seed],
            seed_source="nominal_rollout",
            rng_seed=attempt,
            replay_steps=REPLAY_STEPS,
        )
        steps += len(MU_GRID) * 8 * REPLAY_STEPS
        if found:
            return seed, found, {"nominal_rollouts_tried": attempt + 1, "search_steps": steps}
    raise RuntimeError("no nominal entry seed with an identified boundary")


def train_replay_arm(cfg, es, theta0, seed_obj, boundaries, *, seed, seed_source):
    sampler = BoundarySampler(
        boundaries,
        NEIGHBORHOOD,
        nominal_fraction=1.0 - REPLAY_SHARE,
        rng_seed=seed,
        one_sided=True,
        base_mu_range=cfg.broad_mu,
    )
    by_id = {seed_obj.seed_id: seed_obj}

    def make_batch(rng, n):
        return draws_to_batch(cfg, sampler.sample(n), by_id, rng, NEIGHBORHOOD, noise=SEED_NOISE)

    theta, log = es_train(cfg, es, theta0, make_batch, seed=seed)
    return theta, log, sampler.describe()


def train_broad(cfg, es, theta0, *, seed):
    sampler = BroadSampler(cfg.broad_mu, rng_seed=seed)

    def make_batch(rng, n):
        return draws_to_batch(cfg, sampler.sample(n), {}, rng, NEIGHBORHOOD)

    return es_train(cfg, es, theta0, make_batch, seed=seed)


def train_adr(cfg, es, theta0, *, seed):
    """One-parameter ADR on the patch-friction lower bound (OpenAI et al. 2019, simplified)."""
    state = {"lo": ADR["lo0"], "history": [], "probe_steps": 0}
    eval_rng = np.random.default_rng(seed + 99_991)

    def refresh(theta, it):
        batch = nominal_batch(cfg, eval_rng, ADR["eval_n"], mu=np.full(ADR["eval_n"], state["lo"]))
        success = 1.0 - float(traverse_failure(simulate(cfg, theta, batch)).mean())
        state["probe_steps"] += ADR["eval_n"] * cfg.steps
        if success >= ADR["t_high"]:
            state["lo"] = max(ADR["lo_min"], state["lo"] - ADR["step"])
        elif success <= ADR["t_low"]:
            state["lo"] = min(ADR["hi"] - ADR["step"], state["lo"] + ADR["step"])
        state["history"].append({"iteration": it, "success_at_lo": success, "lo": state["lo"]})

    def make_batch(rng, n):
        return nominal_batch(cfg, rng, n, mu=rng.uniform(state["lo"], ADR["hi"], n))

    adr_es = ESConfig(**{**asdict(es), "refresh_every": ADR["eval_every"]})
    theta, log = es_train(cfg, adr_es, theta0, make_batch, seed=seed, on_refresh=refresh)
    log["adr"] = state["history"]
    log["probe_steps"] = state["probe_steps"]
    return theta, log


def run_study(
    output: Path,
    *,
    training_seeds=TRAINING_SEEDS,
    base_iterations=BASE_ITERATIONS,
    worlds=WORLDS,
    baseline_iterations=200,
):
    from ashfall.stats import paired_seed_effect

    cfg = ToySlipConfig()
    output.mkdir(parents=True, exist_ok=True)
    es_base = ESConfig(iterations=base_iterations)
    steps_per_iteration = 2 * es_base.pairs * es_base.contexts * cfg.steps
    spec = {
        "toy": config_dict(cfg, es_base),
        "worlds": worlds,
        "held_out": {k: held_out(w) for k, w in worlds.items()},
        "training_seeds": list(training_seeds),
        "replay_share": REPLAY_SHARE,
        "band_w": BAND_W,
        "mu_min": MU_MIN,
        "max_lead_s": MAX_LEAD_S,
        "seed_noise": SEED_NOISE,
        "acceptance": asdict(ACCEPTANCE),
        "adr": ADR,
        "base_iterations": base_iterations,
        "compute_rule": "B and ADR get extra ES iterations until their total simulator steps "
        "are at least the largest out-of-training overhead of C or D plus base training",
        "baseline_iterations": baseline_iterations,
        "arms": ARMS,
        "evidence_kind": "toy_mechanism",
        "preregistration": "docs/research/TOY_V2_PREREGISTRATION.md",
    }
    spec_id = content_hash(spec)

    theta0, _ = es_train(
        cfg,
        ESConfig(iterations=baseline_iterations),
        initial_params(),
        lambda g, n: nominal_batch(cfg, g, n),
        seed=0,
    )
    np.save(output / "baseline.npy", theta0)
    policy_sha = file_hash(output / "baseline.npy")

    # Capsules and boundaries (one per world), then C's seed. All steps counted.
    capsules = {}
    for name, world in worlds.items():
        capsule, seed_pt, toy_seed, info = entry_capsule(cfg, theta0, policy_sha, world)
        bounds, binfo = boundaries_for(
            cfg, theta0, [toy_seed], seed_source="deployment_failure", replay_steps=REPLAY_STEPS
        )
        info["boundary_steps"] = len(MU_GRID) * 8 * REPLAY_STEPS
        if not bounds:
            raise RuntimeError(f"{name}: entry seed has no identified boundary: {binfo['dropped']}")
        capsule.save(output / f"capsule_{name}.json")
        capsules[name] = {"seed": toy_seed, "bounds": bounds, "info": info}
    c_seed, c_bounds, c_info = nominal_entry_seed(cfg, theta0)

    info_mass = {}
    rng_probe = np.random.default_rng(777)

    def mass(sampler, by_id, noise=None, n=256, reps=16):
        batch = draws_to_batch(cfg, sampler.sample(n), by_id, rng_probe, NEIGHBORHOOD, noise=noise)
        rep = batch.take(np.repeat(np.arange(n), reps))
        rep.noise_seed = rep.noise_seed * 31 + np.tile(np.arange(reps), n)
        p = traverse_failure(simulate(cfg, theta0, rep))[0].reshape(n, reps).mean(axis=1)
        return boundary_mass(p)

    info_mass["B_broad_dr"] = mass(BroadSampler(cfg.broad_mu, rng_seed=1), {})
    info_mass["C_entry"] = mass(
        BoundarySampler(
            c_bounds,
            NEIGHBORHOOD,
            nominal_fraction=1 - REPLAY_SHARE,
            rng_seed=1,
            one_sided=True,
            base_mu_range=cfg.broad_mu,
        ),
        {c_seed.seed_id: c_seed},
        SEED_NOISE,
    )
    for name, c in capsules.items():
        info_mass[f"D_fbr_entry:{name}"] = mass(
            BoundarySampler(
                c["bounds"],
                NEIGHBORHOOD,
                nominal_fraction=1 - REPLAY_SHARE,
                rng_seed=1,
                one_sided=True,
                base_mu_range=cfg.broad_mu,
            ),
            {c["seed"].seed_id: c["seed"]},
            SEED_NOISE,
        )

    # Charge the out-of-training simulation of C and D to B and ADR: both get
    # extra ES iterations until their total steps cover the largest overhead.
    overhead = max(
        [c_info["search_steps"]]
        + [
            c["info"]["reconstruction_steps"] + c["info"]["boundary_steps"]
            for c in capsules.values()
        ]
    )
    b_iterations = base_iterations + math.ceil(overhead / steps_per_iteration)
    adr_iterations = base_iterations
    while True:
        adr_probe = math.ceil(adr_iterations / ADR["eval_every"]) * ADR["eval_n"] * cfg.steps
        if (adr_iterations - base_iterations) * steps_per_iteration + adr_probe >= overhead:
            break
        adr_iterations += 1
    es_b = ESConfig(iterations=b_iterations)
    es_adr = ESConfig(iterations=adr_iterations)
    per = {arm: {} for arm in ARMS}  # arm -> seed -> world -> metrics
    nominal = {arm: {} for arm in ARMS}
    budget = {arm: {} for arm in ARMS}

    def nominal_metrics(theta):
        m = evaluate(cfg, theta)
        return {
            "nominal_success": m["nominal_success"],
            "nominal_tracking_rmse": m["nominal_tracking_rmse"],
        }

    nominal_a = nominal_metrics(theta0)
    for s in training_seeds:
        th_b, log_b = train_broad(cfg, es_b, theta0, seed=s)
        th_adr, log_adr = train_adr(cfg, es_adr, theta0, seed=s)
        th_c, log_c, _ = train_replay_arm(
            cfg, es_base, theta0, c_seed, c_bounds, seed=s, seed_source="nominal_rollout"
        )
        budget["B_broad_dr"][s] = log_b["rollout_slots"] * cfg.steps
        budget["ADR"][s] = log_adr["rollout_slots"] * cfg.steps + log_adr["probe_steps"]
        budget["C_entry"][s] = log_c["rollout_slots"] * cfg.steps + c_info["search_steps"]
        per["A_no_repair"][s] = {
            w: evaluate_world(cfg, theta0, world) for w, world in worlds.items()
        }
        per["B_broad_dr"][s] = {w: evaluate_world(cfg, th_b, world) for w, world in worlds.items()}
        per["ADR"][s] = {w: evaluate_world(cfg, th_adr, world) for w, world in worlds.items()}
        per["C_entry"][s] = {w: evaluate_world(cfg, th_c, world) for w, world in worlds.items()}
        nominal["A_no_repair"][s] = nominal_a
        nominal["B_broad_dr"][s] = nominal_metrics(th_b)
        nominal["ADR"][s] = nominal_metrics(th_adr)
        nominal["C_entry"][s] = nominal_metrics(th_c)
        per["D_fbr_entry"][s], d_nom, d_budget = {}, [], []
        for w, c in capsules.items():
            th_d, log_d, _ = train_replay_arm(
                cfg,
                es_base,
                theta0,
                c["seed"],
                c["bounds"],
                seed=s,
                seed_source="deployment_failure",
            )
            per["D_fbr_entry"][s][w] = evaluate_world(cfg, th_d, worlds[w])
            d_nom.append(nominal_metrics(th_d))
            d_budget.append(
                log_d["rollout_slots"] * cfg.steps
                + c["info"]["reconstruction_steps"]
                + c["info"]["boundary_steps"]
            )
        nominal["D_fbr_entry"][s] = {k: float(np.mean([d[k] for d in d_nom])) for k in d_nom[0]}
        budget["D_fbr_entry"][s] = max(d_budget)
        budget["A_no_repair"][s] = 0
    for s in training_seeds:
        for arm in ("C_entry", "D_fbr_entry"):
            for strong in ("B_broad_dr", "ADR"):
                if budget[arm][s] > budget[strong][s]:
                    raise RuntimeError(
                        f"compute accounting violated: {arm} {budget[arm][s]} > "
                        f"{strong} {budget[strong][s]}"
                    )

    def seed_mean(arm, key):
        return {s: float(np.mean([per[arm][s][w][key] for w in worlds])) for s in training_seeds}

    def nom(arm, key):
        return {s: nominal[arm][s][key] for s in training_seeds}

    comparisons = {}
    for key in ("held_out_failure", "deployment_condition"):
        for a, b in (
            ("ADR", "D_fbr_entry"),
            ("B_broad_dr", "D_fbr_entry"),
            ("C_entry", "D_fbr_entry"),
            ("B_broad_dr", "ADR"),
            ("A_no_repair", "D_fbr_entry"),
            ("A_no_repair", "B_broad_dr"),
            ("A_no_repair", "ADR"),
        ):
            comparisons[f"{key}:{b}-{a}"] = paired_seed_effect(
                seed_mean(a, key), seed_mean(b, key), metric=key
            ).to_dict()
    for key in ("nominal_success", "nominal_tracking_rmse"):
        for arm in ("D_fbr_entry", "B_broad_dr", "ADR", "C_entry"):
            e90 = paired_seed_effect(
                nom("A_no_repair", key), nom(arm, key), metric=key, confidence=0.90
            )
            comparisons[f"{key}:{arm}-A_no_repair"] = e90.to_dict()

    d_adr = comparisons["held_out_failure:D_fbr_entry-ADR"]
    d_b = comparisons["held_out_failure:D_fbr_entry-B_broad_dr"]
    ni_s = comparisons["nominal_success:D_fbr_entry-A_no_repair"]
    ni_t = comparisons["nominal_tracking_rmse:D_fbr_entry-A_no_repair"]
    iut = (
        d_adr["mean"] < 0
        and d_adr["p_exact_two_sided"] <= 0.05
        and d_b["mean"] < 0
        and d_b["p_exact_two_sided"] <= 0.05
    )
    ni = ni_s["ci_low"] > -0.02 and ni_s["sd"] > 0 and ni_t["ci_high"] < 0.05
    go = bool(iut and d_adr["mean"] <= -0.05 and ni)
    decision = {
        "iut_rejects": bool(iut),
        "d_minus_adr_pp": 100 * d_adr["mean"],
        "non_degradation_passes": bool(ni),
        "GO_for_GO2_E0": go,
        "rule": "GO iff D beats both ADR and B (exact sign-flip, alpha 0.05 each), "
        "D minus ADR <= -5 pp, and non-degradation passes",
    }
    per_world = {
        w: {
            arm: float(np.mean([per[arm][s][w]["held_out_failure"] for s in training_seeds]))
            for arm in ARMS
        }
        for w in worlds
    }
    result = {
        "spec_id": spec_id,
        "spec": spec,
        "evidence_kind": "toy_mechanism",
        "baseline_sha256": policy_sha,
        "capsules": {
            w: {**c["info"], "boundary": [b.to_dict() for b in c["bounds"]]}
            for w, c in capsules.items()
        },
        "arm_C_seed": {**c_info, "boundary": [b.to_dict() for b in c_bounds]},
        "boundary_mass_at_baseline": info_mass,
        "simulator_steps_per_seed": {arm: budget[arm][training_seeds[0]] for arm in ARMS},
        "iterations": {
            "base": base_iterations,
            "B": b_iterations,
            "ADR": adr_iterations,
            "out_of_training_overhead_steps": overhead,
        },
        "per_world_held_out_failure": per_world,
        "summary": {
            arm: {
                "held_out_failure": float(
                    np.mean(list(seed_mean(arm, "held_out_failure").values()))
                ),
                "deployment_condition": float(
                    np.mean(list(seed_mean(arm, "deployment_condition").values()))
                ),
                "nominal_success": float(np.mean(list(nom(arm, "nominal_success").values()))),
                "nominal_tracking_rmse": float(
                    np.mean(list(nom(arm, "nominal_tracking_rmse").values()))
                ),
            }
            for arm in ARMS
        },
        "comparisons": comparisons,
        "decision": decision,
        "per_seed": {arm: {str(s): v for s, v in d.items()} for arm, d in per.items()},
        "nominal_per_seed": {arm: {str(s): v for s, v in d.items()} for arm, d in nominal.items()},
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--quick", action="store_true", help="software check: 2 seeds, 2 worlds, tiny budget"
    )
    args = parser.parse_args(argv)
    if args.quick:
        result = run_study(
            args.output,
            training_seeds=TRAINING_SEEDS[:2],
            base_iterations=6,
            worlds={k: WORLDS[k] for k in ("W1", "W2")},
            baseline_iterations=30,
        )
        print(
            json.dumps(
                {
                    "steps": result["simulator_steps_per_seed"],
                    "capsules": {
                        w: {k: c[k] for k in ("trial", "lead_s", "reproduced_range")}
                        for w, c in result["capsules"].items()
                    },
                },
                indent=1,
            )
        )
    else:
        result = run_study(args.output)
        print(
            json.dumps(
                {
                    "summary": result["summary"],
                    "decision": result["decision"],
                    "per_world": result["per_world_held_out_failure"],
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
