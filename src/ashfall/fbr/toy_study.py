"""The slip cart-pole study: arms A to E under one budget, one capsule, twelve seeds.

Run:  python -m ashfall.fbr.toy_study --output results/fbr_toy

Order of operations, fixed before any arm is trained:

1. Train the baseline once on the nominal distribution.
2. Deployment trials: run the baseline in the hidden deployment world until
   its first failure (the toy's "first reviewed failure" rule). Extract the
   capsule with the real FBR code, with sensor noise on the recorded channels.
   The deployment friction is not written into the capsule.
3. Reconstruction: from the capsule's seed state, matched (control, treated)
   replays over a friction grid; the three acceptance checks decide which
   friction values REPRODUCE the failure.
4. Seeds for arm C: the first nominal rollout whose states, at the same timing
   relative to patch entry as the capsule's seeds, have an identified boundary.
5. Train arms B, C, D, E for each training seed; evaluate A to E on the frozen
   held-out cells with common random numbers.

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
from ashfall.fbr.boundary import LocalNeighborhood, boundary_mass
from ashfall.fbr.capsule import extract_capsule, seed_points
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
    frames_from_trace,
    initial_params,
    nominal_batch,
    run_arm,
    seed_batch,
    seed_from_frame,
    simulate,
    traverse_failure,
)
from ashfall.gates import InterventionReceipt
from ashfall.ontology import Intervention
from ashfall.provenance import content_hash, file_hash

TRAINING_SEEDS = (11, 23, 47, 59, 71, 83, 97, 101, 113, 127, 131, 149)
DEPLOYMENT = {"mu": 0.08, "patch_start": 1.0, "v_cmd": 1.1}
HELD_OUT = {
    "h1_easier": {"mu": 0.11, "patch_start": 1.0, "v_cmd": 1.1},
    "h1_harder": {"mu": 0.06, "patch_start": 1.0, "v_cmd": 1.1},
    "h2_patch_near": {"mu": 0.08, "patch_start": 0.7, "v_cmd": 1.1},
    "h2_patch_far": {"mu": 0.08, "patch_start": 1.3, "v_cmd": 1.1},
    "h3_slower": {"mu": 0.08, "patch_start": 1.0, "v_cmd": 0.85},
    "h3_faster": {"mu": 0.08, "patch_start": 1.0, "v_cmd": 1.35},
}
LEAD_TIMES = (0.2, 0.4, 0.6)
NOMINAL_FRACTION = 0.3
SENSOR_NOISE = {"th": 0.005, "thd": 0.02, "v": 0.02}
NEIGHBORHOOD = LocalNeighborhood(mu_half_width=0.05, mu_min=0.02, mu_max=0.8)
ARMS = ("A_no_repair", "B_broad_dr", "C_nominal_seed_boundary", "D_fbr", "E_friction_band")


def _noisy_rows(rows, rng):
    out = []
    for r in rows:
        r = dict(r)
        qx, qy, qz, qw = r["base_quat"]
        th = 2 * math.atan2(qy, qw) + rng.normal(0, SENSOR_NOISE["th"])
        r["base_quat"] = (0.0, math.sin(th / 2), 0.0, math.cos(th / 2))
        r["base_ang_vel_body"] = (
            0.0,
            r["base_ang_vel_body"][1] + rng.normal(0, SENSOR_NOISE["thd"]),
            0.0,
        )
        r["base_lin_vel_body"] = (
            r["base_lin_vel_body"][0] + rng.normal(0, SENSOR_NOISE["v"]),
            0.0,
            0.0,
        )
        out.append(r)
    return out


def _replay_rows(cfg, theta, seed: ToySeed, mu, noise_seed, steps, t0, *, control):
    batch = seed_batch([seed], np.array([mu]), np.array([noise_seed]), noise_scale=0.0)
    res = simulate(
        cfg,
        theta,
        batch,
        steps=steps,
        record_trace=True,
        friction_override=cfg.mu_ground if control else None,
    )
    rows = frames_from_trace(res, 0, batch, cfg)
    for i, r in enumerate(rows):
        r["timestamp_s"] = t0 + (i + 1) * cfg.dt
    return rows


def deployment_failures(cfg, theta, policy_sha, *, max_trials=400):
    """Baseline trials in the deployment world, yielding each failure as a capsule in order."""
    rng = np.random.default_rng(2026)
    for trial in range(max_trials):
        batch = condition_batch(cfg, seeds=np.array([90_000 + trial]), **DEPLOYMENT)
        res = simulate(cfg, theta, batch, record_trace=True)
        if not res.failed[0, 0]:
            continue
        rows = _noisy_rows(frames_from_trace(res, 0, batch, cfg), rng)
        capsule = extract_capsule(
            rows,
            criterion=TOY_CRITERION,
            source="toy_deployment_world",
            robot="slip_cartpole",
            policy_sha256=policy_sha,
            history_s=1.0,
            post_s=0.0,
            control_dt=cfg.dt,
            surface_terrain_metadata={
                "patch_start_m": DEPLOYMENT["patch_start"],
                "patch_length_m": cfg.patch_length,
                "patch_friction": None,
            },
        )
        yield trial + 1, capsule


def reconstruct(cfg, theta, capsule, seeds, *, replicates=6):
    """Acceptance checks over the friction grid from the 0.4 s seed."""
    seed_pt = min(seeds, key=lambda p: abs(p.lead_s - 0.4))
    toy_seed = seed_from_frame(seed_pt.frame, seed_pt.seed_id, DEPLOYMENT["patch_start"])
    onset = capsule.failure_onset_index
    observed = [asdict(f) for f in capsule.frames[seed_pt.row : onset + 1]]
    t0 = capsule.frames[seed_pt.row].timestamp_s
    observed_onset_s = capsule.frames[onset].timestamp_s - t0
    steps = onset - seed_pt.row + int(round(0.6 / cfg.dt))
    verdicts = []
    for mu in MU_GRID:
        pairs = []
        for k in range(replicates):
            noise = 70_000 + k
            ctrl = _replay_rows(cfg, theta, toy_seed, mu, noise, steps, t0, control=True)
            trt = _replay_rows(cfg, theta, toy_seed, mu, noise, steps, t0, control=False)
            coefficients = {"static_friction": mu, "dynamic_friction": mu * cfg.kinetic_ratio}
            receipt = InterventionReceipt(
                Intervention("friction_reduction", tuple(coefficients.items())),
                dict(coefficients),
                dict(coefficients),  # the toy writes exactly what it is asked to
                "toy_parameter_readback",
            )
            pairs.append(([dict(observed[0])] + ctrl, [dict(observed[0])] + trt, receipt))
        verdicts.append(
            accept_reconstruction(
                mu,
                observed,
                observed_onset_s,
                pairs,
                TOY_CRITERION,
                AcceptanceConfig(
                    scales={"tilt_rad": 0.1, "base_ang_vel_body": 0.5, "base_lin_vel_body": 0.2}
                ),
            )
        )
    reproduced = [v.mu for v in verdicts if v.status == "REPRODUCED"]
    return {
        "seed_row": seed_pt.row,
        "reproduced_mu": reproduced,
        "reproduced_range": [min(reproduced), max(reproduced)] if reproduced else None,
        "pass_fraction_by_mu": {f"{v.mu:.4f}": v.pass_fraction for v in verdicts},
        "deployment_mu_inside_reproduced_range": bool(
            reproduced and min(reproduced) - 1e-9 <= DEPLOYMENT["mu"] <= max(reproduced) + 1e-9
        ),
    }


def nominal_seeds(cfg, theta, capsule_seeds, capsule, *, max_rollouts=50):
    """Arm C's seeds: nominal states at the capsule's timing relative to patch entry."""
    entry_t = next(
        f.timestamp_s for f in capsule.frames if f.base_pos[0] >= DEPLOYMENT["patch_start"]
    )
    offsets = [entry_t - p.frame.timestamp_s for p in capsule_seeds]  # seconds before entry
    rng = np.random.default_rng(4242)
    for attempt in range(max_rollouts):
        batch = nominal_batch(cfg, rng, 1)
        res = simulate(cfg, theta, batch, record_trace=True)
        if traverse_failure(res)[0, 0]:
            continue
        x = res.trace["x"][0, 0]
        entry = int(np.argmax(x >= batch.patch_start[0]))
        seeds = []
        for k, off in enumerate(offsets):
            t = entry - int(round(off / cfg.dt)) - 1
            if t < 0:
                break
            seeds.append(
                ToySeed(
                    f"nominal{attempt}:{k}",
                    float(x[t]),
                    float(res.trace["v"][0, 0, t]),
                    float(res.trace["th"][0, 0, t]),
                    float(res.trace["thd"][0, 0, t]),
                    float(res.trace["phase"][0, 0, t]) % (2 * math.pi),
                    float(batch.v_cmd[0]),
                    float(batch.patch_start[0]),
                )
            )
        if len(seeds) != len(offsets):
            continue
        found, _ = boundaries_for(
            cfg, theta, seeds, seed_source="nominal_rollout", rng_seed=attempt
        )
        if found:
            return seeds, {"nominal_rollouts_tried": attempt + 1, "identified": len(found)}
    raise RuntimeError("no nominal rollout offered an identified boundary")


def evaluate(cfg, theta, *, episodes=256):
    out = {}
    seeds = np.arange(episodes) + 500_000
    for name, cond in {**HELD_OUT, "deployment_condition": DEPLOYMENT}.items():
        res = simulate(cfg, theta, condition_batch(cfg, seeds=seeds, **cond))
        out[name] = float(traverse_failure(res).mean())
    rng = np.random.default_rng(31_337)
    res = simulate(cfg, theta, nominal_batch(cfg, rng, 2 * episodes))
    out["nominal_success"] = float(1 - traverse_failure(res).mean())
    out["nominal_tracking_rmse"] = float(res.track_rmse.mean())
    out["held_out_failure"] = float(np.mean([out[k] for k in HELD_OUT]))
    return out


def informativeness_of(cfg, theta, sampler, seeds_by_id, *, n=256, replicates=16):
    rng = np.random.default_rng(777)
    batch = draws_to_batch(cfg, sampler.sample(n), seeds_by_id, rng, NEIGHBORHOOD)
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


def run_study(output: Path, *, training_seeds=TRAINING_SEEDS, es=None, baseline_iterations=200):
    from ashfall.fbr.boundary import BoundarySampler, BroadSampler
    from ashfall.stats import paired_seed_effect

    cfg = ToySlipConfig()
    es = es or ESConfig()
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "toy": config_dict(cfg, es),
        "training_seeds": list(training_seeds),
        "deployment": DEPLOYMENT,
        "held_out": HELD_OUT,
        "lead_times": LEAD_TIMES,
        "nominal_fraction": NOMINAL_FRACTION,
        "neighborhood": asdict(NEIGHBORHOOD),
        "baseline_iterations": baseline_iterations,
        "arms": ARMS,
        "evidence_kind": "toy_mechanism",
    }
    spec_id = content_hash(spec)

    base_es = ESConfig(
        iterations=baseline_iterations,
        pairs=es.pairs,
        contexts=es.contexts,
        sigma=es.sigma,
        learning_rate=es.learning_rate,
    )
    theta0, base_log = es_train(
        cfg, base_es, initial_params(), lambda g, n: nominal_batch(cfg, g, n), seed=0
    )
    np.save(output / "baseline.npy", theta0)
    policy_sha = file_hash(output / "baseline.npy")

    # The first failure that the acceptance checks reproduce is the capsule. A
    # failure the matched full-friction control also produces (a start-up fall,
    # say) is not a friction failure and is logged and skipped, never trained on.
    rejected = []
    for trial, capsule in deployment_failures(cfg, theta0, policy_sha):
        try:
            seeds = seed_points(capsule, lead_times_s=LEAD_TIMES)
        except ValueError as exc:
            rejected.append({"trial": trial, "reason": str(exc)})
            continue
        recon = reconstruct(cfg, theta0, capsule, seeds)
        if recon["reproduced_mu"]:
            break
        rejected.append({"trial": trial, "reason": "UNREPRODUCED"})
    else:
        raise RuntimeError("no deployment failure was reproduced")
    dep_info = {"trials_until_capsule": trial, "rejected_failures": rejected}
    capsule.save(output / "capsule.json")
    d_seeds = [seed_from_frame(p.frame, p.seed_id, DEPLOYMENT["patch_start"]) for p in seeds]
    c_seeds, c_info = nominal_seeds(cfg, theta0, seeds, capsule)
    d_bounds, d_binfo = boundaries_for(cfg, theta0, d_seeds, seed_source="deployment_failure")
    if not d_bounds:
        raise RuntimeError("no identified boundary from the capsule seeds")
    band = (
        max(
            NEIGHBORHOOD.mu_min,
            min(b.mu_interval[0] for b in d_bounds) - NEIGHBORHOOD.mu_half_width,
        ),
        min(
            NEIGHBORHOOD.mu_max,
            max(b.mu_interval[1] for b in d_bounds) + NEIGHBORHOOD.mu_half_width,
        ),
    )
    c_bounds, _ = boundaries_for(cfg, theta0, c_seeds, seed_source="nominal_rollout")

    seeds_by_id = {s.seed_id: s for s in [*d_seeds, *c_seeds]}
    info = {
        "B_broad_dr": informativeness_of(
            cfg, theta0, BroadSampler(cfg.broad_mu, rng_seed=1), seeds_by_id
        ),
        "C_nominal_seed_boundary": informativeness_of(
            cfg,
            theta0,
            BoundarySampler(c_bounds, NEIGHBORHOOD, nominal_fraction=NOMINAL_FRACTION, rng_seed=1),
            seeds_by_id,
        ),
        "D_fbr": informativeness_of(
            cfg,
            theta0,
            BoundarySampler(d_bounds, NEIGHBORHOOD, nominal_fraction=NOMINAL_FRACTION, rng_seed=1),
            seeds_by_id,
        ),
    }

    per_seed = {arm: {} for arm in ARMS}
    base_eval = evaluate(cfg, theta0)
    budgets = {}
    for s in training_seeds:
        per_seed["A_no_repair"][s] = base_eval
        for arm in ARMS[1:]:
            run = run_arm(
                arm,
                cfg,
                es,
                theta0,
                seed=s,
                neighborhood=NEIGHBORHOOD,
                nominal_fraction=NOMINAL_FRACTION,
                seeds=d_seeds if arm == "D_fbr" else c_seeds if arm.startswith("C") else (),
                seed_source="deployment_failure"
                if arm == "D_fbr"
                else "nominal_rollout"
                if arm.startswith("C")
                else None,
                band=band if arm.startswith("E") else None,
            )
            per_seed[arm][s] = evaluate(cfg, run.theta)
            budgets.setdefault(arm, set()).add(run.log["rollout_slots"])
    if len({v for b in budgets.values() for v in b}) != 1:
        raise RuntimeError(f"unequal rollout budgets: {budgets}")

    def metric(arm, key):
        return {s: per_seed[arm][s][key] for s in training_seeds}

    comparisons = {}
    for key in (
        "held_out_failure",
        "deployment_condition",
        *HELD_OUT,
        "nominal_success",
        "nominal_tracking_rmse",
    ):
        for a, b in (
            ("B_broad_dr", "D_fbr"),
            ("C_nominal_seed_boundary", "D_fbr"),
            ("E_friction_band", "D_fbr"),
            ("A_no_repair", "D_fbr"),
            ("A_no_repair", "B_broad_dr"),
        ):
            base, treat = metric(a, key), metric(b, key)
            if (
                len(set(treat[s] - base[s] for s in training_seeds)) == 1
                and len(training_seeds) > 1
                and all(treat[s] == base[s] for s in training_seeds)
            ):
                comparisons[f"{key}:{b}-{a}"] = {"mean": 0.0, "note": "all deltas zero"}
                continue
            comparisons[f"{key}:{b}-{a}"] = paired_seed_effect(base, treat, metric=key).to_dict()
    primary = comparisons["held_out_failure:D_fbr-B_broad_dr"]
    secondary = comparisons["held_out_failure:D_fbr-C_nominal_seed_boundary"]
    primary_rejects = primary["mean"] < 0 and primary["p_exact_two_sided"] <= 0.05
    fixed_sequence = {
        "step1_D_vs_B": {
            "rejects": primary_rejects,
            "p": primary["p_exact_two_sided"],
            "mean": primary["mean"],
        },
        "step2_D_vs_C": {
            "tested": primary_rejects,
            "rejects": bool(
                primary_rejects and secondary["mean"] < 0 and secondary["p_exact_two_sided"] <= 0.05
            ),
            "p": secondary["p_exact_two_sided"],
            "mean": secondary["mean"],
        },
    }
    summary = {
        arm: {
            k: float(np.mean([per_seed[arm][s][k] for s in training_seeds]))
            for k in per_seed[arm][training_seeds[0]]
        }
        for arm in ARMS
    }
    result = {
        "spec_id": spec_id,
        "spec": spec,
        "evidence_kind": "toy_mechanism",
        "baseline": {
            "policy_sha256": policy_sha,
            "final_train_failure": base_log["failure_rate"][-1],
        },
        "deployment": dep_info,
        "capsule_id": capsule.capsule_id,
        "capsule_onset_reason": capsule.event_descriptor["onset_reason"],
        "seed_rows": [p.row for p in seeds],
        "seed_leads_s": [p.lead_s for p in seeds],
        "reconstruction": recon,
        "arm_C_seed_search": c_info,
        "friction_band_E": band,
        "boundaries": {
            "D": [b.to_dict() for b in d_bounds],
            "C": [b.to_dict() for b in c_bounds],
            "D_dropped": d_binfo["dropped"],
        },
        "informativeness_at_baseline": info,
        "rollout_slots_per_arm": sorted(next(iter(budgets.values()))),
        "summary_mean_over_seeds": summary,
        "per_seed": {arm: {str(s): v for s, v in d.items()} for arm, d in per_seed.items()},
        "comparisons": comparisons,
        "fixed_sequence": fixed_sequence,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_efficiency(output: Path, *, primary: Path, snapshot_at=(30, 60, 90)):
    """EXPLORATORY, added after the primary result: held-out failure at intermediate budgets.

    Re-trains arms B, C and D with the primary run's seeds and inputs, keeping
    parameter snapshots, and evaluates each snapshot and the final parameters
    on the same held-out cells. The final values must equal the primary run's;
    a mismatch is a determinism failure and aborts.
    """
    primary_result = json.loads((primary / "result.json").read_text())
    cfg, es = ToySlipConfig(), ESConfig()
    theta0 = np.load(primary / "baseline.npy")
    from ashfall.capsule import FailureCapsule

    capsule = FailureCapsule.load(primary / "capsule.json")
    seeds = seed_points(capsule, lead_times_s=LEAD_TIMES)
    d_seeds = [seed_from_frame(p.frame, p.seed_id, DEPLOYMENT["patch_start"]) for p in seeds]
    c_seeds, _ = nominal_seeds(cfg, theta0, seeds, capsule)
    output.mkdir(parents=True, exist_ok=True)
    curves: dict = {}
    for arm in ("B_broad_dr", "C_nominal_seed_boundary", "D_fbr"):
        curves[arm] = {}
        for s in primary_result["spec"]["training_seeds"]:
            run = run_arm(
                arm,
                cfg,
                es,
                theta0,
                seed=s,
                neighborhood=NEIGHBORHOOD,
                nominal_fraction=NOMINAL_FRACTION,
                seeds=d_seeds if arm == "D_fbr" else c_seeds if arm.startswith("C") else (),
                seed_source="deployment_failure"
                if arm == "D_fbr"
                else "nominal_rollout"
                if arm.startswith("C")
                else None,
                snapshot_at=snapshot_at,
            )
            final = evaluate(cfg, run.theta)
            if final != primary_result["per_seed"][arm][str(s)]:
                raise RuntimeError(f"determinism failure: {arm} seed {s} differs from primary run")
            row = {
                str(it): evaluate(cfg, th)["held_out_failure"]
                for it, th in run.log["snapshots"].items()
            }
            row[str(es.iterations)] = final["held_out_failure"]
            curves[arm][str(s)] = row
    slots_per_iteration = 2 * es.pairs * es.contexts
    result = {
        "status": "EXPLORATORY: not preregistered, added after the primary null",
        "primary_spec_id": primary_result["spec_id"],
        "slots_per_iteration": slots_per_iteration,
        "curves": curves,
        "mean_curve": {
            arm: {
                it: float(np.mean([curves[arm][s][it] for s in curves[arm]]))
                for it in curves[arm][next(iter(curves[arm]))]
            }
            for arm in curves
        },
        "determinism": "final parameters reproduce the primary run exactly for every arm and seed",
    }
    (output / "efficiency.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--quick", action="store_true", help="3 seeds, short training (software check)"
    )
    parser.add_argument(
        "--efficiency-from", type=Path, help="EXPLORATORY: budget curves from a primary run"
    )
    args = parser.parse_args(argv)
    if args.efficiency_from is not None:
        eff = run_efficiency(args.output, primary=args.efficiency_from)
        print(json.dumps(eff["mean_curve"], indent=1))
        return
    if args.quick:
        result = run_study(
            args.output,
            training_seeds=TRAINING_SEEDS[:3],
            es=ESConfig(iterations=12, refresh_every=6),
            baseline_iterations=30,
        )
    else:
        result = run_study(args.output)
    print(
        json.dumps(
            {
                "summary": result["summary_mean_over_seeds"],
                "fixed_sequence": result["fixed_sequence"],
                "informativeness": result["informativeness_at_baseline"],
                "reconstruction": {
                    k: result["reconstruction"][k]
                    for k in ("reproduced_range", "deployment_mu_inside_reproduced_range")
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
