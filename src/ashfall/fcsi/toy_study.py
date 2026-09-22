"""FCSI ground-truth toy: can failure-conditioned identification repair the simulator?

Preregistered in ``docs/research/FCSI_TOY_PREREGISTRATION.md`` (committed with this code
before the registered run). Evidence kind ``toy_mechanism``; nothing here is a GO2 result.

Run:
    python -m ashfall.fcsi.toy_study --output results/fcsi_toy/<commit>
    python -m ashfall.fcsi.toy_study --output <scratch> --quick   # software check only

``--quick`` uses a separate, unregistered world (``QUICK_WORLD``) with its own seeds and tiny
budgets, so a software check never looks at a registered instance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from ashfall.fbr.toy_slip import ToySlipConfig
from ashfall.fcsi import acceptance as acc
from ashfall.fcsi import baselines as bl
from ashfall.fcsi import divergence as dv
from ashfall.fcsi import evaluation as evl
from ashfall.fcsi import objective as ob
from ashfall.fcsi import search as fs
from ashfall.fcsi.alignment import validate_toy_log
from ashfall.fcsi.toy_world import (
    MAX_DELAY,
    Condition,
    Hidden,
    Physics,
    failure_rate,
    record,
)
from ashfall.provenance import file_hash

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "results" / "fbr_toy_v2" / "8f49f02" / "baseline.npy"
BASELINE_SHA = "2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3"

NUISANCE = {"pole_mass": 0.31, "cart_damping": 0.03}
WORLDS = {
    "K1_kinetic": {
        "kind": "known",
        "truth": "kinetic_ratio",
        "physics": Physics.of(kinetic_ratio=0.4, **NUISANCE),
        "n": 6,
    },
    "K2_latency": {
        "kind": "known",
        "truth": "action_delay",
        "physics": Physics.of(action_delay=1.5, **NUISANCE),
        "n": 6,
    },
    "K3_braking": {
        "kind": "known",
        "truth": "aniso",
        "physics": Physics.of(aniso=0.4, **NUISANCE),
        "n": 6,
    },
    "U1_strip": {
        "kind": "unknown",
        "truth": None,
        "physics": Physics.of(hidden=Hidden(strip=(0.1, 0.6, 0.15)), **NUISANCE),
        "n": 4,
    },
    "U2_dropout": {
        "kind": "unknown",
        "truth": None,
        "physics": Physics.of(hidden=Hidden(dropout=(0.0, 0.0)), **NUISANCE),
        "n": 4,
    },
    "N_harmless": {
        "kind": "negative",
        "truth": None,
        "physics": Physics.of(pole_mass=0.31, cart_damping=0.4),
        "n": 3,
    },
}
QUICK_WORLD = {
    "Q_kinetic": {
        "kind": "known",
        "truth": "kinetic_ratio",
        "physics": Physics.of(kinetic_ratio=0.5, **NUISANCE),
        "n": 1,
    }
}

N_NOMINAL, N_HELDOUT_NOMINAL, N_HELDOUT_CONDITIONS = 8, 4, 12
NOMINAL_MAX, REAL_MIN = 0.05, 0.25
SCREEN_N = 128
RATE_N = 256


def _benign(rng) -> Condition:
    return Condition(
        float(rng.uniform(0.5, 0.8)), float(rng.uniform(0.7, 1.2)), float(rng.uniform(0.9, 1.3))
    )


def _hazard(rng, lo=0.15, hi=0.35) -> Condition:
    return Condition(
        float(rng.uniform(lo, hi)), float(rng.uniform(0.7, 1.2)), float(rng.uniform(0.9, 1.3))
    )


def _discrepant(cfg, theta, real, rng, *, negative=False, tries=400):
    nominal = Physics.of()
    for _ in range(tries):
        c = _hazard(rng)
        fr = failure_rate(cfg, theta, real, c, n=SCREEN_N, base=2_000_000)
        fn = failure_rate(cfg, theta, nominal, c, n=SCREEN_N, base=2_000_000)
        if negative and fr <= NOMINAL_MAX and fn <= NOMINAL_MAX:
            return c, fr, fn
        if not negative and fn <= NOMINAL_MAX and fr >= REAL_MIN:
            return c, fr, fn
    raise RuntimeError("no discrepant condition found")


def _log_with(cfg, theta, real, cond, seed0, rng_meas, *, want_failure: bool, tries=400):
    for k in range(tries):
        lg = record(cfg, theta, real, cond, seed0 + k, rng_meas)
        if (lg.outcome != "success") == want_failure and len(lg) > MAX_DELAY + 30:
            return lg, seed0 + k
    raise RuntimeError("no log with the wanted outcome")


def truth_onset(cfg, theta, real, cond, seed, cal) -> float | None:
    """First row where the truth's one-step dynamics depart from the nominal model's
    (noise-free log, d^2 above the 99.9% chi-square point). Instrumentation only."""
    clean = record(cfg, theta, real, cond, seed, np.random.default_rng(0), meas=0.0)
    rows = np.arange(MAX_DELAY, len(clean) - 1)
    r = dv.residuals(cfg, [Physics.of()], clean, k=1, rows=rows)[0]
    hit = np.flatnonzero(cal.scale.d2(r) > chi2.ppf(0.999, 5))
    return float(clean.t[rows[hit[0]]]) if len(hit) else None


def build_instance(cfg, theta, world: str, spec: dict, idx: int, global_idx: int) -> dict:
    rng = np.random.default_rng(10_000 + 97 * global_idx)
    base = 10_000_000 + 100_000 * global_idx
    rng_meas = np.random.default_rng(base + 1)
    real = spec["physics"]
    nominal_logs, held_nom, k = [], [], 0
    while len(nominal_logs) + len(held_nom) < N_NOMINAL + N_HELDOUT_NOMINAL:
        lg = record(cfg, theta, real, _benign(rng), base + 10 + k, rng_meas)
        k += 1
        if lg.outcome == "success":
            (nominal_logs if len(nominal_logs) < N_NOMINAL else held_nom).append(lg)
    negative = spec["kind"] == "negative"
    cond, fr, fn = _discrepant(cfg, theta, real, rng, negative=negative)
    target, tseed = _log_with(
        cfg, theta, real, cond, base + 50_000, rng_meas, want_failure=not negative
    )
    validation = None
    if not negative:
        vcond, _, _ = _discrepant(cfg, theta, real, rng)
        validation, _ = _log_with(
            cfg, theta, real, vcond, base + 60_000, rng_meas, want_failure=True
        )
    heldout = [_hazard(rng, 0.1, 0.45) for _ in range(N_HELDOUT_CONDITIONS)]
    for lg in nominal_logs + held_nom + [target] + ([validation] if validation else []):
        validate_toy_log(lg, cfg.dt)
    return {
        "id": f"{world}#{idx}",
        "world": world,
        "kind": spec["kind"],
        "truth": spec["truth"],
        "truth_physics": real.to_dict(),
        "target_condition": cond.to_dict(),
        "target_real_rate_screen": fr,
        "target_nominal_rate_screen": fn,
        "target_seed": tseed,
        "evidence": ob.Evidence(nominal_logs, target, validation),
        "heldout_nominal_logs": held_nom,
        "heldout_conditions": heldout,
        "heldout_real_rates": evl.failure_rates(cfg, theta, real, heldout, n=RATE_N),
        "heldout_nominal_rates": evl.failure_rates(cfg, theta, Physics.of(), heldout, n=RATE_N),
    }


_JOB: dict = {}


def run_instance(args) -> dict:
    world, spec, idx, gidx = args
    cfg, theta, quick = _JOB["cfg"], _JOB["theta"], _JOB["quick"]
    t0 = time.time()
    inst = build_instance(cfg, theta, world, spec, idx, gidx)
    ev = inst["evidence"]
    base = Physics.of()
    cal = dv.calibrate(cfg, base, ev.nominal_logs)
    cal5 = dv.calibrate(cfg, base, ev.nominal_logs, k=bl.G0B_K)
    rep = dv.locate(cfg, base, ev.failure_log, cal)
    inst["event_row"] = ob.event_start_row(cfg, ev.failure_log, rep.t_div)
    if ev.validation_log is not None:
        rv = dv.locate(cfg, base, ev.validation_log, cal)
        inst["validation_event_row"] = ob.event_start_row(cfg, ev.validation_log, rv.t_div)
    onset = None
    if ev.failure_log.outcome != "success":
        onset = truth_onset(
            cfg, theta, spec["physics"], ev.failure_log.cond, inst["target_seed"], cal
        )
    out = {
        "id": inst["id"],
        "world": world,
        "kind": spec["kind"],
        "truth": spec["truth"],
        "target_condition": inst["target_condition"],
        "target_outcome": ev.failure_log.outcome,
        "target_fail_time": ev.failure_log.fail_time,
        "calibration": cal.to_dict(),
        "divergence_nominal": rep.to_dict(),
        "divergence_text": rep.text(),
        "truth_effect_onset": onset,
        "heldout_real_rates": inst["heldout_real_rates"],
        "heldout_nominal_rates": inst["heldout_nominal_rates"],
        "methods": {},
    }
    if quick:
        for mod in (bl,):
            mod.HALTON, mod.NM_EVALS = 16, 10
    diag = fs.identify(cfg, theta, ev, cal)
    a = acc.check(cfg, theta, diag, ev, cal)
    status = acc.finalize(diag, a)
    patched = diag.patch(base) if status in ("identified", "ambiguous") else base
    out["methods"]["FCSI"] = {
        "status": status,
        "raw_status": diag.status,
        "diagnosis": diag.to_dict(),
        "acceptance": None if a is None else a.to_dict(),
        "mechanism": diag.top.mechanisms[0]
        if status in ("identified", "ambiguous") and len(diag.top.mechanisms) == 1
        else ("+".join(diag.top.mechanisms) if status in ("identified", "ambiguous") else None),
        "physics": patched.to_dict(),
        "score": evl.score(cfg, theta, patched, inst, cal),
        "budget": diag.budget,
    }
    if ev.failure_log.outcome != "success":
        dn = fs.identify(cfg, theta, ev, cal, use_event=False)
        pn = dn.patch(base)
        out["methods"]["FCSI_no_event"] = {
            "mechanism": dn.top.mechanisms[0] if dn.top else None,
            "physics": pn.to_dict(),
            "score": evl.score(cfg, theta, pn, inst, cal),
            "budget": dn.budget,
        }
    for name, fit in (
        ("G0_whole_trajectory", lambda: bl.fit_g0(cfg, theta, ev, seed=gidx)),
        ("G0b_multiple_shooting", lambda: bl.fit_g0b(cfg, ev, cal5, seed=gidx)),
        ("G1_dropo_like", lambda: bl.fit_g1(cfg, ev, cal, seed=gidx)),
    ):
        g = fit()
        ph = g.physics(base)
        mech, nchg = bl.implied_mechanism(g.values)
        out["methods"][name] = {
            "mechanism": mech,
            "fit": g.to_dict(),
            "physics": ph.to_dict(),
            "score": evl.score(cfg, theta, ph, inst, cal),
            "budget": g.budget,
        }
    out["nominal_model_score"] = evl.score(cfg, theta, base, inst, cal)
    out["seconds"] = time.time() - t0
    print(
        f"  {inst['id']}: FCSI {status} {out['methods']['FCSI']['mechanism']} "
        f"({out['seconds']:.0f} s)",
        flush=True,
    )
    return out


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #

GLOBALS = ("G0_whole_trajectory", "G0b_multiple_shooting", "G1_dropo_like")


def gate(results: list) -> dict:
    from ashfall.stats.primary import paired_seed_effect

    known = [r for r in results if r["kind"] == "known"]
    unknown = [r for r in results if r["kind"] == "unknown"]
    negative = [r for r in results if r["kind"] == "negative"]

    def correct(r, m):
        return r["methods"][m]["mechanism"] == r["truth"]

    ident = {m: sum(correct(r, m) for r in known) for m in ("FCSI", "FCSI_no_event", *GLOBALS)}
    mae = {
        m: {i: r["methods"][m]["score"]["heldout_mae"] for i, r in enumerate(known)}
        for m in ("FCSI", "FCSI_no_event", *GLOBALS)
    }
    mean_mae = {m: float(np.mean(list(v.values()))) for m, v in mae.items()}
    best_global = min(GLOBALS, key=lambda m: mean_mae[m])
    eff = {
        m: paired_seed_effect(mae[m], mae["FCSI"], metric="heldout_mae").to_dict() for m in GLOBALS
    }
    repro = sum(
        fs.event_ok(
            r["methods"]["FCSI"]["score"]["target_p_event"],
            r["nominal_model_score"]["target_p_event"],
        )
        for r in known
    )
    nom_fcsi = [r["methods"]["FCSI"]["score"]["nominal_d2_per_row"] for r in known]
    nom_base = [r["nominal_model_score"]["nominal_d2_per_row"] for r in known]
    abstain = sum(r["methods"]["FCSI"]["status"] == "unexplained" for r in unknown)
    neg_ok = sum(
        r["methods"]["FCSI"]["status"] in ("no_failure", "no_patch_needed") for r in negative
    )
    g = {
        "G2a_identification": bool(
            ident["FCSI"] >= 14 and all(ident["FCSI"] > ident[m] for m in GLOBALS)
        ),
        "G2b_heldout_prediction": bool(
            all(mean_mae["FCSI"] < mean_mae[m] for m in GLOBALS)
            and eff[best_global]["p_exact_two_sided"] <= 0.05
        ),
        "G2c_reproduction": bool(repro >= 14),
        "G2d_nominal_fidelity": bool(np.mean(nom_fcsi) <= 1.05 * np.mean(nom_base)),
        "G2e_abstention": bool(abstain >= 6),
        "G2f_negative_control": bool(neg_ok == len(negative)),
    }
    return {
        "gates": g,
        "GO": all(g.values()),
        "identification_correct": ident,
        "known_instances": len(known),
        "mean_heldout_mae": mean_mae,
        "best_global": best_global,
        "fcsi_minus_global_mae": eff,
        "fcsi_reproduced": repro,
        "nominal_d2": {"FCSI": float(np.mean(nom_fcsi)), "nominal": float(np.mean(nom_base))},
        "unknown_abstained": abstain,
        "unknown_instances": len(unknown),
        "negative_no_patch": neg_ok,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SWEEP_WORKERS", 20)))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if file_hash(BASELINE) != BASELINE_SHA:
        raise RuntimeError("baseline policy hash mismatch")
    worlds = QUICK_WORLD if args.quick else WORLDS
    jobs, g = [], 0
    for w, spec in worlds.items():
        for i in range(spec["n"]):
            jobs.append((w, spec, i, g + (900 if args.quick else 0)))
            g += 1
    _JOB.update(cfg=ToySlipConfig(), theta=np.load(BASELINE), quick=args.quick)
    t0 = time.time()
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(run_instance, jobs, chunksize=1)
    summary = gate(results) if not args.quick else {"quick": True}
    spec = {
        w: {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in s.items()}
        for w, s in worlds.items()
    }
    doc = {
        "evidence_kind": "toy_mechanism",
        "preregistration": "docs/research/FCSI_TOY_PREREGISTRATION.md",
        "baseline_sha256": BASELINE_SHA,
        "worlds": spec,
        "spec_id": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "wall_seconds": time.time() - t0,
        "summary": summary,
        "instances": results,
    }
    (args.output / "result.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()


__all__ = ["asdict", "run_instance", "gate"]
