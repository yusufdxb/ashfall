"""Active simulator diagnosis, ground-truth toy: development calibration and confirmatory run.

Preregistered in ``docs/research/ACTIVE_DIAGNOSIS_PREREGISTRATION.md``. Evidence kind
``toy_mechanism``; nothing here is a GO2, quadruped or Isaac Lab result.

    python -m ashfall.active_diag.study dev  --output results/active_diag/<commit>
    python -m ashfall.active_diag.study eval --output results/active_diag/<commit>

``dev`` builds development instances only (known mechanisms and development-only unknown
mechanisms), fits the temperature, selects the fixed probe and sets every method's UNKNOWN
threshold, and writes ``calibration.json``. ``eval`` refuses to run without that file, builds the
evaluation instances (new seeds; evaluation-only unknown mechanisms), runs every method and the
oracle, and applies the frozen gate. Nothing in ``eval`` is fitted.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from ashfall.fbr.toy_slip import ToySlipConfig
from ashfall.fcsi import toy_study as ft
from ashfall.fcsi.toy_world import Hidden, Physics, record
from ashfall.provenance import file_hash

from . import loop as lp
from . import posterior as po
from . import probe_search as ps
from . import unknown as uk
from .prediction import execute
from .probe_space import Probe, probe_grid

NUIS = ft.NUISANCE
KNOWN = {
    "KA_patch_friction": ("mu_scale", Physics.of(mu_scale=0.5, **NUIS)),
    "KB_sliding_friction": ("kinetic_ratio", Physics.of(kinetic_ratio=0.4, **NUIS)),
    "KC_braking_friction": ("aniso", Physics.of(aniso=0.4, **NUIS)),
    "KD_action_latency": ("action_delay", Physics.of(action_delay=1.5, **NUIS)),
    "KE_sensor_latency": ("obs_delay", Physics.of(obs_delay=1.0, **NUIS)),
}
DEV_UNKNOWN = {
    "DU1_deadzone": Physics.of(hidden=Hidden(deadzone=9.0), **NUIS),
    "DU2_speed_friction": Physics.of(hidden=Hidden(speed_friction=0.6), **NUIS),
}
EVAL_UNKNOWN = {  # the frozen FCSI unknown mechanisms
    "U1_strip": Physics.of(hidden=Hidden(strip=(0.1, 0.6, 0.15)), **NUIS),
    "U2_dropout": Physics.of(hidden=Hidden(dropout=(0.0, 0.0)), **NUIS),
}
DEV_PER_KNOWN, DEV_PER_UNKNOWN = 3, 3
EVAL_PER_KNOWN, EVAL_PER_UNKNOWN = 6, 6
DEV_BASE, EVAL_BASE = 2000, 3000
TEMPERATURES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
METHODS = ("passive_abstain", "fixed", "random", "active")


def specs(stage: str) -> list:
    out, g = [], DEV_BASE if stage == "dev" else EVAL_BASE
    nk = DEV_PER_KNOWN if stage == "dev" else EVAL_PER_KNOWN
    nu = DEV_PER_UNKNOWN if stage == "dev" else EVAL_PER_UNKNOWN
    unknown = DEV_UNKNOWN if stage == "dev" else EVAL_UNKNOWN
    for name, (truth, phys) in KNOWN.items():
        for i in range(nk):
            out.append(
                {
                    "id": f"{name}#{i}",
                    "world": name,
                    "kind": "known",
                    "truth": truth,
                    "physics": phys,
                    "gidx": g,
                }
            )
            g += 1
    for name, phys in unknown.items():
        for i in range(nu):
            out.append(
                {
                    "id": f"{name}#{i}",
                    "world": name,
                    "kind": "unknown",
                    "truth": None,
                    "physics": phys,
                    "gidx": g,
                }
            )
            g += 1
    return out


def build(cfg, theta, spec: dict) -> dict:
    """Nominal logs and one failure log the nominal model does not predict (FCSI's rules)."""
    gidx = spec["gidx"]
    rng = np.random.default_rng(10_000 + 97 * gidx)
    base = 10_000_000 + 100_000 * gidx
    rng_meas = np.random.default_rng(base + 1)
    real = spec["physics"]
    nominal, k = [], 0
    while len(nominal) < ft.N_NOMINAL:
        lg = record(cfg, theta, real, ft._benign(rng), base + 10 + k, rng_meas)
        k += 1
        if lg.outcome == "success":
            nominal.append(lg)
    cond, fr, fn = ft._discrepant(cfg, theta, real, rng)
    failure, _ = ft._log_with(cfg, theta, real, cond, base + 50_000, rng_meas, want_failure=True)
    return {
        **spec,
        "real": real,
        "nominal_logs": nominal,
        "failure_log": failure,
        "target_condition": cond.to_dict(),
        "screen_rates": [fr, fn],
    }


_JOB: dict = {}


def _build_one(spec):
    cfg, theta, base = _JOB["cfg"], _JOB["theta"], Physics.of()
    inst = build(cfg, theta, spec)
    return spec["id"], inst, po.calibrate_scale(cfg, theta, base, inst["nominal_logs"])


def _prepare(sp, workers):
    """Build every instance once; forked workers then read them from ``_JOB``."""
    with _pool(workers) as pool:
        built = pool.map(_build_one, sp)
    _JOB["inst"] = {i: inst for i, inst, _ in built}
    _JOB["scale"] = {i: sc for i, _, sc in built}


def _setup(spec):
    cfg, theta, base = _JOB["cfg"], _JOB["theta"], Physics.of()
    return cfg, theta, base, _JOB["inst"][spec["id"]], _JOB["scale"][spec["id"]]


def _passive_ll(spec):
    cfg, theta, base, inst, scale = _setup(spec)
    hyps = po.hypothesis_grid()
    ev = [po.EvidenceLog(inst["failure_log"])]
    ll = po.log_likelihoods(cfg, theta, base, hyps, ev, scale)
    return spec["id"], ll.tolist()


def _run_methods(args):
    spec, temperature, fixed, methods, oracle = args
    cfg, theta, base, inst, scale = _setup(spec)
    t0 = time.time()
    out = {
        "id": spec["id"],
        "world": spec["world"],
        "kind": spec["kind"],
        "truth": spec["truth"],
        "target_condition": inst["target_condition"],
        "runs": {},
    }
    for m in methods:
        out["runs"][m] = lp.run_method(
            cfg, theta, base, inst, scale, temperature, m, fixed_probe=fixed, seed=spec["gidx"]
        )
    if oracle is not None:
        out["oracle"] = _oracle(cfg, theta, base, inst, scale, temperature, oracle, spec["gidx"])
    out["seconds"] = time.time() - t0
    print(
        f"  {spec['id']}: "
        + ", ".join(
            f"{m}={r['top']}/{r['top_weight']:.2f}/p{r['n_probes']}" for m, r in out["runs"].items()
        ),
        flush=True,
    )
    return out


def _oracle(cfg, theta, base, inst, scale, temperature, threshold, seed):
    """Upper bound: every admissible single probe, executed, scored with the active threshold."""
    hyps = po.hypothesis_grid()
    ev0 = [po.EvidenceLog(inst["failure_log"])]
    w = po.weights(hyps, po.log_likelihoods(cfg, theta, base, hyps, ev0, scale), temperature)
    ok, _, _, _, _, _ = ps.admissible(cfg, theta, base, w, probe_grid())
    good, n = 0, 0
    for j, p in enumerate(ok):
        lg, u, _ = execute(
            cfg,
            theta,
            inst["real"],
            p,
            lp.PROBE_SEED_BASE + seed * 10 + 7 + j,
            np.random.default_rng(seed + 999 + j),
        )
        ev = ev0 + [po.EvidenceLog(lg, u)]
        w1 = po.weights(hyps, po.log_likelihoods(cfg, theta, base, hyps, ev, scale), temperature)
        best, _ = uk.refine(cfg, theta, base, w1, ev, scale)
        stat = uk.statistic(cfg, theta, best, ev, scale)["peak"]
        status = (
            "UNKNOWN"
            if stat > threshold
            else ("KNOWN" if w1.top_weight >= lp.TAU_KNOWN else "UNRESOLVED")
        )
        run = {"top": w1.top}
        good += lp.correct(run, status, inst["kind"], inst["truth"])
        n += 1
    return {"admissible": n, "correct_probes": good, "any_correct": bool(good > 0)}


def _fixed_candidate(args):
    spec, temperature, probe = args
    cfg, theta, base, inst, scale = _setup(spec)
    r = lp.run_method(
        cfg, theta, base, inst, scale, temperature, "fixed", fixed_probe=probe, seed=spec["gidx"]
    )
    return (
        spec["id"],
        probe.name,
        {k: r[k] for k in ("top", "top_weight", "status_hint", "n_probes", "safety_violations")}
        | {"stat": r["unknown_stat"]["peak"], "steps": r["budget"]["steps"]},
    )


def _pool(workers):
    return mp.get_context("fork").Pool(workers)


def threshold_from(dev_runs: list, key=lambda r: r["unknown_stat"]["peak"]) -> float:
    """Largest statistic among development cases with known mechanisms (zero dev false UNKNOWN)."""
    return float(max(key(r) for r in dev_runs))


def dev(output: Path, workers: int) -> dict:
    cfg, theta = ToySlipConfig(), np.load(ft.BASELINE)
    _JOB.update(cfg=cfg, theta=theta)
    sp = specs("dev")
    t0 = time.time()
    _prepare(sp, workers)
    with _pool(workers) as pool:
        lls = dict(pool.map(_passive_ll, sp))
    hyps = po.hypothesis_grid()
    losses = {}
    for T in TEMPERATURES:
        nll = []
        for s in sp:
            if s["kind"] != "known":
                continue
            w = po.weights(hyps, np.array(lls[s["id"]]), T).as_dict()
            nll.append(-np.log(max(w[s["truth"]], 1e-6)))
        losses[T] = float(np.mean(nll))
    best = min(losses.values())
    temperature = max(T for T, v in losses.items() if v <= best + 1e-9)
    # Fixed probe: every candidate on every development case, with its own threshold.
    grid = probe_grid()
    with _pool(workers) as pool:
        fixed_rows = pool.map(_fixed_candidate, [(s, temperature, p) for p in grid for s in sp])
    per_probe = {}
    for sid, pname, row in fixed_rows:
        per_probe.setdefault(pname, {})[sid] = row
    kinds = {s["id"]: (s["kind"], s["truth"]) for s in sp}
    fixed_scores = {}
    for pname, rows in per_probe.items():
        known_stats = [r["stat"] for sid, r in rows.items() if kinds[sid][0] == "known"]
        thr = max(known_stats) if known_stats else float("inf")
        score = 0
        for sid, r in rows.items():
            run = {
                "method": "fixed",
                "status_hint": r["status_hint"],
                "top": r["top"],
                "top_weight": r["top_weight"],
                "unknown_stat": {"peak": r["stat"]},
            }
            score += lp.correct(run, lp.decide(run, thr), *kinds[sid])
        cost = next(p.cost for p in grid if p.name == pname)
        fixed_scores[pname] = {"score": score, "threshold": thr, "cost": cost}
    fixed_name = max(
        fixed_scores, key=lambda n: (fixed_scores[n]["score"], -fixed_scores[n]["cost"])
    )
    fixed_probe = next(p for p in grid if p.name == fixed_name)
    # Other methods on development cases, then thresholds.
    with _pool(workers) as pool:
        runs = pool.map(_run_methods, [(s, temperature, fixed_probe, METHODS, None) for s in sp])
    thresholds = {}
    for m in METHODS:
        known = [r["runs"][m] for r in runs if r["kind"] == "known"]
        thresholds[m] = threshold_from(known) if known else float("inf")
    dev_scores = {}
    for m in METHODS:
        c = [
            lp.correct(r["runs"][m], lp.decide(r["runs"][m], thresholds[m]), r["kind"], r["truth"])
            for r in runs
        ]
        dev_scores[m] = int(sum(c))
    cal = {
        "temperature": temperature,
        "temperature_losses": losses,
        "fixed_probe": fixed_probe.to_dict(),
        "fixed_probe_dev_scores": fixed_scores,
        "thresholds": thresholds,
        "dev_scores": dev_scores,
        "dev_instances": len(sp),
        "wall_seconds": time.time() - t0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(cal, indent=1, default=float) + "\n")
    (output / "dev_runs.json").write_text(json.dumps(runs, indent=1, default=float) + "\n")
    return cal


def gate(results: list, thresholds: dict) -> dict:
    from scipy.stats import binomtest

    known = [r for r in results if r["kind"] == "known"]
    unknown = [r for r in results if r["kind"] == "unknown"]
    for r in results:
        r["runs"]["passive"] = {**r["runs"]["passive_abstain"], "method": "passive"}
    st = {
        m: [lp.decide(r["runs"][m], thresholds.get(m)) for r in results]
        for m in ("passive", *METHODS)
    }

    def ok(m):
        return [lp.correct(r["runs"][m], s, r["kind"], r["truth"]) for r, s in zip(results, st[m])]

    corr = {m: ok(m) for m in st}
    idx_k = [i for i, r in enumerate(results) if r["kind"] == "known"]
    idx_u = [i for i, r in enumerate(results) if r["kind"] == "unknown"]
    known_acc = {m: int(sum(corr[m][i] for i in idx_k)) for m in st}
    unk_rej = {m: int(sum(corr[m][i] for i in idx_u)) for m in st}
    false_unknown = {m: int(sum(st[m][i] == "UNKNOWN" for i in idx_k)) for m in st}
    unresolved = {m: int(sum(s == "UNRESOLVED" for s in st[m])) for m in st}
    aborts = {m: int(sum(s == "ABORT" for s in st[m])) for m in st}
    total = {m: known_acc[m] + unk_rej[m] for m in st}

    def mcnemar(a, b):
        ab = sum(x and not y for x, y in zip(corr[a], corr[b]))
        ba = sum(y and not x for x, y in zip(corr[a], corr[b]))
        n = ab + ba
        return {"a_only": ab, "b_only": ba, "p": float(binomtest(ab, n, 0.5).pvalue) if n else 1.0}

    viol = {m: int(sum(r["runs"][m]["safety_violations"] for r in results)) for m in METHODS}
    maxp = {m: int(max(r["runs"][m]["n_probes"] for r in results)) for m in METHODS}
    ent = {m: float(np.mean([r["runs"][m]["entropy"] for r in known])) for m in METHODS}
    nk, nu = len(known), len(unknown)
    g = {
        "A1_known_non_inferior": bool(
            known_acc["active"] >= max(known_acc[m] for m in ("passive_abstain", "fixed", "random"))
            or (known_acc["active"] >= 0.9 * nk and ent["active"] <= 0.5 * ent["passive_abstain"])
        ),
        "A2_unknown_gain": bool(
            unk_rej["active"] >= unk_rej["passive_abstain"] + 3
            and unk_rej["active"] >= (2 * nu) // 3
        ),
        "A3_false_unknown": bool(false_unknown["active"] <= nk // 10),
        "A4_safety": bool(viol["active"] == 0),
        "A5_probe_budget": bool(maxp["active"] <= lp.MAX_PROBES),
        "A6_beats_random": bool(
            total["active"] > total["random"] and mcnemar("active", "random")["p"] <= 0.05
        ),
        "A7_beats_fixed": bool(total["active"] >= total["fixed"] + max(2, (nk + nu) // 10)),
    }
    return {
        "gates": g,
        "GO": all(g.values()),
        "known_correct": known_acc,
        "unknown_rejected": unk_rej,
        "false_unknown": false_unknown,
        "unresolved": unresolved,
        "abort": aborts,
        "total_correct": total,
        "n_known": nk,
        "n_unknown": nu,
        "mcnemar_active_vs": {
            m: mcnemar("active", m) for m in ("passive_abstain", "fixed", "random")
        },
        "safety_violations": viol,
        "max_probes": maxp,
        "mean_probes": {
            m: float(np.mean([r["runs"][m]["n_probes"] for r in results])) for m in METHODS
        },
        "mean_entropy_known": ent,
        "mean_steps": {
            m: float(np.mean([r["runs"][m]["budget"]["steps"] for r in results])) for m in METHODS
        },
        "oracle_any_correct": int(
            sum(r.get("oracle", {}).get("any_correct", False) for r in results)
        ),
        "statuses": st,
    }


def evaluate(output: Path, workers: int) -> dict:
    cal_path = output / "calibration.json"
    if not cal_path.exists():
        raise SystemExit("no calibration.json: run the dev stage first")
    cal = json.loads(cal_path.read_text())
    cfg, theta = ToySlipConfig(), np.load(ft.BASELINE)
    _JOB.update(cfg=cfg, theta=theta)
    fixed = Probe(
        **{k: cal["fixed_probe"][k] for k in ("mu", "speed", "kind", "amplitude", "frequency")}
    )
    sp = specs("eval")
    t0 = time.time()
    _prepare(sp, workers)
    with _pool(workers) as pool:
        results = pool.map(
            _run_methods,
            [(s, cal["temperature"], fixed, METHODS, cal["thresholds"]["active"]) for s in sp],
        )
    summary = gate(results, cal["thresholds"])
    doc = {
        "evidence_kind": "toy_mechanism",
        "preregistration": "docs/research/ACTIVE_DIAGNOSIS_PREREGISTRATION.md",
        "calibration_sha256": file_hash(cal_path),
        "calibration": cal,
        "wall_seconds": time.time() - t0,
        "summary": summary,
        "instances": results,
    }
    (output / "eval_result.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("dev", "eval"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SWEEP_WORKERS", 22)))
    args = ap.parse_args(argv)
    if file_hash(ft.BASELINE) != ft.BASELINE_SHA:
        raise RuntimeError("baseline policy hash mismatch")
    res = (
        dev(args.output, args.workers)
        if args.stage == "dev"
        else evaluate(args.output, args.workers)
    )
    print(json.dumps({k: v for k, v in res.items() if k != "statuses"}, indent=1, default=float))


if __name__ == "__main__":
    main()
