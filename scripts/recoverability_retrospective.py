"""Phases 2 to 6: estimate recoverability for every stored replay state, then analyse.

Plan: docs/research/RECOVERABILITY_RETROSPECTIVE_PLAN.md (frozen before this ran).

    python scripts/recoverability_retrospective.py --output results/recoverability/retro_<commit>

No training. Reads the frozen precursor sweep; writes a new directory only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
from pathlib import Path

import numpy as np

import ashfall.fbr.toy_study_v2 as v2
from ashfall.fbr.time_sweep import load_baseline
from ashfall.fbr.toy_slip import ToySlipConfig
from ashfall.recoverability import analysis as an
from ashfall.recoverability.dataset import (
    artifact_provenance,
    attach_repair,
    candidate_states,
    repair_outcomes,
)
from ashfall.recoverability.estimator import ContextDistribution, estimate

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "results/precursor_toy/3205f02/stage1.json"
STAGE2 = ROOT / "results/precursor_toy/3205f02/stage2/stage2.json"
N = 1024
NOISE = tuple(sorted(v2.SEED_NOISE.items()))
REFERENCE = ContextDistribution("reference_0.02_0.20", 0.02, 0.20, NOISE)

_JOB: dict = {}


def _estimate_job(c):
    cfg, theta = _JOB["cfg"], _JOB["theta"]
    band = ContextDistribution(f"band_{c.band_hi:.4f}", v2.MU_MIN, c.band_hi, NOISE)
    seed = c.toy_seed()
    e1 = estimate(cfg, theta, seed, band, n=N, repeat=0)
    e2 = estimate(cfg, theta, seed, band, n=N, repeat=1)
    er = estimate(cfg, theta, seed, REFERENCE, n=N, repeat=0)
    return c.state_id, {
        "band": e1.to_dict(),
        "band_repeat": e2.to_dict(),
        "reference": er.to_dict(),
    }


def build(output: Path, workers: int) -> list[dict]:
    cfg = ToySlipConfig()
    theta, sha = load_baseline()
    cands = candidate_states(STAGE1, "discovery") + candidate_states(STAGE2, "hidden")
    _JOB.update(cfg=cfg, theta=theta)
    with mp.get_context("fork").Pool(workers) as pool:
        est = dict(pool.map(_estimate_job, cands))
    rows = []
    for c in cands:
        e = est[c.state_id]
        rows.append(
            {
                **c.to_dict(),
                "r_band": e["band"]["r"],
                "r_band_ci": [e["band"]["ci_low"], e["band"]["ci_high"]],
                "r_band_repeat": e["band_repeat"]["r"],
                "r_ref": e["reference"]["r"],
                "r_ref_ci": [e["reference"]["ci_low"], e["reference"]["ci_high"]],
                "survival_s_band": e["band"]["mean_survival_s"],
                "estimation_steps": sum(x["sim_steps"] for x in e.values()),
                "estimates": e,
                "policy_sha256": sha,
            }
        )
    # Recoverability is fixed from here on; only now are outcomes joined.
    (output / "recoverability_only.json").write_text(json.dumps(rows, indent=1) + "\n")
    o1, o2 = repair_outcomes(STAGE1), repair_outcomes(STAGE2)
    disc = attach_repair([r for r in rows if r["stage"] == "discovery"], o1)
    hid = attach_repair([r for r in rows if r["stage"] == "hidden"], o2)
    return disc + hid


def analyse(rows: list[dict]) -> dict:
    disc = [r for r in rows if r["stage"] == "discovery"]
    hid = [r for r in rows if r["stage"] == "hidden"]
    yd = np.array([r["repair_value"] for r in disc])
    yh = np.array([r["repair_value"] for r in hid])
    res: dict = {}
    r1 = np.array([r["r_band"] for r in rows])
    r2 = np.array([r["r_band_repeat"] for r in rows])
    legacy = [(r["r_band"], r["legacy_r_band"]) for r in rows if r["legacy_r_band"] is not None]
    res["reliability"] = {
        "test_retest_pearson": float(np.corrcoef(r1, r2)[0, 1]),
        "test_retest_mean_abs_diff": float(np.mean(np.abs(r1 - r2))),
        "vs_sweep_diagnostic_pearson": float(np.corrcoef(*zip(*legacy))[0, 1]),
        "mean_ci_half_width": float(
            np.mean([(r["r_band_ci"][1] - r["r_band_ci"][0]) / 2 for r in rows])
        ),
        "r_band_vs_r_ref_pearson": float(np.corrcoef(r1, [r["r_ref"] for r in rows])[0, 1]),
    }
    res["collinearity_discovery"] = {
        "r_band~lead_spearman": float(
            an.spearmanr([r["r_band"] for r in disc], [r["lead_s"] for r in disc]).statistic
        ),
        "r_band~distance_spearman": float(
            an.spearmanr(
                [r["r_band"] for r in disc], [r["distance_to_hazard_m"] for r in disc]
            ).statistic
        ),
    }
    res["models_discovery"] = an.compare_models(disc, yd)
    res["models_discovery_r_ref"] = {
        k: v
        for k, v in an.compare_models(
            [{**r, "r_band": r["r_ref"]} for r in disc], yd, families=("quadratic",)
        ).items()
        if k.startswith(("D_", "E_", "F_"))
    }
    q = res["models_discovery"]
    res["rule_R_beats_time"] = bool(
        q["D_recoverability:quadratic"]["lowo"]["rmse"] < q["A_time:quadratic"]["lowo"]["rmse"]
        and q["D_recoverability:quadratic"]["lowo"]["r2"] > q["A_time:quadratic"]["lowo"]["r2"]
    )
    res["rule_R_beats_distance"] = bool(
        q["D_recoverability:quadratic"]["lowo"]["rmse"] < q["B_distance:quadratic"]["lowo"]["rmse"]
        and q["D_recoverability:quadratic"]["lowo"]["r2"] > q["B_distance:quadratic"]["lowo"]["r2"]
    )
    by_seed = [{int(k): v for k, v in r["repair_value_by_seed"].items()} for r in disc]
    res["interior_discovery"] = an.bootstrap_interior(disc, by_seed)
    res["interior_discovery_r_ref"] = an.bootstrap_interior(
        [{**r, "r_band": r["r_ref"]} for r in disc], by_seed
    )
    res["provenance_discovery"] = an.provenance_permutation(disc, yd)
    e_rmse = q["E_recoverability_provenance:quadratic"]["lowo"]["rmse"]
    d_rmse = q["D_recoverability:quadratic"]["lowo"]["rmse"]
    res["rule_provenance_adds_little"] = bool(
        res["provenance_discovery"]["p"] >= 0.05 and e_rmse >= 0.95 * d_rmse
    )
    # Held-out (stage 2): fit on discovery only.
    fd = an.fit(disc, yd, an.MODELS[3], "quadratic")
    fa = an.fit(disc, yd, an.MODELS[0], "quadratic")
    r_star = an.argmax_quadratic(fd.beta, 0.0, 1.0)
    leads = [r["lead_s"] for r in disc]
    lead_star = an.argmax_quadratic(fa.beta, min(leads), max(leads))
    res["heldout"] = {
        "R_star": r_star,
        "lead_star": lead_star,
        "R_rule": an.regret_table(hid, an.select(hid, "r_band", r_star)),
        "time_rule": an.regret_table(hid, an.select(hid, "lead_s", lead_star)),
        "onset_rule": an.regret_table(
            hid,
            {
                w: next(r for r in hid if r["world"] == w and r["arm"] == "real_T0")
                for w in sorted({r["world"] for r in hid})
            },
        ),
        "oos": {
            spec.name: an.out_of_sample(disc, yd, hid, yh, spec)
            for spec in an.MODELS
            if spec.continuous
        },
    }
    rt = res["heldout"]
    res["rule_R_regret_not_worse_than_time"] = bool(
        rt["R_rule"]["mean_regret"] <= rt["time_rule"]["mean_regret"] + 1e-12
    )
    inter = res["interior_discovery"]
    res["phase7_gate"] = {
        "R_beats_time": res["rule_R_beats_time"],
        "R_beats_distance": res["rule_R_beats_distance"],
        "shape_ok": bool(inter["interior_optimum"] or inter["monotone_increasing_significant"]),
        "provenance_adds_little": res["rule_provenance_adds_little"],
        "R_regret_not_worse_than_time": res["rule_R_regret_not_worse_than_time"],
    }
    res["phase7_gate"]["pass"] = all(res["phase7_gate"].values())
    return res


def plots(rows, res, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker = {
        "failed_real": ("o", "#b03a2e"),
        "failed_sim": ("s", "#1f618d"),
        "successful": ("^", "#1e8449"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    disc = [r for r in rows if r["stage"] == "discovery"]
    yc = an.center_by_world(disc, np.array([r["repair_value"] for r in disc]))
    for ax, key, xl in (
        (axes[0], "r_band", "recoverability R_band (baseline policy)"),
        (axes[1], "lead_s", "time before failure onset (s)"),
        (axes[2], "distance_to_hazard_m", "distance to patch start (m)"),
    ):
        for p, (m, c) in marker.items():
            idx = [i for i, r in enumerate(disc) if r["provenance"] == p]
            ax.scatter(
                [disc[i][key] for i in idx],
                [100 * yc[i] for i in idx],
                marker=m,
                c=c,
                s=26,
                alpha=0.8,
                label=p,
                edgecolors="none",
            )
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlabel(xl)
    fd = an.fit(disc, np.array([r["repair_value"] for r in disc]), an.MODELS[3], "quadratic")
    g = np.linspace(min(r["r_band"] for r in disc), max(r["r_band"] for r in disc), 200)
    curve = fd.beta[0] * g + fd.beta[1] * g**2
    curve -= np.mean(
        fd.beta[0] * np.array([r["r_band"] for r in disc])
        + fd.beta[1] * np.array([r["r_band"] for r in disc]) ** 2
    )
    axes[0].plot(g, 100 * curve, color="k", lw=1.5, label="quadratic fit (world FE)")
    axes[0].set_ylabel("repair value, world-centred (pp less held-out failure)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Discovery worlds W1-W6: repair value against three candidate explanations (toy)")
    fig.tight_layout()
    fig.savefig(output / "repair_vs_explanations.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for p, (m, c) in marker.items():
        idx = [r for r in rows if r["provenance"] == p]
        ax.scatter(
            [r["r_band"] for r in idx],
            [100 * r["repair_value"] for r in idx],
            marker=m,
            c=c,
            s=26,
            alpha=0.8,
            label=p,
            edgecolors="none",
        )
    ax.set_xlabel("recoverability R_band")
    ax.set_ylabel("repair value (pp, raw, not centred)")
    ax.set_title("Cross-provenance collapse test, all 180 states")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "collapse_by_provenance.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SWEEP_WORKERS", 20)))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    rows = build(args.output, args.workers)
    res = analyse(rows)
    meta = {
        "code_commit": commit,
        "plan": "docs/research/RECOVERABILITY_RETROSPECTIVE_PLAN.md",
        "evidence_kind": "toy_mechanism",
        "n_states": len(rows),
        "n_per_estimate": N,
        "total_estimation_steps": int(sum(r["estimation_steps"] for r in rows)),
        "inputs": artifact_provenance(
            STAGE1, STAGE2, ROOT / "results/fbr_toy_v2/8f49f02/baseline.npy"
        ),
    }
    (args.output / "dataset.json").write_text(json.dumps(rows, indent=1) + "\n")
    (args.output / "analysis.json").write_text(
        json.dumps({"meta": meta, **res}, indent=1, default=float) + "\n"
    )
    plots(rows, res, args.output)
    print(
        json.dumps(
            {
                "meta": meta,
                "phase7_gate": res["phase7_gate"],
                "interior": res["interior_discovery"],
                "reliability": res["reliability"],
            },
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
