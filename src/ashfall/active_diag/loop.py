"""The diagnosis loop: passive weights, optional probes, belief updates, and a final status.

Methods (identical evidence, identical update, identical final rule; they differ only in which
probe, if any, is run):

* ``passive`` (B0): weights from the failure trace and nominal logs; always names the top
  mechanism;
* ``passive_abstain`` (B1): same weights, plus the UNKNOWN test with its own calibrated threshold;
* ``fixed`` (B2): the same pre-selected probe in every round (if admissible);
* ``random`` (B3): a uniformly random admissible probe each round;
* ``active`` (B4): the admissible probe with the highest expected information gain.

Probing methods run at least one probe (a confirmation probe) and at most ``MAX_PROBES``; they
stop early once the top weight reaches ``TAU_KNOWN``, and the active method also stops when the
best admissible probe's EIG is below ``EIG_MIN``. If a probe is needed (top weight below
``TAU_KNOWN``) and none is admissible, the status is ABORT (safety cannot be guaranteed); a
confident diagnosis with no safe confirmation probe goes to the final rule, flagged unconfirmed.
Evidence is the failure trace only; the nominal logs calibrate the residual scale. Final rule:
UNKNOWN if the unknown statistic exceeds the method's calibrated threshold; else KNOWN(top) if
the top weight >= ``TAU_KNOWN``; else UNRESOLVED.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ashfall.fcsi.objective import Budget

from . import posterior as po
from . import probe_search as ps
from . import unknown as uk
from .prediction import execute
from .probe_space import probe_grid

MAX_PROBES = 3
TAU_KNOWN = 0.9
EIG_MIN = 0.01
PROBE_SEED_BASE = 4_000_000


def _weights(cfg, theta, base, hyps, evidence, scale, temperature, budget):
    ll = po.log_likelihoods(cfg, theta, base, hyps, evidence, scale, budget)
    return po.weights(hyps, ll, temperature)


def run_method(
    cfg,
    theta,
    base,
    inst: dict,
    scale,
    temperature: float,
    method: str,
    *,
    fixed_probe=None,
    seed: int = 0,
) -> dict:
    """Run one method on one instance; returns everything needed to score it (no threshold yet)."""
    budget = Budget()
    hyps = po.hypothesis_grid()
    # Evidence is the failure trace (nominal logs only calibrate the noise scale).
    evidence = [po.EvidenceLog(inst["failure_log"], None, "failure")]
    w = _weights(cfg, theta, base, hyps, evidence, scale, temperature, budget)
    history = [{"weights": w.as_dict(), "entropy": w.entropy, "top": w.top}]
    probes_run, safety_events, eigs = [], [], []
    status_hint = None
    rng = np.random.default_rng(seed + 17)
    rng_meas = np.random.default_rng(seed + 29)
    grid = probe_grid()
    if method in ("fixed", "random", "active"):
        for r in range(MAX_PROBES):
            if r >= 1 and w.top_weight >= TAU_KNOWN:
                break
            if method == "fixed":
                ok, _, _, _, _, _ = ps.admissible(cfg, theta, base, w, [fixed_probe], budget)
                choice = ok[0] if ok else None
            elif method == "random":
                ok, _, _, _, _, _ = ps.admissible(cfg, theta, base, w, grid, budget)
                choice = ok[int(rng.integers(len(ok)))] if ok else None
            else:
                ranked, _ = ps.rank(cfg, theta, base, w, grid, scale, temperature, rng, budget)
                choice = None
                if ranked:
                    if ranked[0].eig < EIG_MIN:
                        status_hint = "no_informative_probe"
                        break
                    choice = ranked[0].probe
                    eigs.append(ranked[0].eig)
            if choice is None:
                # ABORT only when a probe is needed (ambiguity) and none is safe; a confident
                # diagnosis without a safe confirmation probe goes to the final rule unconfirmed.
                needed = w.top_weight < TAU_KNOWN
                status_hint = "no_admissible_probe" if needed else "unconfirmed_no_safe_probe"
                break
            lg, u, facts = execute(
                cfg, theta, inst["real"], choice, PROBE_SEED_BASE + seed * 10 + r, rng_meas
            )
            probes_run.append(
                {"probe": choice.to_dict(), "physical_s": float(len(lg) * cfg.dt), **facts}
            )
            safety_events.append(facts["violation"])
            evidence.append(po.EvidenceLog(lg, u, f"probe{r}"))
            w_prev = w
            w = _weights(cfg, theta, base, hyps, evidence, scale, temperature, budget)
            history.append(
                {
                    "weights": w.as_dict(),
                    "entropy": w.entropy,
                    "top": w.top,
                    "entropy_drop": w_prev.entropy - w.entropy,
                }
            )
    best, refined = uk.refine(cfg, theta, base, w, evidence, scale)
    stat = uk.statistic(cfg, theta, best, evidence, scale)
    stat["refined_value"] = refined
    return {
        "method": method,
        "top": w.top,
        "top_weight": w.top_weight,
        "map_value": w.map_value[w.top],
        "entropy0": history[0]["entropy"],
        "entropy": w.entropy,
        "history": history,
        "probes": probes_run,
        "n_probes": len(probes_run),
        "eig": eigs,
        "safety_violations": int(sum(safety_events)),
        "status_hint": status_hint,
        "unknown_stat": stat,
        "budget": asdict(budget),
    }


def decide(run: dict, threshold: float | None) -> str:
    """Final status for a method run given its calibrated UNKNOWN threshold."""
    if run["method"] == "passive":
        return "KNOWN"
    if run["status_hint"] == "no_admissible_probe":
        return "ABORT"
    if threshold is not None and run["unknown_stat"]["peak"] > threshold:
        return "UNKNOWN"
    return "KNOWN" if run["top_weight"] >= TAU_KNOWN else "UNRESOLVED"


def correct(run: dict, status: str, kind: str, truth: str | None) -> bool:
    if kind == "known":
        return status == "KNOWN" and run["top"] == truth
    return status == "UNKNOWN"
