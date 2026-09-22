"""The retrospective recoverability dataset, built from the frozen precursor sweep.

Two strictly separated steps, so repair outcomes cannot leak into recoverability:

1. :func:`candidate_states` reads only the *setup* half of a stage file (restored
   state, provenance, friction band, lead time) and drops every evaluation key.
   Recoverability is estimated from these rows alone.
2. :func:`attach_repair` joins the measured repair outcomes afterwards, by
   ``(stage, world, arm)``.

Provenance labels: ``failed_real`` (the deployment failure the sweep treated as
real), ``failed_sim`` (a failure found by simulator search), ``successful`` (a
successful baseline traverse matched by position).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import ashfall.fbr.toy_study_v2 as v2
from ashfall.fbr.toy_slip import ToySeed
from ashfall.provenance import file_hash

PROVENANCE = {"real": "failed_real", "sim": "failed_sim", "match": "successful"}
GLOBAL_ARMS = ("A", "B", "ADR", "C")
EVAL_KEYS = frozenset({"eval", "nominal", "steps"})


@dataclass(frozen=True)
class CandidateState:
    state_id: str
    stage: str  # "discovery" (W1-W6) or "hidden" (W7-W12)
    world: str
    arm: str
    label: str  # start, entry, d2 ... d0.25, T0, Rsel
    provenance: str
    source_trajectory: str
    lead_s: float  # time before the failure onset the label is defined against
    distance_to_hazard_m: float  # patch start minus position; negative on the patch
    band_hi: float
    x: float
    v: float
    th: float
    thd: float
    phase: float
    v_cmd: float
    patch_start: float
    legacy_r_band: float | None  # the sweep's own n=256 diagnostic, kept for test-retest

    def toy_seed(self) -> ToySeed:
        return ToySeed(
            self.state_id,
            self.x,
            self.v,
            self.th,
            self.thd,
            self.phase,
            self.v_cmd,
            self.patch_start,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def band_hi(bound: dict) -> float:
    return float(
        min(v2.NEIGHBORHOOD.mu_max, bound["mu_interval"][1] + v2.NEIGHBORHOOD.mu_half_width)
    )


def _source_trajectory(prefix: str, setup: dict, world: str) -> str:
    if prefix == "real":
        return str(setup["real_capsule_id"])
    if prefix == "sim":
        return str(setup["sim"]["capsule_id"])
    return f"success:{world}:trial{setup['success']['success_trial']}"


def candidate_states(stage_file: Path, stage: str) -> list[CandidateState]:
    """Per-world replay arms of a stage file as restorable states, with no outcomes."""
    data = json.loads(Path(stage_file).read_text())
    out = []
    for world, arms in sorted(data["arms"].items()):
        setup = data["setup"][world]
        diag = setup.get("diagnostics", {})
        for arm, a in sorted(arms.items()):
            if EVAL_KEYS & set(a):
                raise ValueError(f"{world}/{arm}: arm record carries evaluation keys")
            prefix, label = arm.split("_", 1)
            seed = a["seed"]
            if "lead_s" in a:
                lead = float(a["lead_s"])
            else:  # matched-success states inherit the lead of the real state they match
                lead = float(arms[f"real_{label}"]["lead_s"])
            legacy = diag.get(arm, {}).get("recoverability_band", {}).get("success")
            out.append(
                CandidateState(
                    state_id=f"{stage}:{world}:{arm}",
                    stage=stage,
                    world=world,
                    arm=arm,
                    label=label,
                    provenance=PROVENANCE[prefix],
                    source_trajectory=_source_trajectory(prefix, setup, world),
                    lead_s=lead,
                    distance_to_hazard_m=float(seed["patch_start"] - seed["x"]),
                    band_hi=band_hi(a["bound"]),
                    x=float(seed["x"]),
                    v=float(seed["v"]),
                    th=float(seed["th"]),
                    thd=float(seed["thd"]),
                    phase=float(seed["phase"]),
                    v_cmd=float(seed["v_cmd"]),
                    patch_start=float(seed["patch_start"]),
                    legacy_r_band=None if legacy is None else float(legacy),
                )
            )
    return out


def repair_outcomes(stage_file: Path) -> dict:
    """``(world, arm) -> outcome`` with per-seed deltas against the unrepaired policy A.

    Repair value is ``failure(A) - failure(arm)`` on the world's held-out conditions,
    so positive means the repair reduced failure.
    """
    data = json.loads(Path(stage_file).read_text())
    per, glob = data["raw"]["per"], data["raw"]["global"]
    seeds = sorted(per, key=int)
    out = {}
    for world, arms in data["arms"].items():
        a_fail = {s: glob[s]["A"]["eval"][world]["held_out_failure"] for s in seeds}
        a_nom = {s: glob[s]["A"]["nominal"]["nominal_success"] for s in seeds}
        for arm in arms:
            fail = {s: per[s][world][arm]["eval"]["held_out_failure"] for s in seeds}
            nom = {s: per[s][world][arm]["nominal"]["nominal_success"] for s in seeds}
            deltas = np.array([a_fail[s] - fail[s] for s in seeds])
            out[(world, arm)] = {
                "held_out_failure": float(np.mean(list(fail.values()))),
                "baseline_failure": float(np.mean(list(a_fail.values()))),
                "repair_value": float(deltas.mean()),
                "repair_value_se": float(deltas.std(ddof=1) / np.sqrt(len(deltas))),
                "repair_value_by_seed": {int(s): float(d) for s, d in zip(seeds, deltas)},
                "nominal_change": float(np.mean([nom[s] - a_nom[s] for s in seeds])),
                "train_steps": int(per[seeds[0]][world][arm]["steps"]),
            }
    return out


def attach_repair(rows: list[dict], outcomes: dict) -> list[dict]:
    return [{**r, **outcomes[(r["world"], r["arm"])]} for r in rows]


def artifact_provenance(*paths: Path) -> dict:
    return {str(p): file_hash(Path(p)) for p in paths}
