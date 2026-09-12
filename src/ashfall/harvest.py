"""Failure capsules from physics rollouts, never from constructed kinematics.

A harvested capsule is a :class:`ashfall.capsule.FailureCapsule` at schema 1.1
whose frames are the rows of a real rollout trace, whose ``intervention`` is
the cause that was verifiably applied, whose ``phenotype_label`` is what the
detector observed, whose ``onset_window`` carries the precursor step from the
D2 departure profile, the transition and the established (detector) step, and
whose ``provenance`` names the policy, environment, simulator, seed and
repositories. Every capsule set is written with a :class:`DatasetManifest`
whose kind follows the evidence: ``scientific`` for simulator or hardware
traces, ``fixture`` for the toy surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ashfall.capsule import CapsuleFrame, FailureCapsule
from ashfall.counterfactual import RolloutTrace
from ashfall.datasets import DatasetManifest
from ashfall.h0 import H0Result, H0Spec, PairRecord
from ashfall.ontology import OnsetWindow
from ashfall.provenance import write_artifact
from ashfall.synth.generator import SCHEMA as PHOENIX_SCHEMA


def trace_to_frames(trace: RolloutTrace) -> tuple[CapsuleFrame, ...]:
    return tuple(
        CapsuleFrame(
            timestamp_s=step * trace.control_dt,
            base_pos=tuple(trace.base_pos[step]),
            base_quat=tuple(trace.base_quat[step]),
            base_lin_vel_body=tuple(trace.base_lin_vel_body[step]),
            base_ang_vel_body=tuple(trace.base_ang_vel_body[step]),
            joint_pos=tuple(trace.joint_pos[step]),
            joint_vel=tuple(trace.joint_vel[step]),
            command_vel=tuple(trace.command_vel[step]),
            action=None if trace.actions is None else tuple(trace.actions[step]),
            contact_forces=(
                None if trace.contact_forces is None else tuple(trace.contact_forces[step])
            ),
        )
        for step in range(trace.n_steps)
    )


def write_trace_parquet(
    trace: RolloutTrace, path: str | Path, *, onset_step: int | None, phenotype: str | None
) -> Path:
    """Write a trace in the Phoenix trajectory schema so TrajectoryPool can read it.

    ``failure_flag`` marks rows at or after ``onset_step`` and ``failure_mode``
    carries the detector's label on those rows only, matching the harvest
    convention on the Phoenix side.
    """
    rows = []
    for step in range(trace.n_steps):
        failed = onset_step is not None and step >= onset_step
        rows.append(
            {
                "step": step,
                "timestamp_s": step * trace.control_dt,
                "base_pos": trace.base_pos[step].astype(np.float32).tolist(),
                "base_quat": trace.base_quat[step].astype(np.float32).tolist(),
                "base_lin_vel_body": trace.base_lin_vel_body[step].astype(np.float32).tolist(),
                "base_ang_vel_body": trace.base_ang_vel_body[step].astype(np.float32).tolist(),
                "joint_pos": trace.joint_pos[step].astype(np.float32).tolist(),
                "joint_vel": trace.joint_vel[step].astype(np.float32).tolist(),
                "command_vel": trace.command_vel[step].astype(np.float32).tolist(),
                "action": (
                    trace.actions[step].astype(np.float32).tolist()
                    if trace.actions is not None
                    else [0.0] * 12
                ),
                "contact_forces": (
                    trace.contact_forces[step].astype(np.float32).tolist()
                    if trace.contact_forces is not None
                    else [0.0] * 4
                ),
                "failure_flag": bool(failed),
                "failure_mode": phenotype if failed else None,
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=PHOENIX_SCHEMA), path, compression="zstd")
    return path


def capsule_from_pair(
    record: PairRecord,
    *,
    robot: str,
    post_failure_seconds: float = 1.0,
    extra_provenance: Mapping[str, Any] | None = None,
) -> FailureCapsule | None:
    """Build a schema 1.1 capsule from a DELIVERED pair, or None when it was not delivered.

    Only delivered pairs become capsules: an undelivered pair has no phenotype
    to condition on, and turning it into a capsule would recreate the
    row-0 defect with better paperwork.
    """
    verdict = record.attempt.verdict
    if not verdict.delivered:
        return None
    trace = record.evidence.treatment
    d3 = verdict.d3_phenotype.detail
    established = next(
        o["window"]["established_start"]
        for o in d3["treatment_observations"]
        if o["phenotype"] == record.attempt.intended_phenotype
    )
    precursor = verdict.d2_departure.detail.get("precursor_step")
    if precursor is not None and precursor > established:
        precursor = None
    transition = precursor if precursor is not None else established
    window = OnsetWindow(
        transition_start=int(transition),
        established_start=int(established),
        precursor_start=None if precursor is None else int(precursor),
        end=None,
        label_source="detector",
    )
    frames = trace_to_frames(trace)
    post_frames = int(round(post_failure_seconds / trace.control_dt))
    end = min(len(frames) - 1, established + post_frames)
    if (
        established >= len(frames)
        or transition >= established
        and precursor is None
        and established == 0
    ):
        return None
    provenance = {
        **record.treatment_episode.provenance,
        "control_episode_id": record.control_episode.episode_id,
        "treatment_episode_id": record.treatment_episode.episode_id,
        "attempt_id": record.attempt.attempt_id,
        "initial_state_id": record.state.state_id,
        **(extra_provenance or {}),
    }
    source = record.treatment_episode.source
    return FailureCapsule(
        source=source,
        robot=robot,
        policy_id=record.attempt.policy_id,
        timestamp=None,
        control_dt=trace.control_dt,
        failure_mode=record.attempt.intended_phenotype,
        failure_onset_index=int(established),
        pre_failure_start_index=int(transition),
        post_failure_end_index=int(end),
        frames=frames,
        detector_version=d3.get("detector"),
        event_descriptor={"onset_label_source": "detector", "precursor_source": "d2_departure"},
        intervention=record.attempt.intervention.to_dict(),
        phenotype_label=record.attempt.intended_phenotype,
        onset_window=window.to_dict(),
        provenance=provenance,
    )


@dataclass(frozen=True)
class HarvestSummary:
    directory: str
    dataset_id: str
    dataset_kind: str
    capsules: int
    pairs: int
    delivered: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def harvest_h0_run(
    spec: H0Spec,
    result: H0Result,
    records: Sequence[PairRecord],
    output_dir: str | Path,
    *,
    robot: str,
    provenance: Mapping[str, Any],
) -> HarvestSummary:
    """Persist an H0 run: spec, result, attempts, episodes, traces, capsules, manifest.

    ``provenance`` must carry the repository revisions and the policy identity
    (``ashfall_sha``, ``phoenix_sha``, ``policy_id``); the manifest refuses to
    call the harvest scientific without them.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_artifact(out / "h0_spec.json", spec.to_dict())
    write_artifact(out / "h0_result.json", result.to_dict())
    write_artifact(out / "attempts.json", [r.attempt.to_dict() for r in records])
    write_artifact(
        out / "episodes.json",
        [e.to_dict() for r in records for e in (r.control_episode, r.treatment_episode)],
    )
    capsules = []
    for index, record in enumerate(records):
        d3 = record.attempt.verdict.d3_phenotype.detail
        onset = None
        observed = d3.get("treatment_observations") or []
        for o in observed:
            if o["phenotype"] == record.attempt.intended_phenotype:
                onset = o["window"]["established_start"]
        write_trace_parquet(
            record.evidence.treatment,
            out / f"pair_{index:03d}_treatment.parquet",
            onset_step=onset,
            phenotype=record.attempt.intended_phenotype if onset is not None else None,
        )
        write_trace_parquet(
            record.evidence.control,
            out / f"pair_{index:03d}_control.parquet",
            onset_step=None,
            phenotype=None,
        )
        capsule = capsule_from_pair(record, robot=robot)
        if capsule is not None:
            capsule.save(out / "capsules" / f"{capsule.capsule_id}.json")
            capsules.append(capsule.capsule_id)
    manifest_provenance = {
        **dict(provenance),
        "h0_spec_id": spec.spec_id,
        "h0_verdict": result.verdict,
        "seeds": [s["simulator_seed"] for s in _seed_plan(spec)],
        "env_config_hash": provenance.get(
            "env_config_hash",
            records[0].evidence.treatment.metadata.get("env_config_hash") if records else None,
        ),
        "simulator_version": provenance.get("simulator_version"),
    }
    if result.evidence_kind == "simulation":
        manifest = DatasetManifest(
            "scientific",
            "simulation",
            _hash_dir(out),
            (
                f"H0 matched-pair harvest: {spec.intervention.kind} -> {spec.intended_phenotype}, "
                f"{result.n_pairs} pairs, verdict {result.verdict}."
            ),
            manifest_provenance,
            spec.intervention.to_dict(),
        )
    else:
        manifest = DatasetManifest(
            "fixture",
            "synthetic_generator",
            {k: v for k, v in _hash_dir(out).items()},
            f"Toy-surrogate H0 fixture: {spec.intervention.kind} -> {spec.intended_phenotype}. "
            "Mock evidence; never a scientific input.",
            manifest_provenance,
            spec.intervention.to_dict(),
        )
    manifest.save(out)
    return HarvestSummary(
        str(out), manifest.dataset_id, manifest.kind, len(capsules), len(records), result.delivered
    )


def _seed_plan(spec: H0Spec) -> list[dict]:
    from ashfall.h0 import h0_seed_plan

    return h0_seed_plan(spec)


def _hash_dir(directory: Path) -> dict[str, str]:
    """Relative-path keyed hashes so the manifest verifies capsules in their subdirectory."""
    from ashfall.provenance import file_hash

    files = sorted(
        {
            p
            for pattern in ("*.parquet", "*.json", "capsules/*.json")
            for p in directory.glob(pattern)
        }
    )
    return {str(p.relative_to(directory)): file_hash(p) for p in files if p.name != "dataset.json"}


__all__ = [
    "HarvestSummary",
    "capsule_from_pair",
    "harvest_h0_run",
    "trace_to_frames",
    "write_trace_parquet",
]
