"""Public artifact pipeline. Real simulator operations use an explicit backend config.

Subcommands added by the causal-repair redesign:

``h0``
    Run the preregistered matched-counterfactual delivery calibration for one
    intervention -> phenotype pathway against the toy surrogate (mock
    evidence) or the Phoenix backend (simulation evidence), and write one
    evidence bundle.
``detector-eval``
    Score the phenotype detector on an independently labelled dataset and run
    the mutation suite. The bundled fixture dataset yields a
    ``fixture_regression`` report, never a validation claim.
``env-snapshot``
    Write the interpreter, package, simulator, CUDA and GPU snapshot that every
    bundle carries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from .provenance import file_hash, write_artifact

REPO_ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def _expand(value):
    """Expand ``${VAR}`` in string values of a JSON config, recursively."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def backend_from_config(path):
    from .backends.phoenix import create_backend

    return create_backend(_expand(read(path)))


def _build_parser():
    parser = argparse.ArgumentParser(prog="ashfall", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("demo", help="deterministic CPU software loop, no locomotion claim")
    p.add_argument("--output", required=True)
    p = commands.add_parser(
        "capture", help="archive an existing Phoenix recording with content identity"
    )
    p.add_argument("--trajectory", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("capsule")
    sub = p.add_subparsers(dest="capsule_command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--trajectory", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--robot", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--timestamp", required=True)
    p.add_argument("--pre-seconds", type=float, default=0.5)
    p.add_argument("--output", required=True)
    p = commands.add_parser("reproduce")
    p.add_argument("--capsule", required=True)
    p.add_argument(
        "--config", required=True, help="frozen reproduction protocol and search bounds JSON"
    )
    p.add_argument("--backend-config", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("basin")
    p.add_argument("--reproduction", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("frontier")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--severity", required=True)
    p.add_argument("--split", choices=["train", "validation", "held_out"], default="validation")
    p.add_argument("--output", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--capsules", nargs="+", required=True)
    p.add_argument("--backend-config", required=True)
    p.add_argument("--policy-id", required=True)
    p.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    p.add_argument("--training-seed", type=int)
    p.add_argument("--split", choices=["train", "validation", "held_out"], default="held_out")
    p.add_argument("--output", required=True)
    p = commands.add_parser("repair")
    p.add_argument("--config", required=True, help="complete frozen PPO repair run specification")
    p = commands.add_parser("verdict")
    p.add_argument(
        "--config",
        required=True,
        help="paths to protocol, manifests, reproduction and actual episode evidence",
    )
    p.add_argument("--output", required=True)
    p = commands.add_parser("manifest")
    p.add_argument("--config", required=True, help="build_manifest keyword arguments")
    p.add_argument("--output", required=True)

    p = commands.add_parser("h0", help="matched-counterfactual delivery calibration (H0)")
    p.add_argument("--spec", required=True, help="preregistered H0 spec JSON")
    p.add_argument("--backend", choices=["toy", "phoenix"], required=True)
    p.add_argument("--backend-config", help="Phoenix backend JSON; required for --backend phoenix")
    p.add_argument("--output", required=True, help="parent directory for the evidence bundle")
    p.add_argument("--robot", default="go2")
    p.add_argument(
        "--allow-config-block",
        action="append",
        default=[],
        help="acknowledge a declared-but-unapplied env config block as documentation only",
    )
    p.add_argument("--phoenix-repo", help="path of the go2-phoenix checkout for provenance")
    p.add_argument("--sensitivity-pairs", type=int, default=2)

    p = commands.add_parser("detector-eval", help="independent detector evaluation + mutations")
    p.add_argument("--dataset", help="labelled JSONL dataset; default is the regression fixture")
    p.add_argument("--output", required=True)

    p = commands.add_parser("env-snapshot", help="write the environment snapshot")
    p.add_argument("--output", required=True)
    return parser


def _run_h0(args) -> dict:
    from .config_guard import assert_config_applied, load_effective_env_config
    from .counterfactual import DepartureConfig, sensitivity_analysis
    from .h0 import H0Spec, h0_seed_plan, run_h0, select_states
    from .harvest import harvest_h0_run
    from .ontology import Intervention
    from .provenance import EvidenceBundle, collect_provenance, environment_snapshot

    if args.backend == "phoenix" and not args.backend_config:
        raise SystemExit("--backend-config is required for --backend phoenix")
    raw = read(args.spec)
    departure = DepartureConfig(**raw.get("departure", {}))
    command = tuple(raw.get("command", (0.5, 0.0, 0.0)))
    spec = H0Spec(
        Intervention.from_dict(raw["intervention"]),
        raw["intended_phenotype"],
        n_pairs=int(raw["n_pairs"]),
        replicates_per_pair=int(raw["replicates_per_pair"]),
        horizon_steps=int(raw["horizon_steps"]),
        alpha=float(raw.get("alpha", 0.05)),
        min_delivered_fraction=float(raw.get("min_delivered_fraction", 0.5)),
        departure=departure,
        platform=raw.get("platform", "simulation"),
        seed_base=int(raw.get("seed_base", 10_000)),
    )
    config_paths = {"h0_spec": args.spec}
    policy_paths = {}
    acknowledged: list[str] = []
    if args.backend == "toy":
        from .backends.toy import ToyBackend

        backend = ToyBackend()
        policy_paths["toy_policy"] = args.spec  # the surrogate has no checkpoint file
        phoenix_repo = None
    else:
        backend_config = _expand(read(args.backend_config))
        config_paths.update(
            backend_config_file=args.backend_config,
            env_config=backend_config["env_config"],
            train_config=backend_config["train_config"],
        )
        policy_paths["checkpoint"] = backend_config["checkpoint"]
        effective = load_effective_env_config(backend_config["env_config"])
        acknowledged = assert_config_applied(
            effective, purpose="H0 calibration", allow=tuple(args.allow_config_block)
        )
        backend = backend_from_config(args.backend_config)
        phoenix_repo = args.phoenix_repo or backend_config.get("phoenix_repo")
    bundle = EvidenceBundle.create(
        args.output,
        {
            "kind": "h0_delivery_calibration",
            "spec": spec.to_dict(),
            "command": list(command),
            "backend": args.backend,
            "config_hashes": {k: file_hash(v) for k, v in config_paths.items()},
        },
    )
    bundle.write_environment(environment_snapshot())
    provenance = collect_provenance(
        ashfall_repo=REPO_ROOT,
        phoenix_repo=phoenix_repo,
        config_paths=config_paths,
        policy_paths=policy_paths,
        seeds={"seed_base": spec.seed_base},
    )
    bundle.write_provenance(provenance)
    write_artifact(bundle.root / "seed_plan.json", h0_seed_plan(spec))
    try:
        states = select_states(backend, spec, command=command)
        result, records = run_h0(
            backend,
            spec,
            states=states,
            provenance={
                "ashfall_sha": provenance.ashfall["sha"],
                "phoenix_sha": None if provenance.phoenix is None else provenance.phoenix["sha"],
            },
        )
        sensitivity = []
        for record in records[: max(0, args.sensitivity_pairs)]:
            evidence = record.evidence
            report = sensitivity_analysis(
                evidence.treatment,
                evidence.control,
                evidence.nominal_replicates,
                config=spec.departure,
                phenotype=spec.intended_phenotype,
            )
            sensitivity.append({"attempt_id": record.attempt.attempt_id, **report.to_dict()})
        harvest = harvest_h0_run(
            spec,
            result,
            records,
            bundle.root / "harvest",
            robot=args.robot,
            provenance={
                "ashfall_sha": provenance.ashfall["sha"],
                "phoenix_sha": None if provenance.phoenix is None else provenance.phoenix["sha"],
                "policy_id": backend.policy_id,
                "env_config_hash": backend.provenance().get("env_config_hash"),
                "simulator_version": backend.provenance().get("simulator_version"),
            },
        )
    finally:
        backend.close()
    metrics = {
        "h0": result.to_dict(),
        "per_pair": [
            {
                "attempt_id": r.attempt.attempt_id,
                "status": r.attempt.verdict.status,
                "d2_ratio": r.attempt.verdict.d2_departure.detail.get("ratio_to_null_max"),
                "treatment_phenotypes": r.attempt.verdict.d3_phenotype.detail.get(
                    "observed_phenotypes"
                ),
                "control_phenotypes": r.attempt.verdict.d3_phenotype.detail.get(
                    "control_phenotypes"
                ),
                "treatment_termination": r.evidence.treatment.termination_reason,
                "control_termination": r.evidence.control.termination_reason,
            }
            for r in records
        ],
        "d2_sensitivity": sensitivity,
        "harvest": harvest.to_dict(),
        "acknowledged_unapplied_config_blocks": acknowledged,
    }
    bundle.write_metrics(metrics)
    bundle.write_verdict(
        {
            "hypothesis": "H0_causal_delivery",
            "pathway": f"{spec.intervention.kind} -> {spec.intended_phenotype}",
            "verdict": result.verdict,
            "reasons": list(result.reasons),
            "evidence_kind": result.evidence_kind,
            "is_research_result": result.evidence_kind == "simulation",
            "spec_id": spec.spec_id,
        }
    )
    bundle.finalize()
    return {
        "bundle": str(bundle.root),
        "run_id": bundle.run_id,
        "verdict": result.verdict,
        "evidence_kind": result.evidence_kind,
        "statuses": result.statuses,
        "exact_p": result.exact_p,
        "delivered_fraction": result.delivered_fraction,
        "reasons": list(result.reasons),
        "capsules": harvest.capsules,
    }


def _run_detector_eval(args) -> dict:
    from .detector_eval.dataset import DetectorEvalDataset
    from .detector_eval.fixtures import fixture_dataset
    from .detector_eval.metrics import evaluate_detector, render_markdown
    from .detector_eval.mutation import run_mutation_suite

    dataset = DetectorEvalDataset.load(args.dataset) if args.dataset else fixture_dataset()
    report = evaluate_detector(dataset)
    mutations = run_mutation_suite(dataset)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_artifact(out / "detector_report.json", report.to_dict())
    write_artifact(out / "mutation_report.json", mutations.to_dict())
    (out / "detector_report.md").write_text(render_markdown(report))
    summary = {
        "report_kind": report.report_kind,
        "validation_claim_allowed": report.validation_claim_allowed,
        "episodes": len(dataset.episodes),
        "mutants_caught": [m.mutation for m in mutations.outcomes if m.caught],
        "mutants_missed": [m.mutation for m in mutations.outcomes if not m.caught],
        "output": str(out),
    }
    return summary


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "demo":
        from .demo import run_demo

        result = run_demo(args.output)
        print(json.dumps(result, indent=2))
    elif args.command == "capture":
        # Capture is an immutable ingestion boundary, not a robot command.
        from phoenix.replay.trajectory_reader import TrajectoryReader

        reader = TrajectoryReader(args.trajectory)
        if len(reader) == 0:
            raise ValueError("Cannot capture an empty trajectory")
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        digest = file_hash(args.trajectory)
        target = out / f"{digest}.parquet"
        if target.exists() and file_hash(target) != digest:
            raise ValueError("Existing captured content changed")
        if not target.exists():
            shutil.copyfile(args.trajectory, target)
        write_artifact(
            out / f"{digest}.json",
            {
                "trajectory_sha256": digest,
                "rows": len(reader),
                "artifact": target.name,
                "physical_origin": "requires_review",
            },
        )
    elif args.command == "capsule":
        from .capsule import capsules_from_parquet

        policy_hash = file_hash(args.checkpoint)
        capsules = capsules_from_parquet(
            args.trajectory,
            source=args.source,
            robot=args.robot,
            policy_id=policy_hash,
            timestamp=args.timestamp,
            pre_failure_seconds=args.pre_seconds,
            checkpoint_sha256=policy_hash,
        )
        for capsule in capsules:
            capsule.save(Path(args.output) / f"{capsule.capsule_id}.json")
        if not capsules:
            raise ValueError("No failure events found; manually review onset labels")
    elif args.command == "reproduce":
        from .capsule import FailureCapsule
        from .reproduction import FailureDescriptor, ReproductionConfig, ReproductionGate

        capsule = FailureCapsule.load(args.capsule)
        spec = read(args.config)
        cfg = dict(spec["gate"])
        cfg["target"] = FailureDescriptor(**cfg["target"])
        cfg["seeds"] = tuple(cfg["seeds"])
        backend = backend_from_config(args.backend_config)
        try:
            results = ReproductionGate(ReproductionConfig(**cfg)).search(
                capsule,
                backend,
                spec["bounds"],
                count=spec["count"],
                search_seed=spec["search_seed"],
            )
            for result in results:
                result.save(Path(args.output) / f"{result.reproduction_id}.json")
        finally:
            backend.close()
    elif args.command == "basin":
        from .basin import discover_basin
        from .reproduction import load_reproduction

        manifest = discover_basin(load_reproduction(args.reproduction), **read(args.config))
        manifest.save(args.output)
    elif args.command == "frontier":
        from .basin import BasinObservation
        from .frontier import FrontierEstimator, SeverityMetric
        from .scenarios import ScenarioManifest

        raw = read(args.observations)["observations"]
        fields = {"scenario_id", "policy_id", "evaluation_seeds", "failures", "descriptors"}
        observations = [
            BasinObservation(
                **{k: tuple(v) if isinstance(v, list) else v for k, v in o.items() if k in fields}
            )
            for o in raw
        ]
        estimator = FrontierEstimator(SeverityMetric(**read(args.severity)))
        points = estimator.points(
            ScenarioManifest.load(args.scenarios), observations, splits=(args.split,)
        )
        quantiles = [estimator.estimate(points, q).to_dict() for q in (0.1, 0.5, 0.9)]
        write_artifact(args.output, {"quantiles": quantiles, "points": [asdict(p) for p in points]})
    elif args.command == "evaluate":
        from phoenix.training.episode_outcomes import write_outcomes

        from .basin import BasinObservation, save_observations
        from .capsule import FailureCapsule
        from .scenarios import ScenarioManifest

        capsules = {c.capsule_id: c for c in map(FailureCapsule.load, args.capsules)}
        manifest = ScenarioManifest.load(args.scenarios)
        backend = backend_from_config(args.backend_config)
        try:
            records = backend.evaluate(
                manifest,
                capsules,
                policy_id=args.policy_id,
                evaluation_seeds=args.evaluation_seeds,
                training_seed=args.training_seed,
                split=args.split,
            )
            write_outcomes(args.output, records)
            groups = {}
            for r in records:
                groups.setdefault(r.scenario_id, []).append(r)
            observations = [
                BasinObservation(
                    sid,
                    args.policy_id,
                    tuple(r.evaluation_seed for r in rs),
                    tuple(not r.success for r in rs),
                    tuple(asdict(r) for r in rs),
                )
                for sid, rs in groups.items()
            ]
            save_observations(str(args.output) + ".basin.json", observations)
        finally:
            backend.close()
    elif args.command == "repair":
        from .training import run_repair

        run_repair(read(args.config))
    elif args.command == "verdict":
        from .workflow import verdict_from_files

        write_artifact(args.output, asdict(verdict_from_files(read(args.config))))
    elif args.command == "manifest":
        from .provenance import build_manifest

        build_manifest(**read(args.config)).save(args.output)
    elif args.command == "h0":
        print(json.dumps(_run_h0(args), indent=2))
    elif args.command == "detector-eval":
        print(json.dumps(_run_detector_eval(args), indent=2))
    elif args.command == "env-snapshot":
        from .provenance import environment_snapshot

        write_artifact(args.output, environment_snapshot())
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
