"""Public artifact pipeline. Real simulator operations use an explicit backend config."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .provenance import file_hash, write_artifact


def read(path):
    return json.loads(Path(path).read_text())


def backend_from_config(path):
    from .backends.phoenix import create_backend
    return create_backend(read(path))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ashfall', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('demo', help='deterministic CPU software loop, no locomotion claim')
    p.add_argument('--output', required=True)
    p = commands.add_parser('capture',
                            help='archive an existing Phoenix recording with content identity')
    p.add_argument('--trajectory', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('capsule')
    sub = p.add_subparsers(dest='capsule_command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--trajectory', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--robot', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--timestamp', required=True)
    p.add_argument('--pre-seconds', type=float, default=.5)
    p.add_argument('--output', required=True)
    p = commands.add_parser('reproduce')
    p.add_argument('--capsule', required=True)
    p.add_argument('--config', required=True,
                   help='frozen reproduction protocol and search bounds JSON')
    p.add_argument('--backend-config', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('basin')
    p.add_argument('--reproduction', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('frontier')
    p.add_argument('--scenarios', required=True)
    p.add_argument('--observations', required=True)
    p.add_argument('--severity', required=True)
    p.add_argument('--split', choices=['train', 'validation', 'held_out'], default='validation')
    p.add_argument('--output', required=True)
    p = commands.add_parser('evaluate')
    p.add_argument('--scenarios', required=True)
    p.add_argument('--capsules', nargs='+', required=True)
    p.add_argument('--backend-config', required=True)
    p.add_argument('--policy-id', required=True)
    p.add_argument('--evaluation-seeds', type=int, nargs='+', required=True)
    p.add_argument('--training-seed', type=int)
    p.add_argument('--split', choices=['train', 'validation', 'held_out'], default='held_out')
    p.add_argument('--output', required=True)
    p = commands.add_parser('repair')
    p.add_argument('--config', required=True, help='complete frozen PPO repair run specification')
    p = commands.add_parser('verdict')
    p.add_argument('--config', required=True,
                   help='paths to protocol, manifests, reproduction and actual episode evidence')
    p.add_argument('--output', required=True)
    p = commands.add_parser('manifest')
    p.add_argument('--config', required=True, help='build_manifest keyword arguments')
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'demo':
        from .demo import run_demo
        result = run_demo(args.output)
        print(json.dumps(result, indent=2))
    elif args.command == 'capture':
        # Capture is an immutable ingestion boundary, not a robot command.
        from phoenix.replay.trajectory_reader import TrajectoryReader
        reader = TrajectoryReader(args.trajectory)
        if len(reader) == 0:
            raise ValueError('Cannot capture an empty trajectory')
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        digest = file_hash(args.trajectory)
        target = out / f'{digest}.parquet'
        if target.exists() and file_hash(target) != digest:
            raise ValueError('Existing captured content changed')
        if not target.exists():
            shutil.copyfile(args.trajectory, target)
        write_artifact(out / f'{digest}.json',
                       {'trajectory_sha256': digest, 'rows': len(reader),
                        'artifact': target.name, 'physical_origin': 'requires_review'})
    elif args.command == 'capsule':
        from .capsule import capsules_from_parquet
        policy_hash = file_hash(args.checkpoint)
        capsules = capsules_from_parquet(args.trajectory, source=args.source, robot=args.robot,
                    policy_id=policy_hash, timestamp=args.timestamp,
                    pre_failure_seconds=args.pre_seconds, checkpoint_sha256=policy_hash)
        for capsule in capsules:
            capsule.save(Path(args.output) / f'{capsule.capsule_id}.json')
        if not capsules:
            raise ValueError('No failure events found; manually review onset labels')
    elif args.command == 'reproduce':
        from .capsule import FailureCapsule
        from .reproduction import FailureDescriptor, ReproductionConfig, ReproductionGate
        capsule = FailureCapsule.load(args.capsule)
        spec = read(args.config)
        cfg = dict(spec['gate'])
        cfg['target'] = FailureDescriptor(**cfg['target'])
        cfg['seeds'] = tuple(cfg['seeds'])
        backend = backend_from_config(args.backend_config)
        try:
            results = ReproductionGate(ReproductionConfig(**cfg)).search(capsule, backend,
                spec['bounds'], count=spec['count'], search_seed=spec['search_seed'])
            for result in results:
                result.save(Path(args.output) / f'{result.reproduction_id}.json')
        finally:
            backend.close()
    elif args.command == 'basin':
        from .basin import discover_basin
        from .reproduction import load_reproduction
        manifest = discover_basin(load_reproduction(args.reproduction), **read(args.config))
        manifest.save(args.output)
    elif args.command == 'frontier':
        from .basin import BasinObservation
        from .frontier import FrontierEstimator, SeverityMetric
        from .scenarios import ScenarioManifest
        raw = read(args.observations)['observations']
        fields = {'scenario_id', 'policy_id', 'evaluation_seeds', 'failures', 'descriptors'}
        observations = [BasinObservation(**{k: tuple(v) if isinstance(v, list) else v
                                             for k, v in o.items() if k in fields}) for o in raw]
        estimator = FrontierEstimator(SeverityMetric(**read(args.severity)))
        points = estimator.points(ScenarioManifest.load(args.scenarios), observations,
                                  splits=(args.split,))
        quantiles = [estimator.estimate(points, q).to_dict() for q in (.1, .5, .9)]
        write_artifact(args.output, {'quantiles': quantiles,
                                     'points': [asdict(p) for p in points]})
    elif args.command == 'evaluate':
        from phoenix.training.episode_outcomes import write_outcomes

        from .basin import BasinObservation, save_observations
        from .capsule import FailureCapsule
        from .scenarios import ScenarioManifest
        capsules = {c.capsule_id: c for c in map(FailureCapsule.load, args.capsules)}
        manifest = ScenarioManifest.load(args.scenarios)
        backend = backend_from_config(args.backend_config)
        try:
            records = backend.evaluate(manifest, capsules, policy_id=args.policy_id,
                        evaluation_seeds=args.evaluation_seeds,
                        training_seed=args.training_seed, split=args.split)
            write_outcomes(args.output, records)
            groups = {}
            for r in records:
                groups.setdefault(r.scenario_id, []).append(r)
            observations = [BasinObservation(sid, args.policy_id,
                                tuple(r.evaluation_seed for r in rs),
                                tuple(not r.success for r in rs), tuple(asdict(r) for r in rs))
                            for sid, rs in groups.items()]
            save_observations(str(args.output)+'.basin.json', observations)
        finally:
            backend.close()
    elif args.command == 'repair':
        from .training import run_repair
        run_repair(read(args.config))
    elif args.command == 'verdict':
        from .workflow import verdict_from_files
        write_artifact(args.output, asdict(verdict_from_files(read(args.config))))
    elif args.command == 'manifest':
        from .provenance import build_manifest
        build_manifest(**read(args.config)).save(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
