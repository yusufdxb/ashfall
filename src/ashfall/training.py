"""Executable frontier PPO integration through Phoenix reset primitives.

Frontier probes run in isolated simulator subprocesses at PPO chunk boundaries.
Only the resulting scenario-level boolean outcomes affect reset sampling. Probe
trajectories are never supplied to PPO. This orchestration requires Isaac Lab.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .capsule import FailureCapsule
from .provenance import build_manifest, file_hash, write_artifact
from .repair import FrontierSchedule, RepairCurriculum
from .reproduction import load_reproduction
from .scenarios import ScenarioManifest


def validate_repair_spec(spec):
    required = {'scenarios', 'reproductions', 'capsules', 'train_config', 'backend_config',
                'ashfall_repo', 'phoenix_repo', 'output', 'training_seed', 'iterations',
                'num_envs', 'evaluation_seeds', 'schedule'}
    if required-set(spec):
        raise ValueError(f'Missing repair specification fields: {sorted(required-set(spec))}')
    if spec['iterations'] <= 0 or spec['num_envs'] <= 0:
        raise ValueError('Positive iteration/env counts required')
    manifest = ScenarioManifest.load(spec['scenarios'])
    reproductions = [load_reproduction(p) for p in spec['reproductions']]
    # Disallow mock evidence in this executable simulator path.
    curriculum = RepairCurriculum(manifest, reproductions,
                  schedule=FrontierSchedule(**spec['schedule']),
                  training_seed=spec['training_seed'])
    capsules = {c.capsule_id: c for c in map(FailureCapsule.load, spec['capsules'])}
    states = {r.reproduction_id: r.candidate.seed_row for r in reproductions}
    for scenario in manifest.select('train'):
        if scenario.capsule_id not in capsules:
            raise ValueError('Training capsule artifact missing')
        frame = capsules[scenario.capsule_id].frames[states[scenario.reproduction_id]]
        if frame.missing_reset_fields:
            raise ValueError('Training capsule lacks reset state')
    return manifest, reproductions, curriculum, capsules, states


def run_repair(spec):
    manifest, reproductions, curriculum, capsules, states = validate_repair_spec(spec)
    backend_config = json.loads(Path(spec['backend_config']).read_text())
    baseline_hash = file_hash(backend_config['checkpoint'])
    if any(r.config.baseline_policy_id != baseline_hash for r in reproductions):
        raise ValueError('Repair checkpoint differs from reproduced baseline')
    out = Path(spec['output'])
    out.mkdir(parents=True, exist_ok=False)
    manifest_record = build_manifest(ashfall_repo=spec['ashfall_repo'],
        phoenix_repo=spec['phoenix_repo'],
        config=spec, checkpoint=backend_config['checkpoint'], training_seed=spec['training_seed'],
        scenario_manifest_hash=manifest.manifest_hash, capsule_ids=capsules,
        command=[sys.executable, '-m', 'ashfall.cli', 'repair', '--config', 'repair.json'],
        simulator_version=backend_config.get('simulator_version'))
    manifest_record.save(out / 'experiment.json')
    write_artifact(out / 'repair.json', spec)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    env = None
    try:
        from importlib import metadata

        import gymnasium as gym
        import yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from phoenix.adaptation.scenario_bridge import (
            FrictionScenarioAdapter,
            install_scenario_reset,
            learn_with_frontier_updates,
        )
        from phoenix.replay.trajectory_reader import InitialState
        from phoenix.sim_env import build_env_cfg, load_layered_config
        from phoenix.training.agent_cfg import build_runner_cfg
        from phoenix.training.checkpoint import load_runner_checkpoint
        from phoenix.training.episode_outcomes import load_outcomes
        from rsl_rl.runners import OnPolicyRunner

        loaded = load_layered_config(backend_config['env_config'])
        cfg = build_env_cfg(loaded)
        cfg.scene.num_envs = spec['num_envs']
        cfg.seed = spec['training_seed']
        cfg.sim.device = backend_config.get('device', 'cuda:0')
        task = loaded.to_container()['env']['task_name']
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=1.)
        train_cfg = yaml.safe_load(Path(spec['train_config']).read_text())
        train_cfg['run'].update(seed=spec['training_seed'], device=cfg.sim.device,
                                max_iterations=spec['iterations'])
        runner_cfg = handle_deprecated_rsl_rl_cfg(build_runner_cfg(train_cfg, task),
                                                  metadata.version('rsl-rl-lib'))
        runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(out/'ppo'),
                                device=cfg.sim.device)
        info = load_runner_checkpoint(runner, backend_config['checkpoint'], load_actor=True,
                load_critic=True, load_optimizer=False, load_iteration=False)
        if not info['actor_match']:
            raise RuntimeError('Training baseline checkpoint did not round-trip')
        materials = FrictionScenarioAdapter(env.unwrapped)

        def resolve(scenario):
            capsule = capsules[scenario.capsule_id]
            row = states[scenario.reproduction_id]
            f = capsule.frames[row]
            state = InitialState(**{k: getattr(f, k) for k in ('base_pos', 'base_quat',
                'base_lin_vel_body', 'base_ang_vel_body', 'joint_pos', 'joint_vel', 'command_vel')})
            return state, dict(capsule_id=capsule.capsule_id, requested_seed_row=row,
                 resolved_row=row, failure_onset_row=capsule.failure_onset_index,
                 time_before_onset_s=capsule.frames[capsule.failure_onset_index].timestamp_s-f.timestamp_s)

        telemetry_path = out / 'reset_telemetry.jsonl'
        def record(data):
            with telemetry_path.open('a') as stream:
                stream.write(json.dumps(data, sort_keys=True, allow_nan=False)+'\n')
        control = install_scenario_reset(env, curriculum.sample, resolve,
                    lambda i, params: materials.apply(i, dict(params)),
                    reset_parameters=materials.reset, on_reset=record)

        def reestimate(live_runner, completed):
            checkpoint = out / f'frontier_probe_{completed}.pt'
            live_runner.save(str(checkpoint))
            current_id = file_hash(checkpoint)
            probe_config = {**backend_config, 'checkpoint': str(checkpoint),
                            'train_config': spec['train_config']}
            probe_path = out / f'probe_{completed}.json'
            write_artifact(probe_path, probe_config)
            episodes = out / f'probe_{completed}.episodes.jsonl'
            command = [sys.executable, '-m', 'ashfall.cli', 'evaluate',
                '--scenarios', spec['scenarios'],
                '--capsules', *spec['capsules'], '--backend-config', str(probe_path),
                '--policy-id', current_id, '--training-seed', str(spec['training_seed']),
                '--evaluation-seeds', *map(str, spec['evaluation_seeds']), '--split', 'train',
                '--output', str(episodes)]
            write_artifact(out / f'probe_{completed}.command.json', command)
            with (out / f'probe_{completed}.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            rows = load_outcomes(episodes)
            outcomes = {s.scenario_id: [] for s in manifest.select('train')}
            expected = {(sid, seed) for sid in outcomes for seed in spec['evaluation_seeds']}
            observed = {(r.scenario_id, r.evaluation_seed) for r in rows}
            if observed != expected or len(rows) != len(expected):
                raise RuntimeError('Incomplete or duplicate frontier probe episodes')
            for row in rows:
                if row.policy_id != current_id or row.failure_modes is None:
                    raise RuntimeError('Frontier probe policy/event capture mismatch')
                outcomes[row.scenario_id].append(not row.success or bool(row.failure_modes))
            curriculum.update(outcomes, current_id, completed)
            return current_id

        learn_with_frontier_updates(runner, total_iterations=spec['iterations'],
             update_interval=curriculum.schedule.refresh_interval, reestimate=reestimate,
             control=control)
        runner.save(str(out / 'candidate.pt'))
        write_artifact(out / 'candidate.json', {'policy_id': file_hash(out/'candidate.pt'),
                       'training_seed': spec['training_seed'], 'accepted': False,
                       'status': 'REQUIRES_FROZEN_TARGET_AND_NOMINAL_EVALUATION'})
    finally:
        if env is not None:
            env.close()
        app.close()
