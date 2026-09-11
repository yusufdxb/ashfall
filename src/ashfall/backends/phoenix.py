"""Sequential Isaac Lab replay using Phoenix's state and terminal adapters.

One environment is constructed per episode, so startup randomization is seeded
by the frozen scenario seed. After reset and restoration the evaluation seed
controls rollout randomness. Only robot-shape friction is searched initially;
terrain material and other nuisance dynamics remain the declared environment
configuration, not inferred physical context. No simulation results are created
by importing or constructing this backend.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ashfall.evaluation.metrics import FailureAnalyzer
from ashfall.provenance import content_hash, write_artifact
from ashfall.reproduction import FailureDescriptor, ReproductionCandidate


def checkpoint_identity(path):
    """Policy identity is the exact checkpoint bytes, not its mutable filename."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_scenario_row(capsule, scenario):
    if scenario.capsule_id != capsule.capsule_id:
        raise ValueError("scenario belongs to a different capsule")
    for row in range(capsule.pre_failure_start_index, capsule.failure_onset_index + 1):
        if (
            content_hash({"capsule_id": capsule.capsule_id, "seed_row": row})
            == scenario.initial_state_id
        ):
            return row
    raise ValueError("scenario initial state identity is outside the reviewed capsule window")


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _policy_observations(policy, observations):
    if isinstance(observations, tuple):
        observations = observations[0]
    # rsl_rl MLPModel consumes TensorDict; older ActorCritic consumes a tensor.
    if hasattr(policy, "obs_groups"):
        return observations
    if hasattr(observations, "keys"):
        if "policy" not in observations.keys():
            raise RuntimeError("legacy actor needs explicit policy observation group")
        return observations["policy"]
    return observations


class PhoenixBackend:
    evidence_kind = "simulation"

    def __init__(
        self,
        checkpoint,
        env_config,
        train_config,
        *,
        device="cuda:0",
        horizon_s=10.0,
        horizon_steps=None,
        output_dir=None,
        training_seed=None,
        simulation_app=None,
    ):
        self.checkpoint = Path(checkpoint)
        self.env_config = Path(env_config)
        self.train_config = Path(train_config)
        self.policy_id = checkpoint_identity(self.checkpoint)
        self.config_hashes = {
            "env_config": checkpoint_identity(self.env_config),
            "train_config": checkpoint_identity(self.train_config),
        }
        if not math.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("horizon_s must be positive and finite")
        if horizon_steps is not None and (type(horizon_steps) is not int or horizon_steps < 1):
            raise ValueError("horizon_steps must be a positive integer")
        self.device, self.horizon_s, self.horizon_steps = device, horizon_s, horizon_steps
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.training_seed = training_seed
        self.app, self._owns_app = simulation_app, simulation_app is None
        self.env = self.runner = self.policy = None
        self.backend_id = "phoenix:" + content_hash(
            {
                "checkpoint": self.policy_id,
                **self.config_hashes,
                "horizon_s": horizon_s,
                "horizon_steps": horizon_steps,
                "command_strategy": "hold_seed_command",
                "material_scope": "robot_shapes",
            }
        )
        self.last_reset = None
        self.last_outcome = None
        self.last_telemetry = []
        self.environment_metadata = {}

    def _verify_policy(self, policy_id):
        if policy_id != self.policy_id or checkpoint_identity(self.checkpoint) != self.policy_id:
            raise ValueError("policy_id must match the unchanged checkpoint SHA256")
        for name, path in (("env_config", self.env_config), ("train_config", self.train_config)):
            if checkpoint_identity(path) != self.config_hashes[name]:
                raise ValueError("backend configuration changed after construction")

    def _start_episode(self, scenario_seed):
        """Real simulator construction; tests replace only this external boundary."""
        if self.env is not None:
            self.env.close()
            self.env = self.runner = self.policy = None
        if self.app is None:
            from isaaclab.app import AppLauncher

            self.app = AppLauncher(headless=True).app
        import importlib.metadata as metadata

        import gymnasium as gym
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from omegaconf import OmegaConf
        from phoenix.sim_env import build_env_cfg, load_layered_config
        from phoenix.training.agent_cfg import build_runner_cfg
        from phoenix.training.checkpoint import load_runner_checkpoint
        from rsl_rl.runners import OnPolicyRunner

        loaded = load_layered_config(self.env_config)
        effective = loaded.to_container()
        cfg = build_env_cfg(loaded)
        cfg.scene.num_envs = 1
        cfg.seed = scenario_seed
        cfg.sim.device = self.device
        # Keep the horizon explicit. An early configured time limit is rejected
        # below rather than incorrectly counted as full-horizon success.
        task = effective["env"]["task_name"]
        self.env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=1.0)
        training = OmegaConf.to_container(OmegaConf.load(self.train_config), resolve=True)
        training["run"]["device"] = self.device
        training["run"]["seed"] = scenario_seed
        runner_cfg = build_runner_cfg(training, task)
        runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
        self.runner = OnPolicyRunner(
            self.env, runner_cfg.to_dict(), log_dir=None, device=self.device
        )
        evidence = load_runner_checkpoint(
            self.runner,
            self.checkpoint,
            load_actor=True,
            load_critic=True,
            load_optimizer=False,
            load_iteration=False,
        )
        if not evidence.get("actor_match", False):
            raise RuntimeError("checkpoint actor did not round-trip into inference model")
        self.policy = self.runner.get_inference_policy(device=self.device)
        self.environment_metadata = {
            "effective_env_config_hash": content_hash(effective),
            "rsl_rl_version": metadata.version("rsl-rl-lib"),
            "simulator": "Isaac Lab",
            "material_scope": "robot_shapes",
            "known_physical_context_reconstructed": False,
            "actuator_internal_state_restored": False,
            "action_history_restored": False,
            "command_strategy": "hold_seed_command",
        }
        self.env.seed(scenario_seed)
        self.env.reset()

    def _prepare_seed(self, capsule, row, parameters, evaluation_seed):
        import torch
        from phoenix.adaptation.scenario_bridge import FrictionScenarioAdapter
        from phoenix.replay.state_adapter import restore_state
        from phoenix.replay.trajectory_reader import InitialState

        if not capsule.pre_failure_start_index <= row <= capsule.failure_onset_index:
            raise ValueError("reset row is outside reviewed pre-onset interval")
        frame = capsule.frames[row]
        if frame.missing_reset_fields:
            raise ValueError(f"capsule missing reset state: {frame.missing_reset_fields}")
        if len(frame.joint_pos) != 12:
            raise ValueError("Phoenix GO2 backend requires twelve joints")
        target = self.env.unwrapped
        dt = float(target.step_dt)
        if not math.isclose(dt, capsule.control_dt, rel_tol=0, abs_tol=1e-8):
            raise ValueError("capsule and simulator control periods differ")
        state = InitialState(
            **{
                name: np.asarray(getattr(frame, name), dtype=np.float32)
                for name in InitialState.__dataclass_fields__
            }
        )
        applied = FrictionScenarioAdapter(target).apply(0, parameters)
        restored = restore_state(target, state, 0)
        self.last_reset = {
            "capsule_id": capsule.capsule_id,
            "requested_seed_row": row,
            "resolved_row": row,
            "failure_onset_row": capsule.failure_onset_index,
            "time_before_onset_seconds": capsule.frames[capsule.failure_onset_index].timestamp_s
            - frame.timestamp_s,
            **applied,
            **restored,
        }
        # Seed control-time randomization independently after startup/reset draws.
        self.env.seed(evaluation_seed)
        target.scene.write_data_to_sim()
        target.sim.forward()
        target.scene.update(dt=0.0)
        if hasattr(self.policy, "reset"):
            self.policy.reset(torch.ones(1, dtype=torch.bool, device=target.device))
        return dt

    def _episode(self, capsule, candidate, policy_id, evaluation_seed, *, scenario=None):
        self._verify_policy(policy_id)
        if candidate.capsule_id != capsule.capsule_id:
            raise ValueError("candidate belongs to a different capsule")
        if type(evaluation_seed) is not int or evaluation_seed < 0:
            raise ValueError("evaluation_seed must be a nonnegative integer")
        if type(candidate.seed_row) is not int or not (
            capsule.pre_failure_start_index <= candidate.seed_row <= capsule.failure_onset_index
        ):
            raise ValueError("candidate row is outside reviewed pre-onset interval")
        parameters = dict(candidate.parameters)
        if len(parameters) != len(candidate.parameters):
            raise ValueError("duplicate candidate parameter")
        supported = {"static_friction", "dynamic_friction"}
        if set(parameters) - supported:
            raise ValueError(
                f"unsupported simulator dimensions: {sorted(set(parameters) - supported)}"
            )
        if any(not math.isfinite(value) or value < 0 for value in parameters.values()):
            raise ValueError("friction must be finite and nonnegative")
        if parameters.get("dynamic_friction", 0) > parameters.get("static_friction", math.inf):
            raise ValueError("dynamic friction exceeds static friction")
        scenario_seed = scenario.scenario_seed if scenario is not None else evaluation_seed
        if type(scenario_seed) is not int or scenario_seed < 0:
            raise ValueError("scenario_seed must be a nonnegative integer")
        self._start_episode(scenario_seed)
        dt = self._prepare_seed(capsule, candidate.seed_row, parameters, evaluation_seed)
        steps = self.horizon_steps or math.ceil(self.horizon_s / dt)
        target = self.env.unwrapped
        if int(target.max_episode_length) < steps:
            raise ValueError("environment time limit is shorter than registered evaluation horizon")

        import torch
        from phoenix.training.episode_outcomes import (
            EpisodeOutcome,
            PreResetCapture,
            snapshot_manager_state,
        )
        from phoenix.training.episode_telemetry import quat_wxyz_to_euler

        analyzer = FailureAnalyzer()

        def snapshot():
            return snapshot_manager_state(target, _numpy)

        capture = PreResetCapture(target, snapshot)
        events, errors, angular_errors, attitudes, contacts, commands = [], [], [], [], [], []
        self.last_telemetry = []
        total_return = 0.0
        reason = "evaluation_horizon"
        terminated = False
        observations = self.env.get_observations()
        try:
            with torch.inference_mode():
                for index in range(steps):
                    applied_command = _numpy(target.command_manager.get_command("base_velocity"))[
                        0
                    ].copy()
                    capture.begin_step()
                    action = self.policy(_policy_observations(self.policy, observations))
                    observations, reward, done, extras = self.env.step(action)
                    done_flag = bool(_numpy(done)[0])
                    state = capture.overlay(snapshot())
                    if done_flag and 0 not in capture.terminal:
                        raise RuntimeError("terminal environment missing pre-reset snapshot")
                    roll, pitch, _ = quat_wxyz_to_euler(state["quaternion"][0])
                    contact = state["contacts"][0]
                    signal = dict(
                        timestamp_s=(index + 1) * dt,
                        pitch_rad=pitch,
                        roll_rad=roll,
                        base_height_m=float(state["position"][0, 2]),
                        cmd_lin_vel=applied_command[:2],
                        actual_lin_vel=state["linear"][0, :2],
                        joint_vel=state["joint_velocity"][0],
                        contact_forces=contact if np.isfinite(contact).all() else None,
                    )
                    final = done_flag or index == steps - 1
                    emitted = analyzer.step(done=final, **signal)
                    events.extend(
                        {
                            "mode": event.mode.value,
                            "timestamp_s": event.timestamp_s,
                            "detail": event.detail,
                            "detector": "ashfall.threshold.v1",
                        }
                        for event in emitted
                    )
                    error = float(np.linalg.norm(applied_command[:2] - state["linear"][0, :2]))
                    errors.append(error)
                    angular_errors.append(abs(float(applied_command[2] - state["angular"][0, 2])))
                    attitudes.append(roll * roll + pitch * pitch)
                    contacts.append(contact)
                    commands.append(applied_command)
                    total_return += float(_numpy(reward)[0])
                    self.last_telemetry.append(
                        {
                            "timestamp_s": (index + 1) * dt,
                            "roll_rad": roll,
                            "pitch_rad": pitch,
                            "base_height_m": signal["base_height_m"],
                            "tracking_error_mps": error,
                            "command": applied_command.tolist(),
                            "terminal": done_flag,
                        }
                    )
                    if done_flag:
                        reasons = [
                            name.split(":", 1)[1]
                            for name, value in state.items()
                            if name.startswith("termination:") and bool(value[0])
                        ]
                        if not reasons:
                            raise RuntimeError(
                                "terminal episode has no captured termination reason"
                            )
                        # Use captured term config to distinguish truncation from physical failure.
                        failed_terms = [
                            name
                            for name in reasons
                            if not target.termination_manager.get_term_cfg(name).time_out
                        ]
                        terminated = bool(failed_terms)
                        reason = "|".join(reasons)
                        if not terminated and index + 1 < steps:
                            raise RuntimeError(
                                "environment truncated before the registered horizon"
                            )
                        break
        finally:
            capture.close()
        metrics = analyzer.compute()
        target_events = [event for event in events if event["mode"] == capsule.failure_mode]
        # All observed modes are retained in outcomes. Reproduction describes the
        # requested family if present, otherwise the earliest observed other mode.
        primary = target_events[0] if target_events else (events[0] if events else None)
        descriptor = FailureDescriptor(
            mode=primary["mode"] if primary else f"termination:{reason}" if terminated else None,
            time_to_failure_s=primary["timestamp_s"]
            if primary
            else len(errors) * dt
            if terminated
            else None,
            termination_reason=reason,
            attitude_rms_rad=float(np.sqrt(np.mean(attitudes))),
            velocity_error_mps=float(np.sqrt(np.mean(np.square(errors)))),
            contact_signature=tuple(np.mean(np.asarray(contacts) > 5.0, axis=0))
            if np.isfinite(contacts).all()
            else None,
        )
        outcome = EpisodeOutcome(
            policy_id=policy_id,
            evaluation_seed=evaluation_seed,
            episode_id=0,
            success=not terminated and not events,
            termination_reason=reason,
            episode_length_steps=len(errors),
            control_dt_s=dt,
            episode_return=total_return,
            scenario_id=scenario.scenario_id if scenario is not None else None,
            scenario_seed=scenario_seed if scenario is not None else None,
            parameter_sample_id=scenario.parameter_sample_id if scenario is not None else None,
            training_seed=self.training_seed,
            command=np.mean(commands, axis=0).tolist(),
            tracking_error=float(np.sqrt(np.mean(np.square(errors)))),
            angular_tracking_error=float(np.mean(angular_errors)),
            failure_events=events,
            failure_modes=sorted({event["mode"] for event in events}),
            failure_onset_s=min((e["timestamp_s"] for e in events), default=None),
            intervention_required=bool(metrics.intervention_count),
            intervention_criterion="detected_attitude_or_collapse",
            recovery_outcome="unrecovered"
            if metrics.unrecovered_events
            else "recovered"
            if metrics.recovered_events
            else "no_detected_failure",
            recovery_time_s=metrics.mean_recovery_time_s,
            environment_parameters={
                **parameters,
                **self.environment_metadata,
                "reset_state_id": content_hash(
                    {"capsule_id": capsule.capsule_id, "seed_row": candidate.seed_row}
                ),
                "success_criterion": "full_horizon_without_termination_or_detected_event",
            },
        )
        self.last_outcome = outcome
        if self.output_dir is not None:
            identity = content_hash(
                {
                    "candidate": candidate.candidate_id,
                    "policy": policy_id,
                    "scenario": scenario.scenario_id if scenario else None,
                    "evaluation_seed": evaluation_seed,
                    "backend": self.backend_id,
                }
            )
            write_artifact(
                self.output_dir / f"episode_{identity}.json",
                {
                    "evidence_kind": self.evidence_kind,
                    "backend_id": self.backend_id,
                    "outcome": outcome.to_dict(),
                    "descriptor": asdict(descriptor),
                    "reset_telemetry": self.last_reset,
                    "telemetry": self.last_telemetry,
                },
            )
        return descriptor, outcome

    def replay(self, capsule, candidate, policy_id, seed):
        if capsule.policy_id != policy_id:
            raise ValueError("reproduction requires the capsule baseline checkpoint identity")
        return self._episode(capsule, candidate, policy_id, seed)[0]

    def evaluate_scenario(self, capsule, scenario, policy_id, evaluation_seed):
        row = resolve_scenario_row(capsule, scenario)
        candidate = ReproductionCandidate(capsule.capsule_id, scenario.parameters, row)
        return self._episode(capsule, candidate, policy_id, evaluation_seed, scenario=scenario)[1]

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = self.runner = self.policy = None
        if self.app is not None and self._owns_app:
            self.app.close()
            self.app = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_backend(config):
    """CLI plugin factory. Config paths resolve in the invoking working directory."""
    return PhoenixBackend(**config)
