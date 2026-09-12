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
import inspect
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ashfall.backends.phoenix_compat import KINEMATIC_RESTORE_FIELDS
from ashfall.counterfactual import RestorableState, RolloutTrace
from ashfall.evaluation.metrics import FailureAnalyzer
from ashfall.gates import MatchedPairEvidence
from ashfall.ontology import Intervention
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
        # Only the kinematic restore channels come from the frame. InitialState
        # also carries metadata (position frame, controller history, declared
        # environment parameters); iterating every dataclass field broke the
        # moment Phoenix added those, which is exactly the drift
        # phoenix_compat pins. Ashfall capsules are env-local captures.
        state = InitialState(
            **{
                name: np.asarray(getattr(frame, name), dtype=np.float32)
                for name in KINEMATIC_RESTORE_FIELDS
            },
            position_frame="env_local",
            position_frame_source="ashfall_capsule_env_local",
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
            time_to_failure_s=(
                primary["timestamp_s"] if primary else len(errors) * dt if terminated else None
            ),
            termination_reason=reason,
            attitude_rms_rad=float(np.sqrt(np.mean(attitudes))),
            velocity_error_mps=float(np.sqrt(np.mean(np.square(errors)))),
            contact_signature=(
                tuple(np.mean(np.asarray(contacts) > 5.0, axis=0))
                if np.isfinite(contacts).all()
                else None
            ),
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
            recovery_outcome=(
                "unrecovered"
                if metrics.unrecovered_events
                else "recovered" if metrics.recovered_events else "no_detected_failure"
            ),
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

    # ------------------------------------------------------------------ #
    # Matched counterfactual pairs (H0) on a persistent environment
    # ------------------------------------------------------------------ #

    def _ensure_started(self, scene_seed):
        """Create the simulator once and keep it; matched arms must share one scene.

        Startup-mode domain randomisation (mass, material, motor strength) is
        drawn when the scene is created, so control and treatment share it by
        construction when they run in the same environment instance.
        """
        if self.env is None:
            self._start_episode(scene_seed)
            self.scene_seed = scene_seed
        return self.env.unwrapped

    def _restore(self, state: RestorableState, env_id: int = 0) -> dict:
        from phoenix.replay.state_adapter import restore_state
        from phoenix.replay.trajectory_reader import InitialState

        target = self.env.unwrapped
        initial = InitialState(
            **{
                name: np.asarray(getattr(state, name), dtype=np.float32)
                for name in KINEMATIC_RESTORE_FIELDS
            },
            position_frame="env_local",
            position_frame_source="ashfall_restorable_state_env_local",
        )
        restored = restore_state(target, initial, env_id)
        target.scene.write_data_to_sim()
        target.sim.forward()
        target.scene.update(dt=0.0)
        if hasattr(self.policy, "reset"):
            import torch

            self.policy.reset(torch.ones(1, dtype=torch.bool, device=target.device))
        return restored

    def _rollout(self, horizon_steps: int, *, on_step=None) -> RolloutTrace:
        """Run the policy from the current simulator state and record restorable channels."""
        import torch
        from phoenix.training.episode_outcomes import PreResetCapture, snapshot_manager_state

        target = self.env.unwrapped
        dt = float(target.step_dt)
        if int(target.max_episode_length) < horizon_steps:
            raise ValueError("environment time limit is shorter than the requested horizon")
        rows = {
            name: []
            for name in (
                "base_pos",
                "base_quat",
                "base_lin_vel_body",
                "base_ang_vel_body",
                "joint_pos",
                "joint_vel",
                "command_vel",
                "contact_forces",
                "actions",
            )
        }

        def snapshot():
            state = snapshot_manager_state(target, _numpy)
            state["joint_position"] = _numpy(target.scene["robot"].data.joint_pos).copy()
            return state

        capture = PreResetCapture(target, snapshot)
        termination = "evaluation_horizon"
        terminated_step = None
        observations = self.env.get_observations()
        try:
            with torch.inference_mode():
                for step in range(horizon_steps):
                    if on_step is not None:
                        on_step(step, dt)
                    capture.begin_step()
                    action = self.policy(_policy_observations(self.policy, observations))
                    observations, _reward, done, _extras = self.env.step(action)
                    state = capture.overlay(snapshot())
                    quat_wxyz = state["quaternion"][0]
                    rows["base_pos"].append(state["position"][0].tolist())
                    rows["base_quat"].append([*quat_wxyz[1:], quat_wxyz[0]])
                    rows["base_lin_vel_body"].append(state["linear"][0].tolist())
                    rows["base_ang_vel_body"].append(state["angular"][0].tolist())
                    rows["joint_pos"].append(state["joint_position"][0].tolist())
                    rows["joint_vel"].append(state["joint_velocity"][0].tolist())
                    rows["command_vel"].append(state["command"][0][:3].tolist())
                    contact = state["contacts"][0]
                    rows["contact_forces"].append(
                        contact.tolist() if np.isfinite(contact).all() else [np.nan] * 4
                    )
                    rows["actions"].append(_numpy(action)[0].tolist())
                    if bool(_numpy(done)[0]):
                        reasons = [
                            name.split(":", 1)[1]
                            for name, value in state.items()
                            if name.startswith("termination:") and bool(value[0])
                        ]
                        failed = [
                            r
                            for r in reasons
                            if not target.termination_manager.get_term_cfg(r).time_out
                        ]
                        if failed:
                            termination = "|".join(reasons)
                            terminated_step = step
                            break
                        if step + 1 < horizon_steps:
                            raise RuntimeError("environment truncated before the requested horizon")
        finally:
            capture.close()
        contacts = np.array(rows["contact_forces"])
        return RolloutTrace(
            dt,
            np.array(rows["base_pos"]),
            np.array(rows["base_quat"]),
            np.array(rows["base_lin_vel_body"]),
            np.array(rows["base_ang_vel_body"]),
            np.array(rows["joint_pos"]),
            np.array(rows["joint_vel"]),
            np.array(rows["command_vel"]),
            termination,
            terminated_step,
            contact_forces=None if np.isnan(contacts).any() else contacts,
            actions=np.array(rows["actions"]),
            metadata={
                "backend_id": self.backend_id,
                "env_config_hash": self.config_hashes["env_config"],
                "scene_seed": getattr(self, "scene_seed", None),
            },
        )

    def _arm(self, state, intervention, *, seed, horizon_steps):
        """One arm of a matched pair: seed, reset, restore, apply, roll out, release."""
        from ashfall.backends.phoenix_interventions import PhoenixInterventionAdapter

        target = self._ensure_started(seed)
        self.env.seed(seed)
        self.env.reset()
        self._restore(state, 0)
        adapter = PhoenixInterventionAdapter(target, env_id=0)
        receipt = adapter.apply(intervention)
        completed = {"receipt": receipt}

        def on_step(step, dt):
            if adapter.pending_push is not None:
                done = adapter.apply_pending_push(step, dt, intervention)
                if done is not None:
                    completed["receipt"] = done

        try:
            trace = self._rollout(horizon_steps, on_step=on_step)
        finally:
            adapter.release()
        return trace, completed["receipt"]

    def matched_pair(
        self,
        state: RestorableState,
        intervention: Intervention,
        *,
        simulator_seed: int,
        replicate_seeds,
        horizon_steps: int,
    ) -> MatchedPairEvidence:
        """Control, treatment and nominal replicates from one restored state in one scene."""
        if type(simulator_seed) is not int or simulator_seed < 0:
            raise ValueError("simulator_seed must be a nonnegative integer")
        control, control_receipt = self._arm(
            state, Intervention.none(), seed=simulator_seed, horizon_steps=horizon_steps
        )
        treatment, treatment_receipt = self._arm(
            state, intervention, seed=simulator_seed, horizon_steps=horizon_steps
        )
        replicates = [
            self._arm(state, Intervention.none(), seed=int(s), horizon_steps=horizon_steps)[0]
            for s in replicate_seeds
        ]
        return MatchedPairEvidence(
            treatment,
            control,
            tuple(replicates),
            treatment_receipt,
            control_receipt,
            simulator_seed,
            tuple(int(s) for s in replicate_seeds),
        )

    def nominal_rollout(self, *, seed: int, horizon_steps: int, command) -> RolloutTrace:
        """A no-intervention rollout from the environment's own reset, holding ``command``."""
        from phoenix.replay.state_adapter import VelocityCommandAdapter

        target = self._ensure_started(seed)
        self.env.seed(seed)
        self.env.reset()
        adapter = VelocityCommandAdapter(target)
        import torch

        ids = torch.as_tensor([0], dtype=torch.long, device=adapter.term.vel_command_b.device)
        adapter.restore(ids, [float(v) for v in command], hold_seconds=None)
        return self._rollout(horizon_steps)

    def provenance(self) -> dict:
        import importlib.metadata as metadata

        try:
            version = metadata.version("isaaclab")
        except metadata.PackageNotFoundError:
            version = None
        return {
            "simulator_version": version,
            "env_config_hash": self.config_hashes["env_config"],
            "train_config_hash": self.config_hashes["train_config"],
            "backend_id": self.backend_id,
            "evidence_kind": self.evidence_kind,
            "checkpoint_sha256": self.policy_id,
            "scene_seed": getattr(self, "scene_seed", None),
            **self.environment_metadata,
        }

    def evaluate(
        self, manifest, capsules, *, policy_id, evaluation_seeds, training_seed=None, split
    ):
        """Episode outcomes for every scenario of ``split`` and every evaluation seed.

        This is the producer ``ashfall evaluate`` calls (audit finding A1). The
        training seed is recorded on every outcome and never inferred.
        """
        if not evaluation_seeds or len(set(evaluation_seeds)) != len(evaluation_seeds):
            raise ValueError("distinct evaluation seeds required")
        previous = self.training_seed
        self.training_seed = training_seed
        records = []
        try:
            for scenario in manifest.select(split):
                capsule = capsules[scenario.capsule_id]
                for seed in evaluation_seeds:
                    records.append(self.evaluate_scenario(capsule, scenario, policy_id, int(seed)))
        finally:
            self.training_seed = previous
        if not records:
            raise ValueError(
                f"no scenarios in split {split!r}; an empty evaluation is not evidence"
            )
        return records

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
    """CLI plugin factory. Config paths resolve in the invoking working directory.

    Only keys ``PhoenixBackend.__init__`` accepts are passed through. Provenance
    keys such as ``simulator_version`` that the same JSON legitimately carries
    for the training path are kept on ``backend.extra_config`` instead of
    raising ``TypeError`` (audit finding A3).
    """
    accepted = set(inspect.signature(PhoenixBackend.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in config.items() if k in accepted}
    backend = PhoenixBackend(**kwargs)
    backend.extra_config = {k: v for k, v in config.items() if k not in accepted}
    return backend
