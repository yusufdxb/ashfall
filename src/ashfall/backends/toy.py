"""A deterministic planar quadruped surrogate for exercising the H0 pipeline on the CPU.

This is a MOCK backend. Its evidence kind is ``mock``, its datasets are
fixtures, and nothing it produces may enter a scientific claim. It exists so
the matched-pair, gate and harvest machinery can be tested end to end without
Isaac Lab, with interventions that have a causal, physically motivated effect
in the surrogate:

* friction reduction lowers the traction cone, so a commanded speed the feet
  cannot transmit makes them slide and the body speed decays toward zero;
* actuator weakening caps the achievable speed and lowers the stance height;
* payload mass compresses the stance height in proportion to the added mass;
* an external perturbation is a velocity impulse with a pitch kick;
* command corruption replaces the applied command with the requested one;
* terrain geometry is unsupported, and says so through the receipt.

Noise is drawn from the simulator seed, so a control and a treatment rollout
under the same seed share their noise sequence exactly, and nominal replicates
under other seeds do not. The surrogate has no claim to GO2 fidelity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ashfall.counterfactual import RestorableState, RolloutTrace
from ashfall.gates import InterventionReceipt, MatchedPairEvidence
from ashfall.ontology import Intervention
from ashfall.provenance import content_hash

GRAVITY = 9.81
N_JOINTS = 12
_LEG_OFFSETS = np.array([0.0, math.pi, math.pi, 0.0])
_NOMINAL_JOINTS = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])


@dataclass(frozen=True)
class ToyParameters:
    mass_kg: float = 15.0
    nominal_static_friction: float = 0.8
    nominal_dynamic_friction: float = 0.7
    speed_gain: float = 5.0
    speed_cap_mps: float = 1.2
    stance_height_m: float = 0.30
    height_time_constant_s: float = 0.3
    sag_per_kg_m: float = 0.04
    sag_per_unit_weakening_m: float = 0.10
    pitch_natural_frequency: float = 6.0
    pitch_damping_ratio: float = 0.5
    pitch_kick_per_mps: float = 10.0
    slip_decay_per_s: float = 6.0
    gait_drag_per_s: float = 2.0
    noise_speed: float = 0.01
    noise_height: float = 0.002
    noise_pitch: float = 0.005
    control_dt: float = 0.02
    belly_contact_height_m: float = 0.08
    fall_pitch_rad: float = 1.3


class ToyBackend:
    """See the module docstring. Mock evidence only."""

    evidence_kind = "mock"
    simulator_version = "toy-quadruped-surrogate-1.0"
    policy_id = "toypolicy-" + content_hash("toy-policy-v1")[:56]
    supported_interventions = (
        "none",
        "friction_reduction",
        "actuator_weakening",
        "external_perturbation",
        "command_corruption",
        "payload_mass",
    )

    def __init__(self, parameters: ToyParameters | None = None):
        self.parameters = parameters or ToyParameters()
        self.backend_id = "toy:" + content_hash(self.parameters.__dict__)
        self.env_config_hash = content_hash({"toy": self.parameters.__dict__})

    # ------------------------------------------------------------------ #
    # Interventions
    # ------------------------------------------------------------------ #

    def _apply(self, intervention: Intervention) -> tuple[dict, InterventionReceipt]:
        p = self.parameters
        applied = {
            "static_friction": p.nominal_static_friction,
            "dynamic_friction": p.nominal_dynamic_friction,
            "gain": 1.0,
            "mass_extra": 0.0,
            "push": None,
            "command_override": None,
        }
        params = dict(intervention.parameters)
        if intervention.kind not in self.supported_interventions:
            return applied, InterventionReceipt.unsupported(
                intervention, f"toy surrogate has no {intervention.kind} manipulation"
            )
        if intervention.kind == "friction_reduction":
            applied["static_friction"] = params["static_friction"]
            applied["dynamic_friction"] = params["dynamic_friction"]
            readback = {k: applied[k] for k in params}
        elif intervention.kind == "actuator_weakening":
            applied["gain"] = params.get("stiffness_scale", params.get("effort_scale", 1.0))
            readback = dict(params)
        elif intervention.kind == "payload_mass":
            applied["mass_extra"] = params["mass_offset_kg"]
            readback = dict(params)
        elif intervention.kind == "external_perturbation":
            applied["push"] = (
                params.get("push_vx_mps", 0.0),
                params.get("push_vy_mps", 0.0),
                params.get("push_time_s", 0.5),
            )
            readback = dict(params)
        elif intervention.kind == "command_corruption":
            applied["command_override"] = params
            readback = dict(params)
        else:  # control
            readback = {}
        receipt = InterventionReceipt(
            intervention,
            dict(params),
            readback,
            "toy_parameter_readback",
            baseline={
                "static_friction": p.nominal_static_friction,
                "dynamic_friction": p.nominal_dynamic_friction,
            },
        )
        return applied, receipt

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #

    def spawn_state(self, command: Sequence[float]) -> RestorableState:
        p = self.parameters
        return RestorableState(
            (0.0, 0.0, p.stance_height_m),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            tuple(_NOMINAL_JOINTS),
            (0.0,) * N_JOINTS,
            tuple(float(v) for v in command),
        )

    def rollout(
        self,
        state: RestorableState,
        intervention: Intervention,
        *,
        seed: int,
        horizon_steps: int,
    ) -> tuple[RolloutTrace, InterventionReceipt]:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if horizon_steps < 2:
            raise ValueError("horizon_steps must be at least 2")
        p = self.parameters
        applied, receipt = self._apply(intervention)
        rng = np.random.default_rng(seed)
        dt = p.control_dt
        x, y, z = state.base_pos
        vx, vy, vz = state.base_lin_vel_body
        roll, pitch = _roll_pitch(state.base_quat)
        yaw = 0.0
        pitch_rate = float(state.base_ang_vel_body[1])
        yaw_rate = float(state.base_ang_vel_body[2])
        command = np.array(state.command_vel, dtype=float)
        applied_command = command.copy()
        if applied["command_override"] is not None:
            over = applied["command_override"]
            applied_command = np.array(
                [
                    over.get("command_vx_mps", command[0]),
                    over.get("command_vy_mps", command[1]),
                    over.get("command_yaw_radps", command[2]),
                ]
            )
        phase = 0.0
        gain = float(applied["gain"])
        mass_extra = float(applied["mass_extra"])
        mu = float(applied["dynamic_friction"])
        z_target = (
            p.stance_height_m
            - p.sag_per_kg_m * mass_extra
            - p.sag_per_unit_weakening_m * (1 - gain)
        )
        push = applied["push"]
        push_step = None if push is None else int(round(push[2] / dt))
        omega, zeta = p.pitch_natural_frequency, p.pitch_damping_ratio
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
            )
        }
        termination = "evaluation_horizon"
        terminated_step = None
        for step in range(horizon_steps):
            # propulsion under a traction cone
            target = (
                min(applied_command[0], p.speed_cap_mps * gain)
                if applied_command[0] >= 0
                else max(applied_command[0], -p.speed_cap_mps * gain)
            )
            a_des = p.speed_gain * (target - vx)
            # Traction demand: acceleration plus the tangential load of
            # carrying the gait at speed (impact and drag), pulsing with phase.
            demand = abs(a_des) + p.gait_drag_per_s * abs(vx) * (1 + 0.5 * abs(math.sin(phase)))
            a_max = mu * GRAVITY
            slipping = demand > a_max
            if slipping:
                vx += (-p.slip_decay_per_s * vx + 0.2 * a_max * np.sign(a_des)) * dt
            else:
                vx += a_des * dt
            vy += p.speed_gain * (applied_command[1] - vy) * dt
            yaw_rate += 4.0 * (applied_command[2] - yaw_rate) * dt
            if push_step is not None and step == push_step:
                vx += push[0]
                vy += push[1]
                pitch_rate += p.pitch_kick_per_mps * abs(push[0])
            vx += rng.normal(0.0, p.noise_speed)
            # height relaxes toward its target
            z_prev = z
            z += (z_target - z) * dt / p.height_time_constant_s + rng.normal(0.0, p.noise_height)
            vz = (z - z_prev) / dt
            # pitch as a damped oscillator
            pitch_acc = -(omega**2) * pitch - 2 * zeta * omega * pitch_rate
            pitch_rate += pitch_acc * dt
            pitch += pitch_rate * dt + rng.normal(0.0, p.noise_pitch)
            yaw += yaw_rate * dt
            x += vx * math.cos(yaw) * dt
            y += vx * math.sin(yaw) * dt
            # gait
            rate = 2 * math.pi * (1.5 + abs(vx)) * (1.6 if slipping else 1.0)
            phase += rate * dt
            legs = 0.3 * np.sin(phase + _LEG_OFFSETS)
            leg_vel = 0.3 * np.cos(phase + _LEG_OFFSETS) * rate
            joint_pos = _NOMINAL_JOINTS + np.repeat(legs, 3) * np.array([0.2, 1.0, 1.0] * 4)
            joint_vel = np.repeat(leg_vel, 3) * np.array([0.2, 1.0, 1.0] * 4)
            support = max(0.0, min(1.0, (z - p.belly_contact_height_m) / 0.1))
            contact = (
                (p.mass_kg + mass_extra) * GRAVITY / 4 * (1 + 0.3 * np.sin(phase + _LEG_OFFSETS))
            )
            contact = np.clip(contact * support, 0.0, None)
            quat = _quat_from_roll_pitch_yaw(roll, pitch, yaw)
            rows["base_pos"].append([x, y, z])
            rows["base_quat"].append(list(quat))
            rows["base_lin_vel_body"].append([vx, vy, vz])
            rows["base_ang_vel_body"].append([0.0, pitch_rate, yaw_rate])
            rows["joint_pos"].append(joint_pos.tolist())
            rows["joint_vel"].append(joint_vel.tolist())
            rows["command_vel"].append(applied_command.tolist())
            rows["contact_forces"].append(contact.tolist())
            if z < p.belly_contact_height_m or abs(pitch) > p.fall_pitch_rad:
                termination = "base_contact"
                terminated_step = step
                break
        if len(rows["base_pos"]) < 2:
            raise RuntimeError("surrogate terminated on the first step; no trace to return")
        trace = RolloutTrace(
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
            contact_forces=np.array(rows["contact_forces"]),
            metadata={
                "backend_id": self.backend_id,
                "seed": seed,
                "intervention_id": intervention.intervention_id,
            },
        )
        return trace, receipt

    # ------------------------------------------------------------------ #
    # Backend protocol used by the H0 runner
    # ------------------------------------------------------------------ #

    def nominal_rollout(
        self, *, seed: int, horizon_steps: int, command: Sequence[float]
    ) -> RolloutTrace:
        trace, _ = self.rollout(
            self.spawn_state(command), Intervention.none(), seed=seed, horizon_steps=horizon_steps
        )
        return trace

    def matched_pair(
        self,
        state: RestorableState,
        intervention: Intervention,
        *,
        simulator_seed: int,
        replicate_seeds: Sequence[int],
        horizon_steps: int,
    ) -> MatchedPairEvidence:
        control, control_receipt = self.rollout(
            state, Intervention.none(), seed=simulator_seed, horizon_steps=horizon_steps
        )
        treatment, treatment_receipt = self.rollout(
            state, intervention, seed=simulator_seed, horizon_steps=horizon_steps
        )
        replicates = [
            self.rollout(state, Intervention.none(), seed=s, horizon_steps=horizon_steps)[0]
            for s in replicate_seeds
        ]
        return MatchedPairEvidence(
            treatment,
            control,
            tuple(replicates),
            treatment_receipt,
            control_receipt,
            simulator_seed,
            tuple(replicate_seeds),
        )

    def provenance(self) -> dict:
        return {
            "simulator_version": self.simulator_version,
            "env_config_hash": self.env_config_hash,
            "backend_id": self.backend_id,
            "evidence_kind": self.evidence_kind,
        }

    def close(self) -> None:
        return None


def _roll_pitch(quat_xyzw) -> tuple[float, float]:
    x, y, z, w = (float(v) for v in quat_xyzw)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def _quat_from_roll_pitch_yaw(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
    norm = math.sqrt(sum(v * v for v in q))
    return tuple(v / norm for v in q)  # type: ignore[return-value]


__all__ = ["ToyBackend", "ToyParameters"]
