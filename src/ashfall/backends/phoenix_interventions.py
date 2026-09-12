"""Intervention adapters for the Phoenix / Isaac Lab backend, each with readback.

Every adapter returns an :class:`ashfall.gates.InterventionReceipt` whose
``readback`` is read from the simulator after the write, so gate D1 judges what
the simulator holds rather than what was requested. An intervention kind the
backend cannot write returns an unsupported receipt instead of a silent no-op.

The adapters are written against the Isaac Lab articulation and event
interfaces Phoenix already relies on, and are exercised on the CPU against
fakes in ``tests/test_phoenix_backend.py``. Their behaviour on the real
simulator is UNVERIFIED until the H0 smoke in ``docs/runbooks/h0_isaac.md`` has
been run and its bundle committed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ashfall.gates import InterventionReceipt
from ashfall.ontology import Intervention

SUPPORTED_KINDS = (
    "none",
    "friction_reduction",
    "command_corruption",
    "external_perturbation",
    "actuator_weakening",
    "payload_mass",
)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _as_torch(value):
    """A torch view of a torch tensor or a Warp array (shared memory for device arrays)."""
    if hasattr(value, "detach"):
        return value
    import warp as wp

    return wp.to_torch(value)


class MaterialWriter:
    """Per-environment robot-shape material coefficients with readback.

    Mirrors ``phoenix.adaptation.scenario_bridge.FrictionScenarioAdapter`` but
    decides torch versus Warp from the DATA the view returns rather than from
    which view attribute exists: on the Isaac Lab build probed here the robot
    exposes ``root_physx_view`` and its ``get_material_properties`` still
    returns a ``wp.array``, which the Phoenix adapter's ``.clone()`` cannot
    handle. Columns are static friction, dynamic friction, restitution.
    """

    def __init__(self, env):
        robot = env.scene["robot"]
        self.view = getattr(robot, "root_physx_view", None) or getattr(robot, "root_view", None)
        if self.view is None:
            raise RuntimeError("No supported material physics view")
        self.original: dict[int, Any] = {}

    def _get(self):
        data = self.view.get_material_properties()
        return _as_torch(data).clone()

    def _set(self, values, ids):
        import torch

        indices = torch.as_tensor(ids, dtype=torch.int32, device="cpu")
        raw = self.view.get_material_properties()
        if hasattr(raw, "detach"):
            self.view.set_material_properties(values, indices)
        else:
            import warp as wp

            self.view.set_material_properties(
                wp.from_torch(values.contiguous(), dtype=wp.float32),
                wp.from_torch(indices, dtype=wp.int32),
            )

    def read(self, env_id: int) -> dict[str, float]:
        values = _to_numpy(self._get()[env_id])
        return {
            "static_friction": float(values[:, 0].mean()),
            "dynamic_friction": float(values[:, 1].mean()),
        }

    def apply(self, env_id: int, parameters: dict) -> dict[str, float]:
        import torch

        values = self._get()
        if env_id not in self.original:
            self.original[env_id] = values[env_id].clone()
        for name, column in (("static_friction", 0), ("dynamic_friction", 1)):
            if name in parameters:
                values[env_id, :, column] = float(parameters[name])
        if (values[env_id, :, 1] > values[env_id, :, 0]).any():
            raise ValueError("dynamic friction must not exceed static friction")
        self._set(values, [env_id])
        readback = self._get()[env_id]
        if not torch.allclose(readback, values[env_id], atol=1e-6, rtol=0):
            raise RuntimeError("material readback does not match requested coefficients")
        return self.read(env_id)

    def reset(self, env_ids) -> None:
        ids = [int(i) for i in env_ids if int(i) in self.original]
        if ids:
            values = self._get()
            for i in ids:
                values[i] = self.original.pop(i)
            self._set(values, ids)


class PhoenixInterventionAdapter:
    """Apply one intervention to one environment, verify it, and release it afterwards."""

    def __init__(self, env, *, env_id: int = 0):
        self.env = env
        self.env_id = int(env_id)
        self._friction = None
        self._restore: list = []
        self.pending_push: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    def _friction_adapter(self) -> MaterialWriter:
        if self._friction is None:
            self._friction = MaterialWriter(self.env)
        return self._friction

    def _robot(self):
        return self.env.scene["robot"]

    # ------------------------------------------------------------------ #
    def control_receipt(self) -> InterventionReceipt:
        """Read the nominal material coefficients back so the control arm is verified untouched."""
        baseline = self._friction_adapter().read(self.env_id)
        return InterventionReceipt(
            Intervention.none(), {}, {}, "material_readback", baseline=baseline, env_id=self.env_id
        )

    def apply(self, intervention: Intervention) -> InterventionReceipt:
        kind = intervention.kind
        params = dict(intervention.parameters)
        if kind not in SUPPORTED_KINDS:
            return InterventionReceipt.unsupported(
                intervention,
                f"the Phoenix backend has no per-environment {kind} manipulation "
                "(terrain is fixed by the Isaac Lab task and cannot be edited per env)",
            )
        if kind == "none":
            return self.control_receipt()
        if kind == "friction_reduction":
            adapter = self._friction_adapter()
            readback = adapter.apply(self.env_id, params)
            self._restore.append(lambda: adapter.reset([self.env_id]))
            return InterventionReceipt(
                intervention,
                params,
                readback,
                "material_readback",
                tolerance=1e-5,
                env_id=self.env_id,
            )
        if kind == "command_corruption":
            from phoenix.replay.state_adapter import VelocityCommandAdapter

            adapter = VelocityCommandAdapter(self.env)
            current = _to_numpy(adapter.term.command)[self.env_id]
            command = [
                params.get("command_vx_mps", float(current[0])),
                params.get("command_vy_mps", float(current[1])),
                params.get("command_yaw_radps", float(current[2])),
            ]
            import torch

            ids = torch.as_tensor(
                [self.env_id], dtype=torch.long, device=adapter.term.vel_command_b.device
            )
            written = adapter.restore(ids, command, hold_seconds=None)[0]
            readback = {}
            for name, index in (
                ("command_vx_mps", 0),
                ("command_vy_mps", 1),
                ("command_yaw_radps", 2),
            ):
                if name in params:
                    readback[name] = float(written[index])
            return InterventionReceipt(
                intervention, params, readback, "command_buffer_readback", env_id=self.env_id
            )
        if kind == "external_perturbation":
            self.pending_push = {
                "vx": params.get("push_vx_mps", 0.0),
                "vy": params.get("push_vy_mps", 0.0),
                "time_s": params.get("push_time_s", 0.5),
                "applied": None,
            }
            # The receipt is completed by apply_pending_push when the step arrives.
            return InterventionReceipt(
                intervention, params, None, "root_velocity_write_pending", env_id=self.env_id
            )
        if kind == "actuator_weakening":
            return self._scale_actuators(intervention, params)
        return self._add_mass(intervention, params)

    # ------------------------------------------------------------------ #
    def apply_pending_push(self, step: int, control_dt: float, intervention: Intervention):
        """At the scheduled step, add the push to the root velocity and read it back."""
        push = self.pending_push
        if push is None or push["applied"] is not None:
            return None
        if step != int(round(push["time_s"] / control_dt)):
            return None
        import torch

        robot = self._robot()
        velocity = _as_torch(robot.data.root_vel_w)[self.env_id].clone()
        before = velocity.clone()
        velocity[0] += push["vx"]
        velocity[1] += push["vy"]
        ids = torch.as_tensor([self.env_id], dtype=torch.long, device=velocity.device)
        robot.write_root_velocity_to_sim(velocity[None], env_ids=ids)
        after = _as_torch(robot.data.root_vel_w)[self.env_id]
        delta = _to_numpy(after - before)
        readback = {
            "push_vx_mps": float(delta[0]),
            "push_vy_mps": float(delta[1]),
            "push_time_s": step * control_dt,
        }
        push["applied"] = step
        params = dict(intervention.parameters)
        return InterventionReceipt(
            intervention,
            params,
            {k: readback[k] for k in params},
            "root_velocity_readback",
            tolerance=1e-4,
            env_id=self.env_id,
            applied_at_step=step,
        )

    def _scale_actuators(self, intervention: Intervention, params: dict) -> InterventionReceipt:
        robot = self._robot()
        actuators = getattr(robot, "actuators", None)
        if not actuators:
            return InterventionReceipt.unsupported(intervention, "robot exposes no actuator models")
        readback: dict[str, float] = {}
        for name, scale in (
            ("stiffness_scale", "stiffness"),
            ("damping_scale", "damping"),
            ("effort_scale", "effort_limit"),
        ):
            if name not in params:
                continue
            ratios = []
            for actuator in actuators.values():
                raw = getattr(actuator, scale, None)
                if raw is None:
                    continue
                tensor = _as_torch(raw)
                original = tensor[self.env_id].clone()
                tensor[self.env_id] = original * params[name]
                ratio = tensor[self.env_id] / original
                ratios.append(float(_to_numpy(ratio).mean()))
                self._restore.append(lambda t=tensor, o=original: t.__setitem__(self.env_id, o))
            if not ratios:
                return InterventionReceipt.unsupported(intervention, f"no actuator carries {scale}")
            readback[name] = float(sum(ratios) / len(ratios))
        return InterventionReceipt(
            intervention,
            params,
            readback,
            "actuator_gain_readback",
            tolerance=1e-5,
            env_id=self.env_id,
        )

    def _add_mass(self, intervention: Intervention, params: dict) -> InterventionReceipt:
        robot = self._robot()
        view = getattr(robot, "root_physx_view", None)
        if view is None or not hasattr(view, "get_masses"):
            return InterventionReceipt.unsupported(
                intervention, "no PhysX root view with mass access on this backend"
            )
        import torch

        masses = _as_torch(view.get_masses()).clone()
        original = masses[self.env_id].clone()
        masses[self.env_id, 0] = original[0] + params["mass_offset_kg"]
        ids = torch.as_tensor([self.env_id], dtype=torch.int32, device="cpu")
        view.set_masses(masses, ids)
        after = view.get_masses()[self.env_id]
        readback = {"mass_offset_kg": float(_to_numpy(after[0] - original[0]))}
        self._restore.append(
            lambda: view.set_masses(
                view.get_masses().clone().index_copy_(0, ids.long(), original[None]), ids
            )
        )
        return InterventionReceipt(
            intervention, params, readback, "mass_readback", tolerance=1e-4, env_id=self.env_id
        )

    def release(self) -> None:
        """Undo every per-env write so the next arm starts from the scene defaults."""
        while self._restore:
            self._restore.pop()()
        self.pending_push = None


def is_finite_receipt(receipt: InterventionReceipt) -> bool:
    return receipt.readback is not None and all(math.isfinite(v) for v in receipt.readback.values())


__all__ = ["MaterialWriter", "PhoenixInterventionAdapter", "SUPPORTED_KINDS", "is_finite_receipt"]
