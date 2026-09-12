"""Phoenix intervention adapters against CPU fakes: every write is read back."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ashfall.backends.phoenix_interventions import MaterialWriter, PhoenixInterventionAdapter
from ashfall.ontology import Intervention

torch = pytest.importorskip("torch")


class MaterialView:
    """Torch-returning material view with shape (envs, shapes, 3)."""

    def __init__(self, envs=2):
        self.values = torch.tensor([[[0.9, 0.8, 0.0]] * 4] * envs)
        self.masses = torch.tensor([[6.0, 0.5, 0.5]] * envs)

    def get_material_properties(self):
        return self.values.clone()

    def set_material_properties(self, values, indices):
        self.values[indices.long()] = values[indices.long()]

    def get_masses(self):
        return self.masses.clone()

    def set_masses(self, masses, indices):
        self.masses[indices.long()] = masses[indices.long()]


class CommandTerm:
    def __init__(self, envs=2):
        self.vel_command_b = torch.zeros((envs, 3))
        self.time_left = torch.ones(envs)
        self.is_heading_env = torch.ones(envs, dtype=torch.bool)
        self.is_standing_env = torch.ones(envs, dtype=torch.bool)

    @property
    def command(self):
        return self.vel_command_b


class Robot:
    def __init__(self, envs=2):
        self.root_physx_view = MaterialView(envs)
        self.data = SimpleNamespace(root_vel_w=torch.zeros((envs, 6)))
        self.actuators = {
            "legs": SimpleNamespace(
                stiffness=torch.full((envs, 12), 25.0), damping=torch.full((envs, 12), 0.5)
            )
        }
        self.writes = []

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.data.root_vel_w[env_ids] = velocity
        self.writes.append((velocity.clone(), env_ids.clone()))


def fake_env(envs=2):
    term = CommandTerm(envs)
    return SimpleNamespace(
        scene={"robot": Robot(envs)},
        command_manager=SimpleNamespace(get_term=lambda name: term),
    )


def test_friction_is_written_read_back_and_released():
    env = fake_env()
    adapter = PhoenixInterventionAdapter(env, env_id=1)
    control = adapter.control_receipt()
    assert control.verified
    assert control.baseline == pytest.approx({"static_friction": 0.9, "dynamic_friction": 0.8})
    receipt = adapter.apply(Intervention.friction_reduction(0.3, 0.2))
    assert receipt.verified and receipt.readback == pytest.approx(
        {"static_friction": 0.3, "dynamic_friction": 0.2}
    )
    values = env.scene["robot"].root_physx_view.values
    assert values[1, :, 0].tolist() == pytest.approx([0.3] * 4)
    assert values[0, :, 0].tolist() == pytest.approx([0.9] * 4)  # other env untouched
    adapter.release()
    assert values[1, :, 0].tolist() == pytest.approx([0.9] * 4)


def test_material_writer_refuses_dynamic_above_static():
    writer = MaterialWriter(fake_env())
    with pytest.raises(ValueError, match="dynamic friction"):
        writer.apply(0, {"static_friction": 0.1, "dynamic_friction": 0.5})


def test_command_corruption_reads_back_the_buffer():
    env = fake_env()
    adapter = PhoenixInterventionAdapter(env, env_id=0)
    receipt = adapter.apply(Intervention("command_corruption", (("command_vx_mps", 2.0),)))
    assert receipt.verified and receipt.readback == {"command_vx_mps": 2.0}
    term = env.command_manager.get_term("base_velocity")
    assert term.vel_command_b[0].tolist() == [2.0, 0.0, 0.0]
    assert not term.is_heading_env[0] and not term.is_standing_env[0]


def test_push_applies_at_its_step_and_reads_back_the_delta():
    env = fake_env()
    adapter = PhoenixInterventionAdapter(env, env_id=0)
    intervention = Intervention(
        "external_perturbation", (("push_vx_mps", 1.0), ("push_time_s", 0.1))
    )
    pending = adapter.apply(intervention)
    assert not pending.verified and pending.method == "root_velocity_write_pending"
    assert adapter.apply_pending_push(2, 0.02, intervention) is None
    receipt = adapter.apply_pending_push(5, 0.02, intervention)
    assert receipt is not None and receipt.verified and receipt.applied_at_step == 5
    assert receipt.readback == pytest.approx({"push_vx_mps": 1.0, "push_time_s": 0.1})
    assert adapter.apply_pending_push(5, 0.02, intervention) is None  # applied once


def test_actuator_weakening_scales_and_restores_gains():
    env = fake_env()
    adapter = PhoenixInterventionAdapter(env, env_id=1)
    receipt = adapter.apply(
        Intervention("actuator_weakening", (("stiffness_scale", 0.5), ("damping_scale", 0.5)))
    )
    assert receipt.verified and receipt.readback == pytest.approx(
        {"stiffness_scale": 0.5, "damping_scale": 0.5}
    )
    legs = env.scene["robot"].actuators["legs"]
    assert legs.stiffness[1].tolist() == pytest.approx([12.5] * 12)
    assert legs.stiffness[0].tolist() == pytest.approx([25.0] * 12)
    adapter.release()
    assert legs.stiffness[1].tolist() == pytest.approx([25.0] * 12)


def test_payload_mass_reads_back_the_offset():
    env = fake_env()
    adapter = PhoenixInterventionAdapter(env, env_id=0)
    receipt = adapter.apply(Intervention("payload_mass", (("mass_offset_kg", 2.0),)))
    assert receipt.verified and receipt.readback == pytest.approx({"mass_offset_kg": 2.0})
    assert env.scene["robot"].root_physx_view.masses[0, 0].item() == pytest.approx(8.0)
    adapter.release()
    assert env.scene["robot"].root_physx_view.masses[0, 0].item() == pytest.approx(6.0)


def test_terrain_is_unsupported_not_silent():
    receipt = PhoenixInterventionAdapter(fake_env()).apply(
        Intervention("terrain_geometry", (("step_height_m", 0.1),))
    )
    assert receipt.unsupported_reason and not receipt.verified


def test_missing_actuator_models_are_unsupported():
    env = fake_env()
    env.scene["robot"].actuators = {}
    receipt = PhoenixInterventionAdapter(env).apply(
        Intervention("actuator_weakening", (("stiffness_scale", 0.5),))
    )
    assert receipt.unsupported_reason
