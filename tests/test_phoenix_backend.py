"""CPU protocol integration, not a locomotion or simulator result."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ashfall.backends.phoenix import PhoenixBackend, _policy_observations, resolve_scenario_row
from ashfall.capsule import CapsuleFrame, FailureCapsule
from ashfall.provenance import content_hash
from ashfall.reproduction import ReproductionCandidate
from ashfall.scenarios import Scenario

torch = pytest.importorskip("torch")
pytest.importorskip("phoenix")


class MaterialView:
    def __init__(self):
        self.values = torch.tensor([[[1.0, 0.8, 0.0]]] * 4).reshape(1, 4, 3)

    def get_material_properties(self):
        return self.values

    def set_material_properties(self, values, ids):
        self.values[ids] = values[ids]


class Robot:
    def __init__(self):
        self.root_physx_view = MaterialView()
        self.data = SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.3]]),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_lin_vel_b=torch.zeros((1, 3)),
            root_ang_vel_b=torch.zeros((1, 3)),
            joint_vel=torch.zeros((1, 12)),
        )

    def write_root_pose_to_sim(self, pose, env_ids):
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.data.root_quat_w[env_ids] = pose[:, 3:]

    def write_root_velocity_to_sim(self, velocities, env_ids):
        # Tests use identity orientation; Phoenix transform tests cover yaw.
        self.data.root_lin_vel_b[env_ids] = velocities[:, :3]
        self.data.root_ang_vel_b[env_ids] = velocities[:, 3:]

    def write_joint_state_to_sim(self, positions, velocities, env_ids):
        self.data.joint_vel[env_ids] = velocities


class CommandTerm:
    def __init__(self):
        self.vel_command_b = torch.zeros((1, 3))
        self.time_left = torch.ones(1)
        self.is_heading_env = torch.ones(1, dtype=torch.bool)
        self.is_standing_env = torch.ones(1, dtype=torch.bool)

    @property
    def command(self):
        return self.vel_command_b


class Scene(dict):
    env_origins = torch.zeros((1, 3))

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


class FakeEnv:
    device = "cpu"
    step_dt = 0.02
    max_episode_length = 1000

    def __init__(self, fail=True, capture=True):
        self.unwrapped = self
        self.scene = Scene(robot=Robot())
        self.scene["contact_forces"] = SimpleNamespace(
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            data=SimpleNamespace(
                net_forces_w=torch.tensor([[[0.0, 0.0, 50.0]]] * 4).reshape(1, 4, 3)
            ),
        )
        self.term = CommandTerm()
        self.command_manager = SimpleNamespace(
            get_term=lambda name: self.term, get_command=lambda name: self.term.command
        )
        self.termination_manager = SimpleNamespace(
            active_terms=["fall"],
            terminated=torch.tensor([False]),
            time_outs=torch.tensor([False]),
            get_term=lambda name: torch.tensor([self.failed]),
            get_term_cfg=lambda name: SimpleNamespace(time_out=False),
        )
        self.sim = SimpleNamespace(forward=lambda: None)
        self.count, self.fail, self.capture, self.failed = 0, fail, capture, False
        self.seeds = []
        self.closed = False

    def seed(self, seed):
        self.seeds.append(seed)

    def get_observations(self):
        return {"policy": torch.zeros((1, 48))}

    def _reset_idx(self, ids):
        self.scene["robot"].data.root_pos_w[:, 2] = 0.3
        self.failed = False
        self.termination_manager.terminated[:] = False

    def step(self, action):
        self.count += 1
        done = self.count == 2 and self.fail
        if done:
            self.scene["robot"].data.root_pos_w[:, 2] = 0.08
            self.failed = True
            self.termination_manager.terminated[:] = True
            if self.capture:
                self._reset_idx(torch.tensor([0]))
        return self.get_observations(), torch.tensor([1.0]), torch.tensor([done]), {}

    def close(self):
        self.closed = True


@pytest.fixture
def backend(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"cpu-fixture-checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("fixture: true\n")
    obj = PhoenixBackend(
        checkpoint, config, config, device="cpu", horizon_steps=3, output_dir=tmp_path / "evidence"
    )
    starts = []

    def start(seed):
        starts.append(seed)
        obj.env = FakeEnv()
        obj.policy = lambda observation: torch.zeros((1, 12))

    monkeypatch.setattr(obj, "_start_episode", start)
    obj.starts = starts
    return obj


def capsule(policy_id):
    frames = tuple(
        CapsuleFrame(
            i * 0.02,
            base_pos=(0.0, 0.0, 0.3),
            base_quat=(0.0, 0.0, 0.0, 1.0),
            base_lin_vel_body=(0.5, 0.0, 0.0),
            base_ang_vel_body=(0.0, 0.0, 0.0),
            joint_pos=(0.0,) * 12,
            joint_vel=(0.0,) * 12,
            command_vel=(0.5, 0.0, 0.0),
        )
        for i in range(70)
    )
    return FailureCapsule(
        source="synthetic_fixture",
        robot="go2",
        policy_id=policy_id,
        timestamp=None,
        control_dt=0.02,
        failure_mode="collapse",
        failure_onset_index=50,
        pre_failure_start_index=25,
        post_failure_end_index=69,
        frames=frames,
    )


def candidate(cap):
    return ReproductionCandidate(
        cap.capsule_id, (("static_friction", 0.4), ("dynamic_friction", 0.3)), 40
    )


def scenario(cap):
    return Scenario(
        cap.capsule_id,
        candidate(cap).parameters,
        71,
        "held_out",
        "reproduction",
        content_hash({"capsule_id": cap.capsule_id, "seed_row": 40}),
    )


def test_real_adapters_capture_pre_reset_failure_and_restore_state(backend):
    cap = capsule(backend.policy_id)
    result = backend.replay(cap, candidate(cap), backend.policy_id, 101)
    assert result.mode == "collapse"
    assert result.time_to_failure_s == 0.04
    assert result.termination_reason == "fall"
    assert backend.env.scene["robot"].data.root_pos_w[0, 2] == pytest.approx(0.3)
    assert backend.last_telemetry[-1]["base_height_m"] == pytest.approx(0.08)
    assert backend.last_reset["resolved_row"] == 40
    assert backend.last_reset["time_before_onset_seconds"] == pytest.approx(0.2)
    assert backend.last_reset["restored_command"] == [0.5, 0.0, 0.0]
    assert backend.last_reset["restored_linear_velocity_world"] == [0.5, 0.0, 0.0]
    assert not backend.last_outcome.success
    assert backend.last_outcome.intervention_required
    assert backend.last_outcome.recovery_outcome == "unrecovered"
    assert len(list(backend.output_dir.glob("episode_*.json"))) == 1


def test_frozen_ids_and_separate_seeds(backend):
    cap = capsule(backend.policy_id)
    case = scenario(cap)
    outcome = backend.evaluate_scenario(cap, case, backend.policy_id, 123)
    assert outcome.scenario_id == case.scenario_id
    assert outcome.parameter_sample_id == case.parameter_sample_id
    assert outcome.scenario_seed == 71
    assert outcome.evaluation_seed == 123
    assert outcome.training_seed is None
    assert backend.starts == [71]
    assert backend.env.seeds == [123]


def test_missing_terminal_capture_rejected(backend, monkeypatch):
    original = backend._start_episode

    def start(seed):
        original(seed)
        backend.env.capture = False

    monkeypatch.setattr(backend, "_start_episode", start)
    cap = capsule(backend.policy_id)
    with pytest.raises(RuntimeError, match="pre-reset"):
        backend.replay(cap, candidate(cap), backend.policy_id, 1)


def test_full_horizon_without_events_is_success(backend, monkeypatch):
    original = backend._start_episode

    def start(seed):
        original(seed)
        backend.env.fail = False

    monkeypatch.setattr(backend, "_start_episode", start)
    cap = capsule(backend.policy_id)
    outcome = backend.evaluate_scenario(cap, scenario(cap), backend.policy_id, 1)
    assert outcome.success
    assert outcome.failure_modes == []
    assert outcome.episode_length_steps == 3
    assert outcome.episode_return == 3.0
    assert outcome.termination_reason == "evaluation_horizon"


def test_policy_and_context_validation_precede_simulator(backend):
    cap = capsule(backend.policy_id)
    with pytest.raises(ValueError, match="checkpoint"):
        backend.evaluate_scenario(cap, scenario(cap), "incorrect", 1)
    with pytest.raises(ValueError, match="unsupported"):
        backend.replay(
            cap, replace(candidate(cap), parameters=(("slope", 0.2),)), backend.policy_id, 1
        )
    with pytest.raises(ValueError, match="row"):
        backend.replay(cap, replace(candidate(cap), seed_row=0), backend.policy_id, 1)
    assert backend.starts == []
    backend.checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        backend.replay(cap, candidate(cap), backend.policy_id, 1)


def test_unknown_state_identity_rejected(backend):
    cap = capsule(backend.policy_id)
    assert resolve_scenario_row(cap, scenario(cap)) == 40
    with pytest.raises(ValueError, match="identity"):
        resolve_scenario_row(cap, replace(scenario(cap), initial_state_id="different"))


def test_missing_seed_state_and_dt_mismatch_fail(backend):
    cap = capsule(backend.policy_id)
    altered = replace(cap, control_dt=0.01, capsule_id="")
    with pytest.raises(ValueError, match="periods"):
        backend.replay(altered, candidate(altered), backend.policy_id, 1)
    frames = list(cap.frames)
    frames[40] = replace(frames[40], command_vel=None)
    altered = replace(cap, frames=tuple(frames), capsule_id="")
    with pytest.raises(ValueError, match="missing"):
        backend.replay(altered, candidate(altered), backend.policy_id, 1)


def test_tensor_dictionary_policy_api():
    obs = {"policy": torch.zeros((1, 48))}
    assert _policy_observations(SimpleNamespace(obs_groups=["policy"]), obs) is obs
    assert _policy_observations(lambda x: x, obs) is obs["policy"]


def test_negative_scenario_seed_cannot_invoke_random_seed_mode(backend):
    cap = capsule(backend.policy_id)
    with pytest.raises(ValueError, match="scenario_seed"):
        backend.evaluate_scenario(
            cap, replace(scenario(cap), scenario_seed=-1), backend.policy_id, 1
        )
    assert backend.starts == []
