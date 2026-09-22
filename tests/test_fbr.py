"""Failure-Boundary Replay: criterion, capsule, seeds, boundary, acceptance, samplers.

The two regression groups that matter most:

* ``TestPhaseOneRegression``: the Phase-I curriculum seeded row 0, a nominal
  state far before onset. A far seed must be refused, and in the toy a replay
  from row 0 must be shown not to deliver the failure while the FBR seed does.
* ``TestAcceptance``: a replay is a reproduction only when the friction was
  read back, the treated replay is closer to the recording than its matched
  control, and it fails like the recording while the control does not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashfall.fbr.acceptance import AcceptanceConfig, accept_reconstruction, trajectory_distance
from ashfall.fbr.boundary import (
    BoundaryEstimate,
    BoundarySampler,
    BroadSampler,
    LocalNeighborhood,
    ReplayDraw,
    boundary_mass,
    estimate_boundary,
)
from ashfall.fbr.capsule import (
    assert_near_onset,
    capsule_criterion,
    extract_capsule,
    seed_points,
)
from ashfall.fbr.criterion import FailureCriterion, detect_onset, tilt_from_quat_xyzw
from ashfall.gates import InterventionReceipt
from ashfall.ontology import Intervention

SHA = "a" * 64
DT = 0.02


def pitch_quat(theta):
    return (0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2))


def row(t, theta, *, x=0.0, thd=0.0, terminated=False, v=1.0):
    return {
        "timestamp_s": t,
        "base_pos": (x, 0.0, 0.3),
        "base_quat": pitch_quat(theta),
        "base_lin_vel_body": (v, 0.0, 0.0),
        "base_ang_vel_body": (0.0, thd, 0.0),
        "joint_pos": (0.1,) * 12,
        "joint_vel": (0.0,) * 12,
        "command_vel": (1.0, 0.0, 0.0),
        "terminated": terminated,
    }


def slip_trajectory(n=100, onset=70, *, spike_at=None):
    """Nominal gait until ``onset - 15``, a growing pitch excursion, a fall at ``onset``."""
    rows = []
    for i in range(n):
        if i < onset - 15:
            theta = 0.02 * math.sin(i / 3)
        else:
            theta = 0.02 + 0.9 * (i - (onset - 15)) / 15
        if spike_at is not None and i == spike_at:
            theta = 1.0
        rows.append(row((i + 1) * DT, theta, x=0.02 * i))
    return rows


CRITERION = FailureCriterion(name="test_v1", max_tilt_rad=0.8, debounce_frames=3)


class TestCriterion:
    def test_tilt_is_combined_roll_pitch(self):
        assert tilt_from_quat_xyzw(pitch_quat(0.3)) == pytest.approx(0.3)
        assert tilt_from_quat_xyzw((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)

    def test_debounced_attitude_onset_is_first_frame_of_run(self):
        rows = slip_trajectory()
        onset = detect_onset(rows, CRITERION)
        first_over = next(
            i for i, r in enumerate(rows) if tilt_from_quat_xyzw(r["base_quat"]) > 0.8
        )
        assert onset.index == first_over and onset.reason == "attitude"

    def test_single_frame_spike_is_not_an_onset(self):
        rows = slip_trajectory(spike_at=20)
        assert detect_onset(rows, CRITERION).index > 20

    def test_terminal_flag_wins_when_earlier(self):
        rows = slip_trajectory()
        rows[40]["terminated"] = True
        onset = detect_onset(rows, CRITERION)
        assert (onset.index, onset.reason) == (40, "terminated")

    def test_no_failure_returns_none(self):
        rows = [row((i + 1) * DT, 0.01) for i in range(50)]
        assert detect_onset(rows, CRITERION) is None

    def test_stall_needs_velocity(self):
        crit = FailureCriterion(stall_speed_fraction=0.3)
        rows = [row((i + 1) * DT, 0.0) for i in range(10)]
        del rows[3]["base_lin_vel_body"]
        with pytest.raises(ValueError, match="stall"):
            detect_onset(rows, crit)

    def test_stall_onset(self):
        crit = FailureCriterion(stall_speed_fraction=0.3, stall_seconds=0.2)
        rows = [row((i + 1) * DT, 0.0, v=1.0 if i < 20 else 0.1) for i in range(40)]
        onset = detect_onset(rows, crit)
        assert (onset.index, onset.reason) == (20, "stall")

    def test_criterion_identity_changes_with_definition(self):
        assert CRITERION.criterion_id != FailureCriterion(max_tilt_rad=0.7).criterion_id


class TestCapsule:
    def test_extract_records_criterion_and_policy(self):
        capsule = extract_capsule(
            slip_trajectory(), criterion=CRITERION, source="unit", robot="go2", policy_sha256=SHA
        )
        assert capsule.checkpoint_sha256 == SHA
        assert capsule_criterion(capsule) == CRITERION
        assert capsule.event_descriptor["onset_reason"] == "attitude"

    def test_refuses_rollout_without_failure(self):
        rows = [row((i + 1) * DT, 0.01) for i in range(50)]
        with pytest.raises(ValueError, match="no failure"):
            extract_capsule(rows, criterion=CRITERION, source="u", robot="go2", policy_sha256=SHA)

    def test_refuses_unnamed_policy(self):
        with pytest.raises(ValueError, match="SHA256"):
            extract_capsule(
                slip_trajectory(), criterion=CRITERION, source="u", robot="go2", policy_sha256="x"
            )

    def test_seed_points_are_strictly_pre_onset_at_declared_leads(self):
        capsule = extract_capsule(
            slip_trajectory(), criterion=CRITERION, source="u", robot="go2", policy_sha256=SHA
        )
        points = seed_points(capsule, lead_times_s=(0.2, 0.4, 0.6))
        onset = capsule.failure_onset_index
        assert [p.row for p in points] == sorted((p.row for p in points), reverse=True)
        for p, lead in zip(points, (0.2, 0.4, 0.6)):
            assert p.row < onset
            assert lead - 1e-9 <= p.lead_s < lead + DT + 1e-9

    def test_seed_already_failing_is_refused(self):
        # A one-frame tilt spike is not an onset under debounce 3, but seeding on it
        # would hand the simulator a state that already meets the criterion.
        rows = slip_trajectory(spike_at=56)
        capsule = extract_capsule(
            rows, criterion=CRITERION, source="u", robot="go2", policy_sha256=SHA, history_s=1.5
        )
        onset_time = capsule.frames[capsule.failure_onset_index].timestamp_s
        lead = onset_time - capsule.frames[56].timestamp_s
        with pytest.raises(ValueError, match="already meets"):
            seed_points(capsule, lead_times_s=(lead,))


class TestPhaseOneRegression:
    """Phase I seeded row 0. FBR refuses a far seed and the toy shows why."""

    def test_row_zero_far_before_onset_is_refused(self):
        capsule = extract_capsule(
            slip_trajectory(),
            criterion=CRITERION,
            source="u",
            robot="go2",
            policy_sha256=SHA,
            history_s=2.0,
        )
        assert capsule.pre_failure_start_index == 0  # row 0 is inside the reviewed window
        with pytest.raises(ValueError, match="nominal state, not a failure boundary"):
            assert_near_onset(capsule, 0, max_lead_s=1.0)

    def test_lead_beyond_the_boundary_window_is_refused(self):
        capsule = extract_capsule(
            slip_trajectory(),
            criterion=CRITERION,
            source="u",
            robot="go2",
            policy_sha256=SHA,
            history_s=2.0,
        )
        with pytest.raises(ValueError, match="boundary window"):
            seed_points(capsule, lead_times_s=(1.2,), max_lead_s=1.0)

    def test_toy_replay_from_fbr_seed_delivers_the_failure_and_row_zero_does_not(self):
        from ashfall.fbr import toy_slip as ts

        cfg = ts.ToySlipConfig()
        theta = ts.initial_params()
        reps = np.arange(12) + 5_000

        def failure_rate(frame, window_s, mu):
            seed = ts.seed_from_frame(frame, "s", 1.0)
            b = ts.seed_batch([seed] * len(reps), np.full(len(reps), mu), reps, noise_scale=0.0)
            r = ts.simulate(cfg, theta, b, steps=int(round(window_s / cfg.dt)))
            return float(r.failed[0].mean())

        found = None
        for trial in range(300):
            batch = ts.condition_batch(
                cfg, mu=0.06, patch_start=1.0, v_cmd=1.1, seeds=np.array([trial])
            )
            res = ts.simulate(cfg, theta, batch, record_trace=True)
            rows = ts.frames_from_trace(res, 0, batch, cfg)
            if not (res.failed[0, 0] and rows[-1]["base_pos"][0] >= 1.0):
                continue  # no failure, or a fall before the patch
            capsule = extract_capsule(
                rows,
                criterion=ts.TOY_CRITERION,
                source="toy",
                robot="slip_cartpole",
                policy_sha256=SHA,
                history_s=10.0,
                post_s=0.0,
                control_dt=cfg.dt,
            )
            (seed,) = seed_points(capsule, lead_times_s=(0.4,))
            window = seed.lead_s + 0.3
            # Attributable to friction: the matched full-friction replay survives.
            if failure_rate(seed.frame, window, cfg.mu_ground) <= 1 / 3:
                found = (capsule, seed, window)
                break
        assert found is not None, "no friction-attributable patch failure in 300 trials"
        capsule, seed, window = found
        onset_s = capsule.frames[capsule.failure_onset_index].timestamp_s
        assert onset_s - capsule.frames[0].timestamp_s > 1.0
        # The FBR seed delivers the failure inside the boundary window. The
        # Phase-I row-0 seed (the start line) delivers nothing in the same
        # window under the same friction, and FBR refuses it.
        assert failure_rate(seed.frame, window, 0.06) >= 2 / 3
        assert failure_rate(capsule.frames[0], window, 0.06) <= 1 / 3
        with pytest.raises(ValueError, match="boundary window"):
            assert_near_onset(capsule, 0, max_lead_s=1.0)


class TestBoundary:
    @staticmethod
    def outcomes(mu_b, reps=10):
        grid = np.linspace(0.05, 0.8, 16)
        return {float(mu): [mu < mu_b] * reps for mu in grid}

    def test_crossing_located(self):
        est = estimate_boundary("s", self.outcomes(0.3), seed_source="deployment_failure")
        assert 0.25 <= est.mu_boundary <= 0.35

    def test_always_robust_is_refused(self):
        grid = np.linspace(0.05, 0.8, 8)
        with pytest.raises(ValueError, match="above_support"):
            estimate_boundary(
                "s", {float(m): [False] * 5 for m in grid}, seed_source="nominal_rollout"
            )

    def test_always_failing_is_refused(self):
        grid = np.linspace(0.05, 0.8, 8)
        with pytest.raises(ValueError, match="below_support"):
            estimate_boundary(
                "s", {float(m): [True] * 5 for m in grid}, seed_source="nominal_rollout"
            )

    def test_refusal_carries_its_censoring_kind_even_with_colons_in_the_seed_id(self):
        from ashfall.fbr.boundary import BoundaryNotIdentified

        grid = np.linspace(0.05, 0.8, 8)
        with pytest.raises(BoundaryNotIdentified) as info:
            estimate_boundary(
                "cap_x:118", {float(m): [False] * 5 for m in grid}, seed_source="nominal_rollout"
            )
        assert info.value.censoring == "above_support"

    def test_toy_refresh_clamps_a_seed_that_became_robust(self):
        # Regression: the drop reason used to be parsed out of the message with
        # split(":"), which returned the seed row for ids like "cap_x:118", so a
        # robust seed was dropped instead of clamped to the hardest friction.
        from ashfall.fbr import toy_slip as ts

        cfg = ts.ToySlipConfig()
        robust = ts.ToySeed("cap_x:118", 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, -5.0)  # patch behind it
        found, info = ts.boundaries_for(
            cfg,
            ts.initial_params(),
            [robust],
            seed_source="deployment_failure",
            clamp_robust=True,
            replay_steps=20,
        )
        assert [b.censoring for b in found] == ["clamped_robust"] and not info["dropped"]
        found, info = ts.boundaries_for(
            cfg,
            ts.initial_params(),
            [robust],
            seed_source="deployment_failure",
            clamp_robust=False,
            replay_steps=20,
        )
        assert not found and info["dropped"] == {"cap_x:118": "above_support"}

    def test_sampler_stays_in_the_neighbourhood_and_mixes_nominal(self):
        est = estimate_boundary("s", self.outcomes(0.3), seed_source="deployment_failure")
        nb = LocalNeighborhood(mu_half_width=0.05)
        draws = BoundarySampler([est], nb, nominal_fraction=0.3, rng_seed=0).sample(4000)
        boundary = [d for d in draws if d.kind == "boundary"]
        assert abs(len(boundary) / 4000 - 0.7) < 0.03
        assert all(
            est.mu_interval[0] - 0.05 - 1e-9 <= d.mu <= est.mu_interval[1] + 0.05 + 1e-9
            for d in boundary
        )

    def test_one_sampler_one_seed_source(self):
        a = BoundaryEstimate(
            "a", "deployment_failure", 0.3, (0.3, 0.3), "interpolated_crossing", ()
        )
        b = BoundaryEstimate("b", "nominal_rollout", 0.3, (0.3, 0.3), "interpolated_crossing", ())
        with pytest.raises(ValueError, match="one seed source"):
            BoundarySampler([a, b], LocalNeighborhood(), nominal_fraction=0.3, rng_seed=0)

    def test_broad_sampler_range(self):
        draws = BroadSampler((0.05, 0.8), rng_seed=1).sample(1000)
        assert all(d.kind == "broad" and 0.05 <= d.mu <= 0.8 for d in draws)

    def test_draw_validation(self):
        with pytest.raises(ValueError):
            ReplayDraw("boundary", seed_id=None, mu=0.3)
        with pytest.raises(ValueError):
            ReplayDraw("broad")

    def test_boundary_mass(self):
        assert boundary_mass([0.0, 1.0]) == 0.0
        assert boundary_mass([0.5]) == pytest.approx(0.5)


def receipt(mu, verified=True):
    coefficients = {"static_friction": mu, "dynamic_friction": mu}
    return InterventionReceipt(
        Intervention("friction_reduction", tuple(coefficients.items())),
        dict(coefficients),
        dict(coefficients) if verified else None,
        "unit_readback",
    )


def trace(thetas, t0=0.0):
    return [row(t0 + (i + 1) * DT, th, thd=0.0) for i, th in enumerate(thetas)]


class TestAcceptance:
    CFG = AcceptanceConfig(scales={"tilt_rad": 0.1}, onset_tolerance_s=0.1)
    CRIT = FailureCriterion(name="acc", max_tilt_rad=0.8, debounce_frames=1)

    def observed(self):
        return trace(np.linspace(0.0, 0.9, 20))  # meets the criterion at the end

    def test_reproduced_when_all_three_checks_pass(self):
        obs = self.observed()
        onset = detect_onset(obs, self.CRIT).timestamp_s - obs[0]["timestamp_s"]
        treated = trace(np.linspace(0.0, 0.92, 20))
        control = trace(np.zeros(20))
        v = accept_reconstruction(
            0.1, obs, onset, [(control, treated, receipt(0.1))], self.CRIT, self.CFG
        )
        assert v.status == "REPRODUCED"

    def test_unverified_friction_is_not_a_reproduction(self):
        obs = self.observed()
        onset = detect_onset(obs, self.CRIT).timestamp_s - obs[0]["timestamp_s"]
        treated, control = trace(np.linspace(0.0, 0.92, 20)), trace(np.zeros(20))
        v = accept_reconstruction(
            0.1, obs, onset, [(control, treated, receipt(0.1, False))], self.CRIT, self.CFG
        )
        assert v.status == "UNREPRODUCED" and not v.pairs[0].delivered

    def test_control_that_also_fails_blocks_attribution(self):
        obs = self.observed()
        onset = detect_onset(obs, self.CRIT).timestamp_s - obs[0]["timestamp_s"]
        treated = trace(np.linspace(0.0, 0.92, 20))
        control = trace(np.linspace(0.0, 0.95, 20))
        v = accept_reconstruction(
            0.1, obs, onset, [(control, treated, receipt(0.1))], self.CRIT, self.CFG
        )
        assert not v.pairs[0].failure_match and v.status == "UNREPRODUCED"

    def test_treated_no_closer_than_control_fails_trajectory_match(self):
        obs = self.observed()
        onset = detect_onset(obs, self.CRIT).timestamp_s - obs[0]["timestamp_s"]
        treated = trace(np.concatenate([np.zeros(19), [0.9]]))  # control-like until the end
        control = trace(np.zeros(20))
        v = accept_reconstruction(
            0.1, obs, onset, [(control, treated, receipt(0.1))], self.CRIT, self.CFG
        )
        assert not v.pairs[0].trajectory_match

    def test_distance_is_zero_for_identical_rollouts(self):
        obs = self.observed()
        assert trajectory_distance(obs, obs, {"tilt_rad": 0.1}) == 0.0


class TestToyBudget:
    def test_every_arm_receives_the_same_rollout_budget(self):
        from ashfall.fbr import toy_slip as ts

        cfg = ts.ToySlipConfig()
        es = ts.ESConfig(iterations=2, pairs=2, contexts=4, refresh_every=1)
        theta = ts.initial_params()
        seeds = [ts.ToySeed(f"s{k}", 0.9, 1.0, 0.0, 0.0, 0.5 * k, 1.1, 1.0) for k in range(3)]
        nb = LocalNeighborhood()
        slots = set()
        for arm, kw in (
            ("B_broad_dr", {}),
            ("E_friction_band", {"band": (0.05, 0.2)}),
            ("D_fbr", {"seeds": seeds, "seed_source": "deployment_failure"}),
        ):
            try:
                run = ts.run_arm(
                    arm, cfg, es, theta, seed=1, neighborhood=nb, nominal_fraction=0.3, **kw
                )
            except ValueError as exc:  # a D seed set with no identified boundary is refused
                assert "no seed has an identified boundary" in str(exc)
                continue
            slots.add(run.log["rollout_slots"])
        assert slots == {es.rollout_slots}


class TestV2Pieces:
    def test_seed_before_position_is_the_last_frame_short_of_the_patch(self):
        from ashfall.fbr.capsule import seed_before_position

        capsule = extract_capsule(
            slip_trajectory(),
            criterion=CRITERION,
            source="u",
            robot="go2",
            policy_sha256=SHA,
            history_s=2.0,
        )
        seed = seed_before_position(capsule, 1.0, max_lead_s=2.0)  # x = 0.02 * row
        assert (
            capsule.frames[seed.row].base_pos[0] < 1.0 <= capsule.frames[seed.row + 1].base_pos[0]
        )
        with pytest.raises(ValueError, match="boundary window"):
            seed_before_position(capsule, 0.5, max_lead_s=0.5)

    def test_one_sided_band_and_broad_base(self):
        est = estimate_boundary("s", TestBoundary.outcomes(0.3), seed_source="deployment_failure")
        nb = LocalNeighborhood(mu_half_width=0.05, mu_min=0.02)
        draws = BoundarySampler(
            [est], nb, nominal_fraction=0.5, rng_seed=0, one_sided=True, base_mu_range=(0.05, 0.8)
        ).sample(4000)
        replay = [d.mu for d in draws if d.kind == "boundary"]
        base = [d for d in draws if d.kind != "boundary"]
        assert min(replay) < 0.1 and max(replay) <= est.mu_interval[1] + 0.05 + 1e-9
        assert all(d.kind == "broad" and 0.05 <= d.mu <= 0.8 for d in base)
