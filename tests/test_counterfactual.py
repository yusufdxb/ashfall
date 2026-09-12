"""Matched-pair departure: covariance, null calibration, sensitivity, onset."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ashfall.backends.toy import ToyBackend
from ashfall.counterfactual import (
    DEPARTURE_FEATURES,
    DepartureConfig,
    RestorableState,
    RolloutTrace,
    departure_onset,
    ledoit_wolf_shrinkage,
    measure_departure,
    sensitivity_analysis,
    shrunk_covariance,
)
from ashfall.ontology import Intervention


def trace(n=60, dt=0.02, vx=0.5, height=0.3, noise=0.0, seed=0, drop_after=None):
    rng = np.random.default_rng(seed)
    z = np.full(n, height) + rng.normal(0, noise, n)
    v = np.full(n, vx) + rng.normal(0, noise, n)
    if drop_after is not None:
        z[drop_after:] = np.linspace(height, 0.1, n - drop_after)
        v[drop_after:] = 0.0
    return RolloutTrace(
        dt,
        np.stack([np.arange(n) * vx * dt, np.zeros(n), z], 1),
        np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        np.stack([v, np.zeros(n), np.zeros(n)], 1),
        np.zeros((n, 3)),
        np.zeros((n, 12)),
        rng.normal(0, 0.1 + noise, (n, 12)),
        np.tile([vx, 0.0, 0.0], (n, 1)),
        "evaluation_horizon",
    )


class TestCovariance:
    def test_ledoit_wolf_matches_reference_values(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 4))
        x -= x.mean(0)
        # Value cross-checked against scikit-learn's ledoit_wolf_shrinkage on
        # the same draw (0.89942071...); pinned so a formula edit is visible.
        assert ledoit_wolf_shrinkage(x) == pytest.approx(0.8994207117502612, rel=1e-9)
        # Isotropic data IS the shrinkage target, so the optimal intensity is
        # one; strongly anisotropic data with many samples shrinks little.
        isotropic = rng.normal(size=(2000, 3))
        isotropic -= isotropic.mean(0)
        assert ledoit_wolf_shrinkage(isotropic) > 0.8
        anisotropic = rng.normal(size=(2000, 3)) * np.array([1.0, 4.0, 0.1])
        anisotropic -= anisotropic.mean(0)
        assert ledoit_wolf_shrinkage(anisotropic) < 0.1

    def test_shrunk_covariance_is_positive_definite_even_for_constant_channels(self):
        residuals = np.zeros((10, 3))
        residuals[:, 0] = np.linspace(-1, 1, 10)
        sigma, intensity = shrunk_covariance(residuals, floor=1e-6)
        assert np.all(np.linalg.eigvalsh(sigma) > 0)
        assert 0 <= intensity <= 1
        with pytest.raises(ValueError):
            shrunk_covariance(residuals, shrinkage=1.5)
        with pytest.raises(ValueError):
            shrunk_covariance(residuals[:1])

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=25, deadline=None)
    def test_explicit_intensity_is_honoured(self, intensity):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(30, 3))
        x -= x.mean(0)
        _, used = shrunk_covariance(x, shrinkage=intensity, floor=0.0)
        assert used == pytest.approx(intensity)


class TestDeparture:
    def nominal_set(self, noise=0.01, count=5):
        return [trace(noise=noise, seed=s) for s in range(count)]

    def test_identical_treatment_is_not_a_departure(self):
        nominal = self.nominal_set()
        control = nominal[0]
        result = measure_departure(control, control, nominal)
        assert result.treatment_vs_control == 0.0 and not result.passed
        assert result.replicates == 5 and len(result.null_pairwise) == 10

    def test_real_departure_passes_and_reports_direction(self):
        nominal = self.nominal_set()
        control = nominal[0]
        treatment = trace(noise=0.01, seed=0, drop_after=30)
        result = measure_departure(treatment, control, nominal, phenotype="collapse")
        assert result.passed and result.ratio_to_null_max > 1.5
        assert result.empirical_p == pytest.approx(result.p_floor)
        assert result.directional["moves_in_expected_direction"]
        assert departure_onset(result) is not None and departure_onset(result) >= 30

    def test_too_few_replicates_refused(self):
        nominal = self.nominal_set(count=2)
        with pytest.raises(ValueError, match="replicates"):
            measure_departure(nominal[0], nominal[0], nominal)

    def test_feature_scaling_invariance(self):
        """Mahalanobis distance does not depend on the units of a channel.

        Scaling every trace's forward speed by the same factor rescales that
        feature column everywhere; with a tiny floor the distance is unchanged.
        """
        nominal = self.nominal_set(noise=0.02)
        treatment = trace(noise=0.02, seed=0, drop_after=30)
        config = DepartureConfig(regularization_floor=1e-12, shrinkage=0.0)
        base = measure_departure(treatment, nominal[0], nominal, config=config)

        def scaled(t):
            return RolloutTrace(
                t.control_dt,
                t.base_pos,
                t.base_quat,
                t.base_lin_vel_body * [3.0, 1.0, 1.0],
                t.base_ang_vel_body,
                t.joint_pos,
                t.joint_vel,
                t.command_vel * [3.0, 1.0, 1.0],
                t.termination_reason,
            )

        scaled_nominal = [scaled(t) for t in nominal]
        again = measure_departure(
            scaled(treatment), scaled_nominal[0], scaled_nominal, config=config
        )
        assert again.treatment_vs_control == pytest.approx(base.treatment_vs_control, rel=1e-3)

    def test_sensitivity_reports_stability_over_regularisers(self):
        nominal = self.nominal_set()
        treatment = trace(noise=0.01, seed=0, drop_after=30)
        report = sensitivity_analysis(treatment, nominal[0], nominal, phenotype="collapse")
        assert report.primary_passed and report.stable
        assert len(report.cells) == 24
        quiet = sensitivity_analysis(nominal[1], nominal[0], nominal)
        assert not quiet.primary_passed and quiet.stable

    def test_config_validation(self):
        with pytest.raises(ValueError):
            DepartureConfig(statistic="median")
        with pytest.raises(ValueError):
            DepartureConfig(margin_ratio=0.5)
        with pytest.raises(ValueError):
            DepartureConfig(min_replicates=1)


class TestTraces:
    def test_features_are_restorable_only(self):
        t = trace()
        assert t.features().shape == (60, len(DEPARTURE_FEATURES))
        assert "contact" not in " ".join(DEPARTURE_FEATURES)

    def test_state_round_trip_and_identity(self):
        t = trace()
        state = t.state_at(3)
        assert state.state_id.startswith("st_")
        assert (
            RestorableState(**{k: v for k, v in state.to_dict().items() if k != "state_id"})
            == state
        )
        with pytest.raises(ValueError, match="unrecorded"):
            RestorableState(
                (0, 0, 0.0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0), (0,) * 12, (0,) * 12, (0, 0, 0)
            )

    def test_trace_validation(self):
        with pytest.raises(ValueError, match="same length"):
            RolloutTrace(
                0.02,
                np.zeros((3, 3)),
                np.tile([0, 0, 0, 1.0], (2, 1)),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                np.zeros((3, 12)),
                np.zeros((3, 12)),
                np.zeros((3, 3)),
                "x",
            )
        with pytest.raises(ValueError, match="terminated_step"):
            trace_bad = trace()
            RolloutTrace(
                trace_bad.control_dt,
                trace_bad.base_pos,
                trace_bad.base_quat,
                trace_bad.base_lin_vel_body,
                trace_bad.base_ang_vel_body,
                trace_bad.joint_pos,
                trace_bad.joint_vel,
                trace_bad.command_vel,
                "x",
                terminated_step=99,
            )


class TestToySurrogate:
    def test_same_seed_gives_identical_control_and_shared_noise(self):
        backend = ToyBackend()
        state = backend.nominal_rollout(seed=1, horizon_steps=100, command=(0.6, 0, 0)).state_at(40)
        a, _ = backend.rollout(state, Intervention.none(), seed=5, horizon_steps=80)
        b, _ = backend.rollout(state, Intervention.none(), seed=5, horizon_steps=80)
        c, _ = backend.rollout(state, Intervention.none(), seed=6, horizon_steps=80)
        assert a.content_hash() == b.content_hash() != c.content_hash()

    def test_unsupported_intervention_reports_itself(self):
        backend = ToyBackend()
        state = backend.spawn_state((0.5, 0, 0))
        _, receipt = backend.rollout(
            state,
            Intervention("terrain_geometry", (("step_height_m", 0.1),)),
            seed=1,
            horizon_steps=20,
        )
        assert receipt.unsupported_reason and not receipt.verified
