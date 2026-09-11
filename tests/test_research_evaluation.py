import numpy as np
import pytest

from ashfall.evaluation.metrics import FailureAnalyzer, RecoveryConfig
from ashfall.evaluation.paired import paired_comparison, target_failure_probability
from ashfall.evaluation.regression import RegressionBudget, regression_verdict


def telemetry(t, height=0.3, contacts=True, done=False):
    return dict(
        timestamp_s=t,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=height,
        cmd_lin_vel=np.array([0.5, 0.0]),
        actual_lin_vel=np.array([0.5, 0.0]),
        contact_forces=np.ones(4) * 20 if contacts else None,
        done=done,
    )


def test_event_free_steps_are_not_recovery():
    analyzer = FailureAnalyzer(recovery=RecoveryConfig(consecutive_steps=3))
    analyzer.step(**telemetry(0, height=0.1))
    for i in range(1, 20):
        analyzer.step(**telemetry(i * 0.02, height=0.1))
    result = analyzer.compute()
    assert result.mean_recovery_time_s is None
    assert result.unrecovered_events == 1


def test_recovery_requires_contacts_and_reports_entry_seconds():
    analyzer = FailureAnalyzer(recovery=RecoveryConfig(consecutive_steps=3))
    analyzer.step(**telemetry(0, height=0.1))
    analyzer.step(**telemetry(0.1, contacts=False))
    for t in [0.2, 0.3, 0.4]:
        analyzer.step(**telemetry(t, done=t == 0.4))
    metrics = analyzer.compute()
    assert metrics.mean_recovery_time_s == pytest.approx(0.2)
    assert metrics.recovered_events == 1
    assert metrics.unrecovered_events == 0


def test_episode_boundaries_reset_detector_and_censor_unrecovered():
    analyzer = FailureAnalyzer()
    for _ in range(2):
        analyzer.step(**telemetry(0, height=0.1, done=True))
    result = analyzer.compute()
    assert result.total_failures == 2
    assert result.unrecovered_events == 2
    assert result.episode_incidence_by_mode == {"collapse": 1.0}


def test_incidence_counts_episodes_not_events():
    analyzer = FailureAnalyzer()
    for t in [0.0, 1.1, 2.2]:
        analyzer.step(**telemetry(t, height=0.1, done=t == 2.2))
    analyzer.step(**telemetry(0, done=True))
    result = analyzer.compute()
    assert result.failure_rate == 1.5
    assert result.episode_incidence_by_mode["collapse"] == 0.5
    assert result.repeated_events_after_recovery == {}


def records(success=True, tracking=0.1, intervention=False):
    return [
        dict(
            scenario_id=f"s{i}",
            scenario_seed=i,
            evaluation_seed=3,
            parameter_sample_id=f"p{i}",
            policy_id="baseline",
            training_seed=42,
            success=success,
            tracking_error=tracking,
            intervention_required=intervention,
            failure_modes=[] if success else ["slip"],
            environment_parameters={"friction": 0.5},
        )
        for i in range(5)
    ]


def test_paired_actual_outcomes_and_deterministic_intervals():
    a, b = records(False), records(True)
    result = paired_comparison(a, b)
    assert result.candidate_minus_baseline == 1
    assert result.ci_low == result.ci_high == 1
    assert result == paired_comparison(a, list(reversed(b)))
    assert target_failure_probability(a, "slip") == 1


@pytest.mark.parametrize("damage", ["missing", "duplicate", "seed", "parameters", "unknown"])
def test_pairing_fails_closed(damage):
    a, b = records(), records()
    if damage == "missing":
        b.pop()
    if damage == "duplicate":
        b.append(b[0])
    if damage == "seed":
        b[0]["evaluation_seed"] = 7
    if damage == "parameters":
        b[0]["environment_parameters"] = {}
    if damage == "unknown":
        b[0]["scenario_id"] = None
    with pytest.raises(ValueError):
        paired_comparison(a, b)


def test_regression_rejects_target_only_improvement():
    verdict = regression_verdict(0.2, records(), records(False))
    assert verdict.target_improved and not verdict.regression_passed and not verdict.accepted
    assert regression_verdict(0.2, records(), records()).accepted
    assert not regression_verdict(0.01, records(), records()).accepted
    assert not regression_verdict(float("nan"), records(), records()).accepted
    assert not regression_verdict(0.2, records(), records(intervention=True)).accepted
    with pytest.raises(ValueError):
        RegressionBudget(nominal_success_drop=-0.1)


def test_missing_tracking_or_intervention_cannot_pass():
    candidate = records()
    candidate[0]["intervention_required"] = None
    assert not regression_verdict(0.2, records(), candidate).accepted


def test_nominal_stratum_regression_cannot_be_averaged_away():
    baseline, candidate = records(), records()
    for i, (left, right) in enumerate(zip(baseline, candidate)):
        left["nominal_group"] = right["nominal_group"] = "rough" if i == 0 else "flat"
        left["tracking_error"] = 0.2
        right["tracking_error"] = 0.3 if i == 0 else 0.1
    verdict = regression_verdict(0.2, baseline, candidate)
    assert not verdict.accepted
    assert any("rough/tracking_error" in reason for reason in verdict.reasons)
    assert not regression_verdict(
        0.2, records(), records(), RegressionBudget(required_nominal_groups=("rough",))
    ).accepted
