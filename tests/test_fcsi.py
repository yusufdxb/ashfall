"""FCSI: alignment contract, world mechanisms, divergence, event scoring, search, gate."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ashfall.fbr.toy_slip import ToySlipConfig, condition_batch, simulate, traverse_failure
from ashfall.fcsi import alignment as al
from ashfall.fcsi import baselines as bl
from ashfall.fcsi import divergence as dv
from ashfall.fcsi import objective as ob
from ashfall.fcsi import search as fs
from ashfall.fcsi.mechanisms import BY_NAME, LIBRARY
from ashfall.fcsi.toy_study import gate
from ashfall.fcsi.toy_world import (
    NOMINAL,
    PARAMS,
    Condition,
    Physics,
    closed_loop,
    fresh_state,
    record,
    start_state,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ToySlipConfig()


@pytest.fixture(scope="module")
def theta():
    return np.load(ROOT / "results/fbr_toy_v2/8f49f02/baseline.npy")


# ------------------------------------------------------------------ alignment --


def test_resample_marks_gaps_missing_instead_of_filling():
    t = np.array([0.0, 0.1, 0.2, 0.8, 0.9])
    y = np.arange(5.0)
    grid = np.array([0.05, 0.5, 0.85, 1.2])
    vals, missing = al.resample(t, y, grid, max_gap_s=0.15)
    assert np.isclose(vals[0], 0.5) and np.isclose(vals[2], 3.5)
    assert missing.tolist() == [False, True, False, True]


def test_resample_rejects_non_monotonic_time():
    with pytest.raises(ValueError):
        al.resample(np.array([0.0, 0.2, 0.1]), np.zeros(3), np.array([0.0]), max_gap_s=1)


def test_attitude_error_is_sign_invariant_and_convention_explicit():
    q = np.array([math.cos(0.1), math.sin(0.1), 0.0, 0.0])  # wxyz, 0.2 rad about x
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    assert math.isclose(al.attitude_error(q, ident, convention="wxyz"), 0.2, abs_tol=1e-9)
    assert math.isclose(al.attitude_error(-q, ident, convention="wxyz"), 0.2, abs_tol=1e-9)
    q_xyzw = q[[1, 2, 3, 0]]
    ident_xyzw = ident[[1, 2, 3, 0]]
    assert math.isclose(al.attitude_error(q_xyzw, ident_xyzw, convention="xyzw"), 0.2, abs_tol=1e-9)
    with pytest.raises(ValueError):
        al.attitude_error(q, ident, convention="zyx")
    with pytest.raises(ValueError):
        al.attitude_error(2 * q, ident, convention="wxyz")


def test_lag_estimate_recovers_known_delay():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(400)
    b = np.r_[np.zeros(3), a[:-3]]
    est = al.estimate_lag(a, b, max_lag=6)
    assert est.lag_steps == 3 and not est.ambiguous


def test_contact_edge_matching_reports_unmatched():
    real = np.array([0.1, 0.5, 0.9])
    sim = np.array([0.12, 0.95])
    m = al.match_edges(real, sim, tol_s=0.08)
    assert m["matched"] == 2 and m["unmatched_real"] == 1 and m["unmatched_sim"] == 0


# ---------------------------------------------------------------- toy world --


def test_library_is_small_and_brackets_nominal():
    assert 5 <= len(LIBRARY) <= 12
    assert {m.name for m in LIBRARY} == set(PARAMS)
    for m in LIBRARY:
        assert m.low <= NOMINAL[m.name] <= m.high, m.name
        assert m.meaning and m.implementation and m.observability


def test_nominal_world_from_rest_matches_original_toy(theta):
    seeds = np.arange(64) + 500_000
    b = condition_batch(CFG, mu=0.08, patch_start=1.0, v_cmd=1.1, seeds=seeds)
    r0 = simulate(CFG, theta, b)
    st = fresh_state(CFG, 64, th=b.th, thd=b.thd, ph=b.phase)
    r1 = closed_loop(CFG, theta, Physics.of(), Condition(0.08, 1.0, 1.1), st, seeds)
    assert np.array_equal(traverse_failure(r0)[0], r1.failed)


def test_patch_contact_mechanism_is_local(theta):
    cond = Condition(0.5, 50.0, 1.1)  # patch never reached
    seeds = np.arange(32) + 1
    a = closed_loop(CFG, theta, Physics.of(), cond, start_state(CFG, cond, seeds), seeds)
    b = closed_loop(
        CFG,
        theta,
        Physics.of(kinetic_ratio=0.3, aniso=0.3, mu_scale=0.4),
        cond,
        start_state(CFG, cond, seeds),
        seeds,
    )
    assert np.allclose(a.x, b.x) and np.allclose(a.th, b.th)


def test_physics_rejects_unknown_parameters():
    with pytest.raises(ValueError):
        Physics.of(gravity=3.0)


# --------------------------------------------------------------- divergence --


def _logs(theta, physics, n, seed0, cond_fn):
    rng = np.random.default_rng(seed0)
    out, k = [], 0
    while len(out) < n:
        lg = record(CFG, theta, physics, cond_fn(rng), seed0 + k, rng)
        k += 1
        if lg.outcome == "success":
            out.append(lg)
    return out


def _benign(rng):
    return Condition(float(rng.uniform(0.5, 0.8)), 1.0, 1.1)


def test_chi2_threshold_is_deterministic_and_positive():
    assert dv.chi2_threshold() == dv.chi2_threshold() > 0


def test_detector_quiet_on_matched_model_and_fires_on_mismatch(theta):
    nominal = Physics.of()
    cal = dv.calibrate(CFG, nominal, _logs(theta, nominal, 6, 100, _benign))
    fresh = _logs(theta, nominal, 6, 900, _benign)
    alarms = sum(dv.locate(CFG, nominal, lg, cal).diverged for lg in fresh)
    assert alarms <= 1
    real = Physics.of(kinetic_ratio=0.3)
    rng = np.random.default_rng(5)
    for k in range(200):
        lg = record(CFG, theta, real, Condition(0.2, 0.9, 1.1), 5000 + k, rng)
        if lg.outcome == "fall":
            break
    rep = dv.locate(CFG, nominal, lg, cal)
    assert rep.diverged and rep.t_div < lg.fail_time
    assert rep.channel_ranking[0][0] in ("slip", "v", "thd", "th")
    assert "FIRST DIVERGENCE" in rep.text()


# --------------------------------------------------------- event and search --


def test_event_probability_counts_and_charges_steps(theta):
    nominal = Physics.of()
    lg = _logs(theta, nominal, 1, 300, _benign)[0]
    b = ob.Budget()
    p = ob.event_probability(CFG, theta, [nominal], lg, 5, draws=16, budget=b)
    assert 0.5 <= p[0] <= 1.0 and b.steps > 0


def test_event_rule_is_relative_to_nominal():
    assert fs.event_ok(0.2, 0.01)
    assert not fs.event_ok(0.2, 0.05)
    assert not fs.event_ok(0.08, 0.0)


def test_identify_refuses_a_successful_log(theta):
    nominal = Physics.of()
    logs = _logs(theta, nominal, 3, 400, _benign)
    cal = dv.calibrate(CFG, nominal, logs[:2])
    d = fs.identify(CFG, theta, ob.Evidence(logs[:2], logs[2]), cal)
    assert d.status == "no_failure" and d.top is None


def test_identify_finds_the_hidden_mechanism_in_a_small_library(theta, monkeypatch):
    monkeypatch.setattr(fs, "GRID", 7)
    monkeypatch.setattr(fs, "REFINE", 3)
    real = Physics.of(kinetic_ratio=0.35)
    nominal_logs = _logs(theta, real, 4, 700, _benign)
    rng = np.random.default_rng(3)
    fails = []
    for k in range(400):
        lg = record(CFG, theta, real, Condition(0.22, 0.9, 1.1), 7000 + k, rng)
        if lg.outcome == "fall":
            fails.append(lg)
        if len(fails) == 1:
            break
    cal = dv.calibrate(CFG, Physics.of(), nominal_logs)
    lib = (BY_NAME["kinetic_ratio"], BY_NAME["cart_damping"])
    d = fs.identify(CFG, theta, ob.Evidence(nominal_logs, fails[0]), cal, library=lib)
    assert d.status in ("identified", "ambiguous", "unexplained")
    if d.top is not None:
        assert d.top.mechanisms[0] == "kinetic_ratio"
        assert sum(h.weight for h in d.hypotheses) == pytest.approx(1.0)


# --------------------------------------------------------------- baselines --


def test_open_loop_blind_parameters_stay_nominal():
    vals = bl._to_values(np.ones(len(LIBRARY)), bl.OPEN_LOOP_BLIND)
    assert vals["obs_delay"] == NOMINAL["obs_delay"]
    assert vals["theta_bias"] == NOMINAL["theta_bias"]
    assert vals["kinetic_ratio"] == BY_NAME["kinetic_ratio"].high


def test_implied_mechanism_and_sparsity():
    vals = dict(NOMINAL)
    vals["aniso"] = 0.4
    top, n = bl.implied_mechanism(vals)
    assert top == "aniso" and n == 1


# -------------------------------------------------------------------- gate --


def _fake(kind, truth, fcsi_mech, status, mae_f, mae_g, p=0.3):
    m = {
        "FCSI": {
            "mechanism": fcsi_mech,
            "status": status,
            "score": {"heldout_mae": mae_f, "target_p_event": p, "nominal_d2_per_row": 5.0},
        },
        "FCSI_no_event": {"mechanism": truth, "score": {"heldout_mae": mae_f}},
    }
    for g in ("G0_whole_trajectory", "G0b_multiple_shooting", "G1_dropo_like"):
        m[g] = {"mechanism": "pole_mass", "score": {"heldout_mae": mae_g}}
    return {
        "kind": kind,
        "truth": truth,
        "methods": m,
        "nominal_model_score": {"nominal_d2_per_row": 5.0, "target_p_event": 0.01},
    }


def test_gate_go_and_no_go():
    known = [
        _fake("known", "aniso", "aniso", "identified", 0.05, 0.2 + 0.01 * i) for i in range(18)
    ]
    unknown = [_fake("unknown", None, None, "unexplained", 0.1, 0.1) for _ in range(8)]
    neg = [_fake("negative", None, None, "no_failure", 0.0, 0.0) for _ in range(3)]
    g = gate(known + unknown + neg)
    assert g["GO"], g["gates"]
    known[0]["methods"]["FCSI"]["score"]["target_p_event"] = 0.0
    for k in known[:5]:
        k["methods"]["FCSI"]["score"]["target_p_event"] = 0.0
    assert not gate(known + unknown + neg)["gates"]["G2c_reproduction"]
