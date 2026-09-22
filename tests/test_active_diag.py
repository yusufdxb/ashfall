"""Active diagnosis: likelihood, probe space, safety filter, information, loop, decisions, gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ashfall.active_diag import information as inf
from ashfall.active_diag import loop as lp
from ashfall.active_diag import posterior as po
from ashfall.active_diag import safety as sf
from ashfall.active_diag import study as st
from ashfall.active_diag.prediction import execute, run
from ashfall.active_diag.probe_space import MAX_AMPLITUDE, Probe, probe_grid
from ashfall.fbr.toy_slip import ToySlipConfig
from ashfall.fcsi.toy_world import Condition, Physics, record

ROOT = Path(__file__).resolve().parents[1]
CFG = ToySlipConfig()


@pytest.fixture(scope="module")
def theta():
    return np.load(ROOT / "results/fbr_toy_v2/8f49f02/baseline.npy")


def _log(theta, physics, seed, mu=0.5, meas=1.0):
    return record(
        CFG, theta, physics, Condition(mu, 0.9, 1.1), seed, np.random.default_rng(seed), meas=meas
    )


def test_action_channel_is_exact_under_the_true_sensing_model(theta):
    for truth in (Physics.of(), Physics.of(obs_delay=1.0), Physics.of(theta_bias=0.03)):
        lg = _log(theta, truth, 77, meas=0.0)
        rows = po.rows_of(lg)
        r = po.action_residuals(CFG, theta, [truth, Physics.of(obs_delay=2.0)], lg, rows)
        assert np.abs(r[0]).max() < 1e-9 and np.abs(r[1]).max() > 1.0


def test_hypothesis_grid_contains_none_and_nominal_values():
    hyps = po.hypothesis_grid()
    assert hyps[0][0] == po.NONE
    assert ("kinetic_ratio", 0.85) in hyps and ("action_delay", 0.0) in hyps


def test_weights_normalize_and_temperature_softens():
    hyps = [("none", np.nan), ("a", 1.0), ("a", 2.0), ("b", 1.0)]
    ll = np.array([-100.0, -10.0, -12.0, -50.0])
    sharp, soft = po.weights(hyps, ll, 1.0), po.weights(hyps, ll, 100.0)
    assert sharp.top == "a" and sharp.weights.sum() == pytest.approx(1.0)
    assert soft.top_weight < sharp.top_weight
    assert sharp.map_value["a"] == 1.0


def test_probe_space_is_small_bounded_and_interpretable():
    grid = probe_grid()
    assert len(grid) == 81 and len({p.name for p in grid}) == 81
    assert all(abs(p.amplitude) <= MAX_AMPLITUDE for p in grid)
    pulse = Probe(0.4, 0.8, "pulse", 8.0, 0.0)
    f = pulse.force(np.zeros(3), np.array([0.0, 1.1, 2.0]))
    assert f.tolist() == [0.0, 8.0, 0.0]


def test_safety_filter_rejects_an_oversized_probe(theta):
    big = Probe(0.4, 0.8, "pulse", 20.0, 0.0)
    v, _ = sf.check_many(CFG, theta, [big], {"none": Physics.of()})
    assert not v[big].admissible


def test_gentle_probe_is_admissible_under_nominal_and_reproducible(theta):
    p = Probe(0.5, 0.6, "none", 0.0, 0.0)
    v, _ = sf.check_many(CFG, theta, [p], {"none": Physics.of()})
    assert v[p].admissible
    a = run(CFG, theta, [Physics.of()] * 2, p, np.array([1, 2]))
    b = run(CFG, theta, [Physics.of()] * 2, p, np.array([1, 2]))
    assert np.array_equal(a.x, b.x)


def test_execute_reports_safety_facts(theta):
    lg, u, facts = execute(
        CFG, theta, Physics.of(), Probe(0.5, 0.6, "none", 0.0, 0.0), 5, np.random.default_rng(0)
    )
    assert set(facts) >= {"fell", "aborted", "max_tilt", "violation"}
    assert len(u) == len(lg)


def test_information_prior_floor():
    pi = inf.prior({"a": 1.0, "none": 0.0}, ["none", "a"])
    assert pi.sum() == pytest.approx(1.0) and pi.min() > 0


def test_decide_statuses():
    base = {
        "method": "active",
        "status_hint": None,
        "top": "aniso",
        "top_weight": 0.95,
        "unknown_stat": {"peak": 5.0},
    }
    assert lp.decide(base, 10.0) == "KNOWN"
    assert lp.decide({**base, "unknown_stat": {"peak": 50.0}}, 10.0) == "UNKNOWN"
    assert lp.decide({**base, "top_weight": 0.5}, 10.0) == "UNRESOLVED"
    assert lp.decide({**base, "status_hint": "no_admissible_probe"}, 10.0) == "ABORT"
    assert lp.decide({**base, "method": "passive", "unknown_stat": {"peak": 1e9}}, 1.0) == "KNOWN"
    assert lp.correct(base, "KNOWN", "known", "aniso")
    assert not lp.correct(base, "KNOWN", "unknown", None)


def test_specs_split_dev_and_eval_without_overlap():
    dev, ev = st.specs("dev"), st.specs("eval")
    assert not {s["gidx"] for s in dev} & {s["gidx"] for s in ev}
    assert {s["world"] for s in dev if s["kind"] == "unknown"} == set(st.DEV_UNKNOWN)
    assert {s["world"] for s in ev if s["kind"] == "unknown"} == set(st.EVAL_UNKNOWN)
    assert not set(st.DEV_UNKNOWN) & set(st.EVAL_UNKNOWN)
    fcsi_idx = set(range(0, 29)) | {900}
    assert not ({s["gidx"] for s in dev + ev} & fcsi_idx)


def test_threshold_is_the_largest_known_dev_statistic():
    runs = [{"unknown_stat": {"peak": x}} for x in (3.0, 9.0, 4.0)]
    assert st.threshold_from(runs) == 9.0


def _fake(kind, truth, top, stats, weight=0.95, probes=1, viol=0):
    runs = {}
    for m, s in stats.items():
        runs[m] = {
            "method": m,
            "status_hint": None,
            "top": top,
            "top_weight": weight,
            "unknown_stat": {"peak": s},
            "n_probes": 0 if m == "passive_abstain" else probes,
            "safety_violations": viol,
            "entropy": 0.1,
            "budget": {"steps": 1},
        }
    return {"kind": kind, "truth": truth, "runs": runs}


def test_gate_requires_open_set_gain_over_passive():
    thr = {m: 10.0 for m in st.METHODS}
    known = [_fake("known", "aniso", "aniso", {m: 1.0 for m in st.METHODS}) for _ in range(30)]
    unk = [
        _fake(
            "unknown",
            None,
            "aniso",
            {"passive_abstain": 1.0, "fixed": 1.0, "random": 1.0, "active": 99.0},
        )
        for _ in range(12)
    ]
    g = st.gate(known + unk, thr)
    assert g["GO"], g["gates"]
    unk2 = [_fake("unknown", None, "aniso", {m: 99.0 for m in st.METHODS}) for _ in range(12)]
    assert not st.gate(known + unk2, thr)["gates"]["A2_unknown_gain"]


def test_safety_models_include_map_value_when_weights_tie():
    from ashfall.active_diag import probe_search as ps

    hyps = [("none", np.nan)] + [("aniso", v) for v in (0.3, 0.6, 0.9, 1.2)]
    ll = np.array([-1e6, -5.0, -5.0, -5.0, -9.0])
    w = po.weights(hyps, ll, 1.0)
    models, labels, set_hyps = ps.safety_models(Physics.of(), w)
    assert f"aniso@{w.map_value['aniso']:.4g}" in models
    assert len([k for k in models if k.startswith("aniso@")]) == 2
