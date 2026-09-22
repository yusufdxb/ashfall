"""The precursor time-sweep study: start-state extraction and the registered statistics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ashfall.fbr import time_sweep as ts
from ashfall.fbr.capsule import seed_before_position
from ashfall.fbr.toy_slip import ToySlipConfig


@pytest.fixture(scope="module")
def w1_capsule():
    from ashfall.capsule import FailureCapsule

    return FailureCapsule.load(ts.V2_DIR / "capsule_W1.json")


def test_offsets_are_the_registered_ones():
    assert ts.FIXED == ("d2", "d1.5", "d1", "d0.75", "d0.5", "d0.25", "T0")
    assert ts.PRE_FAILURE == ts.FIXED[:-1]
    assert set(ts.HIDDEN).isdisjoint(ts.DISCOVERY)
    assert set(ts.STAGE1_SEEDS).isdisjoint(ts.STAGE2_SEEDS)


def test_offset_rows_precede_onset_in_order(w1_capsule):
    onset = w1_capsule.failure_onset_index
    rows = [ts.offset_row(w1_capsule, k) for k in ts.FIXED]
    assert rows == sorted(rows) and len(set(rows)) == len(rows)
    assert rows[-1] == onset - 1  # T0 is the last pre-onset frame, never the onset frame
    t_onset = w1_capsule.frames[onset].timestamp_s
    for k, r in zip(ts.PRE_FAILURE, rows):
        lead = t_onset - w1_capsule.frames[r].timestamp_s
        assert float(k[1:]) - 1e-9 <= lead < float(k[1:]) + 0.02 + 1e-9


def test_entry_row_is_toy_v2_arm_d(w1_capsule):
    frozen = json.loads((ts.V2_DIR / "result.json").read_text())["capsules"]["W1"]
    pt = seed_before_position(w1_capsule, ts.DISCOVERY["W1"]["patch_start"], max_lead_s=2.0)
    assert pt.row == frozen["seed_row"]


def test_frozen_v2_capsules_meet_the_onset_rule():
    from ashfall.capsule import FailureCapsule

    for w in ts.DISCOVERY:
        assert ts._onset_ok(FailureCapsule.load(ts.V2_DIR / f"capsule_{w}.json"))


def test_baseline_hash_is_checked():
    theta, sha = ts.load_baseline()
    assert sha == ts.BASELINE_SHA and theta.ndim == 1


def test_holm_is_monotone_and_capped():
    adj = ts.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.09)
    assert adj["b"] == pytest.approx(0.09)  # monotone: never below a smaller p's adjustment
    assert adj["d"] == pytest.approx(0.5)
    assert all(v <= 1.0 for v in adj.values())


def test_omnibus_detects_structure_and_not_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.02, (12, 7))
    assert ts.omnibus(noise, permutations=2000)["p"] > 0.05
    shifted = noise + np.array([0, 0, 0, -0.1, 0, 0, 0])
    assert ts.omnibus(shifted, permutations=2000)["p"] < 0.01


def _eff(mean, p):
    return {"mean": mean, "p_exact_two_sided": p}


def _means(values):
    return {f"real_{k}": v for k, v in zip(ts.FIXED, values)}


def test_classify_rules_in_registered_order():
    base = _means([0.40, 0.35, 0.30, 0.32, 0.38, 0.42, 0.45])
    base["real_entry"] = 0.36
    eff = {
        "real_entry-real_d1": _eff(0.06, 0.001),
        "real_d1-real_T0": _eff(-0.15, 0.001),
        "real_d1-real_d2": _eff(-0.10, 0.001),
    }
    assert ts.classify(base, eff, 0.2)["class"] == "A"
    assert ts.classify(base, eff, 0.001)["class"] == "C"
    unresolved = dict(eff, **{"real_d1-real_d2": _eff(-0.01, 0.4)})
    assert ts.classify(base, unresolved, 0.001)["class"] == "A"
    t0_best = _means([0.40, 0.35, 0.30, 0.32, 0.38, 0.42, 0.20])
    t0_best["real_entry"] = 0.5
    assert ts.classify(t0_best, eff, 0.001)["class"] == "D"
    entry_best = dict(base, real_entry=0.25)
    e2 = dict(eff, **{"real_entry-real_d1": _eff(-0.05, 0.01)})
    assert ts.classify(entry_best, e2, 0.001)["class"] == "E"
    early = _means([0.20, 0.35, 0.30, 0.32, 0.38, 0.42, 0.45])
    early["real_entry"] = 0.36
    e3 = {"real_entry-real_d2": _eff(0.16, 0.001)}
    assert ts.classify(early, e3, 0.001)["class"] == "B"


def test_replay_sampler_differs_only_in_seed_id():
    cfg = ToySlipConfig()
    frozen = json.loads((ts.V2_DIR / "result.json").read_text())["capsules"]["W1"]["boundary"][0]
    from ashfall.fbr.boundary import BoundaryEstimate

    b = BoundaryEstimate(
        frozen["seed_id"],
        frozen["seed_source"],
        frozen["mu_boundary"],
        tuple(frozen["mu_interval"]),
        frozen["censoring"],
        tuple(tuple(r) for r in frozen["raw"]),
    )
    from ashfall.fbr.toy_slip import ToySeed

    s1 = ToySeed("x:1", 0, 0, 0, 0, 0, 1, 1)
    s2 = ToySeed("x:2", 0.5, 1, 0, 0, 0, 1, 1)
    d1 = ts.replay_sampler(b, s1, rng_seed=7, cfg=cfg).sample(50)
    d2 = ts.replay_sampler(b, s2, rng_seed=7, cfg=cfg).sample(50)
    assert [(d.kind, d.mu, d.noise_seed) for d in d1] == [(d.kind, d.mu, d.noise_seed) for d in d2]
