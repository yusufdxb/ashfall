"""Recoverability estimator, dataset separation and analysis rules."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from ashfall.fbr.toy_slip import ToySeed, ToySlipConfig, seed_batch, simulate
from ashfall.recoverability import analysis as an
from ashfall.recoverability.dataset import EVAL_KEYS, candidate_states, repair_outcomes
from ashfall.recoverability.estimator import (
    STREAM_BASE,
    ContextDistribution,
    estimate,
    noise_seeds,
    stream_for,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "results/precursor_toy/3205f02/stage1.json"
STAGE2 = ROOT / "results/precursor_toy/3205f02/stage2/stage2.json"
CFG = ToySlipConfig()
SEED = ToySeed("unit:s", 0.0, 1.0, 0.01, 0.0, 0.5, 1.1, 1.0)
DIST = ContextDistribution(
    "unit", 0.02, 0.2, (("phase", 0.0), ("th", 0.005), ("thd", 0.02), ("v", 0.02))
)


@pytest.fixture(scope="module")
def theta():
    return np.load(ROOT / "results/fbr_toy_v2/8f49f02/baseline.npy")


def test_state_restoration_is_exact_without_jitter(theta):
    zero = {"th": 0.0, "thd": 0.0, "v": 0.0, "phase": 0.0}
    b = seed_batch([SEED] * 3, np.full(3, 0.1), np.array([1, 2, 3]), noise=zero)
    assert np.all(b.x == SEED.x) and np.all(b.v == SEED.v) and np.all(b.th == SEED.th)
    r1 = simulate(CFG, theta, b, record_trace=True).trace["x"]
    r2 = simulate(CFG, theta, b, record_trace=True).trace["x"]
    assert np.array_equal(r1, r2)


def test_estimate_is_deterministic_and_counts_steps(theta):
    a = estimate(CFG, theta, SEED, DIST, n=64)
    b = estimate(CFG, theta, SEED, DIST, n=64)
    assert a == b
    assert a.sim_steps == 64 * CFG.steps
    assert 0 <= a.successes <= 64 and a.r == a.successes / 64


def test_repeat_uses_an_independent_stream(theta):
    assert stream_for("s", "d", 0) != stream_for("s", "d", 1)
    a = noise_seeds(stream_for("s", "d", 0), 64)
    b = noise_seeds(stream_for("s", "d", 1), 64)
    assert not set(a) & set(b)


def test_wilson_interval_brackets_the_estimate(theta):
    e = estimate(CFG, theta, SEED, DIST, n=128)
    assert e.ci_low <= e.r <= e.ci_high
    assert 0 <= e.ci_low and e.ci_high <= 1
    assert e.ci_high - e.ci_low < 0.2


def test_continuation_seeds_never_touch_evaluation_seeds():
    ev = set(range(500_000, 500_256))
    for sid in ("a", "b", "discovery:W1:real_T0"):
        s = noise_seeds(stream_for(sid, "band_0.1000"), 1024)
        assert s.min() >= STREAM_BASE and not ev & set(s.tolist())


def test_estimator_cannot_see_world_or_outcome():
    params = set(inspect.signature(estimate).parameters)
    assert params == {"cfg", "theta", "seed", "dist", "n", "repeat"}


def test_context_distribution_validates():
    with pytest.raises(ValueError):
        ContextDistribution("bad", 0.3, 0.1, ())


def test_candidates_carry_no_outcomes_and_have_labels():
    s1 = candidate_states(STAGE1, "discovery")
    s2 = candidate_states(STAGE2, "hidden")
    assert len(s1) == 144 and len(s2) == 36
    for c in s1 + s2:
        assert not EVAL_KEYS & set(c.to_dict())
        assert c.provenance in an.PROVENANCES
        assert not math.isnan(c.lead_s)
    assert {c.stage for c in s2} == {"hidden"}
    assert not {c.world for c in s1} & {c.world for c in s2}


def test_repair_value_sign_convention():
    o = repair_outcomes(STAGE1)
    r = o[("W1", "real_d1.5")]
    assert math.isclose(r["repair_value"], r["baseline_failure"] - r["held_out_failure"])


# ---------------------------------------------------------------- analysis --


def _synthetic(shape, rng_seed=0, worlds=6, per=20):
    rng = np.random.default_rng(rng_seed)
    rows, by_seed = [], []
    for w in range(worlds):
        for i in range(per):
            r = rng.uniform(0, 1)
            base = shape(r) + 0.1 * w
            deltas = {s: base + rng.normal(0, 0.02) for s in range(12)}
            rows.append(
                {
                    "world": f"W{w}",
                    "r_band": r,
                    "arm": f"a{i}",
                    "provenance": an.PROVENANCES[i % 3],
                    "lead_s": rng.uniform(0, 2),
                    "repair_value": float(np.mean(list(deltas.values()))),
                }
            )
            by_seed.append(deltas)
    return rows, by_seed


def test_interior_optimum_detected_on_inverted_u():
    rows, by_seed = _synthetic(lambda r: -((r - 0.45) ** 2))
    out = an.bootstrap_interior(rows, by_seed, resamples=200)
    assert out["interior_optimum"]
    assert 0.38 < out["point"]["vertex"] < 0.52


def test_interior_optimum_rejected_on_monotone():
    rows, by_seed = _synthetic(lambda r: 0.3 * r)
    out = an.bootstrap_interior(rows, by_seed, resamples=200)
    assert not out["interior_optimum"]
    assert out["monotone_increasing_significant"]


def test_provenance_permutation_null_and_effect():
    rows, _ = _synthetic(lambda r: -((r - 0.5) ** 2))
    y = np.array([r["repair_value"] for r in rows])
    assert an.provenance_permutation(rows, y, permutations=300)["p"] > 0.05
    y2 = y + np.array([0.05 if r["provenance"] == "failed_real" else 0 for r in rows])
    assert an.provenance_permutation(rows, y2, permutations=300)["p"] < 0.05


def test_lowo_prefers_true_predictor():
    rows, _ = _synthetic(lambda r: -((r - 0.5) ** 2))
    y = np.array([r["repair_value"] for r in rows])
    d = an.lowo_cv(rows, y, an.MODELS[3], "quadratic")
    a = an.lowo_cv(rows, y, an.MODELS[0], "quadratic")
    assert d["rmse"] < a["rmse"] and d["r2"] > 0.8


def test_selection_and_regret():
    rows = [
        {"world": "W", "arm": "x", "r_band": 0.1, "repair_value": 0.0},
        {"world": "W", "arm": "y", "r_band": 0.5, "repair_value": 0.3},
        {"world": "W", "arm": "z", "r_band": 0.9, "repair_value": 0.1},
    ]
    choice = an.select(rows, "r_band", 0.45)
    assert choice["W"]["arm"] == "y"
    t = an.regret_table(rows, choice)
    assert t["mean_regret"] == 0.0
    assert math.isclose(t["per_world"]["W"]["random_expected"], 0.4 / 3)
