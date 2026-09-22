"""Does recoverability explain repair value? Interpretable models with world fixed effects.

Implements the analysis frozen in ``docs/research/RECOVERABILITY_RETROSPECTIVE_PLAN.md``:
models A to F, leave-one-world-out CV, AIC, within-world Spearman, a two-level
bootstrap for the interior-optimum test, a within-world permutation test for
provenance, and the held-out selection rules. Plain numpy least squares only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import spearmanr

PROVENANCES = ("failed_real", "failed_sim", "successful")


# --------------------------------------------------------------------------- #
# Feature families
# --------------------------------------------------------------------------- #


def linear(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    return x[:, None]


def quadratic(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    return np.column_stack([x, x**2])


def natural_spline(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Natural cubic spline, 3 df: boundary knots at the reference range, interior at tertiles."""
    r = x if ref is None else ref
    knots = np.array([r.min(), *np.quantile(r, [1 / 3, 2 / 3]), r.max()])
    if len(np.unique(knots)) < 4:
        return quadratic(x)

    def d(k):
        return (np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[-1], 0, None) ** 3) / (
            knots[-1] - knots[k]
        )

    return np.column_stack([x, d(0) - d(2), d(1) - d(2)])


FAMILIES: dict[str, Callable] = {
    "linear": linear,
    "quadratic": quadratic,
    "spline": natural_spline,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    continuous: tuple[str, ...]
    provenance: bool = False


MODELS = (
    ModelSpec("A_time", ("lead_s",)),
    ModelSpec("B_distance", ("distance_to_hazard_m",)),
    ModelSpec("C_provenance", (), provenance=True),
    ModelSpec("D_recoverability", ("r_band",)),
    ModelSpec("E_recoverability_provenance", ("r_band",), provenance=True),
    ModelSpec("F_recoverability_time", ("r_band", "lead_s")),
)


def _col(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def features(
    rows: Sequence[dict], spec: ModelSpec, family: str, ref_rows: Sequence[dict] | None = None
) -> np.ndarray:
    """Non-fixed-effect columns (no intercept) for ``rows``; spline knots from ``ref_rows``."""
    ref_rows = rows if ref_rows is None else ref_rows
    cols = []
    for key in spec.continuous:
        cols.append(FAMILIES[family](_col(rows, key), _col(ref_rows, key)))
    if spec.provenance:
        prov = np.array([r["provenance"] for r in rows])
        cols.append(np.column_stack([(prov == p).astype(float) for p in PROVENANCES[1:]]))
    if not cols:
        return np.zeros((len(rows), 0))
    return np.column_stack(cols)


def world_dummies(worlds: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    levels = sorted(set(worlds))
    return np.column_stack([np.array(worlds) == w for w in levels]).astype(float), levels


@dataclass
class Fit:
    beta: np.ndarray  # feature coefficients (fixed effects excluded)
    fe: dict  # world -> intercept
    rss: float
    n: int
    k: int

    @property
    def aic(self) -> float:
        return self.n * np.log(self.rss / self.n) + 2 * self.k


def fit(rows: Sequence[dict], y: np.ndarray, spec: ModelSpec, family: str, ref_rows=None) -> Fit:
    X = features(rows, spec, family, ref_rows)
    D, levels = world_dummies([r["world"] for r in rows])
    Z = np.column_stack([X, D])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ coef
    p = X.shape[1]
    return Fit(
        beta=coef[:p],
        fe=dict(zip(levels, coef[p:])),
        rss=float(resid @ resid),
        n=len(y),
        k=Z.shape[1] + 1,
    )


def predict_centered(
    f: Fit, rows: Sequence[dict], spec: ModelSpec, family: str, ref_rows
) -> np.ndarray:
    """Within-world centred predictions (world intercepts are unknown for a new world)."""
    pred = features(rows, spec, family, ref_rows) @ f.beta
    worlds = np.array([r["world"] for r in rows])
    for w in set(worlds):
        m = worlds == w
        pred[m] -= pred[m].mean()
    return pred


def center_by_world(rows: Sequence[dict], y: np.ndarray) -> np.ndarray:
    worlds = np.array([r["world"] for r in rows])
    out = y.astype(float).copy()
    for w in set(worlds):
        m = worlds == w
        out[m] -= out[m].mean()
    return out


def within_world_spearman(rows, y, pred) -> float:
    worlds = np.array([r["world"] for r in rows])
    vals = []
    for w in sorted(set(worlds)):
        m = worlds == w
        if m.sum() >= 3 and np.ptp(pred[m]) > 0 and np.ptp(y[m]) > 0:
            vals.append(spearmanr(pred[m], y[m]).statistic)
    return float(np.mean(vals)) if vals else float("nan")


def lowo_cv(rows: Sequence[dict], y: np.ndarray, spec: ModelSpec, family: str) -> dict:
    worlds = np.array([r["world"] for r in rows])
    yc = center_by_world(rows, y)
    pred = np.zeros_like(yc)
    for w in sorted(set(worlds)):
        train = [r for r, keep in zip(rows, worlds != w) if keep]
        test = [r for r, keep in zip(rows, worlds == w) if keep]
        f = fit(train, y[worlds != w], spec, family)
        pred[worlds == w] = predict_centered(f, test, spec, family, train)
    sse = float(np.sum((yc - pred) ** 2))
    sst = float(np.sum(yc**2))
    return {
        "rmse": float(np.sqrt(sse / len(yc))),
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "spearman_within_world": within_world_spearman(rows, yc, pred),
    }


def compare_models(rows: Sequence[dict], y: np.ndarray, families=("quadratic", "linear", "spline")):
    out = {}
    for family in families:
        for spec in MODELS:
            if not spec.continuous and family != families[0]:
                continue
            f = fit(rows, y, spec, family)
            fitted = features(rows, spec, family) @ f.beta
            out[f"{spec.name}:{family}"] = {
                "aic": f.aic,
                "rss": f.rss,
                "k": f.k,
                "in_sample_spearman_within_world": within_world_spearman(
                    rows, center_by_world(rows, y), fitted
                ),
                "lowo": lowo_cv(rows, y, spec, family),
                "beta": f.beta.tolist(),
            }
    return out


# --------------------------------------------------------------------------- #
# Interior optimum
# --------------------------------------------------------------------------- #


def quad_shape(beta: np.ndarray, lo: float, hi: float) -> dict:
    b1, b2 = float(beta[0]), float(beta[1])
    vertex = -b1 / (2 * b2) if b2 != 0 else float("nan")
    return {
        "b1": b1,
        "b2": b2,
        "vertex": vertex,
        "slope_lo": b1 + 2 * b2 * lo,
        "slope_hi": b1 + 2 * b2 * hi,
    }


def argmax_quadratic(beta: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> float:
    grid = np.linspace(lo, hi, 1001)
    return float(grid[np.argmax(beta[0] * grid + beta[1] * grid**2)])


def bootstrap_interior(
    rows: Sequence[dict],
    by_seed: Sequence[dict],
    key: str = "r_band",
    *,
    resamples: int = 2000,
    rng_seed: int = 7,
) -> dict:
    """Two-level bootstrap: worlds with replacement, then paired training seeds.

    ``by_seed[i]`` maps training seed -> repair delta for row ``i``. ``R`` is held fixed.
    """
    spec = ModelSpec("D", (key,))
    x = _col(rows, key)
    lo, hi = np.quantile(x, [0.1, 0.9])
    worlds = sorted({r["world"] for r in rows})
    seeds = sorted(by_seed[0])
    idx_by_world = {w: [i for i, r in enumerate(rows) if r["world"] == w] for w in worlds}
    y0 = np.array([np.mean(list(d.values())) for d in by_seed])
    point = quad_shape(fit(rows, y0, spec, "quadratic").beta, lo, hi)
    rng = np.random.default_rng(rng_seed)
    draws = []
    for _ in range(resamples):
        ws = rng.choice(worlds, len(worlds), replace=True)
        ss = rng.choice(seeds, len(seeds), replace=True)
        sub, ys = [], []
        for j, w in enumerate(ws):
            for i in idx_by_world[w]:
                sub.append({**rows[i], "world": f"{w}#{j}"})
                ys.append(np.mean([by_seed[i][s] for s in ss]))
        draws.append(quad_shape(fit(sub, np.array(ys), spec, "quadratic").beta, lo, hi))

    def ci(k):
        v = np.array([d[k] for d in draws])
        v = v[np.isfinite(v)]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    b2_ci, slo, shi = ci("b2"), ci("slope_lo"), ci("slope_hi")
    vertex_ci = ci("vertex")
    interior = bool(
        point["b2"] < 0 and x.min() < point["vertex"] < x.max() and slo[0] > 0 and shi[1] < 0
    )
    return {
        "point": point,
        "p10_p90": [float(lo), float(hi)],
        "observed_range": [float(x.min()), float(x.max())],
        "b2_ci": b2_ci,
        "slope_lo_ci": slo,
        "slope_hi_ci": shi,
        "vertex_ci": vertex_ci,
        "fraction_negative_b2": float(np.mean([d["b2"] < 0 for d in draws])),
        "interior_optimum": interior,
        "high_r_observed": bool(hi >= 0.8),
        "monotone_increasing_significant": bool(slo[0] > 0 and shi[0] > 0),
        "resamples": resamples,
    }


# --------------------------------------------------------------------------- #
# Provenance after recoverability
# --------------------------------------------------------------------------- #


def provenance_permutation(
    rows: Sequence[dict], y: np.ndarray, *, permutations: int = 10_000, rng_seed: int = 11
) -> dict:
    d = ModelSpec("D", ("r_band",))
    e = ModelSpec("E", ("r_band",), provenance=True)
    rss_d = fit(rows, y, d, "quadratic").rss
    f_e = fit(rows, y, e, "quadratic")
    observed = rss_d - f_e.rss
    worlds = np.array([r["world"] for r in rows])
    prov = np.array([r["provenance"] for r in rows])
    rng = np.random.default_rng(rng_seed)
    hits = 0
    for _ in range(permutations):
        perm = prov.copy()
        for w in set(worlds):
            m = np.flatnonzero(worlds == w)
            perm[m] = prov[rng.permutation(m)]
        prow = [{**r, "provenance": p} for r, p in zip(rows, perm)]
        if rss_d - fit(prow, y, e, "quadratic").rss >= observed - 1e-12:
            hits += 1
    return {
        "rss_reduction": float(observed),
        "fraction_of_rss_d": float(observed / rss_d) if rss_d > 0 else float("nan"),
        "p": (hits + 1) / (permutations + 1),
        "coefficients": dict(zip(PROVENANCES[1:], f_e.beta[2:].tolist())),
        "permutations": permutations,
    }


# --------------------------------------------------------------------------- #
# Held-out selection
# --------------------------------------------------------------------------- #


def select(rows: Sequence[dict], key: str, target: float) -> dict:
    """Per world, the row whose ``key`` is closest to ``target`` (ties: first by arm name)."""
    out = {}
    for w in sorted({r["world"] for r in rows}):
        cands = sorted((r for r in rows if r["world"] == w), key=lambda r: r["arm"])
        out[w] = min(cands, key=lambda r: abs(float(r[key]) - target))
    return out


def regret_table(rows: Sequence[dict], choice: dict) -> dict:
    per, regrets, values = {}, [], []
    for w, r in choice.items():
        cands = [c for c in rows if c["world"] == w]
        best = max(cands, key=lambda c: c["repair_value"])
        rand = float(np.mean([c["repair_value"] for c in cands]))
        per[w] = {
            "chosen": r["arm"],
            "chosen_value": r["repair_value"],
            "best": best["arm"],
            "best_value": best["repair_value"],
            "regret": best["repair_value"] - r["repair_value"],
            "random_expected": rand,
        }
        regrets.append(per[w]["regret"])
        values.append(r["repair_value"])
    return {
        "per_world": per,
        "mean_value": float(np.mean(values)),
        "mean_regret": float(np.mean(regrets)),
    }


def out_of_sample(train, y_train, test, y_test, spec: ModelSpec, family="quadratic") -> dict:
    f = fit(train, y_train, spec, family)
    pred = predict_centered(f, test, spec, family, train)
    yc = center_by_world(test, y_test)
    sse, sst = float(np.sum((yc - pred) ** 2)), float(np.sum(yc**2))
    return {
        "r2": 1 - sse / sst if sst > 0 else float("nan"),
        "rmse": float(np.sqrt(sse / len(yc))),
        "spearman_within_world": within_world_spearman(test, yc, pred),
    }
