"""Sparse, failure-conditioned search for the simulator change that explains a failure.

The primary FCSI rule is **lexicographic**, so no weight trades trajectory error against the
failure event:

1. a candidate ``(mechanism, value)`` is *feasible* only if it (a) makes the logged failure
   event at least ``LR_MIN`` times as likely as the nominal model does, and at least
   ``P_EVENT_MIN`` in absolute terms (a single failure under process noise is a stochastic
   realization: in the unregistered quick world the true physics reproduced its own logged fall
   within the onset tolerance only 23% of the time, the nominal model 1.6%, so an absolute
   threshold such as 0.5 would reject the truth), (b) keeps the successful nominal logs
   successful with probability >= ``P_NOMINAL``, and (c) does not worsen the pre-divergence
   residual fit by more than two standard deviations of that fit's sampling noise;
2. among feasible candidates, fewer mechanisms beat more (singles before pairs);
3. among equally sparse feasible candidates, lower residual NLL over all rows wins; the
   evidence of a mechanism is its best feasible NLL plus a BIC term, turned into posterior
   weights over mechanisms.

If no single mechanism is feasible, pairs of the three most promising singles are tried. If no
pair is feasible either, the answer is **UNEXPLAINED FAILURE**.

Search per mechanism: a 25-point grid over its prior range plus a 9-point refinement around
the best feasible (or best-event) grid point. Derivative-free and exhaustive in one dimension,
so the result does not depend on an optimizer's luck.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ashfall.fcsi import divergence as dv
from ashfall.fcsi import objective as ob
from ashfall.fcsi.mechanisms import BY_NAME, LIBRARY
from ashfall.fcsi.toy_world import MAX_DELAY, Physics

P_EVENT_MIN = 0.1
LR_MIN = 5.0
P_NOMINAL = 0.8
PRE_SD = 2.0
GRID = 25
REFINE = 9
PAIR_GRID = 9
TOP_FOR_PAIRS = 3
EQUIVALENCE_DELTA = 2.3  # evidence gap within which hypotheses are not distinguished (~10:1)
IDENTIFIED_WEIGHT = 0.8


@dataclass
class Candidate:
    mechanisms: tuple
    values: tuple
    p_event: float
    p_nominal: float
    nll_pre: float
    nll_div: float
    feasible: bool

    @property
    def nll(self) -> float:
        return self.nll_pre + self.nll_div

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hypothesis:
    mechanisms: tuple
    best_values: tuple
    feasible_ranges: list
    evidence: float
    weight: float = 0.0
    p_event: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Diagnosis:
    status: str  # identified, ambiguous, unexplained, no_patch_needed, no_failure
    hypotheses: list = field(default_factory=list)
    equivalence_class: list = field(default_factory=list)
    divergence: dict | None = None
    nominal_p_event: float | None = None
    pre_tolerance: float | None = None
    budget: dict | None = None
    candidates: int = 0

    @property
    def top(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    def patch(self, base: Physics) -> Physics:
        if self.top is None:
            return base
        return ob.physics_with(base, **dict(zip(self.top.mechanisms, self.top.best_values)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hypotheses"] = [h.to_dict() for h in self.hypotheses]
        return d


def _split_rows(ev: ob.Evidence, t_div: float | None, cfg) -> tuple:
    lg = ev.failure_log
    anchor = (
        t_div
        if t_div is not None
        else max(0.0, (lg.fail_time if np.isfinite(lg.fail_time) else float(lg.t[-1])) - 1.0)
    )
    cut = int(round(anchor / cfg.dt))
    rows = np.arange(MAX_DELAY, len(lg) - 1)
    return rows[rows < cut], rows[rows >= cut]


def _score(cfg, theta, models, ev, cal, pre_rows, div_rows, start_row, budget):
    pre = sum(
        ob.nll(cfg, models, lg, np.arange(len(lg)), cal.scale, budget=budget)
        for lg in ev.nominal_logs
    )
    pre = pre + ob.nll(cfg, models, ev.failure_log, pre_rows, cal.scale, budget=budget)
    div = ob.nll(cfg, models, ev.failure_log, div_rows, cal.scale, budget=budget)
    pe = ob.event_probability(cfg, theta, models, ev.failure_log, start_row, budget=budget)
    pn = ob.nominal_success(cfg, theta, models, ev.nominal_logs, budget=budget)
    budget.add(0, len(models))
    return pre, div, pe, pn


def event_ok(p: float, p_nominal: float) -> bool:
    return bool(p >= max(P_EVENT_MIN, LR_MIN * p_nominal))


def _candidates(cfg, theta, base, names, value_sets, ev, cal, rows, start_row, tol, pre0, budget):
    combos = list(itertools.product(*value_sets))
    models = [ob.physics_with(base, **dict(zip(names, c))) for c in combos]
    pre, div, pe, pn = _score(cfg, theta, models, ev, cal, rows[0], rows[1], start_row, budget)
    out = []
    for c, a, b, e, n in zip(combos, pre, div, pe, pn):
        feas = bool(event_ok(e, budget.p_nominal_event) and n >= P_NOMINAL and a - pre0 <= tol)
        out.append(
            Candidate(
                tuple(names),
                tuple(float(v) for v in c),
                float(e),
                float(n),
                float(a),
                float(b),
                feas,
            )
        )
    return out


def _refined(m, best: float) -> np.ndarray:
    step = m.span / (GRID - 1)
    return np.clip(np.linspace(best - step, best + step, REFINE), m.low, m.high)


def _hypothesis(cands: list, n_rows: int) -> Hypothesis | None:
    feas = [c for c in cands if c.feasible]
    if not feas:
        return None
    best = min(feas, key=lambda c: c.nll)
    k = len(best.mechanisms)
    ranges = [
        [float(min(c.values[i] for c in feas)), float(max(c.values[i] for c in feas))]
        for i in range(k)
    ]
    evidence = best.nll - math.log(max(best.p_event, 1e-3)) + 0.5 * k * math.log(n_rows)
    return Hypothesis(best.mechanisms, best.values, ranges, float(evidence), p_event=best.p_event)


def identify(
    cfg,
    theta,
    ev: ob.Evidence,
    cal: dv.Calibration,
    *,
    base: Physics | None = None,
    library=LIBRARY,
    use_event: bool = True,
) -> Diagnosis:
    """Failure-conditioned sparse identification (FCSI, primary lexicographic rule)."""
    base = base or Physics.of()
    budget = ob.Budget()
    lg = ev.failure_log
    if lg.outcome == "success":
        return Diagnosis("no_failure", budget=asdict(budget))
    rep = dv.locate(cfg, base, lg, cal)
    budget.add(len(lg), 0)
    start_row = ob.event_start_row(cfg, lg, rep.t_div)
    rows = _split_rows(ev, rep.t_div, cfg)
    pre0, div0, pe0, pn0 = _score(cfg, theta, [base], ev, cal, rows[0], rows[1], start_row, budget)
    n_pre = sum(len(x) for x in ev.nominal_logs) + len(rows[0])
    tol = PRE_SD * 0.5 * math.sqrt(2 * 5 * max(n_pre, 1))
    diag = Diagnosis(
        "unexplained", divergence=rep.to_dict(), nominal_p_event=float(pe0[0]), pre_tolerance=tol
    )
    budget.p_nominal_event = float(pe0[0])
    if pe0[0] > 0.2:
        diag.status = "no_patch_needed"
        diag.budget = asdict(budget)
        return diag
    n_rows = n_pre + len(rows[1])
    singles, all_cands = {}, []
    for m in library:
        grid = np.linspace(m.low, m.high, GRID)
        c1 = _candidates(
            cfg, theta, base, [m.name], [grid], ev, cal, rows, start_row, tol, pre0[0], budget
        )
        pick = min((c for c in c1 if c.feasible), key=lambda c: c.nll, default=None) or max(
            c1, key=lambda c: (c.p_event, -c.nll)
        )
        c2 = _candidates(
            cfg,
            theta,
            base,
            [m.name],
            [_refined(m, pick.values[0])],
            ev,
            cal,
            rows,
            start_row,
            tol,
            pre0[0],
            budget,
        )
        singles[m.name] = c1 + c2
        all_cands += c1 + c2
    if not use_event:
        # Ablation: ignore the event and the feasibility rule; best residual fit wins.
        for cs in singles.values():
            for c in cs:
                c.feasible = True
    hyps = [h for h in (_hypothesis(cs, n_rows) for cs in singles.values()) if h is not None]
    if not hyps:
        ranked = sorted(singles, key=lambda n: -max(c.p_event for c in singles[n]))
        for a, b in itertools.combinations(ranked[:TOP_FOR_PAIRS], 2):
            ma, mb = BY_NAME[a], BY_NAME[b]
            cp = _candidates(
                cfg,
                theta,
                base,
                [a, b],
                [np.linspace(ma.low, ma.high, PAIR_GRID), np.linspace(mb.low, mb.high, PAIR_GRID)],
                ev,
                cal,
                rows,
                start_row,
                tol,
                pre0[0],
                budget,
            )
            all_cands += cp
            h = _hypothesis(cp, n_rows)
            if h is not None:
                hyps.append(h)
    diag.candidates = len(all_cands)
    diag.budget = asdict(budget)
    if not hyps:
        return diag
    hyps.sort(key=lambda h: h.evidence)
    e0 = hyps[0].evidence
    w = np.array([math.exp(-(h.evidence - e0)) for h in hyps])
    w /= w.sum()
    for h, wi in zip(hyps, w):
        h.weight = float(wi)
    diag.hypotheses = hyps
    diag.equivalence_class = [
        list(h.mechanisms) for h in hyps if h.evidence - e0 < EQUIVALENCE_DELTA
    ]
    diag.status = (
        "identified"
        if len(diag.equivalence_class) == 1 and hyps[0].weight >= IDENTIFIED_WEIGHT
        else "ambiguous"
    )
    return diag
