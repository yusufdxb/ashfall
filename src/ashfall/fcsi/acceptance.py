"""Causal acceptance checks for a proposed simulator explanation.

An explanation from :func:`ashfall.fcsi.search.identify` is accepted only when all six hold:

1. **delivered**: the patched simulator really carries the proposed parameter values (readback);
2. **pre-failure preservation**: residual fit before the divergence is within tolerance and the
   successful nominal logs stay successful;
3. **divergence alignment**: with the patch, the detector's alarm on the failure log disappears
   or moves at least ``DIV_SHIFT_S`` later (the patch explains the divergence, not only the fall);
4. **failure reproduction**: the patch makes the logged failure event at least 5 times as likely
   as the nominal model does and at least 0.1 (the search's feasibility rule, re-checked);
5. **removal**: with the patch removed (nominal model) the event probability is <= 0.2;
6. **held-out replay**: on the separate validation failure, not used for fitting, the patch's
   event probability is >= 0.1 and at least ``VALIDATION_LR`` (2) times the nominal model's (a
   consistency check, looser than the discovery rule's factor 5).

A diagnosis that fails any check becomes ``unexplained`` (the evidence does not support it).
Only an accepted diagnosis whose top hypothesis is alone in its equivalence class, with
posterior weight >= 0.8, is called ``identified``; otherwise it is a set of
**failure-compatible simulator hypotheses** (``ambiguous``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ashfall.fcsi import divergence as dv
from ashfall.fcsi import objective as ob
from ashfall.fcsi.search import P_NOMINAL, Diagnosis, event_ok
from ashfall.fcsi.toy_world import Physics

DIV_SHIFT_S = 0.25
REMOVAL_MAX = 0.2
VALIDATION_LR = 2.0


@dataclass
class Acceptance:
    delivered: bool
    pre_preserved: bool
    divergence_aligned: bool
    failure_reproduced: bool
    removal: bool
    held_out: bool | None
    patched_alarm_s: float | None
    validation_p_event: float | None

    @property
    def accepted(self) -> bool:
        core = (
            self.delivered,
            self.pre_preserved,
            self.divergence_aligned,
            self.failure_reproduced,
            self.removal,
        )
        return all(core) and self.held_out is not False

    def to_dict(self) -> dict:
        return {**asdict(self), "accepted": self.accepted}


def check(
    cfg, theta, diag: Diagnosis, ev: ob.Evidence, cal: dv.Calibration, base: Physics | None = None
) -> Acceptance | None:
    if diag.top is None:
        return None
    base = base or Physics.of()
    top = diag.top
    patched = diag.patch(base)
    delivered = all(
        abs(patched.get(n) - v) < 1e-12 for n, v in zip(top.mechanisms, top.best_values)
    )
    lg = ev.failure_log
    rep0 = dv.locate(cfg, base, lg, cal)
    rep1 = dv.locate(cfg, patched, lg, cal)
    aligned = (not rep1.diverged) or (
        rep0.diverged and rep1.alarm_time >= rep0.alarm_time + DIV_SHIFT_S
    )
    start = ob.event_start_row(cfg, lg, rep0.t_div)
    p_patch = float(ob.event_probability(cfg, theta, [patched], lg, start)[0])
    p_base = float(ob.event_probability(cfg, theta, [base], lg, start)[0])
    pn = float(ob.nominal_success(cfg, theta, [patched], ev.nominal_logs)[0])
    held, pv = None, None
    if ev.validation_log is not None:
        rv = dv.locate(cfg, base, ev.validation_log, cal)
        sv = ob.event_start_row(cfg, ev.validation_log, rv.t_div)
        pv = float(ob.event_probability(cfg, theta, [patched], ev.validation_log, sv)[0])
        pv0 = float(ob.event_probability(cfg, theta, [base], ev.validation_log, sv)[0])
        held = bool(pv >= 0.1 and pv >= VALIDATION_LR * pv0)
    return Acceptance(
        delivered=delivered,
        pre_preserved=pn >= P_NOMINAL,
        divergence_aligned=bool(aligned),
        failure_reproduced=event_ok(p_patch, p_base),
        removal=p_base <= REMOVAL_MAX,
        held_out=held,
        patched_alarm_s=rep1.alarm_time,
        validation_p_event=pv,
    )


def finalize(diag: Diagnosis, acc: Acceptance | None) -> str:
    """Final status after acceptance: identified, ambiguous, unexplained, or a no-patch state."""
    if diag.status in ("no_failure", "no_patch_needed", "unexplained"):
        return diag.status
    if acc is None or not acc.accepted:
        return "unexplained"
    return diag.status
