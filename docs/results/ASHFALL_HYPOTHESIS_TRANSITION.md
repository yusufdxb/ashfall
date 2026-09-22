# Ashfall hypothesis transition (after two preregistered NO-GOs)

Status: **record of a change of question.** This file adds nothing to, and changes nothing in,
the two frozen negative results. It exists so that nobody reads the next study as a rescue of
the old ones.

Frozen records this note points to, unchanged and hash-checked by `tests/test_frozen_results.py`
and `tests/test_recoverability_history.py`:

| record | where | commit |
|---|---|---|
| FBR toy v1 + v2 negative result | `docs/results/FBR_TOY_NEGATIVE_RESULT.md` | `08adcff`, frozen at `48c8c71` |
| toy v2 preregistration | `docs/research/TOY_V2_PREREGISTRATION.md` | `8f49f02` |
| precursor sweep preregistration | `docs/research/PRECURSOR_SWEEP_PREREGISTRATION.md` | `3205f02` |
| precursor sweep result | `docs/results/PRECURSOR_SWEEP_RESULT.md` | `dfec130` |
| branch holding that history | `research/ashfall-failure-boundary-replay` | tip `dfec130` (not rewritten) |

## Hypothesis 1

A real failure state is especially valuable: a real failure defines a local simulation
curriculum (Failure-Boundary Replay, FBR) that beats broad domain randomization.

Result: **REJECTED.** Toy v2 (preregistered, `8f49f02`) returned NO-GO. FBR lost to broad DR
by 2.7 pp held-out failure (p 0.105, not significant, wrong direction for the claim), degraded
nominal success by 6.5 pp against a 2 pp margin, and an ordinary patch-entry state from a
successful rollout beat the failure's own seed by 3.6 pp (p 0.003).

## Hypothesis 2

A particular pre-failure offset along the real failed trajectory is especially valuable.

Result: **REJECTED as a provenance claim.** The preregistered sweep (`3205f02`) found real
structure in replay start time (interior optimum at T-1.5 s, 13.5 pp better than onset replay
on hidden worlds, better than broad DR and ADR, nominal non-degradation passed). But at the same
time before the hazard, a simulator-found failure state (+0.5 pp, p 0.50), a successful
trajectory's state at the same position (+0.2 pp, p 0.83) and plain nominal-entry replay
(+0.0 pp, p 0.997) performed the same. The real failure added no demonstrated value. Gates G2
and G5 failed; the decision was NO-GO and no Isaac Lab or GO2 compute was spent.

## New observation (from Hypothesis 2's diagnostics, not a tested claim)

Replay start time strongly affects repair effectiveness, and the baseline policy's
recoverability from a restored state collapses to zero 0.26 to 0.52 s before failure onset on
every failure examined (real and simulated). Replays started after the collapse carry no
learning signal in this toy.

## New hypothesis (H3)

Repair value may be governed by the **recoverability** of the replay start state under the
current policy, rather than by real-failure provenance or by time before failure.

- Recoverability: `R_pi(s)`, the probability that the frozen policy `pi` completes the task when
  restarted from state `s`, under a declared distribution of the remaining uncertainty.
- Repair value: `V_repair(s)`, the reduction in held-out failure of a policy repaired by
  training from (a neighborhood of) `s`, relative to the unrepaired policy, at a fixed repair
  budget.
- The question: what is the relationship between `R_pi(s)` and `V_repair(s)`, and does `R`
  explain `V_repair` better than time-to-failure, distance to the hazard, or provenance?

**This is a NEW hypothesis, motivated by the two negative results above.** It was not
preregistered previously. The discovery data that suggested it (precursor stage 1, worlds
W1 to W6) cannot also confirm it. Any confirmatory claim needs a new preregistration frozen
before new training on new worlds and seeds.

## What this transition does not do

- It does not reinterpret either NO-GO as support for FBR.
- It does not loosen, re-run or re-score any old gate.
- It does not claim the T-1.5 s optimum transfers beyond this toy.
- It does not claim recoverability-guided selection works; that is what the next steps test,
  and the answer may be negative.
