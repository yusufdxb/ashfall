# Ashfall hypothesis transition: from replaying failures to diagnosing the simulator

Status: **record of a change of question.** It changes nothing in the three frozen negative
results below, all of which stay byte-identical and hash-tested (`tests/test_frozen_results.py`,
`tests/test_recoverability_history.py`).

| record | where | commit |
|---|---|---|
| FBR toy v1 + v2 negative result | `docs/results/FBR_TOY_NEGATIVE_RESULT.md` | frozen at `48c8c71` |
| precursor sweep (preregistered `3205f02`) | `docs/results/PRECURSOR_SWEEP_RESULT.md` | `dfec130` |
| recoverability retrospective (plan `2227de1`) | `docs/results/RECOVERABILITY_RETROSPECTIVE_RESULT.md` | `61bb95d` |
| branches holding that history, not rewritten | `research/ashfall-failure-boundary-replay`, `research/ashfall-recoverability` | `dfec130`, `61bb95d` |

## Previous hypothesis 1

Real failure states are unusually valuable replay states (Failure-Boundary Replay).

**REJECTED** (toy v2 NO-GO: lost to broad DR, cost 6.5 pp nominal success, an ordinary entry
state beat the failure's own state).

## Previous hypothesis 2

Pre-failure states from a real failed trajectory are unusually valuable.

**REJECTED** as a provenance claim (precursor sweep NO-GO: simulator-found and successful
states at the same time before the hazard did equally well).

## Previous hypothesis 3

Recoverability of the replay start state explains repair value better than simple trajectory
timing.

**REJECTED** (retrospective NO-GO: an inverted U exists, but time before failure predicted
repair value better and chose better states on hidden worlds).

## New observation

In all three studies the failure was used as a *training state*, and in all three its origin
carried no value. What a real failure can carry that a simulator-generated failure cannot is
information about the world: if the simulator predicted success and reality failed, the
simulator's model of the dynamics is wrong somewhere that matters for the outcome. That
information is not about where to train; it is about what to simulate.

## New hypothesis (H4)

Failure-localized simulator calibration may reveal dynamics discrepancies that calibration on
average trajectory error ignores. Concretely: when a real trajectory agrees with simulation for
most of its length and diverges shortly before a failure the simulator did not predict, a search
for the smallest simulator change that (a) preserves the agreement before the divergence,
(b) diverges at the observed time and (c) reproduces the failure event recovers the
failure-critical mechanism more reliably than fitting the whole trajectory.

Provisional name: Failure-Critical System Identification (FCSI). The name is not used publicly
until the novelty audit (`docs/research/FCSI_RELATED_WORK.md`) is complete.

**This is a NEW hypothesis.** It was not preregistered before, it is not a reinterpretation of
any earlier result, and none of the earlier gates is loosened or re-scored. The first test is
whether the simulator can be repaired (mechanism identification and failure reproduction in a
toy with known ground truth); policy improvement is not the first gate.

## Scope separation from Phoenix

Phoenix asks whether live actuator residuals can guide adaptation to hardware-response drift.
Ashfall asks which modelling assumption was wrong when reality produced an outcome the simulator
did not predict. They may share infrastructure, not questions.
