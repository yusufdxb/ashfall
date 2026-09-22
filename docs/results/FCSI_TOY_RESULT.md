# Failure-critical system identification, ground-truth toy: result (Gate 2)

Status: **FINAL. The preregistered Gate 2 returned NO-GO.** Do not edit the numbers in this file.
Phase J (quadruped cross-simulator study) is not entered; no Isaac Lab or GO2 work is run for
FCSI.

Evidence kind: `toy_mechanism` (cart-pole with a low-friction patch). Nothing here is a GO2,
quadruped or Isaac Lab result.

- Preregistration: `docs/research/FCSI_TOY_PREREGISTRATION.md`, committed with its code at
  `1b657c1` before any registered instance existed.
- Run: `python -m ashfall.fcsi.toy_study --output results/fcsi_toy/1b657c1` at `1b657c1`,
  29 instances, 1,058 s wall on 22 workers. A first launch at the same commit was stopped
  after 15 minutes with nothing written (20 workers each using multithreaded BLAS, load 64 on
  24 cores); the registered run was relaunched with one BLAS thread per worker
  (`results/fcsi_toy/run_notes.txt`). The code is deterministic; nothing was changed.

## Verdict, plainly

1. **FCSI as registered does not repair the simulator better than global system
   identification.** It named the true mechanism in 7 of 18 known-mechanism instances (gate:
   14), its held-out failure prediction was worse than multiple shooting (MAE 0.123 vs 0.051,
   paired sign-flip p 0.008 in the wrong direction), and its patch reproduced the target failure
   in 7 of 18 (gate: 14). G2a, G2b and G2c fail.
2. **It abstained on every unknown mechanism (8 of 8) and proposed no patch on the harmless
   controls (3 of 3)**, and it never degraded nominal fidelity (G2d, G2e, G2f pass). But it also
   abstained on 11 of 18 *known* mechanisms, so the abstention is not discriminating: of 19
   abstentions, 8 were correct (precision 0.42).
3. **The failure-event condition, which is the core of FCSI, is what failed.** A single failure
   under process noise is a rare realization: from the replay point, a patch with the correct
   mechanism (the no-event ablation's fit, correct in 18 of 18) reproduced the logged event only
   2% to 41% of the time, and on the unregistered quick world the exact true physics managed
   23%. As a feasibility constraint the event (a) rejected the true mechanism in several instances and (b) admitted sensing
   mechanisms (observation delay, tilt bias) that are invisible to reset-based residuals, so they
   pass the pre-divergence fit untouched, and that reproduce "a failure" by destabilizing the
   closed loop. The acceptance checks then rejected those spurious fits (divergence not
   explained, validation not reproduced), producing the abstentions.
4. **Therefore: NO-GO.** Per the preregistration, the result is reported as it is; nothing is
   retuned on these instances.

## Gate 2

| gate | rule | result | pass |
|---|---|---|---|
| G2a identification | >= 14/18 and more than each global | FCSI 7; G0 3, G0b 14, G1 7 | **no** |
| G2b held-out prediction | lower mean MAE than every global, p <= 0.05 vs best | FCSI 0.123; G0 0.104, G0b 0.051, G1 0.095; FCSI minus G0b +0.072 [0.022, 0.123], p 0.008 | **no** |
| G2c reproduction | >= 14/18 by the event rule | 7/18 | **no** |
| G2d nominal fidelity | d^2/row <= 1.05 x unpatched | 15.9 vs 20.8 | yes |
| G2e abstention | >= 6/8 unknowns UNEXPLAINED | 8/8 | yes |
| G2f negative control | no patch 3/3 | 3/3 | yes |
| **decision** | | | **NO-GO** |

Mechanism identification by world (correct / 6):

| method | K1 friction collapse | K2 latency | K3 braking friction | total /18 | params changed (mean) |
|---|---|---|---|---|---|
| FCSI (registered) | 3 | 1 | 3 | 7 | 0.4 |
| G0 whole-trajectory | 0 | 0 | 3 | 3 | 7.6 |
| G0b multiple shooting | 6 | 2 | 6 | 14 | 4.1 |
| G1 DROPO-like | 2 | 1 | 4 | 7 | 5.0 |
| FCSI_no_event (ablation) | 6 | 6 | 6 | **18** | 1.0 |

Held-out failure-rate MAE on known worlds (12 conditions each; unpatched nominal model 0.194):
FCSI 0.123, G0 0.104, G0b 0.051, G1 0.095, FCSI_no_event 0.021. On the discrepant subset
(where reality and the nominal model differ by >= 0.15; nominal 0.302): FCSI 0.198, G0 0.146,
G0b 0.074, G1 0.130, FCSI_no_event 0.029.

Compute per known instance (simulator steps, mean): FCSI 15.8M, FCSI_no_event 14.1M, G0 12.8M,
G0b 7.8M, G1 15.9M. The comparison was not compute-limited in FCSI's disfavour.

## What did work (exploratory, not a registered claim)

- **Divergence localization.** On 26 failure logs, the detected first divergence was within
  0.1 s of the instrumented true onset in 23 (median error 0.00 s); two were 0.14 s early and one
  (U1_strip#2) raised a false early alarm 1.48 s before the true onset. Example (K1_kinetic#4):
  match to 1.28 s, first divergence 1.30 s, primary channel slip, fall at 2.14 s.
- **Whole-trajectory error really does miss failure-critical mechanisms** (G0: 3 of 18, and it
  moved 7.6 of 10 parameters on average), as the hypothesis said. But this is already solved by
  reset-based fitting: multiple shooting got 14 of 18 without any failure conditioning.
- **Sparse single-mechanism selection on reset-based residuals** (the no-event ablation: the same
  failure log and nominal logs, one-step residual likelihood, one mechanism at a time, best fit
  wins) named the true mechanism in 18 of 18 and predicted held-out failure rates best (MAE 0.021
  vs 0.051 for multiple shooting, paired p 0.002; vs G1 p 1.5e-5; vs G0 p 8.4e-5). The
  advantage over multiple shooting comes from sparsity: the joint fit spread latency across
  damping and mass (K2: 2 of 6).
- **But the ablation has no abstention.** On all 8 unknown-mechanism instances it confidently named
  actuator strength, and its patch did not improve prediction there (MAE 0.310 vs 0.311
  unpatched). It is a model-selection method, not a diagnosis you can trust when the library is
  wrong, and it does not use the failure event at all. Sparse model selection from a library is
  established (SINDy, Bayesian model selection, multiple-model fault diagnosis; see
  `docs/research/FCSI_RELATED_WORK.md`), so this observation is not a novel method either.

## Falsified in this toy

- "Conditioning system identification on reproducing the failure event identifies the
  failure-critical mechanism better than global identification": **falsified** (7/18 vs 14/18 for
  multiple shooting; worse held-out prediction).
- "The failure event is the informative part of a real failure": **not supported.** Everything
  that identified mechanisms well used the failure log's *dynamics residuals*, not its outcome.

## Consequences (preregistered)

- NO-GO: Phase J (quadruped cross-simulator), simulator repair and policy retraining are not run.
- FCSI as registered is archived as a negative result, with the three earlier Ashfall hypotheses.
- Not proposed here: a rescue that swaps the event constraint for something else. The
  exploratory sparse-residual observation would be a new hypothesis with its own preregistration
  and novelty audit, and the audit already says it overlaps established sparse identification.
- Not claimed: anything about the GO2, Isaac Lab, or other failure families.

## Artifact index (SHA-256; immutable, checked by `tests/test_frozen_results.py`)

```
8b4c8f9fe2b02a8d23279b7cd9c939272782a8a0b40f01e3f0082f7e5dd532db  results/fcsi_toy/1b657c1/result.json
902a555347ae55df259a0cd95f257cdb44b5c2db2bfb2d0ba3d4e60d2ba2ff3c  results/fcsi_toy/1b657c1.log
57a0993d50e109dfbdb45c4a8223dddd5f1a2e514818366a2e33eb7d692657fe  results/fcsi_toy/run_commit.txt
61675d13ab6c85d3beb9e014e77cae10e4d6a50c3bb828b9b64c32a2602cc0ed  results/fcsi_toy/run_notes.txt
```
