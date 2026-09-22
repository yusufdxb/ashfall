# Ashfall: archive record

Status: **archived, 2026-09-22.** This is the formal final record of the Ashfall research
program. It summarizes; where it and a frozen result record differ, the frozen record governs.

- Last research commit: `4186bbf`. Commit hashes in this record belong to the private research
  history; see [`RESEARCH_HISTORY.md`](RESEARCH_HISTORY.md).

## Public release

The research was done on branches kept in a private archive (branch
`archive/ashfall-negative-results-2026`, commit `dfb9c66`, tag `archive/ashfall-2026`). That history
also contained internal planning documents, so it is not published. This public release
(branch `release/ashfall-archive-2026`, tag `archive/ashfall-public-2026`) starts from the previously
public history and adds the final archive content as new commits. Differences from the private
archive are limited to:

- one result artifact published as a sanitized copy with local path text removed
  ([`PUBLIC_ARTIFACT_MANIFEST.md`](PUBLIC_ARTIFACT_MANIFEST.md) records both hashes);
- release documentation: this section, the manifest, the research history index, a README and
  evidence summary reworded for public readers (same results and numbers), and the tests that
  check the release;
- the three FBR-phase planning documents, which were already absent from the private archive's
  final tree and never enter the public history.

## 1. Objective

Test whether a real robot failure carries information that makes policy repair or simulator
diagnosis meaningfully better than simpler alternatives, and stop each line of work at a
preregistered gate before it could consume Isaac Lab or GO2 hardware time.

All five studies used CPU toys: a cart-pole driven across a low-friction patch (slip cart-pole
for the replay studies; a parametric cart-pole with a mechanism library for the diagnosis
studies). In every toy, the "real" system is a hidden simulator configuration. No physical
failure data entered any study.

## 2. Hypotheses, gates and results

### Study 1: Failure-Boundary Replay (FBR)

- **Question.** Can a real failure define a local simulation curriculum near the observed failure
  boundary that improves robustness more efficiently than broad domain randomization?
- **Gate** (toy v2, preregistered at `8f49f02` with its code). GO only if FBR (D) beats both ADR
  and broad DR (B) on held-out failure (exact two-sided sign-flip over 12 seed-paired means,
  alpha 0.05 each), D minus ADR is at most -5 pp, and nominal non-degradation passes (nominal
  success one-sided lower bound above -2 pp; tracking RMSE upper bound below +0.05 m/s).
- **Result: NO-GO** (`08adcff`; record frozen at `48c8c71`).
  D minus ADR -6.7 pp (95% CI -9.4 to -4.1, p 0.0010); D minus B -2.7 pp (95% CI -6.1 to +0.6,
  p 0.105), so the intersection-union primary does not reject. Nominal success -6.5 pp
  (90% CI -8.4 to -4.7) against the -2 pp margin. Replay from an ordinary patch-entry state (C)
  beat the failure-state replay by 3.6 pp (95% CI 1.5 to 5.6, p 0.0029). An earlier exploratory
  toy (v1, `f339087`) was already a null against broad DR (+0.3 pp, p 0.898).
- **Why it stopped.** The primary did not reject, the nominal margin failed, and the failure's
  own state did worse than an ordinary state.
- Record: [`results/FBR_TOY_NEGATIVE_RESULT.md`](results/FBR_TOY_NEGATIVE_RESULT.md).

### Study 2: Precursor replay

- **Question.** Where along a failed trajectory should replay start, and does a real-failure
  precursor state carry value that other states at the same time do not?
- **Gate** (preregistered at `3205f02`; stage 1 on discovery worlds chose the start offset P*,
  stage 2 on six hidden worlds with twelve new seeds decided). GO required all of: G1 beat onset
  replay; G2 beat the old FBR entry seed, nominal entry, a successful trajectory at the same place
  and a simulator-found precursor; G3 beat broad DR and ADR by at least 3 pp; G4 nominal
  non-degradation; G5 the direction reproduces in at least 4 of 6 hidden worlds.
- **Result: NO-GO** (`dfec130`). Stage 1 found an interior optimum at T-1.5 s (omnibus
  p 5e-5). On hidden worlds, T-1.5 s beat onset (-13.5 pp), the old FBR seed (-7.9 pp), broad DR
  (-11.0 pp) and ADR (-12.4 pp), and kept nominal performance (-1.3 pp). It did not beat nominal
  entry (+0.0 pp, p 0.997), the matched successful state (+0.2 pp, p 0.83) or the
  simulator-found precursor (+0.5 pp, p 0.50); G2 and G5 failed.
- **Why it stopped.** The useful information was the start time, not the fact that the state
  came from a real failure. No precursor method was named or built.
- Record: [`results/PRECURSOR_SWEEP_RESULT.md`](results/PRECURSOR_SWEEP_RESULT.md).

### Study 3: Recoverability-guided replay

- **Question.** Does the recoverability of a replay start state (the baseline's success
  probability from it) explain repair value better than time before failure, distance to the
  hazard or provenance?
- **Gate** (exploratory analysis plan frozen with its code at `2227de1`, before any
  recoverability estimate; a novelty audit at `5c744a8` preceded it). The analysis reused
  outcomes already seen in study 2, so it could refute the hypothesis but never confirm it. The
  Phase 7 gate required recoverability to beat time and distance on leave-one-world-out CV, an
  interior optimum or significant slope, little added value from provenance, and hidden-world
  regret for a recoverability rule no worse than for a timing rule.
- **Result: NO-GO** (`61bb95d`). Repair value was non-monotonic in recoverability, peaking at
  0.48 (95% interval 0.45 to 0.50). Time before failure predicted repair value better (LOWO RMSE
  4.04 vs 4.38 pp; CV R^2 0.43 vs 0.32), and so did distance to the hazard (4.29 pp, 0.35). On
  hidden worlds the recoverability rule had 3.4 pp regret against 1.9 pp for the timing rule.
  Provenance added little (permutation p 0.074).
- **Why it stopped.** Three of the five gate conditions failed. The intermediate-difficulty
  pattern is established prior work (reverse curriculum generation, sampling for learnability),
  so it is not a contribution either. No preregistration, training, Isaac Lab or GO2 work followed.
- Record: [`results/RECOVERABILITY_RETROSPECTIVE_RESULT.md`](results/RECOVERABILITY_RETROSPECTIVE_RESULT.md).

### Study 4: Failure-critical system identification (FCSI)

- **Question.** When reality fails while the simulator predicts success, can identification
  conditioned on reproducing the failure event find the smallest missing physics, and beat
  whole-trajectory and reset-based global identification?
- **Gate 2** (preregistered at `1b657c1` with its code; a novelty audit at `1307255` found the
  idea new only as a combination). G2a named the true mechanism in at least 14 of 18 known
  instances and more than each global method; G2b gave the best held-out failure prediction;
  G2c reproduced the failure in at least 14 of 18; G2d to G2f covered nominal fidelity,
  abstention on unknown mechanisms and a negative control.
- **Result: NO-GO** (`d49daeb`). FCSI named 7 of 18; multiple shooting 14; whole-trajectory
  fitting 3; a DROPO-like baseline 7. Held-out failure MAE 0.123 for FCSI against 0.051 for
  multiple shooting (p 0.008, wrong direction). G2a, G2b and G2c failed. FCSI abstained on all
  8 unknown mechanisms but also on 11 of 18 known ones.
- **Why it stopped.** The failure event, the core of the method, was the part that failed: even
  a model with the correct mechanism reproduced the logged failure only 2% to 41% of the time,
  so the event rejected true mechanisms and admitted spurious sensing mechanisms.
- **Exploratory, not registered.** Divergence localization was within 0.1 s of the true onset
  in 23 of 26 logs. A sparse one-step selector on reset-based residuals, without the event,
  named 18 of 18 known mechanisms but never abstains and named a wrong mechanism on all 8
  unknown ones; it overlaps established sparse identification.
- Record: [`results/FCSI_TOY_RESULT.md`](results/FCSI_TOY_RESULT.md).

### Study 5: Active simulator diagnosis

- **Question.** After an unexplained failure, does choosing a safe probe by expected information
  gain improve UNKNOWN detection and known-mechanism diagnosis over passive abstention, a fixed
  probe and a random safe probe?
- **Gate** (preregistered at `b86006d`, code `2b3e40e`, development calibration frozen; a
  novelty audit at `03e9936` found safe closed-set active discrimination already published, so
  the question was narrowed to open-set detection). A1 known-mechanism non-inferiority; A2
  UNKNOWN detection at least 3 above passive abstention; A3 false UNKNOWN at most 3 of 30; A4
  zero safety violations in active probes; A5 at most 3 probes; A6 beats random; A7 beats the
  fixed probe by at least 4.
- **Result: NO-GO** (`4186bbf`). Total correct of 42: active 38, fixed 38, random 37, passive
  abstention 27. Known mechanisms: active 27 of 30, fixed 28. Unknown rejected: active 11 of 12,
  fixed 10, passive abstention 10. Probe selection cost about 3.5M simulated steps per diagnosis
  against about 121k for the fixed probe. Probes certified safe under every plausible library
  hypothesis caused falls in 3 of 18 executed active probes on unknown-mechanism worlds. A2, A4,
  A6 and A7 failed.
- **Why it stopped.** Information-based selection added nothing over simpler probes, open-set
  detection came mostly from the passive failure trace, and library-conditioned safety failed
  exactly when the library was wrong.
- Deviation, recorded before any result existed: the first evaluation launch crashed on a tie
  bug in the safety set and wrote nothing; the fix (`1ea85eb`) changed no untied behaviour and no
  development case had a tie.
- Record: [`results/ACTIVE_DIAGNOSIS_TOY_RESULT.md`](results/ACTIVE_DIAGNOSIS_TOY_RESULT.md).

## 3. How the questions connected

Each new question was written down before its study, in a transition note that left the earlier
verdicts untouched ([`results/ASHFALL_HYPOTHESIS_TRANSITION.md`](results/ASHFALL_HYPOTHESIS_TRANSITION.md),
[`research/FCSI_HYPOTHESIS_TRANSITION.md`](research/FCSI_HYPOTHESIS_TRANSITION.md),
[`results/ACTIVE_DIAGNOSIS_HYPOTHESIS_TRANSITION.md`](results/ACTIVE_DIAGNOSIS_HYPOTHESIS_TRANSITION.md)).

| From | Observation | Next question |
|---|---|---|
| FBR | the failure state was worse than an ordinary state | is the start time the issue? |
| Precursor | start time mattered; provenance did not | is recoverability behind the timing effect? |
| Recoverability | timing predicted better than recoverability | can a failure diagnose the simulator instead of seeding replay? |
| FCSI | one-step dynamics identified known mechanisms; open-set failed | can safe probing detect out-of-library mechanisms? |
| Active diagnosis | a fixed probe matched it; certified probes fell out of library | none: archived |

## 4. Strongest surviving observations

All from CPU toys; none is a new method.

1. Real-failure provenance did not make replay states more valuable once time and position were
   matched (studies 1 to 3).
2. Replay start time mattered far more than provenance. Recoverability reached zero between 0.52
   and 0.26 s before onset on every failure, and replays from there taught almost nothing
   (study 2).
3. Repair value was highest at intermediate recoverability, as curriculum-learning work
   predicts, but time before failure predicted it better (study 3).
4. Whole-trajectory identification missed failure-critical mechanisms (3 of 18), but reset-based
   multiple shooting recovered most of them (14 of 18) without failure conditioning (study 4).
5. Failure-event reproduction is a poor identification criterion, because correct models
   reproduce a rare failure only probabilistically (study 4).
6. One-step dynamics were highly informative for closed-set mechanism selection (exploratory,
   study 4).
7. Open-world diagnosis stayed hard: closed-set selectors were confidently wrong out of library
   (studies 4 and 5).
8. Active information-seeking probes did not outperform a fixed probe (study 5).
9. Safety certified under a finite model library failed when the true dynamics were outside it
   (study 5).

## 5. Limitations

- One family of toys, one hazard family (a low-friction patch), one baseline policy per toy,
  evolution-strategy training in the replay studies.
- Study 3 is exploratory: it reused seen outcomes and its candidate states were built around time
  offsets, which may favour the time model.
- Study 5 used a small probe family of full slow crossings, because the toy policy cannot stand
  still; a finer, lower-energy probe space could behave differently.
- The toys were chosen to be cheap and favourable to the proposed methods. A negative result here
  says the methods did not earn more expensive tests; it does not prove they fail everywhere.

The detailed list is [`limitations.md`](limitations.md).

## 6. No quadruped or hardware claims

No Ashfall study was run in Isaac Lab, on any quadruped model or on the Unitree GO2. Nothing in
this repository supports a claim about quadruped dynamics, cross-simulator behaviour, simulator
repair or retraining on a quadruped, or physical active diagnosis.

Before the five studies, earlier Ashfall designs ran a reset-file curriculum in Isaac Lab
(Phase I, n=11 training seeds, a null) and an Isaac Lab software smoke of the causal-reproduction
machinery. The Phase-I null was later found uninformative because the intended treatment was
never delivered: the reset bridge seeded the first row of each recorded trajectory, a nominal
gait state, and the trajectories were synthetic fixtures. Both are preserved under
[`legacy/`](legacy/README.md) and neither is a result of the five studies.

## 7. Final status

Archived. No study passed its gate, so no method advanced. The code, preregistrations and frozen
results are kept here for reproducibility; the research branches are kept in the private archive. Work under the Ashfall name should
restart only if independent physical robot data shows a phenomenon these studies do not explain,
and then as a new, separately preregistered question.

Archived on 2026-09-22.
