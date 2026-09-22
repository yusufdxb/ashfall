# Claims ledger

> **Archived 2026-09-22. This ledger is closed.** It is the detailed claim-by-claim history of the
> program and is no longer updated. Every row marked NOT YET TESTED stays untested: each gate in
> front of it returned NO-GO. The final evidence summary is [`EVIDENCE.md`](../EVIDENCE.md); the
> project record is [`ARCHIVE.md`](ARCHIVE.md).

The single place where Ashfall's claims are classified. Every other document defers to this one.

- **VERIFIED**: executed in this repository and observed. The evidence column names the test,
  artifact or command.
- **INFERRED**: read from source or derived by reasoning; not executed.
- **NOT YET TESTED**: code exists, or a protocol is registered; nothing has run it in the
  setting the claim needs.
- **REFUTED / NULL / UNINFORMATIVE**: run, and the result does not support the claim.
- **PROHIBITED**: must not be stated anywhere until its row changes.

Revision: branch `research/ashfall-failure-boundary-replay`. Phoenix integration pin: go2-phoenix
`feat/causal-viability-replication` @ `d607e2f` (interface files unchanged at `9df76d7`, read
only in this revision).

## The FBR study

| id | claim | status | evidence |
|---|---|---|---|
| P | Under equal compute, FBR lowers held-out traverse failure on the GO2 (simulation) more than broad friction DR and ADR (A1) | NOT YET TESTED, and gated off: toy v2 returned NO-GO | protocol `docs/research/PREREGISTRATION.md` |
| S | FBR lowers held-out traverse failure more than nominal-seed boundary replay | NOT YET TESTED (fixed sequence after P) | same |
| N | FBR stays within 2 pp nominal success and 0.05 m/s tracking of the baseline | NOT YET TESTED | same |
| HW | FBR improves traverse success on the physical GO2 patch over broad DR | NOT YET TESTED; blocked on Phoenix's walking gate | hardware plan, never run; not part of the public archive |

## Mechanism toy (slip cart-pole, evidence kind `toy_mechanism`)

Run once at `f339087`; 12 training seeds; 92,160 rollout slots per arm; evidence
`results/fbr_toy/f339087/result.json`. Not a GO2 result.

| claim | status | evidence |
|---|---|---|
| Broad DR concentrates less boundary mass than boundary replay at the baseline | VERIFIED in the toy: 0.115 (broad), 0.275 (FBR), 0.294 (nominal-seed); 70% of broad-DR contexts always succeed | `informativeness_at_baseline` |
| FBR lowers held-out failure more than broad DR in the toy | NULL: +0.3 pp (95% CI -4.7 to +5.3), exact p 0.898 | `comparisons["held_out_failure:D_fbr-B_broad_dr"]` |
| FBR is more sample-efficient than broad DR in the toy | NULL (exploratory, added after the primary): D minus B -1.2, +0.6, +1.4, +0.3 pp at 25/50/75/100% of the budget, no p below 0.6 | `results/fbr_toy/f339087_efficiency_exploratory/efficiency.json` |
| FBR stays within the nominal margin in the toy | REFUTED: nominal success -6.8 pp (95% CI -8.8 to -4.8) against the baseline | `comparisons["nominal_success:D_fbr-A_no_repair"]` |
| Both FBR and broad DR improve on no repair in the toy | VERIFIED: -15.0 pp (FBR) and -15.3 pp (broad DR; -15.25 unrounded) held-out failure, exact p 0.0005 each | `comparisons` |
| The failure's friction band applied from nominal states (arm E) is worse than FBR in the toy | VERIFIED: FBR minus E -10.8 pp, exact p 0.0024 | `comparisons["held_out_failure:D_fbr-E_friction_band"]` |
| The toy capsule is reproduced by the acceptance checks, and the hidden friction is inside the reproduced range | VERIFIED: reproduced 0.04 to 0.24, hidden 0.08 | `reconstruction` |
| The v1 result survives the censoring-kind fix (`f3ccc1b`) | VERIFIED: A, B, D, E identical; C 36.9% to 35.1%; D minus B +0.3 pp, p 0.898 unchanged | `results/fbr_toy/f3ccc1b_corrected/result.json` |
| The full v1 run, baseline retraining included, is deterministic | VERIFIED: the corrected rerun reproduced A, B, D and E exactly | same |
| Retraining the toy arms from the saved baseline is deterministic | VERIFIED: retraining B, C, D reproduced every final evaluation exactly; a full rerun including the baseline has not been compared | efficiency run |

## Toy v2, the preregistered gate (`results/fbr_toy_v2/8f49f02/result.json`)

| claim | status | evidence |
|---|---|---|
| Minimal FBR (patch-entry seed) beats both ADR and broad DR on held-out failure | NULL on the intersection-union primary: vs ADR -6.7 pp (95% CI -9.4 to -4.1, p 0.0010); vs broad DR -2.7 pp (-6.1 to +0.6, p 0.105) | `decision`, `comparisons` |
| Minimal FBR stays within the nominal margin | REFUTED: nominal success -6.6 pp (90% CI -8.4 to -4.7); tracking +0.038 m/s | `comparisons` |
| The failure's own entry state adds value over a nominal entry state | REFUTED in the toy: C (nominal entry) beat D by 3.6 pp (95% CI 1.5 to 5.7, p 0.0029) | `comparisons["held_out_failure:D_fbr_entry-C_entry"]` |
| GO for the GO2 study (E0) | NO-GO under the registered rule | `decision.GO_for_GO2_E0 = false` |
| B and ADR received at least the simulator steps of C and D | VERIFIED: 16,262,400 each vs at most 16,215,760 | `simulator_steps_per_seed` |
| Reconstruction from the entry seed recovers the hidden friction | mostly NOT: hidden friction inside the reproduced range in 1 of 6 worlds | `capsules` |

## Precursor time sweep (new hypothesis after FBR; `docs/results/PRECURSOR_SWEEP_RESULT.md`)

Preregistered at `3205f02`; evidence `results/precursor_toy/3205f02/`; evidence kind `toy_mechanism`.

| claim | status | evidence |
|---|---|---|
| Repair effectiveness varies with replay start time along a failed trajectory (toy) | VERIFIED in the toy: omnibus p 5e-5, curve class C (interior optimum at T-1.5 s) | `stage1.json` `omnibus`, `curve_class` |
| Replay from failure onset is near useless in the toy | VERIFIED: recoverability is zero from 0.26 to 0.52 s before onset on all 12 failures; T-1.5 beats T0 by 13.5 pp on hidden worlds (p 0.0005) | `stage1.json` `setup.*.curve_real`; `stage2.json` `comparisons` |
| A pre-failure start beats the old FBR entry seed, broad DR and ADR at equal compute, and keeps the nominal margin (toy) | VERIFIED on hidden worlds: -7.9, -11.0, -12.4 pp (p 0.0005 each); nominal -1.3 pp, 90% lower bound -1.7 pp | `stage2.json` `gates` G1, G3, G4 |
| The real failed trajectory adds value over simulator-found or successful states at the same start time | REFUTED in the toy: vs simulator-found +0.5 pp (p 0.50), vs matched success +0.2 pp (p 0.83), vs nominal entry +0.0 pp (p 0.997); real beats simulator in 3 of 6 worlds | `stage2.json` `gates` G2, G5 |
| GO for a named precursor method and its Isaac Lab / GO2 study | NO-GO under the registered rule | `stage2.json` `GO = false` |

## Recoverability retrospective (new, exploratory; `docs/results/RECOVERABILITY_RETROSPECTIVE_RESULT.md`)

Analysis plan frozen with code at `2227de1` before estimation; evidence
`results/recoverability/retro_2227de1/`; evidence kind `toy_mechanism`; exploratory (reuses seen outcomes).

| claim | status | evidence |
|---|---|---|
| Repair value is non-monotonic in the baseline's recoverability of the start state (toy) | VERIFIED, exploratory: quadratic vertex 0.48 (95% 0.45 to 0.50), both slopes exclude zero | `analysis.json` `interior_discovery` |
| Recoverability explains repair value better than time before failure or distance to the hazard | REFUTED in the toy: LOWO RMSE 4.38 vs 4.04 (time) and 4.29 pp (distance); hidden-world R^2 0.34 vs 0.62 and 0.59 | `analysis.json` `models_discovery`, `heldout.oos` |
| Selecting the state nearest an intermediate recoverability picks better repair states on unseen worlds | REFUTED in the toy: regret 3.4 pp vs 1.9 pp for the timing rule | `analysis.json` `heldout` |
| Provenance adds value after conditioning on recoverability | NOT SUPPORTED: permutation p 0.074, no CV gain | `analysis.json` `provenance_discovery` |
| "Intermediate success probability is the best training start" is a new idea | REFUTED by prior work (RCG 2017, Sampling For Learnability 2024 and others) | `docs/research/RECOVERABILITY_RELATED_WORK.md` |
| GO for a recoverability preregistration, toy study, Isaac Lab or GO2 work | NO-GO under the frozen Phase 7 gate | `analysis.json` `phase7_gate.pass = false` |

## Failure-critical system identification (new, preregistered; `docs/results/FCSI_TOY_RESULT.md`)

Preregistered with code at `1b657c1`; evidence `results/fcsi_toy/1b657c1/`; evidence kind `toy_mechanism`.

| claim | status | evidence |
|---|---|---|
| Failure-event-conditioned sparse identification names the failure-critical mechanism better than global identification (toy) | REFUTED: 7/18 vs 14/18 (multiple shooting) | `result.json` `summary.identification_correct` |
| It predicts held-out failures better than global identification | REFUTED: MAE 0.123 vs 0.051, p 0.008 against | `summary.fcsi_minus_global_mae` |
| It abstains when the mechanism is outside its library | VERIFIED 8/8, but it also abstained on 11/18 known mechanisms (precision 0.42) | `summary.unknown_abstained` |
| Whole-trajectory error misses failure-critical mechanisms | VERIFIED in the toy (3/18), but reset-based multiple shooting already fixes most of it (14/18) | `summary.identification_correct` |
| Sparse single-mechanism selection on reset-based residuals identifies the mechanism | EXPLORATORY, not registered: 18/18, MAE 0.021; no abstention (8/8 wrong on unknowns) | `instances[*].methods.FCSI_no_event` |
| GO for the quadruped cross-simulator study, simulator repair or GO2 | NO-GO under the registered Gate 2 | `summary.GO = false` |

## Active simulator diagnosis (new, preregistered; `docs/results/ACTIVE_DIAGNOSIS_TOY_RESULT.md`)

Preregistered at `b86006d` with code `2b3e40e`; evidence `results/active_diag/2b3e40e/`; `toy_mechanism`.

| claim | status | evidence |
|---|---|---|
| Information-chosen safe probes beat fixed or random probes | REFUTED in the toy: 38 vs 38 and 37 of 42 | `eval_result.json` `summary.total_correct` |
| Active probing improves unknown-mechanism rejection over passive calibrated abstention | NOT SUPPORTED: 11 vs 10 of 12 | `summary.unknown_rejected` |
| A probe after an ambiguous failure trace resolves ties between known mechanisms | VERIFIED in the toy for any probe (27-28 vs 17 of 30); closed-set active discrimination is prior art | `summary.known_correct` |
| Probes certified safe under every plausible library hypothesis are safe | REFUTED when the mechanism is outside the library: falls in 3 of 18 active probes on unknown worlds | `summary.safety_violations`, instances |
| GO for stress test, quadruped cross-simulator study or GO2 | NO-GO | `summary.GO = false` |

## FBR software

| claim | status | evidence |
|---|---|---|
| Onset extraction, capsule contract, seed refusal, boundary estimation, samplers and acceptance checks behave as specified | VERIFIED on CPU (30 tests) | `tests/test_fbr.py` |
| A seed far from onset (the Phase-I row 0) is refused, and in the toy the FBR seed delivers the failure while row 0 does not | VERIFIED | `tests/test_fbr.py::TestPhaseOneRegression` |
| Every toy arm receives the same rollout budget | VERIFIED | `tests/test_fbr.py::TestToyBudget`; `rollout_slots_per_arm` in the result |
| Boundary estimation and refresh add rollouts outside the budget for arms C and D | INFERRED from the code constants (not instrumented): 4 refreshes x 3 seeds x 20 frictions x 8 replicates of 100 steps = 192,000 steps, 1.19% of the 16,128,000 training steps; arm B has none | `fbr/toy_slip.py::run_arm`, `boundaries_for` |
| FBR runs in Isaac Lab | NOT YET TESTED: sampler-to-scenario adapter, spatial patch and in-loop refresh not written | `docs/legacy/architecture/ashfall_fbr.md` |

## Permitted and prohibited statements

Allowed now: "FBR is implemented and CPU-tested"; "a preregistered GO2 simulation study and a
hardware protocol are prepared"; "in a CPU toy, boundary replay concentrated more boundary mass
than broad randomization but did not improve held-out failure and degraded nominal success".

Prohibited until the named row changes:

- "Ashfall learns from real failures", "improves real robot robustness", "closes the
  real-to-sim-to-real loop", "reproduces hardware failures" (row HW; no hardware capsule exists);
- "FBR beats broad domain randomization" or any FBR effect on the GO2 (rows P, S);
- "FBR is more sample-efficient" (toy null; row P);
- "the real failure adds value over a simulator state" (refuted in toy v2, and again at the
  best start time in the precursor sweep);
- "Ashfall chooses the safest informative probe", "active diagnosis detects unmodelled
  dynamics", or any active-diagnosis hardware claim (NO-GO);
- "Ashfall identifies which physics the simulator missed", "failure-conditioned system
  identification works", or any FCSI simulator-repair or policy result (Gate 2 NO-GO);
- "recoverability explains or predicts repair value", "Recoverability-Guided Replay", or any
  recoverability method claim (retrospective NO-GO; the band principle is prior work);
- any named precursor-replay method, "learning before the fall", or a precursor result on the GO2
  or in Isaac Lab (precursor sweep NO-GO; nothing run outside the toy);
- the toy result stated as a GO2 or quadruped result;
- a measured surface friction stated as the simulator coefficient;
- "the detector is validated"; "failure curricula do not work" or "failure curricula work";
  any terrain contrast from Phase I; "recurrence" for episode incidence (carried over from v3).

## Retired v3 hypotheses (H0 to H4, archived preregistration)

| id | claim | status | evidence |
|---|---|---|---|
| H0 | A physical intervention induces its intended phenotype significantly more often than a matched no-intervention counterfactual | NOT ESTABLISHED. The only simulator run is a 4-pair implementation smoke, not preregistered, and its verdict is FAIL | simulator rows below |
| H1 | Reproduced failures dynamically resemble independently labelled reference failures | NOT YET TESTED, and blocked by H0 and by the absence of independently labelled reference data | `ashfall.protocol.hypotheses` refuses execution |
| H2 | Phenotype-conditioned adaptation reduces held-out failure incidence versus equal-compute non-conditioned arms | NOT YET TESTED, blocked by H0 and H1 | same |
| H3 | Repair generalises to withheld causes and intensities of the same phenotype | NOT YET TESTED, blocked | same |
| H4 | Repair stays within a preregistered nominal degradation margin | NOT YET TESTED, blocked; margin is a placeholder in the preregistration | same |

## H0 evidence rows

| pathway | backend | evidence kind | status | evidence |
|---|---|---|---|---|
| friction_reduction -> slip | toy surrogate | mock | VERIFIED as software: 6 of 6 pairs DELIVERED, exact McNemar p = 2/64 = the floor | `tests/test_gates_h0.py::TestH0::test_friction_to_slip_passes_and_wrong_phenotype_fails` |
| friction_reduction -> collapse (deliberately wrong phenotype) | toy surrogate | mock | VERIFIED as software: 6 of 6 WRONG_PHENOTYPE, verdict FAIL | same test |
| unverified intervention on one pair | toy surrogate | mock | VERIFIED as software: verdict UNINFORMATIVE | `tests/test_gates_h0.py::TestH0::test_unverified_intervention_makes_h0_uninformative` |
| friction_reduction -> slip, 4-pair smoke, v3b policy, 100-step horizon, 3 replicates per pair | Phoenix / Isaac Lab 4.5.22 on Isaac Sim 6.0 | simulation, purpose `implementation_smoke` | FAIL, not a research result. D1 4 of 4 (robot-shape friction read back as 0.12 static and 0.10 dynamic); D2 2 of 4 (ratios to the nominal null maximum 1.42, 1.89, 2.28, 1.24 against a margin of 1.5); D3 1 of 4. Statuses: 1 DELIVERED, 2 NO_DEPARTURE, 1 WRONG_PHENOTYPE. Slip detected in 3 of 4 treatment arms and 2 of 4 control arms; exact McNemar p = 1.0 with a floor of 0.125 at 4 pairs. Every control arm was flagged for attitude, consistent with the environment drift in `docs/limitations.md`. The D2 regularisation sweep disagrees with the primary verdict for one pair (agreement 0.875). Four runs with the same seeds produced identical outcomes | `results/software_validation/h0_isaac_smoke_v3b/efd20885abbcad83/` (spec `configs/h0/smoke_friction_slip.json`) |
| friction_reduction -> slip, 8 pairs (`configs/h0/friction_slip.json`) | Phoenix / Isaac Lab | simulation | NOT YET TESTED | runbook `docs/legacy/runbooks/h0_isaac.md` |
| any pathway -> stumble or contact_loss | any | any | UNSUPPORTED by design, excluded from the first study | `ashfall.ontology.PHENOTYPES` exclusion reasons |

## Engineering claims

| claim | status | evidence |
|---|---|---|
| CPU suite, ruff and mypy (typed core) pass on Python 3.10, 3.11 and 3.12, and the simulator-boundary job passes against the pinned Phoenix SHA | VERIFIED on GitHub Actions for `43b43fe` (run 34719374452: lint, test 3.10, test 3.11, test 3.12, simulator-boundary all succeeded) and locally on all three versions | the first push, `d4516b5`, failed CI: mypy pinned to 3.10 could not parse numpy 2.5 stubs, and a pre-existing test called a private scipy helper whose signature changed in scipy 1.16; both were reproduced locally and fixed in `43b43fe` |
| Every protected defect has a guard test that fails when the defect is reintroduced | VERIFIED | `tests/test_scientific_guards.py::test_every_protected_defect_has_a_guard_test` and the nine guards it indexes |
| Fixture data cannot be passed where scientific data is required | VERIFIED | `tests/test_provenance_bundle.py::TestDatasets` |
| Evidence bundles refuse missing provenance | VERIFIED | `tests/test_provenance_bundle.py`, `tests/test_scientific_guards.py` |
| Detector evaluation catches five mutants (always_collapse, always_stumble, always_command_mismatch, no_slip_priority, label_swap) | VERIFIED on the fixture dataset | `tests/test_detector_mutation.py` |
| Equal-compute violations, held-out re-screening and prerequisite-skipping are refused | VERIFIED | `tests/test_protocol_arms.py`, `tests/test_selection_ledger.py`, `tests/test_hypothesis_gates.py` |
| BCa acceleration sign matches an independent jackknife | VERIFIED | `tests/test_stats_primary.py` |
| Degenerate zero-width intervals are not accepted as proof of a budget | VERIFIED | `tests/test_paired_intervals.py` |
| Phoenix interface matches the pinned integration surface | VERIFIED against `5783416` and `d607e2f` | `tests/test_delivery.py::TestRestoreContract` |
| Phoenix intervention adapters write and read back correctly | VERIFIED against CPU fakes; on Isaac Lab see the smoke row | `tests/test_phoenix_interventions.py` |
| With the checkpoint-resolved normalizer flag, the runtime v3b actor equals a plain MLP from the checkpoint | VERIFIED on Isaac Lab 4.5.22 / Isaac Sim 6.0 | probe output recorded in the session log (`max|diff| = 0.000`) |
| `ashfall repair` training path emits a TrainingArtifact | INFERRED (code written, not executed) | `src/ashfall/training.py` |
| The H0 simulator path runs end to end and writes a finalized, index-verified bundle | VERIFIED on Isaac Lab 4.5.22 / Isaac Sim 6.0 | smoke bundle above; `index.json` hashes recomputed |
| The same H0 spec and seeds give identical outcomes on the simulator | VERIFIED for four runs of the smoke spec | statuses, gate counts, incidences, exact p and verdict compared field by field |
| Closing Isaac Sim's SimulationApp ends the interpreter, and the CLI writes every artifact before closing | VERIFIED (source read, and three earlier runs that closed first exited 0 with incomplete bundles) | `docs/legacy/runbooks/h0_isaac.md`, `src/ashfall/cli.py` |
| A bundle's provenance is captured before its directory exists, so an in-repository bundle records a clean tree | VERIFIED | `tests/test_cli_e2e.py::test_provenance_is_collected_before_the_bundle_exists`; the smoke bundle records Ashfall `a3d1c96` clean |
| The smoke ran against the pinned Phoenix revision | NO: Phoenix was at `0132da0`, a local unpushed commit one past `d607e2f`. Its only interface-file change adds episode-generation bookkeeping to `PreResetCapture`; the methods Ashfall calls are unchanged (INFERRED from the diff) | smoke bundle `provenance.json` |
| The CI guard step can fail | VERIFIED: under the Actions shell it exits 0 with the guards and 5 when nothing is selected; the previous step exited 0 in both cases | `.github/workflows/ci.yml` |
| Bundles meant for publication record the GPU, local package names and untracked paths only as digests | VERIFIED | `environment.json` of the smoke bundle; `tests/test_provenance_bundle.py` |
| Historical Phase-I configs reproduce row-0 seeding only through an explicit tag | VERIFIED | `tests/test_scientific_guards.py::test_every_historical_config_reproduces_row0_explicitly` |

## Detector

| claim | status | evidence |
|---|---|---|
| The threshold detector is accurate on independently labelled physics or hardware data | NOT YET TESTED | `docs/legacy/detector/validation_protocol.md` |
| The fixture report shows slip F1 0.89 and command-mismatch recall 0.67 | VERIFIED as a `fixture_regression` report only | `ashfall detector-eval` on the bundled fixture |

## Historical (Phase I)

| claim | status | evidence |
|---|---|---|
| n = 11 paired sign-flip results reproduce from committed metrics (slippery -0.4155 pp, p = 0.726562; rough +0.5485 pp, p = 0.767578) | VERIFIED | `tests/test_multiseed_combined.py` |
| Phase I tested a failure curriculum | REFUTED: row-0 seeding delivered a nominal state | `docs/legacy/README.md` |
| Phase I shows a failure curriculum does not work | UNINFORMATIVE | same |
| Phase I compared terrains | REFUTED: the terrain block was not applied; the contrast was friction randomisation | same |

## Prohibited statements carried over from v3


- "the detector is validated" or any detector accuracy figure from fixtures;
- "failure curricula do not work" or "failure curricula work";
- "Ashfall repairs policies" or any repair effect;
- "H0 passes" for any pathway without a preregistered simulator bundle;
- "sim-to-real" or any hardware result for Ashfall;
- any terrain contrast from Phase I;
- "recurrence" for episode incidence.
