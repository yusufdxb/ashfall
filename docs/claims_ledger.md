# Claims ledger

The single place where Ashfall's claims are classified. Every other document
defers to this one.

- **VERIFIED**: executed in this repository and observed. The evidence column
  names the test, artifact or command.
- **INFERRED**: read from source or derived by reasoning; not executed.
- **NOT YET TESTED**: code exists; nothing has run it in the setting the claim needs.
- **REFUTED / UNINFORMATIVE**: run, and the result does not support the claim.
- **PROHIBITED**: must not be stated anywhere until its row changes.

Revision: branch `research/causal-failure-repair-v3`. Phoenix integration pin:
go2-phoenix `feat/causal-viability-replication` @ `d607e2f` (advanced from `5783416`; interface files unchanged).

## Research hypotheses

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
| friction_reduction -> slip, 8 pairs (`configs/h0/friction_slip.json`) | Phoenix / Isaac Lab | simulation | NOT YET TESTED | runbook `docs/runbooks/h0_isaac.md` |
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
| Closing Isaac Sim's SimulationApp ends the interpreter, and the CLI writes every artifact before closing | VERIFIED (source read, and three earlier runs that closed first exited 0 with incomplete bundles) | `docs/runbooks/h0_isaac.md`, `src/ashfall/cli.py` |
| A bundle's provenance is captured before its directory exists, so an in-repository bundle records a clean tree | VERIFIED | `tests/test_cli_e2e.py::test_provenance_is_collected_before_the_bundle_exists`; the smoke bundle records Ashfall `a3d1c96` clean |
| The smoke ran against the pinned Phoenix revision | NO: Phoenix was at `0132da0`, a local unpushed commit one past `d607e2f`. Its only interface-file change adds episode-generation bookkeeping to `PreResetCapture`; the methods Ashfall calls are unchanged (INFERRED from the diff) | smoke bundle `provenance.json` |
| The CI guard step can fail | VERIFIED: under the Actions shell it exits 0 with the guards and 5 when nothing is selected; the previous step exited 0 in both cases | `.github/workflows/ci.yml` |
| Bundles meant for publication record the GPU, local package names and untracked paths only as digests | VERIFIED | `environment.json` of the smoke bundle; `tests/test_provenance_bundle.py` |
| Historical Phase-I configs reproduce row-0 seeding only through an explicit tag | VERIFIED | `tests/test_scientific_guards.py::test_every_historical_config_reproduces_row0_explicitly` |

## Detector

| claim | status | evidence |
|---|---|---|
| The threshold detector is accurate on independently labelled physics or hardware data | NOT YET TESTED | `docs/detector/validation_protocol.md` |
| The fixture report shows slip F1 0.89 and command-mismatch recall 0.67 | VERIFIED as a `fixture_regression` report only | `ashfall detector-eval` on the bundled fixture |

## Historical (Phase I)

| claim | status | evidence |
|---|---|---|
| n = 11 paired sign-flip results reproduce from committed metrics (slippery -0.4155 pp, p = 0.726562; rough +0.5485 pp, p = 0.767578) | VERIFIED | `tests/test_multiseed_combined.py` |
| Phase I tested a failure curriculum | REFUTED: row-0 seeding delivered a nominal state | `docs/legacy/README.md` |
| Phase I shows a failure curriculum does not work | UNINFORMATIVE | same |
| Phase I compared terrains | REFUTED: the terrain block was not applied; the contrast was friction randomisation | same |

## Prohibited statements

Until the named row changes, none of these may appear in the README, docs,
papers, resumes or talks:

- "the detector is validated" or any detector accuracy figure from fixtures;
- "failure curricula do not work" or "failure curricula work";
- "Ashfall repairs policies" or any repair effect;
- "H0 passes" for any pathway without a preregistered simulator bundle;
- "sim-to-real" or any hardware result for Ashfall;
- any terrain contrast from Phase I;
- "recurrence" for episode incidence.
