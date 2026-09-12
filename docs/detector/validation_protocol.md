# Detector validation protocol

Status: **NOT YET TESTED.** No detector validation exists in this repository. The
machinery to produce one is built and exercised on a regression fixture
(`ashfall.detector_eval`), and that fixture report is not a validation result,
by construction and by label.

## What is being validated

The threshold detector (`ashfall.taxonomy.detector.FailureDetector`) emits events
named `attitude`, `collapse`, `slip`, `stumble`, `contact_loss` and
`command_mismatch`. Under the ontology (`ashfall.ontology`) those names are
failure phenotypes: observed behavioural failures, distinct from the physical
intervention that caused them. Validating the detector means measuring how well
its emissions agree with phenotype labels produced without it.

The wrapper `ashfall.taxonomy.phenotype_detector.PhenotypeDetector` turns events
into `PhenotypeObservation` records with an `OnsetWindow`. A threshold detector
knows one frame, so it reports `transition_start == established_start`, claims no
precursor, and carries `label_source="detector"`. Ground truth may say more.

## Ground truth must not come from the detector

`ashfall.detector_eval.dataset.LabeledEpisode` refuses `label_source="detector"`,
refuses a window whose label source differs from the episode's, and refuses any
evidence field that says the label came from a threshold (`detector_event_frame`,
`detector_id`, `labelled_by: threshold`, and so on). An evaluation set built by
running the detector and keeping what it flagged measures agreement of the
detector with itself and is rejected at construction.

Accepted label sources:

| source | how a label is produced | platform |
|---|---|---|
| `simulator_ground_truth` | termination terms from the simulator's own termination manager (a `base_contact` termination is a collapse whether or not any height threshold fired); the intervention that was applied and verified by readback (a friction-reduction pair whose treatment episode stalls is a slip candidate, its matched control is a negative); physical criteria read from simulator state that the detector does not consume: foot bodies in contact with tangential velocity above a margin (slip), trunk body contact (collapse), commanded versus achieved velocity under a verified out-of-envelope command (command mismatch), quaternion tilt past the recovery envelope (attitude) | simulation |
| `human_review` | a reviewer inspects synchronized video and telemetry per the procedure in `hand_labeling.md`, with an independently reviewed subset and reported disagreement | hardware, simulation |
| `synthetic_fixture` | the generating recipe; regression only | fixture |

The physical criteria are written into each window's `evidence` so a reader can
see which rule produced the label. None of them is the detector's threshold.

## Composition

An evaluation set that holds only detector-triggered snippets cannot estimate a
false-positive rate or recall. Every set carries a category per episode and the
report counts them:

| category | contents | what it measures |
|---|---|---|
| `positive` | one phenotype with a window | recall, onset timing |
| `negative` | nominal locomotion, no phenotype | false positives |
| `near_miss` | approaches a phenotype and recovers (height to 0.20 m, pitch to 0.45 rad) | specificity near the boundary |
| `confusable` | an adjacent phenotype the detector is known to confuse (an obstacle blockage the detector calls slip) | the confusion matrix |
| `recovery` | a phenotype occurs, resolves for at least two seconds, occurs again | re-detection after recovery, and the incidence-versus-recurrence distinction |
| `multi_failure` | two phenotypes in one episode (attitude loss then collapse) | multi-label scoring |

Episodes lacking a channel (no per-foot contact, no joint velocity) are kept.
The phenotypes that need the missing channel are marked UNSUPPORTED on that
episode and are not scored for it; the report lists the count and the reason.
On `go2_hardware` the ontology already marks `collapse`, `slip`, `stumble` and
`contact_loss` unsupported for the reasons recorded in `ashfall.ontology.PHENOTYPES`,
so a hardware evaluation can only claim `attitude` and `command_mismatch`.

## Metrics (`ashfall.detector_eval.metrics`)

For each phenotype, scored over the episodes on which it is supported:

- episode-level one-vs-rest true/false positives and negatives, precision, recall,
  F1; an undefined ratio is `None`, never 1.0;
- window-level matching: each ground-truth window is matched to the earliest
  emission of its phenotype inside `[transition_start, end]`; emissions inside a
  window are never false positives (a detector re-firing while a slip persists is
  repeating a true detection); emissions outside every window of their phenotype
  are false-positive events;
- onset timing error, `(matched frame - transition_start) * dt` in seconds:
  median, 90th percentile, mean absolute, and the fraction of windows detected
  inside their span;
- false-positive events per scored episode.

For the dataset as a whole: false-positive events per episode and per minute of
telemetry, and an exclusive confusion matrix whose rows are the primary
ground-truth phenotype (earliest transition) or `none` and whose columns are the
earliest supported emission, `none`, or `unsupported`. The matrix asks which
phenotype was named and ignores timing; timing lives in the window metrics.

The report (`DetectorEvaluationReport`) carries `report_kind`:
`independent_evaluation` when the dataset's manifest is scientific,
`fixture_regression` when it is a fixture. Only the former sets
`validation_claim_allowed`, and constructing a report that claims otherwise raises.

## Mutation sensitivity is a precondition

Before any figure from an evaluation may be quoted, the mutation suite
(`ashfall.detector_eval.mutation.run_mutation_suite`) must show the evaluation
can see five deliberate defects, each built by `DetectorMutation` in the wrapper
without touching the detector:

| mutant | defect | expected to move |
|---|---|---|
| `always_collapse` | emits collapse on every episode | collapse precision, F1, false-positive load |
| `always_stumble` | emits stumble on every episode | stumble precision, F1, false-positive load |
| `always_command_mismatch` | emits command mismatch on every episode | command-mismatch precision, F1, false-positive load |
| `no_slip_priority` | removes the rule that command mismatch stands down while slip is active | command-mismatch false positives on slip episodes |
| `label_swap(slip, collapse)` | renames each to the other | the confusion matrix's off-diagonal for the pair |

Preregistered catch criterion, per mutant against the unmutated baseline on the
same dataset: for an affected phenotype, F1 or precision falls by at least 0.20
absolute, or false-positive events per scored episode rise by at least 0.25; for
the swap, either off-diagonal cell of the pair increases. A mutant that is not
caught is reported under `missed`; the guard tests in
`tests/test_detector_mutation.py` fail on it. On negatives alone a label swap
cannot be caught, and the suite says so rather than passing.

## Procedure for the first independent evaluation

1. Harvest matched counterfactual pairs in simulation with the pipeline in
   `ashfall.counterfactual` and `ashfall.harvest`, under a scientific manifest
   (`data/scientific/<harvest>/dataset.json`) naming policy hash, environment
   configuration hash, simulator version, seeds, and the intervention.
2. Label windows from simulator ground truth only: termination terms, verified
   intervention identity, and the physical criteria above. Record the criterion
   in each window's evidence.
3. Sample every category. Controls of delivered pairs are negatives; near misses
   are pairs whose treatment departed (gate D2) without the phenotype (gate D3);
   confusables come from interventions whose plausible pathways overlap
   (`ashfall.ontology.PLAUSIBLE_PATHWAYS`); recovery and multi-failure episodes
   come from long horizons with mid-episode interventions.
4. Freeze the detector configuration and version before opening the labels.
   Threshold tuning, if any, uses a disjoint development harvest.
5. Run `evaluate_detector` and `run_mutation_suite`. Report both, with the
   dataset id, report id and mutation report id, in the claims ledger.
6. A hardware evaluation follows `hand_labeling.md` for review and is reported
   separately; it can only cover the phenotypes the ontology marks supported on
   `go2_hardware`.

## Claim status

| claim | status |
|---|---|
| The evaluation machinery scores independently labelled episodes and refuses detector-derived labels | VERIFIED (tests in `tests/test_detector_eval.py`) |
| The evaluation catches all five standard mutants on the regression fixture | VERIFIED (`tests/test_detector_mutation.py`, guard-marked) |
| The regression fixture's numbers say anything about detector accuracy | NOT A CLAIM; the fixture is authored and its report is `fixture_regression` |
| Simulator ground-truth labels from termination terms and physical criteria are attainable with the current Phoenix state capture | INFERRED from the fields `snapshot_manager_state` exposes (termination terms, contact-sensor forces); foot velocity is not yet captured |
| Detector precision, recall, F1, false-positive rate or onset error on simulator or hardware data | NOT YET TESTED |
