# Methodology: causal failure reproduction and phenotype-conditioned repair

This document is the research story Ashfall is built to test, in the order the
evidence has to be produced. Each step names the code that implements it, the
gate that has to pass before the next step is allowed, and the status of the
step as of this revision. Status words are used strictly:

- **VERIFIED** means executed in this repository and observed, with the
  artifact or test named.
- **INFERRED** means read from source or derived by reasoning, not executed.
- **NOT YET TESTED** means the code exists and nothing has run it in the
  setting the claim would need.

The current status of every claim is kept in one place,
[`docs/claims_ledger.md`](claims_ledger.md); this document explains the design.

## 0. Vocabulary: cause, response, phenotype

The earlier design carried one string, `failure_mode`, that named the detector
threshold that fired, the treatment the curriculum was meant to teach, and the
physical cause it was meant to reproduce. Those are three objects.
`ashfall.ontology` separates them:

| object | meaning | examples |
|---|---|---|
| `Intervention` | a physical causal manipulation, applied and verified by readback | friction reduction, actuator weakening, terrain geometry, external perturbation, command corruption, payload mass |
| dynamical response | the trajectory that follows, compared against a matched no-intervention counterfactual | measured, never assumed |
| `PhenotypeSpec` | an observed behavioural failure, with per-platform observability | slip, collapse, stumble or blockage, command mismatch, contact loss, attitude loss |

No function maps an intervention to one phenotype. `PLAUSIBLE_PATHWAYS` lists
hypotheses, many to many, and the H0 gate tests them. A phenotype whose
ground-truth channels a platform lacks is *unsupported* there, and no detector
accuracy claim may be made for it on that platform. Two phenotypes are
excluded from the first confirmatory study for exactly that reason:
`contact_loss` (no inducing intervention the backend can apply, no gait-phase
ground truth, and contact force is an output of the physics, never a written
state) and `stumble` (no terrain geometry intervention is wired).

## 1. Observed physical or simulator failure

A failure is a `FailureEpisode`: one physics rollout with its policy identity,
intervention, command, initial-state hash, termination reason, windowed
phenotype observations, and a provenance mapping that must name the simulator
version, seed, environment configuration hash and backend. An episode that
cannot be traced is refused at construction.

Hand-authored trajectories remain in the repository under `data/fixtures/`
as regression material. Their manifest kind is `fixture` and
`ashfall.datasets.assert_scientific` refuses them wherever a scientific claim
would be made from data. Physics and hardware data live under
`data/scientific/` with a manifest that carries provenance and file hashes.

Status: VERIFIED for the schema and the fixture/scientific refusal on the
CPU suite. Simulator episodes: see the claims ledger, H0 row.

## 2. Phenotype definition and detection

`PhenotypeSpec.definition` states each phenotype behaviourally. Detection is
the threshold detector in `ashfall.taxonomy.detector`, wrapped by
`PhenotypeDetector`, which emits `PhenotypeObservation` records with an
`OnsetWindow`. A threshold detector only knows the frame at which its
criterion was first met, so its windows have `transition_start ==
established_start` and no precursor; ground-truth and reviewed labels may say
more, and the window machinery exists so they can.

Detector accuracy is an empirical question answered by
`ashfall.detector_eval` on independently labelled data (never labelled by the
detector's own thresholds), with per-phenotype precision, recall, F1, an
exclusive confusion matrix, false positives per episode and per minute, and
onset timing error. `run_mutation_suite` must catch five deliberate detector
defects before any evaluation is quoted. The bundled fixture dataset produces
a `fixture_regression` report that cannot claim validation.

Status: the mutation suite catches all five mutants on the fixture
(VERIFIED, `tests/test_detector_mutation.py`). Detector accuracy on
independently labelled physics or hardware data: NOT YET TESTED. See
[`docs/detector/validation_protocol.md`](detector/validation_protocol.md).

## 3. Causal intervention

An intervention is applied by a backend adapter that reads the simulator back
after the write and returns an `InterventionReceipt`. For the Phoenix backend
(`ashfall.backends.phoenix_interventions`): robot-shape friction via the
Phoenix material adapter with readback; command corruption via the velocity
command buffer with readback; an external perturbation as a root-velocity
write at a scheduled step with readback of the delta; actuator weakening by
scaling the actuator gains with readback of the ratio; payload mass through the
PhysX mass view with readback. Terrain geometry is reported unsupported: the
Isaac Lab task fixes the terrain and it cannot be edited per environment.

Status: adapters VERIFIED against CPU fakes; on the simulator, see the ledger.

## 4. Matched counterfactual reproduction

The unit of delivery evidence is a matched pair (`MatchedPairEvidence`): the
same restored initial state, policy, command, simulator seed, environment
configuration and horizon, run once with no intervention (control) and once
with the intervention (treatment), plus independent no-intervention replicates
of the same state under other seeds. On the Phoenix backend all arms run in
one persistent Isaac Lab scene so startup-mode domain randomisation is shared
by construction; the scene seed is recorded.

## 5. Validated delivery: D1, D2, D3

`ashfall.gates.evaluate_delivery` returns a `DeliveryVerdict` that is
`DELIVERED` only when all three gates pass; every other outcome names its
gate.

- **D1 intervention.** The treatment receipt's readback confirms the requested
  parameters and the control receipt confirms the nominal ones. No receipt, an
  unverified readback, or an unsupported kind fails D1. A departure without a
  verified cause is not evidence of causation.
- **D2 departure.** `ashfall.counterfactual.measure_departure` computes the
  Mahalanobis distance between treatment and control features per step, using
  only the seven physically restorable channels, standardised by a covariance
  estimated from the nominal replicates with Ledoit and Wolf shrinkage toward a
  scaled identity plus a small diagonal floor. The statistic (default: the 95th
  percentile over the window) is judged against the same statistic for every
  pair of nominal replicates. D2 passes when the treatment exceeds every null
  pair by the preregistered margin ratio. The floor is numerical
  regularisation, not a sensor resolution, and `sensitivity_analysis` reports
  whether the verdict survives a grid of shrinkage intensities and floors.
  Per-phenotype directional signatures are reported for interpretability; they
  are no longer the criterion.
- **D3 phenotype.** The intended phenotype is detected in the treatment arm
  and not in the control arm. A phenotype present in both arms is not
  attributable to the intervention.

H0 aggregates pairs (`ashfall.h0`): the preregistered `H0Spec` fixes the
pathway, the number of pairs, replicates, horizon, alpha, the minimum
delivered fraction and the D2 configuration, and is content-addressed. The
test is exact McNemar on the discordant pairs. Any pair whose intervention was
not verifiably applied makes the calibration UNINFORMATIVE. Fewer than six
pairs cannot reach alpha 0.05 and the result says so.

Status: VERIFIED end to end on the toy surrogate (mock evidence, never
scientific). Simulator: see the ledger.

## 6. Repair

Only DELIVERED pairs become capsules (`ashfall.harvest`). A capsule at schema
1.1 carries the intervention, the phenotype label, an onset window whose
precursor comes from the D2 departure profile, and provenance. Capsules are
the inputs to the repair arms.

The comparison is preregistered in
[`docs/phase2/PREREGISTRATION.md`](phase2/PREREGISTRATION.md): arm A
standard, B uniform failure replay, C intervention-parameter-conditioned
replay, D phenotype-conditioned repair, under one `ComputeBudget`. Every
training run records a `TrainingArtifact` (fresh transitions, replay
transitions, episodes, optimizer updates, rollout length, environments, wall
clock, GPU time, initial and final checkpoint hashes) and
`assert_equal_compute` rejects a comparison outside tolerance.

Status: protocol machinery VERIFIED on the CPU suite. No repair arm has been
trained under it: NOT YET TESTED.

## 7. Held-out evaluation

`ashfall.selection`: training produces candidate checkpoints; a
content-hashed `SelectionProtocol` chooses one on validation data; the
candidate SHA is frozen; the held-out suite is evaluated once; an
`ImmutableVerdict` records the protocol, Ashfall, Phoenix, policy,
validation-data, held-out-data and config hashes. The `HeldOutLedger` is an
append-only, hash-chained registry of every held-out evaluation; it refuses to
evaluate the same candidate twice and refuses a second candidate under the
same protocol and data without a one-use preregistered amendment.

The confirmatory endpoint (`ashfall.stats.primary`) is held-out failure
incidence aggregated over the preregistered validated phenotype set, one value
per training seed per arm, paired by seed, tested with the exact sign-flip
permutation test. Per-phenotype endpoints are secondary and Holm-corrected.
Evaluation episodes are never counted as replicates.

Status: VERIFIED as software. No held-out evaluation exists: NOT YET TESTED.

## 8. Nominal non-degradation

H4 is a preregistered margin, tested per nominal stratum with
`ashfall.evaluation.regression`. A degenerate zero-width interval (every
matched scenario delta identical, the normal state of a suite where both
policies succeed everywhere) is never accepted as proof that a budget is met:
binary metrics fall back to an exact Clopper-Pearson bound, continuous metrics
fail closed with a named reason.

Status: VERIFIED as software (`tests/test_paired_intervals.py`). NOT YET
TESTED on a trained candidate.

## The gate order

```
H0 causal delivery  ->  H1 phenotype fidelity  ->  H2 repair  ->  H3 generalization  ->  H4 non-degradation
```

`ashfall.protocol.hypotheses.HypothesisLedger.can_execute` refuses a
hypothesis whose prerequisite is not PASS, and refuses H1 to H4 unless H0 is
PASS for every phenotype retained in the study. The milestone before any
adaptation experiment is therefore explicit: **H0 passes for every retained
phenotype**, on simulator evidence, with the bundle committed.

## Evidence bundles

Every scientific run writes one content-addressed bundle
(`ashfall.provenance.EvidenceBundle`): `manifest.json`, `environment.json`,
`provenance.json`, `metrics.json`, `verdict.json`, `index.json` and `logs/`.
`finalize` refuses a bundle that cannot name its Ashfall revision, Phoenix
revision (or the reason it is absent), policy hash, config hashes and seed.
See [`docs/data_provenance.md`](data_provenance.md).

## What this methodology does not do

- It does not claim the threshold detector is accurate. D3 inherits whatever
  accuracy the detector has, and that accuracy is unmeasured on physics data.
- It does not reconstruct hidden actuator state or controller history from a
  single restored row; the Phoenix reset telemetry labels each restore
  `state_only_seed` unless controller history was supplied.
- It does not equate robot-shape friction with surface friction, or a
  simulator intensity axis with a physical coefficient.
- It does not run hardware. The hardware capture path lacks a validated
  ground-relative height and calibrated foot contact, so collapse, slip and
  contact loss are unsupported phenotypes on the robot until that changes.
