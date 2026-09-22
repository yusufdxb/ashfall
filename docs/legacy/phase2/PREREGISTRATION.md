# Ashfall Phase II: preregistration

Status: **PREREGISTERED, NOT YET RUN.** No Phase-II experiment has executed. No H0, H1, H2,
H3 or H4 verdict exists in any ledger. Every number in this document is either a protocol
constant, an explicitly labelled placeholder awaiting justification, or absent.

This document is the protocol. The code that enforces it lives in `src/ashfall/protocol/`
(arms, compute budget, hypothesis ledger), `src/ashfall/selection.py` (checkpoint selection,
held-out ledger, immutable verdicts), `src/ashfall/gates.py` and `src/ashfall/counterfactual.py`
(the H0 delivery gates), and `src/ashfall/stats/` (the confirmatory tests). Where the document
and the code disagree, the code is wrong and must be fixed before anything runs; where a value
below is marked PLACEHOLDER, it must be replaced and the replacement recorded as a protocol
amendment before registration is frozen.

## 1. Vocabulary

Every artifact uses the ontology in `src/ashfall/ontology.py`:

    Intervention (cause) -> dynamical response -> failure phenotype (effect)

An **intervention** is a physical causal manipulation applied to the simulator or robot:
friction reduction, actuator weakening, terrain geometry, external perturbation, command
corruption, payload mass. A **phenotype** is an observed behavioural failure: slip, collapse,
stumble/blockage, command mismatch, contact loss, attitude loss. No one-to-one mapping between
the two is assumed; `PLAUSIBLE_PATHWAYS` lists hypotheses and H0 tests them.

The **retained phenotype set** for the first confirmatory study is
`ashfall.ontology.FIRST_STUDY_PHENOTYPES`, which at registration reads
`slip`, `collapse`, `command_mismatch`, `attitude`. `contact_loss` and `stumble` are excluded
because no supported inducing intervention and no independent ground truth exist for them on the
current backend; the exclusion reasons are recorded on their `PhenotypeSpec`. A phenotype leaves
the retained set if its H0 verdict is not PASS or if its detector validation (section 7) fails.
It never enters the primary endpoint afterwards, and its removal is a recorded amendment.

## 2. Hypotheses and the gating rule

| id | hypothesis | prerequisite | recorded |
|---|---|---|---|
| H0 | **Causal delivery.** A physical intervention induces the intended phenotype significantly more often than a matched no-intervention counterfactual. | none | per phenotype |
| H1 | **Phenotype fidelity.** Reproduced failures dynamically resemble independently labelled reference failures. | H0 | per phenotype |
| H2 | **Repair.** Failure-phenotype-conditioned adaptation reduces held-out failure incidence versus equal-compute non-conditioned alternatives. | H1 | once |
| H3 | **Generalization.** Repair generalizes to withheld causes and intensities that produce the same phenotype. | H2 | once |
| H4 | **Non-degradation.** Repair does not exceed a preregistered nominal-performance degradation margin. | H3 | once |

Gating rule, enforced by `HypothesisLedger.can_execute`:

1. H(n) may not execute unless H(n-1) is PASS.
2. H1 to H4 may not execute unless H0 is PASS for **every** phenotype in the retained set. A
   phenotype whose H0 is FAIL, UNINFORMATIVE or NOT_RUN must be removed from the retained set
   (an amendment) before anything downstream runs.
3. UNINFORMATIVE is a verdict, not a pass. Phase I was a correctly executed measurement of a
   treatment that was never applied; that outcome now has a name and unlocks nothing.
4. Verdicts are appended to a hash-chained ledger. A settled PASS or FAIL can only be changed
   with a preregistered amendment id, used once.

H4 is evaluated on the same candidate as H2. Because the chain is linear, an H3 FAIL blocks the
confirmatory H4 verdict; the H4 measurement is still reported descriptively in that case, labelled
as not a confirmatory result.

### H0: causal delivery

Unit: one **matched pair**, control and treatment rollouts sharing the restored initial state,
policy checkpoint, applied command, simulator seed, environment configuration hash and horizon,
differing only in the intervention. Three gates decide whether a treatment rollout delivered:

* **D1 intervention.** The backend's readback confirms the requested physical manipulation was
  applied (for friction: material coefficients read back within tolerance; for command
  corruption: command buffer readback; for perturbation: root velocity readback at the push
  step). No receipt, no delivery.
* **D2 departure.** The treatment trajectory departs from its matched control on the
  physically restorable channels by more than independent nominal rollouts depart from one
  another, measured as a multivariate distance under a shrinkage covariance estimated from those
  nominal rollouts, with a sensitivity analysis over the regularization value. Per-phenotype
  directional metrics are reported alongside as interpretable diagnostics, not as the criterion.
* **D3 phenotype.** The independently defined intended phenotype is observed in the treatment
  rollout (and its onset window recorded), and is not observed in the control.

`DELIVERED` requires D1 and D2 and D3. Every other outcome names the gate that failed.

Endpoint per phenotype: over N matched pairs, the paired treatment-versus-control phenotype
incidence. Test: exact binomial test on discordant pairs (treatment-only versus control-only),
one-sided in the direction of the intervention, at alpha 0.05. PASS additionally requires the
delivered fraction (D1 and D2 and D3) to reach a minimum. PLACEHOLDER: minimum delivered fraction
and N are not set; they require a pilot on the simulator that has not run. They must be fixed
before H0 is executed and recorded as an amendment.

### H1: phenotype fidelity

Reference failures are episodes labelled by simulator ground truth or human review, never by
the detector under test. Distance between reproduced and reference trajectories on restorable
channels is compared against the reference-to-reference distance distribution. PLACEHOLDER: the
acceptance quantile is not set.

### H2: repair (primary confirmatory hypothesis)

See sections 3 to 6.

### H3: generalization

Withheld intervention kinds and withheld intensity bins are declared before training. The
held-out incidence on withheld cells uses the same pairing and test as H2.

### H4: non-degradation

Per nominal stratum, one-sided bounds on success, tracking error and intervention rate against
the registered margin. A degenerate (zero-width) resampled interval cannot pass; an exact bound
is required in that case. PLACEHOLDER: margins (`nominal_success_drop`, `tracking_error_increase`,
`intervention_rate_increase`) require laboratory task justification; the values in
`RegressionBudget` defaults are software defaults and are not registered.

## 3. Arms

`ashfall.protocol.arms.default_arm_set` builds the registered arms:

| arm | kind | conditioning | replay |
|---|---|---|---|
| A | standard | none | none; the baseline continues training at the same budget |
| B | uniform failure replay | none | resets seeded from the capsule pool, every capsule equally likely |
| C | intervention-conditioned replay | intervention kind and intensity bin | resets stratified by the physical cause that produced each capsule |
| D | phenotype-conditioned repair | phenotype | resets stratified by the observed failure phenotype |

Arm C may name only intervention kinds; arm D may name only retained phenotypes; a name from the
other vocabulary is refused at construction. Seed rows inside a capsule are chosen inside the
recorded onset window (`phenotype_window_fraction`, `transition_start` or `precursor_start`);
row 0 is not a strategy. `replay_fraction` is a protocol input. PLACEHOLDER: its value (0.5 in
the default set) is a software default and requires a pilot before registration.

The exact set of final baselines is configurable through `ArmSet`; what is not configurable is
that exactly one standard arm exists and every arm shares one `ComputeBudget`.

## 4. Compute budget

Every training run emits a `TrainingArtifact` recording fresh environment transitions, replay
transitions, episodes, optimizer updates, rollout length, number of environments, wall-clock
time, GPU time, the initial checkpoint SHA256, the final checkpoint SHA256 and the hashes of its
configs. `assert_equal_compute` refuses a comparison whose artifacts violate the budget's
tolerances:

| field | tolerance |
|---|---|
| fresh_env_transitions | exact |
| replay_transitions | exact for arms B, C, D; exactly zero for arm A |
| optimizer_updates | exact |
| rollout_length | exact |
| num_envs | exact |
| episodes | recorded, not enforced (termination frequency is an outcome) |
| wall_clock_s, gpu_time_s | recorded; PLACEHOLDER relative tolerance to be set from a pilot |

All arms start from one shared initial checkpoint and every arm is trained on the same set of
training seeds; a missing (arm, seed) cell is a violation. Replay transitions are a subset of the
fresh transitions for a reset-seeded curriculum, not an addition to them.

## 5. Primary and secondary endpoints, statistics

**Primary endpoint (one, confirmatory):** held-out failure incidence aggregated over the retained
phenotype set, defined per training seed as the fraction of held-out episodes in which any
retained phenotype is observed. The comparison of interest is arm D versus the best-performing of
arms A, B and C, where "best" is fixed by a rule declared here: PLACEHOLDER, the comparator rule
(pre-specified single comparator versus each comparator with correction) must be chosen before
registration.

**Experimental unit:** the training seed. Each arm is trained once per seed on a preregistered
seed list; the same seeds are used for every arm (paired by seed). Evaluation episodes within a
seed are averaged into that seed's incidence and are never counted as independent replicates.

**Test:** paired-by-seed differences, exact two-sided sign-flip permutation test over the
2^n sign assignments (`ashfall.stats`), alpha 0.05. The exact test's smallest attainable p is
2 / 2^k for k sign-carrying seeds; the seed count must make alpha reachable. PLACEHOLDER: n seeds
and the seed list are not set. Phase I used 11 seeds; a power calculation on the pilot is
required before registration.

**Secondary endpoints:** per-phenotype held-out incidence, one test per retained phenotype, Holm
step-down correction across the family of secondaries. Frontier and R50 estimates are
descriptive: fitted crossings are reported as interpolated, plateaus as intervals,
non-monotone support as unidentifiable.

**Terminology:** episode incidence is the fraction of episodes in which a phenotype occurs.
Recurrence means a phenotype that occurred, resolved, and occurred again within an episode, and
is reported separately when measured.

## 6. Model selection and held-out protection

    training -> candidate checkpoints -> validation selection
             -> freeze candidate SHA256 -> one held-out evaluation
             -> immutable verdict

* The selection rule (`SelectionProtocol`: rule, metric, tie-break, validation split hash) is
  part of this protocol and its content hash is part of the protocol hash. PLACEHOLDER: rule
  and metric are not chosen; `max_validation_metric` on validation success rate with
  `earliest` tie-break is the software default and is not registered.
* Selection refuses validation data whose hash equals the held-out data hash.
* Every held-out evaluation is opened in the append-only `HeldOutLedger` before it runs. The
  ledger refuses to evaluate the same candidate twice and refuses a second candidate under the
  same protocol and held-out data without a preregistered amendment id, which can be spent once.
* The verdict (`ImmutableVerdict`) carries the protocol hash, selection record hash, Ashfall
  commit, Phoenix commit, policy SHA256, validation data hash, held-out data hash and config
  hashes, is written once, and is re-verified from disk before it is quoted.

Repeated held-out screening cannot be made impossible by software; what the ledger guarantees is
that it cannot happen without leaving a record that contradicts this document.

## 7. Detector validation prerequisite

The primary endpoint is computed from detected phenotypes, so a phenotype enters the retained set
only if the detector's precision and recall for it, measured on an evaluation set whose labels
were produced without the detector's thresholds (simulator ground truth or human review),
reach a declared level. PLACEHOLDER: the levels are not set. The evaluation protocol is
`docs/detector/validation_protocol.md`; the mutation suite that shows the evaluation is sensitive
to always-collapse, always-stumble, always-command-mismatch, removed slip priority and label
swaps must pass on the registered evaluation set.

## 8. Provenance

Every run emits an evidence bundle (`ashfall.provenance.EvidenceBundle`) with manifest,
environment, provenance, metrics and verdict files and a logs directory, recording the Ashfall
SHA, Phoenix SHA (registered integration: `feat/causal-viability-replication`, see
`ashfall.backends.phoenix_compat`), Python package set and hash, Isaac Lab and Isaac Sim
versions, CUDA and GPU metadata, config hashes, dataset hashes, policy hashes and seeds. A bundle
missing any of these is refused at `finalize`. Only data under `data/scientific/` with a
scientific manifest may enter any Phase-II analysis; fixtures are refused.

## 9. Kill criteria

* H0 FAIL for every phenotype in the retained set: the study stops; the result is reported as a
  negative delivery result, not as a negative repair result.
* Detector validation below the registered level for every retained phenotype: the study stops
  before H2.
* Any compute-protocol violation in the A to D comparison: the comparison is discarded and rerun;
  it is never reported with a caveat.
* A held-out ledger entry without a matching amendment: the H2 verdict is void.

## 10. What is VERIFIED, INFERRED, NOT YET TESTED

**VERIFIED** (executed on this machine, CPU): the protocol code refuses unequal compute, refuses
phenotype/intervention conflation in arm definitions, refuses selecting on held-out data,
refuses repeated held-out screening without an amendment, refuses executing H1 to H4 without
H0 PASS for every retained phenotype, and detects tampering in both ledgers. See
`tests/test_protocol_arms.py`, `tests/test_selection_ledger.py`, `tests/test_hypothesis_gates.py`.

**INFERRED** (from source, not executed): that Phoenix's `OnPolicyRunner` consumes
`iterations x num_envs x num_steps_per_env` transitions and performs
`iterations x epochs x mini_batches` updates, which `artifact_from_rsl_rl` assumes.

**NOT YET TESTED:** every simulator path. No matched pair has run on Isaac Lab under this
protocol, no training artifact has been produced by a real run, no candidate has been selected,
no held-out evaluation has been opened. Every PLACEHOLDER above is open.

## 11. Amendments

None. The first amendment will be the one that replaces the placeholders in sections 2, 3, 4, 5,
6 and 7 with justified values, after the H0 pilot and before any Phase-II run.
