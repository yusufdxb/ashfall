# Ashfall simplification audit

Audited: Ashfall `37b61f1` (branch `research/causal-failure-repair-v3`, identical to public
`main`), 283 tracked files, 68 modules under `src/ashfall/` (about 17,300 lines), 43 test files,
550 passing CPU tests (run on this machine before any change, output: `550 passed, 1 warning in
8.82s`; `ruff check .` clean). Sibling go2-phoenix read (not modified) at its checked-out branch
`feat/causal-viability-replication` @ `9df76d7`, which carries uncommitted work from another
session. Every module listed below was read; the Phase-I defect was traced to the Phoenix commit
that fixed it.

## 1. The finding in one paragraph

Ashfall already contains most of Failure-Boundary Replay. The v2 design (April to September
2026) built a failure capsule, a reproduction gate, a frozen basin of local scenarios with split
discipline, an isotonic failure frontier, and a curriculum that sampled nominal, frontier and
hard resets (`capsule.py`, `reproduction.py`, `basin.py`, `scenarios.py`, `frontier.py`,
`repair.py`, `training.py`). The v3 redesign then put a causal-inference layer in front of it
(six phenotypes, five interventions, D1/D2/D3, H0 to H4, detector validation) and made every
downstream step wait on it. The science did not need that layer to be the headline. The
simplification is therefore mostly a re-ordering: the frontier-repair core becomes the paper,
the v3 gates shrink to three named checks, and the ontology and hypothesis chain move out of
the main narrative. Very little code has to be deleted, and none is.

## 2. Classification of every major concept

Legend: **CORE PAPER** (in the main story), **SUPPORTING VALIDATION** (runs in the pipeline, is
cited in methods, not in the headline), **APPENDIX** (reported in supplementary material),
**FUTURE WORK** (kept, not used in the first study), **LEGACY** (historical record, preserved
unchanged), **REMOVE FROM MAIN STORY** (still in the code, never in the README or paper body).

### Concepts named in the brief

| concept | classification | reason |
|---|---|---|
| multiple failure phenotypes (slip, collapse, stumble, command mismatch, contact loss, attitude) | FUTURE WORK | The first study has one failure family (low-friction traverse failure). The stock GO2 cannot observe slip or collapse directly anyway (`ontology.PHENOTYPES` observability); the endpoint is a behavioural traverse criterion (`fbr/criterion.py`). |
| causal interventions (friction, actuator weakening, perturbation, command corruption, payload) | CORE for friction only; FUTURE WORK for the rest | Friction is the one physical axis. The readback adapters for the others stay tested. |
| matched counterfactuals | SUPPORTING VALIDATION | The acceptance check compares each treated replay with its matched untreated control from the same restored state and seed. That is the matched-pair idea, used once, for one purpose. |
| D1 / D2 / D3 delivery gates | CORE, renamed and restated | D1 becomes **delivered** (friction read back). D3 becomes **failure match** (treated meets the same criterion near the recorded onset, control does not). D2's role becomes **trajectory match** (treated closer to the recording than control). The Ledoit-Wolf Mahalanobis departure against nominal replicates (D2 as implemented) moves to the APPENDIX as a stricter variant. |
| detector validation (mutation suite, per-phenotype precision/recall) | FUTURE WORK | Needed only when the endpoint is a detected phenotype. The FBR endpoint is traverse failure under a predeclared attitude/termination criterion. |
| H0 to H4 hypothesis chain | LEGACY (preregistration archived unchanged); code REMOVE FROM MAIN STORY | Replaced by one primary hypothesis, one key secondary in fixed sequence, and one non-degradation condition (`docs/research/PREREGISTRATION.md`). |
| multiple repair arms (A standard, B uniform replay, C intervention-conditioned, D phenotype-conditioned) | REMOVE FROM MAIN STORY | Replaced by A no repair, B broad DR, C nominal-seed boundary replay, D FBR, E friction-band ablation. `protocol/arms.py` stays for its conflation guards. |
| held-out locking (`selection.py`, `HeldOutLedger`, `ImmutableVerdict`) | SUPPORTING VALIDATION | One frozen held-out manifest, evaluated once per (arm, seed). No checkpoint selection (final checkpoint only), which is simpler and neutral across arms. |
| provenance systems (`provenance.py`, `EvidenceBundle`, `datasets.py`, `config_guard.py`) | SUPPORTING VALIDATION | The `evidence/` role in the new architecture. Never in the headline, always in the bundle. |
| frontier analysis (`frontier.py`) | CORE PAPER | The isotonic crossing is how a failure boundary is located per seed state (`fbr/boundary.py`). Its censoring vocabulary is why a curriculum is never centred on an unidentified boundary. |
| nominal non-degradation (`evaluation/regression.py`, exact bounds, degenerate-interval refusal) | CORE PAPER (as a condition) | The repaired policy must stay inside a preregistered nominal margin. |
| causal reproduction (H0 calibration, exact McNemar, `h0.py`) | APPENDIX | A friction-to-failure delivery calibration is a useful supplementary sanity check. It is no longer a gate on the study. |
| phenotype-conditioned repair (the v3 headline) | LEGACY | The old primary question. Not tested, not claimed. |

### Modules

| module | classification | note |
|---|---|---|
| `capsule.py` (`FailureCapsule`, `CapsuleFrame`, `capsules_from_rows`) | CORE PAPER | The capsule object. FBR adds a narrow contract in `fbr/capsule.py` instead of a second type. |
| `fbr/criterion.py`, `fbr/capsule.py`, `fbr/boundary.py`, `fbr/acceptance.py` | CORE PAPER (new) | Onset extraction, seed states, boundary per seed, sampler, three checks. |
| `fbr/toy_slip.py`, `fbr/toy_study.py` | CORE PAPER (mechanism section) | CPU mechanism study. Evidence kind `toy_mechanism`. |
| `frontier.py` | CORE PAPER | See above. |
| `scenarios.py`, `basin.py` | SUPPORTING VALIDATION | Content-addressed scenarios and split-leakage refusal; the Phoenix reset bridge consumes `Scenario` objects. |
| `repair.py` (`RepairCurriculum`, `FrontierSchedule`) | SUPPORTING VALIDATION | The v2 predecessor of `BoundarySampler`; its scenario interface is the Isaac Lab plumbing. The concept (nominal/frontier/hard fractions) is superseded. |
| `training.py` (`run_repair`) | CORE PAPER infrastructure, NOT YET RUN | The Isaac Lab PPO path with scenario resets and in-loop frontier probes. Written, never executed. |
| `reproduction.py` (`ReproductionGate`, Halton search, `FailureSimilarity`) | SUPPORTING VALIDATION | The friction search that proposes candidate values for the hardware capsule; the acceptance checks decide. |
| `backends/phoenix.py`, `backends/phoenix_interventions.py` (`MaterialWriter` with readback) | CORE PAPER infrastructure | Restore, friction write and readback on Isaac Lab. Verified once by the H0 smoke. |
| `backends/phoenix_compat.py` | SUPPORTING VALIDATION | Pins the Phoenix restore contract. |
| `backends/toy.py` (planar surrogate) | SUPPORTING VALIDATION | CPU tests of the matched-pair machinery. |
| `protocol/budget.py` (`ComputeBudget`, `TrainingArtifact`, `assert_equal_compute`) | CORE PAPER | Equal compute is the fairness condition of every comparison. |
| `stats/primary.py` (exact sign-flip, `paired_seed_effect`) | CORE PAPER | The primary test. |
| `evaluation/paired.py`, `evaluation/regression.py` | CORE PAPER | Non-degradation bounds. |
| `selection.py` | SUPPORTING VALIDATION | Held-out ledger. |
| `provenance.py`, `datasets.py`, `config_guard.py` | SUPPORTING VALIDATION | Evidence bundles. |
| `ontology.py` | SUPPORTING VALIDATION | `Intervention` and `InterventionReceipt` are reused by the delivered check; the phenotype registry is FUTURE WORK. |
| `counterfactual.py`, `gates.py`, `h0.py`, `harvest.py` | APPENDIX | H0 calibration and the Mahalanobis departure test. |
| `delivery.py` | SUPPORTING VALIDATION (`implausible_reset_fields`), LEGACY (single-feature z-test) | The implausible-state refusal is reused by `fbr/capsule.py`. |
| `detector_eval/`, `taxonomy/phenotype_detector.py`, `taxonomy/labeling.py` | FUTURE WORK | Six-phenotype detection. |
| `protocol/arms.py`, `protocol/hypotheses.py` | REMOVE FROM MAIN STORY | Kept for their tests and guards. |
| `experiment/`, `analysis/`, `synth/`, `evaluation/harness.py`, `evaluation/metrics.py`, `evaluation/protocol.py`, `evaluation/episode_records.py`, `taxonomy/detector.py`, `workflow.py`, `demo.py` | LEGACY | Phase I and v2 artifact commands. Still tested, never cited. |
| `cli.py` | SUPPORTING VALIDATION | `h0`, `detector-eval` and the v2 artifact commands remain; FBR's study runs as `python -m ashfall.fbr.toy_study`. |

### Data and documents

| item | classification |
|---|---|
| `results/legacy_row0_curriculum/` (Phase I, n = 11) | LEGACY, byte-preserved |
| `results/software_validation/h0_isaac_smoke_v3b/` | APPENDIX (implementation smoke, verdict FAIL) |
| `data/fixtures/` | SUPPORTING VALIDATION (tests only, never evidence) |
| `docs/phase2/` (H0 to H4 preregistration, hypothesis, inputs) | LEGACY, moved unchanged to `docs/legacy/phase2/` |
| `docs/methodology.md` (v3 causal methodology) | LEGACY, moved unchanged to `docs/legacy/methodology_v3_causal.md`; replaced |
| `docs/architecture/ashfall_v2.md`, `ashfall_v3.md` | APPENDIX (supporting infrastructure design) |
| `docs/experiments/low_friction_protocol.md` | LEGACY (already marked superseded) |
| `docs/detector/*` | FUTURE WORK |
| `docs/hardware/controlled_failure_protocol.md` | SUPPORTING VALIDATION, refined by `docs/research/HARDWARE_DEMO.md` |
| `docs/audits/*` | LEGACY (history) |
| `docs/runbooks/h0_isaac.md` | APPENDIX |
| `docs/go2_field_notes.md` | SUPPORTING VALIDATION |
| `docs/claims_ledger.md`, `docs/limitations.md` | CORE, rewritten around FBR with the old rows kept |

## 3. The old bug, validated

The claim to validate: "the earlier Ashfall/Phoenix curriculum work seeded nominal states rather
than actual failure states." It is true, and there were four compounding defects, not one.

**What happened.**

1. **Row 0 was hardcoded.** Phoenix's reset bridge cache, before go2-phoenix commit `057e541`
   (2026-05-17, "H0 bridge fix: configurable seed-row + opt-in velocity write"), called
   `load_initial_state(path, row=0)` for every pool entry (verified: `git show
   057e541^:src/phoenix/adaptation/reset_bridge.py`, line 38). Row 0 of every shipped synthetic
   trajectory is the start of an upright trot; the synthetic slip file first raises
   `failure_flag` at row 70, 1.4 s later at 50 Hz. The commit message records this.
2. **Velocities were dropped.** The same commit records that `base_lin_vel_body` and
   `base_ang_vel_body` were parsed but never written to the simulator, so every seeded
   environment started from rest.
3. **The cause was never restored.** Even at the correct row, a seeded reset restores
   kinematics only. The low friction (or other cause) that produced the recorded failure was
   not applied: Phoenix's `resolve_seed` still returns `"environment_context_restored": False`
   (verified in the current `reset_bridge.py`). A slip state replayed on a high-friction floor
   is a state on a high-friction floor.
4. **The trajectories were not physics.** The pool was 18 hand-authored synthetic parquets,
   authored around the detector's thresholds (`docs/legacy/README.md`).

**Why the intervention was not delivered.** The treatment, "train on failure states", never
reached the simulator: about half the environments reset to a jittered upright stand, at rest,
on the arm's ordinary friction. The n = 11 null (slippery -0.4155 pp, p = 0.726562) is a correct
measurement of a treatment that was not applied, which is why it is uninformative about
failure curricula.

**How FBR prevents each defect.**

| defect | FBR guard | where |
|---|---|---|
| far seed row (row 0) | a seed must lie strictly before onset and within `max_lead_s` (1.0 s) of it; row 0 gets no exemption, and a far request is refused, never clamped | `fbr.capsule.assert_near_onset`, `seed_points` |
| seed already at the failure | a seed that already meets the failure criterion is refused (reproducing it would be tautological) | `fbr.capsule.seed_points` |
| missing or zero-filled state | missing reset channels and physically impossible values (base height exactly zero) are refused | `delivery.implausible_reset_fields` via `seed_points` |
| cause not restored | every boundary draw carries its friction value; the **delivered** check requires the friction to be read back from the simulator | `fbr.boundary.ReplayDraw`, `fbr.acceptance` |
| replayed state does not lead to the failure | the **failure match** check requires the treated replay to meet the same criterion within tolerance of the recorded onset while the matched full-friction control does not | `fbr.acceptance.check_pair` |
| fixtures as data | unchanged: fixture manifests are refused wherever a scientific claim is made | `datasets.assert_scientific` |

**Regression tests** (`tests/test_fbr.py::TestPhaseOneRegression`, all passing):

- `test_row_zero_far_before_onset_is_refused`: a capsule whose reviewed window includes row 0,
  1.4 s before onset, refuses row 0.
- `test_lead_beyond_the_boundary_window_is_refused`: a 1.2 s lead is refused under a 1.0 s
  window.
- `test_toy_replay_from_fbr_seed_delivers_the_failure_and_row_zero_does_not`: in the slip
  cart-pole, the first friction-attributable patch failure is captured; replayed from the FBR
  seed (0.4 s before onset) under the patch friction, at least 2 of 3 replicates fail within the
  boundary window; replayed from row 0 (the Phase-I choice) in the same window, at most 1 of 3
  do; and `assert_near_onset` refuses row 0.

This defect motivates the rigour of the new design ("failure replay is only meaningful if the
failure boundary is actually delivered"). It is not the contribution.

## 4. What changed in the repository, and what did not

- **Added:** `src/ashfall/fbr/` (criterion, capsule contract, boundary and samplers, acceptance,
  toy study), `tests/test_fbr.py`, `docs/research/`, the new preregistration, the rewritten
  README, methodology, architecture page, claims ledger and limitations.
- **Moved unchanged:** `docs/phase2/` to `docs/legacy/phase2/`; the v3 methodology to
  `docs/legacy/methodology_v3_causal.md`.
- **Not deleted:** any module, test, result or evidence file. The 550 prior tests still pass.
- **Not changed:** go2-phoenix (read only; it carries another session's uncommitted work).
