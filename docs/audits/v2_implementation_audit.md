# Ashfall v2 implementation audit

Audited source: Ashfall `667158e` on `research/ashfall-mode-conditioned-adaptation`, working tree clean.
Simulator sibling: go2-phoenix `9434269` on `feat/causal-viability-replication`, which is the branch the
editable install resolves to. Method: five independent module audits, every finding carried to a quoted
line and, where the claim was executable, to a CPU probe whose output is recorded. Isaac Lab, GPU and
hardware were not run, so every simulator-runtime statement below is read from source and labelled.

The audit was commissioned after Gate A (hypothesis H0, delivery) ran for the first time and failed. It
asks one question: does the v2 rebuild fix the defects that made the Phase-I result uninformative, or
repeat them?

## Verdict

The rebuild fixed the bookkeeping and none of the delivery defects. Content-addressed identity,
write-once artifacts, a real reproduction gate, split-leakage refusal and exact pairing are all genuine
and tested. The three mechanisms that decide whether a counterexample is delivered at all are unchanged,
and the executable path that would exercise them cannot run.

## A. The executable pipeline cannot run

| # | Finding | Evidence |
|---|---|---|
| A1 | `PhoenixBackend` has no `evaluate` method, and `evaluate` is the only producer of episode evidence | `cli.py:152` calls `backend.evaluate(...)`; the class exposes `close`, `evaluate_scenario`, `evidence_kind`, `replay`. `git log -S "def evaluate(self, manifest"` finds nothing on any branch |
| A2 | The frontier-probe loop invokes A1 with `check=True` before the first PPO iteration | `training.py:130-138` spawns `ashfall evaluate`; `scenario_bridge.py:149-156` calls `reestimate` at `completed=0`. `candidate.pt` is unreachable |
| A3 | `create_backend` splats the config unfiltered, so a provenance key is fatal | `phoenix.py:456` is `PhoenixBackend(**config)`; `__init__` accepts no `**kwargs`. Adding `simulator_version`, the key `training.py:59` reads from that same dict, raises `TypeError` |
| A4 | The verdict gate can never accept real backend evidence, and accepts the mock's | `protocol.py:349` requires `environment_parameters == dict(scenario.parameters)` exactly; `phoenix.py:396-403` injects ten metadata keys into that field. `protocol.py:365` requires `nominal_group`, which is not an `EpisodeOutcome` field. `demo.py:94` writes the one shape that passes |
| A5 | No example backend config or repair spec exists anywhere in the repo | No `.json`, `.yaml` or `.md` in the tree carries the keys the executable path requires |
| A6 | The Isaac paths have never produced an artifact | No `probe_*.json`, `candidate.pt`, `reset_telemetry.jsonl` or episode JSONL in any commit on any branch, and `write_artifact` refuses overwrite, so a real run would have left durable files |

A4 is the trap: fixing A1 to A3 and obtaining a clean simulator run would still return `accepted=False`,
with a reason implying the simulator drifted off the frozen parameters.

## B. The delivery defect is repeated, not fixed

| # | Finding | Evidence |
|---|---|---|
| B1 | The guard against seeding inside the nominal prefix is dead by construction | `capsule.py:198` defaults `offset_seconds=0.5`; `capsule.py:263` defaults `pre_failure_seconds=0.5`. Equal defaults make `resolve_seed_index()` return exactly `pre_failure_start_index`, so the check at `capsule.py:218` can never fire. It validates the offset against a copy of itself |
| B2 | Five of six modes still seed a nominal walking state | Measured from this repo's own generator: onsets 50, 55, 62, 69, 70, 80 for stumble, contact_loss, collapse, attitude, slip, command_mismatch. A 25-row rollback lands in the stable prefix for all but command_mismatch. Seeded frames read height 0.30 m, forward speed 0.52 m/s, max joint speed 5 rad/s, minimum contact force 45 N |
| B3 | `n_stable=50` is source-hardcoded with no config surface | `generator.py:99,127,154,185,211,237`; `generate_all_failures` never passes it (`:306-309`). Failure development time spans 5 to 30 rows, a factor of six, against one global rollback |
| B4 | A test pins the falsified seed row as the expected answer | `tests/test_capsule.py:24` asserts `resolve_seed_index() == 25`, inside a test named `test_original_row_zero_bug` |
| B5 | The legacy runner writes back the row-0 strategy with no comment and no test | `runner.py:168` sets `seed_row_strategy="first"`, overriding Phoenix's `failure_onset_minus_seconds` default at `fine_tune.py:195`. Reachable only from the three legacy shell scripts, never from the v2 CLI. `grep seed_row_strategy tests/` is empty, and `test_experiment.py:221-231` opens that generated file and asserts only `failure_sample_fraction` |
| B6 | No distinctness or direction check exists anywhere | Nothing in `src/` compares a resolved seed state to the simulator's nominal reset distribution, or tests that it moves toward the failure |
| B7 | The canonical test fixture has no physical content | `demo.py:47-53` returns 80 frames byte-identical modulo timestamp, so its seed row 25 equals row 0. It is the fixture for 36 tests across three files |
| B8 | A later capsule's pre-onset window can lie inside an earlier failure | `capsule.py:335` hands every capsule the full frame set; measured, a stumble window's rows 35 to 39 are slip-failure rows |

## C. Hardware captures and undeliverable modes

| # | Finding | Evidence |
|---|---|---|
| C1 | A present-but-identically-zero capture is accepted as a reset state | `capsule.py:84-98` tests `is None` only, and zero is finite, so `_vector` passes it. Executed: `missing_reset_fields` is empty and `reset_frame()` accepts `base_pos=(0,0,0)`. Only `base_quat` is protected, by its unit-norm check |
| C2 | The repo already has the right guard, in the wrong layer | `_is_zero_filled` at `analysis/recurrence.py:121-131` refuses zero-fill with tests at `test_recurrence.py:111-125`. It is never applied to the capsule or reset path |
| C3 | `contact_loss` is undeliverable by construction | The restore contract is `InitialState`'s seven fields and `state_adapter.py:86-91` issues exactly root pose, root velocity, joint state and command writes. `generator.py:221-231` perturbs only `contact_forces` for this mode, so its entire signature lies outside the writable set |
| C4 | Nothing anywhere records deliverability | `grep -rn "deliverab" src/ tests/ docs/` was empty before this audit. All six taxonomy entries carry the identical `reproduction_status="requires_capsule_reproduction_gate"`, which is false a priori for `contact_loss` |
| C5 | An all-zero `contact_forces` capture reads as the `contact_loss` signature | A broken capture is indistinguishable from the failure mode it would be used to seed |
| C6 | Provenance can be laundered by passing a keyword | `capsule.py:316-321` stamps `onset_label_source="explicit_review"` because `onset_indices` was not `None`, and defaults a missing mode to `"unclassified"`. An unlabelled zero-filled capture acquires a human-review provenance string |

## D. Statistics

| # | Finding | Evidence |
|---|---|---|
| D1 | The BCa acceleration sign is inverted, in the one place the comment claims otherwise | `evaluation/significance.py:158-160` claims it "recovers the right acceleration sign". `pseudo = concatenate([jack_a.mean()-jack_a, jack_b-jack_b.mean()])` is the negative of the Efron and Tibshirani influence values. Measured ratio against that convention: exactly -1.0000, magnitude correct to four figures. On a near-ceiling configuration typical of this repo the repo interval includes zero where the correct BCa interval excludes it |
| D2 | `paired_delta` fails open, and the documented reproduction command now reaches that path | `analysis/multiseed.py:266-280` returns `mean 0.0`, `CI [0,0]`, `p=1.0` for an empty pairing instead of raising. Executed: the path named in the archived analysis note yields `n=0, mean +0.000 pp, p=1.0000`, while the archive yields the true `n=11, -0.416 pp, p=0.7266`. For a repo whose headline is a null, this manufactures a null from missing data |
| D3 | Mode-subset Stage-1 selects on a sign pattern across six arms sharing one baseline, uncorrected | `analysis/mode_subsets.py:284-287` promotes on `all_positive`; `holm_adjust` is never called in that module, and the computed `permutation_p_two_sided` is not used in the selection. Family-wise probability of at least one spurious promotion is about 0.55 |
| D4 | What the code calls recurrence is per-episode mode incidence | `analysis/recurrence.py:322-339` computes mode-labelled episodes over episodes, and each episode carries at most one label. `evaluation/metrics.py:65-66` already distinguishes `episode_incidence_by_mode` from `repeated_events_after_recovery`; the summary layer computes the former and names it the latter |
| D5 | The isotonic fit emits a point threshold labelled `observed_crossing` from data with no monotone signal | `frontier.py:136-188`. Raw probabilities 0.9, 0.1, 0.8 collapse to a single pooled block at 0.6, which bypasses the `unidentified_plateau` guard and reports R50 = 1.0 as an observed crossing. `monotonic_assumption` at `frontier.py:79` is a hardcoded default never assigned anywhere |
| D6 | A plateau exactly at the quantile is an identified set reported as a point | Same function; the left edge is returned with no interval and no plateau label |
| D7 | The cluster bootstrap returns a zero-width 95 percent interval when all clusters agree | `evaluation/paired.py:74-80`. That is the default state of a nominal suite where both policies succeed everywhere, which is exactly when `require_interval_within_budget` is supposed to bite. `regression.py:77-86` then gates on that interval, so the interval gate silently becomes a point-estimate gate |
| D8 | The published p-floor is understated by a factor of two on the slippery arm | `analysis/multiseed.py:508-514` computes `2/2**n`. With one exactly-zero delta present (seed 4096) the achievable minimum is `4/2048`, not `2/2048`. The stated conclusion survives, since 0.7266 clears either floor by three orders of magnitude |
| D9 | Held-out checkpoint selection leaves no trace | `evidence_verdict` is a pure re-runnable function evaluated against the held-out split, the candidate `policy_id` is a free-form string checked against nothing in the protocol, and the function writes no artifact. The `validation` split exists and no code path uses it for candidate selection |
| D10 | The five-arm design is declarative | `protocol.py:22-28` names arms A to E and `training_steps_per_arm` is a single shared integer, but neither is read anywhere outside `__post_init__`, and `evidence_verdict` never references `protocol.arms`. Equal budget is declared, never verified |
| D11 | Multiple-comparison family is split per terrain on a non-statistical rationale | `analysis/significance.py:147-154` with the argument at `docs/methodology/ff_sweep_rigor.md:99-105`. Holm is correctly implemented and valid under arbitrary dependence, so the shared control is not a problem; the family definition is. Ten reported tests read as one result set give a family-wise rate near 0.0975. The multiseed family, four tests produced by adding seeds after observing a null, receives no correction at all |
| D12 | `_norm_ppf`'s lower tail is misparenthesised | `evaluation/significance.py:195-199` divides only the final coefficient. Measured with scipy forced off: q=0.005 returns -134.58 against a true -2.576. Dead today because scipy is a hard dependency |

## E. Fixtures, detector and test suite

| # | Finding | Evidence |
|---|---|---|
| E1 | The detector's validation evidence is not in the repo | `git ls-files data/failures` and its history are both empty; `.gitignore:23` excludes the parquets. `generator.py:277` records that the on-disk files predate a seeding fix and cannot be bit-reproduced, and regeneration digests do differ |
| E2 | Every detector test is the threshold plus an epsilon | `tests/test_taxonomy.py` sets each input one step past the constant it tests, and `generator.py:119,146` duplicate the detector's literals with no sync test. There is no independently labelled evaluation and no precision, recall or false-positive rate anywhere |
| E3 | The adversarial-negatives test is mutation-blind | `test_detector_labeling.py:65` intersects predictions with only the modes the fixture author anticipated. Always-collapse, always-stumble, always-command-mismatch and deletion of the slip-priority exclusion at `detector.py:194` all pass it. `negatives.py:47` also gives every fixture zero joint velocity, impossible for a walking robot |
| E4 | The slip fixture is a coin flip | Detected in 498 of 1000 seeds. The three on-disk variants fire by luck. At the parameters `test_synth.py` uses the mode is mathematically undetectable, and the three sustained-duration modes have no detector test there |
| E5 | The slip and command_mismatch fixtures are kinematically impossible | `generator.py:82` derives position open-loop from the command and the velocity-perturbing modes never override it. Measured on the last slip row: `dx/dt = +0.600 m/s` with logged body velocity `-0.000` |
| E6 | `failure_flag` is not the detector's decision | `generator.py:177,256` use `n_failure//2`. Divergence from the detector's first event: slip +5 rows, command_mismatch +20 rows. `capsule.py:292-295` derives the capsule onset from that column, so the one mode whose seed row is genuinely perturbed has its window mislocated by 0.4 s |
| E7 | CI has no lint gate and skips the simulator boundary | `.github/workflows/ci.yml` runs only `pytest -q -m "not sim"` on Python 3.10. `ruff`, `black` and `mypy` are declared in the dev extra and configured in `pyproject.toml`, and none is invoked; `black --check` reports 28 files. `test_phoenix_backend.py:14-15` skips without torch and phoenix, so CI runs 243 of 252 tests |
| E8 | 19 percent of statements have zero coverage, concentrated in the v2 path | `training.py`, `cli.py`, `workflow.py`, `sweep_report.py`, `failure_modes.py`, `plots.py`, `report.py`, plus `provenance.git_identity` and `build_manifest`. `validate_repair_spec` is pure CPU logic holding the only reset-state gate that reaches training, and is untested |
| E9 | `test_scientific_invariants.py` enforces provenance, not physics | Seven real guards, no tautologies, and every one is about identity, hash agreement, count consistency or retargeting. `seed_state_id` is a hash of which row, never of what is in the row, which is why section B went unnoticed |
| E10 | The no-torch guard is inert | `test_significance.py:220-239` compares `torch in sys.modules` before and after, but `test_phoenix_backend.py:14` imports torch at collection time, so the assertion is trivially true locally and trivially true in CI |
| E11 | The headline regression guard can degrade to a silent skip | `test_multiseed_combined.py:318-325` calls `pytest.skip` if the results path is absent, and there is no n=11 guard at all |
| E12 | Docstrings and filenames that contradict their contents | `multiseed.py:13-15` says one-sided where the code is two-sided; `:231` claims add-one smoothing that is absent; `paired.py:3-4` says replicates are retained where they are collapsed to means first; `results/legacy_row0_curriculum/multiseed_n11_scale_verdict.md` contains the n=7 analysis; `test_multiseed_combined.py:358` attributes the worst slippery delta to seed 1729 where the reproduction shows seed 2718 |

## What is sound and should be preserved

Verified, with the same evidence discipline as the findings above.

- The Phoenix interface is clean. All sixteen Ashfall calls into the simulator bind correctly against the
  live sibling source, `EpisodeOutcome` keywords match exactly, and no required field is missing. Every
  breakage found by this audit is inside Ashfall, which is the cheaper place for them to be.
- Frontier-probe isolation holds by construction. Probes run out of process and only per-scenario
  booleans cross back into reset sampling. `install_scenario_reset` hard-fails on split leakage.
- Friction and command writes verify by readback and raise on mismatch. Root pose, root velocity and
  joint writes do not, and the emitted telemetry label says exactly that rather than overclaiming.
- `assert_eligible` is default-deny and fires: unreproduced capsules are blocked, mock evidence is
  fenced behind an opt-in the production path never passes, and `load_reproduction` recomputes scores
  and rejects a changed verdict.
- Split discipline holds under attack. A scenario identity excludes its split label, so relabelling a
  held-out case cannot evade checks, and identical parameter and state cases are rejected across splits
  even when the seed, reproduction or family differs. One residual hole: `initial_state_id` is a
  free-form string that the cross-split key trusts.
- The repair curriculum refuses held-out and validation input on every refresh, by strict set equality
  against the training scenario ids, and refuses non-boolean or empty replicate outcomes.
- Holm correction is correctly implemented and valid under the dependence the shared control induces.
- The analysis layer is genuinely strong and regression-guarded, including a test that pins the worst
  slippery seed so it cannot be quietly dropped.
- Import discipline is clean. No module-level torch or Isaac import anywhere in `src/`, and all
  forty-three modules import with the simulator stack blocked.
- Synthetic seeding is correct and byte-stable across interpreter hash seeds.
- Hardware zero-fill is correctly refused in the recurrence path, with tests, and the detector's false
  positives are recorded in tests rather than hidden.
- The published n=11 null reproduces exactly from the committed archive: slippery -0.4155 pp with
  p=0.726562, rough +0.5485 pp with p=0.767578, over seeds 7, 42, 99, 123, 314, 1618, 1729, 2024, 2718,
  4096 and 6022.

## What this means for the claims

Phase I remains a correctly executed and correctly analysed measurement of a treatment that was not
applied, and section B says the same is currently true of the v2 path. No Phase-II claim can rest on
this code until B1, B2, B6, C1 and C3 are closed and a delivery gate measures distinguishability rather
than index arithmetic. A1 to A4 must be closed before any simulator session, because a session run
against A4 would return a rejection that reads like a fidelity problem.
