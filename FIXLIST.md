# Ashfall Fix List

Source: skeptical code and research-rigor audit, 2026-05-21. Read-only review, nothing was changed.
Ordered by severity. None of these change the research conclusions (the n=7 null result still holds);
they are code-hygiene and reproducibility problems.

## Critical

- [x] **`intervention_count` is broken, always returns 0 or 1.** FIXED 2026-05-21.
  `src/ashfall/evaluation/metrics.py`. The analyzer now records an episode index per event
  (`_event_episodes`, aligned with `_events`) at collection time, and `compute()` counts the
  distinct episodes among intervention-mode events. The weak `>= 1` test was rewritten to an
  exact-count assertion, and a new multi-episode test
  (`test_intervention_count_reflects_episodes_with_interventions`) asserts 2 of 3 episodes.
  112 tests pass, ruff clean.

## High

- [x] **Synthetic failure data cannot be regenerated.** FIXED.
  `src/ashfall/synth/generator.py`. `variant_seed` now derives from a fixed
  `_MODE_SEED_OFFSETS` integer map (line 315: `seed + _MODE_SEED_OFFSETS[mode] + v`), not
  process-salted `hash()`. `generate_all_failures` is now bit-stable from `seed=42` forward.
  Note: the 18 parquets committed under `data/failures/` predate the fix and cannot be
  bit-reproduced; regenerating yields a stable dataset going forward.

## Medium

- [x] **README test count is wrong.** FIXED. `README.md` now says 113 unit tests
  (pytest collected/passed count today; the earlier "56"/"111" figures were both stale).

- [ ] **Contradictory Phoenix commit hash for the seed-propagation patch.**
  README:45 and `docs/methodology/2026-05-07-ff-sweep-rigor.md:276,321` say `d42ee01`;
  `notes/2026-05-09-curriculum-redesign-spike.md:62` says `129ea52`. The n=7 result depends
  on knowing the exact Phoenix commit. Reconcile to one hash across all docs.

- [ ] **Predefined `ABLATION_FAILURE_MODES` axis produces names the analysis pipeline drops.**
  `src/ashfall/experiment/sweep.py:118-129` vs `:54-61`. `_FAILURE_MODES_LABELS` only maps
  six specific tuples; the predefined axis values are not keys, so cell directory names get a
  `+` that `SUBSET_CELL_PATTERN` in `mode_subsets.py:38` (`[A-Za-z_]+`) will not match. Any
  sweep using the constant directly would silently analyze zero cells. The constant is
  effectively dead, untested code. Either wire it correctly or delete it.

- [ ] **`collect_results` writes fake `wall_time_s` and a non-existent `checkpoint_path`.**
  `src/ashfall/experiment/runner.py:328-329`. `wall_time_s` is hardcoded `0.0` and never
  filled; `checkpoint_path` points at `run_dir/checkpoint` which is never created (real
  checkpoints land in the Phoenix repo). Both are silent wrong values baked into every
  `ExperimentResult`.

- [x] **Two-sample BCa is an approximation, but labeled and advertised as BCa.** RELABELED.
  The implementation in `evaluation/significance.py` is unchanged (still the sign-corrected
  pooled-influence shortcut), but the user-facing label is now honest: the report generator
  (`analysis/significance.py`) and `results/REPORT.md` say "approximate BCa bootstrap
  (sign-corrected two-sample acceleration term)" and explicitly note the CI and p-value come
  from different procedures and can disagree at the margin. A true joint-jackknife BCa is
  still open if magnitude accuracy is ever needed.

## Low / smaller

- [ ] `Condition.CONTROL_NOREPLAY` is handled in `runner.py:260-275` but has no config under
  `configs/experiments/`. Dead, untested branch.
- [ ] `CurriculumSpec.num_variations` (`schema.py:46`) is set in every config but never read
  anywhere in `src/`. Dead config knob.
- [ ] `harness.py` has its own percentile `bootstrap_ci` (`:145-169`), separate from the BCa
  in `evaluation/significance.py`. Two bootstrap implementations, easy to call the wrong one.
- [ ] `harness.py` docstring (`:5`) claims a "paired t-test" feature; no t-test exists there.
- [ ] `MetricComparison.significant` (`harness.py:39`) is never set to `True` by
  `compare_conditions`; the significance flag in the comparison report is always False.
- [ ] `README.md:181-186` ablation-plan table still lists parameter-sweep axes the project
  has moved past (the curriculum mechanism itself is now suspect). Stale.
- [ ] `_T_975_BY_DF` (`multiseed.py:48-66`) silently falls back to 1.96 for any df not in the
  hardcoded table. Correct for n=7, but a future n=18 run would get overconfident CIs with no
  warning.
- [ ] LICENSE file is missing from the working tree (shows as deleted vs main) while the
  README still says "License: MIT". Restore it or update the README.
- [ ] Branch `curriculum-redesign-2026-05-09` has 6 uncommitted modified files (IsaacLab path
  fixes that look complete). Commit them so repo state is clean.

## Not verified (out of scope)

- Whether Phoenix tolerates `--trajectory-dir /dev/null` for the `control_random` condition
  (`runner.py:251`). If Phoenix globs that dir for parquets, the control condition crashes.
- Whether the `reset_bridge.py` defects cited in the curriculum-redesign spike doc are
  accurately quoted (separate repo, not reviewed in that pass; see go2-phoenix FIXLIST).
