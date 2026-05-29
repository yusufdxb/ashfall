# Curriculum Redesign — Spike (2026-05-09, populated 2026-05-17)

## Why

- v0.3.0 single-seed +5.1pp slippery lift at ff=0.25 did NOT replicate against a proper no-curriculum control (ff=0.0): the original "lift" was confounded with "fine-tune on slippery", not the curriculum (`SWEEP_2026-04-28.md`).
- n=7 paired-seed verdict on ff in {0.0, 0.5}: slippery mean delta -1.10 pp, 95% CI [-5.20, +3.01], sign-flip p=0.5625. Rough -1.58 pp, CI [-6.31, +3.15] (`MULTISEED_SCALE_2026-05-08.md`). Neither comes close to alpha=0.05; n=7 floor is p=0.0156.
- 18-cell Stage-1 mode-subset ablation at ff=0.5 (`MODE_SUBSETS_STAGE1_2026-05-08.md`): no subset rescues the curriculum. `slip_only` is actively damaging (mean -9.80 pp, 3/3 negative). `all_modes` is the only directionally positive subset (+2.14 pp, p=0.25 = n=3 floor — the same n=7-killed configuration).
- Both levers (failure_fraction, failure_mode subset) are saturated within the current synth pool + current mechanism. Three viable next directions were identified: (1) re-design curriculum, (2) mode-subset with exploratory framing only, (3) hardware data.
- This spike supersedes that menu after a 4-agent investigation (2026-05-17). The headline finding inverts the framing: the curriculum mechanism has two specific code-level defects that plausibly explain the null verdict without any redesign. Fix those first; the redesign decision is gated on whether the bug-fix re-pilot recovers signal.

## Headline finding — the curriculum may be running on a null seed

Two mechanism defects identified in `~/workspace/go2-phoenix` `audit-fixes-2026-04-16`:

1. **Wrong row.** `_InitialStateCache.get()` always reads `row=0` from each failure parquet (`reset_bridge.py:38`). For `synth_slip_000.parquet`, `failure_flag=True` first appears at row 70; row 0 is upright at 0.299 m base height. The curriculum is seeding **pre-failure normal stance**, not failure-imminent kinematics. Across all 6 modes the seed row is the trajectory's first frame of stable-phase noise, not the moment the failure becomes inevitable.

2. **Velocity is silently dropped.** `load_initial_state` parses `base_lin_vel_body` and `base_ang_vel_body` (`trajectory_reader.py:63-71`), but the bridge writes only pose + joints (`reset_bridge.py:88-96`). There is no `write_root_velocity_to_sim` call. Whatever pre-failure velocity made the synth state physically meaningful is discarded; the env starts from rest.

Net effect: a "failure-seeded" env is initialized at upright stance with zero velocity. That is approximately the default Isaac Lab reset distribution. Toggling ff from 0.0 → 0.5 changes almost nothing the policy can see. **The n=7 null result is consistent with the curriculum mechanism being a near no-op, not with the curriculum being saturated.**

Convergent evidence from Agent 2 (literature scan) and Agent 3 (pool diagnostic):
- Agent 2's rank-2 candidate (RFCL — reset from failure-onset frames, ICLR 2024) IS exactly the row=N fix above, dressed up with a forward-time curriculum.
- Agent 3 independently verified `failure_flag=True` first appears at row 70 for slip; observed that synth `slip_only` being the worst single-mode subset is consistent with the policy training on a wrong stimulus.

## Hypotheses to test (ordered, gated)

- [ ] **H0 — bug-fix recovers signal.** Patch `reset_bridge.py` to (a) seed from a configurable row offset, e.g. `failure_onset_row - K` for small K, and (b) write `base_lin_vel_body` + `base_ang_vel_body` to sim. Re-run n=3 pilot at ff in {0.0, 0.5} on slippery. If mean paired delta at ff=0.5 ≥ +3 pp and ≥2/3 seeds positive, the curriculum mechanism is fine and the v0.2.0/v0.3.0 lifts were on the right side of a buggy distribution. Cost: ~50 LOC + 90 min on RTX 5070.

- [ ] **H1 — pool is the bottleneck.** If H0 null, command-diversify the synth pool (Agent 3 recommendation): sweep `(vx, vy, wz)` across 5 settings per mode, regenerate 90 parquets, re-run n=3 pilot. Eval-logger patch (Agent 3 §3, ~60 LOC in `evaluate.py`) lands in the same change so per-mode eval failure counts become observable. Cost: pool regen 20 min + 90 min training + logger patch.

- [ ] **H2 — mechanism redesign needed.** If H1 null, formalize RFCL (Agent 2 rank 2): reverse-forward curriculum with seed row walked from `failure_onset_row` toward earlier rows as the policy improves. Phoenix-side change is the same `reset_bridge.py` plus a curriculum scheduler keyed to slippery success rate. Cost: ~200 LOC + 1 day.

- [ ] **H3 — environment-side curriculum.** Independent of H0-H2: PLR over friction/terrain bins (Agent 2 rank 1). Curriculum-over-environment has stronger sim-to-real evidence than curriculum-over-data. ~300 LOC, 2-3 days. Run only after H0-H2 are resolved so the two changes don't confound.

- [ ] **H4 — hardware ground-truth.** Parallel track, not gated on the above. Bridge patch (Agent 4 §3, ~30-50 LOC publishing `foot_force` + `base_lin_vel_body`) is a prerequisite for any real-data collection AND is required before any sim-to-real deployment of a curriculum-trained policy. Run the 30-min minimum-useful pass (slip + collapse only, ~12 induction reps) before scheduling a full session. If bug-fix succeeds (H0), hardware pass shifts from data-collection to verification.

## Design knobs in scope

- [ ] `reset_bridge.py:38` row selector: hardcoded `row=0` → `cfg.curriculum.seed_row_strategy` ∈ {first, failure_onset, failure_onset_minus_K, sample_window}
- [ ] `reset_bridge.py:88-96` state write: add `write_root_velocity_to_sim` for parsed `base_lin_vel_body`/`base_ang_vel_body`
- [ ] Pool composition: command-diversity sweep (Agent 3 §5), 5× cmd_velocity × 3 noise variants per mode = 90 parquets
- [ ] `evaluate.py` eval-time logger: instantiate `FailureDetector` per-env, accumulate `failures_by_mode` into `metrics_*.json` (Agent 3 §3)
- [ ] Optional later: trajectory-level weighting in `assign()` (currently uniform `rng.integers(0, len(pool))`), advantage bonus on failure-conditioned rollouts (no such term exists today)
- [ ] Optional later: PLR scoring layer over Isaac Lab terrain manager (Agent 2 rank 1)

## Out of scope

- Further n-scaling at the current mechanism + pool (n=7 was the agreed ceiling).
- QARL / adversarial curriculum (Agent 2 rank 3). High-variance, hyperparameter-sensitive, brought in only if H0-H3 all null. Documented for completeness, not in the active branch.
- TransCurriculum, CTS, FR-Net (Agent 2 candidates 3-4-6). Higher implementation cost than current evidence justifies.
- Literature citations from Agent 2 that **fail verification** ( `arXiv:2603.14156` TransCurriculum, `arXiv:2509.11504` FR-Net). The arXiv ID formats and dates were suspicious; treat those candidates as not-cited unless human-verified before any external use.

## Success criteria

- [ ] **For H0 (the gating experiment):** patched bridge passes all existing Phoenix tests (currently 178 non-Isaac). Re-pilot at ff in {0.0, 0.5}, n=3 seeds {42, 123, 7}, 200-iter slippery fine-tune from `rough_baseline`. Decision rule: paired delta ≥ +3 pp at ff=0.5 AND ≥ 2/3 seeds positive → H0 confirmed, recurse on n=5 then n=7. Below threshold → fall through to H1.
- [ ] Mechanism patch is observable: a unit test in `tests/test_reset_bridge.py` writes a known velocity into a fake parquet, runs the patched cache loader, and asserts the velocity reaches the sim write call (mock the Isaac Lab handle).
- [ ] Eval-logger patch surfaces per-mode failure counts in `metrics_*.json` and the analysis pipeline picks them up automatically.
- [ ] Spike outcome is one of: (a) H0 succeeds, v0.4.0 cut on the bug-fix branch + retraction note for v0.2.0/v0.3.0 framing; (b) H0 null but H1 succeeds, v0.4.0 on pool diversification + bug-fix; (c) H0+H1 null, RFCL becomes the next branch; (d) all null, hardware data is the only remaining lever.

## Risks

- Phoenix `audit-fixes-2026-04-16` may have diverged from main — verify before basing on it. (Confirmed by Agent 1: `129ea52` is pushed; the curriculum patches live on this branch.)
- Writing root velocity to Isaac Lab sim from an arbitrary parquet row may produce unstable initial states (interpenetration, joint-state-velocity mismatch). Mitigation: clamp velocity magnitude, run a 1-step zero-action settle, fall back to row=0 if the post-write contact normals exceed a threshold.
- Per-row selection for `failure_onset` requires the parquet to expose where `failure_flag` first becomes True. Either compute this on parquet load and cache, or pre-bake into a sidecar JSON during synth generation.
- Reproducibility risk: changing the seed-row strategy changes the initial-state distribution, which changes the PPO trajectory distribution, which means the bit-for-bit reproducibility of v0.3.0 ff=0.0/seed=42 (currently confirmed) will be lost. Document this in the v0.4.0 release notes.
- Lit-scan risk: 2 of 6 candidates from Agent 2 had unverifiable citations. Do not cite TransCurriculum or FR-Net externally without independent verification.
- Schema risk for any future real-data parquets (Agent 4 §7): real-data writer must produce all 13 fields; `base_pos` and `action` zero-fill is acceptable, but `foot_force` and `base_lin_vel_body` require the bridge patch as a prerequisite.

## Next concrete action

1. **Today**: branch `H0-bridge-fix-2026-05-17` off `audit-fixes-2026-04-16` on Phoenix. Patch `reset_bridge.py` for (a) configurable seed-row strategy and (b) velocity write. Add unit test. Run existing 178-test suite to confirm no regressions.
2. **Today/tomorrow**: re-run the multiseed pilot (n=3 at ff in {0.0, 0.5}, seeds 42/123/7) on the patched bridge. ~90 min wall on RTX 5070. Use the existing `multiseed_pilot_2026-05-07.yaml` config, only the Phoenix branch changes.
3. **After verdict**: write `notes/2026-05-17-h0-bridge-fix-verdict.md`. If H0 succeeds, scope v0.4.0; if null, scope H1 (pool diversification) as the next 1-day task.
4. **In parallel, independent of H0-H3**: scope the `lowstate_bridge_node.py` patch (Agent 4 §3) to publish `foot_force` and `base_lin_vel_body`. Required for any hardware pass, useful for sim-to-real verification regardless of curriculum direction.

## Appendix — Agent outputs

Full 4-agent reports (Phoenix mechanism audit, lit scan, pool diagnostic, hardware plan) live in this branch's session log. Key file:line citations:
- `reset_bridge.py:38` — hardcoded row=0
- `reset_bridge.py:55-101` — `install()` monkey-patch on `_reset_idx`
- `reset_bridge.py:88-96` — pose + joints write; no velocity write
- `curriculum.py:116-119` — `ff` semantics: `round(K × ff)` envs of the terminating subset
- `trajectory_reader.py:63-71` — `load_initial_state` reads 6 columns
- `synth_slip_000.parquet` row 70 — first `failure_flag=True`
- `evaluate.py` `RolloutMetrics` — no per-mode failure counts
- `lowstate_bridge_node.py` — does not publish foot_force or base_lin_vel
- `trajectory_logger.py:222-230` — fills foot_force/base_lin_vel with zeros
