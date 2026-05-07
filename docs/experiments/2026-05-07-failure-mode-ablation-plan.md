# Failure-Mode Subset Ablation: Experiment Plan (drafted 2026-05-07)

Status: scaffolded, not yet run. GPU was reserved for Zeus when this
plan was written.

## Hypothesis

The v0.3.0 sweep found a point-estimate slippery success-rate optimum
at `failure_fraction = 0.5` (+5.1 pp over the ff=0.0 control), but
the BCa 95% CI on that difference straddles zero
(`[-0.017, +0.118]`, Holm-adjusted Fisher p=0.76). With one seed
and n approximately 130 episodes per cell, we cannot say that the
all-modes-at-ff=0.5 cell is meaningfully better than the no-replay
control.

The next question is therefore not "is the lift real?" (we need
multi-seed for that), but: **assuming the point estimate is real,
which failure modes carry it?** The pool composition (see
`docs/methodology/2026-05-07-ff-sweep-rigor.md`, section 4) is
skewed toward `command_mismatch` (0.34 of active steps) and `slip`
(0.22), so a step-count-driven explanation predicts those two modes
account for most of the lift; a severity-driven explanation
predicts `attitude + collapse` carry it; a synergy explanation
predicts no individual subset reproduces the all-modes result.

## Conditions

Six cells, all at `failure_fraction = 0.5` and `seed = 42`:

| cell                   | failure_modes                          | hypothesis tested                          |
|:-----------------------|:---------------------------------------|:-------------------------------------------|
| all_modes              | `[]` (all 6)                           | reproducer; matches v0.3.0 ff=0.5 within seed noise |
| slip_only              | `[slip]`                               | friction-axis alone explains slippery lift |
| command_mismatch_only  | `[command_mismatch]`                   | step-count majority alone is sufficient    |
| slip_plus_cm           | `[slip, command_mismatch]`             | step-count majority pair is sufficient     |
| severe_only            | `[attitude, collapse]`                 | high-severity short-event modes are sufficient |
| severe_plus_slip       | `[attitude, collapse, slip]`           | minimal severity + friction set is sufficient |

This is six runs. With v0.3.0's per-cell wall-clock of approximately
15 to 25 minutes (200 fine-tune iters at 4096 envs on the RTX 5070,
plus eval) the budget is 90 to 150 minutes total.

## Expected runtime

Reference from `results/_logs/sweep_master.log`:

| ff   | duration |
|-----:|---------:|
| 0.00 | 911 s    |
| 0.10 | 908 s    |
| 0.25 | unknown* |
| 0.50 | 1298 s   |
| 0.75 | 1212 s   |
| 1.00 | 2398 s   |

(*ff=0.25 logged into a different format; manifest entry incomplete.)

Wall-clock scales roughly linearly with `failure_fraction` because
the curriculum-replay step is the dominant per-iter cost. At ff=0.5
fixed across all six cells we should see approximately 15 to 22
minutes per cell, total approximately 100 to 130 minutes. Plan for
2 hours of GPU time, ideally on a quiet workstation (no Zeus, no
Phoenix, no Isaac Sim viewer).

## Required pre-flight

1. Verify `~/IsaacLab/isaaclab.sh` resolves and Phoenix repo is at
   `~/workspace/go2-phoenix`.
2. Verify `data/failures/` contains the 18 synth parquets used in
   v0.3.0 (the runner does not regenerate them).
3. Verify the ashfall-baseline checkpoint at
   `~/workspace/go2-phoenix/checkpoints/ashfall-baseline/latest.pt`
   exists. This is the same warm-start used by v0.3.0.
4. Verify that Phoenix's curriculum loader honours
   `curriculum.failure_modes` as a mode-name filter list. The
   runner change committed on 2026-05-07 wires the field through;
   Phoenix-side support is REQUIRED for the cells to mean anything.
   If absent, every cell will silently use all modes and the
   ablation degenerates to a 6-way replication of the all-modes
   cell.

   **Verification command (read-only)** before running:

   ```bash
   grep -n "failure_modes" ~/workspace/go2-phoenix/src/phoenix/curriculum/*.py
   ```

   If no match, the curriculum loader needs a small extension. The
   runner-side wiring is at
   `src/ashfall/experiment/runner.py` `_write_adapt_override`.

5. Confirm `nvidia-smi` shows the GPU idle (no Zeus, no other
   training).

## Eval-time logger patch (recommended)

Phoenix's `evaluate.py` currently writes only aggregate scalars to
`metrics_*.json`. To enable per-mode breakdowns at eval time
(which the v0.4.0 report should have), patch the eval rollout to:

- Run the existing `FailureDetector` against each step's telemetry.
- Aggregate per-episode failure-mode counts.
- Append a `failures_by_mode: {mode: int}` field to the per-env
  metrics dict before serialising.

This is a less-than-100-line change confined to Phoenix's
`evaluate.py`. The Ashfall side already has all the infrastructure
to consume the field (`FailureMetrics.failures_by_mode` in
`src/ashfall/evaluation/metrics.py`).

## Success criteria

The ablation is informative regardless of outcome. We log:

- Each cell's slippery and rough success rate, with Wilson 95% CIs.
- Each cell's BCa 95% CI on the difference vs the all-modes cell
  (within-ablation control) and against the v0.3.0 ff=0.0 control
  (cross-ablation control).
- Holm-Bonferroni adjusted Fisher's exact p-values across the 5
  non-control cells per terrain.

Three readable outcomes:

1. **Step-count win.** `slip_plus_cm` reproduces all-modes within
   the CI on slippery, severe-only does not. Recommendation:
   v0.4.0 pool can drop `attitude + collapse` and run cheaper.
2. **Severity win.** `severe_plus_slip` reproduces all-modes,
   `slip_plus_cm` does not. Recommendation: v0.4.0 pool focuses on
   high-severity short-event recovery.
3. **No subset wins.** None of the subset cells reproduce all-modes
   on slippery. Recommendation: v0.4.0 keeps all 6 modes; lift is a
   synergy effect.

A null result (no subset and no all-modes-reproducer significantly
beats the cross-ablation ff=0.0 control) is also informative: it
strengthens the call for multi-seed before any further ablation
investment.

## Out of scope for this run

- Multi-seed. We will repeat each cell at 3 to 5 seeds in v0.4.1
  once the mode-axis result is known. Doing seeds and modes
  together would be a 30-cell sweep and not justified yet.
- Real-hardware. CaresLab GO2 hardware validation is queued for the
  next lab visit and lives in a separate playbook.
- Eval-time real failure logging. Requires the Phoenix patch
  described above; if the patch lands before this ablation runs,
  the v0.4.0 report gains the eval-time per-mode breakdown
  automatically.
