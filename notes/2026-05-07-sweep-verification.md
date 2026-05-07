# Sweep verification, 2026-05-07

Ground-truth pass over the v0.3.0 failure_fraction sweep.
Goal: confirm `results/REPORT.md` matches the underlying `metrics_*.json`
data, and inventory what raw signal exists for further analysis.

## Data sources

Per cell, the picked-up artifacts are at:
`results/ablation_failure_fraction_failure_fraction=<ff>/2026-04-28_00-17-07/metrics/metrics_{slippery,rough}.json`.

Each cell's earlier 2026-04-19 stamp directory is empty (no metrics);
`sweep_report._load_cells` correctly prefers the most-recent stamp dir
that has metrics.

## Per-cell metrics, read directly from JSON

| ff   | env       | n_eps | success_rate | failure_rate | mean_return | slew_sat |
|-----:|:----------|------:|-------------:|-------------:|------------:|---------:|
| 0.00 | slippery  | 134   | 0.8881       | 0.1119       | 7.473       | 0.3269   |
| 0.00 | rough     | 130   | 0.9077       | 0.0923       | 7.834       | 0.3908   |
| 0.10 | slippery  | 132   | 0.9015       | 0.0985       | 7.382       | 0.2513   |
| 0.10 | rough     | 128   | 0.9688       | 0.0312       | 8.717       | 0.3168   |
| 0.25 | slippery  | 140   | 0.8214       | 0.1786       | 7.833       | 0.2915   |
| 0.25 | rough     | 129   | 0.8837       | 0.1163       | 8.307       | 0.3681   |
| 0.50 | slippery  | 131   | 0.9389       | 0.0611       | 8.328       | 0.2977   |
| 0.50 | rough     | 130   | 0.9231       | 0.0769       | 7.982       | 0.3771   |
| 0.75 | slippery  | 129   | 0.9225       | 0.0775       | 7.994       | 0.3810   |
| 0.75 | rough     | 129   | 0.9535       | 0.0465       | 8.652       | 0.4129   |
| 1.00 | slippery  | 130   | 0.9000       | 0.1000       | 7.343       | 0.5052   |
| 1.00 | rough     | 128   | 0.9609       | 0.0391       | 8.305       | 0.4999   |

n_eps varies (128 to 140) even though `evaluation.num_episodes=128` in
the config. Phoenix reports `num_episodes` as the count of *terminated*
episodes inside the eval window, so cells with shorter mean episode
length see more terminations within the same wall-clock budget. The
per-cell n is what we trust.

## Cross-check vs REPORT.md

REPORT.md table values match the JSON to 3 decimal places across all 12
(ff, env) cells. Per-cell n is not printed in REPORT.md but is used
correctly to compute the listed Wilson CIs (verified: e.g. ff=0.5
slippery is `wilson_ci(123, 131) = (0.884, 0.969)` which matches
REPORT.md `[0.884, 0.969]`).

## Pareto deltas vs ff=0.0 control (re-derived)

Slippery (control 0.8881):
- ff=0.10: +1.34 pp
- ff=0.25: -6.67 pp
- ff=0.50: +5.08 pp
- ff=0.75: +3.44 pp
- ff=1.00: +1.19 pp

Rough (control 0.9077):
- ff=0.10: +6.11 pp
- ff=0.25: -2.40 pp
- ff=0.50: +1.54 pp
- ff=0.75: +4.58 pp
- ff=1.00: +5.32 pp

The headline ("ff=0.5 slippery optimum, ff=0.25 regresses") matches.
Rough optimum is actually ff=0.10 (+6.11 pp), not ff=0.75 as a stale
memory line claimed. ff=0.75 is the joint Pareto pick (slippery +3.44
and rough +4.58, neither absolute best but no regression on either
axis). REPORT.md does not currently call out the joint-Pareto cell;
the rigor pass will.

## Sample-size and seed inventory

Single seed across the whole sweep: `training.seed = 42` in every
`config.yaml` and `adapt_override.yaml`. No multi-seed replicates.
This is the most important rigor caveat: every CI we compute is a
**within-run binomial CI**, not a between-seed CI. Variance estimates
shrink with bigger n inside the eval rollout but say nothing about
training-seed sensitivity.

## Per-episode raw data: NOT collected

Each `metrics_<env>.json` stores only aggregate scalars
(`num_episodes`, `success_rate`, `mean_episode_return`, etc.). There
is no per-episode parquet, no per-episode success indicator vector,
and no per-episode failure-mode label. Phoenix's `evaluate.py` collapses
the eval rollout into a single dict before writing.

Implications:
- Paired tests across ff cells (same episode index across two cells)
  are impossible. Episodes are independent rollouts and not aligned.
- A "true" per-episode bootstrap is equivalent to bootstrapping a
  Bernoulli(p_hat) vector of length n. We can do this; it gives
  near-identical CIs to Wilson for n>=120 and adds no information
  beyond Wilson, but it lets us also produce p-values via
  permutation/Fisher's exact, which the report currently lacks.
- Per-episode failure-mode counts at eval time are unrecoverable from
  this sweep. The "6-mode breakdown" requested by the rigor task
  cannot be produced from the existing artifacts. The next-best is
  the curriculum-input mode breakdown (what was sampled into the
  fine-tune buffer), which is **identical across cells** because the
  pool is identical. Per-mode breakdown is therefore deferred to the
  next ablation (Block 5).

## Synth-pool curriculum composition (input side)

18 trajectories total in `data/failures/`:

| mode             | n_traj | n_steps | active_steps |
|:-----------------|-------:|--------:|-------------:|
| attitude         | 3      | 240     | 33           |
| collapse         | 3      | 210     | 24           |
| slip             | 3      | 270     | 60           |
| stumble          | 3      | 180     | 30           |
| contact_loss     | 3      | 195     | 30           |
| command_mismatch | 3      | 330     | 90           |

Active-step count = sum of `failure_flag == True` rows. The pool is
balanced at 3 trajectories per mode but skewed in active failure
duration toward command_mismatch and slip. This composition is the
same across the whole ff sweep; only the per-minibatch sampling
fraction changed.

## What the rigor pass can and cannot do

Can:
- Bootstrap difference-in-proportions CI for each ff vs ff=0.0,
  per terrain (slippery, rough).
- Permutation / Fisher's exact p-value for each pairwise comparison.
- Multiple-comparison adjustment (Holm-Bonferroni) across the
  five ff comparisons per terrain.
- Honestly-labeled curriculum-pool composition table.

Cannot (without re-running eval):
- Per-episode failure-mode breakdown at eval time.
- Cross-seed variance.
- Real-hardware corroboration.

These limits go in the methodology doc and REPORT.md.
