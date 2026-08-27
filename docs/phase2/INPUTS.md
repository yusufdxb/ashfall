# Ashfall Phase II: the input the analysis needs

Status: the consumer is built and tested. The data does not exist yet. No
Phase-II experiment has been run.

`ashfall.evaluation.episode_records` and `ashfall.analysis.recurrence` are
plumbed and unit-tested against synthetic fixtures. What they have never seen
is a real Phoenix artifact, because none has been produced. This file states
exactly what has to land for the recurrence analysis to execute, so that step
is one command rather than a fresh design problem.

## 1. The artifact

One parquet per (arm, seed), written by Phoenix's evaluator. Ashfall never
imports Phoenix: the parquet schema is the whole contract, restated and
validated in `ashfall/evaluation/episode_records.py`.

* Schema version `1.0.0`, stamped in both the Arrow metadata key
  `phoenix_episode_records_schema_version` and a per-row `schema_version`
  column.
* A units map under the Arrow metadata key `phoenix_episode_records_units`,
  non-empty, a JSON object.
* All 21 required columns present and non-null. They are listed in
  `REQUIRED_COLUMNS`: `schema_version`, `run_id`, `episode_id`, `seed`,
  `env_index`, `terrain_id`, `challenge_id`, `success`, `termination_reason`,
  `time_to_failure_steps`, `time_to_failure_s`, `episode_return`,
  `episode_length_steps`, `episode_length_s`, `mean_lin_vel_error_mps`,
  `max_lin_vel_error_mps`, `mean_ang_vel_error_radps`, `max_ang_vel_error_radps`,
  `control_dt_s`, `policy_path`, `policy_sha256`.
* `time_to_failure_steps` and `time_to_failure_s` carry the sentinel `-1` on a
  successful episode, never a null.
* At least one row. A zero-row parquet is rejected: an empty arm is not a
  readable result.

Any deviation raises `EpisodeRecordSchemaError` at read time rather than
producing a table.

## 2. The producing command

Emitted by `phoenix.training.evaluate` on the go2-phoenix branch
`research/ashfall-curriculum-delivery` (commit `df66807` adds the emitter;
commit `cc44261` is the curriculum delivery fix that H0 gates on). The
per-episode parquet is additive: it is written next to the existing metrics
JSON and does not alter it, so the Phase-I numbers keep reproducing.

Run from the go2-phoenix repository root, with its Isaac environment
activated (`scripts/_activate.sh`), following the shape of the shipped
`scripts/eval_stand_v3.sh`:

```
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" python3 -m phoenix.training.evaluate \
  --checkpoint <policy.pt> \
  --env-config <terrain.yaml> \
  --num-envs 32 \
  --num-episodes <N, equal across arms> \
  --seed <SEED> \
  --metrics-out <run>/metrics_<arm>_seed<SEED>.json \
  --episode-records-out <run>/episodes_<arm>_seed<SEED>.parquet \
  --run-id <arm>_seed<SEED> \
  --terrain-id <terrain> \
  --challenge-id <challenge>
```

`--episode-records-out` is what switches the artifact on. Without it Phoenix
writes only the aggregate metrics JSON, which averages the per-episode
identity away and cannot answer a recurrence question at all.

## 3. Arms and seeds

The evaluator stamps `args.seed` onto every row it writes, so one invocation
produces one seed. The paired table therefore needs one invocation per
(arm, seed) cell, not one per arm.

Minimum for the H1 comparison:

| arm | meaning |
|:----|:--------|
| baseline | compute-matched uniform failure curriculum |
| treatment | adaptation conditioned on the named failure mode |

Both arms must be evaluated on the same frozen mode-specific challenge suite,
at the same `--num-episodes`, over the same seed set. Only seeds present in
both arms enter the table; an unpaired seed is dropped, and if the arms share
no seed the analysis raises rather than returning an empty table. Unequal
episode counts between arms are tolerated because cells are rates, but they
should still be matched, since an unequal count is usually a sign that
something else differed too.

Phase I paired 11 seeds. ASSUMPTION: Phase II reuses the same count and the
same seed list for continuity. The preregistration in `HYPOTHESIS.md` does not
pin a seed list, so this is an extrapolation and not a registered commitment.

The novelty-boundary baselines named in `HYPOTHESIS.md` (RMA, automatic domain
randomization, terrain-curriculum PPO, Robust PLR) each need the same
per-(arm, seed) treatment before they can be compared on recurrence.

## 4. The open blocker: per-episode telemetry

The recurrence table is only as informative as the mode labels attached to its
failed episodes, and those labels come from Ashfall's six-mode detector, which
needs per-step `pitch_rad`, `roll_rad`, `base_height_m`, `cmd_lin_vel` and
`actual_lin_vel` for each episode.

The episode-record parquet does not carry them, and Phoenix's existing
`--telemetry-out` CSV cannot substitute: it logs env index 0 only, has no
episode identifier, and carries commanded and actual base velocity without
attitude or base height. With today's artifacts, `attach_modes(records)` is
called with no telemetry and every failed episode is labelled
`MODE_UNKNOWN`, which collapses the entire table into the UNKNOWN row.

So a per-episode telemetry artifact, keyed by `(run_id, seed, episode_id)` to
match `episode_key()`, is a prerequisite for the H1 endpoint and does not
exist yet. Until it does, the recurrence machinery runs, and reports honestly
that it classified nothing. `unknown_fraction()` is what makes that visible,
and it should be reported next to every recurrence result.

Note that four of the six modes are undetectable on real GO2 telemetry
regardless, because the hardware path fills `base_pos` and `contact_forces`
with zeros. Ashfall treats an identically-zero base height as a missing
signal rather than as a collapse, so hardware episodes label as UNKNOWN with
the reason attached instead of acquiring an invented mode.
