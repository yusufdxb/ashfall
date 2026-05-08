# 2026-05-08 — Phoenix failure_modes propagation audit

## TL;DR

The mode-subset ablation cannot run as-designed: ashfall writes
`curriculum.failure_modes` into the per-cell adapt YAML, but Phoenix's
`fine_tune` / `FailureCurriculum` / `TrajectoryPool` ignores the field.
Without a patch, every cell in the planned 18-cell sweep would silently
sample from the full unfiltered pool, and 6 hours of GPU time would
produce 18 noisy reproductions of the v0.3.0 ff=0.5 cell.

## Pipeline trace

1. `~/Projects/ashfall/src/ashfall/experiment/runner.py:173-176` —
   ashfall's `_write_adapt_override` injects `failure_modes` into the
   per-cell YAML at `curriculum.failure_modes`. **Honored.**
2. `~/workspace/go2-phoenix/src/phoenix/adaptation/fine_tune.py:84-95`
   — Phoenix's `_run` reads `cfg["curriculum"]["failure_sample_fraction"]`
   and `cfg["curriculum"]["trajectory_dir"]` only. **`failure_modes` is
   not read.** Pool is instantiated unfiltered.
3. `~/workspace/go2-phoenix/src/phoenix/adaptation/curriculum.py` —
   `TrajectoryPool.from_directory(dir, pattern="*.parquet")` globs all
   parquets in the dir; no mode filter. `FailureCurriculum.__init__`
   takes `pool, failure_fraction, seed` only — no mode parameter.
4. Pool composition in `~/Projects/ashfall/data/failures/`: 18
   parquets across 6 modes (3 each). Inspected via pyarrow:
   each parquet's non-null `failure_mode` column entries are uniform
   per file (matches the filename's mode token, e.g.
   `synth_slip_000.parquet` rows are either `null` or `"slip"`).

## Implication

End-to-end gap. The ablation's six subset cells (`all_modes`,
`slip_only`, `command_mismatch_only`, `slip_plus_cm`, `severe_only`,
`severe_plus_slip`) would all run on the same 18-parquet pool and only
differ by stochastic seed.

## Required patch (Block 2)

Phoenix-side surgical changes:

1. `TrajectoryPool.from_directory(directory, *, pattern, failure_modes)`
   accepts an optional `failure_modes: list[str] | None`. None / empty
   list = unchanged behavior (load all parquets).
2. When `failure_modes` is set, peek at each candidate parquet's
   `failure_mode` column with pyarrow and keep it only if any non-null
   value matches a member of the filter list.
3. `phoenix.adaptation.fine_tune._run` reads
   `cfg["curriculum"].get("failure_modes", None)` and passes it to
   `TrajectoryPool.from_directory`. Log the filtered pool size + which
   modes survived for provenance.
4. Phoenix-side unit test in `tests/test_curriculum.py` covering the
   filter (synthetic parquets with three modes, filter to one,
   filter to two, no filter).
5. Commit on `audit-fixes-2026-04-16` (Phoenix's working branch). No push.

The patch is bounded: ~30 LOC in curriculum.py, ~5 LOC in fine_tune.py,
one new test. Well under the 50-LOC stop threshold.
