# Seed-propagation audit (2026-05-07)

Goal: verify whether `training.seed` actually reaches the PPO trainer
and the env RNG, so the multi-seed pilot we are about to run will
produce genuinely different cells.

## Trace

1. **Ashfall sweep YAML**
   `configs/experiments/ablation_sweep.yaml` exposes `training.seed: 42`.
   The new pilot YAML adds a `seed` ablation axis.

2. **Ashfall sweep expansion**
   `src/ashfall/experiment/sweep.py::generate_sweep` walks the
   axes via `_set_nested(cfg, axis.param_path, value)`. With
   `param_path: training.seed` the per-cell `ExperimentConfig.training.seed`
   is overwritten correctly.

3. **Ashfall runner -> per-cell adapt YAML**
   `src/ashfall/experiment/runner.py::_write_adapt_override` (line 147):
   ```python
   adapt_cfg["run"]["seed"] = t.seed
   ```
   Confirmed that the generated
   `configs/_generated/adapt_<cell>.yaml` carries the correct seed.
   Spot-checked existing v0.3.0 cells: `seed: 42` is present.

4. **Phoenix `phoenix.adaptation.fine_tune._run`**
   Reads `cfg["run"]["seed"]` and pushes it onto `env_cfg.seed`
   (line 79 of `fine_tune.py`):
   ```python
   env_cfg.seed = int(cfg["run"].get("seed", 42))
   ```
   Then `gym.make(task_name, cfg=env_cfg, ...)` constructs the
   `ManagerBasedEnv`.

5. **IsaacLab seed plumbing**
   `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py`
   lines 96 to 98 call:
   ```python
   if self.cfg.seed is not None:
       self.cfg.seed = self.seed(self.cfg.seed)
   ```
   `seed()` -> `configure_seed(seed)` in
   `IsaacLab/source/isaaclab/isaaclab/utils/seed.py` which seeds:
   `random`, `np.random`, `torch.manual_seed`,
   `torch.cuda.manual_seed`, `torch.cuda.manual_seed_all`,
   `wp.rand_init`, plus `PYTHONHASHSEED`. So global RNG seeding
   happens at env construction time, BEFORE the rsl_rl
   `OnPolicyRunner` is built and BEFORE `runner.learn(...)` is
   called. Policy network init in `OnPolicyRunner.__init__`
   therefore inherits the seeded torch RNG. Good.

6. **rsl_rl itself does NOT call `manual_seed` anywhere.**
   Verified by grepping
   `/home/yusuf/isaac-sim-venv/lib/python3.12/site-packages/rsl_rl/`.
   `agent_cfg.py:43` sets `cfg.seed = int(run.get("seed", 42))`
   on the runner cfg, but rsl_rl reads it only as metadata.
   Reliance on IsaacLab's pre-seed is therefore load-bearing;
   if IsaacLab ever stops seeding torch globally inside
   `ManagerBasedEnv.__init__`, every Phoenix run silently
   becomes non-reproducible.

## Identified gaps

### Gap A: FailureCurriculum is NOT seed-coupled
`phoenix.adaptation.curriculum.FailureCurriculum.__init__` accepts
`seed: int = 0` and `phoenix.adaptation.fine_tune._run` builds the
curriculum WITHOUT passing one (line 86 of `fine_tune.py`):
```python
curriculum = FailureCurriculum(
    pool, failure_fraction=float(cfg["curriculum"]["failure_sample_fraction"])
)
```
So the per-env failure-vs-clean assignment is identical across
training seeds. For ff=0 cells this is irrelevant (assignment is
all -1). For ff>0 cells it means the curriculum's reset-source
choices come from the same RNG draw across seeds. The PPO update
itself still differs because torch is seeded per-cell via
IsaacLab, but one stochastic input is held constant. We should
fix this so the multiseed pilot actually varies the curriculum
sampling too.

### Gap B: evaluate.py uses a hardcoded `--seed 1234` default
`phoenix.training.evaluate.parse_args` defaults `--seed 1234`. The
ashfall runner's `eval_<env>` command does not pass `--seed`, so
all cells (and all training-seed variants) are evaluated with the
same env seed (1234) and therefore the same eval-time env RNG. For
the multiseed pilot this is acceptable, even desirable: it gives an
apples-to-apples eval across training seeds, so any across-seed
spread reflects the trained policy itself and not the eval-env
randomization. We will leave this as-is and note the choice.

### Gap C: Variation sampler RNG is config-bound, not training-bound
`phoenix.replay.variations.VariationSampler.__init__` defaults
`seed: int = 17` and reads `seed` from the variation YAML. The
variation YAML is checked into the repo and not overridden by the
sweep harness, so variation samples are identical across cells.
This is by design (we want the same DR distribution across
ff cells). For the multiseed pilot this is also acceptable: keeping
DR fixed isolates the training-seed effect. No patch needed.

## Patches needed

- **Phoenix**: pass `seed` from `cfg["run"]["seed"]` into
  `FailureCurriculum(...)` so per-seed curriculum draws actually
  differ. One-line edit in `phoenix/adaptation/fine_tune.py`.
  Optional unit test that constructs two curricula with different
  seeds and asserts assignment arrays differ.
- **Ashfall**: no patch needed. The seed already lands in the
  per-cell YAML.

## What is already correct

- Sweep expansion correctly overwrites `training.seed` per cell
  via `_set_nested`.
- Runner correctly writes `run.seed` into the per-cell adapt YAML.
- fine_tune.py correctly pushes `run.seed` into `env_cfg.seed`.
- IsaacLab `ManagerBasedEnv.__init__` correctly calls
  `configure_seed` before any policy or env RNG draws.
- agent_cfg.py copies `run.seed` to the rsl_rl runner cfg
  (cosmetic, but matches the convention).
- Variation sampler is intentionally fixed-seed across cells
  (DR distribution held constant).

## Verdict

Seed propagation is mostly correct. One genuine gap (curriculum RNG
hardcoded to seed=0) and one cosmetic-but-structural risk (rsl_rl
relies on IsaacLab seeding implicitly). I will patch the curriculum
gap; the rsl_rl observation goes into the audit note as a known
limitation, not a blocker.
