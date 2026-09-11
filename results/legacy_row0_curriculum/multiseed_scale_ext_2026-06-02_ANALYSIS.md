# Ashfall multi-seed extension: n=7 -> n=11 paired analysis (2026-06-02)

Status: COMPLETE. Null verdict from the published n=7 analysis HOLDS at n=11
on both terrains.

## What this is

The published Ashfall headline rests on n=7 seeds
`{42, 123, 7, 99, 314, 1729, 2718}` (pilot + scale, 2026-05-07). Tonight 4 new
seeds `{1618, 2024, 4096, 6022}` were trained at `failure_fraction in {0.0, 0.5}`
(8 cells, all rc=0), taking the paired-by-seed sample to n=11. This document
re-runs the repo's OWN paired sign-flip test on the pooled 11 seeds and compares
the verdict to n=7.

## Methodology (repo tooling, not reinvented)

All statistics come from the exact functions the published n=7 analysis uses:

- `src/ashfall/analysis/multiseed.py:453` `combine_pilot_runs` -- pools cells
  from multiple sweep prefixes, dedup on `(ff, seed, terrain)`.
- `src/ashfall/analysis/multiseed.py:475` `run_combined_analysis` -- per-ff
  summary + paired-delta per terrain.
- `src/ashfall/analysis/multiseed.py:250` `paired_delta` -- pairs cells by seed,
  delta = `success_rate(ff=0.5) - success_rate(ff=0.0)`, t-CI at df=n-1.
- `src/ashfall/analysis/multiseed.py:224` `_exact_sign_flip_p` -- exact
  two-sided sign-flip permutation p over `2**n` sign assignments, |mean| stat,
  no add-one (raw hits/total).

The published n=7 analysis combines exactly these two specs:
`(results, "multiseed_pilot_2026-05-07")` + `(results, "multiseed_scale_2026-05-07")`.
The n=11 run adds a third spec `(results, "multiseed_scale_ext_2026-06-02")` and
changes nothing else. The metric consumed is `success_rate` from each cell's
`metrics/metrics_{slippery,rough}.json`; the new cells have an identical schema
(verified: same keys, same `success_rate`/`num_episodes` fields).

## Exact commands run

```bash
cd /home/yusuf/Projects/ashfall && source .venv/bin/activate
PYTHONPATH=src python3 - <<'EOF'
from pathlib import Path
from ashfall.analysis.multiseed import combine_pilot_runs, run_combined_analysis
R = Path("/home/yusuf/Projects/ashfall/results")
combined = combine_pilot_runs(
    (R, "multiseed_pilot_2026-05-07"),
    (R, "multiseed_scale_2026-05-07"),
    (R, "multiseed_scale_ext_2026-06-02"),   # the 4 new seeds
)
rep = run_combined_analysis(combined)   # ff_a=0.0, ff_b=0.5 (defaults)
for d in rep.deltas:
    print(d.terrain, d.n_seeds, d.mean_delta, d.sem_delta,
          d.ci95_lo, d.ci95_hi, d.permutation_p_two_sided)
EOF
```

## Per-seed deltas at n=11 (delta = ff0.5 - ff0.0)

### Slippery (prior single-seed optimum)

| seed | source | ff=0.0 | ff=0.5 | delta |
|-----:|:-------|-------:|-------:|------:|
| 7    | pilot  | 0.8797 | 0.9077 | +0.0280 |
| 42   | pilot  | 0.8881 | 0.9237 | +0.0356 |
| 99   | scale  | 0.8478 | 0.8561 | +0.0083 |
| 123  | pilot  | 0.9167 | 0.9173 | +0.0006 |
| 314  | scale  | 0.9030 | 0.8931 | -0.0099 |
| 1618 | ext    | 0.9922 | 1.0000 | +0.0078 |
| 1729 | scale  | 0.9308 | 0.8797 | -0.0511 |
| 2024 | ext    | 0.9844 | 1.0000 | +0.0156 |
| 2718 | scale  | 0.9231 | 0.8345 | -0.0885 |
| 4096 | ext    | 1.0000 | 1.0000 | +0.0000 |
| 6022 | ext    | 0.9922 | 1.0000 | +0.0078 |

### Rough (retention check)

| seed | source | ff=0.0 | ff=0.5 | delta |
|-----:|:-------|-------:|-------:|------:|
| 7    | pilot  | 0.9766 | 0.8846 | -0.0919 |
| 42   | pilot  | 0.9077 | 0.9219 | +0.0142 |
| 99   | scale  | 0.9225 | 0.8626 | -0.0599 |
| 123  | pilot  | 0.9531 | 0.9302 | -0.0229 |
| 314  | scale  | 0.8702 | 0.9375 | +0.0673 |
| 1618 | ext    | 0.8722 | 0.9323 | +0.0602 |
| 1729 | scale  | 0.8657 | 0.8571 | -0.0085 |
| 2024 | ext    | 0.9766 | 0.9538 | -0.0227 |
| 2718 | scale  | 0.9462 | 0.9375 | -0.0087 |
| 4096 | ext    | 0.8417 | 0.9609 | +0.1192 |
| 6022 | ext    | 0.8806 | 0.8947 | +0.0141 |

## Test results: n=7 vs n=11

| terrain  | n  | mean delta | 95% CI (t, df=n-1)   | signs (+) | exact two-sided sign-flip p | p-floor | clears a=0.05 | CI excludes 0 |
|:---------|---:|-----------:|:---------------------|:----------|:----------------------------|--------:|:--------------|:--------------|
| slippery |  7 | -1.099 pp  | [-5.204, +3.006] pp  | 4/7       | 0.5625                      | 0.0156  | no            | no            |
| slippery | 11 | -0.416 pp  | [-2.823, +1.992] pp  | 7/11      | 0.7266                      | 0.0010  | no            | no            |
| rough    |  7 | -1.578 pp  | [-6.308, +3.153] pp  | 2/7       | 0.3906                      | 0.0156  | no            | no            |
| rough    | 11 | +0.548 pp  | [-3.463, +4.560] pp  | 5/11      | 0.7676                      | 0.0010  | no            | no            |

(p-floor = `2 / 2**n`, the minimum achievable exact sign-flip p. At n=11 the
floor is 0.0010, so alpha=0.05 IS structurally reachable; the test simply does
not reach it.)

## Verdict

The adapted-vs-control delta NULL VERDICT HOLDS at n=11. On both terrains the
mean delta shrinks toward zero, the 95% CI straddles zero, and the exact
two-sided sign-flip p moves further from significance, not toward it:

- Slippery: p went 0.5625 (n=7) -> 0.7266 (n=11); mean delta -1.10 pp -> -0.42 pp;
  signs 4/7 -> 7/11. No significant adapted-vs-control effect. The v0.3.0
  single-seed +5.1 pp slippery lift does NOT replicate as a cross-seed effect.
- Rough: p went 0.3906 (n=7) -> 0.7676 (n=11); mean delta -1.58 pp -> +0.55 pp
  (sign of the mean flipped, magnitude near zero); signs 2/7 -> 5/11. Rough
  retention is preserved within noise; no significant degradation.

Adding 4 seeds did not surface a hidden effect. It tightened the CIs (slippery
half-width 4.1 pp -> 2.4 pp) and pushed both terrains closer to a clean null.

## Verified vs assumed

Verified (executed, output shown above):
- All 11 seeds load and appear at BOTH ff=0.0 and ff=0.5 on BOTH terrains
  (`combine_pilot_runs` seed list = `[7,42,99,123,314,1618,1729,2024,2718,4096,6022]`).
- New ext cells use the identical metrics schema (`success_rate`, `num_episodes`)
  as the old cells.
- Numbers above are the raw return values of the repo's `paired_delta` /
  `_exact_sign_flip_p`; no hand statistics.

Config comparability (VERIFIED, confound ruled out):
- Diffed `commands.sh`, `config.yaml`, and `adapt_override.yaml` between an old
  cell (scale seed 99, ff=0.0) and a new cell (ext seed 4096, ff=0.0). The eval
  pipeline is identical except seed/name/tags: same `configs/env/slippery.yaml`
  and `configs/env/rough.yaml`, same `--num-episodes 128`, same fine-tune args
  (`--num-envs 4096 --max-iterations 200`). `adapt_override.yaml` differs ONLY
  in `name` and `seed`. The one path difference (`~/isaac-sim-venv` ->
  `~/Sim/isaac-sim-venv`) is the documented 2026-05-20 Isaac Sim relocation, not
  a config change. So the new cells were evaluated on the SAME slippery terrain
  difficulty as the old ones; the high ff=0.0 slippery scores are a genuine
  seed-to-seed outcome, not an easier-eval artifact.

Riskiest remaining caveat (real, not a methodology bug):
- The 4 new ext slippery cells sit much higher than the old 7: ff=0.0 slippery
  for {1618, 2024, 4096, 6022} = {0.9922, 0.9844, 1.0000, 0.9922} vs the old
  range ~0.85-0.93. Three of four new slippery ff=0.0 cells are at or near a
  100% success ceiling, so their delta is mechanically clamped toward 0 (e.g.
  seed 4096 = +0.0000, seed 1618/6022 = +0.0078). Since the terrain config is
  verified identical, this is not a confound but it IS a ceiling effect: those
  seeds had little room to move, so they contribute "delta near zero" partly
  for mechanical reasons. The null verdict is robust regardless (the slippery
  mean was already negative at n=7 and the p never approached significance), but
  the magnitude of the n=11 shrinkage toward zero is partly ceiling-driven, not
  purely additional independent evidence. A future extension should pick seeds
  whose baseline ff=0.0 slippery success leaves headroom (<0.95) to avoid
  diluting the paired contrast.
