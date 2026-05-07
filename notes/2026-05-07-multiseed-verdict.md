# Multi-seed pilot verdict (2026-05-07)

Status: COMPLETE

## Mission recap

The v0.3.0 single-seed sweep reported a +5.1 pp slippery lift at ff=0.5 vs ff=0.0 and a +6.1 pp rough lift at ff=0.1. The 2026-05-07 rigor pass found that neither cell survives Holm-Bonferroni correction with single-seed data, and cross-seed variance was unmeasured. This pilot fills that gap with 2 ff values (0.0, 0.5) x 3 seeds (42, 123, 7), 200-iter PPO fine-tune, 128-episode eval on rough + slippery.

## Statistical floor with n=3

With n_seeds=3 the exact two-sided sign-flip permutation test has 2**3 = 8 permutations and a minimum achievable p of 2/8 = 0.25. This is a structural ceiling, not a bug. A 'significant' result at alpha=0.05 is impossible with n=3; the verdict therefore rests on effect-size CIs and per-seed pattern, not on a single p-value.

## Per-cell raw success rates

| ff   | seed | terrain  |   n |   k | success |
|-----:|-----:|:---------|----:|----:|--------:|
| 0.00 |    7 | rough    | 128 | 125 | 0.9766 |
| 0.00 |   42 | rough    | 130 | 118 | 0.9077 |
| 0.00 |  123 | rough    | 128 | 122 | 0.9531 |
| 0.50 |    7 | rough    | 130 | 115 | 0.8846 |
| 0.50 |   42 | rough    | 128 | 118 | 0.9219 |
| 0.50 |  123 | rough    | 129 | 120 | 0.9302 |
| 0.00 |    7 | slippery | 133 | 117 | 0.8797 |
| 0.00 |   42 | slippery | 134 | 119 | 0.8881 |
| 0.00 |  123 | slippery | 132 | 121 | 0.9167 |
| 0.50 |    7 | slippery | 130 | 118 | 0.9077 |
| 0.50 |   42 | slippery | 131 | 121 | 0.9237 |
| 0.50 |  123 | slippery | 133 | 122 | 0.9173 |

## Per-ff cross-seed summary (mean +/- SE)

| terrain  | ff   |  n_seeds | mean   | std    | SE     | per-seed         |
|:---------|-----:|---------:|-------:|-------:|-------:|:-----------------|
| rough    | 0.00 |        3 | 0.9458 | 0.0350 | 0.0202 | 123=0.953, 42=0.908, 7=0.977 |
| rough    | 0.50 |        3 | 0.9122 | 0.0243 | 0.0140 | 123=0.930, 42=0.922, 7=0.885 |
| slippery | 0.00 |        3 | 0.8948 | 0.0194 | 0.0112 | 123=0.917, 42=0.888, 7=0.880 |
| slippery | 0.50 |        3 | 0.9162 | 0.0080 | 0.0046 | 123=0.917, 42=0.924, 7=0.908 |

## Paired-by-seed delta (ff=0.5 - ff=0.0)

| terrain  | n_seeds | deltas (per seed)              | mean  | SE    | 95% CI            | exact perm p (two-sided) |
|:---------|--------:|:-------------------------------|------:|------:|:------------------|-------------------------:|
| slippery |       3 | +0.0280, +0.0356, +0.0006 | +0.0214 | 0.0106 | [-0.0243, +0.0671] | 0.2500 |
| rough    |       3 | -0.0919, +0.0142, -0.0229 | -0.0336 | 0.0311 | [-0.1674, +0.1003] | 0.5000 |


## Verdict

### Slippery (prior optimum at ff=0.5)

- Prior single-seed lift: +5.1 pp.
- Cross-seed mean delta: +2.14 pp (SE 1.06, 95% CI [-2.43, +6.71] pp).
- Per-seed signs: 3/3 positive.
- Exact two-sided sign-flip p: 0.250 (floor at n=3 is 0.25).
- CI crosses zero: yes.

### Rough (retention check)

- Prior single-seed lift: +1.5 pp.
- Cross-seed mean delta: -3.36 pp (SE 3.11, 95% CI [-16.74, +10.03] pp).
- Per-seed signs: 1/3 positive.
- Exact two-sided sign-flip p: 0.500 (floor at n=3 is 0.25).
- CI crosses zero: yes.


## Honest claim Ashfall can now make

- Slippery: the failure curriculum at ff=0.5 yields a positive lift on slippery in 3/3 seeds, but the 95% CI on the cross-seed mean still crosses zero at n=3, so the effect is directionally consistent but not yet rigour-passing.
- Rough: cross-seed mean delta -3.36 pp (95% CI [-16.74, +10.03] pp); the curriculum did appear to degrade rough retention; see CI.

Reviewer-defensible one-liner: "The failure-fraction sweep's v0.3.0 ff=0.5 slippery optimum is directionally seed-stable across 3 seeds (3/3 positive) but does not clear an n=3 95% CI on the paired-seed mean. Multi-seed scaling (5+ seeds) is the gate before any v0.4.0 claim that the lift is real."
