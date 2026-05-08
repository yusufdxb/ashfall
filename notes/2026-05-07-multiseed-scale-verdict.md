wrote /home/yusuf/Projects/ashfall/notes/2026-05-07-multiseed-scale-verdict.md

# Multi-seed scale verdict (n=7, 2026-05-07)

Status: COMPLETE

## Mission recap

The 2026-05-07 n=3 pilot found a directionally consistent slippery lift at ff=0.5 (3/3 positive seeds, mean +2.14 pp) but with a structural permutation p-floor of 2/8 = 0.25 that prevents alpha=0.05 significance regardless of effect size. This scale pass adds 4 new seeds (99, 314, 1729, 2718) at ff in {0.0, 0.5} for an n=7 paired-by-seed analysis. At n=7 the exact sign-flip permutation test has 2**7 = 128 permutations, so p-floor = 2/128 = 0.0156 < 0.05; alpha=0.05 IS reachable when the effect is consistent across seeds.

## Statistical floor at n=7

With n_seeds=7 the exact two-sided sign-flip permutation test has 2**7 = 128 permutations and a minimum achievable p of 2/128 = 0.0156. CIs use the t-multiplier at df=6 (t_975 = 2.447 for n=7).

## Per-cell raw success rates

| ff   | seed | terrain  |   n |   k | success |
|-----:|-----:|:---------|----:|----:|--------:|
| 0.00 |    7 | rough    | 128 | 125 | 0.9766 |
| 0.00 |   42 | rough    | 130 | 118 | 0.9077 |
| 0.00 |   99 | rough    | 129 | 119 | 0.9225 |
| 0.00 |  123 | rough    | 128 | 122 | 0.9531 |
| 0.00 |  314 | rough    | 131 | 114 | 0.8702 |
| 0.00 | 1729 | rough    | 134 | 116 | 0.8657 |
| 0.00 | 2718 | rough    | 130 | 123 | 0.9462 |
| 0.50 |    7 | rough    | 130 | 115 | 0.8846 |
| 0.50 |   42 | rough    | 128 | 118 | 0.9219 |
| 0.50 |   99 | rough    | 131 | 113 | 0.8626 |
| 0.50 |  123 | rough    | 129 | 120 | 0.9302 |
| 0.50 |  314 | rough    | 128 | 120 | 0.9375 |
| 0.50 | 1729 | rough    | 133 | 114 | 0.8571 |
| 0.50 | 2718 | rough    | 128 | 120 | 0.9375 |
| 0.00 |    7 | slippery | 133 | 117 | 0.8797 |
| 0.00 |   42 | slippery | 134 | 119 | 0.8881 |
| 0.00 |   99 | slippery | 138 | 117 | 0.8478 |
| 0.00 |  123 | slippery | 132 | 121 | 0.9167 |
| 0.00 |  314 | slippery | 134 | 121 | 0.9030 |
| 0.00 | 1729 | slippery | 130 | 121 | 0.9308 |
| 0.00 | 2718 | slippery | 130 | 120 | 0.9231 |
| 0.50 |    7 | slippery | 130 | 118 | 0.9077 |
| 0.50 |   42 | slippery | 131 | 121 | 0.9237 |
| 0.50 |   99 | slippery | 139 | 119 | 0.8561 |
| 0.50 |  123 | slippery | 133 | 122 | 0.9173 |
| 0.50 |  314 | slippery | 131 | 117 | 0.8931 |
| 0.50 | 1729 | slippery | 133 | 117 | 0.8797 |
| 0.50 | 2718 | slippery | 139 | 116 | 0.8345 |

## Per-ff cross-seed summary (mean +/- SE)

| terrain  | ff   |  n_seeds | mean   | std    | SE     | per-seed         |
|:---------|-----:|---------:|-------:|-------:|-------:|:-----------------|
| rough    | 0.00 |        7 | 0.9203 | 0.0420 | 0.0159 | 7=0.977, 42=0.908, 99=0.922, 123=0.953, 314=0.870, 1729=0.866, 2718=0.946 |
| rough    | 0.50 |        7 | 0.9045 | 0.0354 | 0.0134 | 7=0.885, 42=0.922, 99=0.863, 123=0.930, 314=0.938, 1729=0.857, 2718=0.938 |
| slippery | 0.00 |        7 | 0.8984 | 0.0289 | 0.0109 | 7=0.880, 42=0.888, 99=0.848, 123=0.917, 314=0.903, 1729=0.931, 2718=0.923 |
| slippery | 0.50 |        7 | 0.8874 | 0.0329 | 0.0124 | 7=0.908, 42=0.924, 99=0.856, 123=0.917, 314=0.893, 1729=0.880, 2718=0.835 |

## Paired-by-seed delta (ff=0.5 - ff=0.0)

| terrain  | n_seeds | deltas (per seed)              | mean  | SE    | 95% CI            | exact perm p (two-sided) |
|:---------|--------:|:-------------------------------|------:|------:|:------------------|-------------------------:|
| slippery |       7 | +0.0280, +0.0356, +0.0083, +0.0006, -0.0099, -0.0511, -0.0885 | -0.0110 | 0.0168 | [-0.0520, +0.0301] | 0.5625 |
| rough    |       7 | -0.0919, +0.0142, -0.0599, -0.0229, +0.0673, -0.0085, -0.0087 | -0.0158 | 0.0193 | [-0.0631, +0.0315] | 0.3906 |


## Pilot-only (n=3) vs scaled (n=7) comparison

| terrain  | n  | mean delta | 95% CI                | exact perm p |
|:---------|---:|-----------:|:----------------------|-------------:|
| slippery (pilot, n=3)      |  3 | +2.14 pp | [-2.43, +6.71] pp | 0.2500 |
| slippery (scaled, n=7)     |  7 | -1.10 pp | [-5.20, +3.01] pp | 0.5625 |
| rough (pilot, n=3)         |  3 | -3.36 pp | [-16.74, +10.03] pp | 0.5000 |
| rough (scaled, n=7)        |  7 | -1.58 pp | [-6.31, +3.15] pp | 0.3906 |

## Verdict

### Slippery (prior optimum at ff=0.5)

- Prior single-seed lift: +5.1 pp.
- n=3 pilot mean delta (subset): see notes/2026-05-07-multiseed-verdict.md.
- Cross-seed mean delta (n=7): -1.10 pp (SE 1.68, 95% CI [-5.20, +3.01] pp).
- Per-seed signs: 4/7 positive.
- Exact two-sided sign-flip p: 0.5625 (floor at n=7 is 0.0156).
- CI crosses zero: yes.
- Clears alpha=0.05: no.

### Rough (retention check)

- Prior single-seed lift: +1.5 pp.
- n=3 pilot mean delta (subset): see notes/2026-05-07-multiseed-verdict.md.
- Cross-seed mean delta (n=7): -1.58 pp (SE 1.93, 95% CI [-6.31, +3.15] pp).
- Per-seed signs: 2/7 positive.
- Exact two-sided sign-flip p: 0.3906 (floor at n=7 is 0.0156).
- CI crosses zero: yes.
- Clears alpha=0.05: no.


## Honest claim Ashfall can now make

- Slippery: the failure curriculum at ff=0.5 lifts slippery success in 4/7 seeds; permutation p 0.5625 does not clear alpha=0.05; effect is directionally plausible but unreliable across seeds.
- Rough: rough retention is NOT preserved on average (mean -1.58 pp, 95% CI [-6.31, +3.15] pp, only 2/7 positive); the curriculum trades rough proficiency for slippery training.

