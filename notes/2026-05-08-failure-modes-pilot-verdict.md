# Mode-subset Stage-1 verdict (2026-05-08)

Stage-1 triage: 6 mode subsets x 3 pilot seeds at fixed ff=0.5, paired against the existing n=3 ff=0.0 multiseed_pilot baseline.
Exact sign-flip p-floor at n=3 is 2/8 = 0.2500; alpha=0.05 cannot be reached here. Stage-1 is directional triage only.

## Headline

**Stage-2 candidates (3/3 positive slippery sign):** `all_modes`

**Clear losers (3/3 negative slippery sign):** `slip_only`, `severe_only`, `severe_plus_slip`

## 18-cell raw success rates

| subset                  | seed | terrain  |   n |   k | success |
|:------------------------|-----:|:---------|----:|----:|--------:|
| all_modes              |    7 | rough    | 130 | 115 | 0.8846 |
| all_modes              |   42 | rough    | 128 | 118 | 0.9219 |
| all_modes              |  123 | rough    | 129 | 120 | 0.9302 |
| all_modes              |    7 | slippery | 130 | 118 | 0.9077 |
| all_modes              |   42 | slippery | 131 | 121 | 0.9237 |
| all_modes              |  123 | slippery | 133 | 122 | 0.9173 |
| command_mismatch_only  |    7 | rough    | 128 | 123 | 0.9609 |
| command_mismatch_only  |   42 | rough    | 128 | 122 | 0.9531 |
| command_mismatch_only  |  123 | rough    | 129 | 121 | 0.9380 |
| command_mismatch_only  |    7 | slippery | 139 | 113 | 0.8129 |
| command_mismatch_only  |   42 | slippery | 135 | 120 | 0.8889 |
| command_mismatch_only  |  123 | slippery | 135 | 116 | 0.8593 |
| severe_only            |    7 | rough    | 130 | 122 | 0.9385 |
| severe_only            |   42 | rough    | 130 | 120 | 0.9231 |
| severe_only            |  123 | rough    | 131 | 114 | 0.8702 |
| severe_only            |    7 | slippery | 136 | 119 | 0.8750 |
| severe_only            |   42 | slippery | 137 | 119 | 0.8686 |
| severe_only            |  123 | slippery | 133 | 119 | 0.8947 |
| severe_plus_slip       |    7 | rough    | 128 | 119 | 0.9297 |
| severe_plus_slip       |   42 | rough    | 129 | 120 | 0.9302 |
| severe_plus_slip       |  123 | rough    | 129 | 123 | 0.9535 |
| severe_plus_slip       |    7 | slippery | 143 | 116 | 0.8112 |
| severe_plus_slip       |   42 | slippery | 137 | 120 | 0.8759 |
| severe_plus_slip       |  123 | slippery | 134 | 122 | 0.9104 |
| slip_only              |    7 | rough    | 129 | 123 | 0.9535 |
| slip_only              |   42 | rough    | 130 | 115 | 0.8846 |
| slip_only              |  123 | rough    | 129 | 118 | 0.9147 |
| slip_only              |    7 | slippery | 148 | 114 | 0.7703 |
| slip_only              |   42 | slippery | 128 |  92 | 0.7188 |
| slip_only              |  123 | slippery | 132 | 119 | 0.9015 |
| slip_plus_cm           |    7 | rough    | 129 | 126 | 0.9767 |
| slip_plus_cm           |   42 | rough    | 129 | 123 | 0.9535 |
| slip_plus_cm           |  123 | rough    | 130 | 118 | 0.9077 |
| slip_plus_cm           |    7 | slippery | 133 | 120 | 0.9023 |
| slip_plus_cm           |   42 | slippery | 131 | 123 | 0.9389 |
| slip_plus_cm           |  123 | slippery | 134 | 121 | 0.9030 |

## Baseline cells (ff=0.0, multiseed_pilot 2026-05-07)

| seed | terrain  |   n |   k | success |
|-----:|:---------|----:|----:|--------:|
|    7 | rough    | 128 | 125 | 0.9766 |
|   42 | rough    | 130 | 118 | 0.9077 |
|  123 | rough    | 128 | 122 | 0.9531 |
|    7 | slippery | 133 | 117 | 0.8797 |
|   42 | slippery | 134 | 119 | 0.8881 |
|  123 | slippery | 132 | 121 | 0.9167 |

## Per-subset paired delta vs ff=0.0 baseline (slippery)

| subset                  | n | per-seed deltas               | mean   | SE     | 95% CI            | exact perm p | signs (+/-) |
|:------------------------|--:|:------------------------------|-------:|-------:|:------------------|-------------:|:------------|
| all_modes              | 3 | +0.0280 / +0.0356 / +0.0006   | +0.0214 | 0.0106 | [-0.0243, +0.0671] | 0.2500 | 3/3 pos, 0/3 neg |
| slip_only              | 3 | -0.1094 / -0.1693 / -0.0152   | -0.0980 | 0.0449 | [-0.2910, +0.0951] | 0.2500 | 0/3 pos, 3/3 neg |
| command_mismatch_only  | 3 | -0.0667 / +0.0008 / -0.0574   | -0.0411 | 0.0211 | [-0.1321, +0.0499] | 0.5000 | 1/3 pos, 2/3 neg |
| slip_plus_cm           | 3 | +0.0226 / +0.0509 / -0.0137   | +0.0199 | 0.0187 | [-0.0605, +0.1003] | 0.5000 | 2/3 pos, 1/3 neg |
| severe_only            | 3 | -0.0047 / -0.0194 / -0.0219   | -0.0154 | 0.0054 | [-0.0385, +0.0078] | 0.2500 | 0/3 pos, 3/3 neg |
| severe_plus_slip       | 3 | -0.0685 / -0.0121 / -0.0062   | -0.0290 | 0.0198 | [-0.1144, +0.0565] | 0.2500 | 0/3 pos, 3/3 neg |

## Per-subset paired delta vs ff=0.0 baseline (rough)

| subset                  | n | per-seed deltas               | mean   | SE     | 95% CI            | exact perm p | signs (+/-) |
|:------------------------|--:|:------------------------------|-------:|-------:|:------------------|-------------:|:------------|
| all_modes              | 3 | -0.0919 / +0.0142 / -0.0229   | -0.0336 | 0.0311 | [-0.1674, +0.1003] | 0.5000 | 1/3 pos, 2/3 neg |
| slip_only              | 3 | -0.0231 / -0.0231 / -0.0384   | -0.0282 | 0.0051 | [-0.0502, -0.0062] | 0.2500 | 0/3 pos, 3/3 neg |
| command_mismatch_only  | 3 | -0.0156 / +0.0454 / -0.0151   | +0.0049 | 0.0203 | [-0.0823, +0.0921] | 1.0000 | 1/3 pos, 2/3 neg |
| slip_plus_cm           | 3 | +0.0002 / +0.0458 / -0.0454   | +0.0002 | 0.0263 | [-0.1131, +0.1135] | 0.7500 | 2/3 pos, 1/3 neg |
| severe_only            | 3 | -0.0381 / +0.0154 / -0.0829   | -0.0352 | 0.0284 | [-0.1574, +0.0870] | 0.5000 | 1/3 pos, 2/3 neg |
| severe_plus_slip       | 3 | -0.0469 / +0.0225 / +0.0004   | -0.0080 | 0.0205 | [-0.0961, +0.0801] | 1.0000 | 2/3 pos, 1/3 neg |

## Caveats

- n=3 sign-flip p-floor is 0.25; alpha=0.05 cannot be cleared at this seed budget. Stage-1 is directional only.
- 3/3 positive slippery sign DOES NOT mean the subset works; it means the subset survives Stage-1 triage and warrants n>=5 scaling.
- The ff=0.0 baseline cells are not re-run in this sweep; we reuse the existing 2026-05-07 multiseed_pilot ff=0.0 cells. This holds all training inputs except the curriculum constant across the paired comparison.

## Interpretation

- The only Stage-2 candidate is `all_modes`, which is the same configuration the 2026-05-07 n=7 scaling found to have near-zero reliable effect. The mean slippery delta here (+2.14pp at n=3 paired vs n=3 baseline) reproduces the pilot's directional lift in sign and magnitude, but the n=7 expansion already showed the wider 95% CI crosses zero. Stage-1 evidence is consistent with the n=7 verdict: no curriculum-restriction trick at fixed ff=0.5 unlocks a robust slippery improvement.
- Every restricted subset (`slip_only`, `command_mismatch_only`, `slip_plus_cm`, `severe_only`, `severe_plus_slip`) underperforms the unfiltered pool on slippery in the paired test. Three of them (`slip_only`, `severe_only`, `severe_plus_slip`) clear the 3/3-negative bar and are formal clear losers.
- `slip_only` is the most striking single-axis loser: per-seed deltas -0.109, -0.169, -0.015. Restricting the curriculum to slip parquets actively damages the slippery-terrain success rate vs the no-curriculum baseline. Hypothesis: the slip parquets are too narrow a state distribution to induce broad slippery competence, and removing the other modes loses useful generalization signal.
- `command_mismatch_only` and `slip_plus_cm` come out ambiguous (mixed signs). Both have one seed pull strongly positive on slippery, the others negative or mixed. At n=3 they cannot be distinguished from noise.

## Next-step recommendation

Stage-1 produced no shortlist for Stage-2 scaling beyond the trivial "the full pool was the best of a bad set." A redesign of the curriculum (per the 2026-05-07 next-steps menu, item 1) is more productive than further mode-subset triage. Specifically: this Stage-1 sweep adds evidence that the failure-fraction lever and the mode-composition lever are both saturated within the current synthesis pool. The next bet is changing what gets generated, not how it is sampled.


