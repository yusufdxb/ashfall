# Failure-Boundary Replay: negative result in the CPU toy

Status: **FINAL. Do not edit the numbers in this file.** Any later study is a new hypothesis
with its own preregistration; it does not revise this one.

Evidence kind: `toy_mechanism` (slip cart-pole). Nothing here is a GO2 or quadruped result.

## Verdict, plainly

1. The original FBR hypothesis ("a real failure defines a local simulation curriculum that
   beats broad domain randomization") **failed its preregistered toy gate** (toy v2, NO-GO).
2. FBR **did not beat broad domain randomization**: -2.7 pp held-out failure, 95% CI -6.1 to
   +0.6, exact sign-flip p 0.105 (toy v2); +0.3 pp, p 0.898 (toy v1).
3. FBR **degraded nominal performance beyond the registered margin**: nominal success -6.5 pp
   against the unrepaired baseline (90% CI -8.4 to -4.7; margin was -2 pp; the claims ledger rounds the same -6.55 pp to -6.6) in toy v2, and
   -6.8 pp in toy v1. Tracking RMSE rose 0.038 m/s.
4. The failure's own seed state **underperformed an ordinary patch-entry state** taken from a
   successful nominal rollout: C (nominal entry) beat D (failure entry) by 3.6 pp held-out
   failure, 95% CI 1.5 to 5.6, p 0.0029.
5. Therefore the claim "the real failure state itself carries special training value" is
   **unsupported**, and in this toy it is refuted.

FBR did beat ADR in toy v2 (-6.7 pp, p 0.0010). That is not enough: the registered primary was
an intersection-union test that required beating both ADR and broad DR.

The next study (precursor-time sweep, `docs/research/PRECURSOR_SWEEP_PREREGISTRATION.md`) asks a
**new question**. It is not a rescue of FBR and does not reinterpret these results.

## Toy v1 (exploratory design, then corrected)

| item | value |
|---|---|
| code | `f339087` (`src/ashfall/fbr/toy_study.py`, `toy_slip.py`); rerun after the censoring-kind fix at `f3ccc1b` |
| config | `ToySlipConfig()` and `ESConfig()` defaults; baseline ES seed 0, 200 iterations; arms 120 iterations, 16 antithetic pairs, 24 contexts |
| deployment world | patch friction 0.08, patch start 1.0 m, command 1.1 m/s |
| training seeds | 11, 23, 47, 59, 71, 83, 97, 101, 113, 127, 131, 149 |
| test | paired by training seed, exact two-sided sign-flip, Student-t 95% CI |
| FBR (D) minus broad DR (B), held-out failure | +0.3 pp (95% CI -4.7 to +5.3), p 0.898: NULL |
| FBR minus nominal-seed boundary replay (C) | +0.5 pp (-3.5 to +4.5), p 0.783 |
| FBR nominal success minus baseline | -6.8 pp (-8.8 to -4.8), p 0.0005: fails the -2 pp margin |
| artifacts | `results/fbr_toy/f339087/`, `results/fbr_toy/f3ccc1b_corrected/`, logs beside them |

## Toy v2 (the preregistered decision gate)

| item | value |
|---|---|
| preregistration | `docs/research/TOY_V2_PREREGISTRATION.md`, committed with its code at `8f49f02` (`8f49f025a7abe7acb13ecfa421ed7a041342b039`) before the run |
| code | `src/ashfall/fbr/toy_study_v2.py` at `8f49f02`; spec id `5b4196a676aad213345998507feef22c47026e3e6c2738c26aa82413361202e6` |
| baseline checkpoint | `baseline.npy`, SHA-256 `2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3` (identical to v1) |
| worlds | W1 (0.08, 1.0, 1.1), W2 (0.06, 0.8, 1.0), W3 (0.10, 1.2, 1.2), W4 (0.12, 0.9, 1.3), W5 (0.07, 1.1, 0.9), W6 (0.09, 0.7, 1.15); 6 held-out cells each, 256 episodes per cell |
| training seeds | the v1 twelve |
| compute | B and ADR 16,262,400 simulator steps per seed; C 16,144,175; D at most 16,215,760 (out-of-training steps charged) |
| statistics | exact two-sided sign-flip over 12 seed-paired means, alpha 0.05; IUT primary (D beats ADR and D beats B) |

Held-out failure, mean over seeds and worlds:

| arm | held-out failure | nominal success | tracking RMSE (m/s) |
|---|---|---|---|
| A no repair | 0.480 | 0.973 | 0.513 |
| B broad DR | 0.351 | 0.963 | 0.515 |
| ADR | 0.391 | 0.961 | 0.511 |
| C nominal-entry replay | 0.288 | 0.918 | 0.550 |
| D FBR (failure patch-entry replay) | 0.324 | 0.907 | 0.550 |

Registered tests:

| comparison | mean | 95% CI | exact p | outcome |
|---|---|---|---|---|
| D minus ADR, held-out failure | -6.7 pp | -9.4 to -4.1 | 0.0010 | D better |
| D minus B, held-out failure | -2.7 pp | -6.1 to +0.6 | 0.105 | not significant |
| IUT primary | | | | **does not reject** |
| D minus A, nominal success (90% CI, one-sided lower bound vs -2 pp) | -6.5 pp | -8.4 to -4.7 | | **fails** |
| D minus A, tracking RMSE (upper bound vs +0.05 m/s) | +0.038 | 0.032 to 0.043 | | passes |
| D minus C, held-out failure (descriptive) | +3.6 pp | +1.5 to +5.6 | 0.0029 | the ordinary entry state is better |
| Decision | | | | **NO-GO** for the GO2 FBR study |

Also recorded: the reconstruction step recovered the hidden friction inside the reproduced
range in only 1 of 6 worlds.

## Artifact index (SHA-256; these files are immutable)

`tests/test_frozen_results.py` recomputes every hash below and fails if any file changes.

```
446197b78cd3d841aca65d5106dea347ec1a64ec58f608b6e293d70a2b22f42f  results/fbr_toy/f339087/result.json
2922adfd54dbfa1e30acc372aefa8eb12aaa6d9a9de4e272f721cf5d9e33d27a  results/fbr_toy/f339087/capsule.json
2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3  results/fbr_toy/f339087/baseline.npy
f3cb03fb3adf679e7fda474f28183775ee6f64cb078316720e8c61952d89c1c7  results/fbr_toy/f3ccc1b_corrected/result.json
742497cb8c54283e3028ba475d2723551731046c6de6a3df207ff5f56874897b  results/fbr_toy/f339087_efficiency_exploratory/efficiency.json
990ffc19646977079bad46f6d6ad0a326c9314735935e14b720869b588b3719a  results/fbr_toy_v2/8f49f02/result.json
2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3  results/fbr_toy_v2/8f49f02/baseline.npy
5013f3336576ea9c3db6e15ec10bb01da06027d0ddaa5be6790ef86c828f67e1  results/fbr_toy_v2/8f49f02/capsule_W1.json
15d202713094f5e546e95d89c82700a12a38a6bdebf0716da22d8cc7172d00ef  results/fbr_toy_v2/8f49f02/capsule_W2.json
efaa1cdfe4d192753c2998b698bcff6dbb1f884699139a7836b02817dec0f672  results/fbr_toy_v2/8f49f02/capsule_W3.json
191881942cb993c10fe69f402604990f46a74c80721354c159075bae96ee779d  results/fbr_toy_v2/8f49f02/capsule_W4.json
3b5c6cd49e98851d8bea0fb4770f03e5432ac95808fdf3bb1551585bd98dc016  results/fbr_toy_v2/8f49f02/capsule_W5.json
809ea58594940184733f897ac13d297ab9cafafe76c7db72a37559412af0c4ab  results/fbr_toy_v2/8f49f02/capsule_W6.json
3d0ce59d36ab7e8ded25711f01a71a950d9c7eb0a76c8d9f1a1a13e02d0cb221  results/fbr_toy_v2/8f49f02.log
```

Housekeeping note: the toy v2 artifacts were produced at `8f49f02` but had been excluded by the
`results/*/**` ignore rule, so they existed only on this machine. They are committed unchanged
alongside this note (same bytes, hashes above).

## What this does and does not say

It says that, in a one-axis toy chosen to favour FBR, seeding replay from the failed
trajectory's patch-entry state did not beat broad DR, cost nominal performance, and did worse
than an ordinary entry state. It does not say that failure curricula never work, and it says
nothing about the GO2.
