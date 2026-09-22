# Historical reset-file curriculum (Phase I)

`results/legacy_row0_curriculum/` preserves, byte for byte and with a SHA-256
provenance index, every metrics file, report and note of the Phase-I
experiment: the failure-fraction sweep, the multi-seed pilot, the n=7 and
n=11 seed scaling, and the mode-subset ablation. Nothing under that prefix is
modified, regenerated or reinterpreted at the file level. This page is where
its interpretation lives.

## What was measured

Phase I fine-tuned a policy while a fraction of environment resets were
seeded from recorded failure trajectories, and compared success rates against
a no-curriculum control across 11 training seeds with an exact paired
sign-flip test. The numbers reproduce exactly from the committed per-seed
metrics and are pinned by `tests/test_multiseed_combined.py`: slippery mean
delta -0.4155 pp with p = 0.726562, rough +0.5485 pp with p = 0.767678
(n = 11, seeds 7, 42, 99, 123, 314, 1618, 1729, 2024, 2718, 4096, 6022).

## Why it answers a different question than its title

Two defects, both found after the numbers were published, decide the scope:

1. **The treatment was never delivered.** The reset bridge seeded row 0 of
   each trajectory, and row 0 of every shipped synthetic trajectory is a
   nominal gait state. Roughly half the environments reset to a lightly
   jittered stand. In the vocabulary of the current design there was no
   intervention, no verified departure and no phenotype in the seeded state:
   gates D1 to D3 would all have failed. The null is therefore uninformative
   about failure curricula, neither evidence for nor against them.
2. **The terrain contrast is mislabelled.** The `terrain` block of the
   environment configs is declared but not applied, so the rough and slippery
   arms ran on the same generated terrain and differed only in friction
   randomisation. The contrast is real; the axis label was wrong.

In the new ontology the historical `failure_fraction` names the share of
reset files drawn from the trajectory pool. It is not a phenotype-conditioned
anything, and the trajectories it drew from were fixtures, not physics.

## Caveats about the archived documents themselves

The archive is sealed, so these are recorded here rather than corrected in
place:

- The reproduction command inside `multiseed_scale_ext_2026-06-02_ANALYSIS.md`
  names the pre-archive `results/` path. Run it against
  `results/legacy_row0_curriculum` instead; as written it pairs no seeds, and
  the analysis code now raises on an empty pairing instead of reporting a
  null.
- The confidence intervals in `REPORT.md` were computed with a sign-inverted
  BCa bias-correction term. The production code was fixed afterwards and a
  guard (`tests/test_stats_primary.py`) compares it against an independent
  jackknife implementation.
- `multiseed_n11_scale_verdict.md` is named for n=11 but contains the n=7
  analysis.
- Several archived notes carry the author's original absolute filesystem
  paths.

## What the archive still supports

- The statistical machinery: exact paired sign-flip testing across training
  seeds, with the achievable p floor stated, and reproducibility of the
  headline from committed metrics.
- A correctly executed measurement of a treatment that was not applied,
  which is the reason the delivery gates exist.

## What it does not support

- Any statement that a failure curriculum does or does not improve
  robustness.
- Any statement about a terrain contrast.
- Any detector accuracy claim: the 18 synthetic trajectories were authored
  around the detector's thresholds.

The legacy runner (`ashfall.experiment.runner`) now refuses
`seed_row_strategy: first` unless a config carries the tag
`legacy_row0_reproduction`, which exists only to regenerate the archived
commands.

## Archived design documents

Moved here unchanged when the study was simplified around Failure-Boundary Replay
(`docs/legacy/audits/ASHFALL_SIMPLIFICATION_AUDIT.md`):

- `phase2/PREREGISTRATION.md`, `phase2/HYPOTHESIS.md`, `phase2/INPUTS.md`: the H0 to H4
  preregistration of the causal-reproduction design. Superseded by
  `docs/research/PREREGISTRATION.md`. Never run.
- `methodology_v3_causal.md`: the v3 methodology (six phenotypes, D1/D2/D3, H0 to H4).
  Superseded by the FBR methodology, itself archived as `methodology_fbr.md`.

Their contents are byte-identical to the originals, so relative links inside them point to
their old locations (for example `claims_ledger.md` means `docs/claims_ledger.md`).

## Moved here at the program archive (2026-09-22)

Moved unchanged (byte-identical, only their paths differ) when the program was archived after five
NO-GO studies. None of them describes a result of those studies; relative links inside them
point to their old locations. The final record is [`../ARCHIVE.md`](../ARCHIVE.md).

| Now under `docs/legacy/` | Was | What it is |
|---|---|---|
| `architecture/` | `docs/architecture/` | architecture notes and diagrams for v2, v3 and the FBR GO2 design |
| `audits/` | `docs/audits/`, `docs/research/ASHFALL_SIMPLIFICATION_AUDIT.md` | code and design audits of earlier Ashfall versions |
| `detector/` | `docs/detector/` | event-detector labeling and validation protocol; never validated |
| `experiments/low_friction_protocol.md` | `docs/experiments/` | the v2 counterexample protocol, already superseded before the archive |
| `hardware/controlled_failure_protocol.md` | `docs/hardware/` | GO2 low-friction protocol; never run |
| `runbooks/` | `docs/runbooks/` | FBR and Isaac Lab H0 runbooks |
| `methodology_fbr.md` | `docs/methodology.md` | FBR methodology with its planned GO2 steps |
| `methodology/ff_sweep_rigor.md` | `docs/methodology/` | Phase-I failure-fraction sweep methodology |
| `go2_field_notes.md` | `docs/go2_field_notes.md` | measured GO2 platform notes, not an Ashfall result |
| `limitations_pre_archive.md` | `docs/limitations.md` | the limitations list as it stood before the archive |

Three planning documents from the FBR phase (a paper outline, a hardware demo plan and a note on
paper structure) are not part of the public archive. They contained plans, not evidence.

