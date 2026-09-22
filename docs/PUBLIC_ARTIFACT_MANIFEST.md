# Public artifact manifest

Every file under `results/` in this public release, compared with the private canonical archive
(commit `dfb9c66`, not published; see [`RESEARCH_HISTORY.md`](RESEARCH_HISTORY.md)). The
machine-readable form, with both hashes for every file, is
[`results/PUBLIC_ARTIFACT_MANIFEST.json`](../results/PUBLIC_ARTIFACT_MANIFEST.json); `tests/test_public_release.py`
checks it against the files.

Statuses: **IDENTICAL** (same bytes as the private archive), **SANITIZED_PATHS_ONLY** (machine-local
path text replaced, nothing else changed), **DERIVED_SUMMARY_ONLY** and **PRIVATE_ARCHIVE_ONLY** (none
in this release: no scientific artifact was withheld).

## Summary

| Study | Files | IDENTICAL | SANITIZED_PATHS_ONLY | Hash-pinned by a result record |
|---|---|---|---|---|
| FBR (toy v1, exploratory) | 11 | 11 | 0 | 5 |
| FBR (toy v2 gate) | 9 | 9 | 0 | 9 |
| Precursor | 33 | 33 | 0 | 16 |
| Recoverability | 5 | 4 | 1 | 5 |
| FCSI | 4 | 4 | 0 | 4 |
| Active diagnosis | 12 | 12 | 0 | 5 |
| Pre-study legacy (Phase I, Isaac Lab) | 107 | 107 | 0 | 0 |
| Pre-study legacy (Isaac Lab software smoke) | 21 | 21 | 0 | 0 |
| **Total** | 202 | 201 | 1 | 44 |

## Sanitized artifact

`results/recoverability/retro_2227de1/analysis.json`: **SANITIZED PUBLIC COPY**, not byte-identical to the frozen artifact.

| | |
|---|---|
| study | Recoverability |
| original private path | `results/recoverability/retro_2227de1/analysis.json` |
| original SHA-256 (the value pinned in `RECOVERABILITY_RETROSPECTIVE_RESULT.md`) | `cd6c72a2ce4681df6bd9962bebb7a49198ca6867151f01f5a748ac1e2d5ff55a` |
| public SHA-256 | `4e1bc129709d6afa809368346dea461502a214aab2173c476da0c57b7f192ab6` |
| transformation | every occurrence of the absolute checkout prefix (29 bytes, of the form `/home/<user>/Projects/ashfall/`) replaced by the literal `<workspace>/`; 3 occurrences, all in the input-hash keys; no other byte changed |
| SHA-256 of the replaced prefix | `e9bb3b0e3225c6fd53ee9855a3eb4655d104be6c30b08bc1d725534c5872d864` |
| reason | machine-local path metadata removed from public artifact set; the input hashes it keys on are kept |
| reproducibility | rerun at archive: identical except code_commit and the hash-seed-dependent provenance p |

Anyone holding the original prefix can restore the frozen bytes by replacing `<workspace>/` with it; the
archive check did this and obtained the original SHA-256 exactly.

## Already-public files with local paths

These files contain an absolute local path in their text. Each has been public, byte-for-byte, in the
repository history since before this release. They are left IDENTICAL: rewriting them at the tip would not
remove them from public history and would break their provenance-index hashes.

- `results/legacy_row0_curriculum/REPORT.md` (SHA-256 `641444c84b379c03...`)
- `results/legacy_row0_curriculum/multiseed_n11_scale_verdict.md` (SHA-256 `5462fce357dff108...`)
- `results/legacy_row0_curriculum/multiseed_scale_ext_2026-06-02_ANALYSIS.md` (SHA-256 `5f35ae5d2f62df70...`)

## Study artifacts

Columns: status; SHA-256 of the public file (equal to the original unless SANITIZED); the result record that
pins it, if any; reproducibility.

### FBR (toy v1, exploratory)

| Public path (= private path) | Status | SHA-256 | Pinned in | Reproducibility |
|---|---|---|---|---|
| `results/fbr_toy/corrected.log` | IDENTICAL | `07e1c845e9f1dc63589ce1c2531d1806af48a630bba5d77774467df277f0947f` |  | frozen; not rerun at archive |
| `results/fbr_toy/efficiency.log` | IDENTICAL | `528eb01c6234f3c51408d08fd321cd344d5a69fc2e240df375edd27c78704e91` |  | frozen; not rerun at archive |
| `results/fbr_toy/f339087.log` | IDENTICAL | `e442cc11f57293ec0021172212b0c1c63e4ec529087bf2c5c6d9bbe9254f0b1b` |  | frozen; not rerun at archive |
| `results/fbr_toy/f339087/baseline.npy` | IDENTICAL | `2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3` | FBR_TOY_NEGATIVE_RESULT.md | rerun at archive: byte-identical |
| `results/fbr_toy/f339087/capsule.json` | IDENTICAL | `2922adfd54dbfa1e30acc372aefa8eb12aaa6d9a9de4e272f721cf5d9e33d27a` | FBR_TOY_NEGATIVE_RESULT.md | rerun at archive: byte-identical |
| `results/fbr_toy/f339087/result.json` | IDENTICAL | `446197b78cd3d841aca65d5106dea347ec1a64ec58f608b6e293d70a2b22f42f` | FBR_TOY_NEGATIVE_RESULT.md | superseded by the f3ccc1b fix rerun (documented in the FBR record) |
| `results/fbr_toy/f339087_efficiency_exploratory/efficiency.json` | IDENTICAL | `742497cb8c54283e3028ba475d2723551731046c6de6a3df207ff5f56874897b` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy/f3ccc1b_corrected/baseline.npy` | IDENTICAL | `2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3` |  | rerun at archive: byte-identical |
| `results/fbr_toy/f3ccc1b_corrected/capsule.json` | IDENTICAL | `2922adfd54dbfa1e30acc372aefa8eb12aaa6d9a9de4e272f721cf5d9e33d27a` |  | rerun at archive: byte-identical |
| `results/fbr_toy/f3ccc1b_corrected/result.json` | IDENTICAL | `f3cb03fb3adf679e7fda474f28183775ee6f64cb078316720e8c61952d89c1c7` | FBR_TOY_NEGATIVE_RESULT.md | rerun at archive: byte-identical |
| `results/fbr_toy/run_commit.txt` | IDENTICAL | `460a81e26305184f1d5583f0a0c64fda5a822ffdcfd6d95d3b564305bbe1015f` |  | frozen; not rerun at archive |

### FBR (toy v2 gate)

| Public path (= private path) | Status | SHA-256 | Pinned in | Reproducibility |
|---|---|---|---|---|
| `results/fbr_toy_v2/8f49f02.log` | IDENTICAL | `3d0ce59d36ab7e8ded25711f01a71a950d9c7eb0a76c8d9f1a1a13e02d0cb221` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy_v2/8f49f02/baseline.npy` | IDENTICAL | `2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy_v2/8f49f02/capsule_W1.json` | IDENTICAL | `5013f3336576ea9c3db6e15ec10bb01da06027d0ddaa5be6790ef86c828f67e1` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy_v2/8f49f02/capsule_W2.json` | IDENTICAL | `15d202713094f5e546e95d89c82700a12a38a6bdebf0716da22d8cc7172d00ef` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy_v2/8f49f02/capsule_W3.json` | IDENTICAL | `efaa1cdfe4d192753c2998b698bcff6dbb1f884699139a7836b02817dec0f672` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy_v2/8f49f02/capsule_W4.json` | IDENTICAL | `191881942cb993c10fe69f402604990f46a74c80721354c159075bae96ee779d` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy_v2/8f49f02/capsule_W5.json` | IDENTICAL | `3b5c6cd49e98851d8bea0fb4770f03e5432ac95808fdf3bb1551585bd98dc016` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy_v2/8f49f02/capsule_W6.json` | IDENTICAL | `809ea58594940184733f897ac13d297ab9cafafe76c7db72a37559412af0c4ab` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |
| `results/fbr_toy_v2/8f49f02/result.json` | IDENTICAL | `990ffc19646977079bad46f6d6ad0a326c9314735935e14b720869b588b3719a` | FBR_TOY_NEGATIVE_RESULT.md | frozen; not rerun at archive |

### Precursor

| Public path (= private path) | Status | SHA-256 | Pinned in | Reproducibility |
|---|---|---|---|---|
| `results/precursor_toy/3205f02/capsule_W1.json` | IDENTICAL | `5013f3336576ea9c3db6e15ec10bb01da06027d0ddaa5be6790ef86c828f67e1` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/capsule_W2.json` | IDENTICAL | `15d202713094f5e546e95d89c82700a12a38a6bdebf0716da22d8cc7172d00ef` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/capsule_W3.json` | IDENTICAL | `efaa1cdfe4d192753c2998b698bcff6dbb1f884699139a7836b02817dec0f672` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/capsule_W4.json` | IDENTICAL | `191881942cb993c10fe69f402604990f46a74c80721354c159075bae96ee779d` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/capsule_W5.json` | IDENTICAL | `3b5c6cd49e98851d8bea0fb4770f03e5432ac95808fdf3bb1551585bd98dc016` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/capsule_W6.json` | IDENTICAL | `809ea58594940184733f897ac13d297ab9cafafe76c7db72a37559412af0c4ab` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/figures/stage1_per_world.png` | IDENTICAL | `cc4502febc9f8256de63a5170d321662498e4c750bb052c88b7db983aeeba761` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/figures/stage1_time_sweep.png` | IDENTICAL | `4e014d194ee7aadfa163bd41f7362499ca9348fadb7c5fa20e4ab1bc16799737` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/sim_capsule_W1.json` | IDENTICAL | `3ab803598ff9668994dca12da5a8d672dd5d3c4f7f63b45f1f7f203fc4dbb47a` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/sim_capsule_W2.json` | IDENTICAL | `ce09d4528a2cea7ee5b2a4c7a2656ef71abd0dcef9f2ba4231031f07d2e805c3` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/sim_capsule_W3.json` | IDENTICAL | `40074f0712d9d2fadb9f2a964428552f6e6a1f7536b7846436e58709471fcac2` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/sim_capsule_W4.json` | IDENTICAL | `d2ca421d5bdcc2dccf383bd49ccceed81325213b3ec9c9bc90a1fb7de642f9ac` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/sim_capsule_W5.json` | IDENTICAL | `235f56237cd426a5551584ef054dcaaec7e3f3098383e82a29e24191073e0e5b` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/sim_capsule_W6.json` | IDENTICAL | `0e2b49c81695b5e5adf01894defd47c0e101993b3bee500e2da276615554bb42` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage1.json` | IDENTICAL | `4af4500580271936910b352da1f09e2ebae1bbe0f62f180c696e92d760d0e63d` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage1.sha256` | IDENTICAL | `f9cdb1003606a3c5716df81485be8997e92680d37809833d0428fd94043c3a55` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/capsule_W10.json` | IDENTICAL | `1d4f690cdc12aa35435d2d00ebfe5aa870d65a284be2d7aa048867ed4aa16a71` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/capsule_W11.json` | IDENTICAL | `4559170d18ad708a0c9f4dd9f377e259b165fefd2990c3975d8dea28ced285be` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/capsule_W12.json` | IDENTICAL | `0bb6a2181ce35256cc133601c47d5cd90077fb1e3bf2cb4e34bee46692f26528` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/capsule_W7.json` | IDENTICAL | `9e5f61e4cefdd8f7e56cfc4c5ebeb2f1dbf2ba67473f1d639021a1d8fcfb6a7f` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/capsule_W8.json` | IDENTICAL | `c3a111b3cbf76b6d12b5d4fd58e800236e24206c601840a8a128f5e07c1011d9` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/capsule_W9.json` | IDENTICAL | `4ee703cf1ac5e8d24bbdc81e97c1d5a0ac6e7d5a6210d13a8ea7a79169739319` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/figures/stage2_arms.png` | IDENTICAL | `7a2b5ac8bab0d423bdd4a433abb9283e8e5d655c5ec8691b6934e5bd1a7a63bf` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/sim_capsule_W10.json` | IDENTICAL | `80135086a70ae7492aa94e086939d47c50a1f8fd30c3e328bd6a2b0c61af27d3` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/sim_capsule_W11.json` | IDENTICAL | `4c87947ef415ded55feaf0eb8e58aaae6e3d217b457116a09cc6f95a5b624241` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/sim_capsule_W12.json` | IDENTICAL | `dd89904c8bc16844f05c1e2d6dfdb347d040c2df2b61f7baa80be2e89f840c7e` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/sim_capsule_W7.json` | IDENTICAL | `6383d15a6a424da93c9d91bed812e6bcfd785dcce23bf8caefd85a68c3179472` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/sim_capsule_W8.json` | IDENTICAL | `d47153bc4db55e6d40fa1109c67442acbb33f03ec48c53a83549ad328e7eecc7` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/sim_capsule_W9.json` | IDENTICAL | `180d467ae9afb4cb44400da2455a8f112fc1ef6636f494a8e97fabd4ca92aeff` |  | frozen; not rerun at archive |
| `results/precursor_toy/3205f02/stage2/stage2.json` | IDENTICAL | `778854d2484b44aa2402893ba618970ed671bdfd10354805bef06df3f2c4724c` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02_stage1.log` | IDENTICAL | `d9dcce4018bde8aafbffe402c71691002c7d915f5e2c463c6381f689e52926bc` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/3205f02_stage2.log` | IDENTICAL | `9c49b0c0baa6389914a47cb6d4ef8241051807c1853a40e1d14be48abfefa477` | PRECURSOR_SWEEP_RESULT.md | frozen; not rerun at archive |
| `results/precursor_toy/run_commit.txt` | IDENTICAL | `68390b0cb3924cb81a5f3ebc0b3fa00a974922deb962658f72e59b7b73ec205a` |  | frozen; not rerun at archive |

### Recoverability

| Public path (= private path) | Status | SHA-256 | Pinned in | Reproducibility |
|---|---|---|---|---|
| `results/recoverability/retro_2227de1/analysis.json` | SANITIZED_PATHS_ONLY | `4e1bc129709d6afa809368346dea461502a214aab2173c476da0c57b7f192ab6` | RECOVERABILITY_RETROSPECTIVE_RESULT.md | rerun at archive: identical except code_commit and the hash-seed-dependent provenance p |
| `results/recoverability/retro_2227de1/collapse_by_provenance.png` | IDENTICAL | `207fc3d11dfb5cdab6443d793d2973637b43c562ea0840b16e25cbd2f498e8bc` | RECOVERABILITY_RETROSPECTIVE_RESULT.md | frozen; not rerun at archive |
| `results/recoverability/retro_2227de1/dataset.json` | IDENTICAL | `7566f41164adbfa59721909d78c009aff36884f4dacd784fe5088b80b4c2788a` | RECOVERABILITY_RETROSPECTIVE_RESULT.md | rerun at archive: byte-identical |
| `results/recoverability/retro_2227de1/recoverability_only.json` | IDENTICAL | `9cc8abb0d33167243df56b1db3be48250abd2c4b33c15892aef29946c65a4e8f` | RECOVERABILITY_RETROSPECTIVE_RESULT.md | rerun at archive: byte-identical |
| `results/recoverability/retro_2227de1/repair_vs_explanations.png` | IDENTICAL | `2acdd62522a376a1b12025e42b7cf1997b57dde4e680f0379c449e37ad4313a1` | RECOVERABILITY_RETROSPECTIVE_RESULT.md | frozen; not rerun at archive |

### FCSI

| Public path (= private path) | Status | SHA-256 | Pinned in | Reproducibility |
|---|---|---|---|---|
| `results/fcsi_toy/1b657c1.log` | IDENTICAL | `902a555347ae55df259a0cd95f257cdb44b5c2db2bfb2d0ba3d4e60d2ba2ff3c` | FCSI_TOY_RESULT.md | frozen; not rerun at archive |
| `results/fcsi_toy/1b657c1/result.json` | IDENTICAL | `8b4c8f9fe2b02a8d23279b7cd9c939272782a8a0b40f01e3f0082f7e5dd532db` | FCSI_TOY_RESULT.md | frozen; not rerun at archive |
| `results/fcsi_toy/run_commit.txt` | IDENTICAL | `57a0993d50e109dfbdb45c4a8223dddd5f1a2e514818366a2e33eb7d692657fe` | FCSI_TOY_RESULT.md | frozen; not rerun at archive |
| `results/fcsi_toy/run_notes.txt` | IDENTICAL | `61675d13ab6c85d3beb9e014e77cae10e4d6a50c3bb828b9b64c32a2602cc0ed` | FCSI_TOY_RESULT.md | frozen; not rerun at archive |

### Active diagnosis

| Public path (= private path) | Status | SHA-256 | Pinned in | Reproducibility |
|---|---|---|---|---|
| `results/active_diag/2b3e40e/calibration.json` | IDENTICAL | `efca9e4ceadde1325c2e713d67980ed96adbbc8cd656d680b645aa55804ffe72` | ACTIVE_DIAGNOSIS_TOY_RESULT.md | rerun at archive: identical except wall-clock fields |
| `results/active_diag/2b3e40e/dev_runs.json` | IDENTICAL | `99ee054e0993e6ee778270f7d05e0da325e3f83fbf189c7a3ba3cf6686cf2c18` | ACTIVE_DIAGNOSIS_TOY_RESULT.md | rerun at archive: identical except wall-clock fields |
| `results/active_diag/2b3e40e/eval_result.json` | IDENTICAL | `0726bdf5d0a89fa1e9da14927321e20aa8a16ccddc4fe033f918b4054582cd08` | ACTIVE_DIAGNOSIS_TOY_RESULT.md | rerun at archive: identical except wall-clock fields |
| `results/active_diag/2b3e40e_dev.log` | IDENTICAL | `c9932d98905b279d5f1a308e718cf6eed729723237947fc72c15e4999a501850` |  | frozen; not rerun at archive |
| `results/active_diag/2b3e40e_eval.log` | IDENTICAL | `db83fc04b87d023b2fdc9dd8a710dd905f144a7ad2cf67eff502453308a57d4a` | ACTIVE_DIAGNOSIS_TOY_RESULT.md | frozen; not rerun at archive |
| `results/active_diag/513be90/calibration.json` | IDENTICAL | `7791c6d1ca98256d47c09918d9408c8ef85a04042acb4c404dacda32be6ee9ac` |  | frozen; not rerun at archive |
| `results/active_diag/513be90/dev_runs.json` | IDENTICAL | `cba9df25a61b977dd2fec44819ceea62bd02ae68a41fbd83d02230fa954f5902` |  | frozen; not rerun at archive |
| `results/active_diag/513be90_dev.log` | IDENTICAL | `e4f2587b01419dc64b62b6ffc09e26f298d60fc0b692868e788e9d0921a8aecb` |  | frozen; not rerun at archive |
| `results/active_diag/ad32ac0/calibration.json` | IDENTICAL | `c10ff04464e04b313d466ce4df95d342f929f9e8532d5875c9ecd8603f7f6219` |  | frozen; not rerun at archive |
| `results/active_diag/ad32ac0/dev_runs.json` | IDENTICAL | `4039714e4eae740060b31ef46da02a08114989277de22e9c10b2424db90c3421` |  | frozen; not rerun at archive |
| `results/active_diag/ad32ac0_dev.log` | IDENTICAL | `585c8ec19da01991b0240600434ac847dafe78124921ce43d3a9201cdec3e77c` |  | frozen; not rerun at archive |
| `results/active_diag/eval_notes.txt` | IDENTICAL | `9f24a2fdb65484d671865ff4408e2222b6a766f68695386250269b97997cff5c` | ACTIVE_DIAGNOSIS_TOY_RESULT.md | frozen; not rerun at archive |

## Pre-study legacy artifacts

`results/legacy_row0_curriculum/` (107 files) and `results/software_validation/` (21 files) predate the five
studies. All 128 are IDENTICAL to the private archive and were already public before this release; their
per-file hashes are in the JSON manifest and in `results/legacy_row0_curriculum/provenance_index.json`.
