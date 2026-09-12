# Data and provenance

## Two kinds of data

| tree | manifest kind | source | may support a claim |
|---|---|---|---|
| `data/fixtures/` | `fixture` | `synthetic_generator` | no |
| `data/scientific/` | `scientific` | `simulation` or `hardware` | yes, within its provenance |

Every dataset directory carries a `dataset.json` written by
`ashfall.datasets.DatasetManifest`. A directory without one is unusable
(`DatasetManifest.load` raises). A `scientific` manifest must name
`policy_id`, `env_config_hash`, `simulator_version`, `seeds`, `ashfall_sha`
and `phoenix_sha`; a `fixture` manifest cannot be constructed as scientific,
and `assert_scientific(manifest, purpose=...)` is the call every scientific
consumer makes. Parquet bytes are not tracked in git; manifests are.

The 18 hand-authored parquets under `data/fixtures/synthetic/` are the
historical detector fixtures. Their known physical defects (open-loop
position under a zeroed body velocity in two generators, a fixed stable prefix
with no physical meaning, a `failure_flag` that is the generator's recipe
rather than a detector decision) are deliberately kept: they are regression
material, not data.

## Harvests

`ashfall h0` writes a `harvest/` directory inside its evidence bundle: both
arms of every matched pair in the Phoenix trajectory schema
(`pair_NNN_{control,treatment}.parquet`), `attempts.json`, `episodes.json`,
the spec and result, one schema-1.1 capsule per DELIVERED pair under
`capsules/`, and a `dataset.json`. The manifest's kind follows the evidence:
`scientific` for the Phoenix backend, `fixture` for the toy surrogate.

A capsule at schema 1.1 (`ashfall.capsule.FailureCapsule`) carries, beside
the frames and the reviewed window indices: `intervention` (the cause that was
verifiably applied), `phenotype_label` (what the detector observed),
`onset_window` (precursor from the D2 departure profile, transition,
established), and `provenance` (policy, environment hash, simulator version,
seed, backend, the control and treatment episode ids and the attempt id). A
capsule refuses an intervention kind in the phenotype slot and vice versa.
Schema 1.0 capsules still load; they carry none of these fields.

## Evidence bundles

```
<bundle>/
  manifest.json      specification and the run id (hash of the specification)
  environment.json   interpreter, package set and its hash, simulator packages, CUDA driver, GPU
  provenance.json    Ashfall and Phoenix git identity (sha, branch, dirty flag, patch hash,
                     untracked-file hashes), config hashes, dataset hashes, policy hashes,
                     named seeds, simulator version, packages hash
  metrics.json       what was measured
  verdict.json       what was concluded, with the hashes of the files it rests on
  index.json         sha256 of every file above, written by finalize
  logs/
```

Files are write-once: an identical rewrite is allowed, a changed one raises.
`finalize` refuses a bundle whose provenance lacks any required identity.
`environment.json` is captured without importing torch (nvidia-smi supplies
the driver and GPU fields), so capturing it cannot initialise CUDA.

## Identities

Everything scientific is content-addressed through canonical JSON and
SHA-256 (`ashfall.provenance.content_hash`): interventions (`int_`),
restorable states (`st_`), episodes (`ep_`), attempts (`att_`), H0 specs
(`h0_`), datasets (`ds_`), capsules (`cap_`), experiment manifests and
protocols. A timestamp is never part of an identity. Policies are identified
by the SHA-256 of the checkpoint bytes, never by a filename.

## Verifying a bundle or dataset

```bash
python3 - <<'PY'
from ashfall.datasets import DatasetManifest
m = DatasetManifest.load("results/h0/<run>/harvest"); m.verify("results/h0/<run>/harvest")
print(m.kind, m.dataset_id)
PY
```

`index.json` lists the hash of every bundle file; recompute with
`ashfall.provenance.file_hash` to check nothing was edited after `finalize`.

## The Phoenix pin

`ashfall.backends.phoenix_compat` records the go2-phoenix branch and commit
Ashfall was integrated against and checks the live sibling's interface
(kinematic restore fields, `restore_state` keywords, the friction adapter's
supported set, `EpisodeOutcome` fields) before any simulator run. The
revision actually used is written into every bundle's `provenance.json`; the
pin is a record and an interface check, not a lock on the commit.

## Historical evidence

`results/legacy_row0_curriculum/**` is byte-preserved with its own SHA-256
index and is never modified or regenerated. Its interpretation is in
[`docs/legacy/README.md`](legacy/README.md) and the claims ledger.
