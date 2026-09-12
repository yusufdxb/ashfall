# Trajectory data

Two trees, kept apart by manifest (`ashfall.datasets`), not by convention.

| tree | kind | may support a scientific claim |
|---|---|---|
| `data/fixtures/` | hand-authored or generated trajectories used to exercise schema, wiring and regression behaviour | **no** |
| `data/scientific/` | trajectories produced by physics (simulator rollouts) or by the robot, each set with a `dataset.json` naming policy, environment hash, simulator version, seeds, intervention and file hashes | yes, within the provenance the manifest states |

Parquet bytes are not tracked in git; manifests are. `DatasetManifest.load` refuses a directory
with no manifest, and `assert_scientific` refuses a fixture manifest wherever a detector accuracy,
delivery or learning claim would be made from the data.

`data/fixtures/synthetic/` is written by `python3 -m ashfall.synth.generator`. Its 18 files are
the historical detector fixtures: threshold regressions, not validation. Their known physical
defects are listed in the generator's module docstring and are deliberately not corrected.

`data/scientific/` is written by the harvest pipeline (`python3 -m ashfall.cli harvest`), one
directory per harvest, and by hardware capture review. The Phoenix-side raw harvests live in the
sibling repository under `data/failures/sim_harvest/`; Ashfall ingests them, never edits them.
