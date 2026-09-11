"""Schema and reset-window contracts independent of simulator availability."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ashfall.capsule import CapsuleFrame, FailureCapsule, capsules_from_parquet, capsules_from_rows
from ashfall.synth.generator import SCHEMA, generate_stumble_failure


def capsule():
    rows = generate_stumble_failure(n_stable=50, n_failure=20)
    return capsules_from_rows(
        rows, source="synthetic_fixture", robot="go2", control_dt=0.02, pre_failure_seconds=0.5
    )[0]


def test_original_row_zero_bug():
    cap = capsule()
    assert cap.failure_onset_index == 50
    assert cap.resolve_seed_index("failure_onset_minus_steps", offset_steps=10) == 40
    assert cap.resolve_seed_index(offset_seconds=0.2) == 40
    assert cap.resolve_seed_index() == 25
    assert cap.resolve_seed_index("first") == 0
    assert cap.resolve_seed_index("failure_onset") == 50
    assert len(cap.pre_failure_frames) == 25
    assert cap.pre_failure_frames[-1] == cap.frames[49]


def test_no_silent_fallback_to_row_zero():
    cap = capsule()
    with pytest.raises(ValueError, match="outside reviewed"):
        cap.resolve_seed_index(offset_seconds=2)
    with pytest.raises(ValueError, match="outside reviewed"):
        cap.resolve_seed_index("failure_onset_minus_steps", offset_steps=51)
    with pytest.raises(ValueError):
        cap.resolve_seed_index("bogus")


def test_roundtrip_and_identity(tmp_path):
    cap = capsule()
    saved = cap.save(tmp_path / "capsule.json")
    assert FailureCapsule.load(saved) == cap
    assert json.loads(saved.read_text())["environment_metadata"] is None
    assert cap == capsule()
    data = cap.to_dict()
    data["failure_onset_index"] = 51
    with pytest.raises(ValueError, match="capsule_id"):
        FailureCapsule.from_dict(data)


def test_version_and_unknown_field_rejection():
    data = capsule().to_dict()
    data["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="unsupported"):
        FailureCapsule.from_dict(data)
    data = capsule().to_dict()
    data["misspelled_field"] = 1
    with pytest.raises(TypeError):
        FailureCapsule.from_dict(data)


@pytest.mark.parametrize(
    "change",
    [
        {"control_dt": 0},
        {"control_dt": float("nan")},
        {"failure_onset_index": 70},
        {"pre_failure_start_index": -1},
        {"post_failure_end_index": 49},
        {"timestamp": "2026-09-04T00:00:00"},
        {"checkpoint_sha256": "invalid"},
    ],
)
def test_invalid_capsule_rejected(change):
    with pytest.raises(ValueError):
        replace(capsule(), capsule_id="", **change)


def test_frame_schema():
    with pytest.raises(ValueError, match="unit"):
        CapsuleFrame(0.0, base_quat=(0, 0, 0, 2))
    with pytest.raises(ValueError, match="non-finite"):
        CapsuleFrame(0.0, base_pos=(0, 0, float("inf")))
    with pytest.raises(ValueError, match="dimensions"):
        CapsuleFrame(0.0, joint_pos=(0, 0), joint_vel=(0,))
    with pytest.raises(ValueError):
        CapsuleFrame(0.0, command_vel=(1, 2))


def test_missing_physical_context_stays_unknown_and_blocks_reset():
    cap = capsule()
    frames = list(cap.frames)
    frames[25] = CapsuleFrame(timestamp_s=0.5)
    partial = replace(cap, frames=tuple(frames), capsule_id="")
    assert partial.environment_metadata is None
    assert partial.frames[25].command_vel is None
    with pytest.raises(ValueError, match="missing state"):
        partial.reset_frame()
    assert cap.reset_frame().command_vel == cap.frames[25].command_vel


def test_multiple_events_and_parquet_migration(tmp_path):
    rows = generate_stumble_failure(n_stable=50, n_failure=20)
    for row in rows[55:60]:
        row["failure_flag"] = False
        row["failure_mode"] = None
    path = tmp_path / "recording.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    caps = capsules_from_parquet(path, source="recorded", robot="go2")
    assert [c.failure_onset_index for c in caps] == [50, 60]
    assert caps[0].trajectory_sha256 == caps[1].trajectory_sha256
    assert caps[0].capsule_id != caps[1].capsule_id
    assert caps[0].event_descriptor["onset_label_source"] == "legacy_failure_flag"
    assert caps[0].policy_id is None


def test_explicit_review_onset_and_no_failure():
    rows = generate_stumble_failure(n_stable=50, n_failure=20)
    for row in rows:
        row["failure_flag"] = False
        row["failure_mode"] = None
    assert capsules_from_rows(rows, source="recorded", robot="go2") == []
    caps = capsules_from_rows(rows, source="recorded", robot="go2", onset_indices=[50])
    assert caps[0].failure_onset_index == 50
    assert caps[0].failure_mode == "unclassified"
    assert caps[0].event_descriptor["onset_label_source"] == "explicit_review"


def test_timestamp_selection_handles_missing_samples():
    cap = capsule()
    frames = tuple(replace(f, timestamp_s=f.timestamp_s * 2) for f in cap.frames)
    cap = replace(cap, frames=frames, capsule_id="")
    assert cap.resolve_seed_index(offset_seconds=0.2) == 45


def test_metadata_mutation_cannot_be_saved(tmp_path):
    cap = replace(capsule(), environment_metadata={"friction": None}, capsule_id="")
    cap.environment_metadata["friction"] = 0.5
    with pytest.raises(ValueError, match="changed"):
        cap.save(tmp_path / "invalid.json")
