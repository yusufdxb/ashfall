"""Fail-closed reader tests for the Phoenix per-episode record artifact."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ashfall.evaluation.episode_records import (
    NO_FAILURE,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    UNITS_KEY,
    EpisodeRecord,
    EpisodeRecordSchemaError,
    load_episode_records,
    load_run_records,
    table_units,
    validate_table,
)

UNITS = {
    "schema_version": "semver",
    "run_id": "identifier",
    "episode_id": "count",
    "seed": "identifier",
    "env_index": "index",
    "terrain_id": "identifier",
    "challenge_id": "identifier",
    "success": "bool",
    "termination_reason": "enum",
    "time_to_failure_steps": "env_steps",
    "time_to_failure_s": "seconds",
    "episode_return": "reward",
    "episode_length_steps": "env_steps",
    "episode_length_s": "seconds",
    "mean_lin_vel_error_mps": "m/s",
    "max_lin_vel_error_mps": "m/s",
    "mean_ang_vel_error_radps": "rad/s",
    "max_ang_vel_error_radps": "rad/s",
    "control_dt_s": "seconds",
    "policy_path": "filesystem_path",
    "policy_sha256": "hex_digest",
}

ARROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("run_id", pa.string()),
        ("episode_id", pa.int64()),
        ("seed", pa.int64()),
        ("env_index", pa.int64()),
        ("terrain_id", pa.string()),
        ("challenge_id", pa.string()),
        ("success", pa.bool_()),
        ("termination_reason", pa.string()),
        ("time_to_failure_steps", pa.int64()),
        ("time_to_failure_s", pa.float64()),
        ("episode_return", pa.float64()),
        ("episode_length_steps", pa.int64()),
        ("episode_length_s", pa.float64()),
        ("mean_lin_vel_error_mps", pa.float64()),
        ("max_lin_vel_error_mps", pa.float64()),
        ("mean_ang_vel_error_radps", pa.float64()),
        ("max_ang_vel_error_radps", pa.float64()),
        ("control_dt_s", pa.float64()),
        ("policy_path", pa.string()),
        ("policy_sha256", pa.string()),
    ],
    metadata={
        SCHEMA_VERSION_KEY: SCHEMA_VERSION.encode(),
        UNITS_KEY: json.dumps(UNITS, sort_keys=True).encode(),
    },
)


def make_row(**overrides) -> dict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-a",
        "episode_id": 0,
        "seed": 11,
        "env_index": 2,
        "terrain_id": "rough",
        "challenge_id": "ff0p0",
        "success": False,
        "termination_reason": "termination",
        "time_to_failure_steps": 90,
        "time_to_failure_s": 1.8,
        "episode_return": 33.0,
        "episode_length_steps": 90,
        "episode_length_s": 1.8,
        "mean_lin_vel_error_mps": 0.3,
        "max_lin_vel_error_mps": 0.9,
        "mean_ang_vel_error_radps": 0.15,
        "max_ang_vel_error_radps": 0.5,
        "control_dt_s": 0.02,
        "policy_path": "/ckpt/model_799.pt",
        "policy_sha256": "ef" * 32,
    }
    row.update(overrides)
    return row


def write_rows(path, rows, schema=ARROW_SCHEMA):
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


class TestRoundTrip:
    def test_schema_round_trip(self, tmp_path):
        rows = [make_row(episode_id=i, env_index=i) for i in range(3)]
        path = write_rows(tmp_path / "e.parquet", rows)
        loaded = load_episode_records(path)
        assert len(loaded) == 3
        assert all(isinstance(r, EpisodeRecord) for r in loaded)
        assert [r.episode_id for r in loaded] == [0, 1, 2]
        assert loaded[0].policy_sha256 == "ef" * 32
        assert loaded[0].failed is True

    def test_seed_propagates(self, tmp_path):
        rows = [make_row(episode_id=i, seed=100 + i) for i in range(3)]
        path = write_rows(tmp_path / "e.parquet", rows)
        assert [r.seed for r in load_episode_records(path)] == [100, 101, 102]

    def test_units_preserved(self, tmp_path):
        path = write_rows(tmp_path / "e.parquet", [make_row()])
        units = table_units(pq.read_table(path))
        assert units == UNITS
        assert units["mean_lin_vel_error_mps"] == "m/s"
        assert units["time_to_failure_s"] == "seconds"
        assert set(units) == set(REQUIRED_COLUMNS)

    def test_no_failure_sentinel_readable(self, tmp_path):
        path = write_rows(
            tmp_path / "e.parquet",
            [
                make_row(
                    success=True,
                    termination_reason="time_out",
                    time_to_failure_steps=NO_FAILURE,
                    time_to_failure_s=float(NO_FAILURE),
                )
            ],
        )
        rec = load_episode_records(path)[0]
        assert rec.success is True
        assert rec.failed is False
        assert rec.time_to_failure_steps == NO_FAILURE

    def test_load_run_records_concatenates(self, tmp_path):
        write_rows(tmp_path / "a.parquet", [make_row(run_id="a", seed=1)])
        write_rows(tmp_path / "b.parquet", [make_row(run_id="b", seed=2)])
        loaded = load_run_records([tmp_path])
        assert sorted(r.seed for r in loaded) == [1, 2]

    def test_load_run_records_accepts_files(self, tmp_path):
        p1 = write_rows(tmp_path / "a.parquet", [make_row(seed=1)])
        p2 = write_rows(tmp_path / "b.parquet", [make_row(seed=2)])
        assert len(load_run_records([p1, p2])) == 2


class TestFailClosed:
    def test_version_mismatch_raises(self, tmp_path):
        path = write_rows(tmp_path / "old.parquet", [make_row(schema_version="0.9.0")])
        with pytest.raises(EpisodeRecordSchemaError, match="schema version mismatch"):
            load_episode_records(path)

    def test_metadata_version_mismatch_raises(self, tmp_path):
        schema = ARROW_SCHEMA.with_metadata(
            {
                SCHEMA_VERSION_KEY: b"2.0.0",
                UNITS_KEY: json.dumps(UNITS, sort_keys=True).encode(),
            }
        )
        path = write_rows(tmp_path / "meta.parquet", [make_row()], schema=schema)
        with pytest.raises(EpisodeRecordSchemaError, match="schema version mismatch"):
            load_episode_records(path)

    def test_missing_column_raises(self, tmp_path):
        table = pa.Table.from_pylist([make_row()], schema=ARROW_SCHEMA).drop(["seed"])
        path = tmp_path / "noseed.parquet"
        pq.write_table(table, path)
        with pytest.raises(EpisodeRecordSchemaError, match="missing required column"):
            load_episode_records(path)

    def test_missing_units_raises(self, tmp_path):
        schema = ARROW_SCHEMA.with_metadata({SCHEMA_VERSION_KEY: SCHEMA_VERSION.encode()})
        path = write_rows(tmp_path / "nounits.parquet", [make_row()], schema=schema)
        with pytest.raises(EpisodeRecordSchemaError, match="units map missing"):
            load_episode_records(path)

    def test_null_version_raises(self):
        table = pa.Table.from_pylist([make_row(schema_version=None)], schema=ARROW_SCHEMA)
        with pytest.raises(EpisodeRecordSchemaError, match="null schema_version"):
            validate_table(table)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(EpisodeRecordSchemaError, match="does not exist"):
            load_episode_records(tmp_path / "absent.parquet")

    def test_empty_directory_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(EpisodeRecordSchemaError, match="no \\*.parquet"):
            load_run_records([empty])

    def test_valid_table_passes(self):
        validate_table(pa.Table.from_pylist([make_row()], schema=ARROW_SCHEMA))
