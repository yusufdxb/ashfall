"""Fail-closed reader for Phoenix per-episode evaluation records.

Phoenix writes one parquet row per finished evaluation episode
(``phoenix.training.episode_records``, schema version 1.0.0). Ashfall reads
those rows to build the per-failure-mode recurrence table that the Phase-II
primary endpoint needs, which the aggregate ``metrics_*.json`` cannot supply.

Ashfall does not import Phoenix. The contract between the two repos is the
parquet schema, so the schema constants are restated here and validated on
every read. That is deliberate: if Phoenix bumps its schema, an Ashfall read
raises instead of quietly analysing columns that no longer mean what this
module thinks they mean.

Fail-closed rules:

* A ``schema_version`` other than :data:`SCHEMA_VERSION`, in the per-row
  column or in the Arrow schema metadata, raises.
* A missing required column raises.
* A missing file, an empty units map, or a null version raises.

Nothing is defaulted, inferred, or skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import pyarrow.parquet as pq

#: The single Phoenix schema version this reader understands.
SCHEMA_VERSION = "1.0.0"

SCHEMA_VERSION_KEY = b"phoenix_episode_records_schema_version"
UNITS_KEY = b"phoenix_episode_records_units"

#: Value Phoenix writes into ``time_to_failure_*`` for a successful episode.
NO_FAILURE = -1

REQUIRED_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "episode_id",
    "seed",
    "env_index",
    "terrain_id",
    "challenge_id",
    "success",
    "termination_reason",
    "time_to_failure_steps",
    "time_to_failure_s",
    "episode_return",
    "episode_length_steps",
    "episode_length_s",
    "mean_lin_vel_error_mps",
    "max_lin_vel_error_mps",
    "mean_ang_vel_error_radps",
    "max_ang_vel_error_radps",
    "control_dt_s",
    "policy_path",
    "policy_sha256",
)


class EpisodeRecordSchemaError(ValueError):
    """Raised when an episode-record artifact fails validation."""


@dataclass
class EpisodeRecord:
    """One finished Phoenix evaluation episode, as read by Ashfall."""

    run_id: str
    episode_id: int
    seed: int
    env_index: int
    terrain_id: str
    challenge_id: str
    success: bool
    termination_reason: str
    time_to_failure_steps: int
    time_to_failure_s: float
    episode_return: float
    episode_length_steps: int
    episode_length_s: float
    mean_lin_vel_error_mps: float
    max_lin_vel_error_mps: float
    mean_ang_vel_error_radps: float
    max_ang_vel_error_radps: float
    control_dt_s: float
    policy_path: str
    policy_sha256: str
    schema_version: str = SCHEMA_VERSION

    @property
    def failed(self) -> bool:
        return not self.success


def validate_table(table, *, source: str = "<table>") -> None:
    """Fail-closed validation of an episode-record Arrow table."""
    missing = [name for name in REQUIRED_COLUMNS if name not in table.schema.names]
    if missing:
        raise EpisodeRecordSchemaError(
            f"{source}: missing required column(s): {', '.join(sorted(missing))}"
        )

    versions: set[str] = set()
    meta = table.schema.metadata or {}
    if SCHEMA_VERSION_KEY in meta:
        versions.add(meta[SCHEMA_VERSION_KEY].decode())
    for value in table.column("schema_version").to_pylist():
        if value is None:
            raise EpisodeRecordSchemaError(f"{source}: null schema_version in a row")
        versions.add(str(value))

    if not versions:
        raise EpisodeRecordSchemaError(f"{source}: no schema version found")
    unexpected = sorted(v for v in versions if v != SCHEMA_VERSION)
    if unexpected:
        raise EpisodeRecordSchemaError(
            f"{source}: schema version mismatch, Ashfall expects {SCHEMA_VERSION!r} "
            f"but artifact declares {unexpected!r}"
        )


def table_units(table, *, source: str = "<table>") -> dict[str, str]:
    """Return the units map stamped into the artifact, raising if absent."""
    meta = table.schema.metadata or {}
    if UNITS_KEY not in meta:
        raise EpisodeRecordSchemaError(f"{source}: units map missing from schema metadata")
    units = json.loads(meta[UNITS_KEY].decode())
    if not units:
        raise EpisodeRecordSchemaError(f"{source}: units map is empty")
    return units


def load_episode_records(path: str | Path) -> list[EpisodeRecord]:
    """Load and validate one episode-record parquet."""
    src = Path(path)
    if not src.exists():
        raise EpisodeRecordSchemaError(f"{src}: episode-record artifact does not exist")
    table = pq.read_table(src)
    validate_table(table, source=str(src))
    table_units(table, source=str(src))
    known = {f.name for f in fields(EpisodeRecord)}
    return [
        EpisodeRecord(**{k: v for k, v in row.items() if k in known})
        for row in table.to_pylist()
    ]


def load_run_records(paths) -> list[EpisodeRecord]:
    """Load and concatenate episode records for a set of runs.

    ``paths`` may be individual parquet files or directories, in which case
    every ``*.parquet`` directly inside the directory is read. A directory
    that contains no parquet raises rather than contributing zero rows, so a
    mistyped results path cannot masquerade as an arm with no episodes.
    """
    out: list[EpisodeRecord] = []
    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            found = sorted(p.glob("*.parquet"))
            if not found:
                raise EpisodeRecordSchemaError(f"{p}: directory contains no *.parquet")
            for fp in found:
                out.extend(load_episode_records(fp))
        else:
            out.extend(load_episode_records(p))
    return out


__all__ = [
    "NO_FAILURE",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_KEY",
    "UNITS_KEY",
    "EpisodeRecord",
    "EpisodeRecordSchemaError",
    "load_episode_records",
    "load_run_records",
    "table_units",
    "validate_table",
]
