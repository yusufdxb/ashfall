"""Paired per-seed, per-mode failure recurrence from Phoenix episode records.

The Phase-II primary endpoint asks whether a given failure mode recurs less
often after adaptation. Phoenix's aggregate ``metrics_*.json`` cannot answer
it: it reports a single success rate per cell with the per-episode identity
already averaged away. This module consumes the per-episode record artifact
instead, attaches Ashfall's six-mode detector to every failed episode, and
produces a table paired by seed so that a per-seed sign-flip test can be run
on the deltas.

Unobservable signals are recorded, never guessed. An audit of the real GO2
hardware path found that ``base_pos`` and ``contact_forces`` are hardcoded
zeros there, which makes collapse, stumble, contact loss and (through the
missing height) part of the attitude picture undetectable on hardware. Four
of the six modes are therefore unobservable on real GO2 telemetry as it
stands. Rather than let those episodes silently fall into whichever mode the
remaining signals happen to trip, this module labels them
:data:`MODE_UNKNOWN`, and the recurrence table carries UNKNOWN as a first
class row so the size of the blind spot is visible in every result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ashfall.evaluation.episode_records import EpisodeRecord
from ashfall.taxonomy.detector import SEVERITY, FailureDetector, FailureMode, FailureThresholds

#: Label for a failed episode whose mode could not be determined because a
#: required signal was unavailable or the detector fired on nothing.
MODE_UNKNOWN = "unknown"

#: Signals the six-mode detector cannot run without.
REQUIRED_SIGNALS: tuple[str, ...] = (
    "pitch_rad",
    "roll_rad",
    "base_height_m",
    "cmd_lin_vel",
    "actual_lin_vel",
)

#: Row order of the recurrence table: the six modes plus the explicit
#: unknown bucket. Always emitted, including as zeros, so a mode that never
#: fires is distinguishable from a mode that was never looked for.
TABLE_MODES: tuple[str, ...] = tuple(m.value for m in FailureMode) + (MODE_UNKNOWN,)


class RecurrenceError(ValueError):
    """Raised when a recurrence table cannot be built from the given records."""


@dataclass
class EpisodeTelemetry:
    """Per-step signals for one episode, as fed to the six-mode detector.

    ``joint_vel`` and ``contact_forces`` are optional. Their absence disables
    stumble and contact-loss detection but does not by itself make the
    episode UNKNOWN, since the remaining four modes stay detectable.
    """

    dt_s: float
    pitch_rad: Optional[np.ndarray] = None
    roll_rad: Optional[np.ndarray] = None
    base_height_m: Optional[np.ndarray] = None
    cmd_lin_vel: Optional[np.ndarray] = None
    actual_lin_vel: Optional[np.ndarray] = None
    joint_vel: Optional[np.ndarray] = None
    contact_forces: Optional[np.ndarray] = None


@dataclass
class LabeledEpisode:
    """An episode record with its detected failure mode attached."""

    record: EpisodeRecord
    #: ``None`` for a successful episode, a :class:`FailureMode` value string
    #: for a diagnosed failure, :data:`MODE_UNKNOWN` otherwise.
    mode: Optional[str]
    #: Why the mode is UNKNOWN, empty when a mode was detected or the
    #: episode succeeded. Kept so a downstream reader can tell "hardware
    #: zero-filled signal" apart from "detector saw nothing".
    unknown_reason: str = ""

    @property
    def seed(self) -> int:
        return self.record.seed


@dataclass
class RecurrenceRow:
    """One (seed, mode) cell of the paired recurrence table."""

    seed: int
    mode: str
    baseline_episodes: int
    baseline_failures: int
    baseline_rate: float
    treatment_episodes: int
    treatment_failures: int
    treatment_rate: float
    delta_rate: float

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "mode": self.mode,
            "baseline_episodes": self.baseline_episodes,
            "baseline_failures": self.baseline_failures,
            "baseline_rate": self.baseline_rate,
            "treatment_episodes": self.treatment_episodes,
            "treatment_failures": self.treatment_failures,
            "treatment_rate": self.treatment_rate,
            "delta_rate": self.delta_rate,
        }


def _is_zero_filled(arr: Optional[np.ndarray]) -> bool:
    """True when a signal is present but identically zero.

    The GO2 hardware path fills ``base_pos`` and ``contact_forces`` with
    zeros, so a physically impossible all-zero base height is evidence that
    the signal was never measured, not evidence of a collapse.
    """
    if arr is None:
        return False
    a = np.asarray(arr, dtype=np.float64)
    return a.size > 0 and bool(np.all(a == 0.0))


def missing_signals(telemetry: Optional[EpisodeTelemetry]) -> list[str]:
    """Return the required signals that are absent or hardware zero-filled."""
    if telemetry is None:
        return list(REQUIRED_SIGNALS)
    missing: list[str] = []
    for name in REQUIRED_SIGNALS:
        value = getattr(telemetry, name, None)
        if value is None or np.asarray(value).size == 0:
            missing.append(name)
        elif name == "base_height_m" and _is_zero_filled(value):
            missing.append(f"{name} (hardware zero-filled)")
    return missing


def detect_episode_mode(
    telemetry: Optional[EpisodeTelemetry],
    thresholds: Optional[FailureThresholds] = None,
) -> tuple[Optional[str], str]:
    """Run the six-mode detector over one episode.

    Returns ``(mode, unknown_reason)``. ``mode`` is :data:`MODE_UNKNOWN` when
    a required signal is unavailable or when no detector fired; in that case
    ``unknown_reason`` says which. Never guesses a mode.
    """
    missing = missing_signals(telemetry)
    if missing:
        return MODE_UNKNOWN, "missing signals: " + ", ".join(missing)

    assert telemetry is not None  # narrowed by missing_signals
    pitch = np.asarray(telemetry.pitch_rad, dtype=np.float64).ravel()
    roll = np.asarray(telemetry.roll_rad, dtype=np.float64).ravel()
    height = np.asarray(telemetry.base_height_m, dtype=np.float64).ravel()
    cmd = np.atleast_2d(np.asarray(telemetry.cmd_lin_vel, dtype=np.float64))
    act = np.atleast_2d(np.asarray(telemetry.actual_lin_vel, dtype=np.float64))
    n_steps = min(len(pitch), len(roll), len(height), cmd.shape[0], act.shape[0])
    if n_steps == 0:
        return MODE_UNKNOWN, "telemetry has zero steps"

    joint_vel = (
        np.atleast_2d(np.asarray(telemetry.joint_vel, dtype=np.float64))
        if telemetry.joint_vel is not None
        else None
    )
    contact = (
        np.atleast_2d(np.asarray(telemetry.contact_forces, dtype=np.float64))
        if telemetry.contact_forces is not None and not _is_zero_filled(telemetry.contact_forces)
        else None
    )

    detector = FailureDetector(thresholds)
    dt = float(telemetry.dt_s)
    best_mode: Optional[FailureMode] = None
    best_severity = -1
    best_time = float("inf")
    for i in range(n_steps):
        events = detector.step(
            timestamp_s=i * dt,
            pitch_rad=float(pitch[i]),
            roll_rad=float(roll[i]),
            base_height_m=float(height[i]),
            cmd_lin_vel=cmd[min(i, cmd.shape[0] - 1)],
            actual_lin_vel=act[min(i, act.shape[0] - 1)],
            joint_vel=None if joint_vel is None else joint_vel[min(i, joint_vel.shape[0] - 1)],
            contact_forces=None if contact is None else contact[min(i, contact.shape[0] - 1)],
        )
        for ev in events:
            severity = SEVERITY.get(ev.mode, ev.severity)
            # Most severe wins; on a tie the earlier event wins, so the
            # label names the failure that started the episode's collapse.
            if severity > best_severity or (
                severity == best_severity and ev.timestamp_s < best_time
            ):
                best_mode = ev.mode
                best_severity = severity
                best_time = ev.timestamp_s

    if best_mode is None:
        return MODE_UNKNOWN, "detector fired no event on available signals"
    return best_mode.value, ""


def episode_key(record: EpisodeRecord) -> tuple[str, int, int]:
    """Stable identity of an episode across artifacts."""
    return (record.run_id, record.seed, record.episode_id)


TelemetrySource = Union[
    Mapping[tuple[str, int, int], EpisodeTelemetry],
    Callable[[EpisodeRecord], Optional[EpisodeTelemetry]],
    None,
]


def attach_modes(
    records: Iterable[EpisodeRecord],
    telemetry: TelemetrySource = None,
    thresholds: Optional[FailureThresholds] = None,
) -> list[LabeledEpisode]:
    """Attach a detected failure mode to every failed episode.

    ``telemetry`` may be a mapping keyed by :func:`episode_key` or a callable
    taking a record. Passing ``None`` is legal and means no telemetry is
    available at all, in which case every failed episode is UNKNOWN, which is
    exactly the state of the real GO2 hardware path today.
    """
    if telemetry is None:

        def lookup(_record: EpisodeRecord) -> Optional[EpisodeTelemetry]:
            return None

    elif isinstance(telemetry, Mapping):

        def lookup(record: EpisodeRecord) -> Optional[EpisodeTelemetry]:
            return telemetry.get(episode_key(record))

    else:
        lookup = telemetry

    out: list[LabeledEpisode] = []
    for record in records:
        if record.success:
            out.append(LabeledEpisode(record=record, mode=None))
            continue
        mode, reason = detect_episode_mode(lookup(record), thresholds)
        out.append(LabeledEpisode(record=record, mode=mode, unknown_reason=reason))
    return out


def _by_seed(episodes: Iterable[LabeledEpisode]) -> dict[int, list[LabeledEpisode]]:
    grouped: dict[int, list[LabeledEpisode]] = {}
    for ep in episodes:
        grouped.setdefault(ep.seed, []).append(ep)
    return grouped


def paired_recurrence_table(
    baseline: Iterable[LabeledEpisode],
    treatment: Iterable[LabeledEpisode],
    modes: Iterable[str] = TABLE_MODES,
) -> list[RecurrenceRow]:
    """Build the per-seed, per-mode paired recurrence table.

    A cell's rate is the fraction of that arm's episodes for that seed which
    failed with that mode, so the two arms stay comparable even when they ran
    different episode counts. ``delta_rate`` is treatment minus baseline, so
    a negative delta means the mode recurred less often after adaptation.

    Only seeds present in both arms are emitted, because an unpaired seed
    cannot enter a paired test. Raises if the arms share no seed.
    """
    base_by_seed = _by_seed(baseline)
    treat_by_seed = _by_seed(treatment)
    shared = sorted(set(base_by_seed) & set(treat_by_seed))
    if not shared:
        raise RecurrenceError(
            "no seed appears in both arms: "
            f"baseline seeds={sorted(base_by_seed)}, treatment seeds={sorted(treat_by_seed)}"
        )

    mode_list = list(modes)
    rows: list[RecurrenceRow] = []
    for seed in shared:
        base_eps = base_by_seed[seed]
        treat_eps = treat_by_seed[seed]
        n_base = len(base_eps)
        n_treat = len(treat_eps)
        if n_base == 0 or n_treat == 0:
            raise RecurrenceError(f"seed {seed}: an arm contributed zero episodes")
        for mode in mode_list:
            b_fail = sum(1 for ep in base_eps if ep.mode == mode)
            t_fail = sum(1 for ep in treat_eps if ep.mode == mode)
            b_rate = b_fail / n_base
            t_rate = t_fail / n_treat
            rows.append(
                RecurrenceRow(
                    seed=seed,
                    mode=mode,
                    baseline_episodes=n_base,
                    baseline_failures=b_fail,
                    baseline_rate=b_rate,
                    treatment_episodes=n_treat,
                    treatment_failures=t_fail,
                    treatment_rate=t_rate,
                    delta_rate=t_rate - b_rate,
                )
            )
    return rows


def recurrence_deltas_by_mode(rows: Iterable[RecurrenceRow]) -> dict[str, list[float]]:
    """Group the paired per-seed deltas by mode, ready for a sign-flip test."""
    out: dict[str, list[float]] = {}
    for row in rows:
        out.setdefault(row.mode, []).append(row.delta_rate)
    return out


def unknown_fraction(episodes: Iterable[LabeledEpisode]) -> float:
    """Fraction of FAILED episodes whose mode is UNKNOWN.

    Report this next to any recurrence result. On the real GO2 path it is
    expected to be high, and a recurrence table read without it would
    understate how much of the failure population is unclassified.
    """
    failures = [ep for ep in episodes if ep.mode is not None]
    if not failures:
        return 0.0
    return sum(1 for ep in failures if ep.mode == MODE_UNKNOWN) / len(failures)


__all__ = [
    "MODE_UNKNOWN",
    "REQUIRED_SIGNALS",
    "TABLE_MODES",
    "EpisodeTelemetry",
    "LabeledEpisode",
    "RecurrenceError",
    "RecurrenceRow",
    "attach_modes",
    "detect_episode_mode",
    "episode_key",
    "missing_signals",
    "paired_recurrence_table",
    "recurrence_deltas_by_mode",
    "unknown_fraction",
]
