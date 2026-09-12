"""Paired per-seed, per-mode failure INCIDENCE from Phoenix episode records.

Terminology, fixed on purpose. *Episode incidence* is the fraction of an
arm's episodes in which a phenotype occurred at all. *Recurrence* is a failure
that occurs, resolves into sustained recovery, and occurs again within the
same episode. The table built here is an incidence table: each episode
carries at most one label and nothing in it says whether the robot recovered.
Earlier versions of this module called that recurrence; the names
``paired_recurrence_table``, ``recurrence_deltas_by_mode`` and
``RecurrenceRow`` survive only as deprecated aliases, and
:func:`episode_recurrence` is the function that counts actual recurrence
when per-event recovery information exists.

Phoenix's aggregate ``metrics_*.json`` cannot supply either quantity: it
reports a single success rate per cell with the per-episode identity already
averaged away. This module consumes the per-episode record artifact instead,
attaches Ashfall's six-mode detector to every failed episode, and produces a
table paired by seed so that a per-seed sign-flip test can be run on the
deltas.

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

#: Row order of the incidence table: the six modes plus the explicit
#: unknown bucket. Always emitted, including as zeros, so a mode that never
#: fires is distinguishable from a mode that was never looked for.
TABLE_MODES: tuple[str, ...] = tuple(m.value for m in FailureMode) + (MODE_UNKNOWN,)


class RecurrenceError(ValueError):
    """Raised when an incidence or recurrence table cannot be built from the given records."""


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
class IncidenceRow:
    """One (seed, mode) cell of the paired episode-incidence table.

    ``*_rate`` is the fraction of that arm's episodes in which the mode was
    the episode's label. It is an incidence, not a recurrence: the episode
    may have failed once and never recovered.
    """

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


def paired_incidence_table(
    baseline: Iterable[LabeledEpisode],
    treatment: Iterable[LabeledEpisode],
    modes: Iterable[str] = TABLE_MODES,
) -> list[IncidenceRow]:
    """Build the per-seed, per-mode paired episode-incidence table.

    A cell's rate is the fraction of that arm's episodes for that seed which
    failed with that mode, so the two arms stay comparable even when they ran
    different episode counts. ``delta_rate`` is treatment minus baseline, so
    a negative delta means the mode occurred in fewer episodes after
    adaptation. This says nothing about recurrence within an episode.

    Only seeds present in both arms are emitted, because an unpaired seed
    cannot enter a paired test. Raises if the arms share no seed.

    ``modes`` must cover every mode observed in either arm. A restricted list
    that would drop observed failures raises rather than emitting a table
    whose cells do not add up to the failures that were actually recorded.
    """
    baseline = list(baseline)
    treatment = list(treatment)
    base_by_seed = _by_seed(baseline)
    treat_by_seed = _by_seed(treatment)
    shared = sorted(set(base_by_seed) & set(treat_by_seed))
    if not shared:
        raise RecurrenceError(
            "no seed appears in both arms: "
            f"baseline seeds={sorted(base_by_seed)}, treatment seeds={sorted(treat_by_seed)}"
        )

    mode_list = list(modes)
    if not mode_list:
        raise RecurrenceError("no modes requested; an empty recurrence table is not a result")
    duplicates = sorted({m for m in mode_list if mode_list.count(m) > 1})
    if duplicates:
        raise RecurrenceError(f"duplicate mode(s) requested: {', '.join(duplicates)}")

    observed = {ep.mode for ep in baseline + treatment if ep.mode is not None}
    unaccounted = sorted(observed - set(mode_list))
    if unaccounted:
        raise RecurrenceError(
            "observed failure mode(s) absent from the requested modes, which would "
            f"drop them from the table: {', '.join(unaccounted)}"
        )

    rows: list[IncidenceRow] = []
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
                IncidenceRow(
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


def incidence_deltas_by_mode(rows: Iterable[IncidenceRow]) -> dict[str, list[float]]:
    """Group the paired per-seed incidence deltas by mode, ready for a sign-flip test."""
    out: dict[str, list[float]] = {}
    for row in rows:
        out.setdefault(row.mode, []).append(row.delta_rate)
    return out


def unknown_fraction(episodes: Iterable[LabeledEpisode]) -> float:
    """Fraction of FAILED episodes whose mode is UNKNOWN.

    Report this next to any incidence result. On the real GO2 path it is
    expected to be high, and an incidence table read without it would
    understate how much of the failure population is unclassified.

    Returns ``0.0`` when there are no failed episodes at all. That is the
    labelled empty case, not a measured blind spot of zero: with no failures
    there is nothing to classify, so check the failure count before reading
    this number as evidence that the detector covered the population.
    """
    failures = [ep for ep in episodes if ep.mode is not None]
    if not failures:
        return 0.0
    return sum(1 for ep in failures if ep.mode == MODE_UNKNOWN) / len(failures)


@dataclass(frozen=True)
class RecurrenceCount:
    """Within-episode recurrences of a mode, or why they could not be counted.

    ``count`` is None whenever the episode carries no per-event recovery
    information. An episode that failed once and never recovered has zero
    recurrences and a defined count; an episode whose recovery was never
    observed has an undefined count, and the two are not the same thing.
    """

    mode: str
    count: Optional[int]
    reason: str = ""

    def to_dict(self) -> dict:
        return {"mode": self.mode, "count": self.count, "reason": self.reason}


def episode_recurrence(episode: Mapping) -> dict[str, RecurrenceCount]:
    """Count true recurrences per mode in one episode record.

    Recurrence requires three things in order within the episode: an event of
    the mode, a sustained recovery after it, and a later event of the same
    mode. Two sources are accepted, in this order of preference:

    * ``repeated_events_after_recovery``: the per-mode counts that
      :class:`ashfall.evaluation.metrics.FailureAnalyzer` computes online,
      where recovery is the sustained state defined by its ``RecoveryConfig``;
    * ``failure_events`` carrying a ``recovered_at_s`` timestamp per event, in
      which case a later event of the same mode after that timestamp counts.

    Without either, every mode in ``failure_modes`` is reported with
    ``count=None`` and the reason. A ``recovery_outcome`` string alone is not
    enough: it says whether the episode ended recovered, not when.
    """
    modes = episode.get("failure_modes")
    if modes is None:
        return {}
    repeated = episode.get("repeated_events_after_recovery")
    if isinstance(repeated, Mapping):
        return {
            mode: RecurrenceCount(mode, int(repeated.get(mode, 0)), "failure_analyzer_recovery")
            for mode in modes
        }
    events = episode.get("failure_events") or []
    if events and all("recovered_at_s" in event for event in events):
        counts: dict[str, int] = {mode: 0 for mode in modes}
        recovered_after: dict[str, float] = {}
        for event in sorted(events, key=lambda e: float(e["timestamp_s"])):
            mode = str(event["mode"])
            when = float(event["timestamp_s"])
            if mode in recovered_after and when > recovered_after[mode]:
                counts[mode] = counts.get(mode, 0) + 1
            recovered = event.get("recovered_at_s")
            if recovered is not None:
                recovered_after[mode] = float(recovered)
            else:
                recovered_after.pop(mode, None)
        return {
            mode: RecurrenceCount(mode, counts.get(mode, 0), "per_event_recovery_timestamps")
            for mode in modes
        }
    reason = (
        "no per-event recovery information: incidence is known, recurrence is not"
        if modes
        else "no failure in this episode"
    )
    return {mode: RecurrenceCount(mode, None, reason) for mode in modes}


def _deprecated(old: str, new: str, target):
    def wrapper(*args, **kwargs):
        import warnings

        warnings.warn(
            f"{old} is deprecated: the table it builds is an episode INCIDENCE table, not a "
            f"recurrence table. Use {new}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return target(*args, **kwargs)

    wrapper.__name__ = old
    wrapper.__doc__ = f"Deprecated alias for :func:`{new}`. Episode incidence is not recurrence."
    return wrapper


#: Deprecated names. Episode incidence is not recurrence; see the module docstring.
paired_recurrence_table = _deprecated(
    "paired_recurrence_table", "paired_incidence_table", paired_incidence_table
)
recurrence_deltas_by_mode = _deprecated(
    "recurrence_deltas_by_mode", "incidence_deltas_by_mode", incidence_deltas_by_mode
)
RecurrenceRow = IncidenceRow  # deprecated alias; the row is an incidence cell


__all__ = [
    "MODE_UNKNOWN",
    "REQUIRED_SIGNALS",
    "TABLE_MODES",
    "EpisodeTelemetry",
    "LabeledEpisode",
    "IncidenceRow",
    "RecurrenceCount",
    "RecurrenceError",
    "RecurrenceRow",
    "attach_modes",
    "detect_episode_mode",
    "episode_key",
    "episode_recurrence",
    "incidence_deltas_by_mode",
    "missing_signals",
    "paired_incidence_table",
    "paired_recurrence_table",
    "recurrence_deltas_by_mode",
    "unknown_fraction",
]
