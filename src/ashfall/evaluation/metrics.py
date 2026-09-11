"""Event incidence, sustained recovery, and explicit intervention metrics.

Detector event counts are capture statistics, not independent labeled failures.
Matched counterexample recurrence is computed from episode records in ``paired``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ashfall.taxonomy.detector import FailureDetector, FailureEvent, FailureMode, FailureThresholds


@dataclass(frozen=True)
class RecoveryConfig:
    max_attitude_rad: float = 0.25
    min_height_m: float = 0.20
    max_tracking_error_mps: float = 0.15
    min_contacts: int = 2
    contact_threshold_n: float = 5.0
    consecutive_steps: int = 10

    def __post_init__(self):
        if self.consecutive_steps < 1 or not 0 <= self.min_contacts <= 4:
            raise ValueError("invalid recovery duration or contact count")
        for value in (
            self.max_attitude_rad,
            self.min_height_m,
            self.max_tracking_error_mps,
            self.contact_threshold_n,
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError("recovery thresholds must be finite and nonnegative")

    def recovered(self, telemetry: dict) -> bool:
        contacts = telemetry.get("contact_forces")
        if contacts is None:
            return False  # Unknown contact state cannot establish recovery.
        error = np.linalg.norm(
            np.asarray(telemetry["cmd_lin_vel"])[:2] - np.asarray(telemetry["actual_lin_vel"])[:2]
        )
        return bool(
            abs(telemetry["pitch_rad"]) <= self.max_attitude_rad
            and abs(telemetry["roll_rad"]) <= self.max_attitude_rad
            and telemetry["base_height_m"] >= self.min_height_m
            and error <= self.max_tracking_error_mps
            and np.sum(np.asarray(contacts) > self.contact_threshold_n) >= self.min_contacts
        )


@dataclass
class FailureMetrics:
    total_steps: int = 0
    total_episodes: int = 0
    total_failures: int = 0
    failures_by_mode: dict[str, int] = field(default_factory=dict)
    failure_rate: float = 0.0  # detected events per observed episode, not probability
    intervention_count: int = 0
    intervention_rate: float = 0.0
    mean_recovery_time_s: float | None = None
    recovered_events: int = 0
    unrecovered_events: int = 0
    episode_incidence_by_mode: dict[str, float] = field(default_factory=dict)
    repeated_events_after_recovery: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class FailureAnalyzer:
    """Capture event incidence and recovery, independently within each episode.

    Intervention is an explicit external flag or an attitude/collapse event.
    This is a declared intervention criterion, not observed human assistance.
    Recovery duration starts at detected onset (or a supplied disturbance end)
    and ends at the FIRST step of a subsequently confirmed sustained state.
    Unrecovered events are censored, never assigned zero recovery duration.
    """

    INTERVENTION_MODES = {FailureMode.COLLAPSE, FailureMode.ATTITUDE}

    def __init__(
        self, thresholds: FailureThresholds | None = None, recovery: RecoveryConfig | None = None
    ) -> None:
        self.detector = FailureDetector(thresholds)
        self.recovery = recovery or RecoveryConfig()
        self._events: list[FailureEvent] = []
        self._event_episodes: list[int] = []
        self._total_steps = 0
        self._completed = 0
        self._episode_steps = 0
        self._interventions: set[int] = set()
        self._durations: list[float] = []
        self._pending: list[float] = []
        self._censored = 0
        self._stable_start: float | None = None
        self._stable_steps = 0
        self._last_time: float | None = None
        self._seen: set[str] = set()
        self._recovered_modes: set[str] = set()
        self._repeat_counts: dict[str, int] = {}

    def step(
        self,
        *,
        done: bool = False,
        intervention_required: bool = False,
        disturbance_end_s: float | None = None,
        **telemetry,
    ) -> list[FailureEvent]:
        timestamp = float(telemetry["timestamp_s"])
        if not np.isfinite(timestamp) or (
            self._last_time is not None and timestamp <= self._last_time
        ):
            raise ValueError("timestamps must be finite and strictly increasing within episodes")
        self._last_time = timestamp
        self._total_steps += 1
        self._episode_steps += 1
        events = self.detector.step(**telemetry)
        self._events.extend(events)
        self._event_episodes.extend([self._completed] * len(events))
        if intervention_required or any(ev.mode in self.INTERVENTION_MODES for ev in events):
            self._interventions.add(self._completed)
        for ev in events:
            mode = ev.mode.value
            if mode in self._recovered_modes:
                self._repeat_counts[mode] = self._repeat_counts.get(mode, 0) + 1
                self._recovered_modes.remove(mode)
            self._seen.add(mode)
            self._pending.append(max(timestamp, disturbance_end_s or timestamp))
        if self._pending and not events and self.recovery.recovered(telemetry):
            if self._stable_start is None:
                self._stable_start = timestamp
            self._stable_steps += 1
            if self._stable_steps >= self.recovery.consecutive_steps:
                eligible = [onset for onset in self._pending if onset <= self._stable_start]
                self._durations.extend(self._stable_start - onset for onset in eligible)
                self._pending = [onset for onset in self._pending if onset > self._stable_start]
                self._recovered_modes.update(self._seen)
        else:
            self._stable_steps = 0
            self._stable_start = None
        if done:
            self._completed += 1
            self._episode_steps = 0
            self._censored += len(self._pending)
            self._pending.clear()
            self._stable_steps = 0
            self._stable_start = self._last_time = None
            self._seen.clear()
            self._recovered_modes.clear()
            self.detector = FailureDetector(self.detector.thresholds)
        return events

    def compute(self) -> FailureMetrics:
        n_episodes = self._completed + int(self._episode_steps > 0)
        counts: dict[str, int] = {}
        episodes_by_mode: dict[str, set[int]] = {}
        for event, episode in zip(self._events, self._event_episodes):
            mode = event.mode.value
            counts[mode] = counts.get(mode, 0) + 1
            episodes_by_mode.setdefault(mode, set()).add(episode)
        denom = max(n_episodes, 1)
        return FailureMetrics(
            total_steps=self._total_steps,
            total_episodes=n_episodes,
            total_failures=len(self._events),
            failures_by_mode=counts,
            failure_rate=len(self._events) / denom,
            intervention_count=len(self._interventions),
            intervention_rate=len(self._interventions) / denom,
            mean_recovery_time_s=float(np.mean(self._durations)) if self._durations else None,
            recovered_events=len(self._durations),
            unrecovered_events=self._censored + len(self._pending),
            episode_incidence_by_mode={
                mode: len(eps) / denom for mode, eps in episodes_by_mode.items()
            },
            repeated_events_after_recovery=dict(self._repeat_counts),
        )
