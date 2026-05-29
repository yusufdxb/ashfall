"""Extended metrics for failure-driven evaluation.

Adds failure-specific metrics beyond Phoenix's base set:
- Failure recurrence rate (same failure mode re-occurs after adaptation)
- Recovery time (steps from failure detection to stable state)
- Intervention count (how many times the policy would require human help)
- Per-mode failure counts and rates
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ashfall.taxonomy.detector import FailureDetector, FailureEvent, FailureMode, FailureThresholds


@dataclass
class FailureMetrics:
    """Aggregate failure metrics from one evaluation run."""

    total_steps: int = 0
    total_episodes: int = 0
    total_failures: int = 0
    failures_by_mode: dict[str, int] = field(default_factory=dict)
    failure_rate: float = 0.0  # failures per episode
    intervention_count: int = 0  # episodes requiring human intervention (collapse/attitude)
    intervention_rate: float = 0.0
    mean_recovery_steps: float = 0.0
    failure_recurrence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "total_failures": self.total_failures,
            "failures_by_mode": self.failures_by_mode,
            "failure_rate": self.failure_rate,
            "intervention_count": self.intervention_count,
            "intervention_rate": self.intervention_rate,
            "mean_recovery_steps": self.mean_recovery_steps,
            "failure_recurrence": self.failure_recurrence,
        }


class FailureAnalyzer:
    """Analyze failure events from an evaluation trajectory.

    Feed timestep-level telemetry through :meth:`step`, then call
    :meth:`compute` to get aggregate metrics.
    """

    INTERVENTION_MODES = {FailureMode.COLLAPSE, FailureMode.ATTITUDE}

    def __init__(self, thresholds: FailureThresholds | None = None) -> None:
        self.detector = FailureDetector(thresholds)
        self._events: list[FailureEvent] = []
        self._event_episodes: list[int] = []  # episode index per event, aligned with _events
        self._episode_boundaries: list[int] = [0]
        self._total_steps = 0
        self._recovery_steps: list[int] = []
        self._steps_since_failure: int | None = None
        self._stable_threshold = 10  # steps of no failure = recovered

    def step(self, *, done: bool = False, **telemetry) -> list[FailureEvent]:
        """Process one timestep of telemetry."""
        self._total_steps += 1
        events = self.detector.step(**telemetry)

        if events:
            self._events.extend(events)
            episode_idx = len(self._episode_boundaries) - 1
            self._event_episodes.extend([episode_idx] * len(events))
            if self._steps_since_failure is not None and self._steps_since_failure > 0:
                self._recovery_steps.append(self._steps_since_failure)
            self._steps_since_failure = 0
        elif self._steps_since_failure is not None:
            self._steps_since_failure += 1
            if self._steps_since_failure >= self._stable_threshold:
                self._recovery_steps.append(self._steps_since_failure)
                self._steps_since_failure = None

        if done:
            self._episode_boundaries.append(self._total_steps)

        return events

    def compute(self) -> FailureMetrics:
        """Compute aggregate failure metrics."""
        n_episodes = max(len(self._episode_boundaries) - 1, 1)

        mode_counts: dict[str, int] = {}
        for ev in self._events:
            mode_counts[ev.mode.value] = mode_counts.get(ev.mode.value, 0) + 1

        # Count distinct episodes with at least one intervention-level failure.
        intervention_episodes = {
            self._event_episodes[i]
            for i, ev in enumerate(self._events)
            if ev.mode in self.INTERVENTION_MODES
        }

        mean_recovery = (
            float(np.mean(self._recovery_steps)) if self._recovery_steps else 0.0
        )

        # Recurrence: fraction of modes that appear more than once
        recurrence: dict[str, float] = {}
        for mode, count in mode_counts.items():
            recurrence[mode] = min(count / max(n_episodes, 1), 1.0)

        return FailureMetrics(
            total_steps=self._total_steps,
            total_episodes=n_episodes,
            total_failures=len(self._events),
            failures_by_mode=mode_counts,
            failure_rate=len(self._events) / max(n_episodes, 1),
            intervention_count=len(intervention_episodes),
            intervention_rate=len(intervention_episodes) / max(n_episodes, 1),
            mean_recovery_steps=mean_recovery,
            failure_recurrence=recurrence,
        )
