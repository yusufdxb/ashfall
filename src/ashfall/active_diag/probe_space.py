"""A small, interpretable space of diagnostic probes for the cart-pole toy.

A probe is one gentle crossing of a *test lane*: a sample of the suspect surface material
with a known mapped friction ``mu``, crossed at commanded speed ``speed`` by the frozen policy,
with an optional additive excitation force on top of the policy's command:

* ``none``: the crossing alone;
* ``sine(A, f)``: ``A sin(2 pi f t)`` while the cart is within [start - 0.3, start + 0.7] m;
* ``pulse(A)``: a constant ``A`` (sign = direction) while within [start + 0.1, start + 0.3] m.

A stationary probe is impossible in this toy: the frozen walking policy cannot hold the cart
still under the gait excitation (47% falls with zero command and no probe, measured while
designing the study), so every probe is a slow crossing. Contact mechanisms act only on the
suspect material, so the lane always contains it.

81 probes: mu in {0.3, 0.5, 0.8} x speed in {0.6, 0.8, 1.0} x 9 excitations (mu 0.8 is ordinary
floor, for mechanisms not tied to the material).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np

from ashfall.fcsi.toy_world import Condition

LANE_START = 0.9
MUS = (0.3, 0.5, 0.8)
SPEEDS = (0.6, 0.8, 1.0)
EXCITATIONS = (
    ("none", 0.0, 0.0),
    ("sine", 3.0, 1.0),
    ("sine", 3.0, 3.0),
    ("sine", 6.0, 1.0),
    ("sine", 6.0, 3.0),
    ("pulse", 4.0, 0.0),
    ("pulse", 8.0, 0.0),
    ("pulse", -4.0, 0.0),
    ("pulse", -8.0, 0.0),
)
MAX_AMPLITUDE = 8.0


@dataclass(frozen=True)
class Probe:
    mu: float
    speed: float
    kind: str
    amplitude: float
    frequency: float

    @property
    def name(self) -> str:
        if self.kind == "none":
            return f"cross(mu={self.mu},v={self.speed})"
        if self.kind == "sine":
            return (
                f"cross(mu={self.mu},v={self.speed})+sine({self.amplitude:g}N,{self.frequency:g}Hz)"
            )
        return f"cross(mu={self.mu},v={self.speed})+pulse({self.amplitude:+g}N)"

    @property
    def condition(self) -> Condition:
        return Condition(self.mu, LANE_START, self.speed)

    @property
    def cost(self) -> float:
        """Tie-breaker only: prefer gentler excitation."""
        return 0.01 * abs(self.amplitude) / MAX_AMPLITUDE

    def force(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.kind == "sine":
            on = (x > LANE_START - 0.3) & (x < LANE_START + 0.7)
            return np.where(on, self.amplitude * np.sin(2 * math.pi * self.frequency * t), 0.0)
        if self.kind == "pulse":
            on = (x > LANE_START + 0.1) & (x < LANE_START + 0.3)
            return np.where(on, self.amplitude, 0.0)
        return np.zeros_like(x)

    def to_dict(self) -> dict:
        return {**asdict(self), "name": self.name}


def probe_grid() -> list[Probe]:
    return [
        Probe(mu, v, k, a, f) for mu, v, (k, a, f) in itertools.product(MUS, SPEEDS, EXCITATIONS)
    ]
