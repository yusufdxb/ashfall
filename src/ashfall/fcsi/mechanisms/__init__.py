"""The small, interpretable library of simulator mismatches FCSI may propose.

Each entry names one parameter of :mod:`ashfall.fcsi.toy_world`, what it means physically,
how the simulator implements it, a prior range, and what limits its observability. A
mechanism outside this list cannot be diagnosed; the correct answer then is "unexplained".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ashfall.fcsi.toy_world import NOMINAL


@dataclass(frozen=True)
class Mechanism:
    name: str  # the toy_world parameter
    subsystem: str  # contact, actuation, mechanical, sensing
    meaning: str
    implementation: str
    low: float
    high: float
    observability: str

    @property
    def nominal(self) -> float:
        return NOMINAL[self.name]

    @property
    def span(self) -> float:
        return self.high - self.low

    def normalized_change(self, value: float) -> float:
        return abs(value - self.nominal) / self.span

    def to_dict(self) -> dict:
        return {**asdict(self), "nominal": self.nominal}


LIBRARY: tuple[Mechanism, ...] = (
    Mechanism(
        "mu_scale",
        "contact",
        "patch friction differs from its mapped value",
        "multiplies the patch friction coefficient",
        0.4,
        1.6,
        "only on the patch, and only when the drive force approaches traction",
    ),
    Mechanism(
        "kinetic_ratio",
        "contact",
        "friction collapses once the contact slides",
        "sliding force = ratio x static traction limit",
        0.3,
        1.0,
        "invisible until the contact slips; then dominant",
    ),
    Mechanism(
        "aniso",
        "contact",
        "braking-direction friction differs from driving-direction friction",
        "multiplies friction when the drive force is negative",
        0.3,
        1.5,
        "only while braking near the traction limit",
    ),
    Mechanism(
        "force_scale",
        "actuation",
        "actuator strength differs from the model",
        "multiplies the delivered force",
        0.5,
        1.3,
        "whenever force is large; confounded with mass at low force",
    ),
    Mechanism(
        "action_delay",
        "actuation",
        "command-to-force latency",
        "fractional delay of the commanded force, in control steps",
        0.0,
        4.0,
        "visible in open-loop replay of logged commands",
    ),
    Mechanism(
        "force_max",
        "actuation",
        "actuator saturation",
        "clips the delivered force at +/- this value (N)",
        6.0,
        30.0,
        "only when the policy requests large force",
    ),
    Mechanism(
        "pole_mass",
        "mechanical",
        "pole mass differs from the model",
        "point-mass pole mass (kg)",
        0.2,
        0.45,
        "always, weakly",
    ),
    Mechanism(
        "cart_damping",
        "mechanical",
        "unmodelled viscous drag on the cart",
        "force -= damping x cart speed",
        0.0,
        0.6,
        "always, proportional to speed",
    ),
    Mechanism(
        "obs_delay",
        "sensing",
        "sensor latency",
        "fractional delay of the policy's observations, in control steps",
        0.0,
        4.0,
        "NOT visible in open-loop replay of logged commands (the commands already carry it); "
        "only closed-loop behaviour reveals it",
    ),
    Mechanism(
        "theta_bias",
        "sensing",
        "tilt sensor bias",
        "added to the observed tilt (rad)",
        -0.08,
        0.08,
        "NOT visible in open-loop replay; only closed-loop",
    ),
)
BY_NAME = {m.name: m for m in LIBRARY}
