"""Refuse a scientific run whose configuration declares things the simulator ignores.

Phoenix's ``build_env_cfg`` logs a warning when a YAML block is present but not
wired (``terrain`` is the case that mislabelled every Phase-I contrast). A
warning in a log is not a gate. This module reads the same declaration the
Phoenix loader uses, when the sibling is importable, and otherwise a pinned
copy of it, and turns "declared but unapplied" into an exception for any run
that will be quoted as evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

#: Pinned copy of the Phoenix declaration at the integration revision
#: (``phoenix.sim_env.go2_env_cfg``). The live sibling is preferred when it
#: imports; this copy is what CI uses.
UNWIRED_TOP_LEVEL: tuple[str, ...] = ("termination", "terrain")
UNWIRED_ROBOT_SUB: tuple[str, ...] = ("init_state", "actuator")
APPLIED_DR_KEYS: tuple[str, ...] = (
    "enabled",
    "friction_range",
    "restitution_range",
    "mass_offset_kg",
    "motor_strength_scale",
    "actuator_latency_steps",
)
APPLIED_PERTURBATION_KEYS: tuple[str, ...] = ("enabled", "push_velocity_xy", "push_velocity_yaw")


class IgnoredConfigError(ValueError):
    """A configuration block that the simulator would silently drop."""


def declared_but_unapplied(config: Mapping[str, Any]) -> list[str]:
    """Names of blocks or keys in ``config`` that the Phoenix env builder does not apply.

    Uses the live Phoenix declaration when importable so the two cannot drift
    unnoticed, and the pinned copy above otherwise.
    """
    try:
        from phoenix.sim_env import go2_env_cfg as live

        return list(live._unwired_sections_present(dict(config)))
    except ImportError:
        pass
    unwired: list[str] = []
    for key in UNWIRED_TOP_LEVEL:
        if key in config:
            unwired.append(key)
    observation = config.get("observation") or {}
    if "include" in observation:
        unwired.append("observation.include")
    robot = config.get("robot") or {}
    for sub in UNWIRED_ROBOT_SUB:
        if sub in robot:
            unwired.append(f"robot.{sub}")
    for sub in config.get("domain_randomization") or {}:
        if sub not in APPLIED_DR_KEYS:
            unwired.append(f"domain_randomization.{sub}")
    for sub in config.get("perturbation") or {}:
        if sub not in APPLIED_PERTURBATION_KEYS:
            unwired.append(f"perturbation.{sub}")
    return unwired


def load_effective_env_config(path: str | Path) -> dict:
    """The merged env config Phoenix would build from, via its own layered loader."""
    from phoenix.sim_env import load_layered_config

    return load_layered_config(path).to_container()


def assert_config_applied(
    config: Mapping[str, Any], *, purpose: str, allow: tuple[str, ...] = ()
) -> list[str]:
    """Raise when a scientific run's config declares blocks the simulator ignores.

    ``allow`` names blocks the caller has explicitly acknowledged as
    documentation (for example a ``terrain`` block that only restates what the
    task id already fixes). Anything else declared-but-unapplied is refused,
    because a reader of the config would otherwise believe it was in force.
    """
    unwired = declared_but_unapplied(config)
    refused = [name for name in unwired if name not in allow and name.split(".")[0] not in allow]
    if refused:
        raise IgnoredConfigError(
            f"{purpose}: the environment config declares {refused}, which the simulator does "
            "not apply. Remove the block, wire it on the Phoenix side, or acknowledge it "
            "explicitly with allow=(...) so the run's record says it was documentation only."
        )
    return unwired


__all__ = [
    "APPLIED_DR_KEYS",
    "APPLIED_PERTURBATION_KEYS",
    "UNWIRED_ROBOT_SUB",
    "UNWIRED_TOP_LEVEL",
    "IgnoredConfigError",
    "assert_config_applied",
    "declared_but_unapplied",
    "load_effective_env_config",
]
