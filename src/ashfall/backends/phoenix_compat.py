"""The exact Phoenix revision and interface Ashfall was integrated against.

Ashfall never imports Phoenix at module scope; the integration surface is a
handful of symbols in the sibling repository. Those symbols can drift while
Ashfall's CPU test suite stays green, which is how the restore-contract check
in ``tests/test_delivery.py`` went stale when ``InitialState`` gained four
metadata fields. This module pins what Ashfall was integrated against and
checks the live sibling against that pin at run time, so a drift is a loud
error with a named cause rather than a silent change of meaning.

``EXPECTED_PHOENIX_SHA`` is a record, not a lock: a run against a different
Phoenix revision is allowed, and the revision actually used is written into the
evidence bundle by :mod:`ashfall.provenance`. What is refused is a Phoenix whose
*interface* no longer matches, because that changes what a reset writes.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field

#: Branch and commit of go2-phoenix this Ashfall revision was integrated and
#: CPU-tested against. Update deliberately, with the interface check below.
PHOENIX_BRANCH = "feat/causal-viability-replication"
EXPECTED_PHOENIX_SHA = "5783416bf3d966a148a671d2ce7cb5f649f57c4c"

#: The kinematic restore contract: the channels ``restore_state`` writes into
#: the simulator. ``InitialState`` also carries metadata fields
#: (``position_frame``, ``position_frame_source``, ``controller_history``,
#: ``environment_parameters``) that describe how to write these, and are not
#: themselves written as robot state.
KINEMATIC_RESTORE_FIELDS: tuple[str, ...] = (
    "base_pos",
    "base_quat",
    "base_lin_vel_body",
    "base_ang_vel_body",
    "joint_pos",
    "joint_vel",
    "command_vel",
)

#: ``InitialState`` fields that are metadata about the restore, not restored
#: state. Listed explicitly so a new metadata field is a deliberate addition
#: here rather than a silent widening of the restore contract.
INITIAL_STATE_METADATA_FIELDS: tuple[str, ...] = (
    "position_frame",
    "position_frame_source",
    "controller_history",
    "environment_parameters",
)

#: Simulator dimensions the Phoenix friction adapter accepts.
FRICTION_ADAPTER_SUPPORTED: frozenset[str] = frozenset({"static_friction", "dynamic_friction"})

#: Keyword parameters ``restore_state`` must accept.
RESTORE_STATE_KEYWORDS: tuple[str, ...] = (
    "command_adapter",
    "command_hold_seconds",
    "command_telemetry",
    "position_frame",
    "controller_history",
    "require_exact_replay",
)


@dataclass(frozen=True)
class InterfaceReport:
    importable: bool
    problems: tuple[str, ...] = field(default_factory=tuple)
    phoenix_file: str | None = None

    @property
    def compatible(self) -> bool:
        return self.importable and not self.problems


def check_phoenix_interface() -> InterfaceReport:
    """Compare the live Phoenix sibling against the pinned integration surface.

    Pure introspection: nothing is constructed, no simulator is touched.
    """
    try:
        phoenix = importlib.import_module("phoenix")
        reader = importlib.import_module("phoenix.replay.trajectory_reader")
        adapter = importlib.import_module("phoenix.replay.state_adapter")
        bridge = importlib.import_module("phoenix.adaptation.scenario_bridge")
        outcomes = importlib.import_module("phoenix.training.episode_outcomes")
    except ImportError as exc:
        return InterfaceReport(False, (f"phoenix not importable: {exc}",))
    problems: list[str] = []
    fields_present = tuple(getattr(reader.InitialState, "__dataclass_fields__", {}))
    kinematic = tuple(f for f in fields_present if f not in INITIAL_STATE_METADATA_FIELDS)
    if set(kinematic) != set(KINEMATIC_RESTORE_FIELDS):
        problems.append(
            "InitialState kinematic fields differ from the pinned restore contract: "
            f"live={sorted(kinematic)} pinned={sorted(KINEMATIC_RESTORE_FIELDS)}"
        )
    unknown_metadata = set(fields_present) - set(kinematic) - set(INITIAL_STATE_METADATA_FIELDS)
    if unknown_metadata:
        problems.append(f"InitialState has unclassified fields: {sorted(unknown_metadata)}")
    signature = inspect.signature(adapter.restore_state)
    missing = [k for k in RESTORE_STATE_KEYWORDS if k not in signature.parameters]
    if missing:
        problems.append(f"restore_state lacks keyword parameters: {missing}")
    supported = getattr(bridge.FrictionScenarioAdapter, "supported", None)
    if supported is None or set(supported) != set(FRICTION_ADAPTER_SUPPORTED):
        problems.append(f"FrictionScenarioAdapter.supported changed: {supported}")
    for name in (
        "policy_id",
        "evaluation_seed",
        "success",
        "failure_modes",
        "environment_parameters",
    ):
        if name not in getattr(outcomes.EpisodeOutcome, "__dataclass_fields__", {}):
            problems.append(f"EpisodeOutcome lacks field {name}")
    return InterfaceReport(True, tuple(problems), getattr(phoenix, "__file__", None))


def assert_phoenix_interface() -> InterfaceReport:
    report = check_phoenix_interface()
    if not report.compatible:
        raise RuntimeError(
            "Phoenix interface does not match the revision Ashfall was integrated against "
            f"({PHOENIX_BRANCH} @ {EXPECTED_PHOENIX_SHA[:12]}): " + "; ".join(report.problems)
        )
    return report
