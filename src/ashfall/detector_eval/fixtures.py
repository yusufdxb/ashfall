"""A deterministic REGRESSION FIXTURE for the detector evaluation machinery.

Everything in here is authored, not measured. The episodes exist so the
metrics, the confusion matrix, the onset-timing arithmetic and the mutation
suite can be exercised on the CPU with known answers. Their labels come from
the recipe that generated them (``label_source="synthetic_fixture"``), their
manifest is a fixture manifest, and any report computed from them is a
``fixture_regression`` that no accuracy claim may rest on.

Recipes use wide physical margins rather than the detector's thresholds plus
an epsilon: a collapse drives the trunk to 0.05 m, a slip drives body speed to
zero under a 0.6 m/s command, an attitude loss tips to 1.2 rad. Ground-truth
``established_start`` is where the recipe's own physical criterion is met
(trunk at floor height, body stationary, tipped past recovery), which is not
where the detector's threshold happens to sit.
"""

from __future__ import annotations

import numpy as np

from ashfall.datasets import DatasetManifest
from ashfall.detector_eval.dataset import DetectorEvalDataset, LabeledEpisode, Telemetry
from ashfall.ontology import OnsetWindow, PhenotypeObservation

DT = 0.02
SOURCE = "synthetic_fixture"
FIXTURE_FILE = "detector_eval_fixture.jsonl"


def _nominal(n: int, cmd=(0.5, 0.0), *, height=0.30, joints=True, contacts=True) -> dict:
    t = np.arange(n) * DT
    cmd_arr = np.tile(np.asarray(cmd, dtype=float), (n, 1))
    data = {
        "pitch_rad": 0.03 * np.sin(2 * np.pi * 1.5 * t),
        "roll_rad": 0.02 * np.cos(2 * np.pi * 1.5 * t),
        "base_height_m": height + 0.004 * np.sin(2 * np.pi * 3.0 * t),
        "cmd_lin_vel": cmd_arr,
        "actual_lin_vel": cmd_arr
        + 0.01 * np.stack([np.sin(2 * np.pi * 3 * t), np.cos(2 * np.pi * 3 * t)], 1),
    }
    if joints:
        phase = 2 * np.pi * 1.5 * t[:, None] + np.arange(12)[None, :] * np.pi / 6
        data["joint_vel"] = 3.0 * np.sin(phase)
    if contacts:
        data["contact_forces"] = np.full((n, 4), 50.0)
    return data


def _ramp(n: int, start: int, stop: int, a: float, b: float) -> np.ndarray:
    """Values: ``a`` before ``start``, linear to ``b`` by ``stop``, ``b`` after."""
    out = np.full(n, float(a))
    span = max(stop - start, 1)
    for i in range(start, n):
        out[i] = b if i >= stop else a + (b - a) * (i - start + 1) / span
    return out


def _window(transition: int, established: int, precursor=None, end=None) -> OnsetWindow:
    return OnsetWindow(transition, established, precursor, end, SOURCE)


def _episode(eid, data, truth, category, **evidence) -> LabeledEpisode:
    return LabeledEpisode(
        eid, Telemetry(DT, **data), tuple(truth), category, SOURCE, {"recipe": eid, **evidence}
    )


def _collapse(eid: str, n: int = 150, start: int = 60, floor: float = 0.05) -> LabeledEpisode:
    data = _nominal(n)
    h = _ramp(n, start, start + 20, 0.30, floor)
    data["base_height_m"] = np.minimum(data["base_height_m"], h)
    established = int(np.argmax(h <= 0.08))  # trunk reaches floor height, recipe criterion
    truth = [PhenotypeObservation("collapse", _window(start, established, precursor=start - 5))]
    return _episode(eid, data, truth, "positive", physical_criterion="trunk height <= 0.08 m")


def _slip(eid: str, cmd: float = 0.6, n: int = 150, start: int = 50) -> LabeledEpisode:
    data = _nominal(n, cmd=(cmd, 0.0))
    speed = _ramp(n, start, start + 10, 1.0, 0.0)
    data["actual_lin_vel"] = data["actual_lin_vel"] * speed[:, None]
    established = int(np.argmax(speed <= 0.02))
    truth = [PhenotypeObservation("slip", _window(start, established))]
    return _episode(
        eid, data, truth, "positive", physical_criterion="feet in contact, body stationary"
    )


def _attitude(eid: str, n: int = 150, start: int = 60) -> LabeledEpisode:
    data = _nominal(n)
    pitch = _ramp(n, start, start + 30, 0.0, 1.2)
    data["pitch_rad"] = data["pitch_rad"] + pitch
    data["base_height_m"] = np.minimum(
        data["base_height_m"], _ramp(n, start, start + 30, 0.30, 0.20)
    )
    established = int(np.argmax(pitch >= 1.0))
    truth = [PhenotypeObservation("attitude", _window(start, established))]
    return _episode(
        eid, data, truth, "positive", physical_criterion="pitch past 1.0 rad, unrecoverable"
    )


def _command_mismatch(eid: str, n: int = 150, start: int = 50) -> LabeledEpisode:
    data = _nominal(n, cmd=(0.5, 0.2))
    target = np.asarray([0.1, -0.1])
    frac = _ramp(n, start, start + 15, 0.0, 1.0)[:, None]
    data["actual_lin_vel"] = (1 - frac) * data["actual_lin_vel"] + frac * target
    error = np.linalg.norm(data["cmd_lin_vel"] - data["actual_lin_vel"], axis=1)
    established = int(np.argmax(error >= 0.3))
    truth = [PhenotypeObservation("command_mismatch", _window(start, established))]
    return _episode(
        eid, data, truth, "positive", physical_criterion="planar error >= 0.3 m/s, body moving"
    )


def _stumble(eid: str, n: int = 150, start: int = 70) -> LabeledEpisode:
    data = _nominal(n)
    jv = data["joint_vel"].copy()
    jv[start : start + 8, 3:6] = 22.0
    data["joint_vel"] = jv
    truth = [PhenotypeObservation("stumble", _window(start, start, end=start + 7))]
    return _episode(
        eid, data, truth, "positive", physical_criterion="swing foot arrested, leg spike"
    )


def _contact_loss(eid: str, n: int = 150, start: int = 70) -> LabeledEpisode:
    data = _nominal(n)
    cf = data["contact_forces"].copy()
    cf[start : start + 12, :2] = 0.0
    data["contact_forces"] = cf
    truth = [PhenotypeObservation("contact_loss", _window(start, start, end=start + 11))]
    return _episode(
        eid, data, truth, "positive", physical_criterion="two feet unloaded outside gait"
    )


def _negative(eid: str, n: int = 150, **kw) -> LabeledEpisode:
    return _episode(eid, _nominal(n, **kw), [], "negative")


def _near_miss_height(eid: str, n: int = 150) -> LabeledEpisode:
    data = _nominal(n)
    profile = np.concatenate([_ramp(75, 60, 75, 0.30, 0.20), _ramp(n - 75, 0, 20, 0.20, 0.30)])
    data["base_height_m"] = np.minimum(data["base_height_m"], profile)
    return _episode(eid, data, [], "near_miss", note="height dips to 0.20 m and recovers")


def _near_miss_attitude(eid: str, n: int = 150) -> LabeledEpisode:
    data = _nominal(n)
    profile = np.concatenate([_ramp(75, 60, 75, 0.0, 0.45), _ramp(n - 75, 0, 15, 0.45, 0.0)])
    data["pitch_rad"] = data["pitch_rad"] + profile
    return _episode(eid, data, [], "near_miss", note="pitch excursion to 0.45 rad and recovers")


def _wall_blockage(eid: str, n: int = 150, start: int = 50) -> LabeledEpisode:
    data = _nominal(n)
    data["actual_lin_vel"][start:] = 0.0
    jv = data["joint_vel"].copy()
    jv[start:] *= 0.5  # legs cycle in place against the wall
    data["joint_vel"] = jv
    truth = [PhenotypeObservation("command_mismatch", _window(start, start))]
    return _episode(
        eid,
        data,
        truth,
        "confusable",
        environment="wall",
        note="blocked by an obstacle; a tracking stall the detector will call slip",
    )


def _recovery(eid: str, n: int = 300) -> LabeledEpisode:
    data = _nominal(n, cmd=(0.6, 0.0))
    speed = np.ones(n)
    speed[50:120] = np.concatenate(
        [_ramp(10, 0, 10, 1.0, 0.0), np.zeros(50), _ramp(10, 0, 10, 0.0, 1.0)]
    )
    speed[220:290] = np.concatenate(
        [_ramp(10, 0, 10, 1.0, 0.0), np.zeros(50), _ramp(10, 0, 10, 0.0, 1.0)]
    )
    data["actual_lin_vel"] = data["actual_lin_vel"] * speed[:, None]
    truth = [
        PhenotypeObservation("slip", _window(50, 60, end=119)),
        PhenotypeObservation("slip", _window(220, 230, end=289)),
    ]
    return _episode(eid, data, truth, "recovery", note="slip, two seconds nominal, slip again")


def _multi_failure(eid: str, n: int = 150) -> LabeledEpisode:
    data = _nominal(n)
    data["pitch_rad"] = data["pitch_rad"] + _ramp(n, 60, 80, 0.0, 1.1)
    h = _ramp(n, 90, 110, 0.30, 0.05)
    data["base_height_m"] = np.minimum(data["base_height_m"], h)
    truth = [
        # Both phenotypes persist to the end: the body stays tipped while the
        # trunk reaches the floor, so neither window is closed.
        PhenotypeObservation("attitude", _window(60, int(np.argmax(data["pitch_rad"] >= 1.0)))),
        PhenotypeObservation("collapse", _window(90, int(np.argmax(h <= 0.08)))),
    ]
    return _episode(
        eid, data, truth, "multi_failure", note="tips over, then the trunk reaches the floor"
    )


def fixture_episodes() -> tuple[LabeledEpisode, ...]:
    return (
        _negative("neg_walk_01"),
        _negative("neg_walk_02", cmd=(0.3, 0.1)),
        _negative("neg_walk_03_no_leg_channels", joints=False, contacts=False),
        _collapse("collapse_01"),
        _collapse("collapse_02", start=80, floor=0.04),
        _slip("slip_01"),
        _slip("slip_02", cmd=0.8),
        _slip("slip_03", start=70),
        _attitude("attitude_01"),
        _attitude("attitude_02", start=40),
        _command_mismatch("command_mismatch_01"),
        _command_mismatch("command_mismatch_02", start=70),
        _stumble("stumble_01"),
        _contact_loss("contact_loss_01"),
        _near_miss_height("near_miss_height_01"),
        _near_miss_attitude("near_miss_attitude_01"),
        _wall_blockage("confusable_wall_blockage_01"),
        _recovery("recovery_slip_01"),
        _multi_failure("multi_attitude_then_collapse_01"),
    )


def fixture_dataset() -> DetectorEvalDataset:
    """The regression fixture; its manifest file hash is the SHA256 of the episode lines."""
    episodes = fixture_episodes()
    provisional = DetectorEvalDataset(
        episodes,
        DatasetManifest("fixture", "synthetic_generator", {FIXTURE_FILE: "0" * 64}, "provisional"),
        "synthetic_fixture",
    )
    digest = provisional.episodes_sha256()
    manifest = DatasetManifest(
        "fixture",
        "synthetic_generator",
        {FIXTURE_FILE: digest},
        "Authored detector-evaluation regression fixture. Labels come from the generating "
        "recipe. Not detector validation, not physics, not hardware evidence.",
        {"generator": "ashfall.detector_eval.fixtures", "dt_s": DT, "episodes": len(episodes)},
    )
    return DetectorEvalDataset(episodes, manifest, "synthetic_fixture")


__all__ = ["DT", "FIXTURE_FILE", "SOURCE", "fixture_dataset", "fixture_episodes"]
