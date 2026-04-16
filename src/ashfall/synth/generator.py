"""Synthetic failure trajectory generator.

Generates realistic-looking failure trajectories in Phoenix's Parquet schema
for all 6 Ashfall failure modes. These serve as:
1. Curriculum training data when real hardware failures are not yet available
2. Unit test fixtures for the full pipeline
3. Baselines for evaluating the failure detector

Each generator produces a trajectory that starts stable, transitions into
the failure mode, and (optionally) includes a partial recovery attempt.
Physics are approximate — not sim-grade — but structurally correct for
the Parquet schema and failure detector thresholds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ashfall.taxonomy.detector import FailureMode

logger = logging.getLogger("ashfall.synth.generator")

# Phoenix Parquet schema (must match trajectory_logger.py exactly)
SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("timestamp_s", pa.float64()),
        ("base_pos", pa.list_(pa.float32(), 3)),
        ("base_quat", pa.list_(pa.float32(), 4)),
        ("base_lin_vel_body", pa.list_(pa.float32(), 3)),
        ("base_ang_vel_body", pa.list_(pa.float32(), 3)),
        ("joint_pos", pa.list_(pa.float32(), 12)),
        ("joint_vel", pa.list_(pa.float32(), 12)),
        ("command_vel", pa.list_(pa.float32(), 3)),
        ("action", pa.list_(pa.float32(), 12)),
        ("contact_forces", pa.list_(pa.float32(), 4)),
        ("failure_flag", pa.bool_()),
        ("failure_mode", pa.string()),
    ]
)

# GO2 nominal standing joint positions (radians)
GO2_NOMINAL_JOINTS = np.array(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5],
    dtype=np.float32,
)

DT = 0.02  # 50 Hz control


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler (XYZ) to quaternion (xyzw)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )


def _stable_row(
    step: int,
    rng: np.random.Generator,
    cmd_vel: np.ndarray | None = None,
) -> dict:
    """Generate one row of stable walking telemetry."""
    t = step * DT
    if cmd_vel is None:
        cmd_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    # Slight noise on actual velocity
    actual_vel = cmd_vel[:2] + rng.normal(0, 0.02, size=2).astype(np.float32)
    base_height = 0.30 + rng.normal(0, 0.005)

    return {
        "step": step,
        "timestamp_s": t,
        "base_pos": [0.0 + cmd_vel[0] * t, 0.0 + cmd_vel[1] * t, float(base_height)],
        "base_quat": _quat_from_euler(
            rng.normal(0, 0.02), rng.normal(0, 0.02), cmd_vel[2] * t
        ).tolist(),
        "base_lin_vel_body": [float(actual_vel[0]), float(actual_vel[1]), rng.normal(0, 0.01)],
        "base_ang_vel_body": [rng.normal(0, 0.05), rng.normal(0, 0.05), float(cmd_vel[2])],
        "joint_pos": (GO2_NOMINAL_JOINTS + rng.normal(0, 0.1, size=12).astype(np.float32)).tolist(),
        "joint_vel": rng.normal(0, 2.0, size=12).astype(np.float32).tolist(),
        "command_vel": cmd_vel.tolist(),
        "action": rng.normal(0, 0.5, size=12).astype(np.float32).tolist(),
        "contact_forces": (50.0 + rng.normal(0, 5, size=4)).clip(0).astype(np.float32).tolist(),
        "failure_flag": False,
        "failure_mode": None,
    }


def generate_attitude_failure(
    n_stable: int = 50,
    n_failure: int = 30,
    seed: int = 0,
) -> list[dict]:
    """Attitude loss: pitch/roll ramps past threshold."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stable):
        rows.append(_stable_row(i, rng))

    for j in range(n_failure):
        step = n_stable + j
        progress = j / max(n_failure - 1, 1)
        pitch = 0.3 + progress * 0.8  # ramps to ~1.1 rad
        roll = rng.normal(0, 0.05 + progress * 0.3)
        base_height = 0.30 - progress * 0.1

        row = _stable_row(step, rng)
        row["base_quat"] = _quat_from_euler(float(roll), float(pitch), 0.0).tolist()
        row["base_pos"][2] = float(base_height)
        row["failure_flag"] = pitch > 0.8 or abs(roll) > 0.6
        row["failure_mode"] = FailureMode.ATTITUDE.value if row["failure_flag"] else None
        rows.append(row)

    return rows


def generate_collapse_failure(
    n_stable: int = 50,
    n_failure: int = 20,
    seed: int = 1,
) -> list[dict]:
    """Body collapse: height drops below threshold."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stable):
        rows.append(_stable_row(i, rng))

    for j in range(n_failure):
        step = n_stable + j
        progress = j / max(n_failure - 1, 1)
        base_height = 0.30 - progress * 0.25  # drops to ~0.05 m

        row = _stable_row(step, rng)
        row["base_pos"][2] = float(base_height)
        cf = (10.0 + rng.normal(0, 5, size=4)).clip(0).astype(np.float32)
        row["contact_forces"] = cf.tolist()
        row["failure_flag"] = base_height < 0.15
        row["failure_mode"] = FailureMode.COLLAPSE.value if row["failure_flag"] else None
        rows.append(row)

    return rows


def generate_slip_failure(
    n_stable: int = 50,
    n_failure: int = 40,
    seed: int = 2,
) -> list[dict]:
    """Foot slip: high command but zero actual velocity."""
    rng = np.random.default_rng(seed)
    cmd = np.array([0.6, 0.0, 0.0], dtype=np.float32)
    rows = []
    for i in range(n_stable):
        rows.append(_stable_row(i, rng, cmd_vel=cmd))

    for j in range(n_failure):
        step = n_stable + j
        row = _stable_row(step, rng, cmd_vel=cmd)
        # Actual velocity drops to near zero
        row["base_lin_vel_body"] = [
            float(rng.normal(0, 0.02)),
            float(rng.normal(0, 0.02)),
            float(rng.normal(0, 0.01)),
        ]
        row["contact_forces"] = (
            rng.uniform(2, 15, size=4).astype(np.float32).tolist()
        )
        row["failure_flag"] = j >= max(n_failure // 2, 1)  # after sustained slip
        row["failure_mode"] = FailureMode.SLIP.value if row["failure_flag"] else None
        rows.append(row)

    return rows


def generate_stumble_failure(
    n_stable: int = 50,
    n_failure: int = 10,
    seed: int = 3,
) -> list[dict]:
    """Stumble: sudden joint velocity spike during swing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stable):
        rows.append(_stable_row(i, rng))

    for j in range(n_failure):
        step = n_stable + j
        row = _stable_row(step, rng)
        # Spike one leg's joint velocities
        jv = rng.normal(0, 2.0, size=12).astype(np.float32)
        leg_idx = rng.integers(0, 4) * 3
        jv[leg_idx : leg_idx + 3] = rng.uniform(16, 25, size=3).astype(np.float32)
        row["joint_vel"] = jv.tolist()
        row["failure_flag"] = True
        row["failure_mode"] = FailureMode.STUMBLE.value
        rows.append(row)

    return rows


def generate_contact_loss_failure(
    n_stable: int = 50,
    n_failure: int = 15,
    seed: int = 4,
) -> list[dict]:
    """Contact loss: multiple feet lose ground contact."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stable):
        rows.append(_stable_row(i, rng))

    for j in range(n_failure):
        step = n_stable + j
        row = _stable_row(step, rng)
        # Two feet lose contact
        forces = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)
        lost_feet = rng.choice(4, size=2, replace=False)
        forces[lost_feet] = rng.uniform(0, 3, size=2).astype(np.float32)
        row["contact_forces"] = forces.tolist()
        row["failure_flag"] = j >= 5  # after sustained contact loss
        row["failure_mode"] = FailureMode.CONTACT_LOSS.value if row["failure_flag"] else None
        rows.append(row)

    return rows


def generate_command_mismatch_failure(
    n_stable: int = 50,
    n_failure: int = 60,
    seed: int = 5,
) -> list[dict]:
    """Command mismatch: robot moves but not as commanded."""
    rng = np.random.default_rng(seed)
    cmd = np.array([0.5, 0.2, 0.0], dtype=np.float32)
    rows = []
    for i in range(n_stable):
        rows.append(_stable_row(i, rng, cmd_vel=cmd))

    for j in range(n_failure):
        step = n_stable + j
        row = _stable_row(step, rng, cmd_vel=cmd)
        # Robot moves but in wrong direction
        actual = np.array([0.1, -0.1, 0.0], dtype=np.float32) + rng.normal(0, 0.03, size=3).astype(
            np.float32
        )
        row["base_lin_vel_body"] = actual.tolist()
        row["failure_flag"] = j >= max(n_failure // 2, 1)  # after sustained mismatch
        row["failure_mode"] = FailureMode.COMMAND_MISMATCH.value if row["failure_flag"] else None
        rows.append(row)

    return rows


GENERATORS = {
    FailureMode.ATTITUDE: generate_attitude_failure,
    FailureMode.COLLAPSE: generate_collapse_failure,
    FailureMode.SLIP: generate_slip_failure,
    FailureMode.STUMBLE: generate_stumble_failure,
    FailureMode.CONTACT_LOSS: generate_contact_loss_failure,
    FailureMode.COMMAND_MISMATCH: generate_command_mismatch_failure,
}


def generate_all_failures(
    output_dir: str | Path,
    n_variants: int = 3,
    seed: int = 42,
) -> list[Path]:
    """Generate synthetic failure parquets for all 6 modes.

    Creates ``n_variants`` trajectories per mode with different seeds.
    Returns list of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for mode, gen_fn in GENERATORS.items():
        for v in range(n_variants):
            variant_seed = seed + hash(mode.value) % 1000 + v
            rows = gen_fn(seed=variant_seed)
            table = pa.Table.from_pylist(rows, schema=SCHEMA)

            fname = f"synth_{mode.value}_{v:03d}.parquet"
            path = output_dir / fname
            pq.write_table(table, path, compression="zstd")
            paths.append(path)
            logger.info("Generated %s (%d rows)", fname, len(rows))

    logger.info("Generated %d synthetic failure files in %s", len(paths), output_dir)
    return paths


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else "data/failures"
    generate_all_failures(out)
