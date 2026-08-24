"""Utilities for loading the X1 12-DOF retargeted walking reference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


X1_12DOF_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


@dataclass(frozen=True)
class RetargetReference:
    timestamps: np.ndarray
    joint_names: Tuple[str, ...]
    joint_positions: np.ndarray
    root_positions: np.ndarray
    root_quaternions_xyzw: np.ndarray
    contacts: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return tuple(reader.fieldnames or ()), rows


def _column_matrix(rows, names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[float(row[name]) for name in names] for row in rows], dtype=np.float64
    )


def load_retarget_reference(joint_path, root_path) -> RetargetReference:
    """Load and cross-check joint-only data plus auxiliary root/contact data."""

    joint_path = Path(joint_path)
    root_path = Path(root_path)
    joint_fields, joint_rows = _read_rows(joint_path)
    root_fields, root_rows = _read_rows(root_path)

    expected_joint_fields = ("timestamp",) + X1_12DOF_JOINT_NAMES
    if joint_fields != expected_joint_fields:
        raise ValueError(
            f"{joint_path} must contain timestamp plus exactly the 12 X1 joints"
        )
    required_root_fields = {
        "timestamp",
        "root_pos_x",
        "root_pos_y",
        "root_pos_z",
        "root_quat_x",
        "root_quat_y",
        "root_quat_z",
        "root_quat_w",
        "left_contact",
        "right_contact",
        *X1_12DOF_JOINT_NAMES,
    }
    missing = required_root_fields.difference(root_fields)
    if missing:
        raise ValueError(f"{root_path} is missing fields: {sorted(missing)}")
    if not joint_rows or len(joint_rows) != len(root_rows):
        raise ValueError("joint and root references must contain the same nonzero rows")

    timestamps = _column_matrix(joint_rows, ("timestamp",))[:, 0]
    root_timestamps = _column_matrix(root_rows, ("timestamp",))[:, 0]
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("reference timestamps must be strictly increasing")
    if not np.allclose(timestamps, root_timestamps, atol=1e-9, rtol=0.0):
        raise ValueError("joint and root timestamps do not match")

    joints = _column_matrix(joint_rows, X1_12DOF_JOINT_NAMES)
    root_joints = _column_matrix(root_rows, X1_12DOF_JOINT_NAMES)
    if not np.allclose(joints, root_joints, atol=1e-9, rtol=0.0):
        raise ValueError("auxiliary root reference changes the 12 joint trajectory")

    root_positions = _column_matrix(
        root_rows, ("root_pos_x", "root_pos_y", "root_pos_z")
    )
    root_quaternions = _column_matrix(
        root_rows,
        ("root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"),
    )
    norms = np.linalg.norm(root_quaternions, axis=1)
    if np.any(norms < 1e-9):
        raise ValueError("root reference contains a zero quaternion")
    root_quaternions = root_quaternions / norms[:, None]
    contacts = _column_matrix(root_rows, ("left_contact", "right_contact"))

    return RetargetReference(
        timestamps=timestamps,
        joint_names=X1_12DOF_JOINT_NAMES,
        joint_positions=joints,
        root_positions=root_positions,
        root_quaternions_xyzw=root_quaternions,
        contacts=contacts,
    )


def interpolate_reference(reference: RetargetReference, time_s: float):
    """Linearly interpolate translation/joints and normalized-lerp quaternion."""

    time_s = float(np.clip(time_s, reference.timestamps[0], reference.timestamps[-1]))
    upper = int(np.searchsorted(reference.timestamps, time_s, side="right"))
    upper = min(max(upper, 1), len(reference.timestamps) - 1)
    lower = upper - 1
    span = reference.timestamps[upper] - reference.timestamps[lower]
    alpha = 0.0 if span <= 0.0 else (time_s - reference.timestamps[lower]) / span

    joints = (1.0 - alpha) * reference.joint_positions[lower] + alpha * reference.joint_positions[upper]
    root_pos = (1.0 - alpha) * reference.root_positions[lower] + alpha * reference.root_positions[upper]
    q0 = reference.root_quaternions_xyzw[lower]
    q1 = reference.root_quaternions_xyzw[upper]
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    root_quat = (1.0 - alpha) * q0 + alpha * q1
    root_quat /= np.linalg.norm(root_quat)
    contacts = reference.contacts[lower if alpha < 0.5 else upper]
    return joints, root_pos, root_quat, contacts
