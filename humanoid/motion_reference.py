"""Name-mapped joint-motion references for X1 imitation training.

The retargeting CSV may contain a floating base and joints that are not part of
the 12-DOF training model.  This module deliberately reads only the requested
joint columns, preserving the simulator's DOF order instead of relying on CSV
column positions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class JointMotionTable:
    """A validated joint-position trajectory in an explicit joint order."""

    joint_names: tuple[str, ...]
    timestamps: np.ndarray
    positions: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])


def _validate_joint_names(joint_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(joint_names)
    if not names:
        raise ValueError("joint_names must not be empty")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"joint_names contains duplicates: {duplicates}")
    return names


def load_joint_motion_csv(
    path: PathLike,
    joint_names: Sequence[str],
    *,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    close_loop: bool = False,
) -> JointMotionTable:
    """Load selected joint positions from a retargeting CSV.

    ``start_time`` is inclusive and ``end_time`` is exclusive.  When
    ``close_loop`` is enabled, a duplicate of the first selected pose is
    appended at ``end_time``.  This makes normalized phase interpolation
    continuous without consuming any non-leg columns from the source file.
    """

    names = _validate_joint_names(joint_names)
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Motion reference does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        required = ("timestamp",) + names
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(
                f"Motion reference {csv_path} is missing columns: {missing}"
            )

        timestamps = []
        positions = []
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["timestamp"])
                pose = [float(row[name]) for name in names]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric motion value in {csv_path} at row {row_number}"
                ) from exc
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp >= end_time:
                continue
            timestamps.append(timestamp)
            positions.append(pose)

    if len(timestamps) < 2:
        raise ValueError(
            f"Motion reference {csv_path} must contain at least two selected frames"
        )

    time_array = np.asarray(timestamps, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float32)
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(position_array)):
        raise ValueError(f"Motion reference {csv_path} contains NaN or Inf")
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError(
            f"Motion reference {csv_path} timestamps must be strictly increasing"
        )

    if close_loop:
        if end_time is None:
            step = float(np.median(np.diff(time_array)))
            loop_time = float(time_array[-1] + step)
        else:
            loop_time = float(end_time)
        if loop_time <= time_array[-1]:
            raise ValueError("Loop end_time must be later than the last selected frame")
        time_array = np.concatenate((time_array, np.asarray([loop_time])))
        position_array = np.concatenate((position_array, position_array[:1]), axis=0)

    time_array = time_array - time_array[0]
    return JointMotionTable(names, time_array, position_array)
