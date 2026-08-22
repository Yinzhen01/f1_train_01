"""Contact-aware root reconstruction for periodic retargeted walking motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContactSchedule:
    cycle_frames: int
    half_cycle_frames: int
    switch_phase: int
    switch_frames: np.ndarray
    support_foot: np.ndarray
    smoothed_relative_height: np.ndarray


@dataclass(frozen=True)
class RootReconstruction:
    positions: np.ndarray
    anchor_positions_xy: np.ndarray
    contact_weights: np.ndarray
    ground_height: float


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def estimate_cycle_frames(
    timestamps: Sequence[float],
    joint_positions: np.ndarray,
    *,
    min_period_seconds: float = 2.0,
    max_period_seconds: float = 6.0,
) -> int:
    """Estimate the first strong full-body cycle using normalized pose MSE."""

    time = np.asarray(timestamps, dtype=np.float64)
    positions = np.asarray(joint_positions, dtype=np.float64)
    if positions.ndim != 2 or len(time) != len(positions) or len(time) < 8:
        raise ValueError("timestamps and joint_positions must describe at least 8 frames")
    step = float(np.median(np.diff(time)))
    if step <= 0.0 or np.any(np.diff(time) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    normalized = (positions - positions.mean(axis=0)) / (positions.std(axis=0) + 1e-9)
    min_lag = max(2, int(round(min_period_seconds / step)))
    max_lag = min(len(time) // 2, int(round(max_period_seconds / step)))
    if min_lag >= max_lag:
        raise ValueError("period search interval does not fit the motion")
    errors = []
    for lag in range(min_lag, max_lag + 1):
        errors.append(float(np.mean(np.square(normalized[:-lag] - normalized[lag:]))))
    candidates = []
    for index in range(1, len(errors) - 1):
        if errors[index] < errors[index - 1] and errors[index] < errors[index + 1]:
            candidates.append((errors[index], min_lag + index))
    if not candidates:
        return min_lag + int(np.argmin(errors))
    return min(candidates)[1]


def periodic_contact_schedule(
    relative_foot_height: Sequence[float],
    cycle_frames: int,
    *,
    smoothing_frames: int = 11,
) -> ContactSchedule:
    """Build an alternating, exactly periodic support schedule.

    The phase is initialized from smoothed left-minus-right sole height
    crossings. Crossing residues modulo half a gait cycle are combined using a
    circular mean, which rejects a transient extra crossing without allowing
    the support schedule to lose periodicity.
    """

    relative = np.asarray(relative_foot_height, dtype=np.float64)
    if relative.ndim != 1 or len(relative) < cycle_frames:
        raise ValueError("relative_foot_height must contain at least one cycle")
    if cycle_frames < 4:
        raise ValueError("cycle_frames must be at least 4")
    half_cycle = int(round(cycle_frames / 2.0))
    smoothed = _moving_average(relative, smoothing_frames)
    raw_switches = np.flatnonzero(np.signbit(smoothed[1:]) != np.signbit(smoothed[:-1])) + 1
    if len(raw_switches) < 2:
        raise ValueError("motion does not contain alternating foot-height crossings")
    residues = np.mod(raw_switches, half_cycle)
    angles = residues * (2.0 * np.pi / half_cycle)
    phase_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    if phase_angle < 0.0:
        phase_angle += 2.0 * np.pi
    phase = int(round(phase_angle * half_cycle / (2.0 * np.pi))) % half_cycle
    switch_frames = np.arange(phase, len(relative), half_cycle, dtype=np.int64)
    switch_frames = switch_frames[switch_frames > 0]

    support = np.full(len(relative), 0 if smoothed[0] <= 0.0 else 1, dtype=np.int64)
    current = int(support[0])
    for frame in switch_frames:
        current = 1 - current
        support[frame:] = current
    return ContactSchedule(
        cycle_frames,
        half_cycle,
        phase,
        switch_frames,
        support,
        smoothed,
    )


def _support_segments(support_foot: np.ndarray, switch_frames: np.ndarray) -> list[tuple[int, int, int]]:
    bounds = [0, *[int(value) for value in switch_frames], len(support_foot)]
    return [
        (bounds[index], bounds[index + 1] - 1, int(support_foot[bounds[index]]))
        for index in range(len(bounds) - 1)
    ]


def reconstruct_root_translation(
    raw_positions: np.ndarray,
    root_rotations: np.ndarray,
    foot_points_base: np.ndarray,
    schedule: ContactSchedule,
    *,
    overlap_frames: int = 5,
    contact_weight: float = 100.0,
    acceleration_weight: float = 2.0,
    velocity_weight: float = 0.05,
    initial_position_weight: float = 10000.0,
) -> RootReconstruction:
    """Globally solve root translation while retaining the source orientation.

    Horizontal stance anchors are optimized jointly with every root position.
    In the transition window both feet receive soft contact constraints. The
    vertical solve uses one shared ground plane, preventing recursive height
    drift. No raw horizontal velocity prior is used, because that is the signal
    being audited for foot skating.
    """

    raw_positions = np.asarray(raw_positions, dtype=np.float64)
    rotations = np.asarray(root_rotations, dtype=np.float64)
    foot_points = np.asarray(foot_points_base, dtype=np.float64)
    frame_count = len(raw_positions)
    if raw_positions.shape != (frame_count, 3):
        raise ValueError("raw_positions must have shape (frames, 3)")
    if rotations.shape != (frame_count, 3, 3):
        raise ValueError("root_rotations must have shape (frames, 3, 3)")
    if foot_points.shape != (frame_count, 2, 3):
        raise ValueError("foot_points_base must have shape (frames, 2, 3)")
    if len(schedule.support_foot) != frame_count:
        raise ValueError("contact schedule length does not match motion")

    offsets = np.einsum("fij,fkj->fki", rotations, foot_points)
    segments = _support_segments(schedule.support_foot, schedule.switch_frames)
    segment_count = len(segments)
    contact_weights = np.zeros((frame_count, 2), dtype=np.float64)
    horizontal_rows = []
    horizontal_rhs = [[], []]
    for segment_index, (start, end, foot) in enumerate(segments):
        expanded_start = max(0, start - overlap_frames)
        expanded_end = min(frame_count - 1, end + overlap_frames)
        for frame in range(expanded_start, expanded_end + 1):
            distance = max(start - frame, 0, frame - end)
            confidence = (
                1.0
                if distance == 0
                else (overlap_frames + 1 - distance) / (overlap_frames + 1)
            )
            contact_weights[frame, foot] = max(contact_weights[frame, foot], confidence)
            weight = np.sqrt(contact_weight * confidence)
            row = np.zeros(frame_count + segment_count, dtype=np.float64)
            row[frame] = weight
            row[frame_count + segment_index] = -weight
            horizontal_rows.append(row)
            for axis in range(2):
                horizontal_rhs[axis].append(-weight * offsets[frame, foot, axis])

    for frame in range(1, frame_count - 1):
        row = np.zeros(frame_count + segment_count, dtype=np.float64)
        weight = np.sqrt(acceleration_weight)
        row[frame - 1] = weight
        row[frame] = -2.0 * weight
        row[frame + 1] = weight
        horizontal_rows.append(row)
        horizontal_rhs[0].append(0.0)
        horizontal_rhs[1].append(0.0)
    for frame in range(frame_count - 1):
        row = np.zeros(frame_count + segment_count, dtype=np.float64)
        weight = np.sqrt(velocity_weight)
        row[frame] = -weight
        row[frame + 1] = weight
        horizontal_rows.append(row)
        horizontal_rhs[0].append(0.0)
        horizontal_rhs[1].append(0.0)

    horizontal_matrix = np.vstack(horizontal_rows)
    positions = np.zeros_like(raw_positions)
    anchors = np.zeros((segment_count, 2), dtype=np.float64)
    for axis in range(2):
        pin = np.zeros(frame_count + segment_count, dtype=np.float64)
        pin[0] = np.sqrt(initial_position_weight)
        matrix = np.vstack((horizontal_matrix, pin))
        rhs = np.concatenate(
            (
                np.asarray(horizontal_rhs[axis]),
                [np.sqrt(initial_position_weight) * raw_positions[0, axis]],
            )
        )
        solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        positions[:, axis] = solution[:frame_count]
        anchors[:, axis] = solution[frame_count:]

    initial_foot = int(schedule.support_foot[0])
    ground_height = float(raw_positions[0, 2] + offsets[0, initial_foot, 2])
    vertical_rows = []
    vertical_rhs = []
    for start, end, foot in segments:
        expanded_start = max(0, start - overlap_frames)
        expanded_end = min(frame_count - 1, end + overlap_frames)
        for frame in range(expanded_start, expanded_end + 1):
            distance = max(start - frame, 0, frame - end)
            confidence = (
                1.0
                if distance == 0
                else (overlap_frames + 1 - distance) / (overlap_frames + 1)
            )
            weight = np.sqrt(contact_weight * confidence)
            row = np.zeros(frame_count, dtype=np.float64)
            row[frame] = weight
            vertical_rows.append(row)
            vertical_rhs.append(weight * (ground_height - offsets[frame, foot, 2]))
    for frame in range(1, frame_count - 1):
        row = np.zeros(frame_count, dtype=np.float64)
        weight = np.sqrt(acceleration_weight)
        row[frame - 1] = weight
        row[frame] = -2.0 * weight
        row[frame + 1] = weight
        vertical_rows.append(row)
        vertical_rhs.append(0.0)
    positions[:, 2] = np.linalg.lstsq(
        np.vstack(vertical_rows), np.asarray(vertical_rhs), rcond=None
    )[0]
    return RootReconstruction(positions, anchors, contact_weights, ground_height)


def stance_slip_metrics(
    timestamps: Sequence[float],
    root_positions: np.ndarray,
    root_rotations: np.ndarray,
    foot_points_base: np.ndarray,
    schedule: ContactSchedule,
    *,
    transition_margin_frames: int = 5,
) -> dict[str, float]:
    time = np.asarray(timestamps, dtype=np.float64)
    positions = np.asarray(root_positions, dtype=np.float64)
    offsets = np.einsum("fij,fkj->fki", root_rotations, foot_points_base)
    world = positions[:, None, :] + offsets
    velocity = np.diff(world, axis=0) / np.diff(time)[:, None, None]
    primary = velocity[np.arange(len(time) - 1), schedule.support_foot[:-1]]
    mask = np.ones(len(primary), dtype=bool)
    for frame in schedule.switch_frames:
        mask[
            max(0, int(frame) - transition_margin_frames) : min(
                len(mask), int(frame) + transition_margin_frames
            )
        ] = False
    horizontal_speed = np.linalg.norm(primary[mask, :2], axis=1)
    return {
        "stance_slip_rms_mps": float(np.sqrt(np.mean(np.square(horizontal_speed)))),
        "stance_slip_p95_mps": float(np.percentile(horizontal_speed, 95)),
    }
