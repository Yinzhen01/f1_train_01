"""Reconstruct a periodic X1 root trajectory from alternating foot contacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from humanoid.contact_motion import (
    estimate_cycle_frames,
    periodic_contact_schedule,
    reconstruct_root_translation,
    stance_slip_metrics,
)
from humanoid.motion_kinematics import (
    chain_to_link,
    evaluate_chain,
    parse_urdf,
    quat_matrix,
    sole_center_from_mesh,
)


FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
CONTROLLED_JOINTS = (
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
ROOT_COLUMNS = (
    "root_pos_x",
    "root_pos_y",
    "root_pos_z",
    "root_quat_x",
    "root_quat_y",
    "root_quat_z",
    "root_quat_w",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = ("timestamp", *ROOT_COLUMNS, *CONTROLLED_JOINTS)
    missing = [name for name in required if not rows or name not in rows[0]]
    if missing:
        raise ValueError(f"Source motion is missing columns: {missing}")
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    root_positions = np.asarray(
        [[float(row[f"root_pos_{axis}"]) for axis in "xyz"] for row in rows],
        dtype=np.float64,
    )
    joint_positions = np.asarray(
        [[float(row[name]) for name in CONTROLLED_JOINTS] for row in rows],
        dtype=np.float64,
    )
    return rows, timestamps, root_positions, joint_positions


def reconstruct(source: Path, urdf: Path, output: Path, diagnostics: Path) -> dict[str, object]:
    rows, timestamps, raw_root_positions, joint_positions = _load_source(source)
    by_child, limits = parse_urdf(urdf)
    chains = [chain_to_link(by_child, link) for link in FOOT_LINKS]
    zero_poses = [evaluate_chain(chain, {}, limits) for chain in chains]
    sole_points = np.asarray(
        [
            sole_center_from_mesh(urdf, link, zero_pose)
            for link, zero_pose in zip(FOOT_LINKS, zero_poses)
        ],
        dtype=np.float64,
    )

    root_rotations = np.asarray(
        [
            quat_matrix([float(row[f"root_quat_{axis}"]) for axis in "xyzw"])
            for row in rows
        ]
    )
    foot_points_base = []
    for row in rows:
        values = {name: float(row[name]) for name in CONTROLLED_JOINTS}
        frame_points = []
        for chain, sole_point in zip(chains, sole_points):
            pose = evaluate_chain(chain, values, limits)
            frame_points.append((pose @ np.asarray([*sole_point, 1.0]))[:3])
        foot_points_base.append(frame_points)
    foot_points_base = np.asarray(foot_points_base, dtype=np.float64)

    cycle_frames = estimate_cycle_frames(timestamps, joint_positions)
    world_offsets = np.einsum("fij,fkj->fki", root_rotations, foot_points_base)
    relative_height = world_offsets[:, 0, 2] - world_offsets[:, 1, 2]
    schedule = periodic_contact_schedule(relative_height, cycle_frames)
    result = reconstruct_root_translation(
        raw_root_positions,
        root_rotations,
        foot_points_base,
        schedule,
    )

    raw_slip = stance_slip_metrics(
        timestamps,
        raw_root_positions,
        root_rotations,
        foot_points_base,
        schedule,
    )
    reconstructed_slip = stance_slip_metrics(
        timestamps,
        result.positions,
        root_rotations,
        foot_points_base,
        schedule,
    )
    path_direction = raw_root_positions[-1, :2] - raw_root_positions[0, :2]
    path_direction /= np.linalg.norm(path_direction)
    step_lengths = np.diff(result.anchor_positions_xy, axis=0) @ path_direction
    duration = float(timestamps[-1] - timestamps[0])
    reconstructed_speed = float(
        ((result.positions[-1, :2] - result.positions[0, :2]) @ path_direction)
        / duration
    )
    raw_speed = float(
        ((raw_root_positions[-1, :2] - raw_root_positions[0, :2]) @ path_direction)
        / duration
    )
    metrics: dict[str, object] = {
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "urdf_file": urdf.name,
        "urdf_sha256": _sha256(urdf),
        "frame_count": len(rows),
        "sample_rate_hz": float(1.0 / np.median(np.diff(timestamps))),
        "cycle_frames": int(schedule.cycle_frames),
        "cycle_seconds": float(schedule.cycle_frames * np.median(np.diff(timestamps))),
        "half_cycle_seconds": float(schedule.half_cycle_frames * np.median(np.diff(timestamps))),
        "cadence_steps_per_min": float(
            60.0 / (schedule.half_cycle_frames * np.median(np.diff(timestamps)))
        ),
        "switch_frames": [int(value) for value in schedule.switch_frames],
        "switch_times_seconds": [float(timestamps[value]) for value in schedule.switch_frames],
        "sole_points_in_ankle_roll_link": {
            "left": sole_points[0].tolist(),
            "right": sole_points[1].tolist(),
        },
        "forward_step_lengths_m": step_lengths.tolist(),
        "mean_forward_step_length_m": float(np.mean(step_lengths)),
        "std_forward_step_length_m": float(np.std(step_lengths)),
        "raw_root_speed_mps": raw_speed,
        "reconstructed_root_speed_mps": reconstructed_speed,
        "raw": raw_slip,
        "reconstructed": reconstructed_slip,
        "slip_rms_reduction_fraction": float(
            1.0
            - reconstructed_slip["stance_slip_rms_mps"]
            / raw_slip["stance_slip_rms_mps"]
        ),
        "ground_height_m": result.ground_height,
        "reconstructed_root_z_range_m": [
            float(result.positions[:, 2].min()),
            float(result.positions[:, 2].max()),
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "timestamp",
        *ROOT_COLUMNS,
        *CONTROLLED_JOINTS,
        "left_contact",
        "right_contact",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for frame, row in enumerate(rows):
            record = {"timestamp": f"{timestamps[frame]:.15g}"}
            for axis_index, axis in enumerate("xyz"):
                record[f"root_pos_{axis}"] = f"{result.positions[frame, axis_index]:.12g}"
            for axis in "xyzw":
                record[f"root_quat_{axis}"] = row[f"root_quat_{axis}"]
            for name in CONTROLLED_JOINTS:
                record[name] = row[name]
            record["left_contact"] = f"{result.contact_weights[frame, 0]:.9g}"
            record["right_contact"] = f"{result.contact_weights[frame, 1]:.9g}"
            writer.writerow(record)

    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    args = parser.parse_args()
    metrics = reconstruct(args.source, args.urdf, args.output, args.diagnostics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
