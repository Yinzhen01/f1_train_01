"""Derive phase-aligned stance geometry targets from a retargeted X1 motion.

The generated targets are expressed in the robot base frame.  Foot heading is
the projected local +Z axis of each ankle-roll link, matching the diagnostic
definition used by ``play_gm.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from humanoid.motion_kinematics import chain_to_link, evaluate_chain, parse_urdf


FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
KNEE_LINKS = ("left_knee_pitch_link", "right_knee_pitch_link")
FOOT_FORWARD_LOCAL = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


def _wrap_to_pi(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def build_geometry(source: Path, urdf: Path, output: Path, start_time: float, end_time: float) -> None:
    by_child, limits = parse_urdf(urdf)
    foot_chains = [chain_to_link(by_child, link) for link in FOOT_LINKS]
    knee_chains = [chain_to_link(by_child, link) for link in KNEE_LINKS]

    samples = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        timestamp = float(row["timestamp"])
        if timestamp < start_time or timestamp >= end_time:
            continue
        joint_positions = {name: float(row[name]) for name in limits if name in row}
        foot_poses = [
            evaluate_chain(chain, joint_positions, limits) for chain in foot_chains
        ]
        knee_poses = [
            evaluate_chain(chain, joint_positions, limits) for chain in knee_chains
        ]
        foot_lateral = abs(float(foot_poses[1][1, 3] - foot_poses[0][1, 3]))
        knee_lateral = abs(float(knee_poses[1][1, 3] - knee_poses[0][1, 3]))
        headings = []
        for pose in foot_poses:
            forward = pose[:3, :3] @ FOOT_FORWARD_LOCAL
            headings.append(_wrap_to_pi(np.arctan2(forward[1], forward[0])))
        samples.append((timestamp, foot_lateral, knee_lateral, *headings))

    if len(samples) < 2:
        raise ValueError("Selected motion range must contain at least two frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "foot_lateral_distance",
                "knee_lateral_distance",
                "left_foot_heading",
                "right_foot_heading",
            )
        )
        for sample in samples:
            writer.writerow(tuple(f"{value:.12g}" for value in sample))

    values = np.asarray(samples, dtype=np.float64)
    print(
        f"wrote={output} frames={len(samples)} "
        f"foot_lateral_min_mean_max={values[:, 1].min():.6f},"
        f"{values[:, 1].mean():.6f},{values[:, 1].max():.6f} "
        f"knee_lateral_min_mean_max={values[:, 2].min():.6f},"
        f"{values[:, 2].mean():.6f},{values[:, 2].max():.6f} "
        f"heading_abs_max_deg={np.degrees(np.abs(values[:, 3:])).max():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-time", type=float, default=0.6)
    parser.add_argument("--end-time", type=float, default=5.466666666666667)
    args = parser.parse_args()
    build_geometry(args.source, args.urdf, args.output, args.start_time, args.end_time)


if __name__ == "__main__":
    main()
