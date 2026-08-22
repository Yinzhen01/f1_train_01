"""Derive X1 swing-foot clearance targets from a retargeted motion CSV.

The sole material point is derived from the ankle-roll mesh and its zero-pose
orientation. This avoids assuming that a particular foot-link local axis is
vertical, which is not true for the X1 ankle-roll links.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from humanoid.motion_kinematics import (
    chain_to_link,
    evaluate_chain,
    parse_urdf,
    quat_matrix,
    sole_center_from_mesh,
    transform,
)


FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def build_clearance(source, urdf, output, start_time, end_time, foot_offset, ground_percentile):
    by_child, limits = parse_urdf(urdf)
    chains = [chain_to_link(by_child, link) for link in FOOT_LINKS]
    zero_poses = [evaluate_chain(chain, {}, limits) for chain in chains]
    sole_points = [
        sole_center_from_mesh(urdf, link, pose)
        for link, pose in zip(FOOT_LINKS, zero_poses)
    ]
    samples = []
    with Path(source).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = float(row["timestamp"])
            if timestamp < start_time or timestamp >= end_time:
                continue
            base = transform(
                quat_matrix([float(row[f"root_quat_{axis}"]) for axis in "xyzw"]),
                [float(row[f"root_pos_{axis}"]) for axis in "xyz"],
            )
            joint_positions = {name: float(row[name]) for name in limits if name in row}
            foot_z = []
            for chain, sole_point in zip(chains, sole_points):
                pose = base @ evaluate_chain(chain, joint_positions, limits)
                point = pose @ np.asarray([*sole_point, 1.0])
                z_value = float(point[2])
                if foot_offset is not None:
                    # Explicit compatibility mode for reproducing the old
                    # ankle-origin-minus-world-Z approximation.
                    z_value = float(pose[2, 3] - foot_offset)
                foot_z.append(z_value)
            samples.append((timestamp, *foot_z))

    if len(samples) < 2:
        raise ValueError("Selected motion range must contain at least two frames")
    values = np.asarray(samples, dtype=np.float64)
    ground = float(np.percentile(values[:, 1:], ground_percentile))
    clearance = np.maximum(values[:, 1:] - ground, 0.0)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("timestamp", "left_foot_clearance", "right_foot_clearance"))
        for timestamp, (left, right) in zip(values[:, 0], clearance):
            writer.writerow((f"{timestamp:.15g}", f"{left:.9f}", f"{right:.9f}"))

    peaks = np.max(clearance, axis=0)
    print(
        f"wrote={output} frames={len(samples)} ground={ground:.6f} "
        f"left_peak={peaks[0]:.6f} right_peak={peaks[1]:.6f} "
        f"sole_left={np.asarray(sole_points[0]).round(6).tolist()} "
        f"sole_right={np.asarray(sole_points[1]).round(6).tolist()}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-time", type=float, default=0.6)
    parser.add_argument("--end-time", type=float, default=5.533333333333333)
    parser.add_argument(
        "--foot-offset",
        type=float,
        default=None,
        help="legacy world-Z ankle offset; omit to derive the sole point from the mesh",
    )
    parser.add_argument("--ground-percentile", type=float, default=2.0)
    args = parser.parse_args()
    build_clearance(
        args.source,
        args.urdf,
        args.output,
        args.start_time,
        args.end_time,
        args.foot_offset,
        args.ground_percentile,
    )


if __name__ == "__main__":
    main()
