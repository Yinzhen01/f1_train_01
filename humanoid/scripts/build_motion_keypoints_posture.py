"""Derive foot sole keypoints and base posture from an X1 motion CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from humanoid.motion_kinematics import (
    chain_to_link,
    evaluate_chain,
    mesh_path_for_link,
    parse_urdf,
    quat_matrix,
    read_stl_bounds,
)


FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def sole_keypoints(urdf, link, zero_pose, inset=0.01):
    """Return heel/toe material points on the sole in the foot-link frame."""

    lower, upper = read_stl_bounds(mesh_path_for_link(urdf, link))
    rotation = zero_pose[:3, :3]
    vertical_axis = int(np.argmax(np.abs(rotation[2, :])))
    forward_axis = int(np.argmax(np.abs(rotation[0, :])))
    if vertical_axis == forward_axis:
        raise ValueError(f"Unable to separate vertical/forward axes for {link}")

    center = 0.5 * (lower + upper)
    sole = lower[vertical_axis] if rotation[2, vertical_axis] > 0.0 else upper[vertical_axis]
    forward_min = lower[forward_axis] + inset
    forward_max = upper[forward_axis] - inset
    if forward_min >= forward_max:
        raise ValueError(f"Foot-keypoint inset is too large for {link}")

    heel = center.copy()
    toe = center.copy()
    heel[vertical_axis] = sole
    toe[vertical_axis] = sole
    if rotation[0, forward_axis] > 0.0:
        heel[forward_axis], toe[forward_axis] = forward_min, forward_max
    else:
        heel[forward_axis], toe[forward_axis] = forward_max, forward_min
    return heel, toe


def build_references(source, urdf, keypoint_output, posture_output, inset=0.01):
    by_child, limits = parse_urdf(urdf)
    chains = [chain_to_link(by_child, link) for link in FOOT_LINKS]
    zero_poses = [evaluate_chain(chain, {}, limits) for chain in chains]
    keypoints = [
        sole_keypoints(urdf, link, pose, inset=inset)
        for link, pose in zip(FOOT_LINKS, zero_poses)
    ]

    keypoint_rows = []
    posture_rows = []
    with Path(source).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = float(row["timestamp"])
            joint_positions = {name: float(row[name]) for name in limits if name in row}
            values = [timestamp]
            for chain, (heel, toe) in zip(chains, keypoints):
                pose = evaluate_chain(chain, joint_positions, limits)
                for point in (heel, toe):
                    position = pose @ np.asarray([*point, 1.0])
                    values.extend((float(position[0]), float(position[1])))
            keypoint_rows.append(values)

            root_rotation = quat_matrix(
                [float(row[f"root_quat_{axis}"]) for axis in "xyzw"]
            )
            projected_gravity = root_rotation.T @ np.asarray([0.0, 0.0, -1.0])
            posture_rows.append(
                (timestamp, float(projected_gravity[0]), float(projected_gravity[1]))
            )

    keypoint_output = Path(keypoint_output)
    keypoint_output.parent.mkdir(parents=True, exist_ok=True)
    with keypoint_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "left_heel_x",
                "left_heel_y",
                "left_toe_x",
                "left_toe_y",
                "right_heel_x",
                "right_heel_y",
                "right_toe_x",
                "right_toe_y",
            )
        )
        writer.writerows(keypoint_rows)

    posture_output = Path(posture_output)
    posture_output.parent.mkdir(parents=True, exist_ok=True)
    with posture_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("timestamp", "projected_gravity_x", "projected_gravity_y"))
        writer.writerows(posture_rows)

    print(
        f"keypoints={keypoint_output} posture={posture_output} frames={len(keypoint_rows)} "
        f"offsets={np.asarray(keypoints).round(9).tolist()}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--keypoint-output", required=True)
    parser.add_argument("--posture-output", required=True)
    parser.add_argument("--inset", type=float, default=0.01)
    args = parser.parse_args()
    build_references(
        args.source,
        args.urdf,
        args.keypoint_output,
        args.posture_output,
        inset=args.inset,
    )


if __name__ == "__main__":
    main()
