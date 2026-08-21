"""Derive X1 swing-foot clearance targets from a retargeted motion CSV.

This utility is intentionally independent of Isaac Gym. It evaluates the X1
URDF kinematic tree with the floating-base pose and the 12 leg joints in the
retargeted CSV, then writes ground-normalized left/right foot-bottom heights.
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def _rpy_matrix(values):
    roll, pitch, yaw = values
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _quat_matrix(values):
    x, y, z, w = values
    scale = 2.0 / (x * x + y * y + z * z + w * w)
    return np.asarray(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ]
    )


def _axis_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ]
    )


def _transform(rotation=None, translation=None):
    value = np.eye(4)
    if rotation is not None:
        value[:3, :3] = rotation
    if translation is not None:
        value[:3, 3] = translation
    return value


def _parse_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    by_child = {}
    limits = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib["type"]
        child = joint.find("child").attrib["link"]
        parent = joint.find("parent").attrib["link"]
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()]
        axis_node = joint.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_node is not None:
            axis = [float(value) for value in axis_node.attrib.get("xyz", "1 0 0").split()]
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        by_child[child] = (name, joint_type, parent, _transform(_rpy_matrix(rpy), xyz), axis)
    return by_child, limits


def _chain(by_child, link):
    result = []
    while link != "base_link":
        item = by_child[link]
        result.append(item)
        link = item[2]
    return tuple(reversed(result))


def build_clearance(source, urdf, output, start_time, end_time, foot_offset, ground_percentile):
    by_child, limits = _parse_urdf(urdf)
    chains = [_chain(by_child, link) for link in FOOT_LINKS]
    samples = []
    with Path(source).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = float(row["timestamp"])
            if timestamp < start_time or timestamp >= end_time:
                continue
            base = _transform(
                _quat_matrix([float(row[f"root_quat_{axis}"]) for axis in "xyzw"]),
                [float(row[f"root_pos_{axis}"]) for axis in "xyz"],
            )
            foot_z = []
            for chain in chains:
                pose = base.copy()
                for name, joint_type, _, origin, axis in chain:
                    pose = pose @ origin
                    if joint_type in ("revolute", "continuous"):
                        angle = float(row[name])
                        if name in limits:
                            angle = float(np.clip(angle, *limits[name]))
                        pose = pose @ _transform(_axis_matrix(axis, angle))
                foot_z.append(float(pose[2, 3] - foot_offset))
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
        f"left_peak={peaks[0]:.6f} right_peak={peaks[1]:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-time", type=float, default=0.6)
    parser.add_argument("--end-time", type=float, default=5.533333333333333)
    parser.add_argument("--foot-offset", type=float, default=0.041)
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
