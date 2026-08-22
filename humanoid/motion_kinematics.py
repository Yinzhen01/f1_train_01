"""Small NumPy-only kinematics helpers for offline X1 motion processing."""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JointKinematic:
    name: str
    joint_type: str
    parent: str
    origin: np.ndarray
    axis: np.ndarray


def rpy_matrix(values: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = values
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def quat_matrix(values: Sequence[float]) -> np.ndarray:
    x, y, z, w = values
    norm_squared = x * x + y * y + z * z + w * w
    if norm_squared <= np.finfo(np.float64).eps:
        raise ValueError("Quaternion norm must be positive")
    scale = 2.0 / norm_squared
    return np.asarray(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= np.linalg.norm(axis_array)
    x, y, z = axis_array
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def transform(rotation: np.ndarray | None = None, translation: Sequence[float] | None = None) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if rotation is not None:
        value[:3, :3] = rotation
    if translation is not None:
        value[:3, 3] = translation
    return value


def parse_urdf(urdf_path: str | Path) -> tuple[dict[str, JointKinematic], dict[str, tuple[float, float]]]:
    root = ET.parse(urdf_path).getroot()
    by_child: dict[str, JointKinematic] = {}
    limits: dict[str, tuple[float, float]] = {}
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
        by_child[child] = JointKinematic(
            name,
            joint_type,
            parent,
            transform(rpy_matrix(rpy), xyz),
            np.asarray(axis, dtype=np.float64),
        )
    return by_child, limits


def chain_to_link(by_child: Mapping[str, JointKinematic], link: str, root_link: str = "base_link") -> tuple[JointKinematic, ...]:
    result = []
    while link != root_link:
        if link not in by_child:
            raise ValueError(f"Link {link!r} is not connected to {root_link!r}")
        item = by_child[link]
        result.append(item)
        link = item.parent
    return tuple(reversed(result))


def evaluate_chain(
    chain: Sequence[JointKinematic],
    joint_positions: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    limits = limits or {}
    for joint in chain:
        pose = pose @ joint.origin
        if joint.joint_type in ("revolute", "continuous"):
            angle = float(joint_positions.get(joint.name, 0.0))
            if joint.name in limits:
                angle = float(np.clip(angle, *limits[joint.name]))
            pose = pose @ transform(axis_matrix(joint.axis, angle))
    return pose


def read_stl_bounds(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return mesh bounds for binary or ASCII STL without adding a dependency."""

    mesh_path = Path(path)
    data = mesh_path.read_bytes()
    vertices: np.ndarray
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + triangle_count * 50
    else:
        triangle_count = 0
        expected_size = -1
    if triangle_count and expected_size == len(data):
        vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
        for index in range(triangle_count):
            values = struct.unpack_from("<12fH", data, 84 + index * 50)
            vertices[index * 3 : index * 3 + 3] = np.asarray(values[3:12]).reshape(3, 3)
    else:
        text = data.decode("utf-8", errors="ignore")
        matches = re.findall(
            r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            text,
        )
        if not matches:
            raise ValueError(f"Unable to read STL vertices from {mesh_path}")
        vertices = np.asarray(matches, dtype=np.float64)
    return vertices.min(axis=0), vertices.max(axis=0)


def mesh_path_for_link(urdf_path: str | Path, link_name: str) -> Path:
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    link = next((node for node in root.findall("link") if node.attrib.get("name") == link_name), None)
    if link is None:
        raise ValueError(f"URDF does not contain link {link_name!r}")
    geometry = link.find("collision/geometry/mesh")
    if geometry is None:
        geometry = link.find("visual/geometry/mesh")
    if geometry is None or "filename" not in geometry.attrib:
        raise ValueError(f"Link {link_name!r} does not contain a mesh geometry")
    return (urdf_path.parent / geometry.attrib["filename"]).resolve()


def sole_center_from_mesh(
    urdf_path: str | Path,
    link_name: str,
    zero_pose: np.ndarray,
) -> np.ndarray:
    """Find the sole-center material point using mesh bounds and zero-pose axes."""

    lower, upper = read_stl_bounds(mesh_path_for_link(urdf_path, link_name))
    point = 0.5 * (lower + upper)
    local_axes_in_base = zero_pose[:3, :3]
    vertical_axis = int(np.argmax(np.abs(local_axes_in_base[2, :])))
    # If the positive local axis points upward, the lower mesh bound is the sole;
    # otherwise the upper bound is the sole. This handles mirrored feet.
    point[vertical_axis] = (
        lower[vertical_axis]
        if local_axes_in_base[2, vertical_axis] > 0.0
        else upper[vertical_axis]
    )
    return point
