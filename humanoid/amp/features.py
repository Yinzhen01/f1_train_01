"""Shared demo/policy SI-unit feature contract; quaternions are explicit xyzw.

Derivatives use causal 100 Hz differences on both sides. Dataset frame zero is
not sampled; policy callers must prime a pre-step state after every reset.
"""
from dataclasses import dataclass
from typing import Tuple
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def name_indices(source, target):
    source, target = list(source), list(target)
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("Duplicate joint/body names")
    if not set(target).issubset(source):
        raise ValueError("Missing named fields: %s" % sorted(set(target) - set(source)))
    return [source.index(n) for n in target]


@dataclass(frozen=True)
class FeatureSpec:
    joint_names: Tuple[str, ...]
    body_names: Tuple[str, ...]
    root_body: str = "base_link"
    fps: int = 100
    history: int = 10

    def __post_init__(self):
        if len(self.joint_names) != 12 or len(self.body_names) != 4:
            raise ValueError("F1 v1 contract requires 12 joints and 4 key bodies")
        name_indices(self.joint_names, self.joint_names)
        name_indices(self.body_names, self.body_names)
        if self.fps != 100 or self.history != 10:
            raise ValueError("This contract is strictly 10 frames at 100 Hz")

    @property
    def dim(self):
        return 3 + 2 * len(self.joint_names) + 3 * len(self.body_names)

    def description(self):
        return dict(version="f1-amp-v1-causal-100hz", fps=self.fps, history=self.history,
                    span_s=(self.history - 1) / self.fps, per_frame_dim=self.dim,
                    flat_dim=self.dim * self.history, root_body=self.root_body,
                    joint_names=list(self.joint_names), body_names=list(self.body_names),
                    fields=[dict(name="base_ang_vel_b", width=3, unit="rad/s"),
                            dict(name="joint_pos", width=12, unit="rad"),
                            dict(name="joint_vel", width=12, unit="rad/s"),
                            dict(name="key_body_pos_b", width=12, unit="m")],
                    frame="root body; rigid link origins, not COM or sole centers",
                    derivative="backward difference; SO(3) shortest relative rotation",
                    quaternion_internal="xyzw", quaternion_source="wxyz")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.description(), sort_keys=True).encode()).hexdigest()


def normalize_quat(q):
    if q.shape[-1] != 4 or not torch.isfinite(q).all():
        raise ValueError("Nonfinite or malformed quaternion")
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if (norm < 1e-8).any():
        raise ValueError("Zero quaternion")
    return q / norm


def wxyz_to_xyzw(q):
    return normalize_quat(q[..., [1, 2, 3, 0]])


def xyzw_to_wxyz(q):
    return normalize_quat(q)[..., [3, 0, 1, 2]]


def quat_mul(a, b):
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3]
    xyz = xyz + torch.linalg.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def quat_conj(q):
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_rotate(q, v):
    q = normalize_quat(q)
    xyz = q[..., :3].expand_as(v)
    cross = 2 * torch.linalg.cross(xyz, v, dim=-1)
    return v + q[..., 3:] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def angular_velocity_body(previous_xyzw, current_xyzw, dt=.01):
    # The relative rotation axis is invariant under its own rotation, hence
    # log(R_prev^T R_cur) is also the interval angular velocity in current body.
    prev, cur = normalize_quat(previous_xyzw), normalize_quat(current_xyzw)
    delta = normalize_quat(quat_mul(quat_conj(prev), cur))
    delta = torch.where(delta[..., 3:] < 0, -delta, delta)
    s = torch.linalg.vector_norm(delta[..., :3], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, delta[..., 3:].clamp_min(0))
    factor = torch.where(s > 1e-8, angle / s.clamp_min(1e-8), 2 * torch.ones_like(s))
    return delta[..., :3] * factor / dt


def build_features(spec, angular_b, joint_pos, joint_vel, joint_names,
                   key_positions_b, body_names):
    ji, bi = name_indices(joint_names, spec.joint_names), name_indices(body_names, spec.body_names)
    batch_shape = joint_pos.shape[:-1]
    if angular_b.shape != batch_shape + (3,) or joint_vel.shape != joint_pos.shape:
        raise ValueError("Angular/joint feature shape mismatch")
    if joint_pos.shape[-1] != len(joint_names) or key_positions_b.shape != batch_shape + (len(body_names), 3):
        raise ValueError("Named field width mismatch")
    result = torch.cat((angular_b, joint_pos[..., ji], joint_vel[..., ji],
                        key_positions_b[..., bi, :].reshape(batch_shape + (-1,))), dim=-1)
    if result.shape[-1] != spec.dim or not torch.isfinite(result).all():
        raise ValueError("Invalid AMP features")
    return result


def from_world_state(spec, root_quat_xyzw, root_pos_w, angular_b, joint_pos,
                     joint_vel, joint_names, key_positions_w, body_names):
    relative_b = quat_rotate(quat_conj(normalize_quat(root_quat_xyzw)).unsqueeze(-2),
                             key_positions_w - root_pos_w.unsqueeze(-2))
    return build_features(spec, angular_b, joint_pos, joint_vel, joint_names, relative_b, body_names)


class URDFKinematics:
    """Read-only FK on training URDF link frames; never imports/modifies dynamics."""
    def __init__(self, path, spec):
        self.path, self.spec = str(path), spec
        with open(path, "rb") as stream:
            raw = stream.read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.lf_sha256 = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        tree = ET.fromstring(raw)
        bodies = [n.attrib["name"] for n in tree.findall("link")]
        name_indices(bodies, (spec.root_body,) + spec.body_names)
        self.joints = []
        limits = {}
        for j in tree.findall("joint"):
            kind, name = j.attrib["type"], j.attrib["name"]
            if kind not in ("fixed", "revolute"):
                raise ValueError("Unsupported URDF joint type " + kind)
            if j.find("mimic") is not None:
                raise ValueError("Mimic joints require an explicit kinematic adapter")
            origin = j.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            axis = np.fromstring(j.find("axis").get("xyz", "1 0 0"), sep=" ") if kind != "fixed" else np.zeros(3)
            if kind != "fixed":
                axis /= np.linalg.norm(axis)
                limit = j.find("limit")
                limits[name] = [float(limit.get(k)) for k in ("lower", "upper", "velocity")]
            self.joints.append(dict(name=name, kind=kind, parent=j.find("parent").get("link"),
                                    child=j.find("child").get("link"), xyz=xyz,
                                    rotation=Rotation.from_euler("xyz", rpy).as_matrix(), axis=axis))
        if set(limits) != set(spec.joint_names):
            raise ValueError("Training model does not have exactly the named 12 active joints")
        self.limits = np.array([limits[n] for n in spec.joint_names])

    def key_positions(self, joint_pos, joint_names):
        poses = self.link_poses(joint_pos, joint_names)
        return np.stack([poses[b][1] for b in self.spec.body_names], axis=1)

    def link_poses(self, joint_pos, joint_names):
        """Root-relative rotations and positions, also used for reset geometry."""
        q = np.asarray(joint_pos, dtype=np.float64)
        qi = name_indices(joint_names, self.spec.joint_names)
        q = q[:, qi]
        count = len(q)
        poses = {self.spec.root_body: (np.broadcast_to(np.eye(3), (count, 3, 3)), np.zeros((count, 3)))}
        remaining = list(self.joints)
        while remaining:
            progressed = False
            for j in list(remaining):
                if j["parent"] not in poses:
                    continue
                parent_r, parent_p = poses[j["parent"]]
                r = np.broadcast_to(j["rotation"], (count, 3, 3))
                if j["kind"] != "fixed":
                    angle = q[:, self.spec.joint_names.index(j["name"])]
                    r = r @ Rotation.from_rotvec(angle[:, None] * j["axis"]).as_matrix()
                poses[j["child"]] = (parent_r @ r, parent_p + np.einsum("nij,j->ni", parent_r, j["xyz"]))
                remaining.remove(j)
                progressed = True
            if not progressed:
                raise ValueError("URDF has disconnected/cyclic/unsupported root chain")
        return poses
