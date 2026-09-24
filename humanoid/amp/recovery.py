"""Opt-in AMP recovery mechanisms. No simulator imports or asset writes."""
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def centered_velocity_reward(velocity_xy, command_xy, sigma=5.):
    """Original walking kernel, zero at rest and one at the desired velocity.

    Valid for this nonzero fixed forward command, not a generic stand command.
    No target gait phase or target joint angle appears in the task reward.
    """
    weights = velocity_xy.new_tensor([1., 10.])
    stationary = torch.exp(-sigma * (command_xy.square() * weights).sum(-1))
    if (stationary > .99).any():
        raise ValueError("Centered walking reward requires a nonzero walking command")
    tracking = torch.exp(-sigma * ((velocity_xy-command_xy).square() * weights).sum(-1))
    return (tracking-stationary)/(1.-stationary)


def foot_collision_vertices(urdf):
    """Read exact training-foot collision meshes including origin and scale."""
    urdf = Path(urdf)
    root = ET.parse(urdf).getroot()
    result = {}
    for name in FOOT_NAMES:
        collisions = root.findall("link[@name='%s']/collision" % name)
        if not collisions:
            raise ValueError("Missing training-foot collision geometry")
        meshes = []
        for collision in collisions:
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Expected training-foot STL collision mesh")
            path = (urdf.parent/mesh.get("filename")).resolve()
            raw = path.read_bytes()
            count = struct.unpack_from("<I", raw, 80)[0]
            if len(raw) != 84+50*count:
                raise ValueError("Expected binary STL")
            dtype = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            vertices = np.unique(np.frombuffer(raw, dtype=dtype, offset=84)["v"].reshape(-1, 3), axis=0).astype(float)
            vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = collision.find("origin")
            if origin is not None:
                r = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                vertices = vertices @ r.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            meshes.append(vertices)
        result[name] = np.concatenate(meshes)
    return result


def reference_initial_states(experiment, clearance=.005):
    """Training-model RSI states, never edited demonstration features.

    Source root-Z/lift derivatives are NOT injected as reset vertical velocity.
    Root height is projected once onto the training collision mesh at reset.
    No contact labels, fake AMP history or kinematic motion playback is used.
    """
    if not 0 < clearance <= .02:
        raise ValueError("Invalid reset clearance")
    c = experiment.clip
    q = c.joint_pos[1:].copy()
    poses = experiment.kinematics.link_poses(q, experiment.spec.joint_names)
    quat = c.arrays["qpos"][1:, [4, 5, 6, 3]].copy()
    root_r = Rotation.from_quat(quat).as_matrix()
    vertices = foot_collision_vertices(experiment.kinematics.path)
    min_heights = []
    for name in FOOT_NAMES:
        r, p = poses[name]
        world_r = root_r @ r
        world_p = np.einsum("nij,nj->ni", root_r, p)
        # Only the z projection is required, so avoid [frames,vertices,3].
        heights = world_r[:, 2, :] @ vertices[name].T + world_p[:, 2, None]
        min_heights.append(heights.min(axis=1))
    height = clearance-np.minimum(*min_heights)
    root = np.zeros((len(q), 13), dtype=np.float32)
    root[:, 2] = height
    root[:, 3:7] = quat
    root[:, 7:9] = np.diff(c.arrays["qpos"][:, :2], axis=0)*100.
    root[:, 10:13] = np.einsum("nij,nj->ni", root_r, c.features[1:, :3].numpy())
    velocity = c.features[1:, 15:27].numpy().copy()
    if not all(np.isfinite(x).all() for x in (root, q, velocity)) or np.any(height < .3):
        raise ValueError("Invalid reference reset states")
    return dict(root=torch.from_numpy(root), q=torch.tensor(q, dtype=torch.float32),
                qd=torch.tensor(velocity, dtype=torch.float32),
                frame_ids=torch.arange(1, c.frames), clearance_m=float(clearance))


def static_negative_windows(demo):
    """Synthetic held-pose negatives, explicitly not observed policy samples."""
    if demo.ndim != 3 or demo.shape[1:] != (10, 39):
        raise ValueError("Expected the unchanged 10x39 feature protocol")
    result = demo[:, -1:, :].expand(-1, 10, -1).clone()
    result[:, :, :3] = 0.       # body angular velocity
    result[:, :, 15:27] = 0.   # joint velocity
    return result


class PolicyReplay:
    """Bounded ring of valid, detached, real policy windows; no demo mixing."""
    def __init__(self, capacity=50000):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Invalid replay capacity")
        self.capacity, self.count, self.cursor = capacity, 0, 0
        self.buffer = None

    def add(self, windows):
        if windows.ndim != 3 or windows.shape[1:] != (10, 39) or not torch.isfinite(windows).all():
            raise ValueError("Replay only accepts complete finite 10x39 histories")
        if not len(windows):
            return
        if self.buffer is None:
            self.buffer = torch.empty((self.capacity, 10, 39), dtype=windows.dtype, device=windows.device)
        if windows.device != self.buffer.device or windows.dtype != self.buffer.dtype:
            raise ValueError("Replay dtype/device mismatch")
        rows = windows.detach()[-self.capacity:]
        first = min(len(rows), self.capacity-self.cursor)
        self.buffer[self.cursor:self.cursor+first].copy_(rows[:first])
        if first < len(rows):
            self.buffer[:len(rows)-first].copy_(rows[first:])
        self.cursor = (self.cursor+len(rows)) % self.capacity
        self.count = min(self.capacity, self.count+len(rows))

    def sample(self, size, generator):
        if self.count == 0 or size < 1:
            raise ValueError("Cannot sample empty replay")
        ids = torch.randint(self.count, (size,), generator=generator, device=self.buffer.device)
        return self.buffer[ids].detach().clone()

    def state_dict(self):
        return dict(capacity=self.capacity, count=self.count, cursor=self.cursor,
                    windows=None if self.buffer is None else self.buffer[:self.count].detach().clone())

    def load_state_dict(self, state, device="cpu"):
        if not isinstance(state, dict) or state["capacity"] != self.capacity:
            raise ValueError("Replay checkpoint capacity mismatch")
        count, cursor, windows = state["count"], state["cursor"], state["windows"]
        if not 0 <= count <= self.capacity or not 0 <= cursor < self.capacity:
            raise ValueError("Invalid replay counters")
        if count == 0 or windows is None or windows.shape != (count, 10, 39) or not torch.isfinite(windows).all():
            raise ValueError("Missing/nonfinite replay windows")
        if count < self.capacity and cursor != count:
            raise ValueError("Partial replay cursor mismatch")
        self.buffer = torch.empty((self.capacity, 10, 39), dtype=windows.dtype, device=device)
        self.buffer[:count].copy_(windows.detach().to(device))
        self.count, self.cursor = count, cursor
