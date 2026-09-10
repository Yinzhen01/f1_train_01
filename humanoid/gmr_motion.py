"""Validated, name-mapped, clamped (never wrapped) motion sampling."""
import hashlib
import json

import numpy as np
import torch


class GMRMotion:
    temporal_keys = ("dof_pos", "dof_vel", "root_pos", "root_quat", "root_vel", "root_ang_vel",
                     "foot_pos_base", "foot_quat_base", "foot_contact")

    def __init__(self, path, dof_names, device="cpu", expected_sha256=None):
        if expected_sha256:
            with open(path, "rb") as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            if digest != expected_sha256:
                raise ValueError("GMR reference SHA256 mismatch")
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata_json"]))
            names = archive["joint_names"].astype(str).tolist()
            if len(names) != 12 or len(set(names)) != 12 or set(names) != set(dof_names) or len(dof_names) != 12:
                raise ValueError("GMR reference and robot require the same 12 named joints")
            order = [names.index(name) for name in dof_names]
            self.fps = float(archive["fps"])
            self.frames = len(archive["dof_pos"])
            if self.frames < 2 or not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Invalid reference length or FPS")
            if self.metadata.get("cyclic") is not False or self.metadata.get("quaternion_order") != "xyzw":
                raise ValueError("Only noncyclic xyzw reference files are supported")
            self.duration = (self.frames - 1) / self.fps
            shapes = {"dof_pos": (12,), "dof_vel": (12,), "root_pos": (3,), "root_quat": (4,),
                      "root_vel": (3,), "root_ang_vel": (3,), "foot_pos_base": (2, 3),
                      "foot_quat_base": (2, 4), "foot_contact": (2,)}
            self.data = {}
            for key in self.temporal_keys:
                value = archive[key].copy()
                if value.shape != (self.frames,) + shapes[key] or not np.isfinite(value).all():
                    raise ValueError("Invalid reference field " + key)
                if key.startswith("dof_"):
                    value = value[:, order]
                if "quat" in key:
                    norm = np.linalg.norm(value, axis=-1, keepdims=True)
                    if np.max(np.abs(norm - 1)) > 1e-4:
                        raise ValueError("Non-unit quaternion in " + key)
                    value /= norm
                    for frame in range(1, self.frames):
                        sign = np.where(np.sum(value[frame] * value[frame-1], axis=-1, keepdims=True) < 0, -1, 1)
                        value[frame] *= sign
                self.data[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
            self.lower = torch.as_tensor(archive["lower"][order], dtype=torch.float32, device=device)
            self.upper = torch.as_tensor(archive["upper"][order], dtype=torch.float32, device=device)
            if torch.any(self.data["dof_pos"] < self.lower - 1e-6) or torch.any(self.data["dof_pos"] > self.upper + 1e-6):
                raise ValueError("Reference exceeds exported joint bounds")
            if torch.any(self.data["foot_contact"] < 0) or torch.any(self.data["foot_contact"] > 1):
                raise ValueError("Invalid contact probabilities")
            self.sole_local = torch.as_tensor(archive["sole_local"], dtype=torch.float32, device=device)
            self.sole_normal_local = torch.as_tensor(archive["sole_normal_local"], dtype=torch.float32, device=device)

    def sample(self, times):
        position = torch.clamp(times * self.fps, 0., self.frames - 1.)
        lo = position.long()
        hi = torch.clamp(lo + 1, max=self.frames - 1)
        fraction = position - lo
        result = {}
        for key, values in self.data.items():
            alpha = fraction.reshape(fraction.shape + (1,) * (values.ndim - 1))
            value = values[lo] * (1 - alpha) + values[hi] * alpha
            if "quat" in key:
                value = value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-8)
            result[key] = value
        return result
