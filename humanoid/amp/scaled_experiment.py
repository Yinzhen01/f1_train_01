"""A hash-bound single-clip AMP experiment; old ten-clip gates stay intact.

The source remains training_ready=false (not physically accepted). Admission to
an explicitly authorized simulation experiment is distinct from source/hardware
acceptance and requires static checks plus a matching real-simulator smoke.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .dataset import Clip, load_config, sha256
from .features import FeatureSpec, URDFKinematics
from .quality import derivative_metrics, support_proxy


def canonical_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_runtime_timing(control_dt, physics_dt, decimation):
    """Accept native float32 rounding, never a different control/physics rate."""
    if (decimation != 10 or
            not math.isclose(control_dt, .01, rel_tol=1e-6, abs_tol=1e-10) or
            not math.isclose(physics_dt, .001, rel_tol=1e-6, abs_tol=1e-10)):
        raise ValueError("AMP requires 100 Hz control / 1 kHz physics / decimation 10; "
                         "got control_dt=%.17g physics_dt=%.17g decimation=%s" %
                         (control_dt, physics_dt, decimation))
    return True


class ScaledExperiment:
    def __init__(self, repo, config_path=None):
        self.repo = Path(repo)
        self.path = Path(config_path) if config_path else self.repo/"configs/amp/lafan_walk02_s092.json"
        self.cfg = json.loads(self.path.read_text(encoding="utf-8"))
        self.config_sha = canonical_sha(self.path)
        cfg = self.cfg
        if (cfg["source_clip_id"] != "LAFAN_WALK_02" or
                cfg["candidate_id"] != "LAFAN_WALK_02_LEFT_HIP_ROLL_S092" or cfg["training_ready"] is not False):
            raise ValueError("Unexpected single-clip experiment identity/admission")
        base = load_config(self.repo/cfg["protocol"])
        self.spec = FeatureSpec(tuple(base["joint_names"]), tuple(base["body_names"]), base["root_body"])
        self.kinematics = URDFKinematics(self.repo/base["urdf"], self.spec)
        self.motion_path = self.repo/cfg["motion"]
        if sha256(self.motion_path) != cfg["motion_sha256"]:
            raise ValueError("Scaled motion hash mismatch")
        with np.load(self.motion_path, allow_pickle=False) as z:
            meta = json.loads(z["metadata_json"].item())
        transform = meta["proportional_scaling"]
        if (meta["parent_id"] != cfg["source_clip_id"] or transform["factor"] != .92 or
                transform["joint"] != "left_hip_roll_joint" or transform["parent_npz_sha256"] != cfg["parent_sha256"]):
            raise ValueError("Wrong scaling provenance")
        record = dict(id=cfg["candidate_id"], source_sha256=meta["source_sha256"], split="train_candidate",
                      npz_sha256=cfg["motion_sha256"], training_ready=False, frames=600, retarget_fps=100)
        self.clip = Clip(self.motion_path, record, self.spec, self.kinematics)
        c = self.clip
        ix = 7+c.all_joint_names.index("left_hip_roll_joint")
        expected = c.arrays["parent_qpos"].copy(); expected[:, ix] *= .92
        if not np.array_equal(expected, c.arrays["qpos"]):
            raise ValueError("Unexpected edits beyond left hip roll scaling")
        lo, hi, vmax = self.kinematics.limits.T
        q = c.joint_pos; velocity = np.diff(q, axis=0)*100
        if np.any(q < lo-1e-6) or np.any(q > hi+1e-6) or np.any(np.abs(velocity) > vmax+1e-6):
            raise ValueError("Reference exceeds training hard joint/velocity limits")
        if (np.sum(c.arrays["qpos"][1:, 3:7]*c.arrays["qpos"][:-1, 3:7], axis=1) < 0).any():
            raise ValueError("Quaternion sign discontinuity")
        x = c.features[1:].double()
        self.mean = x.mean(0).float()
        self.std = x.std(0, unbiased=False).clamp_min(cfg["normalizer_std_floor"]).float()
        self.windows = c.features.unfold(0, self.spec.history, 1).permute(0, 2, 1)[1:].contiguous()
        assert self.windows.shape == (590, 10, 39)
        self.rng = np.random.default_rng(cfg["seed"])
        self.diagnostics = dict(clip_id=c.id, frames=c.frames, fps=100,
            angle_violations=0, velocity_violations=0, joint_derivatives=derivative_metrics(q),
            root_z_derivatives=derivative_metrics(c.arrays["qpos"][:, 2]),
            root_z_lift_derivatives=derivative_metrics(c.arrays["root_z_lift"]),
            source_mesh_min_height_mm=float(c.arrays["sole_min_height"].min()*1000),
            source_support_proxy=support_proxy(c.arrays["sole_center"], c.arrays["source_support"]),
            legal_window_starts=[1, 590], window_count=590, fit_clip_ids=[c.id],
            training_ready=False, dynamic_feasibility="not established")
        if self.diagnostics["source_mesh_min_height_mm"] < 0:
            raise ValueError("Source foot geometry penetrates floor")

    def sample(self, batch_size):
        if batch_size < 1:
            raise ValueError("Empty batch")
        starts = self.rng.integers(0, len(self.windows), size=batch_size)
        return self.windows[torch.as_tensor(starts)].clone(), [dict(clip_id=self.clip.id, start_frame=int(i+1), end_frame=int(i+10)) for i in starts]

    def identity(self):
        return dict(experiment=self.cfg["experiment"], config_sha256=self.config_sha,
                    motion_sha256=sha256(self.motion_path), urdf_lf_sha256=self.kinematics.lf_sha256,
                    feature_fingerprint=self.spec.fingerprint)


def implementation_fingerprint(repo):
    repo = Path(repo)
    paths = list((repo/"humanoid").rglob("*.py")) + list((repo/"configs").rglob("*.json"))
    paths += [p for p in (repo/"resources/robots/x1").rglob("*") if p.is_file()]
    digest = hashlib.sha256()
    for p in sorted(paths, key=lambda item: item.relative_to(repo).as_posix()):
        content = p.read_bytes()
        if p.suffix.lower() in (".py", ".json", ".xml", ".urdf", ".yaml", ".yml"):
            content = content.replace(b"\r\n", b"\n")
        digest.update(p.relative_to(repo).as_posix().encode()+b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def validate_cloud_smoke(experiment, certificate, fingerprint):
    """Fail closed: a data flag alone never admits a long GPU run."""
    if certificate.get("identity") != experiment.identity() or certificate.get("implementation_fingerprint") != fingerprint:
        raise ValueError("Smoke identity/commit does not match this experiment")
    required = ("physics_dt_verified", "body_frames_verified", "reset_history_verified",
                "nonzero_style_reward", "actor_updated", "discriminator_updated", "all_finite", "complete")
    if any(certificate.get(key) is not True for key in required):
        raise ValueError("Real-simulator smoke gates incomplete")
    if (certificate.get("num_envs"), certificate.get("updates")) != (32, 10):
        raise ValueError("Wrong smoke budget")
    if certificate.get("valid_windows", 0) <= 0:
        raise ValueError("No valid AMP history observed")
    if experiment.cfg.get("recovery"):
        if (certificate.get("no_dr_configuration_verified") is not True or
                certificate.get("smoke_evaluation_verified") is not True or
                certificate.get("rsi_reset_count", 0) < 32 or certificate.get("replay_window_count", 0) < 1):
            raise ValueError("Recovery smoke must verify no-DR, RSI and policy replay")
    return True
