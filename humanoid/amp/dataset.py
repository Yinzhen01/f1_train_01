"""Hash-bound, noncyclic multiclips; no train/holdout normalization leakage."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .features import (FeatureSpec, URDFKinematics, angular_velocity_body,
                       build_features, name_indices, wxyz_to_xyzw)

TRAIN_IDS = tuple("LAFAN_WALK_%02d" % i for i in (1, 2, 3, 4))
RESERVE_IDS = ("LAFAN_WALK_08", "LAFAN_WALK_09", "LAFAN_WALK_10")
HOLDOUT_IDS = ("LAFAN_WALK_15",)
STAND_IDS = ("R2_009", "R2_010")
ALL_IDS = TRAIN_IDS + RESERVE_IDS + HOLDOUT_IDS + STAND_IDS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, required in (("allowed_train_ids", TRAIN_IDS), ("reserve_ids", RESERVE_IDS),
                          ("holdout_ids", HOLDOUT_IDS), ("standing_ids", STAND_IDS)):
        if tuple(cfg[key]) != required:
            raise ValueError("Unauthorized split change: " + key)
    if cfg["permanent_hold_ids"] != ["LAFAN_WALK_10"]:
        raise ValueError("WALK_10 hold must remain")
    if tuple(cfg["normalization"]["fit_ids"]) != TRAIN_IDS or not cfg["normalization"]["frozen"]:
        raise ValueError("Normalizer must fit train clips only, then remain frozen")
    if cfg["training_ready"] is not False:
        raise ValueError("Physical acceptance is not available; training_ready must stay false")
    return cfg


class Clip:
    def __init__(self, path, record, spec, kinematics, quat_tolerance=1e-5):
        self.path, self.record, self.spec = Path(path), dict(record), spec
        self.id = record["id"]
        self.digest = sha256(path)
        if self.digest != record["npz_sha256"]:
            raise ValueError("Motion hash mismatch: " + self.id)
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {k: archive[k].copy() for k in archive.files}
        z = self.arrays
        required = ("qpos", "raw_qpos", "pre_lift_qpos", "time_s", "fps", "joint_names",
                    "leg_joint_names", "leg_dof_pos", "metadata_json", "source_support",
                    "root_z_lift", "sole_center", "sole_min_height", "quiet_stance_weight")
        if any(k not in z for k in required):
            raise ValueError("Missing NPZ field: " + self.id)
        self.metadata = json.loads(str(z["metadata_json"].item()))
        meta = self.metadata
        if meta["id"] != self.id or meta["quaternion_order"] != "wxyz" or meta["cyclic"]:
            raise ValueError("Identity/quaternion/cyclic contract mismatch")
        if meta["source_sha256"] != record["source_sha256"] or meta["split"] != record["split"]:
            raise ValueError("Provenance/split mismatch")
        if meta["training_ready"] is not False or record["training_ready"] is not False:
            raise ValueError("Unexpected promotion of unvalidated source")
        self.frames = len(z["qpos"])
        if self.frames != record["frames"] or self.frames < spec.history + 1:
            raise ValueError("Invalid frame count")
        if float(z["fps"]) != spec.fps or record["retarget_fps"] != spec.fps:
            raise ValueError("Only exact 100 Hz inputs are supported")
        if z["time_s"].shape != (self.frames,) or not np.allclose(z["time_s"], np.arange(self.frames) / spec.fps, atol=1e-10, rtol=0):
            raise ValueError("Invalid time grid")
        for k, value in z.items():
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError("Nonfinite " + k)
        for k in ("qpos", "raw_qpos", "pre_lift_qpos"):
            if z[k].shape != (self.frames, 36):
                raise ValueError("Expected N x 36 " + k)
            if np.max(np.abs(np.linalg.norm(z[k][:, 3:7], axis=-1) - 1)) > quat_tolerance:
                raise ValueError("Nonunit root quaternion " + k)
        for k, shape in (("source_support", (self.frames, 2)), ("root_z_lift", (self.frames,)),
                         ("sole_center", (self.frames, 2, 3)), ("sole_min_height", (self.frames, 2)),
                         ("leg_dof_pos", (self.frames, 12))):
            if z[k].shape != shape:
                raise ValueError("Invalid shape for " + k)
        self.all_joint_names = tuple(map(str, z["joint_names"]))
        leg_names = tuple(map(str, z["leg_joint_names"]))
        if len(self.all_joint_names) != 29 or set(leg_names) != set(spec.joint_names):
            raise ValueError("Unexpected model joint names")
        indices = name_indices(self.all_joint_names, spec.joint_names)
        self.joint_pos = z["qpos"][:, np.array(indices) + 7]
        reordered_leg = z["leg_dof_pos"][:, name_indices(leg_names, spec.joint_names)]
        if not np.allclose(self.joint_pos, reordered_leg, atol=1e-10, rtol=0):
            raise ValueError("leg_dof_pos is not final qpos")
        upper = [7 + i for i, n in enumerate(self.all_joint_names) if n not in spec.joint_names]
        if len(upper) != 17 or not np.allclose(z["qpos"][:, upper], 0, atol=1e-10):
            raise ValueError("Unexpected moving upper body")
        q = torch.from_numpy(self.joint_pos).float()
        quat = wxyz_to_xyzw(torch.from_numpy(z["qpos"][:, 3:7]).float())
        dq, omega = torch.zeros_like(q), torch.zeros((self.frames, 3), dtype=q.dtype)
        dq[1:] = (q[1:] - q[:-1]) * spec.fps
        omega[1:] = angular_velocity_body(quat[:-1], quat[1:], 1 / spec.fps)
        self.key_positions_b = kinematics.key_positions(self.joint_pos, spec.joint_names)
        self.features = build_features(spec, omega, q, dq, spec.joint_names,
                                       torch.from_numpy(self.key_positions_b).float(), spec.body_names)
        # Frame zero derivatives are undefined. Never sample it or fit stats on it.
        self.first_start, self.last_start = 1, self.frames - spec.history
        self.duration = (self.frames - 1) / spec.fps

    def window(self, start):
        if not isinstance(start, (int, np.integer)) or not self.first_start <= start <= self.last_start:
            raise ValueError("Window outside this finite clip (no clamping, wrap, padding)")
        return self.features[start:start + self.spec.history].clone()


class AMPDataset:
    def __init__(self, data_root, repo_root, config_path):
        self.root, self.repo_root = Path(data_root).resolve(), Path(repo_root).resolve()
        self.cfg = load_config(config_path)
        self.config_sha256 = sha256(config_path)
        self.spec = FeatureSpec(tuple(self.cfg["joint_names"]), tuple(self.cfg["body_names"]),
                                self.cfg["root_body"], self.cfg["fps"], self.cfg["history"])
        self.kinematics = URDFKinematics(self.repo_root / self.cfg["urdf"], self.spec)
        self.manifest_path = self.root / "retarget_manifest.json"
        self.manifest_sha256 = sha256(self.manifest_path)
        self.source_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = self.source_manifest["records"]
        if len(records) != 10 or {r["id"] for r in records} != set(ALL_IDS):
            raise ValueError("Expected exactly the approved 10 clips")
        expected_splits = {**{k: "train_candidate" for k in TRAIN_IDS},
                           **{k: "reserve_excluded_from_current_split" for k in RESERVE_IDS},
                           **{k: "holdout_candidate" for k in HOLDOUT_IDS},
                           **{k: "same_actor_group" for k in STAND_IDS}}
        self.clips = {}
        for record in records:
            cid = record["id"]
            if record["split"] != expected_splits[cid]:
                raise ValueError("Unexpected split for " + cid)
            if bool(record["manual_postprocess_review_hold"]) != (cid == "LAFAN_WALK_10"):
                raise ValueError("Unexpected manual hold")
            self.clips[cid] = Clip(self.root / cid / "motion.npz", record, self.spec, self.kinematics,
                                   self.cfg["quality"]["quat_norm_tolerance"])

    def fit_normalization(self, ids=TRAIN_IDS):
        if tuple(ids) != TRAIN_IDS:
            raise ValueError("Only 01/02/03/04 may fit normalization")
        # Concatenation here is statistics only, never a sampling trajectory.
        x = torch.cat([self.clips[cid].features[1:] for cid in ids], dim=0).double()
        floor = float(self.cfg["normalization"]["std_floor"])
        if floor <= 0:
            raise ValueError("Normalizer std_floor must be positive")
        return x.mean(0).float(), x.std(0, unbiased=False).clamp_min(floor).float()

    def sampler(self, seed=0, ids=TRAIN_IDS, purpose="dry_run", weighting=None):
        if tuple(ids) != TRAIN_IDS:
            raise ValueError("First-batch sampler is restricted to 01/02/03/04")
        if purpose != "dry_run":
            raise ValueError("training_ready=false: only data-pipeline dry runs are allowed")
        return ClipSampler(self, seed, ids, weighting)

    def verify_unchanged(self):
        if sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError("Source manifest changed during preparation")
        for c in self.clips.values():
            if sha256(c.path) != c.digest:
                raise ValueError("Source motion changed: " + c.id)


class ClipSampler:
    def __init__(self, dataset, seed, ids, weighting=None):
        if tuple(ids) != TRAIN_IDS:
            raise ValueError("Sampler split violation")
        self.dataset, self.ids = dataset, tuple(ids)
        self.rng = np.random.default_rng(seed)
        cfg = dataset.cfg["sampling"]
        mode = weighting or cfg["weighting"]
        if mode not in ("clip", "duration"):
            raise ValueError("Explicit clip or duration weighting required")
        if set(cfg["clip_weights"]) != set(ids):
            raise ValueError("Clip weights must name exactly the train clips")
        weights = np.array([cfg["clip_weights"][cid] for cid in ids], dtype=float)
        if mode == "duration":
            weights *= np.array([dataset.clips[cid].duration for cid in ids])
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Weights must be finite and positive")
        self.probabilities = weights / weights.sum()

    def sample(self, batch_size):
        if batch_size < 1:
            raise ValueError("Empty sample")
        choices = self.rng.choice(len(self.ids), size=batch_size, p=self.probabilities)
        windows, provenance = [], []
        for i in choices:
            clip = self.dataset.clips[self.ids[int(i)]]
            start = int(self.rng.integers(clip.first_start, clip.last_start + 1))
            windows.append(clip.window(start))
            provenance.append(dict(clip_id=clip.id, start_frame=start,
                                   end_frame=start + clip.spec.history - 1,
                                   start_s=start / clip.spec.fps,
                                   end_s=(start + clip.spec.history - 1) / clip.spec.fps))
        return torch.stack(windows), provenance
