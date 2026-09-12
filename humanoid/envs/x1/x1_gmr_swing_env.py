"""Add ONLY swing clearance and unintended swing contact costs to smooth v1."""
import hashlib
import json
from pathlib import Path

from isaacgym.torch_utils import quat_apply, quat_mul
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.gmr_swing import (load_geometry, minimum_height, swing_envelope,
                                sample_envelope, swing_terms)
from .x1_gmr_smooth_env import X1GMRSmoothEnv


class X1GMRSwingEnv(X1GMRSmoothEnv):
    def _init_buffers(self):
        super()._init_buffers()
        if self.cfg.terrain.mesh_type != "plane":
            raise ValueError("Swing world-z clearance is defined only for the existing z=0 plane")
        settings = self.cfg.swing
        if not (settings.gate_height_full_m > settings.gate_height_low_m >= 0
                and settings.height_sigma_m > 0 and settings.height_tolerance_m >= 0
                and settings.max_height_cost > 0 and settings.contact_threshold_n > 0):
            raise ValueError("Invalid swing reward settings")
        geometry_path = settings.geometry_file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        urdf = self.cfg.asset.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        self.swing_hulls, geometry = load_geometry(geometry_path, self.cfg.motion_reference.sha256,
                                                  urdf, self.motion.metadata["foot_names"], self.device)
        if not torch.allclose(torch.tensor(geometry["sole_local"], device=self.device),
                              self.motion.sole_local, atol=1e-7, rtol=0):
            raise ValueError("Swing geometry uses a different sole calibration")
        self.swing_envelope = swing_envelope(self.motion.data["foot_contact"], self.motion.fps,
                                             settings.boundary_s, settings.ramp_s)
        if not torch.all(self.swing_envelope.sum(dim=0) > 0):
            raise ValueError("Swing gating disabled every sample of a foot")
        self.swing_diagnostics = {}
        print("[gmr-swing-geometry] " + json.dumps(dict(
            geometry_sha256=hashlib.sha256(Path(geometry_path).read_bytes()).hexdigest(),
            hull_points=geometry["hull_counts"], reference_sha256=self.cfg.motion_reference.sha256,
            settings={k: getattr(settings, k) for k in dir(settings) if not k.startswith("_")},
            active_reference_frames=(self.swing_envelope > 0).sum(dim=0).tolist())), flush=True)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        for key in ("gmr_swing_clearance", "gmr_swing_contact"):
            if self.reward_scales.get(key, 0.) >= 0:
                raise ValueError("Missing negative swing penalty: " + key)

    def compute_reward(self):
        # The callback already refreshed the reference at the current motion time.
        # Compute once, BEFORE reward summation and reset/history overwrite.
        base = self.reference["root_quat"][:, None].expand(-1, 2, -1).reshape(-1, 4)
        ref_sole = quat_apply(base, self.reference["foot_pos_base"].reshape(-1, 3)).reshape(-1, 2, 3)
        ref_sole += self.reference["root_pos"][:, None, :] + self.env_origins[:, None, :]
        ref_quat = quat_mul(base, self.reference["foot_quat_base"].reshape(-1, 4)).reshape(-1, 2, 4)
        self.swing_actual_height = minimum_height(self._sole_positions(),
            self.rigid_state[:, self.feet_indices, 3:7], self.swing_hulls) - self.env_origins[:, 2:3]
        self.swing_target_height = minimum_height(ref_sole, ref_quat, self.swing_hulls) - self.env_origins[:, 2:3]
        phase = sample_envelope(self.swing_envelope, self.motion_time(), self.motion.fps)
        self.swing_costs = swing_terms(self.swing_actual_height, self.swing_target_height, phase,
            self.contact_forces[:, self.feet_indices, 2], self.episode_length_buf > 1, self.cfg.swing)
        super().compute_reward()
        gate = self.swing_costs["gate"]
        count = gate.sum().clamp_min(1.)
        contact_gate = self.swing_costs["contact_gate"]
        speed = torch.linalg.vector_norm(self._sole_velocities()[:, :, :2], dim=-1)
        unexpected = self.swing_costs["unexpected_contact"]
        self.swing_diagnostics = {
            "gmr_swing_gate_mean": gate.mean(),
            "gmr_swing_contact_gate_mean": contact_gate.mean(),
            "gmr_swing_deficit_mean_m": (gate * self.swing_costs["height_deficit"]).sum() / count,
            "gmr_swing_unexpected_contact_fraction": unexpected.sum() / contact_gate.sum().clamp_min(1.),
            "gmr_swing_contact_speed_mps": (speed * unexpected).sum() / unexpected.sum().clamp_min(1.),
            "gmr_swing_clearance_cost": self.swing_costs["clearance_cost"].mean(),
            "gmr_swing_contact_cost": self.swing_costs["contact_cost"].mean(),
        }

    def _reward_gmr_swing_clearance(self):
        return self.swing_costs["clearance_cost"]

    def _reward_gmr_swing_contact(self):
        return self.swing_costs["contact_cost"]

    def get_command_tracking_debug(self):
        result = super().get_command_tracking_debug()
        result.update(self.swing_diagnostics)
        return result
