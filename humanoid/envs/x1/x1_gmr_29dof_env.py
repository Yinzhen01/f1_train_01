"""Full 29-actuator imitation with separate legs, arms, waist and head budgets."""
import json
from isaacgym.torch_utils import quat_mul, quat_rotate_inverse
import torch

from .x1_gmr_clip_env import X1GMRClipEnv


class X1GMR29DOFEnv(X1GMRClipEnv):
    def _init_buffers(self):
        super()._init_buffers()
        if self.num_actions != 29 or self.motion.metadata.get("reference_variant") != "whole_body_29dof_v1":
            raise ValueError("29-DOF task requires its whole-body reference")
        self.joint_groups = {
            "leg": [i for i, name in enumerate(self.dof_names) if any(x in name for x in ("hip_", "knee_", "ankle_"))],
            "upper": [i for i, name in enumerate(self.dof_names) if any(x in name for x in ("shoulder_", "elbow_", "wrist_"))],
            "waist": [i for i, name in enumerate(self.dof_names) if name.startswith("lumbar_")],
            "neck": [i for i, name in enumerate(self.dof_names) if name.startswith(("neck_", "head_"))],
        }
        if [len(self.joint_groups[k]) for k in ("leg", "upper", "waist", "neck")] != [12, 12, 3, 2]:
            raise ValueError("Unexpected 29-DOF joint groups")
        ids = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
               for name in self.motion.tracking_body_names]
        if min(ids) < 0:
            raise ValueError("Chest/wrist rigid body missing")
        self.tracking_body_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.smoothness_diagnostics = {}
        for name, values in (("Kp", self.p_gains), ("Kd", self.d_gains),
                             ("armature", self.nominal_joint_armatures), ("effort", self.torque_limits)):
            if values.numel() != 29 or not torch.isfinite(values).all() or not (values > 0).all():
                raise ValueError("Invalid 29-DOF runtime " + name)
        properties = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        armature = torch.as_tensor(properties["armature"].copy(), device=self.device)
        if not torch.allclose(armature, self.nominal_joint_armatures, atol=1e-7, rtol=1e-5):
            raise ValueError("PhysX runtime armature differs from the nominal manifest")
        print("[gmr-29dof-runtime] " + json.dumps(dict(
            dof_names=self.dof_names, kp=self.p_gains.tolist(), kd=self.d_gains.tolist(),
            armature=armature.tolist(), torque_limits=self.torque_limits.tolist(),
            passive_damping=properties["damping"].tolist(), passive_friction=properties["friction"].tolist(),
            control_dt=self.dt, action_scale=self.cfg.control.action_scale)), flush=True)
        print("[gmr-29dof] actor_frame=98 critic_frame=141 actions=29 scratch_only=True "
              "reference_sha256=%s" % self.cfg.motion_reference.sha256, flush=True)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        required = {"action_smoothness", "gmr_torque_rate", "gmr_upper_joint_pos",
                    "gmr_waist_joint_pos", "gmr_neck_joint_pos", "gmr_chest_rotation", "gmr_hand_pos"}
        if not required.issubset(self.reward_scales):
            raise ValueError("Whole-body rewards missing from runtime accumulator")
        print("[gmr-29dof-rewards] weights_before_dt=" + json.dumps(
            {key: value / self.dt for key, value in self.reward_scales.items()}, sort_keys=True), flush=True)

    def _joint_group_reward(self, group):
        indices = self.joint_groups[group]
        error = self.dof_pos[:, indices] - self.ref_dof_pos[:, indices]
        return torch.exp(-error.square().mean(dim=1) / .04)

    def _reward_gmr_joint_pos(self):
        return self._joint_group_reward("leg")

    def _reward_gmr_upper_joint_pos(self):
        return self._joint_group_reward("upper")

    def _reward_gmr_waist_joint_pos(self):
        return self._joint_group_reward("waist")

    def _reward_gmr_neck_joint_pos(self):
        return self._joint_group_reward("neck")

    def _reward_gmr_hand_pos(self):
        positions = self.rigid_state[:, self.tracking_body_ids[1:], :3] - self.root_states[:, None, :3]
        base = self.base_quat[:, None].expand(-1, 2, -1).reshape(-1, 4)
        actual = quat_rotate_inverse(base, positions.reshape(-1, 3)).reshape(self.num_envs, 2, 3)
        error = (actual - self.reference["body_pos_base"][:, 1:]).square().sum(dim=-1).mean(dim=1)
        return torch.exp(-error / .01)

    def _reward_gmr_chest_rotation(self):
        target = quat_mul(self.reference["root_quat"], self.reference["body_quat_base"][:, 0])
        actual = self.rigid_state[:, self.tracking_body_ids[0], 3:7]
        error = 1. - (target * actual).sum(dim=-1).square().clamp(max=1.)
        return torch.exp(-error / .01)

    def _reward_action_smoothness(self):
        # Gate histories after reset; no penalty between unrelated clip phases.
        first = (self.actions - self.last_actions).square().mean(dim=1)
        second = (self.actions - 2 * self.last_actions + self.last_last_actions).square().mean(dim=1)
        return first * (self.episode_length_buf > 1) + second * (self.episode_length_buf > 2)

    def _reward_gmr_torque_rate(self):
        rate = (self.torques - self.last_torques) / self.torque_limits.clamp_min(1e-6) / self.dt
        return rate.square().mean(dim=1) * (self.episode_length_buf > 1)

    def compute_reward(self):
        super().compute_reward()
        # Snapshot BEFORE the parent updates last_actions / last_torques and before
        # resets; reading differences from the runner after step would return zero.
        valid = self.episode_length_buf > 1
        count = valid.sum().clamp_min(1)
        action_delta = (self.actions - self.last_actions).square().mean(dim=1)
        torque_delta = (self.torques - self.last_torques).square().mean(dim=1)
        ankle_ids = [i for i, name in enumerate(self.dof_names) if "ankle_" in name]
        ankle_ratio = (self.torques[:, ankle_ids].abs() / self.torque_limits[ankle_ids]).amax(dim=1)
        self.smoothness_diagnostics = {
            "gmr_action_delta_rms": torch.sqrt((action_delta * valid).sum() / count),
            "gmr_torque_delta_rms_nm": torch.sqrt((torque_delta * valid).sum() / count),
            "gmr_ankle_saturation_fraction": (ankle_ratio >= .99).float().mean(),
            "gmr_contact_force_peak_n": torch.linalg.vector_norm(self.contact_forces[:, self.feet_indices], dim=-1).max(),
        }

    def compute_observations(self):
        super().compute_observations()
        if self.obs_buf.shape[1] != self.cfg.env.num_observations or self.privileged_obs_buf.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError("29-DOF observation dimension mismatch")

    def get_command_tracking_debug(self):
        result = super().get_command_tracking_debug()
        result.update(self.smoothness_diagnostics)
        for group, ids in self.joint_groups.items():
            result["gmr_" + group + "_rmse_rad"] = torch.sqrt((self.dof_pos[:, ids] - self.ref_dof_pos[:, ids]).square().mean())
        return result
