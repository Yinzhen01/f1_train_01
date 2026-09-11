"""Reset-safe temporal penalties; reference, observations and dynamics unchanged."""
import json
import torch

from .x1_gmr_upright_env import X1GMRUprightEnv


class X1GMRSmoothEnv(X1GMRUprightEnv):
    def _init_buffers(self):
        super()._init_buffers()
        if self.num_actions != 12:
            raise ValueError("Smooth continuation requires the 12-DOF policy")
        self.smoothness_diagnostics = {}
        self.smooth_ankle_ids = [i for i, name in enumerate(self.dof_names) if "ankle_" in name]
        if len(self.smooth_ankle_ids) != 4:
            raise ValueError("Expected four named ankle joints")
        props = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        armature = torch.as_tensor(props["armature"].copy(), device=self.device)
        if not torch.allclose(armature, self.nominal_joint_armatures, atol=1e-7, rtol=1e-5):
            raise ValueError("Runtime armature differs from the nominal manifest")
        print("[gmr-smooth-runtime] " + json.dumps(dict(
            dof_names=self.dof_names, kp=self.p_gains.tolist(), kd=self.d_gains.tolist(),
            armature=armature.tolist(), torque_limits=self.torque_limits.tolist(),
            passive_damping=props["damping"].tolist(), passive_friction=props["friction"].tolist(),
            control_dt=self.dt, action_scale=self.cfg.control.action_scale)), flush=True)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        for key in ("action_smoothness", "gmr_torque_rate", "gmr_trunk_tilt"):
            if key not in self.reward_scales:
                raise ValueError("Missing smoothness/trunk reward: " + key)
        print("[gmr-smooth-rewards] weights_before_dt=" + json.dumps(
            {key: value / self.dt for key, value in self.reward_scales.items()}, sort_keys=True), flush=True)

    def _reward_action_smoothness(self):
        first = (self.actions - self.last_actions).square().mean(dim=1)
        second = (self.actions - 2 * self.last_actions + self.last_last_actions).square().mean(dim=1)
        return first * (self.episode_length_buf > 1) + second * (self.episode_length_buf > 2)

    def _reward_gmr_torque_rate(self):
        rate = (self.torques - self.last_torques) / self.torque_limits.clamp_min(1e-6) / self.dt
        return rate.square().mean(dim=1) * (self.episode_length_buf > 1)

    def compute_reward(self):
        super().compute_reward()
        # Capture BEFORE reset/history overwrite. Last-substep samples only:
        # this does not measure all torque changes inside the 1 kHz PD loop.
        valid = self.episode_length_buf > 1
        count = valid.sum().clamp_min(1)
        action_delta = (self.actions - self.last_actions).square().mean(dim=1)
        torque_delta = (self.torques - self.last_torques).square().mean(dim=1)
        ankle = self.smooth_ankle_ids
        ratio = (self.torques[:, ankle].abs() / self.torque_limits[ankle]).amax(dim=1)
        self.smoothness_diagnostics = {
            "gmr_action_delta_rms": torch.sqrt((action_delta * valid).sum() / count),
            "gmr_torque_delta_rms_nm": torch.sqrt((torque_delta * valid).sum() / count),
            "gmr_ankle_saturation_fraction": (ratio >= .99).float().mean(),
            "gmr_contact_force_peak_n": torch.linalg.vector_norm(self.contact_forces[:, self.feet_indices], dim=-1).max(),
        }

    def get_command_tracking_debug(self):
        result = super().get_command_tracking_debug()
        result.update(self.smoothness_diagnostics)
        return result
