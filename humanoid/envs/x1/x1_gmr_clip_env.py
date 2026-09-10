"""PPO tracking of a finite GMR clip; the clip end is a true terminal state."""
import hashlib

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, quat_mul, quat_rotate_inverse
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.gmr_motion import GMRMotion
from .x1_dh_stand_env import X1DHStandEnv


class X1GMRClipEnv(X1DHStandEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_ref_state()

    def _init_buffers(self):
        super()._init_buffers()
        path = self.cfg.motion_reference.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        self.motion = GMRMotion(path, self.dof_names, self.device, self.cfg.motion_reference.sha256)
        asset_path = self.cfg.asset.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        # Git checkouts normalize newlines; match canonical XML bytes, not platform EOL.
        with open(asset_path, "rb") as stream:
            urdf_bytes = stream.read()
        if hashlib.sha256(urdf_bytes.replace(b"\r\n", b"\n")).hexdigest() != self.motion.metadata["training_urdf_lf_sha256"]:
            raise ValueError("Training URDF differs from the reference preparation model")
        self.reference_start = torch.zeros(self.num_envs, device=self.device)
        self.clip_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i, name in enumerate(self.motion.metadata["foot_names"]):
            body = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
            if body != int(self.feet_indices[i]):
                raise ValueError("Unexpected left/right foot order")
        properties = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        hard_lower = torch.as_tensor(properties["lower"].copy(), device=self.device)
        hard_upper = torch.as_tensor(properties["upper"].copy(), device=self.device)
        if torch.any(self.motion.data["dof_pos"] < hard_lower - 1e-5) or torch.any(self.motion.data["dof_pos"] > hard_upper + 1e-5):
            raise ValueError("GMR reference exceeds simulator DOF limits")
        if abs(self.motion.duration - self.cfg.env.episode_length_s) > 1e-5:
            raise ValueError("Clip duration/config mismatch")
        print("[gmr-clip] frames=%d duration=%.3fs joints=%s cyclic=False no_DR=True urdf_sha256=%s" %
              (self.motion.frames, self.motion.duration, self.dof_names, hashlib.sha256(urdf_bytes).hexdigest()), flush=True)

    def motion_time(self):
        return self.reference_start + self.episode_length_buf * self.dt

    def _get_phase(self):
        # Inherited observation builds sin(2*pi*phase), cos(2*pi*phase).
        # A half-circle distinguishes start/end and cannot wrap during an episode.
        return .5 * torch.clamp(self.motion_time() / self.motion.duration, 0., 1.)

    def compute_ref_state(self):
        self.reference = self.motion.sample(self.motion_time())
        self.ref_dof_pos = self.reference["dof_pos"]
        self.ref_action = (self.ref_dof_pos - self.default_dof_pos) / self.cfg.control.action_scale
        velocity = quat_rotate_inverse(self.reference["root_quat"], self.reference["root_vel"])
        angular = quat_rotate_inverse(self.reference["root_quat"], self.reference["root_ang_vel"])
        self.commands[:, :2] = velocity[:, :2]
        self.commands[:, 2] = angular[:, 2]

    def _resample_commands(self):
        self.compute_ref_state()

    def _post_physics_step_callback(self):
        self.phase_length_buf += 1
        # Refresh targets BEFORE reward evaluation, not one control step later.
        self.compute_ref_state()

    def _get_stance_mask(self):
        return self.reference["foot_contact"]

    def check_termination(self):
        super().check_termination()
        self.clip_finished = self.motion_time() >= self.motion.duration - 1e-6
        self.reset_buf |= self.clip_finished
        # Finite motion completion is terminal, not an infinite-MDP time truncation.
        self.time_out_buf[self.clip_finished] = False
        self.reset_buf |= self.root_states[:, 2] < .3

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        settings = self.cfg.motion_reference
        if settings.random_start:
            self.reference_start[env_ids] = torch.rand(len(env_ids), device=self.device) * (self.motion.duration - settings.minimum_remaining_s)
            from_start = torch.rand(len(env_ids), device=self.device) < settings.start_at_zero_probability
            self.reference_start[env_ids[from_start]] = 0.
        else:
            self.reference_start[env_ids] = 0.
        super().reset_idx(env_ids)
        self.feet_height[env_ids] = 0.

    def _reset_dofs(self, env_ids):
        reference = self.motion.sample(self.reference_start[env_ids])
        self.dof_pos[env_ids] = reference["dof_pos"]
        self.dof_vel[env_ids] = reference["dof_vel"]
        ids = env_ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(ids), len(ids))

    def _reset_root_states(self, env_ids):
        reference = self.motion.sample(self.reference_start[env_ids])
        self.root_states[env_ids, :3] = reference["root_pos"] + self.env_origins[env_ids]
        self.root_states[env_ids, 3:7] = reference["root_quat"]
        self.root_states[env_ids, 7:10] = reference["root_vel"]
        self.root_states[env_ids, 10:13] = reference["root_ang_vel"]
        ids = env_ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(ids), len(ids))

    def _sole_positions(self):
        foot = self.rigid_state[:, self.feet_indices]
        local = self.motion.sole_local[None].expand(self.num_envs, -1, -1)
        return foot[:, :, :3] + quat_apply(foot[:, :, 3:7].reshape(-1, 4), local.reshape(-1, 3)).reshape(self.num_envs, 2, 3)

    def _reward_gmr_joint_pos(self):
        return torch.exp(-torch.mean((self.dof_pos - self.ref_dof_pos).square(), dim=1) / .04)

    def _reward_gmr_joint_vel(self):
        return torch.exp(-torch.mean((self.dof_vel - self.reference["dof_vel"]).square(), dim=1) / 4.)

    def _reward_gmr_foot_pos(self):
        difference = self._sole_positions() - self.root_states[:, None, :3]
        base = self.base_quat[:, None].expand(-1, 2, -1).reshape(-1, 4)
        actual = quat_rotate_inverse(base, difference.reshape(-1, 3)).reshape(self.num_envs, 2, 3)
        return torch.exp(-torch.sum((actual - self.reference["foot_pos_base"]).square(), dim=(1, 2)) / .005)

    def _reward_gmr_foot_rotation(self):
        base = self.reference["root_quat"][:, None].expand(-1, 2, -1).reshape(-1, 4)
        target = quat_mul(base, self.reference["foot_quat_base"].reshape(-1, 4)).reshape(self.num_envs, 2, 4)
        actual = self.rigid_state[:, self.feet_indices, 3:7]
        error = 1. - torch.sum(target * actual, dim=-1).square().clamp(max=1.)
        return torch.exp(-torch.mean(error, dim=1) / .01)

    def _reward_gmr_contact(self):
        actual = (self.contact_forces[:, self.feet_indices, 2] > 5.).float()
        return 1. - torch.mean(torch.abs(actual - self.reference["foot_contact"]), dim=1)

    def _reward_gmr_root_pos(self):
        error = self.root_states[:, :3] - self.env_origins - self.reference["root_pos"]
        return torch.exp(-torch.sum(error.square(), dim=1) / .0625)

    def _reward_gmr_root_rotation(self):
        error = 1. - torch.sum(self.base_quat * self.reference["root_quat"], dim=1).square().clamp(max=1.)
        return torch.exp(-error / .01)

    def _reward_gmr_velocity(self):
        linear = torch.sum((self.root_states[:, 7:10] - self.reference["root_vel"]).square(), dim=1)
        angular = torch.sum((self.root_states[:, 10:13] - self.reference["root_ang_vel"]).square(), dim=1)
        return torch.exp(-linear / .25 - angular / 2.)

    def _reward_gmr_slip(self):
        foot = self.rigid_state[:, self.feet_indices]
        lever = self._sole_positions() - foot[:, :, :3]
        velocity = foot[:, :, 7:10] + torch.cross(foot[:, :, 10:13], lever, dim=-1)
        contact = (self.contact_forces[:, self.feet_indices, 2] > 5.).float()
        return torch.sum(velocity[:, :, :2].square().sum(dim=-1) * contact, dim=1)

    def get_command_tracking_debug(self):
        result = super().get_command_tracking_debug()
        result.update({"gmr_joint_rmse_rad": torch.sqrt(torch.mean((self.dof_pos - self.ref_dof_pos).square())),
                       "gmr_phase": torch.mean(self.motion_time() / self.motion.duration),
                       "gmr_contact_match": torch.mean(self._reward_gmr_contact())})
        return result
