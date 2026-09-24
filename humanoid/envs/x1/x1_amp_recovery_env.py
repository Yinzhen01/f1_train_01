"""Reference-state initialization, never reference actions or phase tracking."""
import torch
from isaacgym import gymtorch
from .x1_amp_env import X1AMPEnv
from humanoid.amp.recovery import reference_initial_states, centered_velocity_reward


class X1AMPRecoveryEnv(X1AMPEnv):
    def configure_reference_initialization(self, experiment):
        if tuple(self.dof_names) != experiment.spec.joint_names:
            raise ValueError("RSI joint order mismatch")
        data = reference_initial_states(experiment, experiment.cfg["recovery"]["reset_clearance_m"])
        self.rsi = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        self.rsi_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rsi_reset_count = 0
        self.rsi_enabled = True

    def _reset_dofs(self, ids):
        if not getattr(self, "rsi_enabled", False):
            return super()._reset_dofs(ids)
        index = torch.randint(len(self.rsi["q"]), (len(ids),), device=self.device)
        self.rsi_indices[ids] = index
        self.dof_pos[ids] = self.rsi["q"][index]
        self.dof_vel[ids] = self.rsi["qd"][index]
        ids32 = ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                             gymtorch.unwrap_tensor(ids32), len(ids))
        self.rsi_reset_count += len(ids)

    def _reset_root_states(self, ids):
        if not getattr(self, "rsi_enabled", False):
            return super()._reset_root_states(ids)
        self.root_states[ids] = self.rsi["root"][self.rsi_indices[ids]]
        self.root_states[ids, :3] += self.env_origins[ids]
        ids32 = ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(ids32), len(ids))

    def reset_idx(self, ids):
        super().reset_idx(ids)
        # Resetting a moving pose is not a physical acceleration from zero.
        if len(ids):
            self.last_dof_vel[ids] = self.dof_vel[ids]
            self.last_root_vel[ids] = self.root_states[ids, 7:13]

    def _reward_recovery_progress(self):
        return centered_velocity_reward(self.base_lin_vel[:, :2], self.commands[:, :2])

    def _reward_recovery_tilt(self):
        return self.projected_gravity[:, :2].square().sum(-1)

    def _reward_recovery_yaw_rate(self):
        return (self.base_ang_vel[:, 2]-self.commands[:, 2]).square()

    def _reward_recovery_vertical_velocity(self):
        return self.base_lin_vel[:, 2].square()

    def _reward_recovery_smoothness(self):
        cost = (self.actions-self.last_actions).square().sum(-1)
        cost += (self.actions-2*self.last_actions+self.last_last_actions).square().sum(-1)
        return cost*(self.episode_length_buf > 2)
