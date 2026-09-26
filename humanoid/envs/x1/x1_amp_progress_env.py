"""Change task velocity frame only; observe both rewards on real transitions."""
import torch

from .x1_amp_contact_env import X1AMPContactEnv
from humanoid.amp.progress import progress_rewards


class X1AMPProgressEnv(X1AMPContactEnv):
    def reset_progress_diagnostics(self):
        self.progress_calls = 0
        self.progress_sums = torch.zeros(4, device=self.device, dtype=torch.float64)

    def _reward_recovery_progress(self):
        if not hasattr(self, 'progress_calls'): self.reset_progress_diagnostics()
        selected, body, world = progress_rewards(self.base_lin_vel[:, :2], self.root_states[:, 7:9],
            self.commands[:, :2], self.cfg.rewards.progress_velocity_frame)
        self.progress_calls += 1
        self.progress_sums += torch.stack([body.sum(), world.sum(), selected.sum(), (body-world).abs().sum()]).detach().double()
        return selected

    def progress_diagnostics(self):
        values = self.progress_sums.cpu().tolist()
        return dict(velocity_frame=self.cfg.rewards.progress_velocity_frame, calls=self.progress_calls,
            env_steps=self.progress_calls*self.num_envs, runtime_scale=self.reward_scales['recovery_progress'],
            body_sum=values[0], world_sum=values[1], selected_sum=values[2], abs_body_world_difference_sum=values[3])
