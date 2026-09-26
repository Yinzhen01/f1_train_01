"""Keep AMP/contact dynamics; measure actual bounded direction task rewards."""
import torch

from .x1_amp_contact_env import X1AMPContactEnv
from humanoid.amp.direction import direction_rewards


class X1AMPDirectionEnv(X1AMPContactEnv):
    def reset_direction_diagnostics(self):
        self.direction_calls = 0
        self.direction_heading_calls = 0
        self.direction_sums = torch.zeros(5, device=self.device, dtype=torch.float64)

    def _reward_recovery_progress(self):
        if not hasattr(self, 'direction_calls'): self.reset_direction_diagnostics()
        selected, body, world = direction_rewards(self.base_lin_vel[:, :2], self.root_states[:, 7:9],
            self.commands[:, :2], self.cfg.rewards.direction_world_fraction)
        self.direction_calls += 1
        self.direction_sums[:4] += torch.stack([body.sum(), world.sum(), selected.sum(),
                                               (body-world).abs().sum()]).detach().double()
        return selected

    def _reward_refine_heading(self):
        if not hasattr(self, 'direction_calls'): self.reset_direction_diagnostics()
        cost = super()._reward_refine_heading()
        self.direction_heading_calls += 1
        self.direction_sums[4] += cost.sum().detach().double()
        return cost

    def direction_diagnostics(self):
        values = self.direction_sums.cpu().tolist()
        return dict(world_fraction=self.cfg.rewards.direction_world_fraction, calls=self.direction_calls,
            heading_calls=self.direction_heading_calls, env_steps=self.direction_calls*self.num_envs,
            progress_scale=self.reward_scales['recovery_progress'], heading_scale=self.reward_scales['refine_heading'],
            body_sum=values[0], world_sum=values[1], selected_sum=values[2],
            abs_body_world_difference_sum=values[3], heading_cost_sum=values[4])
