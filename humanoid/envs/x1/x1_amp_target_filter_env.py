"""Filter clipped targets before PD; observations retain the applied-action state."""
import torch

from .x1_amp_jitter_env import X1AMPJitterEnv
from humanoid.amp.target_filter import filter_alpha, filter_target


class X1AMPTargetFilterEnv(X1AMPJitterEnv):
    def reset_direction_diagnostics(self):
        super().reset_direction_diagnostics()
        self.filter_calls = 0
        self.filter_reset_count = 0
        self.filter_sums = torch.zeros(4, device=self.device, dtype=torch.float64)

    def _begin_physics_step(self):
        # LeggedRobot has clipped the raw policy sample, but not applied PD yet.
        # Do not mutate that tensor: PPO keeps the original sample/log probability.
        if not hasattr(self, 'target_filter_state'):
            self.target_filter_state = torch.zeros_like(self.actions)
            self.target_previous_raw = torch.zeros_like(self.actions)
        if not hasattr(self, 'filter_sums'):
            self.reset_direction_diagnostics()
        spec = self.cfg.target_filter
        alpha = filter_alpha(spec['cutoff_hz'], self.dt)
        self.target_filter_before = self.target_filter_state.clone()
        self.target_raw = self.actions.clone()
        self.actions = filter_target(self.target_raw, self.target_filter_before, alpha)
        self.target_filter_state.copy_(self.actions)
        valid = (self.episode_length_buf > 2)[:, None]
        self.filter_sums += torch.stack((((self.target_raw-self.target_previous_raw).square()*valid).sum(),
            ((self.actions-self.target_filter_before).square()*valid).sum(),
            ((self.target_raw-self.actions).square()*valid).sum(), valid.sum())).detach().double()
        self.target_previous_raw.copy_(self.target_raw)
        self.filter_calls += 1
        super()._begin_physics_step()

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if len(ids) and hasattr(self, 'target_filter_state'):
            for name in ('target_filter_state', 'target_previous_raw', 'target_raw', 'target_filter_before'):
                getattr(self, name)[ids] = 0.
            # Parent resets all action histories before constructing observations.
            self.filter_reset_count += len(ids)

    def direction_diagnostics(self):
        result = super().direction_diagnostics()
        values = self.filter_sums.cpu().tolist()
        result.update(target_filter=dict(self.cfg.target_filter), filter_calls=self.filter_calls,
            filter_alpha=filter_alpha(self.cfg.target_filter['cutoff_hz'], self.dt),
            filter_reset_count=self.filter_reset_count, filter_input_delta_squared_sum=values[0],
            filter_applied_delta_squared_sum=values[1], filter_residual_squared_sum=values[2],
            filter_valid_intervals=values[3])
        return result

    def jitter_capture(self):
        result = super().jitter_capture()
        result.update(control_raw_action=self.target_raw,
                      control_filter_before=self.target_filter_before)
        return result
