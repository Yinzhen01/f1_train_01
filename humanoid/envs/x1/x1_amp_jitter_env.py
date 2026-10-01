"""Same controller/dynamics; reward-call audit and read-only 1 kHz telemetry."""
import torch

from .x1_amp_direction_env import X1AMPDirectionEnv


class X1AMPJitterEnv(X1AMPDirectionEnv):
    def reset_direction_diagnostics(self):
        super().reset_direction_diagnostics()
        self.jitter_smoothness_calls = 0
        self.jitter_physics_steps = 0
        self.jitter_sums = torch.zeros(3, device=self.device, dtype=torch.float64)

    def _begin_physics_step(self):
        super()._begin_physics_step()
        if not hasattr(self, 'jitter_sums'):
            self.reset_direction_diagnostics()
        self.jitter_previous_velocity = self.dof_vel.clone()
        self.jitter_previous_torque = self.torques.clone()
        # Histories across reset are not physical slew/acceleration samples.
        self.jitter_valid = self.episode_length_buf > 2
        self.jitter_accel_squared = torch.zeros_like(self.dof_vel)
        self.jitter_torque_delta_squared = torch.zeros_like(self.torques)
        self.jitter_accel_peak = torch.zeros_like(self.dof_vel)
        self.jitter_force_peak = torch.zeros(self.num_envs, 2, device=self.device)

    def _after_physics_substep(self, substep):
        super()._after_physics_substep(substep)
        acceleration = (self.dof_vel-self.jitter_previous_velocity)/self.sim_params.dt
        delta = self.torques-self.jitter_previous_torque
        self.jitter_accel_squared += acceleration.square()/self.cfg.control.decimation
        self.jitter_torque_delta_squared += delta.square()/self.cfg.control.decimation
        self.jitter_accel_peak = torch.maximum(self.jitter_accel_peak, acceleration.abs())
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.jitter_force_peak = torch.maximum(self.jitter_force_peak,
            self.contact_forces[:, self.feet_indices, 2].clamp_min(0.))
        self.jitter_previous_velocity.copy_(self.dof_vel)
        self.jitter_previous_torque.copy_(self.torques)
        self.jitter_physics_steps += 1
        if substep == self.cfg.control.decimation-1:
            valid = self.jitter_valid[:, None]
            self.jitter_sums[1] += (self.jitter_accel_squared*valid).sum().detach().double()
            self.jitter_sums[2] += (self.jitter_torque_delta_squared*valid).sum().detach().double()

    def _reward_recovery_smoothness(self):
        value = super()._reward_recovery_smoothness()
        if not hasattr(self, 'jitter_sums'):
            self.reset_direction_diagnostics()
        self.jitter_smoothness_calls += 1
        self.jitter_sums[0] += value.sum().detach().double()
        return value

    def direction_diagnostics(self):
        result = super().direction_diagnostics()
        values = self.jitter_sums.cpu().tolist()
        result.update(smoothness_calls=self.jitter_smoothness_calls,
            smoothness_scale=self.reward_scales['recovery_smoothness'],
            physics_substeps=self.jitter_physics_steps, smoothness_cost_sum=values[0],
            physics_accel_squared_sum=values[1], physics_torque_delta_squared_sum=values[2])
        return result

    def jitter_capture(self):
        return dict(physics_valid=self.jitter_valid,
            physics_accel_squared=self.jitter_accel_squared,
            physics_accel_peak=self.jitter_accel_peak,
            physics_torque_delta_squared=self.jitter_torque_delta_squared,
            physics_foot_force_peak=self.jitter_force_peak)
