"""Two independent penalties; unchanged actions, PD and simulation dynamics."""
import torch

from .x1_amp_jitter_env import X1AMPJitterEnv
from humanoid.amp.refinement import robust_acceleration_cost


def robust_normalized_cost(value, scale):
    return (2.*(torch.sqrt(1.+(value/scale).square())-1.)).mean(-1)


class X1AMPSubstepEnv(X1AMPJitterEnv):
    def reset_direction_diagnostics(self):
        super().reset_direction_diagnostics()
        self.substep_acceleration_calls = 0
        self.substep_torque_calls = 0
        self.substep_sums = torch.zeros(5, device=self.device, dtype=torch.float64)

    def _begin_physics_step(self):
        super()._begin_physics_step()
        self.substep_acceleration_cost = torch.zeros(self.num_envs, device=self.device)
        self.substep_torque_cost = torch.zeros(self.num_envs, device=self.device)

    def _after_physics_substep(self, substep):
        # Compute BEFORE the parent's telemetry advances the two histories.
        spec = self.cfg.substep_penalty
        acceleration = (self.dof_vel-self.jitter_previous_velocity)/self.sim_params.dt
        delta = self.torques-self.jitter_previous_torque
        self.substep_acceleration_cost += robust_normalized_cost(
            acceleration, spec['acceleration_normalizer'])/self.cfg.control.decimation
        self.substep_torque_cost += robust_normalized_cost(
            delta, spec['torque_delta_normalizer'])/self.cfg.control.decimation
        super()._after_physics_substep(substep)

    def _reward_refine_acceleration(self):
        valid = self.jitter_valid
        physical = self.substep_acceleration_cost*valid
        torque = self.substep_torque_cost*valid
        endpoint = robust_acceleration_cost((self.dof_vel-self.last_dof_vel)/self.dt)*(self.episode_length_buf > 2)
        selected = physical if self.cfg.substep_penalty['acceleration'] else endpoint
        self.substep_acceleration_calls += 1
        self.substep_sums += torch.stack((physical.sum(), endpoint.sum(), torque.sum(),
                                         selected.sum(), valid.sum())).detach().double()
        return selected

    def _reward_substep_torque(self):
        self.substep_torque_calls += 1
        return self.substep_torque_cost*self.jitter_valid

    def direction_diagnostics(self):
        result = super().direction_diagnostics()
        values = self.substep_sums.cpu().tolist()
        result.update(substep_penalty=dict(self.cfg.substep_penalty),
            substep_acceleration_calls=self.substep_acceleration_calls,
            substep_torque_calls=self.substep_torque_calls,
            acceleration_scale=self.reward_scales['refine_acceleration'],
            torque_scale=self.reward_scales.get('substep_torque', 0.),
            substep_acceleration_cost_sum=values[0], endpoint_acceleration_cost_sum=values[1],
            substep_torque_cost_sum=values[2], used_acceleration_cost_sum=values[3],
            substep_valid_intervals=values[4])
        return result

    def jitter_capture(self):
        result = super().jitter_capture()
        result.update(physics_acceleration_cost=self.substep_acceleration_cost,
                      physics_torque_cost=self.substep_torque_cost)
        return result
