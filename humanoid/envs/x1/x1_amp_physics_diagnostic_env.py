"""Capture env0's physical substeps without changing policy, reward or control."""
import torch

from .x1_amp_jitter_env import X1AMPJitterEnv


class X1AMPPhysicsDiagnosticEnv(X1AMPJitterEnv):
    physics_raw_body_names = (
        'base_link', 'left_ankle_roll_link', 'right_ankle_roll_link')

    def _begin_physics_step(self):
        super()._begin_physics_step()
        if self.cfg.control.decimation != 10:
            raise ValueError('Physics diagnosis requires ten substeps per control interval')
        if not hasattr(self, 'physics_raw_body_ids'):
            self.physics_raw_body_ids = [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name)
                for name in self.physics_raw_body_names]
            if (min(self.physics_raw_body_ids) < 0 or
                    len(set(self.physics_raw_body_ids)) != 3):
                raise ValueError('Physics diagnosis requires three distinct collision bodies')
        self.physics_interval_initial_velocity = self.dof_vel[0].clone()
        self.physics_interval_initial_torque = self.torques[0].clone()
        self.physics_raw_velocity_samples = []
        self.physics_raw_torque_samples = []
        self.physics_raw_body_force_samples = []
        self.physics_raw_body_state_samples = []

    def _after_physics_substep(self, substep):
        super()._after_physics_substep(substep)
        if substep != len(self.physics_raw_velocity_samples):
            raise ValueError('Physics substeps must be captured once and in order')
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        bodies = self.physics_raw_body_ids
        state = self.rigid_state[0, bodies].clone()
        state[:, :3] -= self.env_origins[0]
        self.physics_raw_velocity_samples.append(self.dof_vel[0].clone())
        self.physics_raw_torque_samples.append(self.torques[0].clone())
        self.physics_raw_body_force_samples.append(self.contact_forces[0, bodies].clone())
        self.physics_raw_body_state_samples.append(state)

    def jitter_capture(self):
        result = super().jitter_capture()
        if len(self.physics_raw_velocity_samples) != self.cfg.control.decimation:
            raise ValueError('Incomplete raw physical interval')
        result.update(
            physics_raw_velocity=torch.stack(self.physics_raw_velocity_samples),
            physics_raw_torque=torch.stack(self.physics_raw_torque_samples),
            physics_raw_body_force=torch.stack(self.physics_raw_body_force_samples),
            physics_raw_body_state=torch.stack(self.physics_raw_body_state_samples),
            physics_interval_initial_velocity=self.physics_interval_initial_velocity.clone(),
            physics_interval_initial_torque=self.physics_interval_initial_torque.clone())
        return result
