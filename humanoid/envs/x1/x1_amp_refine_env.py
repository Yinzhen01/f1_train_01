"""AMP task regularizers, not frame/phase reference tracking."""
import torch
import numpy as np
from scipy.spatial import ConvexHull
from .x1_amp_recovery_env import X1AMPRecoveryEnv
from humanoid.amp.features import quat_rotate, quat_conj
from humanoid.amp.recovery import FOOT_NAMES, foot_collision_vertices
from humanoid.amp.refinement import robust_acceleration_cost, ground_force_weight


class X1AMPRefineEnv(X1AMPRecoveryEnv):
    def configure_reference_initialization(self, experiment):
        super().configure_reference_initialization(experiment)
        vertices = foot_collision_vertices(experiment.kinematics.path)
        self.refine_sole_local = torch.tensor(np.asarray([
            vertices[name][vertices[name][:, 2] <= vertices[name][:, 2].min()+1e-5].mean(0)
            for name in FOOT_NAMES]), dtype=torch.float32, device=self.device)
        # Hull vertices preserve the exact minimum projection, unlike center height.
        self.refine_collision_hulls = [torch.tensor(vertices[name][ConvexHull(vertices[name]).vertices],
            dtype=torch.float32, device=self.device) for name in FOOT_NAMES]

    def _reward_refine_heading(self):
        # The existing actor already observes all 3 Euler angles, including yaw.
        return 1-torch.cos(self.base_euler_xyz[:, 2])

    def _reward_refine_acceleration(self):
        acceleration = (self.dof_vel-self.last_dof_vel)/self.dt
        return robust_acceleration_cost(acceleration)*(self.episode_length_buf > 2)

    def _reward_refine_slip(self):
        if not hasattr(self, "refine_sole_local"):
            return torch.zeros(self.num_envs, device=self.device)
        feet = self.rigid_state[:, self.feet_indices]
        arm = quat_rotate(feet[:, :, 3:7], self.refine_sole_local.expand(self.num_envs, -1, -1))
        velocity = feet[:, :, 7:10]+torch.cross(feet[:, :, 10:13], arm, dim=-1)
        up = torch.zeros_like(feet[:, :, :3]); up[:, :, 2] = 1.
        local_z = quat_rotate(quat_conj(feet[:, :, 3:7]), up)
        height = torch.stack([(local_z[:, i] @ mesh.T).min(-1).values+feet[:, i, 2]
                              for i, mesh in enumerate(self.refine_collision_hulls)], dim=-1)
        height -= self.env_origins[:, 2:3]
        weight = ground_force_weight(self.contact_forces[:, self.feet_indices, 2], height)
        # No per-foot target position, gait clock or reference contact label.
        return (weight*velocity[:, :, :2].square().sum(-1)).mean(-1)*(self.episode_length_buf > 2)
