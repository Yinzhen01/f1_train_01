"""Real-simulator pre-reset capture for AMP; old task classes are untouched."""
import torch
import numpy as np
from .x1_dh_stand_env import X1DHStandEnv
from humanoid.amp.history import PolicyState
from humanoid.amp.features import quat_rotate, quat_conj
from humanoid.amp.scaled_experiment import validate_runtime_timing


class X1AMPEnv(X1DHStandEnv):
    def __init__(self, *args, **kwargs):
        self.amp_bridge = None
        super().__init__(*args, **kwargs)
        self.amp_episode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.amp_reset_total = 0
        self.amp_frame_check_max_error = 0.
        self.amp_verified_frames = 0

    def attach_amp(self, bridge, experiment):
        validate_runtime_timing(self.dt, self.sim_params.dt, self.cfg.control.decimation)
        self.amp_bridge, self.amp_experiment = bridge, experiment
        self.amp_body_ids = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n)
                             for n in experiment.spec.body_names]
        if min(self.amp_body_ids) < 0 or tuple(self.dof_names) != experiment.spec.joint_names:
            raise ValueError("Runtime body/joint name mismatch")
        self.amp_prime()

    def amp_state(self):
        return PolicyState(self.root_states[:, :3], self.root_states[:, 3:7], self.dof_pos,
            self.rigid_state[:, self.amp_body_ids, :3], tuple(self.dof_names), self.amp_experiment.spec.body_names)

    def amp_steps(self):
        return torch.full_like(self.amp_episode, self.common_step_counter)

    def amp_prime(self, ids=None):
        # prime consumes only root quaternion/joint pose, not stale reset body FK.
        self.amp_bridge.prime(self.amp_state(), self.amp_episode, self.amp_steps(), ids)

    def _get_phase(self):
        # Keep 47 observation fields but do not impose the old 0.7 s sine gait.
        return torch.zeros_like(self.episode_length_buf, dtype=torch.float)

    def compute_ref_state(self):
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1)
        self.ref_action = torch.zeros_like(self.dof_pos)

    def _get_stance_mask(self):
        # Critic diagnostic only; no source-support labels or phase rewards.
        return (self.contact_forces[:, self.feet_indices, 2] > 5.).float()

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= self.root_states[:, 2] < .3
        if getattr(self, "amp_force_reset_step", -1) == self.common_step_counter:
            self.reset_buf[0] = True
            self.time_out_buf[0] = False

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if len(ids) and hasattr(self, "amp_episode"):
            self.amp_episode[ids] += 1
            self.amp_reset_total += len(ids)

    def compute_reward(self):
        super().compute_reward()
        if self.amp_bridge is not None:
            state = self.amp_state()
            if self.amp_verified_frames < 12:
                # Verify actual Gym rigid states are link frames, not COM frames.
                q = self.dof_pos[:4].detach().cpu().numpy()
                expected = self.amp_experiment.kinematics.key_positions(q, self.dof_names)
                actual = quat_rotate(quat_conj(state.root_quat_xyzw[:4])[:, None],
                    state.key_positions_w[:4]-state.root_pos_w[:4, None]).detach().cpu().numpy()
                error = float(np.max(np.linalg.norm(expected-actual, axis=-1)))
                self.amp_frame_check_max_error = max(self.amp_frame_check_max_error, error)
                if error > 5e-4:
                    raise ValueError("Gym link-frame/FK mismatch: %.6f m" % error)
                self.amp_verified_frames += 1
            self.amp_bridge.capture_pre_reset(state, self.amp_episode, self.amp_steps())
