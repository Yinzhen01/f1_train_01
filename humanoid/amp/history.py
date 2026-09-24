"""Episode-local 10-step history with compact valid-only discriminator batches."""
from dataclasses import dataclass
import torch

from .features import angular_velocity_body, from_world_state, name_indices, normalize_quat


@dataclass
class PolicyState:
    root_pos_w: torch.Tensor
    root_quat_xyzw: torch.Tensor
    joint_pos: torch.Tensor
    key_positions_w: torch.Tensor
    joint_names: tuple
    body_names: tuple


class AMPHistory:
    def __init__(self, num_envs, spec, device="cpu"):
        self.spec = spec
        self.buffer = torch.full((num_envs, spec.history, spec.dim), float("nan"), device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode = torch.full_like(self.count, -1)
        self.step = torch.full_like(self.count, -1)

    def reset(self, ids):
        self.buffer[ids] = float("nan")
        self.count[ids] = 0
        self.episode[ids] = -1
        self.step[ids] = -1

    @torch.no_grad()
    def append(self, features, episode_ids, step_ids):
        if features.shape != (len(self.count), self.spec.dim) or not torch.isfinite(features).all():
            raise ValueError("Invalid policy feature frame")
        if episode_ids.shape != self.count.shape or step_ids.shape != self.count.shape:
            raise ValueError("Episode/step ID shape mismatch")
        discontinuity = (self.episode != episode_ids) | (step_ids != self.step + 1)
        self.reset(discontinuity)
        self.buffer[:, :-1] = self.buffer[:, 1:].clone()
        self.buffer[:, -1] = features.detach()
        self.count.add_(1).clamp_(max=self.spec.history)
        self.episode.copy_(episode_ids)
        self.step.copy_(step_ids)
        valid = self.count == self.spec.history
        # No padded histories are returned. Caller retains env mapping via mask.
        return self.buffer[valid].clone(), valid.clone()


class PolicyAMPStream:
    """Prime after reset; observe post-physics BEFORE auto-reset, once per 10 ms.

    The first state is only a derivative seed, not a discriminator frame.
    Actor observations and their history are not accessed or changed.
    """
    def __init__(self, num_envs, spec, device="cpu", control_dt=.01):
        if abs(control_dt - 1 / spec.fps) > 1e-10:
            raise ValueError("Policy AMP stream must run at 100 Hz")
        self.spec, self.dt = spec, control_dt
        self.history = AMPHistory(num_envs, spec, device)
        self.prev_q = torch.zeros((num_envs, 12), device=device)
        self.prev_quat = torch.zeros((num_envs, 4), device=device)
        self.primed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episode = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.step = torch.full_like(self.episode, -1)

    def _pose(self, state):
        q = state.joint_pos[:, name_indices(state.joint_names, self.spec.joint_names)]
        quat = normalize_quat(state.root_quat_xyzw)
        if q.shape != self.prev_q.shape or quat.shape != self.prev_quat.shape or not torch.isfinite(q).all():
            raise ValueError("Invalid policy pose")
        return q, quat

    @torch.no_grad()
    def prime(self, state, episode_ids, step_ids, env_ids=None):
        q, quat = self._pose(state)
        ids = slice(None) if env_ids is None else env_ids
        self.prev_q[ids] = q[ids]
        self.prev_quat[ids] = quat[ids]
        self.episode[ids], self.step[ids] = episode_ids[ids], step_ids[ids]
        self.primed[ids] = True
        self.history.reset(ids)

    @torch.no_grad()
    def observe_pre_reset(self, state, episode_ids, step_ids):
        if not self.primed.all():
            raise ValueError("Prime reset state before the first physics step")
        if not torch.equal(episode_ids, self.episode) or not torch.equal(step_ids, self.step + 1):
            raise ValueError("Reset or missed control tick: explicitly prime a new history")
        q, quat = self._pose(state)
        omega = angular_velocity_body(self.prev_quat, quat, self.dt)
        velocity = (q - self.prev_q) / self.dt
        frame = from_world_state(self.spec, quat, state.root_pos_w, omega, q, velocity,
                                 self.spec.joint_names, state.key_positions_w, state.body_names)
        windows, valid = self.history.append(frame, episode_ids, step_ids)
        self.prev_q.copy_(q)
        self.prev_quat.copy_(quat)
        self.step.copy_(step_ids)
        return windows, valid
