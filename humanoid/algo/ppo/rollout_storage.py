# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

import torch

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.next_proprio_obs = None
            self.mu_metadata = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, num_single_obs=None, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        if num_single_obs is not None:
            self.next_proprio_obs = torch.zeros(num_transitions_per_env, num_envs, num_single_obs, device=self.device)
            
        self.num_single_obs = num_single_obs
        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0
        # Opt-in metadata never changes the original PPO tensors or sampling RNG.
        self.mu_startup_steps = None
        self.mu_metadata = None
        self.mu_metadata_recorded = None

    def configure_mu_temporal(self, startup_steps=66):
        if isinstance(startup_steps, bool) or not isinstance(startup_steps, int) or startup_steps < 0:
            raise ValueError("mu startup_steps must be a nonnegative integer")
        if self.step:
            raise RuntimeError("Cannot enable temporal metadata during a rollout")
        self.mu_startup_steps = startup_steps
        shape = (self.num_transitions_per_env, self.num_envs)
        self.mu_metadata = {key: torch.zeros(shape, device=self.observations.device, dtype=torch.long)
                            for key in ("episode_ids", "control_ticks", "episode_lengths")}
        self.mu_metadata_recorded = torch.zeros(shape[0], device=self.observations.device, dtype=torch.bool)

    def clone_mu_metadata(self, episode_ids, control_ticks, episode_lengths, commands):
        """Validate and snapshot pre-action metadata, before mutable env buffers advance."""
        if self.mu_startup_steps is None:
            raise RuntimeError("Temporal metadata is not enabled")
        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        result = {}
        for key, value in (("episode_ids", episode_ids), ("control_ticks", control_ticks),
                           ("episode_lengths", episode_lengths), ("commands", commands)):
            if not isinstance(value, torch.Tensor) or value.device != self.observations.device:
                raise ValueError("mu metadata tensors must share the storage device")
            if key == "commands":
                if value.ndim != 2 or value.shape[0] != self.num_envs or value.shape[1] not in (3, 4):
                    raise ValueError("mu commands must have shape [num_envs, 3 or 4]")
                if not value.is_floating_point() or not torch.isfinite(value).all():
                    raise ValueError("mu commands must be finite floating-point values")
                existing = self.mu_metadata.get("commands")
                if existing is not None and (existing.shape[-1] != value.shape[-1] or existing.dtype != value.dtype):
                    raise ValueError("mu command shape/dtype cannot change within storage")
            elif value.shape != (self.num_envs,) or value.dtype not in integer_dtypes or (value < 0).any():
                raise ValueError("mu IDs, ticks and lengths must be nonnegative integer [num_envs] tensors")
            result[key] = value.detach().clone()
        return result

    def _record_mu_metadata(self, metadata):
        if not isinstance(metadata, dict) or set(metadata) != {"episode_ids", "control_ticks", "episode_lengths", "commands"}:
            raise ValueError("Each temporal transition requires all four pre-action metadata tensors")
        values = self.clone_mu_metadata(**metadata)
        if "commands" not in self.mu_metadata:
            commands = values["commands"]
            self.mu_metadata["commands"] = torch.zeros(self.num_transitions_per_env, *commands.shape,
                                                       device=commands.device, dtype=commands.dtype)
        for key, value in values.items():
            self.mu_metadata[key][self.step].copy_(value)
        self.mu_metadata_recorded[self.step] = True

    def mu_temporal_data(self):
        """Return time-major real neighbours and unique-rollout validity diagnostics."""
        if self.mu_startup_steps is None or self.step != self.num_transitions_per_env:
            raise RuntimeError("Temporal batches require a complete enabled rollout")
        if not self.mu_metadata_recorded.all() or "commands" not in self.mu_metadata:
            raise RuntimeError("Incomplete temporal metadata")
        n, t = self.num_envs, self.num_transitions_per_env
        indices = torch.arange(t * n, device=self.observations.device)
        time = indices // n
        # Clamp only for safe indexing; rollout-boundary frames can never be valid.
        prev = (indices - n).clamp_min(0)
        older = (indices - 2 * n).clamp_min(0)
        boundary = time < 2
        candidate = ~boundary
        flat = {key: value.flatten(0, 1) for key, value in self.mu_metadata.items()}
        dones = self.dones.flatten(0, 1).squeeze(-1).bool()
        reasons = dict(
            rollout_boundary=boundary,
            previous_done=candidate & (dones[prev] | dones[older]),
            episode_changed=candidate & ((flat["episode_ids"] != flat["episode_ids"][prev]) |
                                          (flat["episode_ids"][prev] != flat["episode_ids"][older])),
            tick_discontinuity=candidate & ((flat["control_ticks"] - flat["control_ticks"][prev] != 1) |
                                            (flat["control_ticks"][prev] - flat["control_ticks"][older] != 1)),
            command_changed=candidate & ((flat["commands"] != flat["commands"][prev]).any(-1) |
                                          (flat["commands"][prev] != flat["commands"][older]).any(-1)),
            startup=candidate & ((flat["episode_lengths"] < self.mu_startup_steps) |
                                  (flat["episode_lengths"][prev] < self.mu_startup_steps) |
                                  (flat["episode_lengths"][older] < self.mu_startup_steps)))
        invalid = torch.zeros_like(boundary)
        for reason in reasons.values():
            invalid |= reason
        valid = ~invalid
        diagnostics = dict(total_samples=t*n, candidate_triplets=int(candidate.sum()),
                           valid_triplets=int(valid.sum()), invalid_triplets=int(invalid.sum()),
                           invalid_reasons={key: int(value.sum()) for key, value in reasons.items()},
                           invalid_reason_counts_overlap=True, metadata_observed=True,
                           startup_steps=self.mu_startup_steps)
        return dict(indices=indices, prev_indices=prev, older_indices=older,
                    valid_mask=valid, rollout_diagnostics=diagnostics)

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        if self.mu_startup_steps is not None:
            self._record_mu_metadata(transition.mu_metadata)
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        if self.num_single_obs is not None:
            self.next_proprio_obs[self.step].copy_(transition.next_proprio_obs)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0
        if self.mu_metadata_recorded is not None:
            self.mu_metadata_recorded.zero_()

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8, with_temporal=False):
        temporal = self.mu_temporal_data() if with_temporal else None
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        if self.num_single_obs is not None:
            next_proprio_obs = self.next_proprio_obs.flatten(0, 1)
            rewards = self.rewards.flatten(0, 1)
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                if self.num_single_obs is not None:
                    next_proprio_obs_batch = next_proprio_obs[batch_idx]
                    rewards_batch = rewards[batch_idx]
                    batch = (next_proprio_obs_batch, rewards_batch, obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
                             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None)
                else:
                    batch = (obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
                             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None)
                if with_temporal:
                    previous = temporal["prev_indices"][batch_idx]
                    older = temporal["older_indices"][batch_idx]
                    batch += (dict(prev_obs=observations[previous], older_obs=observations[older],
                                   valid_mask=temporal["valid_mask"][batch_idx], batch_indices=batch_idx,
                                   prev_indices=previous, older_indices=older, time_indices=batch_idx // self.num_envs,
                                   env_indices=batch_idx % self.num_envs,
                                   rollout_diagnostics=temporal["rollout_diagnostics"]),)
                yield batch
