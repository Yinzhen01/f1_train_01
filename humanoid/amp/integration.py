"""Preparation-only bridge for existing DHPPO's process_env_step/update interface.

No task is registered and no old runner is modified. A future simulator task
must capture pre-reset state and explicitly prime reset environments. The bridge
can be exercised with DHPPO or a recording test double on CPU now.
"""
import torch
from .history import PolicyAMPStream
from .recovery import PolicyReplay, static_negative_windows


class AMPBridge:
    def __init__(self, spec, discriminator_trainer, num_envs, reward_config, device="cpu", recovery_config=None):
        self.stream = PolicyAMPStream(num_envs, spec, device, reward_config["dt"])
        self.trainer = discriminator_trainer
        self.reward_config = dict(reward_config)
        for key in ("task_weight", "style_weight", "style_scale"):
            if not 0 <= float(self.reward_config[key]) < float("inf"):
                raise ValueError("Invalid reward weight")
        self.pending = None
        self.policy_batches = []
        self.recovery = dict(recovery_config) if recovery_config else None
        self.replay = None if self.recovery is None else PolicyReplay(self.recovery["replay_capacity"])
        if self.recovery is not None:
            if not 0 <= self.recovery["static_negative_fraction"] <= .5:
                raise ValueError("Invalid static negative fraction")
            if self.recovery["discriminator_updates"] < 1 or self.recovery["replay_insert"] < 1:
                raise ValueError("Invalid recovery discriminator budget")

    def prime(self, state, episode_ids, step_ids, env_ids=None):
        if self.pending is not None:
            raise RuntimeError("Consume pre-reset transition before priming the next episode")
        self.stream.prime(state, episode_ids, step_ids, env_ids)

    @torch.no_grad()
    def capture_pre_reset(self, state, episode_ids, step_ids):
        if self.pending is not None:
            raise RuntimeError("Previous AMP transition was not consumed")
        self.pending = self.stream.observe_pre_reset(state, episode_ids, step_ids)

    @torch.no_grad()
    def process_env_step(self, ppo, task_reward, dones, infos):
        if self.pending is None:
            raise RuntimeError("Missing pre-reset AMP capture; cannot read auto-reset states")
        windows, valid = self.pending
        self.pending = None
        if task_reward.shape != valid.shape or dones.shape != valid.shape:
            raise ValueError("PPO transition shape mismatch")
        style = torch.zeros_like(task_reward)
        if len(windows):
            style[valid] = self.trainer.discriminator.style_reward(
                windows, self.reward_config["style_scale"], self.reward_config["dt"])
            # Only valid, detached histories can enter discriminator updates.
            self.policy_batches.append(windows.detach().clone())
        mixed = self.reward_config["task_weight"] * task_reward.detach() + self.reward_config["style_weight"] * style
        ppo.process_env_step(mixed.detach(), dones, infos)
        # A terminal pre-reset frame is valid for its old episode, never the next.
        self.stream.primed[dones.bool()] = False
        self.stream.history.reset(dones.bool())
        return {"mixed_reward": mixed, "style_reward": style, "valid_mask": valid}

    def update_discriminator(self, sampler, batch_size=64, seed=0):
        if self.pending is not None:
            raise RuntimeError("Cannot update with an unconsumed transition")
        if not self.policy_batches:
            return {"skipped": True, "reason": "no complete policy histories"}
        policy = torch.cat(self.policy_batches, dim=0).detach()
        self.policy_batches.clear()
        generator = torch.Generator(device=policy.device).manual_seed(seed)
        if self.recovery is not None:
            return self._recovery_update(sampler, policy, batch_size, generator)
        indices = torch.randint(len(policy), (batch_size,), generator=generator, device=policy.device)
        demo, _ = sampler.sample(batch_size)
        # PPO.update() is intentionally separate; discriminator loss never touches it.
        return self.trainer.step(demo.to(policy.device), policy[indices])

    def _recovery_update(self, sampler, policy, batch_size, generator):
        """Current/replay real policy negatives, optional labelled synthetic subset."""
        auxiliary = int(batch_size*self.recovery["static_negative_fraction"])
        real_count = batch_size-auxiliary
        totals = {}
        for _ in range(self.recovery["discriminator_updates"]):
            demo, _ = sampler.sample(batch_size)
            demo = demo.to(policy.device)
            replay_count = real_count//2 if self.replay.count else 0
            ids = torch.randint(len(policy), (real_count-replay_count,), generator=generator, device=policy.device)
            real = policy[ids]
            if replay_count:
                real = torch.cat((real, self.replay.sample(replay_count, generator)))
            negatives = real
            if auxiliary:
                negatives = torch.cat((real, static_negative_windows(demo[:auxiliary])))
            stats = self.trainer.step(demo, negatives)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.) + value/self.recovery["discriminator_updates"]
        # Insert only a bounded random subset of actual policy windows, AFTER
        # training against the older buffer. Synthetic negatives never enter it.
        ids = torch.randint(len(policy), (min(len(policy), self.recovery["replay_insert"]),),
                            generator=generator, device=policy.device)
        self.replay.add(policy[ids])
        with torch.no_grad():
            d = self.trainer.discriminator
            totals.update(real_policy_score=float(d(real).mean()), demo_score=float(d(demo).mean()),
                held_pose_score=float(d(static_negative_windows(demo[:128])).mean()),
                static_negative_fraction=self.recovery["static_negative_fraction"], replay_count=self.replay.count)
        return totals


def assert_formal_training_allowed(dataset):
    # A boolean in a JSON is not a physical acceptance certificate.
    raise RuntimeError("Preparation only: model limits/geometry, dynamics and Isaac Gym integration remain unvalidated")
