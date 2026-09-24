"""PPO/AMP transition adapter, importable without simulator or logging extras."""
import json
import torch


class AMPAlgorithmAdapter:
    def __init__(self, ppo, bridge, env, experiment):
        self.ppo, self.bridge, self.env, self.experiment = ppo, bridge, env, experiment
        self.updates = 0
        self.valid_windows = 0
        self.style_sum = 0.
        self.style_abs_sum = 0.
        self.reset_checks = 0
        self.warmup_excluded = 0
        self.latest = {}
        self.rollout_sums = {}
        self.rollout_steps = 0

    def __getattr__(self, name):
        return getattr(self.ppo, name)

    def process_env_step(self, rewards, dones, infos):
        task = float(rewards.mean())
        metrics = self.bridge.process_env_step(self.ppo, rewards, dones, infos)
        valid = metrics["valid_mask"]
        self.valid_windows += int(valid.sum())
        self.warmup_excluded += int((~valid).sum())
        self.style_sum += float(metrics["style_reward"].sum())
        self.style_abs_sum += float(metrics["style_reward"].abs().sum())
        ids = dones.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.env.amp_prime(ids)
            if (self.bridge.stream.history.count[ids] != 0).any():
                raise RuntimeError("New episode has old AMP history")
            self.reset_checks += len(ids)
        values = dict(task_reward=task, style_reward=float(metrics["style_reward"].mean()),
                      weighted_style_reward=float(metrics["style_reward"].mean())*self.bridge.reward_config["style_weight"],
                      mixed_reward=float(metrics["mixed_reward"].mean()), valid_fraction=float(valid.float().mean()))
        observed_style = metrics["style_reward"][valid]
        values.update(style_negative_fraction=float((observed_style < 0).float().mean()) if len(observed_style) else 0.,
                      style_zero_fraction=float((observed_style == 0).float().mean()) if len(observed_style) else 0.)
        for key, value in values.items():
            self.rollout_sums[key] = self.rollout_sums.get(key, 0.) + value
        self.rollout_steps += 1
        rewards.copy_(metrics["mixed_reward"])

    def update(self):
        if self.rollout_steps < 1:
            raise RuntimeError("No rollout reward samples")
        self.latest = {key: value/self.rollout_steps for key, value in self.rollout_sums.items()}
        self.rollout_sums.clear()
        self.rollout_steps = 0
        losses = self.ppo.update()
        if not all(torch.isfinite(torch.as_tensor(x)) for x in losses):
            raise ValueError("Nonfinite PPO loss")
        stats = self.bridge.update_discriminator(self.experiment,
            self.experiment.cfg["discriminator_batch_size"], self.experiment.cfg["seed"]+self.updates)
        if stats.get("skipped"):
            raise RuntimeError("No valid policy windows in PPO rollout")
        self.latest.update({"discriminator_"+k: v for k, v in stats.items()})
        self.updates += 1
        print("[f1-amp-update] "+json.dumps(dict(iteration=self.updates, **self.latest)), flush=True)
        return losses
