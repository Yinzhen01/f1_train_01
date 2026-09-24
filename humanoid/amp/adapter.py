"""PPO/AMP transition adapter, importable without simulator or logging extras."""
import json
import torch


class AMPAlgorithmAdapter:
    def __init__(self, ppo, bridge, env, experiment):
        self.ppo, self.bridge, self.env, self.experiment = ppo, bridge, env, experiment
        self.updates = 0
        self.valid_windows = 0
        self.style_sum = 0.
        self.reset_checks = 0
        self.warmup_excluded = 0
        self.latest = {}

    def __getattr__(self, name):
        return getattr(self.ppo, name)

    def process_env_step(self, rewards, dones, infos):
        task = float(rewards.mean())
        metrics = self.bridge.process_env_step(self.ppo, rewards, dones, infos)
        valid = metrics["valid_mask"]
        self.valid_windows += int(valid.sum())
        self.warmup_excluded += int((~valid).sum())
        self.style_sum += float(metrics["style_reward"].sum())
        ids = dones.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.env.amp_prime(ids)
            if (self.bridge.stream.history.count[ids] != 0).any():
                raise RuntimeError("New episode has old AMP history")
            self.reset_checks += len(ids)
        self.latest.update(task_reward=task, style_reward=float(metrics["style_reward"].mean()),
                           valid_fraction=float(valid.float().mean()))
        rewards.copy_(metrics["mixed_reward"])

    def update(self):
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
