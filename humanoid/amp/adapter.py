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
        self.mu_run_updates = 0
        self.mu_run_sums = {}
        self.mu_run_triplets = 0
        self.mu_rollout_diagnostics = []

    def __getattr__(self, name):
        return getattr(self.ppo, name)

    def act(self, observations, critic_observations):
        if getattr(self.ppo, 'mu_regularizer', None) is not None:
            # Metadata is sampled before action/environment stepping. Tensor
            # clones in PPO prevent later auto-reset from rewriting this row.
            self.ppo.set_mu_metadata(self.env.amp_episode, self.env.amp_steps(),
                                     self.env.episode_length_buf, self.env.commands)
        return self.ppo.act(observations, critic_observations)

    def mu_loss_report(self):
        regularizer = getattr(self.ppo, 'mu_regularizer', None)
        if regularizer is None:
            return None
        result = {key: value/max(1, self.mu_run_updates) for key, value in self.mu_run_sums.items()}
        result.update(updates=self.mu_run_updates, valid_triplets=self.mu_run_triplets,
            teacher_unchanged=regularizer.teacher_unchanged(), metadata_observed=bool(self.mu_run_updates),
            all_finite=all(torch.isfinite(torch.as_tensor(value)).item() for value in result.values()),
            anchor_coef=regularizer.anchor_coef, temporal_coef=regularizer.temporal_coef,
            rollout_diagnostics=list(self.mu_rollout_diagnostics))
        return result

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
        if getattr(self.ppo, 'mu_regularizer', None) is not None:
            mu = dict(self.ppo.mu_latest)
            self.mu_run_updates += 1
            self.mu_run_triplets += int(mu['valid_triplets'])
            self.mu_rollout_diagnostics.append(mu['rollout_diagnostics'])
            for key in ('anchor_loss', 'temporal_loss', 'weighted_anchor_loss',
                        'weighted_temporal_loss', 'aux_grad_norm'):
                self.mu_run_sums[key] = self.mu_run_sums.get(key, 0.)+float(mu[key])
            if not mu.get('teacher_unchanged') or not mu.get('all_finite'):
                raise ValueError('Deterministic mu teacher/loss invariant failed')
            self.latest.update({'mu_'+key: value for key, value in mu.items()
                                if isinstance(value, (int, float, bool))})
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
