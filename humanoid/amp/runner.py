"""AMP adapter around the unchanged DHPPO and DHOnPolicyRunner.

The environment captures before auto-reset; this adapter consumes the terminal
transition, then seeds only the new episodes. PPO and D optimizers are separate.
"""
import torch
from humanoid.algo.ppo.dh_on_policy_runner import DHOnPolicyRunner
from .discriminator import AMPDiscriminator, DiscriminatorTrainer
from .integration import AMPBridge


from .adapter import AMPAlgorithmAdapter


class AMPOnPolicyRunner(DHOnPolicyRunner):
    def __init__(self, env, train_cfg, experiment, log_dir, device):
        super().__init__(env, train_cfg, log_dir, device)
        d = AMPDiscriminator(experiment.spec, experiment.mean, experiment.std).to(device)
        self.amp_trainer = DiscriminatorTrainer(d)
        bridge = AMPBridge(experiment.spec, self.amp_trainer, env.num_envs, experiment.cfg["reward"], device)
        env.attach_amp(bridge, experiment)
        self.alg = AMPAlgorithmAdapter(self.alg, bridge, env, experiment)
        self.amp_experiment = experiment

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        for key, value in self.alg.latest.items():
            self.writer.add_scalar("AMP/"+key, value, locs["it"])

    def save(self, path, infos=None):
        torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(),
            es_optimizer_state_dict=self.alg.state_estimator_optimizer.state_dict(),
            iter=self.it, completed_updates=self.it+1, infos=infos,
            amp_discriminator_state_dict=self.amp_trainer.discriminator.state_dict(),
            amp_optimizer_state_dict=self.amp_trainer.optimizer.state_dict(),
            amp_identity=self.amp_experiment.identity()), path)

    def load(self, path, load_optimizer=True):
        raise RuntimeError("First AMP experiment is scratch-only; no implicit checkpoint reuse")
