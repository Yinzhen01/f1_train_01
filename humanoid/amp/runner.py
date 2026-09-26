"""AMP adapter around the unchanged DHPPO and DHOnPolicyRunner.

The environment captures before auto-reset; this adapter consumes the terminal
transition, then seeds only the new episodes. PPO and D optimizers are separate.
"""
import torch
from pathlib import Path
from humanoid.algo.ppo.dh_on_policy_runner import DHOnPolicyRunner
from .discriminator import AMPDiscriminator, DiscriminatorTrainer
from .integration import AMPBridge


from .adapter import AMPAlgorithmAdapter
from .learnability import evaluation_checkpoint_name


class AMPOnPolicyRunner(DHOnPolicyRunner):
    def __init__(self, env, train_cfg, experiment, log_dir, device):
        super().__init__(env, train_cfg, log_dir, device)
        signal = experiment.cfg.get("signal", experiment.cfg.get("horizon", experiment.cfg.get('contact_refinement', experiment.cfg.get('progress', experiment.cfg.get('direction', experiment.cfg.get('sustain', {}))))))
        d = AMPDiscriminator(experiment.spec, experiment.mean, experiment.std,
                             style_floor=signal.get("style_floor", 0.)).to(device)
        self.amp_trainer = DiscriminatorTrainer(d, bridge_gradient_penalty=signal.get("bridge_gradient_penalty", 0.))
        bridge = AMPBridge(experiment.spec, self.amp_trainer, env.num_envs, experiment.cfg["reward"], device,
                           recovery_config=experiment.cfg.get("recovery"))
        env.attach_amp(bridge, experiment)
        self.alg = AMPAlgorithmAdapter(self.alg, bridge, env, experiment)
        self.amp_experiment = experiment

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        for key, value in self.alg.latest.items():
            self.writer.add_scalar("AMP/"+key, value, locs["it"])
        name = evaluation_checkpoint_name(locs["it"] + 1,
            self.amp_experiment.cfg.get("evaluation_updates", []))
        if name:
            if self.amp_experiment.cfg.get("artifact_checkpoint_offset"):
                name = "model_%d.pt" % (self.amp_experiment.cfg["artifact_checkpoint_offset"]+locs["it"]+1)
            checkpoint = str(Path(self.log_dir)/name)
            self.save(checkpoint, infos={"purpose": "independent_evaluation",
                      "effectiveness_verified": False, "dr_unlocked": False})
            callback = getattr(self, "evaluation_callback", None)
            if callback is not None:
                callback(checkpoint, locs["it"] + 1)

    def save(self, path, infos=None):
        replay_checkpoint = (self.it+1 in self.amp_experiment.cfg.get("evaluation_updates", []) or
                             Path(path).stem == "model_%d" % (self.it+1))
        torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(),
            es_optimizer_state_dict=self.alg.state_estimator_optimizer.state_dict(),
            iter=self.it, completed_updates=self.it+1, infos=infos,
            amp_discriminator_state_dict=self.amp_trainer.discriminator.state_dict(),
            amp_optimizer_state_dict=self.amp_trainer.optimizer.state_dict(),
            amp_identity=self.amp_experiment.identity(),
            amp_replay_state=None if self.alg.bridge.replay is None or not replay_checkpoint
                             else self.alg.bridge.replay.state_dict()), path)

    def load(self, path, load_optimizer=True):
        raise RuntimeError("First AMP experiment is scratch-only; no implicit checkpoint reuse")
