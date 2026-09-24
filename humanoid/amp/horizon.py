"""Rounds8/9: matched episode-horizon coverage, unchanged AMP/task rewards."""
import copy
import hashlib
import math
from pathlib import Path

import torch

from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment

HORIZON_GROUPS = ('horizon_short', 'horizon_long')
SOURCE_SHA = 'd5f55bb7c25f6ff8d45588372db8d3feaeb3d8b92d6baaa600260f02b64f4dc8'
SOURCE_CONFIG = 'configs/amp/lafan_walk02_signal_bridge.json'


def horizon_contract(group):
    if group not in ('short', 'long'):
        raise ValueError('Unknown horizon group')
    return dict(group=group, round=8 if group == 'short' else 9,
        source_task='TASK_20260924_094', source_config=SOURCE_CONFIG,
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=1500,
        learning_rate=5e-5, episode_length_s=6. if group == 'short' else 60.,
        style_floor=0., bridge_gradient_penalty=1.)


def apply_horizon_config(cfg, group):
    if cfg.env.episode_length_s != 6.:
        raise ValueError('Unexpected nominal source episode length')
    cfg.env.episode_length_s = horizon_contract(group)['episode_length_s']
    return cfg


def validate_horizon(experiment):
    contract = horizon_contract(experiment.cfg.get('horizon', {}).get('group'))
    if experiment.cfg['horizon'] != contract:
        raise ValueError('Horizon/source/round contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg)
    expected.pop('signal')
    expected.update(experiment='f1_amp_walk02_horizon_'+contract['group'], horizon=contract,
        evaluation_updates=[1750], evaluation_duration_s=60.)
    actual = copy.deepcopy(experiment.cfg)
    for key in ('scope', 'known_risks'):
        expected.pop(key, None); actual.pop(key, None)
    if expected != actual:
        raise ValueError('Unapproved change outside episode horizon and continuation identity')
    return source


def horizon_source(cfg):
    h = cfg['horizon']
    return dict(source_task=h['source_task'], source_sha256=h['source_checkpoint_sha256'],
        source_completed_updates=1500, target_completed_updates=1750,
        kind='matched_episode_horizon', episode_length_s=h['episode_length_s'],
        style_floor=0., bridge_gradient_penalty=1.)


def validate_horizon_certificate(experiment, continuation):
    validate_horizon(experiment)
    expected = horizon_source(experiment.cfg)
    if any(continuation.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong horizon continuation proof')
    if (continuation.get('actor_and_discriminator_restored') is not True or
        continuation.get('optimizers_restored') is not True or continuation.get('replay_windows') != 50000):
        raise ValueError('Incomplete horizon source restoration')
    return True


def warm_start_horizon(runner, experiment, checkpoint):
    source = validate_horizon(experiment)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Horizon source SHA mismatch')
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state['amp_identity'] != source.identity() or state['completed_updates'] != 1500 or state['iter'] != 1499:
        raise ValueError('Wrong horizon source identity/update')
    rate = experiment.cfg['horizon']['learning_rate']
    for key in ('optimizer_state_dict', 'es_optimizer_state_dict'):
        if any(group['lr'] != rate for group in state[key]['param_groups']):
            raise ValueError('Source learning rate changed')
    proof = dict(**horizon_source(experiment.cfg),
        **restore_learning_state(runner, experiment, state, learning_rate=rate))
    if runner.amp_trainer.discriminator.style_floor != 0. or runner.amp_trainer.bridge_gradient_penalty != 1.:
        raise ValueError('Horizon experiment changed AMP signal mechanics')
    validate_horizon_certificate(experiment, proof)
    return proof


class HorizonProbe:
    """Pre-reset coverage instrumentation. Only smoke injects two reset tests."""
    def __init__(self, env, smoke=False):
        if env.num_envs < 2:
            raise ValueError('Horizon probe requires at least two environments')
        self.env, self.smoke = env, bool(smoke)
        self.start_step = env.common_step_counter
        self.steps = 0
        self.counts = torch.zeros(6, dtype=torch.long, device=env.device)
        self.max_age_steps = torch.zeros((), dtype=torch.long, device=env.device)
        self.forced_physical_verified = self.forced_timeout_verified = False
        self.thresholds = [round(s/env.dt)+1 for s in (6., 20., 40.)]
        if env.max_episode_length != math.ceil(env.cfg.env.episode_length_s/env.dt):
            raise ValueError('Runtime episode horizon mismatch')
        original = env.check_termination
        if smoke:
            env.amp_force_reset_step = self.start_step+37

        def check():
            elapsed = env.common_step_counter-self.start_step
            if self.smoke and elapsed == 61:
                # Exercise the existing timeout branch, without changing its
                # threshold, robot state or physical failure rules.
                env.episode_length_buf[1] = env.max_episode_length+1
            original()
            if self.smoke and elapsed == 37:
                self.forced_physical_verified = bool(env.reset_buf[0] and not env.time_out_buf[0]
                                                    and env._reward_termination()[0] == 1)
            if self.smoke and elapsed == 61:
                self.forced_timeout_verified = bool(env.reset_buf[1] and env.time_out_buf[1]
                                                   and env._reward_termination()[1] == 0)
            age = env.episode_length_buf
            physical = env.reset_buf.bool() & ~env.time_out_buf.bool()
            self.counts += torch.stack([torch.as_tensor(env.num_envs, device=env.device)] +
                [(age > limit).sum() for limit in self.thresholds] +
                [physical.sum(), (env.reset_buf.bool() & env.time_out_buf.bool()).sum()])
            self.max_age_steps = torch.maximum(self.max_age_steps, age.max())
            self.steps += 1
        env.check_termination = check

    def report(self):
        values = self.counts.cpu().tolist()
        return dict(smoke_injected_resets=self.smoke, control_steps=self.steps,
            captured_transitions=values[0], beyond_6s_reset_horizon=values[1],
            beyond_20s_reset_horizon=values[2], beyond_40s_reset_horizon=values[3],
            physical_failures=values[4], timeouts=values[5],
            max_episode_age_s=int(self.max_age_steps)*self.env.dt,
            configured_episode_length_s=self.env.cfg.env.episode_length_s,
            runtime_max_episode_steps=self.env.max_episode_length,
            forced_physical_reset_verified=self.forced_physical_verified,
            forced_timeout_reset_verified=self.forced_timeout_verified,
            note='Smoke age is artificially advanced once to test timeout; never use smoke coverage as long-walk evidence.')


def validate_horizon_probe(report, episode_length_s, smoke):
    if (report.get('smoke_injected_resets') is not smoke or
        report.get('configured_episode_length_s') != episode_length_s or
        report.get('control_steps', 0) <= 0 or report.get('captured_transitions', 0) <= 0):
        raise ValueError('Missing or wrong horizon coverage diagnostics')
    if smoke and (report.get('forced_physical_reset_verified') is not True or
                  report.get('forced_timeout_reset_verified') is not True):
        raise ValueError('Smoke did not exercise physical and timeout resets separately')
    if not smoke and (report.get('forced_physical_reset_verified') or report.get('forced_timeout_reset_verified')):
        raise ValueError('Synthetic reset leaked into formal training')
    if not smoke and episode_length_s == 6. and report['beyond_6s_reset_horizon'] != 0:
        raise ValueError('Short control passed its actual timeout horizon')
    return True
