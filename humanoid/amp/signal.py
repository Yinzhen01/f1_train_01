"""Rounds 6/7: isolated style-signal interventions from the same smooth1250."""
import copy
import hashlib
from pathlib import Path

import torch

from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment

SIGNAL_GROUPS = ('signal_signed', 'signal_bridge')
SOURCE_SHA = 'e553b2e800d6c307e746e51f17877e9fd4bf27268979fdd7c3cb2c76453495df'
SOURCE_CONFIG = 'configs/amp/lafan_walk02_refine_smooth.json'


def signal_contract(group):
    if group not in ('signed', 'bridge'):
        raise ValueError('Unknown style intervention')
    return dict(group=group, round=6 if group == 'signed' else 7,
                source_task='TASK_20260924_072', source_config=SOURCE_CONFIG,
                source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=1250,
                learning_rate=5e-5, style_floor=-1. if group == 'signed' else 0.,
                bridge_gradient_penalty=1. if group == 'bridge' else 0.)


def validate_signal(experiment):
    cfg = experiment.cfg
    group = cfg.get('signal', {}).get('group')
    contract = signal_contract(group)
    if cfg['signal'] != contract:
        raise ValueError('Signal/source/round contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg)
    expected.pop('refinement')
    expected.update(experiment='f1_amp_walk02_signal_'+group, signal=contract,
                    formal=dict(num_envs=4096, updates=250), evaluation_updates=[1500])
    actual = copy.deepcopy(cfg)
    # Explanatory strings are not experiment mechanics; hashes still bind them.
    for key in ('scope', 'known_risks'):
        expected.pop(key, None); actual.pop(key, None)
    if actual != expected:
        raise ValueError('Unapproved change outside the isolated style intervention')
    return source


def signal_source(cfg):
    signal = cfg['signal']
    return dict(source_task=signal['source_task'], source_sha256=signal['source_checkpoint_sha256'],
                source_completed_updates=1250, target_completed_updates=1500,
                kind='matched_style_signal', style_floor=signal['style_floor'],
                bridge_gradient_penalty=signal['bridge_gradient_penalty'])


def validate_signal_certificate(experiment, continuation):
    validate_signal(experiment)
    expected = signal_source(experiment.cfg)
    if any(continuation.get(key) != value for key, value in expected.items()):
        raise ValueError('Wrong style-signal continuation proof')
    if (continuation.get('actor_and_discriminator_restored') is not True or
        continuation.get('optimizers_restored') is not True or continuation.get('replay_windows') != 50000):
        raise ValueError('Incomplete style-signal source restoration')
    return True


def warm_start_signal(runner, experiment, checkpoint):
    source = validate_signal(experiment)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Style-signal source SHA mismatch')
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state['amp_identity'] != source.identity() or state['completed_updates'] != 1250 or state['iter'] != 1249:
        raise ValueError('Wrong source identity/update')
    if state.get('amp_replay_state') is None:
        raise ValueError('Source must contain complete replay')
    rate = experiment.cfg['signal']['learning_rate']
    for key in ('optimizer_state_dict', 'es_optimizer_state_dict'):
        if any(group['lr'] != rate for group in state[key]['param_groups']):
            raise ValueError('Source learning rate changed')
    proof = dict(**signal_source(experiment.cfg),
                 **restore_learning_state(runner, experiment, state, learning_rate=rate))
    if (runner.amp_trainer.discriminator.style_floor != proof['style_floor'] or
        runner.amp_trainer.bridge_gradient_penalty != proof['bridge_gradient_penalty']):
        raise ValueError('Runtime intervention does not match configuration')
    validate_signal_certificate(experiment, proof)
    return proof
