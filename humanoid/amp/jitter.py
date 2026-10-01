"""Rounds 19/20: isolate action smoothness from the exact control2500 state."""
import copy
import hashlib
import math
from pathlib import Path

import torch

from .sustain import apply_sustain_config, validate_sustain_diagnostics
from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment

JITTER_GROUPS = ('jitter_control', 'jitter_smooth')
SOURCE_CONFIG = 'configs/amp/lafan_walk02_sustain_control.json'
SOURCE_SHA = '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31'


def jitter_contract(group):
    if group not in ('control', 'smooth'):
        raise ValueError('Unknown jitter group')
    return dict(group=group, round=19 if group == 'control' else 20,
        source_task='TASK_20260926_110', source_config=SOURCE_CONFIG,
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=2500,
        learning_rate=5e-5, episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.,
        smoothness_scale=-.02 if group == 'control' else -.2)


def apply_jitter_config(cfg, group):
    cfg = apply_sustain_config(cfg, 'control')
    if cfg.rewards.scales.recovery_smoothness != -.02:
        raise ValueError('Unexpected source smoothness')
    cfg.rewards.scales.recovery_smoothness = jitter_contract(group)['smoothness_scale']
    return cfg


def validate_jitter(experiment):
    c = jitter_contract(experiment.cfg.get('jitter', {}).get('group'))
    if experiment.cfg['jitter'] != c:
        raise ValueError('Jitter contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg); expected.pop('sustain')
    expected.update(experiment='f1_amp_walk02_jitter_'+c['group'], jitter=c, evaluation_updates=[2750])
    actual = copy.deepcopy(experiment.cfg)
    for key in ('scope', 'known_risks'):
        expected.pop(key, None); actual.pop(key, None)
    if expected != actual:
        raise ValueError('Change outside smoothness weight and bounded 2500 continuation')
    return source


def jitter_source(cfg):
    c = cfg['jitter']
    return dict(source_task=c['source_task'], source_sha256=c['source_checkpoint_sha256'],
        source_completed_updates=2500, target_completed_updates=2750, kind='bounded_jitter',
        smoothness_scale=c['smoothness_scale'], episode_length_s=60.,
        style_floor=0., bridge_gradient_penalty=1.)


def validate_jitter_certificate(experiment, continuation):
    validate_jitter(experiment)
    if any(continuation.get(k) != v for k, v in jitter_source(experiment.cfg).items()):
        raise ValueError('Wrong jitter continuation proof')
    if (continuation.get('actor_and_discriminator_restored') is not True or
        continuation.get('optimizers_restored') is not True or continuation.get('replay_windows') != 50000):
        raise ValueError('Incomplete source restoration')
    return True


def warm_start_jitter(runner, experiment, checkpoint):
    source = validate_jitter(experiment)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Jitter source SHA mismatch')
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state['amp_identity'] != source.identity() or state['completed_updates'] != 2500 or state['iter'] != 2499:
        raise ValueError('Wrong source identity/update')
    rate = experiment.cfg['jitter']['learning_rate']
    for key in ('optimizer_state_dict', 'es_optimizer_state_dict'):
        if any(g['lr'] != rate for g in state[key]['param_groups']):
            raise ValueError('Source learning rate changed')
    proof = dict(**jitter_source(experiment.cfg), **restore_learning_state(runner, experiment, state, learning_rate=rate))
    if runner.amp_trainer.discriminator.style_floor != 0. or runner.amp_trainer.bridge_gradient_penalty != 1.:
        raise ValueError('Jitter intervention changed AMP mechanics')
    validate_jitter_certificate(experiment, proof)
    return proof


def validate_jitter_diagnostics(report, group, steps, num_envs):
    validate_sustain_diagnostics(report, 'control', steps, num_envs)
    if (report.get('smoothness_calls') != steps or
        report.get('physics_substeps') != steps*10 or
        not math.isclose(report.get('smoothness_scale', float('nan')),
                         .01*jitter_contract(group)['smoothness_scale'], rel_tol=1e-6)):
        raise ValueError('Wrong smoothness weight or actual reward/substep calls')
    for key in ('smoothness_cost_sum', 'physics_accel_squared_sum', 'physics_torque_delta_squared_sum'):
        if not math.isfinite(report.get(key, float('nan'))) or report[key] <= 0:
            raise ValueError('Missing actual jitter telemetry: '+key)
    return True
