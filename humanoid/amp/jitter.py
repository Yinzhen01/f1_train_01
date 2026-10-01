"""Rounds 19/20: isolate action smoothness from the exact control2500 state."""
import copy
import hashlib
import math
from pathlib import Path

import torch

from .sustain import apply_sustain_config, validate_sustain_diagnostics
from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment

SUBSTEP_GROUPS = ('substep_accel', 'substep_torque')
JITTER_GROUPS = ('jitter_control', 'jitter_smooth') + tuple('jitter_'+g for g in SUBSTEP_GROUPS)
SOURCE_CONFIG = 'configs/amp/lafan_walk02_sustain_control.json'
SOURCE_SHA = '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31'


def jitter_contract(group):
    if group not in ('control', 'smooth') + SUBSTEP_GROUPS:
        raise ValueError('Unknown jitter group')
    result = dict(group=group, round={'control': 19, 'smooth': 20, 'substep_accel': 21, 'substep_torque': 22}[group],
        source_task='TASK_20260926_110', source_config=SOURCE_CONFIG,
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=2500,
        learning_rate=5e-5, episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.,
        smoothness_scale=-.2 if group == 'smooth' else -.02)
    if group in SUBSTEP_GROUPS:
        result['substep'] = dict(acceleration=group == 'substep_accel', acceleration_normalizer=100.,
            torque_delta_normalizer=2., torque_scale=-.2 if group == 'substep_torque' else 0.,
            reset_guard_steps=3, reduction='mean_of_robust_cost_per_joint_per_substep')
    return result


def apply_jitter_config(cfg, group):
    cfg = apply_sustain_config(cfg, 'control')
    if cfg.rewards.scales.recovery_smoothness != -.02:
        raise ValueError('Unexpected source smoothness')
    cfg.rewards.scales.recovery_smoothness = jitter_contract(group)['smoothness_scale']
    if group in SUBSTEP_GROUPS:
        cfg.substep_penalty = copy.deepcopy(jitter_contract(group)['substep'])
        if group == 'substep_torque':
            cfg.rewards.scales.substep_torque = cfg.substep_penalty['torque_scale']
    return cfg


def validate_jitter(experiment):
    c = jitter_contract(experiment.cfg.get('jitter', {}).get('group'))
    if experiment.cfg['jitter'] != c:
        raise ValueError('Jitter contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg); expected.pop('sustain')
    expected.update(experiment='f1_amp_walk02_jitter_'+c['group'], jitter=c, evaluation_updates=[2750])
    if c['group'] in SUBSTEP_GROUPS:
        # User lifted the 20-experiment cap on 2026-10-02. Each run is still
        # bounded to the unchanged smoke/formal budget and requires a new gate.
        expected['max_experiment_rounds'] = None
    actual = copy.deepcopy(experiment.cfg)
    for key in ('scope', 'known_risks'):
        expected.pop(key, None); actual.pop(key, None)
    if expected != actual:
        raise ValueError('Change outside registered jitter intervention and bounded 2500 continuation')
    return source


def jitter_source(cfg):
    c = cfg['jitter']
    result = dict(source_task=c['source_task'], source_sha256=c['source_checkpoint_sha256'],
        source_completed_updates=2500, target_completed_updates=2750, kind='bounded_jitter',
        smoothness_scale=c['smoothness_scale'], episode_length_s=60.,
        style_floor=0., bridge_gradient_penalty=1.)
    if c['group'] in SUBSTEP_GROUPS:
        result['substep'] = copy.deepcopy(c['substep'])
    return result


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
    if group in SUBSTEP_GROUPS:
        spec = jitter_contract(group)['substep']
        if report.get('substep_penalty') != spec or report.get('substep_acceleration_calls') != steps:
            raise ValueError('Missing actual substep acceleration reward calls/contract')
        if report.get('substep_torque_calls') != (steps if spec['torque_scale'] else 0):
            raise ValueError('Missing actual substep torque reward calls')
        for key, expected in (('acceleration_scale', -.005), ('torque_scale', .01*spec['torque_scale'])):
            if not math.isclose(report.get(key, float('nan')), expected, rel_tol=1e-6):
                raise ValueError('Wrong dt-scaled substep reward scale')
        for key in ('substep_acceleration_cost_sum', 'endpoint_acceleration_cost_sum',
                    'substep_torque_cost_sum', 'used_acceleration_cost_sum', 'substep_valid_intervals'):
            if not math.isfinite(report.get(key, float('nan'))) or report[key] <= 0:
                raise ValueError('Missing substep cost evidence: '+key)
        expected = report['substep_acceleration_cost_sum' if spec['acceleration'] else 'endpoint_acceleration_cost_sum']
        if not math.isclose(report['used_acceleration_cost_sum'], expected, rel_tol=1e-6):
            raise ValueError('Acceleration reward does not match registered definition')
        if report['substep_valid_intervals'] > steps*num_envs:
            raise ValueError('Invalid physical interval count')
    return True
