"""Rounds17/18: matched heading2250 continuation, only progress weight may change."""
import copy
import hashlib
import math
from pathlib import Path

import torch

from .direction import apply_direction_config, validate_direction_diagnostics
from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment

SUSTAIN_GROUPS = ('sustain_control', 'sustain_progress3')
SOURCE_CONFIG = 'configs/amp/lafan_walk02_direction_heading.json'
SOURCE_SHA = '268a0e93db4fc80592713bdba16fc05174225182f193f666e84aaed5247edb62'


def sustain_contract(group):
    if group not in ('control', 'progress3'): raise ValueError('Unknown sustain group')
    return dict(group=group, round=17 if group == 'control' else 18,
        source_task='TASK_20260926_089', source_config=SOURCE_CONFIG,
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=2250,
        learning_rate=5e-5, episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.,
        world_fraction=0., heading_scale=-1.5, command_xy=[.45, 0.],
        progress_scale=2. if group == 'control' else 3.,
        refine_slip_scale=-2., contact_force_tail_scale=-.0025)


def apply_sustain_config(cfg, group):
    c = sustain_contract(group)
    cfg = apply_direction_config(cfg, 'heading')
    if cfg.rewards.scales.recovery_progress != 2.:
        raise ValueError('Unexpected source progress scale')
    cfg.rewards.scales.recovery_progress = c['progress_scale']
    return cfg


def validate_sustain(experiment):
    c = sustain_contract(experiment.cfg.get('sustain', {}).get('group'))
    if experiment.cfg['sustain'] != c: raise ValueError('Sustain contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg); expected.pop('direction')
    expected.update(experiment='f1_amp_walk02_sustain_'+c['group'], sustain=c, evaluation_updates=[2500])
    actual = copy.deepcopy(experiment.cfg)
    for key in ('scope', 'known_risks'): expected.pop(key, None); actual.pop(key, None)
    if expected != actual: raise ValueError('Change outside bounded progress weight and2250 continuation')
    return source


def sustain_source(cfg):
    c = cfg['sustain']
    return dict(source_task=c['source_task'], source_sha256=c['source_checkpoint_sha256'],
        source_completed_updates=2250, target_completed_updates=2500, kind='bounded_sustain',
        progress_scale=c['progress_scale'], world_fraction=0., heading_scale=-1.5,
        episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.)


def validate_sustain_certificate(experiment, continuation):
    validate_sustain(experiment)
    if any(continuation.get(k) != v for k, v in sustain_source(experiment.cfg).items()):
        raise ValueError('Wrong sustain continuation proof')
    if (continuation.get('actor_and_discriminator_restored') is not True or
        continuation.get('optimizers_restored') is not True or continuation.get('replay_windows') != 50000):
        raise ValueError('Incomplete source restoration')
    return True


def warm_start_sustain(runner, experiment, checkpoint):
    source = validate_sustain(experiment)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Sustain source SHA mismatch')
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state['amp_identity'] != source.identity() or state['completed_updates'] != 2250 or state['iter'] != 2249:
        raise ValueError('Wrong source identity/update')
    rate = experiment.cfg['sustain']['learning_rate']
    for key in ('optimizer_state_dict', 'es_optimizer_state_dict'):
        if any(g['lr'] != rate for g in state[key]['param_groups']): raise ValueError('Source learning rate changed')
    proof = dict(**sustain_source(experiment.cfg), **restore_learning_state(runner, experiment, state, learning_rate=rate))
    if runner.amp_trainer.discriminator.style_floor != 0. or runner.amp_trainer.bridge_gradient_penalty != 1.:
        raise ValueError('Sustain intervention changed AMP mechanics')
    validate_sustain_certificate(experiment, proof)
    return proof


def validate_sustain_diagnostics(report, group, steps, num_envs):
    c = sustain_contract(group)
    if not math.isclose(report.get('progress_scale', float('nan')), .01*c['progress_scale'], rel_tol=1e-6):
        raise ValueError('Wrong actual dt-scaled progress weight')
    # All other actual reward calls and raw body-only sums must match heading.
    normalized = dict(report, progress_scale=.02)
    return validate_direction_diagnostics(normalized, 'heading', steps, num_envs)
