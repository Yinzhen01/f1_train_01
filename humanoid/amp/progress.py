"""Rounds13/14: matched body/world velocity supervision, unchanged AMP."""
import copy
import hashlib
import math
from pathlib import Path

import torch

from .contact import apply_contact_config
from .recovery import centered_velocity_reward
from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment

PROGRESS_GROUPS = ('progress_body', 'progress_world')
SOURCE_CONFIG = 'configs/amp/lafan_walk02_contact_tail.json'
SOURCE_SHA = '4804076ff5f88be3dbfffc86f7a9e3e342b255107a2f780bd3c700d0f8cb512e'


def progress_contract(group):
    if group not in ('body', 'world'): raise ValueError('Unknown progress frame')
    return dict(group=group, round=13 if group == 'body' else 14,
        source_task='TASK_20260926_071', source_config=SOURCE_CONFIG,
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=2000,
        learning_rate=5e-5, episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.,
        velocity_frame=group, command_xy=[.45, 0.], progress_scale=2.,
        refine_slip_scale=-2., contact_force_tail_scale=-.0025)


def apply_progress_config(cfg, group):
    c = progress_contract(group)
    cfg = apply_contact_config(cfg, 'tail')
    if (cfg.commands.ranges.lin_vel_x != [.45, .45] or cfg.commands.ranges.lin_vel_y != [0., 0.] or
        cfg.rewards.scales.recovery_progress != 2.):
        raise ValueError('Unexpected source velocity command or reward scale')
    cfg.rewards.progress_velocity_frame = c['velocity_frame']
    return cfg


def progress_rewards(body_xy, world_xy, command_xy, frame):
    """Same reward function/scale; only its velocity coordinate changes."""
    if frame not in ('body', 'world') or body_xy.shape != world_xy.shape or body_xy.shape != command_xy.shape:
        raise ValueError('Invalid progress frame/shapes')
    body = centered_velocity_reward(body_xy, command_xy)
    world = centered_velocity_reward(world_xy, command_xy)
    return (body if frame == 'body' else world), body, world


def validate_progress(experiment):
    c = progress_contract(experiment.cfg.get('progress', {}).get('group'))
    if experiment.cfg['progress'] != c: raise ValueError('Progress contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg); expected.pop('contact_refinement')
    expected.update(experiment='f1_amp_walk02_progress_'+c['group'], progress=c, evaluation_updates=[2250])
    actual = copy.deepcopy(experiment.cfg)
    for key in ('scope', 'known_risks'): expected.pop(key, None); actual.pop(key, None)
    if expected != actual: raise ValueError('Change outside matched progress frame and2000 continuation')
    return source


def progress_source(cfg):
    c = cfg['progress']
    return dict(source_task=c['source_task'], source_sha256=c['source_checkpoint_sha256'],
        source_completed_updates=2000, target_completed_updates=2250, kind='matched_progress_frame',
        velocity_frame=c['velocity_frame'], episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.)


def validate_progress_certificate(experiment, continuation):
    validate_progress(experiment)
    if any(continuation.get(k) != v for k, v in progress_source(experiment.cfg).items()):
        raise ValueError('Wrong progress continuation proof')
    if (continuation.get('actor_and_discriminator_restored') is not True or
        continuation.get('optimizers_restored') is not True or continuation.get('replay_windows') != 50000):
        raise ValueError('Incomplete source restoration')
    return True


def warm_start_progress(runner, experiment, checkpoint):
    source = validate_progress(experiment)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Progress source SHA mismatch')
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state['amp_identity'] != source.identity() or state['completed_updates'] != 2000 or state['iter'] != 1999:
        raise ValueError('Wrong source identity/update')
    rate = experiment.cfg['progress']['learning_rate']
    for key in ('optimizer_state_dict', 'es_optimizer_state_dict'):
        if any(g['lr'] != rate for g in state[key]['param_groups']): raise ValueError('Source learning rate changed')
    proof = dict(**progress_source(experiment.cfg), **restore_learning_state(runner, experiment, state, learning_rate=rate))
    if runner.amp_trainer.discriminator.style_floor != 0. or runner.amp_trainer.bridge_gradient_penalty != 1.:
        raise ValueError('Progress intervention changed AMP mechanics')
    validate_progress_certificate(experiment, proof)
    return proof


def validate_progress_diagnostics(report, group, steps, num_envs):
    progress_contract(group)
    if (report.get('velocity_frame') != group or report.get('calls') != steps or
        report.get('env_steps') != steps*num_envs or
        not math.isclose(report.get('runtime_scale', float('nan')), .02, rel_tol=1e-6)):
        raise ValueError('Wrong actual progress frame, calls or dt-scaled reward')
    for field in ('body_sum', 'world_sum', 'selected_sum', 'abs_body_world_difference_sum'):
        if not math.isfinite(report.get(field, float('nan'))): raise ValueError('Nonfinite progress diagnostic')
    if report['abs_body_world_difference_sum'] <= 0 or report['selected_sum'] != report[group+'_sum']:
        raise ValueError('Progress frame selection was not exercised correctly')
    return True
