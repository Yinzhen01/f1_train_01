"""Rounds15/16: limited direction intervention from the exact tail2000 state."""
import copy
import hashlib
import math
from pathlib import Path

import torch

from .contact import apply_contact_config
from .progress import SOURCE_CONFIG, SOURCE_SHA
from .recovery import centered_velocity_reward
from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment

DIRECTION_GROUPS = ('direction_mix15', 'direction_heading')


def direction_contract(group):
    if group not in ('mix15', 'heading'): raise ValueError('Unknown direction group')
    return dict(group=group, round=15 if group == 'mix15' else 16,
        source_task='TASK_20260926_071', source_config=SOURCE_CONFIG,
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=2000,
        learning_rate=5e-5, episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.,
        world_fraction=.15 if group == 'mix15' else 0.,
        heading_scale=-.5 if group == 'mix15' else -1.5,
        command_xy=[.45, 0.], progress_scale=2., refine_slip_scale=-2., contact_force_tail_scale=-.0025)


def apply_direction_config(cfg, group):
    c = direction_contract(group)
    cfg = apply_contact_config(cfg, 'tail')
    if (cfg.commands.ranges.lin_vel_x != [.45, .45] or cfg.commands.ranges.lin_vel_y != [0., 0.] or
        cfg.rewards.scales.recovery_progress != 2. or cfg.rewards.scales.refine_heading != -.5):
        raise ValueError('Unexpected source direction configuration')
    cfg.rewards.direction_world_fraction = c['world_fraction']
    cfg.rewards.scales.refine_heading = c['heading_scale']
    return cfg


def direction_rewards(body_xy, world_xy, command_xy, fraction):
    if fraction not in (0., .15) or body_xy.shape != world_xy.shape or body_xy.shape != command_xy.shape:
        raise ValueError('Invalid bounded direction mixture/shapes')
    body = centered_velocity_reward(body_xy, command_xy)
    world = centered_velocity_reward(world_xy, command_xy)
    return (body if fraction == 0 else .85*body+.15*world), body, world


def validate_direction(experiment):
    c = direction_contract(experiment.cfg.get('direction', {}).get('group'))
    if experiment.cfg['direction'] != c: raise ValueError('Direction contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg); expected.pop('contact_refinement')
    expected.update(experiment='f1_amp_walk02_direction_'+c['group'], direction=c, evaluation_updates=[2250])
    actual = copy.deepcopy(experiment.cfg)
    for key in ('scope', 'known_risks'): expected.pop(key, None); actual.pop(key, None)
    if expected != actual: raise ValueError('Change outside bounded direction and2000 continuation')
    return source


def direction_source(cfg):
    c = cfg['direction']
    return dict(source_task=c['source_task'], source_sha256=c['source_checkpoint_sha256'],
        source_completed_updates=2000, target_completed_updates=2250, kind='bounded_direction',
        world_fraction=c['world_fraction'], heading_scale=c['heading_scale'],
        episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.)


def validate_direction_certificate(experiment, continuation):
    validate_direction(experiment)
    if any(continuation.get(k) != v for k, v in direction_source(experiment.cfg).items()):
        raise ValueError('Wrong direction continuation proof')
    if (continuation.get('actor_and_discriminator_restored') is not True or
        continuation.get('optimizers_restored') is not True or continuation.get('replay_windows') != 50000):
        raise ValueError('Incomplete source restoration')
    return True


def warm_start_direction(runner, experiment, checkpoint):
    source = validate_direction(experiment)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Direction source SHA mismatch')
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state['amp_identity'] != source.identity() or state['completed_updates'] != 2000 or state['iter'] != 1999:
        raise ValueError('Wrong source identity/update')
    rate = experiment.cfg['direction']['learning_rate']
    for key in ('optimizer_state_dict', 'es_optimizer_state_dict'):
        if any(g['lr'] != rate for g in state[key]['param_groups']): raise ValueError('Source learning rate changed')
    proof = dict(**direction_source(experiment.cfg), **restore_learning_state(runner, experiment, state, learning_rate=rate))
    if runner.amp_trainer.discriminator.style_floor != 0. or runner.amp_trainer.bridge_gradient_penalty != 1.:
        raise ValueError('Direction intervention changed AMP mechanics')
    validate_direction_certificate(experiment, proof)
    return proof


def validate_direction_diagnostics(report, group, steps, num_envs):
    c = direction_contract(group)
    if (report.get('world_fraction') != c['world_fraction'] or report.get('calls') != steps or
        report.get('heading_calls') != steps or report.get('env_steps') != steps*num_envs or
        not math.isclose(report.get('progress_scale', float('nan')), .02, rel_tol=1e-6) or
        not math.isclose(report.get('heading_scale', float('nan')), .01*c['heading_scale'], rel_tol=1e-6)):
        raise ValueError('Wrong actual direction mixture, calls or dt-scaled rewards')
    for field in ('body_sum', 'world_sum', 'selected_sum', 'abs_body_world_difference_sum', 'heading_cost_sum'):
        if not math.isfinite(report.get(field, float('nan'))): raise ValueError('Nonfinite direction diagnostic')
    expected = (1-c['world_fraction'])*report['body_sum']+c['world_fraction']*report['world_sum']
    # GPU float32 reductions precede accumulation; linear sums need a small tolerance.
    if (report['abs_body_world_difference_sum'] <= 0 or report['heading_cost_sum'] <= 0 or
        not math.isclose(report['selected_sum'], expected, rel_tol=2e-6, abs_tol=steps*.001)):
        raise ValueError('Direction/heading rewards were not exercised correctly')
    return True
