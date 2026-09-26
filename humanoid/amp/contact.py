"""Rounds10-12: matched short-budget contact regularization, unchanged AMP."""
import copy
import hashlib
import math
from pathlib import Path

import torch

from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment

CONTACT_GROUPS = ('contact_control', 'contact_slip', 'contact_tail')
SOURCE_CONFIG = 'configs/amp/lafan_walk02_horizon_long.json'
SOURCE_SHA = '79196fee08379410c0c9218272e91ef95a20e185714a25bb59195635aedeaa1b'


def contact_contract(group):
    if group not in ('control', 'slip', 'tail'): raise ValueError('Unknown contact group')
    return dict(group=group, round={'control': 10, 'slip': 11, 'tail': 12}[group],
        source_task='TASK_20260924_113', source_config=SOURCE_CONFIG,
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=1750,
        learning_rate=5e-5, episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.,
        refine_slip_scale=-6. if group == 'slip' else -2.,
        contact_force_tail_scale=-.0025 if group == 'tail' else 0.,
        force_threshold_N=700., original_excess_cap_N=400.)


def apply_contact_config(cfg, group):
    c = contact_contract(group)
    if (cfg.env.episode_length_s != 6. or cfg.rewards.max_contact_force != 700 or
        cfg.rewards.scales.refine_slip != -2. or cfg.rewards.scales.feet_contact_forces != -.01):
        raise ValueError('Unexpected nominal source environment')
    cfg.env.episode_length_s = 60.
    cfg.rewards.scales.refine_slip = c['refine_slip_scale']
    cfg.rewards.scales.contact_force_tail = c['contact_force_tail_scale']
    return cfg


def force_tail_cost(force, threshold=700., excess_cap=400.):
    """Continue beyond the original plateau; no gait phase or contact threshold gate."""
    if force.ndim != 3 or force.shape[1:] != (2, 3) or threshold != 700. or excess_cap != 400.:
        raise ValueError('Expected two foot net force vectors and fixed source thresholds')
    return (torch.linalg.vector_norm(force, dim=-1)-threshold-excess_cap).clamp_min(0.).sum(-1)


def validate_contact(experiment):
    contract = contact_contract(experiment.cfg.get('contact_refinement', {}).get('group'))
    if experiment.cfg['contact_refinement'] != contract: raise ValueError('Contact contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg); expected.pop('horizon')
    expected.update(experiment='f1_amp_walk02_contact_'+contract['group'], contact_refinement=contract,
                    evaluation_updates=[2000])
    actual = copy.deepcopy(experiment.cfg)
    for key in ('scope', 'known_risks'): expected.pop(key, None); actual.pop(key, None)
    if expected != actual: raise ValueError('Change outside matched contact intervention and1750 continuation')
    return source


def contact_source(cfg):
    c = cfg['contact_refinement']
    return dict(source_task=c['source_task'], source_sha256=c['source_checkpoint_sha256'],
        source_completed_updates=1750, target_completed_updates=2000, kind='matched_contact_regularization',
        episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.,
        refine_slip_scale=c['refine_slip_scale'], contact_force_tail_scale=c['contact_force_tail_scale'])


def validate_contact_certificate(experiment, continuation):
    validate_contact(experiment)
    if any(continuation.get(k) != v for k, v in contact_source(experiment.cfg).items()):
        raise ValueError('Wrong contact continuation proof')
    if (continuation.get('actor_and_discriminator_restored') is not True or
        continuation.get('optimizers_restored') is not True or continuation.get('replay_windows') != 50000):
        raise ValueError('Incomplete source restoration')
    return True


def warm_start_contact(runner, experiment, checkpoint):
    source = validate_contact(experiment)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Contact source SHA mismatch')
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state['amp_identity'] != source.identity() or state['completed_updates'] != 1750 or state['iter'] != 1749:
        raise ValueError('Wrong source identity/update')
    rate = experiment.cfg['contact_refinement']['learning_rate']
    for key in ('optimizer_state_dict', 'es_optimizer_state_dict'):
        if any(g['lr'] != rate for g in state[key]['param_groups']): raise ValueError('Source learning rate changed')
    proof = dict(**contact_source(experiment.cfg), **restore_learning_state(runner, experiment, state, learning_rate=rate))
    if runner.amp_trainer.discriminator.style_floor != 0. or runner.amp_trainer.bridge_gradient_penalty != 1.:
        raise ValueError('Contact intervention changed AMP mechanics')
    validate_contact_certificate(experiment, proof)
    return proof


def validate_contact_diagnostics(report, group, steps, num_envs):
    c = contact_contract(group)
    expected = {'refine_slip': c['refine_slip_scale']*.01,
                'feet_contact_forces': -.01*.01, 'contact_force_tail': c['contact_force_tail_scale']*.01}
    for name, value in expected.items():
        if not math.isclose(report.get('runtime_scales', {}).get(name, float('nan')), value, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError('Wrong actual dt-scaled contact reward')
    calls = report.get('tail_calls', -1)
    cost = report.get('tail_cost_sum', float('nan'))
    count = report.get('tail_positive_env_steps', -1)
    if not math.isfinite(cost): raise ValueError('Nonfinite force tail')
    if group == 'tail':
        if calls != steps or cost <= 0 or not 0 < count <= steps*num_envs:
            raise ValueError('Force tail was not actually exercised on real transitions')
    elif calls != 0 or cost != 0 or count != 0:
        raise ValueError('Inactive force tail affected a control group')
    return True
