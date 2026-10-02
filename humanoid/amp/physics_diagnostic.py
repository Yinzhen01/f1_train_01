"""Bounded frozen-source physical interventions, separate from training tasks."""
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import re
import subprocess

from .evaluation import validate_evaluation_budget
from .jitter import SOURCE_CONFIG, SOURCE_SHA

GROUPS = ('original', 'velocity1', 'selfoff')
RAW_BODY_NAMES = ('base_link', 'left_ankle_roll_link', 'right_ankle_roll_link')
SHAPE_FIELDS = ('friction', 'rolling_friction', 'torsion_friction', 'restitution',
                'compliance', 'contact_offset', 'rest_offset', 'filter', 'thickness')
PHYSX_FIELDS = ('solver_type', 'num_position_iterations', 'num_velocity_iterations',
                'contact_offset', 'rest_offset', 'max_depenetration_velocity',
                'contact_collection')


def diagnostic_contract(group):
    if group not in GROUPS:
        raise ValueError('Unknown frozen physics diagnostic group')
    intervention = ({'sim.physx.num_velocity_iterations': 1} if group == 'velocity1'
                    else {'asset.self_collisions': 1} if group == 'selfoff' else {})
    return dict(group=group, source_task='TASK_20260926_110', source_config=SOURCE_CONFIG,
                source_checkpoint='model_8802500.pt', source_checkpoint_sha256=SOURCE_SHA,
                source_completed_updates=2500, no_training=True, raw_env=0,
                raw_body_names=list(RAW_BODY_NAMES), intervention=intervention)


def validate_source_state(state, identity):
    if (state.get('amp_identity') != identity or state.get('completed_updates') != 2500
            or state.get('iter') != 2499):
        raise ValueError('Physics diagnosis requires exact original control2500 state')
    return True


def validate_request(group, experiment, checkpoint_sha256, seed, num_envs, duration,
                     extended_validation):
    diagnostic_contract(group)
    validate_evaluation_budget(num_envs, duration, extended_validation)
    if (experiment != 'sustain_control' or checkpoint_sha256 != SOURCE_SHA or seed != 5
            or (not extended_validation and (num_envs, duration) != (2, 1.))):
        raise ValueError('Physics diagnosis permits source-only 2x1 smoke or fixed60 replay')
    return True


def apply_diagnostic_config(cfg, group):
    contract = diagnostic_contract(group)
    if (cfg.sim.dt != .001 or cfg.control.decimation != 10
            or cfg.sim.physx.num_velocity_iterations != 0 or cfg.asset.self_collisions != 0):
        raise ValueError('Unexpected original physical configuration')
    if group == 'velocity1':
        cfg.sim.physx.num_velocity_iterations = contract['intervention']['sim.physx.num_velocity_iterations']
    elif group == 'selfoff':
        cfg.asset.self_collisions = contract['intervention']['asset.self_collisions']
    return cfg


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite physical property readback')
        return result
    # pybind enums (ContactCollection) support int but may not be Real.
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError('Unsupported physical property type')


def physical_readback(env):
    """Read only. Missing optional API fields are unknown, never invented zeros."""
    gym, actor_env, actor = env.gym, env.envs[0], env.actor_handles[0]
    props = gym.get_actor_rigid_shape_properties(actor_env, actor)
    names = list(gym.get_actor_rigid_body_names(actor_env, actor))
    ranges = None
    if hasattr(gym, 'get_actor_rigid_body_shape_indices'):
        ranges = [dict(start=int(value.start), count=int(value.count))
                  for value in gym.get_actor_rigid_body_shape_indices(actor_env, actor)]
        if len(ranges) != len(names):
            raise ValueError('Rigid body/shape range readback mismatch')
        if any(row['start'] < 0 or row['count'] < 0 or row['start']+row['count'] > len(props)
               for row in ranges):
            raise ValueError('Out-of-range rigid shape mapping')
    shape = dict(body_names=names, body_shape_ranges=ranges,
                 properties=[dict(index=index, **{key: _scalar(getattr(prop, key, None))
                     for key in SHAPE_FIELDS}) for index, prop in enumerate(props)])
    actual_sim = gym.get_sim_params(env.sim)
    sim = dict(dt=float(actual_sim.dt), physx={key: _scalar(getattr(actual_sim.physx, key, None))
               for key in PHYSX_FIELDS})
    return dict(physics_sim_parameters=sim, physics_shape_properties=shape)


def cleared_history_diagnostics(env):
    """Prove native reset cleared histories; count signed zeros read-only."""
    import torch
    rows = {}
    for name in ('obs_history', 'critic_history'):
        value = torch.cat([tensor.flatten() for tensor in getattr(env, name)])
        nonzero = int(torch.count_nonzero(value).item())
        nonfinite = int((~torch.isfinite(value)).sum().item())
        if nonzero or nonfinite:
            raise ValueError('Native reset did not clear '+name)
        rows[name] = dict(elements=value.numel(), nonzero=nonzero, nonfinite=nonfinite,
                          negative_zero=int(torch.signbit(value).sum().item()))
    return rows


def reset_input_snapshot(env):
    """Capture actual injected state before reset() advances native physics."""
    result = {}
    for name in ('root_states', 'dof_pos', 'dof_vel', 'actions', 'last_actions',
                 'last_last_actions', 'commands', 'gait_start', 'episode_length_buf',
                 'phase_length_buf', 'rsi_indices'):
        value = getattr(env, name).detach().cpu().numpy().copy()
        if name == 'root_states':
            value[:, :3] -= env.env_origins.detach().cpu().numpy()
        result[name] = value.tolist()
    for name in ('obs_history', 'critic_history'):
        digest = hashlib.sha256()
        for tensor in getattr(env, name):
            value = tensor.detach().cpu().numpy().copy()
            # Native reset uses history *= 0: negative entries retain -0.0.
            # Equal numerical reset inputs must hash alike; do not mutate the
            # live buffers or round any nonzero value.
            value[value == 0] = 0
            digest.update(str((value.shape, str(value.dtype))).encode()+b'\0'+value.tobytes())
        result[name+'_sha256'] = digest.hexdigest()
    return result


def validate_smoke_certificate(certificate, commit, groups, source_sha):
    """Certificate-only commits are allowed, implementation changes are not."""
    from .scaled_experiment import implementation_fingerprint
    repo = Path(__file__).resolve().parents[2]
    flags = ('verified', 'no_training', 'all_finite', 'raw_verified', 'readback_verified',
             'reset_inputs_exact', 'cloud_execution_verified')
    if (any(certificate.get(key) is not True for key in flags)
            or certificate.get('source_checkpoint_sha256') != source_sha or source_sha != SOURCE_SHA
            or (certificate.get('num_envs'), certificate.get('duration_s')) != (2, 1.)
            or certificate.get('effectiveness_verified') is not False
            or certificate.get('dr_unlocked') is not False
            or certificate.get('implementation_fingerprint') != implementation_fingerprint(repo)):
        raise ValueError('Real-cloud physics diagnostic smoke gates incomplete or changed')
    smoke_commit = certificate.get('code_commit', '')
    if (not re.fullmatch('[0-9a-f]{40}', smoke_commit)
            or not re.fullmatch('[0-9a-f]{40}', commit)
            or not re.fullmatch('TASK_[0-9]{8}_[0-9]+', certificate.get('cloud_task_id', ''))):
        raise ValueError('Smoke cloud task/commit provenance missing')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', smoke_commit, commit],
                      cwd=str(repo), check=False).returncode != 0:
        raise ValueError('Smoke commit is not an ancestor of the evaluation checkout')
    actual_groups = certificate.get('groups', [])
    if (len(set(actual_groups)) != len(actual_groups) or set(actual_groups) != set(GROUPS)
            or not set(groups).issubset(set(actual_groups)) or not groups):
        raise ValueError('Smoke groups do not cover the registered physical interventions')
    artifacts = certificate.get('group_artifacts', {})
    if set(artifacts) != set(actual_groups) or any(
            not re.fullmatch('[0-9a-f]{64}', artifacts[group].get('sha256', ''))
            for group in actual_groups):
        raise ValueError('Smoke artifact checksums missing')
    return True
