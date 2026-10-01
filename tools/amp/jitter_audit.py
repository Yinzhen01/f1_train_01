"""Only smoothness weight changes; telemetry is present in both paired groups."""
import copy
import numpy as np

from humanoid.amp.jitter import jitter_contract, SUBSTEP_GROUPS
from tools.amp.sustain_audit import validate_sustain_updates


def validate_jitter_updates(rows, formal=False):
    return validate_sustain_updates([dict(row, iteration=row['iteration']-250) for row in rows], formal)


def compare_environment(source, target, group, smoke=False):
    after = copy.deepcopy(target)
    if (source['rewards']['scales']['recovery_smoothness'] != -.02 or
        after['rewards']['scales']['recovery_smoothness'] != jitter_contract(group)['smoothness_scale']):
        raise ValueError('Wrong smoothness weight')
    after['rewards']['scales']['recovery_smoothness'] = -.02
    if group in SUBSTEP_GROUPS:
        contract = jitter_contract(group)['substep']
        if after.pop('substep_penalty', None) != contract:
            raise ValueError('Wrong substep contract')
        if group == 'substep_torque' and after['rewards']['scales'].pop('substep_torque', None) != contract['torque_scale']:
            raise ValueError('Wrong substep torque weight')
    if smoke:
        after['env']['num_envs'] = source['env']['num_envs']
    if source != after:
        raise ValueError('Environment changed beyond smoothness weight and smoke size')
    return True


def validate_substep_arrays(manifest, arrays):
    """Check every captured interval, including the surviving prefixes after 2s."""
    spec = manifest.get('substep_telemetry', {})
    group = manifest.get('identity', {}).get('experiment', '').replace('f1_amp_walk02_jitter_', '', 1)
    if (spec.get('physics_hz'), spec.get('samples_per_control'), spec.get('reward_used')) != (1000, 10, group in SUBSTEP_GROUPS):
        raise ValueError('Missing or changed substep capture contract')
    if group in SUBSTEP_GROUPS and spec.get('penalty') != jitter_contract(group)['substep']:
        raise ValueError('Wrong recorded substep penalty')
    for mode in manifest['modes']:
        dq = arrays[mode+'_dof_vel']; shape = dq.shape
        mask = arrays[mode+'_physics_valid']
        if mask.shape != shape[:2] or mask.dtype != np.bool_:
            raise ValueError('Invalid physical reset guard')
        active = mask & arrays[mode+'_valid']
        if not active.any(): raise ValueError('No real physical samples')
        if group in SUBSTEP_GROUPS:
            for key in ('acceleration_cost', 'torque_cost'):
                cost = arrays[mode+'_physics_'+key]
                if cost.shape != shape[:2] or not np.isfinite(cost).all() or (cost < 0).any():
                    raise ValueError('Invalid actual substep cost: '+key)
        for key, width in (('accel_squared', 12), ('accel_peak', 12),
                           ('torque_delta_squared', 12), ('foot_force_peak', 2)):
            x = arrays[mode+'_physics_'+key]
            if x.shape != shape[:2]+(width,) or not np.isfinite(x).all() or (x < 0).any():
                raise ValueError('Invalid substep interval values: '+key)
        mean_square = arrays[mode+'_physics_accel_squared']
        peak = arrays[mode+'_physics_accel_peak']
        endpoint = np.diff(np.concatenate((arrays[mode+'_initial_dof_vel'][None], dq)), axis=0)/.01
        # Jensen bound: a mean of squared substep accelerations cannot be smaller
        # than the square of their mean (the endpoint difference).
        if np.any((mean_square*1.001+.01 < endpoint**2)[active]):
            raise ValueError('Substep acceleration is inconsistent with endpoint velocity')
        if np.any((peak**2*1.001+.01 < mean_square)[active]):
            raise ValueError('Substep peak smaller than mean square')
        if group in SUBSTEP_GROUPS:
            previous_torque = np.concatenate((arrays[mode+'_initial_torque'][None], arrays[mode+'_torque']))
            for cost_key, squared, mean, normalizer in (
                ('acceleration_cost', mean_square, endpoint, spec['penalty']['acceleration_normalizer']),
                ('torque_cost', arrays[mode+'_physics_torque_delta_squared'], np.diff(previous_torque, axis=0)/10.,
                 spec['penalty']['torque_delta_normalizer'])):
                cost = arrays[mode+'_physics_'+cost_key]
                # Convex in delta, concave in delta-squared: two independent
                # Jensen bounds audit actual nonlinear substep capture.
                lower = (2*(np.sqrt(1+(mean/normalizer)**2)-1)).mean(-1)
                upper = (2*(np.sqrt(1+squared/normalizer**2)-1)).mean(-1)
                if np.any(((cost+1e-3 < lower) | (cost > upper+1e-3))[active]):
                    raise ValueError('Nonlinear substep cost violates physical bounds: '+cost_key)
    return True
