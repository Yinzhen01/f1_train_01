"""Frozen-policy PhysX interventions: matched prefixes and measured substeps.

This audit makes no checkpoint promotion or hardware claim. Contact forces are
net body readings, not identified contact pairs or confirmed ground friction.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.jitter import SOURCE_SHA
from humanoid.amp.recovery import foot_collision_vertices
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint
from tools.amp.analyze_jitter_events import (WINDOWS, backward_delta,
    command_geometry, window_summary)
from tools.amp.audit_jitter_substeps import measured_summary
from tools.amp.inspect_rollout import load_bundle, episode, heading_metrics
from tools.amp.verify_long_pair import initial_comparison

GROUPS = ('original', 'velocity1', 'selfoff')
RAW_NAMES = ('base_link', 'left_ankle_roll_link', 'right_ankle_roll_link')
FORCE_THRESHOLD = 1100.
SIM_FIELDS = ('solver_type', 'num_position_iterations', 'num_velocity_iterations',
              'contact_offset', 'rest_offset', 'max_depenetration_velocity', 'contact_collection')
SHAPE_FIELDS = ('friction', 'rolling_friction', 'torsion_friction', 'restitution',
                'compliance', 'contact_offset', 'rest_offset', 'filter', 'thickness')


def validate_environment(source, target, group):
    """An intervention is accepted only with its exact single physics change."""
    if group not in GROUPS:
        raise ValueError('Unknown physics diagnostic group')
    before, after = copy.deepcopy(source), copy.deepcopy(target)
    if before['sim']['physx']['num_velocity_iterations'] != 0 or before['asset']['self_collisions'] != 0:
        raise ValueError('Unexpected source physics')
    if group == 'velocity1':
        if after['sim']['physx']['num_velocity_iterations'] != 1:
            raise ValueError('Velocity intervention was not applied')
        after['sim']['physx']['num_velocity_iterations'] = 0
    if group == 'selfoff':
        if after['asset']['self_collisions'] != 1:
            raise ValueError('Self-collision intervention was not applied')
        after['asset']['self_collisions'] = 0
    if before != after:
        raise ValueError('Environment changed beyond the single registered intervention')
    return True


def validate_manifest(manifest, group, identity, smoke=False, fingerprint=None, expected_commit=None):
    from humanoid.amp.physics_diagnostic import diagnostic_contract
    if (manifest.get('physics_diagnostic') != diagnostic_contract(group) or
            manifest.get('diagnostic_only') is not True):
        raise ValueError('Missing or changed frozen physics diagnostic contract')
    if manifest.get('identity') != identity or manifest.get('checkpoint_sha256') != SOURCE_SHA:
        raise ValueError('Frozen source policy/data identity mismatch')
    expected = (1., 2, 100) if smoke else (60, 16, 100)
    protocol = 'physics_diagnostic_smoke' if smoke else 'fixed60_independent_mode_seeds'
    if (manifest.get('duration_s'), manifest.get('num_envs'), manifest.get('fps')) != expected:
        raise ValueError('Wrong diagnostic smoke/full environment count or duration')
    if (manifest.get('modes') != ['standing', 'reference'] or
        manifest.get('mode_seeds') != {'standing': 5, 'reference': 105} or
        manifest.get('evaluation_protocol') != protocol or
        manifest.get('policy_deterministic') is not True or
        manifest.get('effectiveness_verified') is not False or
        manifest.get('dr_unlocked') is not False):
        raise ValueError('Evaluation modes, seeds or deterministic/no-DR contract changed')
    if (not re.fullmatch('[0-9a-f]{40}', manifest.get('code_commit', '')) or
            not re.fullmatch('[0-9a-f]{64}', manifest.get('implementation_fingerprint', ''))):
        raise ValueError('Diagnostic code provenance is missing')
    if (fingerprint is not None and manifest['implementation_fingerprint'] != fingerprint or
            expected_commit is not None and manifest['code_commit'] != expected_commit):
        raise ValueError('Recorded implementation differs from audit checkout or expected commit')
    telemetry = manifest.get('substep_telemetry', {})
    if (telemetry.get('physics_hz'), telemetry.get('control_hz'),
            telemetry.get('samples_per_control'), telemetry.get('reward_used')) != (1000, 100, 10, False):
        raise ValueError('Frozen diagnostics require read-only native substep telemetry')
    validate_initialization(manifest)
    return True


def validate_initialization(manifest):
    initial = manifest.get('physics_initialization', {})
    if (set(initial) != {'protocol', 'warmup_control_dt', 'reset_inputs'} or
            initial['protocol'] != 'identical_reset_inputs_then_native_10ms_warmup' or
            initial['warmup_control_dt'] != .01 or set(initial['reset_inputs']) != {'standing', 'reference'}):
        raise ValueError('Pre-physics reset input protocol is missing or changed')
    numeric = ('root_states', 'dof_pos', 'dof_vel', 'actions', 'last_actions', 'last_last_actions',
               'commands', 'gait_start', 'episode_length_buf', 'phase_length_buf', 'rsi_indices')
    histories = ('obs_history_sha256', 'critic_history_sha256')
    for mode, snapshot in initial['reset_inputs'].items():
        if set(snapshot) != set(numeric+histories):
            raise ValueError('Incomplete reset input snapshot for '+mode)
        for name in numeric:
            value = np.asarray(snapshot[name])
            if not value.ndim or value.shape[0] != manifest['num_envs'] or not np.isfinite(value).all():
                raise ValueError('Invalid reset input '+mode+'/'+name)
        root = np.asarray(snapshot['root_states'])
        if root.shape != (manifest['num_envs'], 13) or not np.allclose(np.linalg.norm(root[:, 3:7], axis=1), 1., rtol=0., atol=1e-4):
            raise ValueError('Invalid injected root-state dimensions or quaternion')
        for name in histories:
            if not re.fullmatch('[0-9a-f]{64}', snapshot[name]):
                raise ValueError('Reset observation-history checksum is missing')
    return True


def compare_initialization(source, target):
    """Interventions may alter warmup response, but not the injected inputs."""
    validate_initialization(source); validate_initialization(target)
    fields = {}
    left = source['physics_initialization']['reset_inputs']; right = target['physics_initialization']['reset_inputs']
    for mode in source['modes']:
        for field in left[mode]:
            exact = (left[mode][field] == right[mode][field])
            fields[mode+'/'+field] = dict(exact_equal=bool(exact))
    if not all(value['exact_equal'] for value in fields.values()):
        raise ValueError('Pre-physics injected reset states/history are not exactly matched')
    return dict(reset_inputs_exact=True, fields=fields,
        native_warmup_seconds=.01, limitation='Equal injected states/history do not prove identical hidden solver state. '
            'Warmup-response and later trajectory differences can be caused by the physics intervention.')


def observed_initial_comparison(left, right):
    # Raw interval fields contain _initial_ inside their names; they are not
    # mode initialization snapshots and must not contaminate this comparison.
    prefixes = ('standing_initial_', 'reference_initial_')
    select = lambda arrays: {key:value for key,value in arrays.items() if key.startswith(prefixes)}
    return initial_comparison(select(left), select(right))


def validate_runtime(manifest):
    """Check actual simulator readback; shape-property null means unknown."""
    runtime = manifest['runtime']
    if (not np.isclose(runtime['control_dt'], .01, rtol=1e-6, atol=1e-10) or
            not np.isclose(runtime['physics_dt'], .001, rtol=1e-6, atol=1e-10) or
            manifest['environment']['control']['decimation'] != 10):
        raise ValueError('Actual policy/physics time steps changed')
    actual = runtime['physics_sim_parameters']; configured = manifest['environment']['sim']
    if set(actual) != {'dt', 'physx'} or set(actual['physx']) != set(SIM_FIELDS):
        raise ValueError('Incomplete actual simulator parameter readback')
    if not np.isclose(actual['dt'], configured['dt'], rtol=1e-6, atol=1e-10):
        raise ValueError('Actual physics time step differs from configuration')
    for name in SIM_FIELDS:
        if not isinstance(actual['physx'][name], (float, int)) or not np.isfinite(actual['physx'][name]):
            raise ValueError('Missing/nonfinite actual simulator field '+name)
        if not np.isclose(actual['physx'][name], configured['physx'][name], rtol=1e-6, atol=1e-10):
            raise ValueError('Actual simulator field differs from configuration: '+name)
    shapes = runtime['physics_shape_properties']
    if set(shapes) != {'body_names', 'body_shape_ranges', 'properties'}:
        raise ValueError('Incomplete rigid-shape property readback schema')
    bodies = shapes['body_names']; properties = shapes['properties']; ranges = shapes['body_shape_ranges']
    if len(set(bodies)) != len(bodies) or not all(name in bodies for name in RAW_NAMES):
        raise ValueError('Raw diagnostic bodies are missing from actual actor')
    if not properties:
        raise ValueError('Actor has no readable rigid shapes')
    unknown = {}
    for index, row in enumerate(properties):
        if set(row) != set(SHAPE_FIELDS+('index',)) or row['index'] != index:
            raise ValueError('Incomplete or unordered actual shape-property fields')
        for field in SHAPE_FIELDS:
            value = row[field]
            if value is None:
                unknown.setdefault(field, []).append(index)
            elif field == 'filter':
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError('Invalid collision filter readback')
            elif not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError('Nonfinite shape property '+field)
    if ranges is not None:
        if len(ranges) != len(bodies):
            raise ValueError('Body/shape range count mismatch')
        covered = []
        for row in ranges:
            if (set(row) != {'start', 'count'} or
                any(not isinstance(row[key], int) or isinstance(row[key], bool) for key in ('start', 'count')) or
                row['start'] < 0 or row['count'] < 0 or row['start']+row['count'] > len(properties)):
                raise ValueError('Invalid body shape range')
            covered.extend(range(row['start'], row['start']+row['count']))
        if sorted(covered) != list(range(len(properties))):
            raise ValueError('Body shape ranges overlap or fail to cover all actor shapes')
    return dict(simulator_parameters_verified=True, shape_count=len(properties),
        body_shape_mapping_available=ranges is not None, unknown_shape_fields=unknown,
        actual_shape_properties=shapes,
        limitation='Null attributes remain unknown; geometric asset inspection does not replace cooked-shape readback.')


def compare_runtime(source, target, group):
    before, after = copy.deepcopy(source), copy.deepcopy(target)
    old_shapes = before.pop('physics_shape_properties')
    new_shapes = after.pop('physics_shape_properties')
    if group == 'velocity1':
        if after['physics_sim_parameters']['physx']['num_velocity_iterations'] != 1:
            raise ValueError('Actual velocity iteration intervention is missing')
        after['physics_sim_parameters']['physx']['num_velocity_iterations'] = 0
    if before != after:
        raise ValueError('Actual DOF, controller, timing or simulator changed beyond intervention')
    if (old_shapes['body_names'] != new_shapes['body_names'] or
        old_shapes['body_shape_ranges'] != new_shapes['body_shape_ranges'] or
        len(old_shapes['properties']) != len(new_shapes['properties'])):
        raise ValueError('Actor rigid-shape layout changed')
    changed_filters = []
    for index, (old, new) in enumerate(zip(old_shapes['properties'], new_shapes['properties'])):
        if group == 'selfoff' and old['filter'] != new['filter']:
            # create_actor's supplied self-filter may override imported shape
            # filters. A known changed filter must match the requested mask=1.
            if new['filter'] != 1 or old['filter'] is None:
                raise ValueError('Unexpected self-collision filter readback change')
            changed_filters.append(dict(index=index, source=old['filter'], diagnostic=new['filter']))
            new['filter'] = old['filter']
        if old != new:
            raise ValueError('Friction, restitution, margin or other shape property changed')
    return dict(runtime_single_intervention_verified=True, changed_shape_filters=changed_filters,
        limitation='An unchanged or unreadable shape filter does not prove its effective actor-level pair filtering.')


def validate_raw(raw_velocity, raw_torque, raw_force, raw_state, initial_velocity,
                 initial_torque, endpoint_velocity, endpoint_torque, endpoint_feet,
                 squared, peak, torque_squared, force_peak, valid):
    """Verify env0 raw ten-substep records against independent interval fields."""
    n = len(raw_velocity)
    expected = ((raw_velocity, (n, 10, 12)), (raw_torque, (n, 10, 12)),
        (raw_force, (n, 10, 3, 3)), (raw_state, (n, 10, 3, 13)),
        (initial_velocity, (n, 12)), (initial_torque, (n, 12)))
    for value, shape in expected:
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid raw substep shape or nonfinite values')
    if valid.shape != (n,) or valid.dtype != np.bool_:
        raise ValueError('Invalid raw physical validity mask')
    if not valid.any():
        return dict(valid_intervals=0, samples=0, max_abs_errors=None,
                    unavailable_due_to_early_failure=True, body_names=list(RAW_NAMES))
    acceleration = np.diff(np.concatenate((initial_velocity[:, None], raw_velocity), axis=1), axis=1)/.001
    torque_delta = np.diff(np.concatenate((initial_torque[:, None], raw_torque), axis=1), axis=1)
    comparisons = ((raw_velocity[:, -1], endpoint_velocity, 'velocity endpoint'),
        (raw_torque[:, -1], endpoint_torque, 'torque endpoint'),
        (raw_state[:, -1, 1:], endpoint_feet, 'foot-state endpoint'),
        ((acceleration**2).mean(1), squared, 'acceleration mean square'),
        (np.abs(acceleration).max(1), peak, 'acceleration peak'),
        ((torque_delta**2).mean(1), torque_squared, 'torque-delta mean square'),
        (np.maximum(raw_force[:, :, 1:, 2], 0).max(1), force_peak, 'foot-force peak'))
    errors = {}
    for measured, recorded, name in comparisons:
        if measured.shape != recorded.shape or not np.allclose(measured[valid], recorded[valid], rtol=1e-4, atol=1e-3):
            raise ValueError('Raw data disagrees with recorded '+name)
        errors[name] = float(np.max(np.abs(measured[valid]-recorded[valid])))
    quaternion_norms = np.linalg.norm(raw_state[..., 3:7], axis=-1)
    if not np.allclose(quaternion_norms[valid], 1., rtol=0., atol=1e-4):
        raise ValueError('Invalid raw body quaternion')
    # This independently checks Jensen using the ten native velocity changes.
    endpoint = (endpoint_velocity-initial_velocity)/.01
    if np.any((endpoint[valid]**2 > squared[valid]*1.001+.1)):
        raise ValueError('Raw substep/endpoint acceleration violates Jensen bound')
    return dict(valid_intervals=int(valid.sum()), samples=int(valid.sum()*10),
        max_abs_errors=errors, body_names=list(RAW_NAMES),
        force_channel='net rigid-body force; contact pairs and friction semantics are not identified')


def validate_telemetry(manifest, arrays):
    result = {}
    for mode in manifest['modes']:
        dq = arrays[mode+'_dof_vel']; n, envs, width = dq.shape
        if envs != manifest['num_envs'] or width != 12:
            raise ValueError('Unexpected DOF telemetry dimensions')
        active = arrays[mode+'_valid'] & arrays[mode+'_physics_valid']
        if active.dtype != np.bool_ or active.shape != dq.shape[:2]:
            raise ValueError('Invalid all-environment physical validity')
        stats = {}
        for field, trailing in (('accel_squared', 12), ('accel_peak', 12),
                                 ('torque_delta_squared', 12), ('foot_force_peak', 2)):
            value = arrays[mode+'_physics_'+field]
            if value.shape != (n, envs, trailing) or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('Invalid all-environment interval statistics: '+field)
            stats[field] = value
        endpoint = np.diff(np.concatenate((arrays[mode+'_initial_dof_vel'][None], dq)), axis=0)/.01
        if np.any((endpoint**2 > stats['accel_squared']*1.001+.1)[active]):
            raise ValueError('All-environment acceleration violates Jensen bound')
        if np.any((stats['accel_squared'] > stats['accel_peak']**2*1.001+.1)[active]):
            raise ValueError('Substep peak is smaller than mean square')
        result[mode] = validate_raw(
            arrays[mode+'_physics_raw_velocity'], arrays[mode+'_physics_raw_torque'],
            arrays[mode+'_physics_raw_body_force'], arrays[mode+'_physics_raw_body_state'],
            arrays[mode+'_physics_interval_initial_velocity'], arrays[mode+'_physics_interval_initial_torque'],
            dq[:, 0], arrays[mode+'_torque'][:, 0], arrays[mode+'_foot_state'][:, 0],
            stats['accel_squared'][:, 0], stats['accel_peak'][:, 0],
            stats['torque_delta_squared'][:, 0], stats['foot_force_peak'][:, 0], active[:, 0])
        # The initial state is measured before each control interval, so retain
        # cross-interval continuity only before a failure, never across reset.
        previous = np.concatenate((arrays[mode+'_initial_dof_vel'][None, 0], dq[:-1, 0]))
        if not np.allclose(arrays[mode+'_physics_interval_initial_velocity'][active[:, 0]], previous[active[:, 0]], rtol=1e-5, atol=1e-5):
            raise ValueError('Interval initial velocity lost control-boundary continuity')
        previous_torque = np.concatenate((arrays[mode+'_initial_torque'][None, 0], arrays[mode+'_torque'][:-1, 0]))
        if not np.allclose(arrays[mode+'_physics_interval_initial_torque'][active[:, 0]], previous_torque[active[:, 0]], rtol=1e-5, atol=1e-5):
            raise ValueError('Interval initial torque lost control-boundary continuity')
    return result


def fast_geometry(vertices):
    return {name: dict(hull=mesh[ConvexHull(mesh).vertices],
        sole=mesh[mesh[:, 2] <= mesh[:, 2].min()+1e-5].mean(0)) for name, mesh in vertices.items()}


def fast_signals(data, manifest, geometry):
    """Same source-mesh height extrema and sole-centroid definition, less memory."""
    from scipy.spatial.transform import Rotation
    cfg = manifest['environment']; props = manifest['runtime']['dof_properties']
    default = [cfg['init_state']['default_joint_angles'][name] for name in manifest['dof_names']]
    target, excess = command_geometry(data['action'], default, cfg['control']['action_scale'], props['lower'], props['upper'])
    du = backward_delta(data['action'], data['initial']['action'])
    qvelocity = backward_delta(data['dof_pos'], data['initial']['dof_pos'])/.01
    height, speed = [], []
    for foot, name in enumerate(manifest['foot_names']):
        state = data['foot_state'][:, foot]
        rotation = Rotation.from_quat(state[:, 3:7]).as_matrix()
        shape = geometry[name]
        minimum = np.concatenate([(rotation[i:i+200, 2] @ shape['hull'].T).min(1)+state[i:i+200, 2]
                                   for i in range(0, len(state), 200)])
        arm = np.einsum('nij,j->ni', rotation, shape['sole'])
        velocity = state[:, 7:10]+np.cross(state[:, 10:13], arm)
        height.append(minimum); speed.append(np.linalg.norm(velocity[:, :2], axis=1))
    return dict(target=target, target_excess=excess, action_delta=du,
        acceleration_native=backward_delta(data['dof_vel'], data['initial']['dof_vel'])/.01,
        acceleration_qfd=backward_delta(qvelocity, qvelocity[0])/.01,
        torque_delta=backward_delta(data['torque'], data['initial']['torque']),
        sole_height=np.stack(height, axis=1), sole_speed=np.stack(speed, axis=1),
        pd_target_jump_nm=du*cfg['control']['action_scale']*np.asarray(manifest['runtime']['p_gains']))


def physics_summary(arrays, mode, index, data, signals, start, end, extra_mask=None):
    n = len(data['time'])
    mask = arrays[mode+'_physics_valid'][:n, index] & (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
    if extra_mask is not None:
        if extra_mask.shape != mask.shape or extra_mask.dtype != np.bool_:
            raise ValueError('Invalid additional physical interval mask')
        mask &= extra_mask
    if not mask.any():
        return None
    squared = arrays[mode+'_physics_accel_squared'][:n, index]
    force_peak = arrays[mode+'_physics_foot_force_peak'][:n, index]
    result = measured_summary(signals['acceleration_native'], squared,
        arrays[mode+'_physics_accel_peak'][:n, index], data['foot_force'][:, :, 2], force_peak, mask)
    burst = force_peak[mask].max(-1) > FORCE_THRESHOLD
    selected = squared[mask].astype(np.float64)
    energy = selected.sum()
    result.update(burst_control_intervals=int(burst.sum()),
        burst_interval_fraction=float(burst.mean()),
        burst_acceleration_energy_fraction=float(selected[burst].sum()/energy) if energy else None,
        quiet_acceleration_energy_fraction=float(selected[~burst].sum()/energy) if energy else None,
        substep_torque_delta_rms_nm=float(np.sqrt(arrays[mode+'_physics_torque_delta_squared'][:n, index][mask].mean())))
    return result


def worst_second(time, energy, valid, minimum_time=2.):
    """Sliding one-second windows; failed/missing samples cannot count as quiet."""
    energy = np.asarray(energy, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & (np.asarray(time) >= minimum_time-1e-8)
    if energy.shape != np.shape(time) or valid.shape != np.shape(time) or not np.isfinite(energy).all() or (energy < 0).any():
        raise ValueError('Invalid one-second diagnostic energy')
    if len(time) < 100:
        return None
    counts = np.convolve(valid.astype(int), np.ones(100, dtype=int), mode='valid')
    sums = np.convolve(np.where(valid, energy, 0.), np.ones(100), mode='valid')
    usable = counts == 100
    if not usable.any():
        return None
    chosen = int(np.argmax(np.where(usable, sums, -np.inf)))
    return dict(start_s=float(time[chosen]), end_exclusive_s=float(time[chosen]+1.),
                rms=float(np.sqrt(sums[chosen]/100)), samples=100)


def common_prefix_samples(left, right, terminal_guard=100):
    """Keep failures as failures, while excluding their terminal one second."""
    n = min(len(left['time']), len(right['time']))
    if not np.array_equal(left['time'][:n], right['time'][:n]):
        raise ValueError('Paired prefixes use different physical time ticks')
    left_failed = bool(left['initial']['failure'] or left['failure'].any())
    right_failed = bool(right['initial']['failure'] or right['failure'].any())
    # A later failure does not shorten an already-earlier surviving common
    # prefix unnecessarily: truncate each failed episode before intersecting.
    for data, failed in ((left, left_failed), (right, right_failed)):
        if failed: n = min(n, max(0, len(data['time'])-terminal_guard))
    return n


def quality_summary(data, signals, manifest, start, end):
    result = window_summary(data, signals, manifest, start, end)
    if result is not None:
        active = (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
        result.update(heading_metrics(data['root_state'][active]),
            world_x_displacement_m=float(data['root_state'][active][-1, 0]-data['root_state'][active][0, 0]),
            body_vy_abs_mean_m_s=float(np.abs(data['base_lin_vel'][active, 1]).mean()))
    return result


def paired_report(original, candidate, geometry):
    """Same time prefix and a shared non-burst mask; all 32 starts retained."""
    rows = []
    for mode in original['manifest']['modes']:
        for index in range(16):
            left = episode(original['arrays'], mode, index)
            right = episode(candidate['arrays'], mode, index)
            n = common_prefix_samples(left, right)
            data = [left, right]; cases = [original, candidate]
            summaries = []
            if n:
                end = float(left['time'][n-1])+.01
                shared_quiet = np.ones(n, dtype=bool)
                for case in cases:
                    shared_quiet &= case['arrays'][mode+'_physics_foot_force_peak'][:n, index].max(-1) <= FORCE_THRESHOLD
                    shared_quiet &= case['arrays'][mode+'_physics_valid'][:n, index]
                for d, case in zip(data, cases):
                    clipped = {key:(value[:n] if key not in ('initial',) else value) for key, value in d.items()}
                    s = fast_signals(clipped, case['manifest'], geometry)
                    summaries.append(dict(post2s=quality_summary(clipped, s, case['manifest'], 2., end),
                        physics_post2s=physics_summary(case['arrays'], mode, index, clipped, s, 2., end),
                        shared_quiet_physics_post2s=physics_summary(case['arrays'], mode, index, clipped, s, 2., end, shared_quiet),
                        windows={name:dict(quality=quality_summary(clipped, s, case['manifest'], *span),
                            physics=physics_summary(case['arrays'], mode, index, clipped, s, *span),
                            shared_quiet_physics=physics_summary(case['arrays'], mode, index, clipped, s, *span, shared_quiet))
                            for name, span in WINDOWS.items()}))
            else:
                end = 0.; summaries = [None, None]
            rows.append(dict(mode=mode, env=index, common_samples=n, common_end_s=end-.01 if n else 0.,
                original_failure=bool(left['initial']['failure'] or left['failure'].any()),
                candidate_failure=bool(right['initial']['failure'] or right['failure'].any()),
                terminal_guard_s=1., original=summaries[0], diagnostic=summaries[1]))
    means = {}
    for label in ('original', 'diagnostic'):
        means[label] = {}
        for section, keys in (('post2s', ('vx_mean', 'action_delta_rms', 'torque_delta_rms_nm',
            'acceleration_native_rms', 'contact_proxy_slip_mean_m_s', 'heading_rms_to_world_x_deg')),
            ('physics_post2s', ('substep_accel_rms', 'no_burst_substep_accel_rms', 'force_peak_p99_n')),
            ('shared_quiet_physics_post2s', ('samples', 'substep_accel_rms', 'substep_torque_delta_rms_nm'))):
            means[label][section] = {}
            for key in keys:
                selected = [r[label][section][key] for r in rows if r[label] is not None and
                    r[label][section] is not None and r[label][section][key] is not None]
                means[label][section][key] = dict(mean=float(np.mean(selected)) if selected else None,
                                                contributing_prefixes=len(selected))
    return dict(rows=rows, equal_initial_condition_means=means,
        same_physical_ticks_verified=True, terminal_failure_guard_s=1.,
        limitation='Both groups use a common valid prefix; failed episodes stay failed. A shared quiet mask excludes '
                    'any tick with >1100N in either group. Same-clock masking cannot establish equal gait phase.')


def summarize(rows):
    valid = [r for r in rows if r['post2s'] is not None]
    metrics = {}
    for key in ('vx_mean', 'action_delta_rms', 'torque_delta_rms_nm', 'acceleration_native_rms',
                'contact_proxy_slip_mean_m_s', 'heading_rms_to_world_x_deg', 'world_x_displacement_m'):
        values = [r['post2s'][key] for r in valid if r['post2s'].get(key) is not None]
        metrics[key] = float(np.mean(values)) if values else None
    for key in ('substep_accel_rms', 'no_burst_substep_accel_rms', 'burst_acceleration_energy_fraction',
                'force_peak_p99_n', 'substep_torque_delta_rms_nm'):
        values = [r['physics_post2s'][key] for r in rows if r['physics_post2s'] is not None and r['physics_post2s'][key] is not None]
        metrics[key] = float(np.mean(values)) if values else None
    return dict(total=len(rows), survived=sum(r['survived'] for r in rows),
                failed=sum(r['failure'] for r in rows), contributing_post2s_prefixes=len(valid),
                equal_initial_condition_mean=metrics, failed_prefixes_retained=True)


def inspect_prefixes(manifest, arrays):
    rows = []
    expected = int(round(manifest['duration_s']*100))
    for mode in manifest['modes']:
        for index in range(manifest['num_envs']):
            data = episode(arrays, mode, index)
            n = len(data['time'])
            failed = bool(data['initial']['failure'] or data['failure'].any())
            if n > expected or not (failed or n == expected):
                raise ValueError('Incomplete observed prefix without physical failure')
            if data['failure'].any() and (not data['failure'][-1] or data['failure'].sum() != 1):
                raise ValueError('Failure is not a single terminal pre-reset frame')
            rows.append(dict(mode=mode, env=index, samples=n, observed_s=n*.01,
                             survived=n == expected and not failed, failure=failed))
    return rows


def smoke_certificate(cases):
    commits = {case['manifest']['code_commit'] for case in cases.values()}
    fingerprints = {case['manifest']['implementation_fingerprint'] for case in cases.values()}
    if set(cases) != set(GROUPS) or len(commits) != 1 or len(fingerprints) != 1:
        raise ValueError('Smoke group/code provenance differs')
    for group, case in cases.items():
        validate_manifest(case['manifest'], group, cases['original']['manifest']['identity'], smoke=True)
        if any(row['valid_intervals'] <= 0 for row in case['raw_audit'].values()):
            raise ValueError('Smoke has no valid env0 raw physical samples')
        if (case['runtime_audit'].get('simulator_parameters_verified') is not True or
                any(not np.isfinite(value).all() for value in case['arrays'].values())):
            raise ValueError('Smoke readback or finite-state verification is missing')
        if group != 'original' and (case['pre_physics_initialization_comparison'].get('reset_inputs_exact') is not True or
                case['runtime_comparison'].get('runtime_single_intervention_verified') is not True):
            raise ValueError('Smoke reset or single-variable runtime comparison is missing')
    return dict(verified=True, no_training=True, all_finite=True,
        raw_verified=True, readback_verified=True, reset_inputs_exact=True,
        code_commit=commits.pop(), implementation_fingerprint=fingerprints.pop(),
        source_checkpoint_sha256=SOURCE_SHA, num_envs=2, duration_s=1., groups=list(GROUPS),
        group_artifacts={group:dict(sha256=case['bundle_sha256']) for group,case in cases.items()},
        group_evidence={group:{key:value for key,value in case.items()
            if key not in ('arrays', 'manifest', 'env0')} for group,case in cases.items()},
        cloud_execution_verified=False, effectiveness_verified=False, dr_unlocked=False,
        limitation='This verifies supplied bundles only. An independent cloud task/log audit must bind '
            'cloud_task_id and cloud_execution_verified before formal diagnostics. One-second smoke is '
            'not fixed60 motion-quality acceptance; native warmup response is allowed to differ.')


def source_archive_comparison(case, path, identity):
    manifest, arrays = load_bundle(path)
    if (manifest['checkpoint_sha256'] != SOURCE_SHA or manifest['identity'] != identity or
            (manifest['num_envs'], manifest['duration_s']) != (16, 60) or
            manifest.get('mode_seeds') != {'standing': 5, 'reference': 105} or
            manifest.get('evaluation_protocol') != 'fixed60_independent_mode_seeds'):
        raise ValueError('Archived source is not the original matched control2500 fixed60 bundle')
    validate_environment(manifest['environment'], case['manifest']['environment'], 'original')
    actual_runtime = copy.deepcopy(case['manifest']['runtime'])
    actual_runtime.pop('physics_sim_parameters'); actual_runtime.pop('physics_shape_properties')
    if actual_runtime != manifest['runtime']:
        raise ValueError('Original diagnostic changed archived source controller/DOF properties')
    proof = observed_initial_comparison(arrays, case['arrays'])
    if not proof['all_captured_fields_exact_equal']:
        raise ValueError('Original diagnostic post-warmup initial state differs from archived source')
    fields = {}
    for mode in manifest['modes']:
        for field in ('time', 'root_state', 'dof_pos', 'dof_vel', 'action', 'torque', 'base_lin_vel',
                      'base_ang_vel', 'foot_state', 'foot_force', 'key_positions_w', 'valid', 'failure'):
            key = mode+'_'+field; old = arrays[key]; new = case['arrays'][key]
            n = min(len(old), len(new)); same_shape = old.shape == new.shape
            fields[key] = dict(same_shape=same_shape, common_captured_ticks=n,
                exact_equal=same_shape and bool(np.array_equal(old, new)),
                max_abs_difference=float(np.max(np.abs(old[:n].astype(np.float64)-new[:n]))) if n else None)
    return dict(bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        archived_code_commit=manifest['code_commit'], post_warmup_initial=proof,
        trajectory_fields=fields, all_trajectory_fields_exact_equal=all(row['exact_equal'] for row in fields.values()),
        limitation='Trajectory differences are reported directly, not removed by smoothing or tolerance; '
            'identical recorded states do not establish hidden solver-state equality.')


def plot_window(cases, output, start, end, label):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 3, figsize=(17, 10), sharex=True, sharey='row')
    for col, group in enumerate(GROUPS):
        case = cases[group]; data, signals = case['env0']
        axes[0, col].set_title(group+' | reference env0')
        if signals is None:
            continue
        mask = (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
        t = data['time'][mask]
        axes[0, col].plot(t, data['base_lin_vel'][mask, 0]); axes[0, col].axhline(.45, c='gray', ls='--')
        axes[1, col].plot(t, np.sqrt((signals['acceleration_native'][mask]**2).mean(-1)), label='100Hz endpoint', lw=.7)
        axes[1, col].plot(t, np.sqrt(case['arrays']['reference_physics_accel_squared'][:len(data['time']), 0][mask].mean(-1)), label='measured 1kHz RMS', lw=.8)
        peak = case['arrays']['reference_physics_foot_force_peak'][:len(data['time']), 0][mask]
        for foot in range(2):
            axes[2, col].plot(t, peak[:, foot], label=('L', 'R')[foot]+' 1kHz peak', lw=.7)
            axes[3, col].plot(t, signals['sole_height'][mask, foot]*1000, label=('L', 'R')[foot])
        axes[4, col].plot(t, np.sqrt((signals['action_delta'][mask]**2).mean(-1)), lw=.8)
        for row in (1, 2): axes[row, col].legend(fontsize=7)
    for row, text in enumerate(('Body vx m/s', 'Acceleration rad/s2', 'Net foot Fz peak N', 'Sole mesh min z mm', 'Action delta RMS')):
        axes[row, 0].set_ylabel(text)
    for ax in axes.flat:
        ax.set_xlim(start, end); ax.grid(alpha=.2)
    axes[-1, 1].set_xlabel('Original physical time s; failed tails are missing')
    fig.suptitle('Same frozen policy and captured initial states; physics diagnostic only')
    fig.tight_layout(); fig.savefig(output/(label+'.png'), dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    for name in GROUPS+('output',): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--smoke', action='store_true', help='Verify 2x1 bundles and write certificate; never fixed60 acceptance')
    parser.add_argument('--source-bundle', type=Path, help='Original control2500 fixed60 archive; required outside smoke')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    torch.set_num_threads(2)
    experiment = ScaledExperiment(ROOT, ROOT/'configs/amp/lafan_walk02_sustain_control.json')
    fingerprint = implementation_fingerprint(ROOT)
    if not args.smoke and args.source_bundle is None:
        parser.error('--source-bundle is required for fixed60 read-only sampling comparison')
    geometry = None if args.smoke else fast_geometry(foot_collision_vertices(experiment.kinematics.path))
    cases = {}
    for group in GROUPS:
        path = getattr(args, group); manifest, arrays = load_bundle(path)
        validate_manifest(manifest, group, experiment.identity(), smoke=args.smoke,
                          fingerprint=fingerprint, expected_commit=args.expected_commit)
        runtime_audit = validate_runtime(manifest)
        raw_audit = validate_telemetry(manifest, arrays)
        if group != 'original':
            source = cases['original']
            validate_environment(source['manifest']['environment'], manifest['environment'], group)
            runtime_comparison = compare_runtime(source['manifest']['runtime'], manifest['runtime'], group)
            if (source['manifest']['code_commit'] != manifest['code_commit'] or
                source['manifest']['urdf_lf_sha256'] != manifest['urdf_lf_sha256']):
                raise ValueError('Code, actor DOF properties, PD, timing or asset identity changed')
            proof = compare_initialization(source['manifest'], manifest)
            warmup_response = observed_initial_comparison(source['arrays'], arrays)
        else:
            validate_environment(manifest['environment'], manifest['environment'], group)
            proof = None
            runtime_comparison = None
            warmup_response = None
        prefix_status = inspect_prefixes(manifest, arrays)
        if args.smoke:
            cases[group] = dict(manifest=manifest, arrays=arrays, raw_audit=raw_audit,
                runtime_audit=runtime_audit, runtime_comparison=runtime_comparison,
                pre_physics_initialization_comparison=proof, native_warmup_response=warmup_response,
                prefixes=prefix_status, bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            continue
        rows = []
        for mode in manifest['modes']:
            for index in range(16):
                data = episode(arrays, mode, index); n = len(data['time'])
                failure = bool(data['initial']['failure'] or data['failure'].any())
                survived = n == 6000 and not failure
                if not (failure or survived):
                    raise ValueError('Incomplete observed prefix without physical failure')
                if data['failure'].any() and (not data['failure'][-1] or data['failure'].sum() != 1):
                    raise ValueError('Physical failure is not a single terminal frame')
                signals = fast_signals(data, manifest, geometry) if n else None
                def window(start, end):
                    return window_summary(data, signals, manifest, start, end) if signals is not None else None
                post = quality_summary(data, signals, manifest, 2., 60.01) if n else None
                def physics(start, end):
                    return physics_summary(arrays, mode, index, data, signals, start, end) if signals is not None else None
                valid = arrays[mode+'_physics_valid'][:n, index]
                worst = {}
                if n:
                    energies = dict(physics_acceleration=arrays[mode+'_physics_accel_squared'][:n, index].mean(-1),
                        endpoint_acceleration=(signals['acceleration_native']**2).mean(-1),
                        action_delta=(signals['action_delta']**2).mean(-1))
                    worst = {key: worst_second(data['time'], energy, valid, minimum_time=0.) for key, energy in energies.items()}
                row = dict(mode=mode, env=index, survived=survived, failure=failure, observed_s=n*.01,
                    post2s=post, physics_post2s=physics(2., 60.01), worst_one_second=worst,
                    windows={name: dict(quality=window(*span), physics=physics(*span)) for name, span in WINDOWS.items()})
                rows.append(row)
                if mode == 'reference' and index == 0: env0 = (data, signals)
        cases[group] = dict(manifest=manifest, arrays=arrays, rows=rows, env0=env0,
            raw_audit=raw_audit, runtime_audit=runtime_audit, runtime_comparison=runtime_comparison,
            pre_physics_initialization_comparison=proof, native_warmup_response=warmup_response, summary=summarize(rows),
            bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(json.dumps(dict(group=group, summary=cases[group]['summary'])), flush=True)
    if args.smoke:
        certificate = smoke_certificate(cases)
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output/'certificate.json').open('x', encoding='utf-8') as stream:
            json.dump(certificate, stream, indent=2, allow_nan=False)
        print(json.dumps(dict(output=str(args.output/'certificate.json'), verified=True,
                              cloud_execution_verified=False, effectiveness_verified=False)), flush=True)
        return
    archive_check = source_archive_comparison(cases['original'], args.source_bundle, experiment.identity())
    paired = {group:paired_report(cases['original'], cases[group], geometry) for group in GROUPS[1:]}
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    for label, span in dict(WINDOWS, whole=(0., 60.)).items():
        plot_window(cases, args.output, *span, label)
    worst_windows = {}
    for group, case in cases.items():
        row = next(r for r in case['rows'] if r['mode'] == 'reference' and r['env'] == 0)
        worst = row['worst_one_second'].get('physics_acceleration')
        if worst is not None:
            worst_windows[group] = worst
            plot_window(cases, args.output, worst['start_s'], worst['end_exclusive_s'], 'worst1s_'+group)
    report = dict(source_checkpoint_sha256=SOURCE_SHA, policy_frozen=True, training_updates=0,
        original_source_archive_comparison=archive_check,
        cases={group:{k:v for k,v in case.items() if k not in ('arrays', 'env0')} for group, case in cases.items()},
        paired_to_original=paired,
        reference_env0_worst_windows=worst_windows, force_threshold_n=FORCE_THRESHOLD,
        effectiveness_verified=False, hardware_effectiveness_verified=False, dr_unlocked=False,
        limitations='32 physical-failure prefixes per group are retained; unequal prefix means are not survival acceptance. '
        'Paired comparisons use common prefixes with the last one second of failed episodes excluded; failures stay failures. '
        'Pre-physics reset inputs are exact; post-reset native warmup states are allowed to respond to the intervention. '
        'Modified physics is a mechanism diagnostic, not a trained or promoted policy. Same timestamps need not have the '
        'same gait phase after an intervention. Force threshold is a diagnostic classification, not a contact label. '
        'Net body forces do not identify collider pairs or complete friction forces; source-mesh height is not solver penetration. '
        'Raw ten-substep signals cover env0 only; interval statistics cover all sixteen environments per mode.')
    with (args.output/'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), policy_frozen=True, effectiveness_verified=False)), flush=True)


if __name__ == '__main__': main()
