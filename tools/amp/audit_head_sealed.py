"""Read-only formal head holdout audit; never solve, repair or simulate.

The seven received stage files are the evidence boundary. Embedded temporary
arm SHA claims are retained, not presented as independently received files.
Native aggregate telemetry covers all environments; raw ten-substep telemetry
covers env0 only. No physical effectiveness or domain-randomization admission
is granted by this archive/protocol audit.
"""
import copy
import hashlib
from pathlib import Path
import re

import numpy as np

from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.recovery import foot_collision_vertices
from humanoid.scripts.collect_amp_head_cohort import INITIAL_FIELDS
from humanoid.scripts.record_amp_head_sealed import (
    ARMS, ARM_SCHEMA, SCHEMA, audit_sealed_arm, audit_sealed_inputs,
    sealed_recorder_contract)
from tools.amp.analyze_jitter_events import WINDOWS
from tools.amp.audit_mu_candidates import overview, paired_masks, section
from tools.amp.audit_physics_diagnostic import (
    fast_geometry, validate_telemetry, worst_second)
from tools.amp.inspect_rollout import episode

ARM_START = '[head-sealed-arm-start] '
ARM_COMPLETE = '[head-sealed-arm-complete] '
PAIR_COMPLETE = '[head-sealed-pair-complete] '
SPANS = dict(whole60s=(.01, 60.01), post2s=(2., 60.01), **WINDOWS)
SOURCE_BODIES = ['left_knee_pitch_link', 'right_knee_pitch_link',
                 'left_ankle_roll_link', 'right_ankle_roll_link']
RAW_BODIES = ['base_link', 'left_ankle_roll_link', 'right_ankle_roll_link']


def require(condition, message):
    if not condition:
        raise ValueError('Head sealed audit: '+message)


def same(actual, expected, path=''):
    """Typed equality: a truthy int is not a native bool proof."""
    if isinstance(expected, dict):
        require(type(actual) is dict and set(actual) == set(expected), 'mapping differs '+path)
        for key, value in expected.items():
            same(actual[key], value, path+'.'+key)
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), 'list differs '+path)
        for i, value in enumerate(expected):
            same(actual[i], value, path+'.'+str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, 'value differs '+path)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_proof_equal(actual, expected):
    """Only a verified registered index's OS-dependent path may differ."""
    values = [copy.deepcopy(actual), copy.deepcopy(expected)]
    for value in values:
        require(type(value) is dict and type(value.get('derived_index')) is dict,
                'registered original source/index proof missing')
        path = value['derived_index'].pop('path', None)
        require(isinstance(path, str) and path.replace('\\', '/').endswith(
            '/resources/amp_inputs/source110_exclusions_v1.npz'), 'unregistered index path')
    same(*values, path='source exclusions')


def validate_arm_arrays(arrays, contract, body_count):
    """No dropped body, dtype coercion, invented raw samples or extra field."""
    require(type(arrays) is dict and all(isinstance(v, np.ndarray) and
        v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in arrays.values()),
        'nonfinite/non-numeric actual arm arrays')
    widths = dict(root_state=(13,), dof_pos=(12,), dof_vel=(12,), action=(12,), torque=(12,),
        base_lin_vel=(3,), base_ang_vel=(3,), foot_state=(2, 13), foot_force=(2, 3),
        key_positions_w=(body_count, 3), valid=(), failure=(), done=(), policy_raw_mu=(12,),
        native_episode=(), physics_valid=(), physics_accel_squared=(12,),
        physics_accel_peak=(12,), physics_torque_delta_squared=(12,), physics_foot_force_peak=(2,))
    raw = dict(physics_raw_velocity=(10, 12), physics_raw_torque=(10, 12),
        physics_raw_body_force=(10, 3, 3), physics_raw_body_state=(10, 3, 13),
        physics_interval_initial_velocity=(12,), physics_interval_initial_torque=(12,))
    time = arrays['reference_time']
    t, n = len(time), contract['num_envs']
    require(time.shape == (t,) and time.dtype == np.float64, 'physical time is not float64')
    expected = {'reference_time', 'reference_initial_full_obs', 'reference_initial_native_episode'}
    for key, shape in widths.items():
        name = 'reference_'+key
        dtype = np.bool_ if key in ('valid', 'failure', 'done', 'physics_valid') else (
            np.int64 if key == 'native_episode' else np.float32)
        require(arrays[name].shape == (t, n)+shape and arrays[name].dtype == dtype,
                'wrong native shape/dtype '+name)
        expected.add(name)
    for key in INITIAL_FIELDS:
        name = 'reference_initial_'+key
        dtype = np.bool_ if key in ('valid', 'failure') else np.float32
        require(arrays[name].shape == (n,)+widths[key] and arrays[name].dtype == dtype,
                'wrong native initial shape/dtype '+name)
        expected.add(name)
    for key, shape in raw.items():
        name = 'reference_'+key
        require(arrays[name].shape == (t,)+shape and arrays[name].dtype == np.float32,
                'wrong raw1ms shape/dtype '+name)
        expected.add(name)
    require(arrays['reference_initial_full_obs'].shape == (n, 3102) and
        arrays['reference_initial_full_obs'].dtype == np.float32 and
        arrays['reference_initial_native_episode'].shape == (n,) and
        arrays['reference_initial_native_episode'].dtype == np.int64, 'incomplete actual initial input/episode')
    require(set(arrays) == expected, 'missing/extra actual native arm array')


def validate_initialization(initialization, sealed_initialization):
    require(type(initialization) is dict and set(initialization) ==
        {'protocol', 'reset_inputs', 'cleared_histories'}, 'missing/extra native reset proof')
    same(initialization['protocol'],
         'original_reference_reset_then_native_10ms_zero_action_warmup', 'reset protocol')
    for key in ('reset_inputs', 'cleared_histories'):
        same(initialization[key], sealed_initialization[key], 'actual sealed '+key)
    reset = initialization['reset_inputs']
    shapes = dict(root_states=(8, 13), dof_pos=(8, 12), dof_vel=(8, 12), actions=(8, 12),
        last_actions=(8, 12), last_last_actions=(8, 12), commands=(8, 4), gait_start=(8,),
        episode_length_buf=(8,), phase_length_buf=(8,), rsi_indices=(8,))
    require(type(reset) is dict and set(reset) == set(shapes)|
        {'obs_history_sha256', 'critic_history_sha256'}, 'reset fields missing/extra')
    for key, shape in shapes.items():
        value = np.asarray(reset[key])
        require(value.shape == shape and value.dtype.kind in 'biuf' and np.isfinite(value).all(),
                'invalid native reset input '+key)
    for key in ('obs_history_sha256', 'critic_history_sha256'):
        require(isinstance(reset[key], str) and re.fullmatch('[0-9a-f]{64}', reset[key]),
                'missing cleared history hash')
    cleared = initialization['cleared_histories']
    require(type(cleared) is dict and set(cleared) == {'obs_history', 'critic_history'},
            'missing cleared histories')
    for row in cleared.values():
        require(type(row) is dict and set(row) == {'elements', 'nonzero', 'nonfinite', 'negative_zero'}
            and all(type(v) is int for v in row.values()) and row['elements'] > 0 and
            row['nonzero'] == row['nonfinite'] == 0 and 0 <= row['negative_zero'] <= row['elements'],
            'uncleared native input history')


def metrics_for_case(case, geometry):
    """Retain every episode and label missing windows; failures are never zeros."""
    rows = []
    for mode in case['manifest']['modes']:
        for i in range(case['manifest']['num_envs']):
            data = episode(case['arrays'], mode, i)
            values = {name: measured_section(data, case, mode, i, geometry, *span)
                      for name, span in SPANS.items()}
            valid = case['arrays'][mode+'_physics_valid'][:len(data['time']), i]
            worst = {}
            if len(data['time']):
                for name, field in (('acceleration', 'physics_accel_squared'),
                                    ('torque_delta', 'physics_torque_delta_squared')):
                    energy = case['arrays'][mode+'_'+field][:len(data['time']), i].mean(-1)
                    worst[name] = worst_second(data['time'], energy, valid, minimum_time=.01)
            failed = bool(data['initial']['failure'] or data['failure'].any())
            rows.append(dict(mode=mode, env=i, valid_ticks=len(data['time']), failed=failed,
                complete60s=bool(len(data['time']) == 6000 and not failed), spans=values,
                worst_one_second_including_terminal=worst))
    return rows


def measured_section(data, case, mode, index, geometry, start, end, **masks):
    value = section(data, case, mode, index, geometry, start, end, **masks)
    if value is not None:
        # The older report inferred mean availability from clipping. These new
        # archives actually export the unclipped mean, including saturated rows.
        inside = (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
        value['coverage'] = dict(observed_ticks=int(inside.sum()),
            requested_ticks=int(round((end-start)*100)),
            complete_window=int(inside.sum()) == int(round((end-start)*100)))
        raw = case['arrays'][mode+'_policy_raw_mu'][:len(data['time']), index]
        value['target']['raw_mu_available'] = True
        value['target']['raw_mu_equals_recorded_action'] = None if not inside.any() else bool(
            np.array_equal(raw[inside], data['action'][inside]))
    return value


def paired_metrics(source, candidate, geometry):
    """All eight pairs, identical masks; separate terminal-inclusive evidence."""
    rows = []
    for i in range(8):
        data = [episode(case['arrays'], 'reference', i) for case in (source, candidate)]
        masks = paired_masks(*data, source['arrays'], candidate['arrays'], 'reference', i)
        n = masks['samples']
        sides = {}
        for arm, actual, case in zip(ARMS, data, (source, candidate)):
            clipped = {key: value[:n] if key != 'initial' else value for key, value in actual.items()}
            sides[arm] = {name: measured_section(clipped, case, 'reference', i, geometry, *span,
                physical=masks['physical'], quiet=masks['quiet']) for name, span in SPANS.items()}
        rows.append(dict(env=i, common_valid_ticks=n, terminal_guard_s=1.,
            common_end_s=float(data[0]['time'][n-1]) if n else None,
            source_failed=bool(data[0]['initial']['failure'] or data[0]['failure'].any()),
            candidate_failed=bool(data[1]['initial']['failure'] or data[1]['failure'].any()),
            shared_physics_intervals=int(masks['physical'].sum()),
            shared_quiet_intervals=int(masks['quiet'].sum()), **sides))
    means = {}
    for span in SPANS:
        means[span] = {}
        for metric in overview(None):
            available = [(row['env'], overview(row['source'][span])[metric],
                          overview(row['candidate'][span])[metric]) for row in rows]
            available = [v for v in available if v[1] is not None and v[2] is not None]
            left = float(np.mean([v[1] for v in available])) if available else None
            right = float(np.mean([v[2] for v in available])) if available else None
            means[span][metric] = dict(source=left, candidate=right,
                contributing_envs=[v[0] for v in available],
                relative_change=None if left in (None, 0.) else (right-left)/left)
    return dict(rows=rows, means=means, all_eight_retained=True,
        terminal_guard_s=1., requested_windows={k: list(v) for k, v in SPANS.items()},
        limitation='Paired comparisons exclude each failed terminal second before intersection. '
            'Independent per-arm spans retain it. Missing windows are unavailable, not quiet; '
            'means use explicitly listed common contributing environments, not all eight by default.')


def audit_sealed_artifacts(manifest, arrays, *, cohorts, artifact, archive, identity,
        binding, commit, fingerprint, source_proof, source_manifest, exclusions,
        sealed_reference, pair_reference, pair_sha, head_sha, experiment=None, geometry=None):
    """Independent actual arrays/proofs readback; all fitting happened earlier."""
    require(set(cohorts) == {'train', 'validation', 'sealed'}, 'three actual cohorts mandatory')
    native_proof = cohorts['train']['manifest']['source_regression']
    source_proof_equal(native_proof, source_proof)
    for value in cohorts.values():
        source_proof_equal(value['manifest']['source_regression'], source_proof)
    inputs = audit_sealed_inputs(cohorts, artifact=artifact, identity=identity,
        code_commit=commit, implementation_fingerprint=fingerprint, exclusions=exclusions,
        source_proof=native_proof, source_manifest=source_manifest)
    cohort_telemetry = {}
    for role, item in cohorts.items():
        data = item['arrays']
        frames = len(data['reference_time'])
        same(item['manifest'].get('modes'), ['reference'], 'actual source-only '+role+' mode')
        for key, shape in dict(physics_raw_velocity=(10, 12), physics_raw_torque=(10, 12),
            physics_raw_body_force=(10, 3, 3), physics_raw_body_state=(10, 3, 13),
            physics_interval_initial_velocity=(12,), physics_interval_initial_torque=(12,)).items():
            value = data['reference_'+key]
            require(value.shape == (frames,)+shape and value.dtype == np.float32 and np.isfinite(value).all(),
                    'missing actual source-cohort raw1ms telemetry '+role+' '+key)
        same(item['manifest'].get('inference_capture'), dict(actor_module='actor.6', hidden_dim=128,
            action_dim=12, actual_forwards=frames, extra_forward=False, actual_output_exact=True,
            model_state_unchanged=True), 'actual '+role+' one-source-forward capture')
        cohort_telemetry[role] = validate_telemetry(item['manifest'], data)
    contract = sealed_recorder_contract('formal', 705, 8, 60.)
    constants = dict(contract, code_commit=commit, implementation_fingerprint=fingerprint,
        source_identity=identity, parent_checkpoint_sha256=binding['file_sha256'],
        parent_model_state_sha256=binding['model_state_sha256'],
        head_artifact_sha256=head_sha,
        candidate_model_state_sha256=archive['candidate_model_state_sha256'],
        cohorts=inputs['cohorts'], cohort_audit=inputs,
        evaluation_protocol='new_reference8_seed705_source_head_paired60',
        array_naming='source_reference_* and candidate_reference_* retain native reference field semantics',
        process_isolation='source and candidate run in separate fresh native processes/simulators',
        capture='all eight first-episode prefixes, terminal ticks and invalid tails; no survivor selection')
    for key, value in constants.items():
        same(manifest.get(key), value, 'pair.'+key)
    require(not {'iter', 'completed_updates', 'optimizer_state_dict'} & set(manifest),
            'physical pair claims PPO continuation')
    sealed = cohorts['sealed']
    same(sealed_reference, dict(artifact_name='model_7100006.pt', sha256=sealed['sha256'],
        seed=705, num_envs=8, duration_s=20, split='sealed', fit_eligible=False,
        episodes=inputs['audits']['sealed']['episodes'], deduplication_passed=True,
        holdout_not_used_for_fitting=True), 'sixth artifact binding')
    same(pair_reference, dict(artifact_name='model_7100007.pt', sha256=pair_sha, type=SCHEMA,
        mode='formal', seed=705, num_envs=8, duration_s=60,
        head_artifact_sha256=manifest['head_artifact_sha256'],
        source_checkpoint_sha256=binding['file_sha256'], sealed_cohort_sha256=sealed['sha256'],
        effectiveness_verified=False, dr_unlocked=False), 'seventh artifact binding')
    bodies = list(experiment.spec.body_names) if experiment is not None else SOURCE_BODIES
    same(sealed['manifest'].get('body_names'), bodies, 'sealed source key bodies')
    if experiment is not None:
        same(sealed['manifest'].get('dof_names'), list(experiment.spec.joint_names), 'sealed source DOFs')
        same(sealed['manifest'].get('foot_names'), RAW_BODIES[1:], 'sealed source feet')
        same(sealed['manifest'].get('urdf_lf_sha256'), experiment.kinematics.lf_sha256, 'sealed source asset SHA')
    t, n = sealed['arrays']['reference_valid'].shape
    for key, shape in (('reference_key_positions_w', (t, n, len(bodies), 3)),
                       ('reference_initial_key_positions_w', (n, len(bodies), 3))):
        require(sealed['arrays'][key].shape == shape and sealed['arrays'][key].dtype == np.float32,
                'wrong sealed source key-body dimensions')
    require(type(manifest.get('arms')) is dict and set(manifest['arms']) == set(ARMS),
            'both complete arms mandatory')
    require(type(arrays) is dict and all(key.startswith(('source_reference_', 'candidate_reference_'))
                                      for key in arrays), 'unregistered pair array route')
    cases, audits, logs = {}, {}, {}
    shas = []
    for arm in ARMS:
        embedded = manifest['arms'][arm]
        require(type(embedded) is dict and set(embedded) == {'manifest', 'arm_file_sha256', 'actual_audit'},
                'missing/extra embedded arm proof')
        arm_manifest = embedded['manifest']
        arm_arrays = {key[len(arm)+1:]: value for key, value in arrays.items() if key.startswith(arm+'_')}
        validate_arm_arrays(arm_arrays, contract, len(bodies))
        result = audit_sealed_arm(arm_manifest, arm_arrays, arm=arm, contract=contract,
            identity=identity, code_commit=commit, implementation_fingerprint=fingerprint,
            head_sha256=manifest['head_artifact_sha256'],
            candidate_model_sha256=archive['candidate_model_state_sha256'], inputs=inputs,
            sealed_arrays=sealed['arrays'], source_manifest=source_manifest)
        same(embedded['actual_audit'], result, arm+' actual arm proof')
        same(arm_manifest.get('head_archive_audit'), archive, arm+' strict head archive')
        same(arm_manifest.get('body_names'), bodies, arm+' source key bodies')
        same(arm_manifest.get('foot_names'), RAW_BODIES[1:], arm+' feet')
        if experiment is not None:
            same(arm_manifest.get('dof_names'), list(experiment.spec.joint_names), arm+' DOF order')
            same(arm_manifest.get('urdf_lf_sha256'), experiment.kinematics.lf_sha256, arm+' asset SHA')
        same(arm_manifest.get('no_dr'), assert_no_domain_randomization(arm_manifest['environment']), arm+' no DR')
        validate_initialization(arm_manifest.get('initialization'), sealed['manifest']['initialization'])
        same(arm_manifest.get('substep_telemetry'), dict(physics_hz=1000, control_hz=100,
            samples_per_control=10, raw_env=0, raw_body_names=RAW_BODIES, reward_used=False,
            reset_guard='native physics_valid excludes first three reset intervals',
            acceleration='native1ms dq/.001 mean squared and peak absolute',
            torque_delta='actual1ms torque difference mean squared; not derivative',
            foot_force='max positive NET footFz, not contact-pair identity'), arm+' raw telemetry protocol')
        pid = arm_manifest.get('process_id')
        require(type(pid) is int and pid > 0, 'missing actual native arm PID')
        sha = embedded['arm_file_sha256']
        require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'missing temporary arm SHA')
        shas.append(sha)
        telemetry = validate_telemetry(arm_manifest, arm_arrays)
        audits[arm] = dict(actual_arm=result, raw_telemetry=telemetry, process_id=pid,
                          temporary_arm_sha256_claim=sha, temporary_arm_bytes_independently_received=False)
        cases[arm] = dict(manifest=arm_manifest, arrays=arm_arrays)
        logs[arm] = dict(start=dict(type=ARM_SCHEMA, arm=arm, code_commit=commit,
            parent_checkpoint_sha256=binding['file_sha256'], head_artifact_sha256=manifest['head_artifact_sha256'],
            policy_model_state_sha256=arm_manifest['policy_model_state_sha256'], num_envs=8, seed=705,
            duration_s=60., control_dt=arm_manifest['runtime']['control_dt'],
            physics_dt=arm_manifest['runtime']['physics_dt'], no_training=True),
            complete=dict(type=ARM_SCHEMA, arm=arm, episodes=result['episodes'],
                policy_model_state_sha256=arm_manifest['policy_model_state_sha256'],
                effectiveness_verified=False, dr_unlocked=False))
    require(len(set(shas)) == 2, 'two arms reuse same temporary archive claim')
    same(manifest['initial_sealed_matches'], audits['source']['actual_arm']['initial_sealed_matches'],
         'pair all-eight sealed initial proof')
    same(audits['candidate']['actual_arm']['initial_sealed_matches'],
         audits['source']['actual_arm']['initial_sealed_matches'], 'source/head actual initials')
    if geometry is None:
        require(experiment is not None, 'physical metrics need actual source foot geometry')
        geometry = fast_geometry(foot_collision_vertices(experiment.kinematics.path))
    logs['pair_complete'] = dict(type=SCHEMA, code_commit=commit,
        parent_checkpoint_sha256=binding['file_sha256'], head_artifact_sha256=manifest['head_artifact_sha256'],
        episodes={arm: audits[arm]['actual_arm']['episodes'] for arm in ARMS},
        effectiveness_verified=False, dr_unlocked=False)
    return dict(cohort_inputs=inputs, source_cohort_raw_telemetry=cohort_telemetry,
        arms=audits, log_records=logs,
        metrics=dict(source=metrics_for_case(cases['source'], geometry),
            candidate=metrics_for_case(cases['candidate'], geometry),
            paired=paired_metrics(cases['source'], cases['candidate'], geometry)),
        all_actual_arrays_verified=True, all_eight_retained=True,
        effectiveness_verified=False, dr_unlocked=False,
        limitation='RSI-initial-state holdout is not new motion/task or hardware generalization. '
            'No per-decision full observations were exported; means are not independently re-forwarded. '
            'Raw 1ms series is env0 only; other environments contain measured interval aggregates. '
            'Embedded temporary arm SHA claims are not independently verified against received arm files. '
            'Exact observed initials do not include hidden PhysX contact state or guarantee future replay.')
