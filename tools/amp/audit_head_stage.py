"""Independent read-only audit of REAL native head-stage artifacts.

No solve/fit/candidate repair, training or simulator import. CPU re-forwards
verify every exported native full observation against the hash-bound source.
Offline mathematical gates do not establish physical quality or unlock DR.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.head_artifact import ORIGINAL110_PARENT, _load_parent, validate_head_artifact
from humanoid.amp.head_smoothing import (
    _admission, evaluate_head_candidate, validate_head_only_change)
from humanoid.amp.head_workflow import (
    audit_cohort, exclusion_add_cohort, fit_episodes, forward_parity, identity_digest)
from humanoid.amp.mu_temporal import state_fingerprint
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.native_hardware import validate_hardware_record
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint
from humanoid.scripts.collect_amp_head_cohort import (
    BUDGETS, SEEDS, SOURCE_ENDPOINT_SHA, INITIAL_FIELDS, PHYSICS_FIELDS,
    _read_bundle, registered_source_exclusions, validate_source_runtime,
    validate_source_forward_context)
from humanoid.scripts.record_amp_head_policy import (
    audit_rollout_arrays, initial_source_matches, recorder_contract)
from humanoid.scripts.run_amp_head_smooth import SMOKE_SCHEMA
from tools.amp.audit_head_sealed import (
    ARM_START, ARM_COMPLETE, PAIR_COMPLETE, audit_sealed_artifacts,
    metrics_for_case)

SCHEMA = 'head_stage_artifact_audit_v1'
START, COHORT, COMPLETE = ('[head-smooth-start] ', '[head-cohort-complete] ',
                            '[head-smooth-complete] ')
INDEX_SHA = 'cf621dc289c74796e37d4d6f547308a17808d4e96c53fb2edb87339f0b326df2'
POLICY_START, POLICY_COMPLETE = '[head-policy-start] ', '[head-policy-complete] '
HARDWARE = '[head-smooth-hardware] '
SOURCE_BODIES = ['left_knee_pitch_link', 'right_knee_pitch_link',
                 'left_ankle_roll_link', 'right_ankle_roll_link']


def require(condition, message):
    if not condition:
        raise ValueError('Head stage audit: '+message)


def _object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'duplicate JSON field '+key)
        value[key] = item
    return value


def _constant(value):
    raise ValueError('Head stage audit: nonfinite JSON number '+value)


def _json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _same(actual, expected, path='', approximate=False):
    """Typed JSON compare; only independent CPU f64 arithmetic gets tolerance."""
    if isinstance(expected, dict):
        require(type(actual) is dict and set(actual) == set(expected), 'JSON mapping differs '+path)
        for key in expected:
            _same(actual[key], expected[key], path+'.'+key, approximate)
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), 'JSON list differs '+path)
        for i, value in enumerate(expected):
            _same(actual[i], value, path+'.'+str(i), approximate)
    elif type(expected) is float and approximate:
        require(type(actual) in (float, int) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-10), 'numeric mismatch '+path)
    else:
        require(type(actual) is type(expected) and actual == expected, 'JSON value differs '+path)


def validate_platform(platform, expected_commit, mode):
    require(mode in BUDGETS and re.fullmatch('[0-9a-f]{40}', expected_commit) is not None,
            'invalid mode/commit')
    if 'data' in platform:
        require(platform.get('success') is True and type(platform['data']) is dict,
                'invalid raw platform info envelope')
        platform = platform['data']
    base, code = platform['taskBaseInfo'], platform['taskCodeInfo']
    task = base.get('taskId')
    require(isinstance(task, str) and re.fullmatch('TASK_[0-9]{8}_[0-9]{3}', task) is not None,
            'missing actual platform TaskId')
    require(str(base.get('userId')) == '4409' and str(base.get('taskStatus')) == '5' and
            base.get('goodsId') == 'ESKU000001' and type(base.get('gpuNum')) is int and
            base['gpuNum'] == 1, 'wrong owner/terminal status/resource/GPU count')
    code_owner = code.get('userId')
    require(code_owner in (None, '') or str(code_owner) == '4409',
            'nonblank platform code owner mismatch')
    require(code.get('taskId') == task and
            code.get('commitId') == expected_commit and str(code.get('codeType')) == '2',
            'platform code/owner/task/commit mismatch')
    script = code.get('startScript')
    require(isinstance(script, str) and not any(c in script for c in '\n\r;|&`'),
            'not a single registered gm-run script')
    arguments = shlex.split(script)
    require(len(arguments) >= 2 and arguments[0] == 'gm-run' and
            arguments[1].replace('\\', '/').endswith('/humanoid/scripts/run_amp_head_smooth.py'),
            'wrong native entry point')
    parsed, flags = {}, []
    i = 2
    while i < len(arguments):
        token = arguments[i]
        require(token.startswith('--'), 'unexpected native positional argument')
        if '=' in token:
            key, value = token.split('=', 1)
        elif token == '--headless':
            key, value = token, True
        else:
            require(i+1 < len(arguments) and not arguments[i+1].startswith('--'),
                    'missing native option value')
            key, value = token, arguments[i+1]
            i += 1
        require(key not in parsed, 'duplicate native option '+key)
        parsed[key] = value
        flags.append(key)
        i += 1
    require(parsed.get('--mode') == mode and parsed.get('--expected-commit') == expected_commit and
            parsed.get('--headless') is True and '--source-endpoint' not in parsed,
            'native startScript role/commit/fixed-index mismatch')
    allowed = {'--mode', '--expected-commit', '--headless', '--checkpoint-file', '--smoke-certificate'}
    require(set(flags) <= allowed and (('--smoke-certificate' in parsed) == (mode == 'formal')),
            'unregistered native options or reused smoke certificate')
    require(not any(isinstance(v, str) and ('://' in v or '?' in v) for v in parsed.values()),
            'URL/credential-like native argument is forbidden')
    return dict(task_id=task, owner='4409', terminal_status='5', goods_id='ESKU000001',
                gpu_count=1, code_commit=expected_commit, mode=mode,
                code_owner_metadata=code_owner, code_owner_metadata_present=code_owner not in (None, ''))


def read_log(path):
    raw = Path(path).read_bytes()
    try:
        text, binary = raw.decode('utf-8'), False
    except UnicodeDecodeError:
        text, binary = raw.decode('utf-8', errors='backslashreplace'), True
    if Path(path).suffix == '.json':
        envelope = _json(text)
        require(envelope.get('success') is True and type(envelope.get('data')) is str,
                'invalid full cloud-log envelope')
        text = envelope['data'].replace('\\n', '\n')
    controls = sum(ord(c) < 32 and c not in '\t\r\n\x1b' for c in text)
    return text, dict(raw_log_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw),
                      escaped_invalid_utf8=binary, control_bytes=controls)


def marker_records(text, marker):
    decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant)
    rows = []
    for match in re.finditer(re.escape(marker), text):
        try:
            value, size = decoder.raw_decode(text[match.end():])
        except json.JSONDecodeError as error:
            raise ValueError('Head stage audit: malformed native marker '+marker) from error
        require(type(value) is dict, 'marker must be a JSON object')
        rows.append((match.start(), match.end()+size, value))
    return rows


def validate_log(text, report, audits, task_id, mode, diagnostics, policy=None, sealed=None):
    starts, cohorts, ends = (marker_records(text, marker) for marker in (START, COHORT, COMPLETE))
    require(sealed is None or (mode == 'formal' and policy is not None),
            'sealed log requires admitted formal physical stage')
    splits = ('train', 'validation', 'sealed') if sealed is not None else ('train', 'validation')
    require(len(starts) == len(ends) == 1 and len(cohorts) == len(splits),
            'missing/duplicate native START/COHORT/COMPLETE')
    _same(starts[0][2], report['start'], 'native START')
    _same(ends[0][2], report, 'native COMPLETE')
    hardware = marker_records(text, HARDWARE)
    require(validate_hardware_record(report['start'].get('native_hardware')) is True,
            'missing/invalid native START hardware snapshot')
    require(len(hardware) == 1, 'missing/duplicate native hardware marker')
    _same(hardware[0][2], report['start']['native_hardware'], 'native hardware snapshot')
    require(hardware[0][1] <= starts[0][0], 'native hardware must be captured before START')
    require(starts[0][1] <= cohorts[0][0] < cohorts[0][1] <= cohorts[1][0] <
            cohorts[1][1] <= ends[0][0], 'native stage order mismatch')
    for split, row in zip(splits, cohorts):
        _same(row[2], dict(split=split, mode=mode, fit_eligible=audits[split]['fit_eligible'],
                          episodes=audits[split]['episodes'],
                          deduplication_passed=audits[split]['deduplication']['passed']),
              'actual '+split+' COMPLETE')
    policy_starts, policy_ends = (marker_records(text, marker) for marker in
                                  (POLICY_START, POLICY_COMPLETE))
    require(len(policy_starts) == len(policy_ends) == (1 if policy is not None else 0),
            'missing/duplicate/unbound native policy recorder START/COMPLETE')
    if policy is not None:
        _same(policy_starts[0][2], policy['start'], 'actual native recorder START')
        _same(policy_ends[0][2], policy['complete'], 'actual native recorder COMPLETE')
        require(cohorts[1][1] <= policy_starts[0][0] < policy_starts[0][1] <=
                policy_ends[0][0] < policy_ends[0][1] <= ends[0][0],
                'native recorder must follow train/validation and precede stage COMPLETE')
    arm_starts, arm_ends, pair_ends = (marker_records(text, marker) for marker in
                                     (ARM_START, ARM_COMPLETE, PAIR_COMPLETE))
    require(len(arm_starts) == len(arm_ends) == (2 if sealed is not None else 0) and
            len(pair_ends) == (1 if sealed is not None else 0),
            'missing/duplicate/unbound sealed arm/pair markers')
    if sealed is not None:
        require(policy_ends[0][1] <= cohorts[2][0] < cohorts[2][1] <= arm_starts[0][0],
                'sealed cohort must follow original32 physical recorder')
        previous = cohorts[2][1]
        for i, arm in enumerate(('source', 'candidate')):
            _same(arm_starts[i][2], sealed['log_records'][arm]['start'], 'native sealed '+arm+' START')
            _same(arm_ends[i][2], sealed['log_records'][arm]['complete'], 'native sealed '+arm+' COMPLETE')
            require(previous <= arm_starts[i][0] < arm_starts[i][1] <= arm_ends[i][0],
                    'fresh sealed source/candidate process order differs')
            previous = arm_ends[i][1]
        _same(pair_ends[0][2], sealed['log_records']['pair_complete'], 'native sealed pair COMPLETE')
        require(previous <= pair_ends[0][0] < pair_ends[0][1] <= ends[0][0],
                'sealed pair must complete before stage COMPLETE')
    exits = list(re.finditer(r'(?m)^【SDK】训练进程状态码：([0-9]+)\r?$', text))
    completions = list(re.finditer(r'(?m)^\[SDK\]\[INFO\] [0-9 :\-]+ '
        r'Task\(([^)]+)\) status updated to: Completed , ret:(True|False)\r?$', text))
    require(len(exits) == len(completions) == 1 and exits[0][1] == '0' and
            completions[0][1] == task_id and completions[0][2] == 'True' and
            ends[0][1] < exits[0].start() < completions[0].start(),
            'missing/mismatched native SDK exit0/task Completed receipt')
    numbers = 7000000 if mode == 'smoke' else 7100000
    uploads = {}
    for offset in range(1, 8 if sealed is not None else (6 if policy is not None else 5)):
        name = 'model_%d.pt' % (numbers+offset)
        found = list(re.finditer(r'(?m)^\[SDK\]\[INFO\] [0-9 :\-]+ PT file \('
            +re.escape(name)+r'\) uploaded successfully\r?$', text))
        require(found and all(starts[0][0] < m.start() < exits[0].start() for m in found),
                'missing actual uploaded '+name+' receipt')
        uploads[name] = [[m.start(), m.end()] for m in found]
    require(not re.search(r'\[f1-amp-update\]|\[f1-amp-complete\]|Learning iteration', text),
            'offline stage contains PPO training markers')
    scanned, warnings = text, []
    spans = [(lower, upper) for lower, upper, _ in
             starts+cohorts+ends+hardware+policy_starts+policy_ends+arm_starts+arm_ends+pair_ends]
    trace_pattern = r'Traceback \(most recent call last\):\r?\n((?:[ \t].*\r?\n)+)([^\r\n]+)'
    traces = list(re.finditer(trace_pattern, text))
    require(len(traces) == text.count('Traceback (most recent call last):'),
            'truncated/unparsed cloud traceback')
    for trace in traces:
        frames = re.findall(r'File "([^"]+)"', trace[1])
        require(frames and not any(re.search(r'[/\\](humanoid|isaacgym|torch|numpy)[/\\]', f)
                                  for f in frames), 'native source/runtime traceback')
        external_sdk = any(re.search(r'[/\\](gradmotion|pika|oss2|requests|urllib3)[/\\]', f)
                           for f in frames)
        require(external_sdk or trace.end() < starts[0][0] or trace.start() > ends[0][1],
                'unclassified exception inside native stage')
        spans.append((trace.start(), trace.end()))
        warnings.append(dict(kind='external_sdk_exception' if external_sdk else 'unclassified_wrapper_exception',
                             character_scope=[trace.start(), trace.end()], cause_verified=False))
    for lower, upper in sorted(spans, reverse=True):
        scanned = scanned[:lower]+' '*(upper-lower)+scanned[upper:]
    require(not re.search(r'CUDA out of memory|\bOOM\b|\bNonfinite\b|\bNaN\b|\bInf\b', scanned,
                          re.IGNORECASE), 'native nonfinite/OOM evidence')
    position = 0
    for line in scanned.splitlines(keepends=True):
        bad = re.search(r'\b(?:[A-Za-z_]+(?:Error|Exception)|ERROR|WARNING|UserWarning|FutureWarning)\b|'
                        r'(?i:error:|task is failed)', line)
        if bad:
            inside = starts[0][1] <= position < ends[0][0]
            sdk = '[SDK]' in line or '【SDK】' in line
            warning = re.search(r'\b(?:WARNING|UserWarning|FutureWarning)\b', line)
            require(not inside or sdk or warning, 'unscoped native error line')
            warnings.append(dict(kind='sdk_or_wrapper_message' if sdk or not inside else 'native_warning',
                                 character_scope=[position, position+len(line)], cause_verified=False))
        position += len(line)
    clean = not warnings and not diagnostics['escaped_invalid_utf8'] and not diagnostics['control_bytes']
    return dict(start_singleton=True, complete_singleton=True, cohort_completions=len(splits),
                native_recorder_verified=policy is not None,
                native_hardware_log_bound=True,
                native_sealed_pair_verified=sealed is not None,
                sdk_native_exit_code=0, sdk_completed_task=task_id, upload_receipts=uploads,
                logs_clean=clean, wrapper_warnings=warnings,
                raw_log_preserved=True, **diagnostics)


def source_policy(state, repo, fingerprint):
    """Construct the ORIGINAL Python architecture directly; no Isaac env import."""
    spec = importlib.util.spec_from_file_location('head_stage_cpu_actor',
                         Path(repo)/'humanoid/algo/ppo/actor_critic_dh.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = state['model_state_dict']
    dimensions = lambda prefix: [int(model[prefix+'.%d.weight' % i].shape[0]) for i in (0, 2, 4)]
    # Native architecture assigns Normal.set_default_validate_args=False
    # (a class attribute, not a function call). Preserve that global side effect
    # around this audit-only construction without changing the original source.
    normal = torch.distributions.Normal
    validation_owned = 'set_default_validate_args' in normal.__dict__
    validation_default = normal.__dict__.get('set_default_validate_args')
    try:
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            policy = module.ActorCriticDH(235, 47, int(model['critic.0.weight'].shape[1]), 12,
                actor_hidden_dims=dimensions('actor'), critic_hidden_dims=dimensions('critic'),
                state_estimator_hidden_dims=dimensions('state_estimator'),
                in_channels=int(model['long_history.0.weight'].shape[1]),
                kernel_size=[int(model['long_history.%d.weight' % i].shape[2]) for i in (0, 2)],
                filter_size=[int(model['long_history.%d.weight' % i].shape[0]) for i in (0, 2)],
                stride_size=[3, 2], lh_output_dim=int(model['long_history.7.weight'].shape[0]))
    finally:
        if validation_owned:
            normal.set_default_validate_args = validation_default
        else:
            del normal.set_default_validate_args
    policy.load_state_dict(model, strict=True)
    policy.eval()
    require(state_fingerprint(policy.state_dict()) == fingerprint,
            'constructed original architecture/model fingerprint mismatch')
    return policy


def _source_proof(actual, expected):
    actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
    for value in (actual, expected):
        require(type(value) is dict and type(value.get('derived_index')) is dict,
                'missing registered source-index proof')
        path = value.get('derived_index', {}).pop('path', None)
        require(isinstance(path, str) and path.replace('\\', '/').endswith(
            '/resources/amp_inputs/source110_exclusions_v1.npz'), 'wrong registered source-index path')
    _same(actual, expected, 'actual source exclusions')


def audit_policy_rollout(path, reference, *, mode, identity, binding, commit,
                         fingerprint, artifact, archive, source_proof,
                         source_manifest, exclusions, experiment=None):
    """Re-audit actual pre-reset states and telemetry; not independent simulation."""
    manifest, arrays = _read_bundle(path, torch)
    contract = recorder_contract(mode, 5, 2 if mode == 'smoke' else 16,
                                 1. if mode == 'smoke' else 60.)
    for key, value in contract.items():
        _same(manifest.get(key), value, 'recorder.'+key)
    head_sha = None if mode == 'smoke' else reference['head_artifact_sha256']
    model_sha = binding['model_state_sha256'] if mode == 'smoke' else archive['candidate_model_state_sha256']
    constants = dict(code_commit=commit, implementation_fingerprint=fingerprint,
        identity=identity, source_identity=identity, parent_checkpoint_sha256=binding['file_sha256'],
        parent_model_state_sha256=binding['model_state_sha256'], head_artifact_sha256=head_sha,
        policy_model_state_sha256=model_sha, policy_variant='source_smoke_probe' if mode == 'smoke'
            else 'offline_final_head_candidate', head_archive_audit=None if mode == 'smoke' else archive,
        policy_deterministic=True, evaluation_protocol='head_source_smoke' if mode == 'smoke'
            else 'head_fixed32_source_matched60',
        capture='pre-reset; retain full failing first-episode prefixes and invalid tails',
        action_semantics='policy_raw_mu is actual unclipped deterministic mean; action is actual clipped PD command')
    for key, value in constants.items():
        _same(manifest.get(key), value, 'recorder.'+key)
    if mode == 'formal':
        require(artifact['provenance']['offline_admitted'] is True,
                'unadmitted head entered formal physical recorder')
    _source_proof(manifest.get('source_regression'), source_proof)
    no_dr = assert_no_domain_randomization(manifest['environment'])
    _same(manifest.get('no_dr'), no_dr, 'actual recorder no-DR gate')
    runtime = validate_source_runtime(manifest['environment'], manifest['runtime'],
                                     source_manifest, contract)
    _same(manifest.get('source_runtime_proof'), runtime, 'actual recorder native runtime')
    body_names = list(experiment.spec.body_names) if experiment is not None else SOURCE_BODIES
    _same(manifest.get('body_names'), body_names, 'actual source-protocol recorder bodies')
    _same(manifest.get('substep_telemetry'), dict(physics_hz=1000, control_hz=100,
        samples_per_control=10, raw_env=0,
        raw_body_names=['base_link', 'left_ankle_roll_link', 'right_ankle_roll_link'], reward_used=False,
        reset_guard='native physics_valid excludes first three reset intervals',
        acceleration='native1ms velocity difference/.001 mean squared and peak absolute',
        torque_delta='actual1ms torque difference mean squared; not derivative',
        foot_force='max positive NET footFz, not contact-pair identity'), 'native recorder telemetry protocol')
    if experiment is not None:
        _same(manifest.get('dof_names'), list(experiment.spec.joint_names), 'recorder actual DOF order')
        _same(manifest.get('body_names'), list(experiment.spec.body_names), 'recorder actual body order')
        _same(manifest.get('foot_names'), ['left_ankle_roll_link', 'right_ankle_roll_link'],
              'recorder actual feet')
        _same(manifest.get('urdf_lf_sha256'), experiment.kinematics.lf_sha256, 'recorder asset SHA')
    require(all(value.dtype.kind in 'biuf' and np.isfinite(value).all() for value in arrays.values()),
            'nonfinite/non-numeric actual recorder arrays')
    expected_keys = set()
    widths = dict(root_state=(13,), dof_pos=(12,), dof_vel=(12,), action=(12,), torque=(12,),
        base_lin_vel=(3,), base_ang_vel=(3,), foot_state=(2, 13), foot_force=(2, 3),
        key_positions_w=(len(body_names), 3), valid=(), failure=(), done=(), policy_raw_mu=(12,),
        physics_valid=(), physics_accel_squared=(12,), physics_accel_peak=(12,),
        physics_torque_delta_squared=(12,), physics_foot_force_peak=(2,))
    raw_shapes = dict(physics_raw_velocity=(10, 12), physics_raw_torque=(10, 12),
        physics_raw_body_force=(10, 3, 3), physics_raw_body_state=(10, 3, 13),
        physics_interval_initial_velocity=(12,), physics_interval_initial_torque=(12,))
    for name in contract['modes']:
        t, n = len(arrays[name+'_time']), contract['num_envs']
        expected_keys.add(name+'_time')
        require(arrays[name+'_time'].dtype == np.float64, 'recorder physical time must retain float64')
        for key, shape in widths.items():
            expected_keys.add(name+'_'+key)
            data = arrays[name+'_'+key]
            dtype = np.bool_ if key in ('valid', 'failure', 'done', 'physics_valid') else np.float32
            require(data.shape == (t, n)+shape and data.dtype == dtype,
                    'wrong actual recorder shape/dtype '+name+'_'+key)
        for key in INITIAL_FIELDS:
            expected_keys.add(name+'_initial_'+key)
            data = arrays[name+'_initial_'+key]
            dtype = np.bool_ if key in ('valid', 'failure') else np.float32
            require(data.shape == (n,)+widths[key] and data.dtype == dtype,
                    'wrong actual recorder initial shape/dtype '+name+'_'+key)
        for key, shape in raw_shapes.items():
            expected_keys.add(name+'_'+key)
            require(arrays[name+'_'+key].shape == (t,)+shape and
                    arrays[name+'_'+key].dtype == np.float32, 'wrong actual native raw telemetry '+key)
        mask = arrays[name+'_valid']
        clip = manifest['environment']['normalization']['clip_actions']
        require(np.array_equal(arrays[name+'_action'][mask],
                np.clip(arrays[name+'_policy_raw_mu'], -clip, clip)[mask]),
                'actual recorder action is not the clipped deterministic mean')
    require(set(arrays) == expected_keys, 'missing/extra actual native recorder array')
    episodes = audit_rollout_arrays(arrays, contract)
    _same(manifest.get('episodes'), episodes, 'actual recorder all first-episode prefixes')
    # Use the existing strict source-native substep/endpoint/Jensen/continuity
    # checker; this verifies archived telemetry, never simulates new physics.
    from tools.amp.audit_physics_diagnostic import validate_telemetry
    telemetry = validate_telemetry(manifest, arrays)
    initialization = manifest.get('initialization')
    require(type(initialization) is dict and set(initialization) ==
            {'protocol', 'reset_inputs', 'cleared_histories', 'initial_source_matches'} and
            initialization['protocol'] == 'identical_reset_inputs_then_native_10ms_warmup',
            'actual recorder reset protocol missing')
    for name in contract['modes']:
        initial = {key: arrays[name+'_initial_'+key] for key in INITIAL_FIELDS}
        matched = initial_source_matches(initial, name, exclusions, enforce=mode == 'formal')
        _same(initialization['initial_source_matches'][name], matched, 'recorder initial binding '+name)
        reset = initialization['reset_inputs'][name]
        reset_shapes = dict(root_states=(n, 13), dof_pos=(n, 12), dof_vel=(n, 12), actions=(n, 12),
            last_actions=(n, 12), last_last_actions=(n, 12), commands=(n, 4), gait_start=(n,),
            episode_length_buf=(n,), phase_length_buf=(n,), rsi_indices=(n,))
        require(set(reset) == set(reset_shapes)|{'obs_history_sha256', 'critic_history_sha256'},
                'actual native reset-input fields missing/extra')
        for key, shape in reset_shapes.items():
            value = np.asarray(reset[key])
            require(value.shape == shape and value.dtype.kind in 'biuf' and np.isfinite(value).all(),
                    'invalid actual native reset input '+key)
        for key in ('obs_history_sha256', 'critic_history_sha256'):
            require(isinstance(reset[key], str) and re.fullmatch('[0-9a-f]{64}', reset[key]) is not None,
                    'missing cleared native history hash')
        cleared = initialization['cleared_histories'][name]
        require(set(cleared) == {'obs_history', 'critic_history'}, 'missing native cleared histories')
        for key in cleared:
            row = cleared[key]
            require(set(row) == {'elements', 'nonzero', 'nonfinite', 'negative_zero'} and
                all(type(v) is int for v in row.values()) and row['elements'] > 0 and
                row['nonzero'] == row['nonfinite'] == 0 and 0 <= row['negative_zero'] <= row['elements'],
                'native reset retained nonzero/nonfinite history')
    base = 7000000 if mode == 'smoke' else 7100000
    expected_reference = dict(artifact_name='model_%d.pt' % (base+5), sha256=file_sha(path),
        type=contract['type'], mode=mode, num_envs=contract['num_envs'], duration_s=contract['duration_s'],
        head_artifact_sha256=head_sha, source_checkpoint_sha256=binding['file_sha256'],
        policy_model_state_sha256=model_sha, evaluation_protocol=constants['evaluation_protocol'], episodes=episodes)
    _same(reference, expected_reference, 'stage actual fifth policy artifact binding')
    start = dict(type=contract['type'], mode=mode, code_commit=commit,
        parent_checkpoint_sha256=binding['file_sha256'], head_artifact_sha256=head_sha,
        policy_model_state_sha256=model_sha, num_envs=contract['num_envs'], duration_s=contract['duration_s'],
        control_dt=manifest['runtime']['control_dt'], physics_dt=manifest['runtime']['physics_dt'],
        parent_completed_updates=2500, no_training=True, effectiveness_verified=False, dr_unlocked=False)
    complete = dict(type=contract['type'], mode=mode, parent_checkpoint_sha256=binding['file_sha256'],
        head_artifact_sha256=head_sha, policy_model_state_sha256=model_sha, episodes=episodes,
        effectiveness_verified=False, dr_unlocked=False)
    return dict(start=start, complete=complete, episodes=episodes, all_arrays_verified=True,
                raw_telemetry=telemetry,
                runtime_verified=True, initial_binding_verified=True, source_only=mode == 'smoke',
                limitation='Archived native pre-reset telemetry is re-audited, not new simulation. '
                    'No rollout full observations were exported, so this does not independently '
                    're-forward every closed-loop policy mean or prove dynamics/hardware quality.')


def _parity_record(value, arrays, capture, hardware):
    required = ('full_obs_verified', 'hidden_verified', 'action_verified', 'max_abs_error',
                'max_hidden_error', 'max_raw_mu_error', 'actual_rows', 'forward_batches',
                'policy_unchanged', 'observations', 'batch_size', 'numerical_context')
    require(type(value) is dict and set(value) == set(required), 'missing native source parity fields')
    for key in ('full_obs_verified', 'hidden_verified', 'action_verified', 'policy_unchanged'):
        require(value[key] is True, 'native source parity flag is not true')
    for key in ('max_abs_error', 'max_hidden_error', 'max_raw_mu_error'):
        require(type(value[key]) in (float, int) and math.isfinite(value[key]) and
                0 <= value[key] < .001, 'native source parity error exceeds bound')
    require(value['max_abs_error'] == max(value['max_hidden_error'], value['max_raw_mu_error']),
            'native source parity error arithmetic mismatch')
    ticks, count = arrays['reference_full_obs'].shape[:2]
    rows = int(ticks*count)
    require(type(value['actual_rows']) is int and value['actual_rows'] == rows and
            type(value['batch_size']) is int and value['batch_size'] == count and
            type(value['forward_batches']) is int and value['forward_batches'] == ticks and
            value['observations'] == 'actual archived 66x47 native inputs, not reconstructed',
            'native source parity row/batch protocol mismatch')
    require(type(capture) is dict, 'missing native collector inference capture')
    context = capture.get('numerical_context')
    require(validate_source_forward_context(context, native=True) is True,
            'invalid native collector numerical context')
    require(validate_source_forward_context(value['numerical_context'], native=True) is True,
            'invalid native parity numerical context')
    _same(capture, dict(actor_module='actor.6', hidden_dim=128, action_dim=12,
        actual_forwards=int(ticks), extra_forward=False, actual_output_exact=True,
        model_state_unchanged=True, numerical_context=context), 'native one-forward capture')
    _same(value['numerical_context'], context, 'native collector/parity numerical context')
    for key in ('torch_version', 'torch_cuda_version'):
        _same(context[key], hardware[key], 'native numerical context/hardware '+key)


def audit_head_stage(*, mode, source_checkpoint, train, validation, head, report,
                     cloud_log, platform_info, expected_commit, repo=ROOT,
                     expected_parent=ORIGINAL110_PARENT, policy_rollout=None,
                     sealed_cohort=None, sealed_policy=None):
    """Audit actual files. Explicit synthetic parent override is TEST-ONLY."""
    synthetic = dict(expected_parent) != dict(ORIGINAL110_PARENT)
    platform_bytes = Path(platform_info).read_bytes()
    platform_sha = hashlib.sha256(platform_bytes).hexdigest()
    platform = validate_platform(_json(platform_bytes.decode('utf-8')), expected_commit, mode)
    source, binding = _load_parent(source_checkpoint, expected_parent)
    identity = source['amp_identity']
    experiment = None
    if not synthetic:
        experiment = ScaledExperiment(repo, Path(repo)/binding['source_config'])
        _same(identity, experiment.identity(), 'fixed original source identity')
        require(subprocess.run(['git', 'merge-base', '--is-ancestor', expected_commit, 'HEAD'],
            cwd=str(repo), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
            'cloud code commit is not an ancestor of the current audit checkout')
    fp = implementation_fingerprint(repo)
    paths = dict(train=Path(train), validation=Path(validation), head=Path(head), report=Path(report))
    require(len({path.resolve() for path in paths.values()}) == 4, 'four stage paths must be distinct')
    hashes = {name: file_sha(path) for name, path in paths.items()}
    require(hashes['train'] != hashes['validation'], 'same train/validation artifact')
    packed = torch.load(io.BytesIO(paths['report'].read_bytes()), weights_only=True, map_location='cpu')
    require(type(packed) is dict and set(packed) == {'report_json'} and type(packed['report_json']) is str,
            'not the actual head-stage PT report_json')
    actual = _json(packed['report_json'])
    expected_fields = {'start', 'cohort_sha256', 'source_parity', 'effectiveness_verified',
        'dr_unlocked', 'solver_runs', 'admitted_to_physical_test', 'float64_fit', 'exported_head',
        'status', 'head_sha256', 'archive_audit', 'frozen_audit', 'provenance_report_json'}
    require(type(actual) is dict, 'malformed fitted stage report')
    sealed_fields = {'sealed_cohort', 'sealed_policy'} & set(actual)
    require(mode != 'smoke' or not sealed_fields,
            'formal sealed physical-pair independent audit is pending for a formal stage; not smoke')
    require(not sealed_fields or sealed_fields == {'sealed_cohort', 'sealed_policy'},
            'both formal sealed artifact references mandatory')
    has_sealed = bool(sealed_fields)
    has_policy = 'policy_probe' in actual
    if has_policy:
        expected_fields.add('policy_probe')
    if has_sealed:
        expected_fields.update(sealed_fields)
    require(set(actual) == expected_fields, 'missing/extra fitted stage report fields')
    require(has_policy == (policy_rollout is not None), 'referenced fifth policy artifact missing/unbound')
    require(mode != 'smoke' or has_policy, 'new smoke admission requires actual new recorder probe')
    require(has_sealed == (sealed_cohort is not None and sealed_policy is not None) and
            ((sealed_cohort is None) == (sealed_policy is None)),
            'referenced sixth/seventh sealed files missing/unbound')
    require(not (mode == 'formal' and has_policy) or has_sealed,
            'admitted formal physical stage requires all seven artifacts')
    if has_policy:
        paths['policy_rollout'] = Path(policy_rollout)
        require(len({path.resolve() for path in paths.values()}) == 5, 'five stage paths must be distinct')
        hashes['policy_rollout'] = file_sha(policy_rollout)
    if has_sealed:
        require(mode == 'formal' and has_policy, 'sealed stage is formal admitted-head only')
        paths.update(sealed_cohort=Path(sealed_cohort), sealed_policy=Path(sealed_policy))
        require(len({path.resolve() for path in paths.values()}) == 7, 'seven stage paths must be distinct')
        hashes.update(sealed_cohort=file_sha(sealed_cohort), sealed_policy=file_sha(sealed_policy))
    n, seconds = BUDGETS[mode]
    start = actual['start']
    # Required for every new REAL stage. The helper validates exact schema and
    # genuine types without conversions; SKU/platform checks remain separate.
    # RTX 4090's reported driver name is not proof of RTX 4090D silicon.
    hardware = start.get('native_hardware')
    require(validate_hardware_record(hardware) is True, 'native hardware validation did not pass')
    for key, value in dict(schema='head_smooth_native_stage_v1', mode=mode,
            code_commit=expected_commit, implementation_fingerprint=fp, identity=identity,
            source_checkpoint_sha256=binding['file_sha256'], num_envs=n, duration_s=seconds,
            seeds=dict(train=305, validation=505), ppo_updates_added=0, optimizer_state_reused=False,
            effectiveness_verified=False, dr_unlocked=False).items():
        _same(start.get(key), value, 'START.'+key)
    require(type(actual['solver_runs']) is int and actual['solver_runs'] == 1 and
            actual['effectiveness_verified'] is False and actual['dr_unlocked'] is False,
            'stage added updates/retuning/effectiveness claims')
    _same(actual['cohort_sha256'], {key: hashes[key] for key in ('train', 'validation')}, 'cohort hashes')
    require(actual['head_sha256'] == hashes['head'], 'head file hash mismatch')
    exclusions, source_proof, source_manifest = registered_source_exclusions(
        repo, Path(repo)/'resources/amp_inputs/source110_exclusions_v1.npz', identity)
    require(source_proof['sha256'] == SOURCE_ENDPOINT_SHA and
            source_proof['derived_index']['sha256'] == INDEX_SHA, 'not the fixed source/index bytes')
    _source_proof(start['source_exclusions'], source_proof)
    protected_exclusions = copy.deepcopy(exclusions)
    manifests, arrays, audits, parities = {}, {}, {}, {}
    policy = source_policy(source, repo, binding['model_state_sha256'])
    for split in ('train', 'validation'):
        manifest, data = _read_bundle(paths[split], torch)
        require(all(value.dtype.kind in 'biuf' and np.isfinite(value).all() for value in data.values()),
                'nonfinite/non-numeric actual native arrays')
        source_bodies = list(experiment.spec.body_names) if experiment is not None else SOURCE_BODIES
        _same(manifest.get('body_names'), source_bodies, split+' actual source-protocol bodies')
        if experiment is not None:
            _same(manifest.get('dof_names'), list(experiment.spec.joint_names), split+' actual DOF order')
            _same(manifest.get('foot_names'), ['left_ankle_roll_link', 'right_ankle_roll_link'],
                  split+' actual feet')
            _same(manifest.get('urdf_lf_sha256'), experiment.kinematics.lf_sha256, split+' asset SHA')
        t, count = data['reference_valid'].shape
        for name, shape in (('reference_key_positions_w', (t, count, len(source_bodies), 3)),
                            ('reference_initial_key_positions_w', (count, len(source_bodies), 3))):
            require(data[name].shape == shape and data[name].dtype == np.float32,
                    'wrong source-protocol cohort key-body dimensions '+name)
        _source_proof(manifest['source_regression'], source_proof)
        prior = [] if split == 'train' else [dict(sha256=hashes['train'], split='train', seed=305,
            num_envs=n, duration_s=seconds, episode_ids=manifests['train']['episode_ids'])]
        _same(manifest.get('previous_cohorts'), prior, split+' prior split proof')
        audit = audit_cohort(manifest, data, split=split, mode=mode, identity=identity,
            code_commit=expected_commit, implementation_fingerprint=fp,
            exclusions=exclusions, source_manifest=source_manifest)
        require(audit['smoke_eligible' if mode == 'smoke' else 'fit_eligible'],
                'incomplete/failed/duplicate actual '+split+' cohort')
        _parity_record(actual['source_parity'][split], data,
                       manifest.get('inference_capture'), hardware)
        parities[split] = forward_parity(policy, data,
            action_clip=manifest['environment']['normalization']['clip_actions'],
            batch_size=count,
            expected_model_fingerprint=binding['model_state_sha256'])
        require(parities[split]['numerical_context']['device_type'] == 'cpu',
                'independent source re-forward is not the CPU audit path')
        exclusion_add_cohort(exclusions, audit)
        manifests[split], arrays[split], audits[split] = manifest, data, audit
    artifact = torch.load(io.BytesIO(paths['head'].read_bytes()), weights_only=True, map_location='cpu')
    archive = validate_head_artifact(artifact, source_checkpoint, expected_parent=expected_parent)
    _same(actual['archive_audit'], archive, 'strict head archive audit')
    frozen = validate_head_only_change(source['model_state_dict'], artifact['model_state_dict'],
                                       identity_digest(identity), identity_digest(identity))
    _same(actual['frozen_audit'], frozen, 'frozen31 tensor audit')
    provenance = artifact['provenance']
    for key, value in (('code_commit', expected_commit), ('implementation_fingerprint', fp), ('mode', mode)):
        _same(provenance[key], value, 'head provenance '+key)
    for split in ('train', 'validation'):
        _same(provenance['cohorts'][split], dict(sha256=hashes[split], seed=SEEDS[split],
            num_envs=n, duration_s=seconds, episode_ids=manifests[split]['episode_ids']), 'head cohort '+split)
    _same(provenance['source_parity'], dict(full_obs_verified=True, hidden_verified=True,
        action_verified=True, max_abs_error=max(value['max_abs_error'] for value in actual['source_parity'].values())),
        'native-recorded parity provenance')
    train_rows, val_rows = [fit_episodes(manifests[split], arrays[split], audits[split], identity=identity)
                           for split in ('train', 'validation')]
    forbidden = [label for labels in exclusions['initial_state'].values() for label in labels
                 if label.startswith('source110/') or '/sealed/' in label]
    old, new = source['model_state_dict'], artifact['model_state_dict']
    exported = evaluate_head_candidate(train_rows, val_rows,
        old['actor.6.weight'].numpy(), old['actor.6.bias'].numpy(),
        new['actor.6.weight'].numpy(), new['actor.6.bias'].numpy(),
        source_identity=identity_digest(identity), candidate_identity=identity_digest(identity),
        action_clip=source_manifest['environment']['normalization']['clip_actions'],
        forbidden_cohort_ids=forbidden)
    _same(actual['exported_head'], exported, 'exported float32 metrics', approximate=True)
    fitted = actual['float64_fit']
    require(fitted.get('audit_origin') == 'float64_closed_form_training' and
            fitted.get('candidate_head_dtype') == 'float64' and type(fitted.get('admitted')) is bool and
            fitted.get('direction_scale') == provenance['solver']['direction_scale'],
            'wrong recorded float64 solve scope/direction')
    solver_extra = {'audit_origin', 'candidate_head_dtype', 'direction_scale',
                    'unconstrained_training_output_change_peak_rad'}
    exported_extra = {'audit_origin', 'candidate_head_dtype', 'candidate_identity',
                      'candidate_adjusted', 'solver_run'}
    require(set(fitted)-solver_extra == set(exported)-exported_extra,
            'recorded float64 solver report schema changed')
    peak = fitted['unconstrained_training_output_change_peak_rad']
    require(type(peak) in (int, float) and math.isfinite(peak) and peak >= 0,
            'invalid recorded unconstrained solve peak')
    for key in set(exported)-exported_extra-{'admitted', 'rejection_reasons', 'train', 'validation',
            'train_normalized_curvature_squared', 'validation_normalized_curvature_squared'}:
        _same(fitted[key], exported[key], 'recorded fixed solve protocol '+key, approximate=True)
    reasons = _admission(fitted['train'], fitted['validation'])
    require(fitted['rejection_reasons'] == reasons and fitted['admitted'] is (not reasons),
            'recorded float64 fit admission disagrees with its fixed gates')
    for split in ('train', 'validation'):
        require(len(fitted[split]) == len(exported[split]), 'missing recorded solve episode')
        for recorded, computed in zip(fitted[split], exported[split]):
            for key in ('episode_id', 'cohort_id', 'seed', 'env_id', 'split', 'design_sha256',
                        'samples', 'first_tick', 'last_tick', 'source'):
                _same(recorded[key], computed[key], 'recorded source statistics '+key, approximate=True)
    admitted = bool(fitted['admitted'] and exported['admitted'])
    require(provenance['offline_admitted'] is admitted and
            actual['status'] == ('offline_pass' if admitted else 'offline_rejected') and
            actual['admitted_to_physical_test'] is bool(admitted and mode == 'formal'),
            'recorded stage/export admission mismatch')
    math_report = {key: value for key, value in actual.items() if key not in
                   ('head_sha256', 'archive_audit', 'frozen_audit', 'provenance_report_json',
                    'policy_probe', 'sealed_cohort', 'sealed_policy')}
    math_text = actual['provenance_report_json']
    require(type(math_text) is str, 'missing bound pre-export report bytes')
    _same(_json(math_text), math_report, 'bound pre-export report')
    require(hashlib.sha256(math_text.encode('utf-8')).hexdigest() == provenance['report_sha256'],
            'head provenance report hash mismatch')
    if has_policy:
        if mode == 'formal':
            require(admitted and actual['policy_probe']['head_artifact_sha256'] == hashes['head'],
                    'formal recorder received another/unadmitted head')
        policy_audit = audit_policy_rollout(policy_rollout, actual['policy_probe'], mode=mode,
            identity=identity, binding=binding, commit=expected_commit, fingerprint=fp,
            artifact=artifact, archive=archive, source_proof=source_proof,
            source_manifest=source_manifest, exclusions=exclusions, experiment=experiment)
    else:
        policy_audit = None
    sealed_audit, regression_metrics = None, None
    if has_sealed:
        require(admitted and actual['admitted_to_physical_test'] is True,
                'unadmitted head cannot have a sealed physical pair')
        manifest, data = _read_bundle(paths['sealed_cohort'], torch)
        manifests['sealed'], arrays['sealed'] = manifest, data
        paired_manifest, paired_arrays = _read_bundle(paths['sealed_policy'], torch)
        sealed_audit = audit_sealed_artifacts(paired_manifest, paired_arrays,
            cohorts={split: dict(manifest=manifests[split], arrays=arrays[split],
                sha256=hashes['sealed_cohort' if split == 'sealed' else split])
                for split in ('train', 'validation', 'sealed')},
            artifact=artifact, archive=archive, identity=identity, binding=binding,
            commit=expected_commit, fingerprint=fp, source_proof=source_proof,
            source_manifest=source_manifest, exclusions=protected_exclusions,
            sealed_reference=actual['sealed_cohort'], pair_reference=actual['sealed_policy'],
            pair_sha=hashes['sealed_policy'], head_sha=hashes['head'], experiment=experiment)
        audits['sealed'] = sealed_audit['cohort_inputs']['audits']['sealed']
        parities['sealed'] = forward_parity(policy, data,
            action_clip=manifest['environment']['normalization']['clip_actions'],
            batch_size=data['reference_full_obs'].shape[1],
            expected_model_fingerprint=binding['model_state_sha256'])
        require(parities['sealed']['numerical_context']['device_type'] == 'cpu',
                'independent sealed source re-forward is not the CPU audit path')
        if experiment is not None:
            from humanoid.amp.recovery import foot_collision_vertices
            from tools.amp.audit_physics_diagnostic import fast_geometry
            geometry = fast_geometry(foot_collision_vertices(experiment.kinematics.path))
            probe_manifest, probe_arrays = _read_bundle(paths['policy_rollout'], torch)
            regression_metrics = dict(rows=metrics_for_case(
                dict(manifest=probe_manifest, arrays=probe_arrays), geometry),
                known_regression_only=True, novel_holdout=False,
                source_baseline_comparison_included=False,
                limitation='Original32 candidate rows are known regression starts. '
                    'Comparison with immutable original110 trajectories is a separate audit, '
                    'not inferred from an exclusion index containing only state/history fingerprints.')
    text, diagnostics = read_log(cloud_log)
    logs = validate_log(text, actual, audits, platform['task_id'], mode, diagnostics,
                        policy=policy_audit, sealed=sealed_audit)
    for name, path in paths.items():
        require(file_sha(path) == hashes[name], 'artifact changed during audit '+name)
    require(file_sha(source_checkpoint) == binding['file_sha256'], 'immutable parent changed during audit')
    require(file_sha(platform_info) == platform_sha and file_sha(cloud_log) == diagnostics['raw_log_sha256'],
            'platform/log snapshot changed during audit')
    require(implementation_fingerprint(repo) == fp, 'implementation changed during audit')
    facts = dict(schema=SCHEMA, mode=mode, verified=True, synthetic_fixture_only=synthetic,
        native_verified=not synthetic, platform=platform, identity=identity, code_commit=expected_commit,
        implementation_fingerprint=fp, source_checkpoint_sha256=binding['file_sha256'],
        native_hardware=hardware, native_hardware_record_verified=True,
        hardware_evidence_scope='Bound actual runtime snapshot plus supplied platform ESKU000001/gpu1 metadata. '
            'Supported driver name RTX4090 or RTX4090D does not establish the device silicon is RTX4090D.',
        source_model_state_sha256=binding['model_state_sha256'], source_endpoint_sha256=SOURCE_ENDPOINT_SHA,
        source_index_sha256=INDEX_SHA, artifact_sha256=hashes,
        platform_info_sha256=platform_sha, cloud_log_sha256=diagnostics['raw_log_sha256'],
        cohort_arrays_verified=True, source_forward_verified=True, head_archive_verified=True,
        native_recorder_verified=has_policy, policy_rollout_audit=policy_audit,
        native_sealed_pair_verified=has_sealed, sealed_pair_audit=sealed_audit,
        original32_candidate_metrics=regression_metrics,
        cohorts={split: dict(episodes=audits[split]['episodes'], history=audits[split]['history'],
            deduplication=audits[split]['deduplication']) for split in audits},
        native_recorded_source_parity=actual['source_parity'], recomputed_cpu_source_parity=parities,
        native_numerical_context_verified=True,
        cpu_forward_scope='Same per-tick cohort batch, but independently recorded CPU device/library '
            'context. CPU error below .001 neither reproduces nor certifies GPU numerical execution.',
        exported_float32_report=exported, recorded_float64_solver_admitted=fitted['admitted'],
        offline_admitted=admitted, formal_admission=False, solver_runs=1, auditor_solver_runs=0,
        logs=logs, logs_clean=logs['logs_clean'], effectiveness_verified=False, dr_unlocked=False,
        limitation='CPU source forward must independently meet .001, not equal GPU error numerically. '
            'Stored float32 head values are promoted to float64 on fixed source designs; '
            'not native float32 GEMM per-point bounds or closed-loop history feedback proof. '
            'Metrics are recomputed without solve or candidate changes. '
            'Recorded float64 solver candidate is not independently reconstructed. Native evidence '
            'binds supplied full platform/log/artifact snapshots, not their cryptographic authenticity. '
            'An offline/smoke pass is not physical quality, formal effectiveness or DR admission.')
    digest = canonical_sha(facts)
    result = dict(facts=facts, audit_facts_sha256=digest,
                  audit_hash_protocol='SHA256 canonical UTF8 sorted-key compact facts JSON; certificate excluded')
    if mode == 'smoke' and not synthetic:
        result['smoke_certificate'] = dict(schema=SMOKE_SCHEMA, mode='smoke', identity=identity,
            implementation_fingerprint=fp, native_verified=True, cohort_arrays_verified=True,
            source_forward_verified=True, head_archive_verified=True, native_recorder_verified=True, train_seed=305,
            validation_seed=505, num_envs=4, duration_s=2, solver_runs=1, formal_admission=False,
            source_checkpoint_sha256=binding['file_sha256'], source_model_state_sha256=binding['model_state_sha256'],
            platform_task_id=platform['task_id'], platform_terminal_status='5', code_commit=expected_commit,
            cloud_log_sha256=diagnostics['raw_log_sha256'], train_sha256=hashes['train'],
            validation_sha256=hashes['validation'], head_sha256=hashes['head'], audit_report_sha256=digest,
            report_artifact_sha256=hashes['report'], platform_info_sha256=platform_sha,
            policy_rollout_sha256=hashes['policy_rollout'],
            logs_clean=logs['logs_clean'], effectiveness_verified=False, dr_unlocked=False)
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', choices=tuple(BUDGETS), required=True)
    for name in ('source-checkpoint', 'train', 'validation', 'head', 'report',
                 'cloud-log', 'platform-info', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--policy-rollout', type=Path)
    parser.add_argument('--sealed-cohort', type=Path)
    parser.add_argument('--sealed-policy', type=Path)
    args = vars(parser.parse_args())
    output = args.pop('output')
    require(not output.exists(), 'audit output already exists')
    torch.set_num_threads(2)
    result = audit_head_stage(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output.resolve()), audit_sha256=file_sha(output),
        audit_facts_sha256=result['audit_facts_sha256'], mode=result['facts']['mode'],
        native_verified=result['facts']['native_verified'], offline_admitted=result['facts']['offline_admitted'],
        logs_clean=result['facts']['logs_clean'], formal_admission=False,
        effectiveness_verified=False, dr_unlocked=False)))


if __name__ == '__main__':
    main()
