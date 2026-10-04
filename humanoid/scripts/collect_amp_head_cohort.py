"""New, source-only reference cohorts for an offline final-head experiment.

Native collection is a separate Linux/CUDA Isaac Gym process, not PPO training.
Pure NumPy protocol helpers remain importable for CPU tests.  The native entry
point imports Isaac Gym BEFORE torch (including all transitive torch imports).
No legacy recorder, reset, reward, PD, physics, or inference implementation is
changed.  Failed first-episode prefixes are archived, never silently selected
away.  Different seeds alone are NOT a proof of independent observations.
"""
import argparse
import codecs
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


SCHEMA = 'head_cohort_v2'
SOURCE_TASK = 'TASK_20260926_110'
SOURCE_SHA = '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31'
SOURCE_MODEL_SHA = 'c06fbd97e32a483d46a2738014215de2721ad7db4b2b86547645ab72d4c555bc'
SOURCE_ENDPOINT_SHA = 'de77d4a0c85cd581ae9dc9418d3fbce144730749b9f746f93b9671ec122a833b'
SOURCE_CONFIG = 'configs/amp/lafan_walk02_sustain_control.json'
SEEDS = dict(train=306, validation=506, sealed=706)
BUDGETS = dict(smoke=(4, 2.), formal=(8, 20.))
HISTORY = 66
OBS = 47
HIDDEN = 128
ACTIONS = 12
INITIAL_FIELDS = ('root_state', 'dof_pos', 'dof_vel', 'action', 'torque',
                  'base_lin_vel', 'base_ang_vel', 'foot_state', 'foot_force',
                  'key_positions_w', 'valid', 'failure')
PHYSICS_FIELDS = ('physics_valid', 'physics_accel_squared', 'physics_accel_peak',
                  'physics_torque_delta_squared', 'physics_foot_force_peak')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_split(split):
    split = 'validation' if split == 'val' else split
    if split not in SEEDS:
        raise ValueError('Unknown head cohort split')
    return split


def cohort_contract(split, mode, seed, num_envs, duration_s):
    split = canonical_split(split)
    if (mode not in BUDGETS or type(seed) is not int or seed != SEEDS[split]
            or type(num_envs) is not int or isinstance(duration_s, bool)
            or (num_envs, duration_s) != BUDGETS[mode]):
        raise ValueError('Head cohort permits only registered reference seeds/budgets')
    return dict(type=SCHEMA, split=split, mode=mode, seed=seed,
                num_envs=num_envs, duration_s=float(duration_s), fps=100,
                expected_ticks=int(duration_s * 100), initialization_mode='reference_only',
                episode_ids=[episode_id(split, seed, i) for i in range(num_envs)])


def validate_initialization_proof(initialization, contract):
    """Keep the reference role separate from the actual native reset evidence.

    The native manifest's ``initialization`` is a proof dictionary, not the
    role string. Checking the dictionary does not recreate a simulator reset
    or independently establish that these recorded values came from PhysX.
    """
    expected_keys = {'protocol', 'reset_inputs', 'cleared_histories',
                     'all12_initial_fields', 'actual_initial_full_obs'}
    if (type(initialization) is not dict or set(initialization) != expected_keys
            or initialization['protocol'] !=
                'original_reference_reset_then_native_10ms_zero_action_warmup'
            or initialization['all12_initial_fields'] != list(INITIAL_FIELDS)
            or initialization['actual_initial_full_obs'] is not True):
        raise ValueError('Missing/changed actual native reference initialization proof')
    n = contract['num_envs']
    shapes = dict(root_states=(n, 13), dof_pos=(n, 12), dof_vel=(n, 12),
        actions=(n, 12), last_actions=(n, 12), last_last_actions=(n, 12),
        commands=(n, 4), gait_start=(n,), episode_length_buf=(n,),
        phase_length_buf=(n,), rsi_indices=(n,))
    reset = initialization['reset_inputs']
    hashes = {'obs_history_sha256', 'critic_history_sha256'}
    if type(reset) is not dict or set(reset) != set(shapes) | hashes:
        raise ValueError('Missing/extra actual native reference reset inputs')
    for key, shape in shapes.items():
        value = np.asarray(reset[key])
        if (value.shape != shape or value.dtype.kind not in 'biuf'
                or not np.isfinite(value).all()):
            raise ValueError('Invalid actual native reference reset input: '+key)
    for key in hashes:
        if (not isinstance(reset[key], str)
                or re.fullmatch('[0-9a-f]{64}', reset[key]) is None):
            raise ValueError('Missing actual cleared reference history hash')
    cleared = initialization['cleared_histories']
    if type(cleared) is not dict or set(cleared) != {'obs_history', 'critic_history'}:
        raise ValueError('Missing actual cleared reference histories')
    for row in cleared.values():
        if (type(row) is not dict
                or set(row) != {'elements', 'nonzero', 'nonfinite', 'negative_zero'}
                or any(type(value) is not int for value in row.values())
                or row['elements'] <= 0 or row['nonzero'] != 0 or row['nonfinite'] != 0
                or not 0 <= row['negative_zero'] <= row['elements']):
            raise ValueError('Actual reference input histories were not cleared')
    return True


def episode_id(split, seed, env_index):
    split = canonical_split(split)
    if (type(seed) is not int or seed != SEEDS[split] or type(env_index) is not int
            or env_index < 0):
        raise ValueError('Invalid canonical first-episode identity')
    return '%s/%s/seed-%d/env-%d/episode-0' % (SCHEMA, split, seed, env_index)


def numerical_fingerprint(fields):
    """Exact numerical bytes, except +0 and -0; never mutate or quantize inputs."""
    digest = hashlib.sha256()
    for key in sorted(fields):
        value = np.asarray(fields[key])
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('Fingerprint requires finite numerical fields: ' + key)
        value = np.ascontiguousarray(value).copy()
        if value.dtype.kind == 'f':
            value[value == 0] = 0
        digest.update(key.encode() + b'\0' + str(value.dtype).encode() + b'\0')
        digest.update(str(tuple(value.shape)).encode() + b'\0' + value.tobytes())
    return digest.hexdigest()


def full_history_fingerprint(value):
    value = np.asarray(value)
    if value.shape != (HISTORY * OBS,) or value.dtype != np.float32:
        raise ValueError('Expected actual float32 66x47 actor history')
    return numerical_fingerprint({'full_observation': value})


def empty_exclusion_index():
    return dict(initial_state={}, initial_full_observation={}, ready_full_history={})


def _index_add(index, kind, digest, label):
    # Preserve all labels: source itself may legitimately contain equal rows.
    labels = index[kind].setdefault(digest, [])
    if label not in labels:
        labels.append(label)


def validate_history(full, single, valid, native_episode, initial_native_episode):
    """Prove actual ready inputs equal their own consecutive, same-episode rows.

    Tick zero uses reset()'s actual 10-ms warmup input. Conservatively discard
    the first 66 decision ticks from fitting, so a ready stencil does not depend
    on zero padding or the reset warmup. No simulated/reconstructed input is fed
    to the native policy.
    """
    full, single, valid = np.asarray(full), np.asarray(single), np.asarray(valid)
    t, n = valid.shape
    if (full.shape != (t, n, HISTORY * OBS) or single.shape != (t, n, OBS)
            or full.dtype != np.float32 or single.dtype != np.float32
            or valid.dtype != np.bool_ or native_episode.shape != (t, n)
            or initial_native_episode.shape != (n,)):
        raise ValueError('Invalid native observation/history dimensions')
    if not np.isfinite(full).all() or not np.isfinite(single).all():
        raise ValueError('Nonfinite actual actor inputs')
    if not np.array_equal(full[:, :, -OBS:], single):
        raise ValueError('Single observation is not the actual full-input last row')
    ready = valid & (np.arange(t)[:, None] >= HISTORY)
    if np.any(native_episode[valid] != np.broadcast_to(initial_native_episode, (t, n))[valid]):
        raise ValueError('A valid prefix crossed the first native episode')
    count = 0
    for tick in range(HISTORY, t):
        # Stack is oldest-to-newest, exactly as compute_observations builds it.
        expected = single[tick-HISTORY+1:tick+1].transpose(1, 0, 2).reshape(n, -1)
        mask = ready[tick]
        if not np.array_equal(full[tick, mask], expected[mask]):
            raise ValueError('Actual full history is not its consecutive episode observations')
        count += int(mask.sum())
    real_frames = np.minimum(np.arange(t) + 1, HISTORY)[:, None]
    real_frames = np.broadcast_to(real_frames, (t, n)).copy().astype(np.int64)
    return ready, real_frames, dict(exact=True, checked_ready_inputs=count,
        warmup_excluded_from_fit=True, minimum_ready_decision_tick=HISTORY,
        no_cross_episode=True, padding_is_not_independence_evidence=True)


def audit_arrays(arrays, contract):
    """Validate all archived rows; retain failed prefixes and return fit status."""
    n, expected = contract['num_envs'], contract['expected_ticks']
    valid = arrays['reference_valid']
    if valid.ndim != 2 or valid.dtype != np.bool_ or valid.shape[1] != n:
        raise ValueError('Malformed first-episode validity mask')
    t = len(valid)
    if not 1 <= t <= expected:
        raise ValueError('Malformed observed tick budget')
    required = INITIAL_FIELDS + PHYSICS_FIELDS + ('done', 'native_episode',)
    for key in required:
        value = arrays['reference_' + key]
        if value.shape[:2] != (t, n) or not np.isfinite(value).all():
            raise ValueError('Malformed/nonfinite captured field ' + key)
    for key in INITIAL_FIELDS:
        value = arrays['reference_initial_' + key]
        if value.shape[0] != n or not np.isfinite(value).all():
            raise ValueError('Malformed/nonfinite initial field ' + key)
    for key in ('valid', 'failure', 'done', 'physics_valid'):
        if arrays['reference_'+key].dtype != np.bool_:
            raise ValueError('Native mask is not boolean: '+key)
    for key in ('native_episode', 'episode_id', 'tick'):
        if arrays['reference_'+key].dtype != np.int64:
            raise ValueError('Native episode/tick is not int64: '+key)
    for key, width in (('full_obs', HISTORY * OBS), ('single_obs', OBS),
                       ('actor_hidden', HIDDEN), ('raw_mu', ACTIONS), ('command', 4)):
        value = arrays['reference_' + key]
        if (value.shape != (t, n, width) or value.dtype != np.float32
                or not np.isfinite(value).all()):
            raise ValueError('Malformed/nonfinite decision field ' + key)
    initial_obs = arrays['reference_initial_full_obs']
    if initial_obs.shape != (n, HISTORY * OBS) or initial_obs.dtype != np.float32:
        raise ValueError('Missing actual first full observation')
    if not np.array_equal(arrays['reference_full_obs'][0], initial_obs):
        raise ValueError('First policy input differs from actual reset-warmup input')
    command = arrays['reference_command'][:, :, :3]
    wanted = np.array([.45, 0., 0.], dtype=np.float32)
    if not np.array_equal(command, np.broadcast_to(wanted, command.shape)):
        raise ValueError('Actual cohort commands changed from original source')
    ticks = np.broadcast_to(np.arange(t, dtype=np.int64)[:, None], (t, n))
    if (not np.array_equal(arrays['reference_tick'], ticks)
            or not np.array_equal(arrays['reference_episode_id'], np.zeros((t, n), dtype=np.int64))
            or not np.array_equal(arrays['reference_time'], (np.arange(t)+1)*.01)):
        raise ValueError('Missing/nonconsecutive first-episode control ticks')
    ready, real_frames, proof = validate_history(arrays['reference_full_obs'],
        arrays['reference_single_obs'], valid, arrays['reference_native_episode'],
        arrays['reference_initial_native_episode'])
    for key, expected_value in (('history_ready', ready), ('history_real_frames', real_frames)):
        if 'reference_'+key in arrays and not np.array_equal(arrays['reference_'+key], expected_value):
            raise ValueError('Archived history readiness metadata differs from actual inputs')
    episodes = []
    for i, label in enumerate(contract['episode_ids']):
        count = int(valid[:, i].sum())
        if not np.array_equal(valid[:, i], np.arange(t) < count):
            raise ValueError('Restarted/missing states in first-episode prefix')
        failures = arrays['reference_failure'][:count, i]
        dones = arrays['reference_done'][:count, i]
        if (failures.any() and (not failures[-1] or failures[:-1].any())) or dones[:-1].any():
            raise ValueError('Terminal first-episode state was lost or crossed')
        failed = bool(arrays['reference_initial_failure'][i] or failures.any())
        complete = bool(count == expected and not failed and not dones.any()
                        and arrays['reference_initial_valid'][i])
        episodes.append(dict(episode_id=label, env=i, observed_ticks=count,
            failed=failed, terminal_tick=None if not dones.any() else count-1,
            complete=complete, ready_inputs=int(ready[:, i].sum()),
            fit_eligible=bool(complete and ready[:, i].any()
                and contract['mode'] == 'formal' and contract['split'] != 'sealed')))
    return ready, real_frames, proof, episodes


def fingerprint_cohort(arrays, contract, ready, exclusions):
    """Compare actual numeric inputs, not seeds or cleared zero histories."""
    result, local = [], empty_exclusion_index()
    overlaps = []
    for i, label in enumerate(contract['episode_ids']):
        state = numerical_fingerprint({key: arrays['reference_initial_'+key][i]
                                       for key in INITIAL_FIELDS})
        first = full_history_fingerprint(arrays['reference_initial_full_obs'][i])
        ready_rows = []
        for tick in np.flatnonzero(ready[:, i]):
            ready_rows.append(dict(tick=int(tick), sha256=full_history_fingerprint(
                                  arrays['reference_full_obs'][tick, i])))
        values = dict(initial_state=[state], initial_full_observation=[first],
                      ready_full_history=[r['sha256'] for r in ready_rows])
        for kind, hashes in values.items():
            for digest in hashes:
                # Repeated inputs inside the SAME episode are valid samples,
                # not evidence of a cross-episode/split duplicate.
                found = exclusions[kind].get(digest, []) + [previous for previous in
                    local[kind].get(digest, []) if previous != label]
                if found:
                    overlaps.append(dict(episode_id=label, kind=kind,
                                         sha256=digest, matched_episode_ids=found))
                _index_add(local, kind, digest, label)
        result.append(dict(episode_id=label, initial_state=state,
            initial_full_observation=first, ready_full_history=ready_rows))
    return result, dict(passed=not overlaps, overlaps=overlaps,
        compared_counts={kind: len(values) for kind, values in exclusions.items()},
        numerical_equivalence='exact shape/dtype/values, signed zeros normalized on copies only',
        limitation='No exact repeated state/history found is not statistical independence, '
                   'semantic novelty, broader task coverage, or effectiveness proof.')


def fit_admission(contract, episodes, history_proof, dedupe_proof):
    # Nothing is dropped: any failed/incomplete environment rejects the cohort.
    return bool(contract['mode'] == 'formal' and contract['split'] != 'sealed'
        and len(episodes) == contract['num_envs']
        and all(row['fit_eligible'] for row in episodes)
        and history_proof.get('exact') is True and dedupe_proof.get('passed') is True)


def validate_source_runtime(environment, runtime, source_manifest, contract):
    """Only seed/count/observation horizon may differ from the immutable source."""
    expected, actual = (copy.deepcopy(source_manifest['environment']),
                        copy.deepcopy(environment))
    if (actual.get('seed') != contract['seed']
            or actual.get('env', {}).get('num_envs') != contract['num_envs']
            or actual.get('env', {}).get('episode_length_s') != contract['duration_s']+.1):
        raise ValueError('Actual cohort seed/count/horizon do not match the registered budget')
    for cfg in (expected, actual):
        cfg.pop('seed')
        cfg['env'].pop('num_envs')
        cfg['env'].pop('episode_length_s')
    if actual != expected:
        raise ValueError('Source configuration changed beyond seed/count/observation horizon')
    for key in ('dof_properties', 'p_gains', 'd_gains', 'control_dt', 'physics_dt'):
        if runtime.get(key) != source_manifest['runtime'].get(key):
            raise ValueError('Actual native source DOF/PD/timing readback changed: '+key)
    sim = runtime['physics_sim_parameters']
    cfg_sim = source_manifest['environment']['sim']
    # SimParams/PhysX expose C++ float32 fields through Python floats. Compare
    # native dt to the original native readback, not the pre-ABI config literal.
    # For fields missing from the old endpoint, require the exact configured
    # float32 representation. This is not a tolerance or a physics change.
    if (type(sim.get('dt')) is not float
            or sim['dt'] != source_manifest['runtime']['physics_dt']):
        raise ValueError('Actual native physics dt differs from original source')
    floating = {'contact_offset', 'rest_offset', 'max_depenetration_velocity'}
    for key in ('solver_type', 'num_position_iterations', 'num_velocity_iterations',
                'contact_offset', 'rest_offset', 'max_depenetration_velocity', 'contact_collection'):
        value = sim.get('physx', {}).get(key)
        expected_value = (float(np.float32(cfg_sim['physx'][key])) if key in floating
                          else cfg_sim['physx'][key])
        expected_type = float if key in floating else int
        if type(value) is not expected_type or value != expected_value:
            raise ValueError('Actual native source PhysX readback changed/missing: '+key)
    return dict(configuration_exact_except=['seed', 'env.num_envs', 'env.episode_length_s'],
        dof_pd_timing_exact=True, native_physx_parameters_exact=True,
        native_float_abi='exact float32 configured PhysX fields; dt equals original native readback',
        source_physical_shape_readback='not present in old endpoint; current shapes preserved, not fabricated')


def validate_source_forward_context(value, *, native=False):
    """Typed numerical readback, not a policy/simulation or platform certificate."""
    if type(native) is not bool:
        raise ValueError('Invalid source forward context validation mode')
    types = dict(schema=str, device_type=str, device_index=(int, type(None)),
        input_dtype=str, grad_enabled=bool, policy_training=bool,
        cudnn_enabled=bool, cudnn_deterministic=bool, cudnn_benchmark=bool,
        cudnn_allow_tf32=bool, cuda_matmul_allow_tf32=bool,
        float32_matmul_precision=str, torch_version=str,
        torch_cuda_version=(str, type(None)), cudnn_version=(int, type(None)))
    if type(value) is not dict or set(value) != set(types):
        raise ValueError('Missing/malformed source forward numerical context')
    for key, expected in types.items():
        allowed = expected if type(expected) is tuple else (expected,)
        if type(value[key]) not in allowed:
            raise ValueError('Wrong source forward numerical context type: '+key)
    if (value['schema'] != 'source_forward_numerical_context_v1'
            or value['device_type'] not in ('cpu', 'cuda')
            or value['input_dtype'] != 'torch.float32'
            or value['grad_enabled'] is not False or value['policy_training'] is not False
            or value['float32_matmul_precision'] not in ('highest', 'high', 'medium')
            or not value['torch_version']
            or (value['torch_cuda_version'] is not None and not value['torch_cuda_version'])
            or (value['cudnn_version'] is not None and value['cudnn_version'] <= 0)
            or (value['device_type'] == 'cpu' and value['device_index'] is not None)
            or (value['device_type'] == 'cuda' and
                (type(value['device_index']) is not int or value['device_index'] < 0))):
        raise ValueError('Invalid source forward numerical context values')
    if native and (value['device_type'] != 'cuda' or value['device_index'] != 0
            or value['cudnn_enabled'] is not True or value['cudnn_deterministic'] is not True
            or value['cudnn_benchmark'] is not False
            or value['torch_cuda_version'] is None or value['cudnn_version'] is None):
        raise ValueError('Native source forward context differs from source inference protocol')
    return True


def source_forward_context(policy, observation):
    """Read actual backend settings inside the ordinary no-grad forward scope."""
    import torch
    device = next(policy.parameters()).device
    if observation.device != device:
        raise ValueError('Source parity input and policy devices differ')
    value = dict(schema='source_forward_numerical_context_v1', device_type=device.type,
        device_index=device.index, input_dtype=str(observation.dtype),
        grad_enabled=torch.is_grad_enabled(), policy_training=policy.training,
        cudnn_enabled=torch.backends.cudnn.enabled,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        torch_version=str(torch.__version__), torch_cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version())
    validate_source_forward_context(value)
    return value


class FrozenActorCapture:
    """Read actor.6's ACTUAL input/output during the one source inference call."""
    def __init__(self, policy):
        import torch
        self.torch, self.policy, self.hidden, self.output = torch, policy, None, None
        self.calls, self.active = 0, False
        modules = dict(policy.named_modules())
        final = modules.get('actor.6')
        if (type(final) is not torch.nn.Linear or final.in_features != HIDDEN
                or final.out_features != ACTIONS or final.bias is None
                or list(policy.actor.children())[-1] is not final):
            raise ValueError('Expected original native 128-to-12 actor.6 final affine head')
        self.handles = [final.register_forward_pre_hook(self._before),
                        final.register_forward_hook(self._after)]

    def _before(self, module, arguments):
        if not self.active or self.hidden is not None or len(arguments) != 1:
            raise ValueError('Expected exactly one original final-head forward per decision')
        value = arguments[0]
        if value.ndim != 2 or value.shape[1] != HIDDEN or not self.torch.isfinite(value).all():
            raise ValueError('Malformed/nonfinite original final hidden vector')
        self.hidden = value.detach().clone()

    def _after(self, module, arguments, output):
        self.output = output.detach().clone()
        self.calls += 1

    def inference(self, observation):
        self.hidden, self.output, self.active = None, None, True
        try:
            result = self.policy.act_inference(observation)
            if (self.hidden is None or self.output is None
                    or not self.torch.equal(result, self.output)
                    or not self.torch.isfinite(result).all()):
                raise ValueError('Recorded affine output is not the original raw action mean')
            return result, self.hidden
        finally:
            self.active = False

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _read_bundle(path, torch):
    torch.serialization.add_safe_globals([codecs.encode])
    packed = torch.load(str(path), map_location='cpu', weights_only=True)
    if set(packed) != {'npz_bytes', 'manifest_json'}:
        raise ValueError('Not a source/cohort archive')
    manifest = json.loads(packed['manifest_json'])
    with np.load(io.BytesIO(packed['npz_bytes']), allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    return manifest, arrays


def source_exclusions(path, identity, torch):
    if file_sha(path) != SOURCE_ENDPOINT_SHA:
        raise ValueError('Source regression endpoint is not the immutable original110 artifact')
    manifest, arrays = _read_bundle(path, torch)
    if (manifest.get('checkpoint_sha256') != SOURCE_SHA or manifest.get('identity') != identity
            or manifest.get('modes') != ['standing', 'reference']
            or (manifest.get('num_envs'), manifest.get('duration_s'), manifest.get('fps')) != (16, 60., 100)
            or manifest.get('evaluation_protocol') != 'fixed60_independent_mode_seeds'
            or manifest.get('mode_seeds') != {'standing': 5, 'reference': 105}
            or manifest.get('policy_deterministic') is not True
            or manifest.get('dr_unlocked') is not False):
        raise ValueError('Wrong source regression endpoint protocol')
    from humanoid.amp.learnability import assert_no_domain_randomization
    from tools.amp.audit_recorded_estimator import single_observations
    from tools.amp.inspect_rollout import episode
    assert_no_domain_randomization(manifest['environment'])
    index = empty_exclusion_index()
    for mode in manifest['modes']:
        for i in range(16):
            label = 'source110/%s/env-%d/episode-0' % (mode, i)
            data = episode(arrays, mode, i)
            if len(data['time']) != 6000 or data['failure'].any() or data['initial']['failure']:
                raise ValueError('Original32 regression endpoint must retain all complete episodes')
            initial = {key: arrays[mode+'_initial_'+key][i] for key in INITIAL_FIELDS}
            _index_add(index, 'initial_state', numerical_fingerprint(initial), label)
            initial_data = {key: np.asarray(value)[None] for key, value in data['initial'].items()}
            initial_data['time'] = np.array([0.])
            first_single = single_observations(initial_data, manifest).numpy()[0].astype(np.float32)
            first_full = np.concatenate((np.zeros((HISTORY-1)*OBS, dtype=np.float32), first_single))
            _index_add(index, 'initial_full_observation', full_history_fingerprint(first_full), label)
            single = single_observations(data, manifest).numpy().astype(np.float32)
            for tick in range(HISTORY-1, len(single)):
                full = single[tick-HISTORY+1:tick+1].reshape(-1)
                _index_add(index, 'ready_full_history', full_history_fingerprint(full), label)
    proof = dict(sha256=SOURCE_ENDPOINT_SHA, source_task=SOURCE_TASK,
        regression_episode_ids=['source110/%s/env-%d/episode-0' % (mode, i)
            for mode in ('standing', 'reference') for i in range(16)],
        initial_state='all12 legacy numerical initial fields from the actual immutable archive',
        initial_full_observation='reconstructed source-only zero65 + actual warmup single observation',
        ready_full_history='reconstructed original66 consecutive no-noise/no-lag single observations',
        reconstruction_not_raw_history=True,
        limitation='Source full-history fingerprints use the existing CPU reconstruction, '
                   'not a claim of bitwise GPU observation replay or statistical independence.')
    return index, proof, manifest


def add_prior_exclusions(path, contract, identity, index, torch):
    manifest, arrays = _read_bundle(path, torch)
    previous = cohort_contract(manifest.get('split'), manifest.get('mode'),
        manifest.get('seed'), manifest.get('num_envs'), manifest.get('duration_s'))
    if (manifest.get('type') != SCHEMA or manifest.get('source_checkpoint_sha256') != SOURCE_SHA
            or manifest.get('source_model_state_sha256') != SOURCE_MODEL_SHA
            or manifest.get('identity') != identity or previous['mode'] != contract['mode']
            or previous['split'] == contract['split']
            or manifest.get('episode_ids') != previous['episode_ids']):
        raise ValueError('Wrong/reused previous-split cohort identity')
    ready, _, _, _ = audit_arrays(arrays, previous)
    computed, _ = fingerprint_cohort(arrays, previous, ready, empty_exclusion_index())
    if computed != manifest.get('fingerprints'):
        raise ValueError('Prior cohort numeric fingerprints do not match its actual rows')
    for row in computed:
        for kind in ('initial_state', 'initial_full_observation'):
            _index_add(index, kind, row[kind], row['episode_id'])
        for value in row['ready_full_history']:
            _index_add(index, 'ready_full_history', value['sha256'], row['episode_id'])
    return dict(sha256=file_sha(path), split=previous['split'], seed=previous['seed'],
                num_envs=previous['num_envs'], duration_s=previous['duration_s'],
                episode_ids=previous['episode_ids'])


def registered_exclusion_path(repo, requested):
    """A published fixed config owns the hash; argv cannot substitute a hash."""
    repo = Path(repo).resolve()
    config = json.loads((repo/'configs/amp/head_smooth_v2.json').read_text(encoding='utf-8'))
    spec = config.get('source_exclusions', {})
    if (set(spec) != {'path', 'sha256'}
            or spec['path'] != 'resources/amp_inputs/source110_exclusions_v1.npz'
            or not isinstance(spec['sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', spec['sha256'])):
        raise ValueError('Unregistered derived source110 exclusion input')
    path = (repo/spec['path']).resolve()
    if Path(requested).resolve() != path or not path.is_file() or file_sha(path) != spec['sha256']:
        raise ValueError('Derived source110 exclusion path/bytes differ from published fixed config')
    return path, spec['sha256']


def registered_source_exclusions(repo, requested, identity):
    path, expected_sha = registered_exclusion_path(repo, requested)
    from humanoid.amp.head_exclusions import load_exclusion_index
    index, proof = load_exclusion_index(path, expected_file_sha=expected_sha, identity=identity)
    if (proof.get('source_regression', {}).get('sha256') != SOURCE_ENDPOINT_SHA
            or not isinstance(proof.get('source_environment'), dict)
            or not isinstance(proof.get('source_runtime'), dict)
            or not isinstance(proof.get('derived_index'), dict)
            or proof['derived_index'].get('sha256') != expected_sha):
        raise ValueError('Derived exclusions lack immutable source/runtime provenance')
    source_manifest = dict(environment=proof['source_environment'], runtime=proof['source_runtime'])
    # Both raw/index modes expose the ORIGINAL endpoint at the same top-level
    # source_regression.sha256 path. Derived-file SHA is separate provenance,
    # never a substitute parent endpoint. The old readback is returned only for
    # direct native comparisons, not duplicated into manifest-facing proof.
    source_proof = dict(proof['source_regression'])
    source_proof['derived_index'] = dict(proof['derived_index'])
    return index, source_proof, source_manifest


def _native_main(extra, remaining):
    # This order is part of the native ABI contract, not just a style convention.
    from isaacgym import gymapi, gymtorch  # noqa: F401 -- must precede torch.
    import torch
    from humanoid import LEGGED_GYM_ROOT_DIR
    from humanoid.algo.ppo.actor_critic_dh import ActorCriticDH
    from humanoid.amp.learnability import assert_no_domain_randomization
    from humanoid.amp.mu_temporal import state_fingerprint
    from humanoid.amp.physics_diagnostic import (apply_diagnostic_config,
        cleared_history_diagnostics, reset_input_snapshot, physical_readback,
        validate_source_state)
    from humanoid.amp.recovery import FOOT_NAMES
    from humanoid.amp.refinement import select_environment
    from humanoid.amp.scaled_experiment import (ScaledExperiment,
        validate_runtime_timing, implementation_fingerprint)
    from humanoid.envs.x1.x1_amp_physics_diagnostic_env import X1AMPPhysicsDiagnosticEnv
    from humanoid.utils import get_args, task_registry
    from humanoid.utils.helpers import class_to_dict, set_seed

    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    if (sys.platform != 'linux' or not torch.cuda.is_available()
            or args.sim_device_type != 'cuda' or args.rl_device != 'cuda:0'
            or args.physics_engine != gymapi.SIM_PHYSX or not args.use_gpu_pipeline
            or not args.headless or args.armature_mode != 'nominal'
            or args.training_profile or args.resume or args.horovod):
        raise ValueError('New cohort collection is native Linux/CUDA headless source inference only')
    contract = cohort_contract(extra.split, extra.mode, args.seed, args.num_envs, extra.duration)
    prior_paths = list(extra.exclude_cohort)
    required_prior = {'train'} if contract['split'] == 'validation' else (
        {'train', 'validation'} if contract['split'] == 'sealed' else set())
    repo = Path(LEGGED_GYM_ROOT_DIR)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    if (not re.fullmatch('[0-9a-f]{40}', extra.expected_commit) or commit != extra.expected_commit
            or file_sha(extra.checkpoint_file) != SOURCE_SHA
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=str(repo)).strip()
            or subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--',
                                         'humanoid', 'configs', 'resources'], cwd=str(repo)).strip()):
        raise ValueError('Cohort requires the exact published checkout and original110 source checkpoint')
    subprocess.check_call(['git', 'ls-files', '--error-unmatch',
        'humanoid/scripts/collect_amp_head_cohort.py'], cwd=str(repo), stdout=subprocess.DEVNULL)
    if extra.output.exists() or extra.output.with_suffix('.json').exists():
        raise FileExistsError(extra.output)
    experiment = ScaledExperiment(repo, repo/SOURCE_CONFIG)
    state = torch.load(str(extra.checkpoint_file), map_location='cpu', weights_only=True)
    validate_source_state(state, experiment.identity())
    if args.task != experiment.cfg['experiment'] or state_fingerprint(state['model_state_dict']) != SOURCE_MODEL_SHA:
        raise ValueError('Wrong original110 task/model state')
    if extra.source_endpoint is not None:
        exclusions, source_proof, source_manifest = source_exclusions(
            extra.source_endpoint, experiment.identity(), torch)
    else:
        exclusions, source_proof, source_manifest = registered_source_exclusions(
            repo, extra.source_exclusion_index, experiment.identity())
    prior_proofs = [add_prior_exclusions(path, contract, experiment.identity(), exclusions, torch)
                    for path in prior_paths]
    if (len({row['split'] for row in prior_proofs}) != len(prior_proofs)
            or {row['split'] for row in prior_proofs} != required_prior):
        raise ValueError('Validation/sealed cohort must compare against every earlier actual split')
    cfg, train_cfg, _ = select_environment('sustain_control')
    cfg = apply_diagnostic_config(cfg, 'original')  # validates; makes NO changes.
    cfg.seed = args.seed
    cfg.env.episode_length_s = extra.duration + .1
    gate = assert_no_domain_randomization(cfg)
    task_registry.register(args.task, X1AMPPhysicsDiagnosticEnv, cfg, train_cfg)
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        validate_runtime_timing(env.dt, env.sim_params.dt, cfg.control.decimation)
        if (tuple(env.dof_names) != experiment.spec.joint_names
                or env.num_envs != contract['num_envs'] or env.num_actions != ACTIONS
                or env.num_single_obs != OBS or env.num_short_obs != 5*OBS
                or cfg.env.frame_stack != HISTORY or env.add_noise):
            raise ValueError('Actual source actor/robot/timing/no-noise contract changed')
        env.configure_reference_initialization(experiment)
        env.rsi_enabled = True
        policy = ActorCriticDH(env.num_short_obs, env.num_single_obs, env.num_privileged_obs,
                              env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
        policy.load_state_dict(state['model_state_dict'], strict=True)
        policy.eval()
        if state_fingerprint(policy.state_dict()) != SOURCE_MODEL_SHA:
            raise ValueError('Actual inference policy is not the original110 state')
        del state
        hook = FrozenActorCapture(policy)
        body_ids = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
                    for name in experiment.spec.body_names]
        feet = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
                for name in FOOT_NAMES]
        if min(body_ids) < 0 or env.feet_indices.cpu().tolist() != feet:
            raise ValueError('Actual source body/foot order changed')
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        frames, pending, recording = [], {}, False
        reset_proof, reset_inputs = {}, {}
        original_reset = env.reset_idx
        reset_pending = True
        def diagnostic_reset(ids):
            nonlocal reset_pending
            original_reset(ids)
            if reset_pending:
                if len(ids) != env.num_envs:
                    raise ValueError('Reference initialization did not reset every cohort environment')
                reset_proof.update(cleared_history_diagnostics(env))
                reset_inputs.update(reset_input_snapshot(env))
                reset_pending = False
        env.reset_idx = diagnostic_reset
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        def capture():
            root = cpu(env.root_states)
            root[:, :3] -= cpu(env.env_origins)
            foot = cpu(env.rigid_state[:, env.feet_indices])
            foot[:, :, :3] -= cpu(env.env_origins)[:, None, :]
            result = dict(root_state=root, dof_pos=cpu(env.dof_pos), dof_vel=cpu(env.dof_vel),
                action=cpu(env.actions), torque=cpu(env.torques), base_lin_vel=cpu(env.base_lin_vel),
                base_ang_vel=cpu(env.base_ang_vel), foot_state=foot,
                foot_force=cpu(env.contact_forces[:, env.feet_indices]),
                key_positions_w=cpu(env.rigid_state[:, body_ids, :3]-env.env_origins[:, None, :]),
                valid=cpu(alive), failure=cpu(env.reset_buf.bool() & ~env.time_out_buf.bool()),
                done=cpu(env.reset_buf.bool()), native_episode=cpu(env.amp_episode))
            result.update({key: cpu(value) for key, value in env.jitter_capture().items()})
            return result
        original_termination = env.check_termination
        def check_and_capture():
            original_termination()
            if recording:
                value = capture()  # includes failing tick BEFORE automatic reset.
                value.update(pending)
                frames.append(value)
        env.check_termination = check_and_capture
        # Same native reference reset/zero-action 10ms warmup; seed is new ONLY.
        set_seed(contract['seed'])
        obs, _ = env.reset()
        initial = capture()
        initial_obs = cpu(obs)
        initial_native_episode = cpu(env.amp_episode)
        alive &= ~env.reset_buf.bool()
        if reset_pending:
            raise ValueError('No actual reference reset/history-clear evidence')
        recording = True
        inference_context = None
        with torch.no_grad():
            for tick in range(contract['expected_ticks']):
                full = cpu(obs)
                command = cpu(env.commands)
                actual_context = source_forward_context(policy, obs)
                validate_source_forward_context(actual_context, native=True)
                if inference_context is None:
                    inference_context = actual_context
                elif actual_context != inference_context:
                    raise ValueError('Source numerical context changed during native collection')
                actions, hidden = hook.inference(obs)
                if source_forward_context(policy, obs) != actual_context:
                    raise ValueError('Source numerical context changed during native forward')
                pending = dict(full_obs=full, single_obs=full[:, -OBS:].copy(),
                    actor_hidden=cpu(hidden), raw_mu=cpu(actions), command=command,
                    tick=np.full(env.num_envs, tick, dtype=np.int64),
                    episode_id=np.zeros(env.num_envs, dtype=np.int64))
                obs, _, _, dones, _ = env.step(actions)
                alive &= ~dones.bool()
                if not alive.any():
                    break
                if (tick+1) % 500 == 0:
                    print('[head-cohort-progress] '+json.dumps(dict(split=contract['split'],
                        ticks=tick+1, alive=int(alive.sum()))), flush=True)
        recording = False
        hook.close()
        if hook.calls != len(frames) or state_fingerprint(policy.state_dict()) != SOURCE_MODEL_SHA:
            raise ValueError('Source inference changed weights or used extra final-head forwards')
        arrays = {'reference_'+key: np.stack([row[key] for row in frames]) for key in frames[0]}
        arrays['reference_time'] = (np.arange(len(frames))+1)*.01
        for key in INITIAL_FIELDS:
            arrays['reference_initial_'+key] = initial[key]
        arrays['reference_initial_full_obs'] = initial_obs
        arrays['reference_initial_native_episode'] = initial_native_episode
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite native archive field '+key)
        clip = cfg.normalization.clip_actions
        if not np.array_equal(arrays['reference_action'][arrays['reference_valid']],
                              np.clip(arrays['reference_raw_mu'], -clip, clip)[arrays['reference_valid']]):
            raise ValueError('Actual native applied action is not the clipped original raw mean')
        ready, real_frames, history_proof, episodes = audit_arrays(arrays, contract)
        arrays['reference_history_ready'] = ready
        arrays['reference_history_real_frames'] = real_frames
        fingerprints, dedupe = fingerprint_cohort(arrays, contract, ready, exclusions)
        eligible = fit_admission(contract, episodes, history_proof, dedupe)
        props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
        environment = class_to_dict(cfg)
        runtime = dict(dof_properties={key: props[key].tolist() for key in props.dtype.names},
            p_gains=cpu(env.p_gains).tolist(), d_gains=cpu(env.d_gains).tolist(),
            control_dt=env.dt, physics_dt=env.sim_params.dt, **physical_readback(env))
        runtime_proof = validate_source_runtime(environment, runtime, source_manifest, contract)
        manifest = dict(contract, code_commit=commit, identity=experiment.identity(),
            implementation_fingerprint=implementation_fingerprint(repo),
            source_task=SOURCE_TASK, source_checkpoint='model_8802500.pt',
            source_checkpoint_sha256=SOURCE_SHA, source_model_state_sha256=SOURCE_MODEL_SHA,
            source_completed_updates=2500, source_regression=source_proof,
            previous_cohorts=prior_proofs, parent_frozen=True, no_training=True,
            policy_deterministic=True, no_dr=gate, environment=environment,
            modes=['reference'], dof_names=env.dof_names,
            body_names=list(experiment.spec.body_names), foot_names=list(FOOT_NAMES),
            urdf_lf_sha256=experiment.kinematics.lf_sha256,
            runtime=runtime, source_runtime_proof=runtime_proof,
            initialization=dict(protocol='original_reference_reset_then_native_10ms_zero_action_warmup',
                reset_inputs=reset_inputs, cleared_histories=reset_proof,
                all12_initial_fields=list(INITIAL_FIELDS), actual_initial_full_obs=True),
            inference_capture=dict(actor_module='actor.6', hidden_dim=HIDDEN,
                action_dim=ACTIONS, actual_forwards=hook.calls, extra_forward=False,
                actual_output_exact=True, model_state_unchanged=True,
                numerical_context=inference_context),
            history_proof=history_proof, fingerprints=fingerprints, deduplication=dedupe,
            episodes=episodes, fit_eligible=eligible,
            capture='all first-episode valid prefixes including failing tick before native reset; invalid tails retained',
            substep_telemetry=dict(physics_hz=1000, control_hz=100, samples_per_control=10,
                reset_guard='native physics_valid excludes first three intervals',
                acceleration='mean square and peak abs native dq / .001',
                torque_delta='mean square actual command difference per 1ms; not derivative',
                foot_force='max positive NET foot Fz over10 intervals, not contact-pair identity'),
            evaluation_protocol=SCHEMA, effectiveness_verified=False, dr_unlocked=False,
            limitation='Smoke validates implementation only. Sealed cohort cannot be fitted. '
                'Archive success is not training continuation, physical improvement or deployment acceptance.')
        validate_initialization_proof(manifest['initialization'], contract)
        packed = io.BytesIO()
        np.savez_compressed(packed, **arrays)
        extra.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves every prior artifact and evidence file.
        with extra.output.open('xb') as stream:
            torch.save(dict(npz_bytes=packed.getvalue(), manifest_json=json.dumps(manifest)), stream)
        with extra.output.with_suffix('.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
        print('[head-cohort-complete] '+json.dumps(dict(split=contract['split'],
            mode=contract['mode'], fit_eligible=eligible, episodes=episodes,
            deduplication_passed=dedupe['passed'])), flush=True)
        return manifest
    finally:
        env.gym.destroy_sim(env.sim)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--split', choices=('train', 'validation', 'val', 'sealed'), required=True)
    parser.add_argument('--mode', choices=tuple(BUDGETS), required=True)
    parser.add_argument('--checkpoint-file', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-endpoint', type=Path)
    source.add_argument('--source-exclusion-index', type=Path)
    parser.add_argument('--exclude-cohort', type=Path, action='append', default=[])
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    extra, remaining = parser.parse_known_args()
    return _native_main(extra, remaining)


if __name__ == '__main__':
    main()
