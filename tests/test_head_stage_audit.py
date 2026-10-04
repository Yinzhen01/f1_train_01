"""Synthetic independent audit tests; NO cloud, real-model fitting or simulation."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from humanoid.amp import head_workflow
from humanoid.amp.head_artifact import (
    ORIGINAL110_PARENT, assemble_head_artifact, validate_head_artifact)
from humanoid.amp.head_smoothing import evaluate_head_candidate, validate_head_only_change
from humanoid.amp.head_workflow import (
    audit_cohort, exclusion_add_cohort, fit_episodes, forward_parity, identity_digest)
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.mu_temporal import state_fingerprint
from humanoid.scripts.collect_amp_head_cohort import (
    INITIAL_FIELDS, FrozenActorCapture, SEEDS, audit_arrays, cohort_contract,
    empty_exclusion_index, fingerprint_cohort, validate_source_runtime)
from humanoid.scripts.record_amp_head_policy import (
    audit_rollout_arrays, initial_source_matches)
from tools.amp import audit_head_stage as stage
from test_head_artifact import synthetic_source, synthetic_provenance
from test_head_cohort import fixture as array_fixture
from test_head_policy_recorder import rollout_fixture
from test_head_workflow import initialization_fixture, native_sim_fixture, source_fixture


ROOT = Path(__file__).resolve().parents[1]
COMMIT, FP, TASK = '1'*40, '2'*64, 'TASK_20990101_001'


def platform_fixture():
    return dict(success=True, data=dict(taskBaseInfo=dict(taskId=TASK, userId=4409,
        taskStatus='5', goodsId='ESKU000001', gpuNum=1), taskCodeInfo=dict(taskId=TASK,
        userId=4409, commitId=COMMIT, codeType='2', startScript='gm-run '
        '/workspace/humanoid/scripts/run_amp_head_smooth.py --mode smoke '
        '--expected-commit '+COMMIT+' --headless')))


def hardware_fixture():
    """Explicit synthetic typed runtime record, never native hardware evidence."""
    return dict(schema='head_native_runtime_hardware_v1', platform='linux', cuda_available=True,
        cuda_device_count=1, devices=[dict(index=0, name='NVIDIA GeForce RTX 4090',
            total_memory_bytes=25393692672, compute_capability=[8, 9])],
        torch_version='2.4.1', torch_cuda_version='12.1')


def synthetic_native_context():
    """Typed fake GPU snapshot for this synthetic protocol fixture, NOT readback."""
    return dict(schema='source_forward_numerical_context_v1', device_type='cuda', device_index=0,
        input_dtype='torch.float32', grad_enabled=False, policy_training=False,
        cudnn_enabled=True, cudnn_deterministic=True, cudnn_benchmark=False,
        cudnn_allow_tf32=True, cuda_matmul_allow_tf32=False,
        float32_matmul_precision='highest', torch_version='2.4.1',
        torch_cuda_version='12.1', cudnn_version=8902)


def bundle(path, manifest, arrays):
    memory = io.BytesIO()
    np.savez_compressed(memory, **arrays)
    torch.save(dict(npz_bytes=memory.getvalue(), manifest_json=json.dumps(manifest)), path)


def synthetic_pipeline(directory):
    """Actual network architecture on CPU-generated arrays, explicitly test-only."""
    spec = importlib.util.spec_from_file_location('stage_fixture_actor',
        ROOT/'humanoid/algo/ppo/actor_critic_dh.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normal = torch.distributions.Normal
    validation_owned = 'set_default_validate_args' in normal.__dict__
    validation_default = normal.__dict__.get('set_default_validate_args')
    try:
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(111)
            policy = module.ActorCriticDH(235, 47, 219, 12, actor_hidden_dims=[16, 12, 128],
                critic_hidden_dims=[8, 8, 8], state_estimator_hidden_dims=[8, 8, 8])
    finally:
        if validation_owned: normal.set_default_validate_args = validation_default
        else: del normal.set_default_validate_args
    policy.eval()
    source = synthetic_source()
    source['model_state_dict'] = policy.state_dict()
    source_path = directory/'synthetic_source.pt'
    torch.save(source, source_path)
    parent = dict(ORIGINAL110_PARENT, source_task='SYNTHETIC_ONLY',
        source_config='synthetic_only.json', checkpoint_name=source_path.name,
        file_sha256=stage.file_sha(source_path), model_state_sha256=state_fingerprint(policy.state_dict()))
    identity = source['amp_identity']
    source_manifest = source_fixture()
    source_manifest['environment']['normalization'] = dict(clip_actions=1000.)
    index = empty_exclusion_index()
    index['initial_state']['a'*64] = ['source110/standing/env-0/episode-0']
    protected = copy.deepcopy(index)
    proof = dict(sha256=stage.SOURCE_ENDPOINT_SHA, source_task='TASK_20260926_110',
        regression_episode_ids=['source110/standing/env-0/episode-0'],
        reconstruction_not_raw_history=True, limitation='SYNTHETIC TEST ONLY',
        derived_index=dict(path=str(ROOT/'resources/amp_inputs/source110_exclusions_v1.npz'),
                           sha256=stage.INDEX_SHA, hash_bits=256))
    paths, manifests, all_arrays, audits, parities = {}, {}, {}, {}, {}
    with patch.object(head_workflow, 'SOURCE_SHA', parent['file_sha256']), \
         patch.object(head_workflow, 'SOURCE_MODEL_SHA', parent['model_state_sha256']):
        for split in ('train', 'validation'):
            contract = cohort_contract(split, 'smoke', SEEDS[split], 4, 2.)
            _, arrays = array_fixture()
            arrays['reference_key_positions_w'] = arrays['reference_key_positions_w'][:, :, :4].copy()
            arrays['reference_initial_key_positions_w'] = arrays['reference_initial_key_positions_w'][:, :4].copy()
            singles = np.random.RandomState(SEEDS[split]).normal(0, .1, (200, 4, 47)).astype(np.float32)
            arrays['reference_single_obs'] = singles
            full = np.zeros((200, 4, 3102), dtype=np.float32)
            for tick in range(200):
                rows = singles[max(0, tick-65):tick+1].transpose(1, 0, 2).reshape(4, -1)
                full[tick, :, -rows.shape[1]:] = rows
            arrays['reference_full_obs'] = full
            arrays['reference_initial_full_obs'] = full[0].copy()
            for key in INITIAL_FIELDS:
                value = arrays['reference_initial_'+key]
                if value.dtype == np.float32:
                    value += 100. if split == 'validation' else 10.
            hook = FrozenActorCapture(policy)
            try:
                with torch.no_grad():
                    mu, hidden = hook.inference(torch.from_numpy(full.reshape(-1, 3102)))
            finally:
                hook.close()
            arrays['reference_actor_hidden'] = hidden.numpy().reshape(200, 4, 128).copy()
            arrays['reference_raw_mu'] = mu.numpy().reshape(200, 4, 12).copy()
            arrays['reference_action'] = np.clip(arrays['reference_raw_mu'], -1000., 1000.)
            ready, _, history, episodes = audit_arrays(arrays, contract)
            fingerprints, dedupe = fingerprint_cohort(arrays, contract, ready, index)
            env, runtime = copy.deepcopy(source_manifest['environment']), copy.deepcopy(source_manifest['runtime'])
            env['seed'] = SEEDS[split]
            env['env'].update(num_envs=4, episode_length_s=2.1)
            runtime['physics_sim_parameters'] = native_sim_fixture(env['sim'])
            prior = [] if split == 'train' else [dict(sha256=stage.file_sha(paths['train']),
                split='train', seed=305, num_envs=4, duration_s=2., episode_ids=manifests['train']['episode_ids'])]
            manifest = dict(contract, source_checkpoint_sha256=parent['file_sha256'],
                source_model_state_sha256=parent['model_state_sha256'], identity=identity,
                source_completed_updates=2500, code_commit=COMMIT, implementation_fingerprint=FP,
                policy_deterministic=True, parent_frozen=True, no_training=True,
                effectiveness_verified=False, dr_unlocked=False, source_regression=copy.deepcopy(proof),
                environment=env, runtime=runtime,
                initialization=initialization_fixture(4),
                source_runtime_proof=validate_source_runtime(env, runtime, source_manifest, contract),
                fingerprints=fingerprints, history_proof=history, episodes=episodes,
                deduplication=dedupe, fit_eligible=False, previous_cohorts=prior,
                inference_capture=dict(actor_module='actor.6', hidden_dim=128, action_dim=12,
                    actual_forwards=200, extra_forward=False, actual_output_exact=True,
                    model_state_unchanged=True, numerical_context=synthetic_native_context()),
                body_names=list(stage.SOURCE_BODIES))
            paths[split] = directory/('model_700000%d.pt' % (1 if split == 'train' else 2))
            bundle(paths[split], manifest, arrays)
            audit = audit_cohort(manifest, arrays, split=split, mode='smoke', identity=identity,
                code_commit=COMMIT, implementation_fingerprint=FP, exclusions=index,
                source_manifest=source_manifest)
            parities[split] = forward_parity(policy, arrays, action_clip=1000., batch_size=4,
                expected_model_fingerprint=parent['model_state_sha256'])
            # Explicit synthetic GPU-shaped claimed metadata only. The audit
            # independently retains its actual CPU context and never certifies
            # this fake native snapshot as a real cloud/runtime result.
            parities[split]['numerical_context'] = synthetic_native_context()
            exclusion_add_cohort(index, audit)
            manifests[split], all_arrays[split], audits[split] = manifest, arrays, audit
    rows = [fit_episodes(manifests[s], all_arrays[s], audits[s], identity=identity)
            for s in ('train', 'validation')]
    old = source['model_state_dict']
    exported = evaluate_head_candidate(*rows, old['actor.6.weight'].numpy(), old['actor.6.bias'].numpy(),
        old['actor.6.weight'].numpy(), old['actor.6.bias'].numpy(),
        source_identity=identity_digest(identity), candidate_identity=identity_digest(identity),
        action_clip=1000., forbidden_cohort_ids=['source110/standing/env-0/episode-0'])
    fitted = {key: copy.deepcopy(value) for key, value in exported.items() if key not in
              ('candidate_identity', 'candidate_adjusted', 'solver_run')}
    fitted.update(audit_origin='float64_closed_form_training', candidate_head_dtype='float64',
                  direction_scale=0., unconstrained_training_output_change_peak_rad=0.)
    start = dict(schema='head_smooth_native_stage_v1', mode='smoke', code_commit=COMMIT,
        native_hardware=hardware_fixture(),
        implementation_fingerprint=FP, identity=identity, source_checkpoint_sha256=parent['file_sha256'],
        source_exclusions=proof, num_envs=4, duration_s=2., seeds=dict(train=305, validation=505),
        ppo_updates_added=0, optimizer_state_reused=False, effectiveness_verified=False, dr_unlocked=False)
    report = dict(start=start, cohort_sha256={s: stage.file_sha(paths[s]) for s in manifests},
        source_parity=parities, effectiveness_verified=False, dr_unlocked=False, solver_runs=1,
        admitted_to_physical_test=False, float64_fit=fitted, exported_head=exported, status='offline_rejected')
    math_text = json.dumps(report, sort_keys=True, allow_nan=False)
    provenance = synthetic_provenance('smoke', False)
    provenance.update(code_commit=COMMIT, implementation_fingerprint=FP,
        report_sha256=hashlib.sha256(math_text.encode()).hexdigest(),
        cohorts={s: dict(sha256=stage.file_sha(paths[s]), seed=SEEDS[s], num_envs=4,
            duration_s=2., episode_ids=manifests[s]['episode_ids']) for s in manifests},
        source_parity=dict(full_obs_verified=True, hidden_verified=True, action_verified=True,
                           max_abs_error=max(row['max_abs_error'] for row in parities.values())))
    provenance['solver']['direction_scale'] = 0.
    artifact = assemble_head_artifact(source_path, {k: old[k] for k in ('actor.6.weight', 'actor.6.bias')},
                                      provenance=provenance, expected_parent=parent)
    paths['head'] = directory/'model_7000003.pt'
    torch.save(artifact, paths['head'])
    archive = validate_head_artifact(artifact, source_path, expected_parent=parent)
    report.update(head_sha256=stage.file_sha(paths['head']), archive_audit=archive,
        frozen_audit=validate_head_only_change(old, artifact['model_state_dict'],
            identity_digest(identity), identity_digest(identity)), provenance_report_json=math_text)
    contract, data = rollout_fixture()
    for name in contract['modes']:
        data[name+'_key_positions_w'] = data[name+'_key_positions_w'][:, :, :4].copy()
        data[name+'_initial_key_positions_w'] = data[name+'_initial_key_positions_w'][:, :4].copy()
        data[name+'_physics_valid'][3:] = True
        # Consistent CPU-only native-shaped telemetry, never a simulation.
        data[name+'_physics_raw_body_state'][..., 6] = 1.
        data[name+'_foot_state'][..., 6] = 1.
        data[name+'_initial_foot_state'][..., 6] = 1.
    env, runtime = copy.deepcopy(source_manifest['environment']), copy.deepcopy(source_manifest['runtime'])
    env['env'].update(num_envs=2, episode_length_s=1.1)
    runtime['physics_sim_parameters'] = native_sim_fixture(env['sim'])
    episodes = audit_rollout_arrays(data, contract)
    resets, cleared, matches = {}, {}, {}
    for name in ('standing', 'reference'):
        resets[name] = dict(root_states=np.zeros((2, 13)).tolist(),
            **{k: np.zeros((2, 12)).tolist() for k in
               ('dof_pos', 'dof_vel', 'actions', 'last_actions', 'last_last_actions')},
            commands=np.zeros((2, 4)).tolist(), **{k: [0, 0] for k in
               ('gait_start', 'episode_length_buf', 'phase_length_buf', 'rsi_indices')},
            obs_history_sha256='b'*64, critic_history_sha256='c'*64)
        cleared[name] = {k: dict(elements=10, nonzero=0, nonfinite=0, negative_zero=0)
                         for k in ('obs_history', 'critic_history')}
        matches[name] = initial_source_matches({k: data[name+'_initial_'+k] for k in INITIAL_FIELDS},
                                               name, index, enforce=False)
    recorder = dict(contract, code_commit=COMMIT, implementation_fingerprint=FP, identity=identity,
        source_identity=identity, parent_checkpoint_sha256=parent['file_sha256'],
        parent_model_state_sha256=parent['model_state_sha256'], head_artifact_sha256=None,
        policy_model_state_sha256=parent['model_state_sha256'], policy_variant='source_smoke_probe',
        head_archive_audit=None, source_regression=proof, no_dr=assert_no_domain_randomization(env),
        environment=env, runtime=runtime, source_runtime_proof=validate_source_runtime(env, runtime, source_manifest, contract),
        policy_deterministic=True, episodes=episodes, evaluation_protocol='head_source_smoke',
        body_names=list(stage.SOURCE_BODIES),
        initialization=dict(protocol='identical_reset_inputs_then_native_10ms_warmup',
            reset_inputs=resets, cleared_histories=cleared, initial_source_matches=matches),
        capture='pre-reset; retain full failing first-episode prefixes and invalid tails',
        action_semantics='policy_raw_mu is actual unclipped deterministic mean; action is actual clipped PD command',
        substep_telemetry=dict(physics_hz=1000, control_hz=100, samples_per_control=10,
            raw_env=0, raw_body_names=['base_link', 'left_ankle_roll_link', 'right_ankle_roll_link'],
            reward_used=False, reset_guard='native physics_valid excludes first three reset intervals',
            acceleration='native1ms velocity difference/.001 mean squared and peak absolute',
            torque_delta='actual1ms torque difference mean squared; not derivative',
            foot_force='max positive NET footFz, not contact-pair identity'))
    paths['policy_rollout'] = directory/'model_7000005.pt'
    bundle(paths['policy_rollout'], recorder, data)
    reference = dict(artifact_name=paths['policy_rollout'].name, sha256=stage.file_sha(paths['policy_rollout']),
        type=contract['type'], mode='smoke', num_envs=2, duration_s=1., head_artifact_sha256=None,
        source_checkpoint_sha256=parent['file_sha256'], policy_model_state_sha256=parent['model_state_sha256'],
        evaluation_protocol='head_source_smoke', episodes=episodes)
    report['policy_probe'] = reference
    policy_audit = stage.audit_policy_rollout(paths['policy_rollout'], reference, mode='smoke',
        identity=identity, binding=parent, commit=COMMIT, fingerprint=FP, artifact=artifact,
        archive=archive, source_proof=proof, source_manifest=source_manifest, exclusions=index)
    paths['report'] = directory/'model_7000004.pt'
    torch.save(dict(report_json=json.dumps(report)), paths['report'])
    log = stage.HARDWARE+json.dumps(start['native_hardware'])+'\n'+stage.START+json.dumps(start)+'\n'
    for split in ('train', 'validation'):
        log += stage.COHORT+json.dumps(dict(split=split, mode='smoke', fit_eligible=False,
            episodes=audits[split]['episodes'], deduplication_passed=True))+'\n'
    log += stage.POLICY_START+json.dumps(policy_audit['start'])+'\n'
    log += stage.POLICY_COMPLETE+json.dumps(policy_audit['complete'])+'\n'
    log += stage.COMPLETE+json.dumps(report)+'\n'
    log += ''.join('[SDK][INFO] 2099-01-01 00:00:00 PT file (model_700000%d.pt) uploaded successfully\n' % i
                   for i in range(1, 6))
    log += '【SDK】训练进程状态码：0\n[SDK][INFO] 2099-01-01 00:00:00 Task('+TASK+') status updated to: Completed , ret:True\n'
    cloud_log, info = directory/'cloud.log', directory/'info.json'
    cloud_log.write_text(log, encoding='utf-8')
    info.write_text(json.dumps(platform_fixture()), encoding='utf-8')
    return dict(source_checkpoint=source_path, expected_parent=parent, cloud_log=cloud_log,
        platform_info=info, expected_commit=COMMIT, mode='smoke', **paths), \
        dict(index=protected, proof=proof, source_manifest=source_manifest, report=report,
             audits=audits, policy=policy_audit, log=log, recorder=recorder, policy_arrays=data)


class IndependentHeadStageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.args, cls.fixture = synthetic_pipeline(Path(cls.temp.name))
        cls.snapshots = {key: value.read_bytes() for key, value in cls.args.items() if isinstance(value, Path)}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        for key, data in self.snapshots.items(): self.args[key].write_bytes(data)

    def audit(self, **overrides):
        args = dict(self.args, **overrides)
        with patch.object(head_workflow, 'SOURCE_SHA', args['expected_parent']['file_sha256']), \
             patch.object(head_workflow, 'SOURCE_MODEL_SHA', args['expected_parent']['model_state_sha256']), \
             patch.object(stage, 'implementation_fingerprint', return_value=FP), \
             patch.object(stage, 'registered_source_exclusions', return_value=(
                 copy.deepcopy(self.fixture['index']), copy.deepcopy(self.fixture['proof']),
                 copy.deepcopy(self.fixture['source_manifest']))), \
             patch('humanoid.amp.head_smoothing._solve', side_effect=AssertionError('audit must not solve')):
            return stage.audit_head_stage(**args)

    def refresh_report_binding(self, value):
        """Keep wrapper file hashes consistent to reach the independent deep gates."""
        torch.save(dict(report_json=json.dumps(value)), self.args['report'])
        log = self.fixture['log'].replace(stage.COMPLETE+json.dumps(self.fixture['report']),
                                         stage.COMPLETE+json.dumps(value))
        self.args['cloud_log'].write_text(log, encoding='utf-8')

    def test_complete_synthetic_five_file_probe_recomputes_but_never_certifies_native(self):
        before = {key: stage.file_sha(path) for key, path in self.args.items() if isinstance(path, Path)}
        rng = torch.get_rng_state().clone()
        result = self.audit()
        facts = result['facts']
        self.assertTrue(facts['verified'])
        self.assertTrue(facts['synthetic_fixture_only'])
        self.assertFalse(facts['native_verified'])
        self.assertNotIn('smoke_certificate', result)
        self.assertTrue(facts['native_recorder_verified'])
        self.assertTrue(facts['native_hardware_record_verified'])
        self.assertTrue(facts['logs']['native_hardware_log_bound'])
        self.assertEqual(facts['native_hardware'], hardware_fixture())
        self.assertEqual(facts['native_hardware']['devices'][0]['name'], 'NVIDIA GeForce RTX 4090')
        self.assertIn('does not establish', facts['hardware_evidence_scope'])
        self.assertTrue(facts['native_numerical_context_verified'])
        for split in ('train', 'validation'):
            native, cpu = (facts[key][split] for key in
                ('native_recorded_source_parity', 'recomputed_cpu_source_parity'))
            self.assertEqual(native['batch_size'], 4)
            self.assertEqual(native['forward_batches'], 200)
            self.assertEqual(native['numerical_context']['device_type'], 'cuda')
            self.assertEqual(cpu['batch_size'], 4)
            self.assertEqual(cpu['forward_batches'], 200)
            self.assertEqual(cpu['numerical_context']['device_type'], 'cpu')
        self.assertIn('neither reproduces nor certifies GPU', facts['cpu_forward_scope'])
        self.assertTrue(facts['logs_clean'])
        self.assertFalse(facts['offline_admitted'])
        self.assertFalse(facts['formal_admission'])
        self.assertFalse(facts['effectiveness_verified'])
        self.assertFalse(facts['dr_unlocked'])
        self.assertEqual(facts['auditor_solver_runs'], 0)
        self.assertEqual(facts['recomputed_cpu_source_parity']['train']['actual_rows'], 800)
        self.assertEqual(len(facts['logs']['upload_receipts']), 5)
        self.assertEqual(result['audit_facts_sha256'], stage.canonical_sha(facts))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, {key: stage.file_sha(path) for key, path in self.args.items() if isinstance(path, Path)})

    def test_required_typed_hardware_missing_unsupported_and_malformed_records_rejected(self):
        def changed(path, value):
            record = hardware_fixture()
            cursor = record
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            return record

        invalid = [None, {}, dict(hardware_fixture(), sku='ESKU000001')]
        cases = [
            (('schema',), 'head_native_runtime_hardware_v0'),
            (('platform',), 'win32'),
            (('cuda_available',), False), (('cuda_available',), 1),
            (('cuda_device_count',), 0), (('cuda_device_count',), 2),
            (('cuda_device_count',), True), (('cuda_device_count',), 1.),
            (('devices',), []),
            (('devices', 0, 'index'), True), (('devices', 0, 'index'), 1),
            (('devices', 0, 'name'), 'NVIDIA A100-SXM4-80GB'),
            (('devices', 0, 'name'), 'NVIDIA L20'),
            (('devices', 0, 'name'), 'NVIDIA GeForce RTX 4090 Laptop GPU'),
            (('devices', 0, 'name'), 'NVIDIA GeForce RTX 4090Dextra'),
            (('devices', 0, 'name'), 'RTX 4090'),
            (('devices', 0, 'name'), 'NVIDIA\tGeForce RTX 4090'),
            (('devices', 0, 'name'), 4090),
            (('devices', 0, 'total_memory_bytes'), 23*1024**3-1),
            (('devices', 0, 'total_memory_bytes'), 25*1024**3+1),
            (('devices', 0, 'total_memory_bytes'), 25393692672.),
            (('devices', 0, 'compute_capability'), [8, 0]),
            (('devices', 0, 'compute_capability'), [True, 9]),
            (('devices', 0, 'compute_capability'), [8., 9]),
            (('torch_version',), ''), (('torch_version',), None),
            (('torch_cuda_version',), ''), (('torch_cuda_version',), None),
        ]
        invalid.extend(changed(path, value) for path, value in cases)
        missing = hardware_fixture(); missing.pop('cuda_available'); invalid.append(missing)
        extra = hardware_fixture(); extra['devices'][0]['sku'] = 'ESKU000001'; invalid.append(extra)
        for number, record in enumerate(invalid):
            report = copy.deepcopy(self.fixture['report'])
            if record is None:
                report['start'].pop('native_hardware')
            else:
                report['start']['native_hardware'] = record
            self.refresh_report_binding(report)
            with self.subTest(case=number), self.assertRaisesRegex(ValueError, 'Native head hardware:'):
                self.audit()

    def test_native_parity_exact_schema_batch_context_capture_and_versions_rejected(self):
        from humanoid.scripts.collect_amp_head_cohort import _read_bundle
        manifest, arrays = _read_bundle(self.args['train'], torch)
        parity = self.fixture['report']['source_parity']['train']
        capture = manifest['inference_capture']
        stage._parity_record(parity, arrays, capture, hardware_fixture())
        cases = [('batch_size', 256), ('batch_size', True), ('batch_size', 4.),
            ('forward_batches', 4), ('forward_batches', True), ('actual_rows', 799),
            ('max_abs_error', .001), ('max_hidden_error', float('nan'))]
        for field, value in cases:
            changed = copy.deepcopy(parity); changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                stage._parity_record(changed, arrays, capture, hardware_fixture())
        for field in ('batch_size', 'numerical_context'):
            changed = copy.deepcopy(parity); changed.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                stage._parity_record(changed, arrays, capture, hardware_fixture())
        context_cases = [('device_type', 'cpu'), ('device_index', True), ('device_index', 1),
            ('input_dtype', 'torch.float64'), ('grad_enabled', 0), ('grad_enabled', True),
            ('policy_training', True), ('cudnn_enabled', False), ('cudnn_deterministic', False),
            ('cudnn_benchmark', True), ('cudnn_allow_tf32', 1), ('cuda_matmul_allow_tf32', 0),
            ('float32_matmul_precision', 'unknown'), ('torch_version', ''),
            ('torch_cuda_version', None), ('cudnn_version', None), ('cudnn_version', True)]
        for field, value in context_cases:
            changed = copy.deepcopy(parity); changed['numerical_context'][field] = value
            with self.subTest(context_field=field), self.assertRaises(ValueError):
                stage._parity_record(changed, arrays, capture, hardware_fixture())
        for field, value in (('cuda_matmul_allow_tf32', True), ('cudnn_allow_tf32', False),
                ('float32_matmul_precision', 'high'), ('cudnn_version', 9100)):
            changed = copy.deepcopy(parity); changed['numerical_context'][field] = value
            with self.subTest(valid_context_mismatch=field), self.assertRaises(ValueError):
                stage._parity_record(changed, arrays, capture, hardware_fixture())
        for field, value in (('torch_version', '2.5.0'), ('torch_cuda_version', '12.2')):
            changed, cap = copy.deepcopy(parity), copy.deepcopy(capture)
            changed['numerical_context'][field] = cap['numerical_context'][field] = value
            with self.subTest(hardware_version_mismatch=field), self.assertRaises(ValueError):
                stage._parity_record(changed, arrays, cap, hardware_fixture())
        for field, value in (('actual_forwards', 199), ('actual_forwards', True),
                ('extra_forward', 0), ('actual_output_exact', False), ('model_state_unchanged', False)):
            cap = copy.deepcopy(capture); cap[field] = value
            with self.subTest(capture_field=field), self.assertRaises(ValueError):
                stage._parity_record(parity, arrays, cap, hardware_fixture())
        for cap in (None, {}, dict(capture, unregistered_field=True)):
            with self.subTest(capture_shape=type(cap).__name__), self.assertRaises(ValueError):
                stage._parity_record(parity, arrays, cap, hardware_fixture())

    def test_valid_hardware_cannot_override_unregistered_platform_resource(self):
        raw = platform_fixture()
        raw['data']['taskBaseInfo']['goodsId'] = 'ESKU000002'
        self.args['platform_info'].write_text(json.dumps(raw), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'wrong owner/terminal status/resource/GPU count'):
            self.audit()

    def test_hardware_marker_missing_duplicate_tampered_or_after_start_rejected(self):
        hardware = stage.HARDWARE+json.dumps(self.fixture['report']['start']['native_hardware'])+'\n'
        start = stage.START+json.dumps(self.fixture['report']['start'])+'\n'
        changed = hardware_fixture(); changed['devices'][0]['name'] = 'NVIDIA GeForce RTX 4090 D'
        wrong = stage.HARDWARE+json.dumps(changed)+'\n'
        log = self.fixture['log']
        mutations = [log.replace(hardware, ''), log+hardware,
            log.replace(hardware, wrong), log.replace(hardware, '').replace(start, start+hardware)]
        for number, value in enumerate(mutations):
            with self.subTest(case=number), self.assertRaises(ValueError):
                stage.validate_log(value, self.fixture['report'], self.fixture['audits'], TASK, 'smoke',
                    dict(escaped_invalid_utf8=False, control_bytes=0), policy=self.fixture['policy'])
        report = copy.deepcopy(self.fixture['report'])
        report['start'].pop('native_hardware')
        no_record = log.replace(hardware, '').replace(start, stage.START+json.dumps(report['start'])+'\n')
        no_record = no_record.replace(stage.COMPLETE+json.dumps(self.fixture['report']),
                                      stage.COMPLETE+json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'Native head hardware:'):
            stage.validate_log(no_record, report, self.fixture['audits'], TASK, 'smoke',
                dict(escaped_invalid_utf8=False, control_bytes=0), policy=self.fixture['policy'])

    def test_wrong_parent_and_modified_learning_archive_rejected(self):
        parent = dict(self.args['expected_parent'], file_sha256='3'*64)
        with self.assertRaises(ValueError): self.audit(expected_parent=parent)
        value = torch.load(self.args['head'], map_location='cpu', weights_only=True)
        value['source_archive']['optimizer_state_dict']['state'][0]['step'] += 1
        torch.save(value, self.args['head'])
        report = copy.deepcopy(self.fixture['report'])
        report['head_sha256'] = stage.file_sha(self.args['head'])
        self.refresh_report_binding(report)
        with self.assertRaisesRegex(ValueError, 'source_archive'): self.audit()

    def test_missing_fifth_artifact_and_tampered_policy_sha_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fifth policy artifact'): self.audit(policy_rollout=None)
        self.args['policy_rollout'].write_bytes(self.snapshots['policy_rollout']+b'changed')
        with self.assertRaises(ValueError): self.audit()

    def test_platform_owner_terminal_resource_gpu_commit_and_options_strict(self):
        cases = [('taskBaseInfo', 'userId', 4408), ('taskBaseInfo', 'taskStatus', '4'),
            ('taskBaseInfo', 'goodsId', 'OTHER'), ('taskBaseInfo', 'gpuNum', True),
            ('taskCodeInfo', 'commitId', '3'*40), ('taskCodeInfo', 'taskId', 'TASK_20990101_002')]
        cases.append(('taskCodeInfo', 'userId', '4408'))
        for group, key, value in cases:
            raw = platform_fixture(); raw['data'][group][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): stage.validate_platform(raw, COMMIT, 'smoke')
        for extra in (' --mode formal', '; echo x', ' --source-endpoint /old', ' --checkpoint-file https://secret?key=x'):
            raw = platform_fixture(); raw['data']['taskCodeInfo']['startScript'] += extra
            with self.subTest(extra=extra), self.assertRaises(ValueError): stage.validate_platform(raw, COMMIT, 'smoke')

    def test_real_blank_optional_code_owner_is_preserved_not_fabricated(self):
        for value in ('', None):
            raw = platform_fixture(); raw['data']['taskCodeInfo']['userId'] = value
            result = stage.validate_platform(raw, COMMIT, 'smoke')
            self.assertEqual(result['owner'], '4409')
            self.assertEqual(result['code_owner_metadata'], value)
            self.assertFalse(result['code_owner_metadata_present'])

    def test_log_missing_duplicate_wrong_order_exit_and_upload_rejected(self):
        log = self.fixture['log']
        mutations = [log.replace(stage.POLICY_START, '[wrong-policy-start] '),
            log+stage.START+json.dumps(self.fixture['report']['start'])+'\n',
            log.replace('【SDK】训练进程状态码：0', '【SDK】训练进程状态码：1'),
            log.replace('model_7000005.pt) uploaded successfully', 'model_7999999.pt) uploaded successfully'),
            log+'Learning iteration 1\n', log.replace(TASK+') status updated', 'TASK_20990101_002) status updated')]
        for value in mutations:
            with self.subTest(value=value[-100:]), self.assertRaises(ValueError):
                stage.validate_log(value, self.fixture['report'], self.fixture['audits'], TASK, 'smoke',
                    dict(escaped_invalid_utf8=False, control_bytes=0), policy=self.fixture['policy'])

    def test_external_sdk_traceback_retained_not_clean_native_error_rejected(self):
        sdk = 'Traceback (most recent call last):\n  File "/opt/pika/client.py", line 1, in receive\n    receive()\nRuntimeError: SDK queue closed\n'
        result = stage.validate_log(sdk+self.fixture['log'], self.fixture['report'],
            self.fixture['audits'], TASK, 'smoke', dict(escaped_invalid_utf8=False, control_bytes=0),
            policy=self.fixture['policy'])
        self.assertFalse(result['logs_clean'])
        self.assertTrue(result['raw_log_preserved'])
        self.assertEqual(result['wrapper_warnings'][0]['kind'], 'external_sdk_exception')
        for failure in (sdk.replace('/opt/pika/client.py', '/repo/humanoid/amp/head.py'),
                        'NaN native output\n', 'Traceback (most recent call last):\n'):
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                stage.validate_log(failure+self.fixture['log'], self.fixture['report'],
                    self.fixture['audits'], TASK, 'smoke', dict(escaped_invalid_utf8=False, control_bytes=0),
                    policy=self.fixture['policy'])

    def test_actual_array_history_hidden_and_duplicate_role_tampering_rejected(self):
        from humanoid.scripts.collect_amp_head_cohort import _read_bundle
        for change in ('history', 'hidden', 'seed', 'initial'):
            manifest, arrays = _read_bundle(self.args['train'], torch)
            if change == 'history': arrays['reference_full_obs'][90, 0, 0] += .1
            elif change == 'hidden': arrays['reference_actor_hidden'][90, 0, 0] += .1
            elif change == 'seed': manifest['seed'] = 505
            else: arrays['reference_initial_dof_pos'][0, 0] += .1
            bundle(self.args['train'], manifest, arrays)
            report = copy.deepcopy(self.fixture['report'])
            report['cohort_sha256']['train'] = stage.file_sha(self.args['train'])
            self.refresh_report_binding(report)
            with self.subTest(change=change), self.assertRaises(ValueError): self.audit()
            self.args['train'].write_bytes(self.snapshots['train'])
            self.args['report'].write_bytes(self.snapshots['report'])
            self.args['cloud_log'].write_bytes(self.snapshots['cloud_log'])

    def test_recorder_missing_raw_samples_wrong_mean_dr_reset_and_initial_binding_rejected(self):
        for change in ('raw', 'action', 'dr', 'reset', 'initial', 'mask', 'fifteen_bodies', 'missing_body', 'raw_stats'):
            manifest = copy.deepcopy(self.fixture['recorder'])
            arrays = {key: value.copy() for key, value in self.fixture['policy_arrays'].items()}
            if change == 'raw': arrays['reference_physics_raw_velocity'] = arrays['reference_physics_raw_velocity'][:, :9]
            elif change == 'action': arrays['reference_action'][80, 0, 0] += 1
            elif change == 'dr': manifest['environment']['domain_rand']['randomize_friction'] = True
            elif change == 'reset': manifest['initialization']['cleared_histories']['standing']['obs_history']['nonzero'] = 1
            elif change == 'initial': arrays['reference_initial_dof_vel'][0, 0] += .1
            elif change == 'mask': arrays['reference_valid'][50, 0] = False
            elif change == 'fifteen_bodies': arrays['reference_key_positions_w'] = np.zeros((100, 2, 15, 3), dtype=np.float32)
            elif change == 'missing_body': manifest['body_names'].pop()
            else: arrays['reference_physics_accel_squared'][40, 0, 0] = 9.
            bundle(self.args['policy_rollout'], manifest, arrays)
            reference = copy.deepcopy(self.fixture['report']['policy_probe'])
            reference['sha256'] = stage.file_sha(self.args['policy_rollout'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                stage.audit_policy_rollout(self.args['policy_rollout'], reference, mode='smoke',
                    identity=manifest['identity'], binding=self.args['expected_parent'], commit=COMMIT,
                    fingerprint=FP, artifact={}, archive={}, source_proof=self.fixture['proof'],
                    source_manifest=self.fixture['source_manifest'], exclusions=self.fixture['index'])

    def test_duplicate_json_keys_and_nan_are_not_silently_accepted(self):
        for value in ('{"x":1,"x":2}', '{"value":NaN}', '{"value":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError): stage._json(value)

    def test_production_cli_has_no_parent_override_or_fit_and_formal_holdout_fail_closed(self):
        import ast
        text = (ROOT/'tools/amp/audit_head_stage.py').read_text(encoding='utf-8')
        tree = ast.parse(text)
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)]
        self.assertNotIn('fit_head_smoothing', calls)
        self.assertNotIn('_solve', calls)
        self.assertNotIn("add_argument('--expected-parent'", text)
        report = copy.deepcopy(self.fixture['report']); report['sealed_cohort'] = {'sha256': 'f'*64}
        torch.save(dict(report_json=json.dumps(report)), self.args['report'])
        with self.assertRaisesRegex(ValueError, 'sealed physical-pair independent audit is pending'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
