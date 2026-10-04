"""Synthetic protocol/telemetry tests: no real fit, cloud or simulation."""
import ast
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from humanoid.amp.head_workflow import exclusion_add_cohort
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.scripts.collect_amp_head_cohort import (
    INITIAL_FIELDS, SOURCE_SHA, SOURCE_MODEL_SHA, audit_arrays,
    empty_exclusion_index, fingerprint_cohort, validate_source_runtime)
from humanoid.scripts.record_amp_head_sealed import assemble_sealed_pair, sealed_recorder_contract
from tools.amp import audit_head_stage as stage
from tools.amp import audit_head_sealed as sealed_audit
from tools.amp.audit_physics_diagnostic import fast_geometry
from test_head_sealed_recorder import (
    COMMIT, IMPLEMENTATION, IDENTITY, HEAD_SHA, CANDIDATE_SHA,
    arm_fixture, input_fixture, source_runtime_fixture)
from test_head_workflow import native_sim_fixture
from test_head_stage_audit import synthetic_native_context


ROOT = Path(__file__).resolve().parents[1]
TASK = 'TASK_20990101_001'


def fixture():
    cohorts, artifact = input_fixture()
    source = source_runtime_fixture()
    names = json.loads((ROOT/'configs/amp/f1_100hz.json').read_text(encoding='utf-8'))['joint_names']
    source['environment'].update(init_state=dict(default_joint_angles={k: 0. for k in names}),
                                safety=dict(torque_limit=1.))
    source['runtime'].update(p_gains=[35.]*12, d_gains=[1.5]*12,
        dof_properties=dict(lower=[-1.]*12, upper=[1.]*12, effort=[100.]*12, armature=[.1]*12))
    proof = copy.deepcopy(cohorts['train']['manifest']['source_regression'])
    proof['derived_index']['path'] = '/workspace/resources/amp_inputs/source110_exclusions_v1.npz'
    index = empty_exclusion_index()
    earlier = []
    for role in ('train', 'validation', 'sealed'):
        item = cohorts[role]
        arrays = item['arrays']
        for key in ('root_state', 'foot_state'):
            initial = arrays['reference_initial_'+key]
            initial[..., 3:7] = 0.
            initial[..., 6] = 1.
            arrays['reference_'+key] = np.broadcast_to(initial,
                (len(arrays['reference_time']),)+initial.shape)
        ready, _, history, episodes = audit_arrays(arrays, item['manifest'])
        fingerprints, dedupe = fingerprint_cohort(arrays, item['manifest'], ready, index)
        environment = copy.deepcopy(source['environment'])
        environment['seed'] = item['manifest']['seed']
        environment['env'].update(num_envs=8, episode_length_s=20.1)
        runtime = copy.deepcopy(source['runtime'])
        runtime['physics_sim_parameters'] = native_sim_fixture(environment['sim'])
        initialization = dict(protocol='original_reference_reset_then_native_10ms_zero_action_warmup',
            reset_inputs=dict(root_states=np.zeros((8, 13)).tolist(),
                **{k: np.zeros((8, 12)).tolist() for k in
                   ('dof_pos', 'dof_vel', 'actions', 'last_actions', 'last_last_actions')},
                commands=np.zeros((8, 4)).tolist(), **{k: [0]*8 for k in
                   ('gait_start', 'episode_length_buf', 'phase_length_buf', 'rsi_indices')},
                obs_history_sha256='b'*64, critic_history_sha256='c'*64),
            cleared_histories={k: dict(elements=100, nonzero=0, nonfinite=0, negative_zero=0)
                               for k in ('obs_history', 'critic_history')},
            all12_initial_fields=list(INITIAL_FIELDS), actual_initial_full_obs=True)
        item['manifest'].update(source_regression=copy.deepcopy(proof), previous_cohorts=copy.deepcopy(earlier),
            fingerprints=fingerprints, deduplication=dedupe, episodes=episodes, history_proof=history,
            environment=environment, runtime=runtime, initialization=initialization,
            modes=['reference'],
            source_runtime_proof=validate_source_runtime(environment, runtime, source, item['manifest']),
            body_names=list(sealed_audit.SOURCE_BODIES), dof_names=names,
            foot_names=sealed_audit.RAW_BODIES[1:], urdf_lf_sha256='6'*64)
        frames = len(arrays['reference_time'])
        for key, shape in dict(physics_raw_velocity=(10, 12), physics_raw_torque=(10, 12),
            physics_raw_body_force=(10, 3, 3), physics_raw_body_state=(10, 3, 13),
            physics_interval_initial_velocity=(12,), physics_interval_initial_torque=(12,)).items():
            arrays['reference_'+key] = np.zeros((frames,)+shape, dtype=np.float32)
        arrays['reference_physics_raw_velocity'][:] = arrays['reference_dof_vel'][:, 0, None]
        arrays['reference_physics_raw_torque'][:] = arrays['reference_torque'][:, 0, None]
        arrays['reference_physics_interval_initial_velocity'][:] = arrays['reference_dof_vel'][:, 0]
        arrays['reference_physics_interval_initial_torque'][:] = arrays['reference_torque'][:, 0]
        arrays['reference_physics_raw_body_state'][:, :, 0] = arrays['reference_root_state'][:, 0, None]
        arrays['reference_physics_raw_body_state'][:, :, 1:] = arrays['reference_foot_state'][:, 0, None]
        arrays['reference_physics_valid'][3:] = True
        item['manifest']['inference_capture'] = dict(actor_module='actor.6', hidden_dim=128,
            action_dim=12, actual_forwards=frames, extra_forward=False, actual_output_exact=True,
            model_state_unchanged=True, numerical_context=synthetic_native_context())
        exclusion_add_cohort(index, dict(fingerprints=fingerprints))
        earlier.append(dict(sha256=item['sha256'], split=role, seed=item['manifest']['seed'],
            num_envs=8, duration_s=20., episode_ids=item['manifest']['episode_ids']))
    from humanoid.scripts.record_amp_head_sealed import audit_sealed_inputs
    inputs = audit_sealed_inputs(cohorts, artifact=artifact, identity=IDENTITY, code_commit=COMMIT,
        implementation_fingerprint=IMPLEMENTATION, exclusions=empty_exclusion_index(),
        source_proof=proof, source_manifest=source)
    archive = dict(candidate_model_state_sha256=CANDIDATE_SHA, fixture='synthetic-only')
    arms = {}
    for arm in ('source', 'candidate'):
        manifest, arrays = arm_fixture(arm, inputs, cohorts['sealed']['arrays'], ticks=620)
        environment = copy.deepcopy(source['environment'])
        environment['seed'] = 706
        environment['env'].update(num_envs=8, episode_length_s=60.1)
        runtime = copy.deepcopy(source['runtime'])
        runtime['physics_sim_parameters'] = native_sim_fixture(environment['sim'])
        manifest.update(environment=environment, runtime=runtime,
            source_runtime_proof=validate_source_runtime(environment, runtime, source, manifest),
            head_archive_audit=archive, body_names=list(sealed_audit.SOURCE_BODIES), dof_names=names,
            foot_names=sealed_audit.RAW_BODIES[1:], urdf_lf_sha256='6'*64,
            no_dr=assert_no_domain_randomization(environment), process_id=101 if arm == 'source' else 102,
            initialization={key: copy.deepcopy(cohorts['sealed']['manifest']['initialization'][key])
                for key in ('protocol', 'reset_inputs', 'cleared_histories')},
            substep_telemetry=dict(physics_hz=1000, control_hz=100, samples_per_control=10,
                raw_env=0, raw_body_names=list(sealed_audit.RAW_BODIES), reward_used=False,
                reset_guard='native physics_valid excludes first three reset intervals',
                acceleration='native1ms dq/.001 mean squared and peak absolute',
                torque_delta='actual1ms torque difference mean squared; not derivative',
                foot_force='max positive NET footFz, not contact-pair identity'))
        arrays['reference_physics_valid'][3:] = True
        arrays['reference_physics_raw_velocity'][:] = arrays['reference_dof_vel'][:, 0, None]
        arrays['reference_physics_raw_torque'][:] = arrays['reference_torque'][:, 0, None]
        arrays['reference_physics_interval_initial_velocity'][:] = arrays['reference_dof_vel'][:, 0]
        arrays['reference_physics_interval_initial_torque'][:] = arrays['reference_torque'][:, 0]
        arrays['reference_physics_raw_body_state'][:, :, 0] = arrays['reference_root_state'][:, 0, None]
        arrays['reference_physics_raw_body_state'][:, :, 1:] = arrays['reference_foot_state'][:, 0, None]
        arms[arm] = dict(manifest=manifest, arrays=arrays)
    contract = sealed_recorder_contract('formal', 706, 8, 60.)
    manifest, arrays = assemble_sealed_pair(arms['source']['manifest'], arms['source']['arrays'],
        arms['candidate']['manifest'], arms['candidate']['arrays'], contract=contract, identity=IDENTITY,
        code_commit=COMMIT, implementation_fingerprint=IMPLEMENTATION, head_sha256=HEAD_SHA,
        candidate_model_sha256=CANDIDATE_SHA, inputs=inputs, sealed_arrays=cohorts['sealed']['arrays'],
        source_runtime_manifest=source, arm_file_shas=dict(source='a'*64, candidate='b'*64))
    reference6 = dict(artifact_name='model_7100006.pt', sha256=cohorts['sealed']['sha256'],
        seed=706, num_envs=8, duration_s=20, split='sealed', fit_eligible=False,
        episodes=inputs['audits']['sealed']['episodes'], deduplication_passed=True,
        holdout_not_used_for_fitting=True)
    reference7 = dict(artifact_name='model_7100007.pt', sha256='d'*64,
        type=sealed_audit.SCHEMA, mode='formal', seed=706, num_envs=8, duration_s=60,
        head_artifact_sha256=HEAD_SHA, source_checkpoint_sha256=SOURCE_SHA,
        sealed_cohort_sha256=cohorts['sealed']['sha256'], effectiveness_verified=False, dr_unlocked=False)
    vertices = np.array([[x, y, z] for x in (-.05, .05) for y in (-.03, .03)
                         for z in (-.01, .01)])
    geometry = fast_geometry({key: vertices for key in sealed_audit.RAW_BODIES[1:]})
    options = dict(cohorts=cohorts, artifact=artifact, archive=archive, identity=IDENTITY,
        binding=dict(file_sha256=SOURCE_SHA, model_state_sha256=SOURCE_MODEL_SHA), commit=COMMIT,
        fingerprint=IMPLEMENTATION, source_proof=proof, source_manifest=source,
        exclusions=empty_exclusion_index(), sealed_reference=reference6, pair_reference=reference7,
        pair_sha='d'*64, head_sha=HEAD_SHA, geometry=geometry)
    return manifest, arrays, options


class SealedArtifactAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.arrays, cls.options = fixture()
        cls.result = sealed_audit.audit_sealed_artifacts(cls.manifest, cls.arrays, **cls.options)

    def test_all8_failures_terminal_prefixes_and_actual1ms_verified_not_effectiveness(self):
        result = self.result
        self.assertTrue(result['all_eight_retained'])
        self.assertFalse(result['effectiveness_verified'])
        self.assertFalse(result['dr_unlocked'])
        self.assertEqual(set(result['source_cohort_numerical_contexts']),
                         {'train', 'validation', 'sealed'})
        for context in result['source_cohort_numerical_contexts'].values():
            self.assertEqual(context, synthetic_native_context())
        self.assertIn('standalone helper does not authenticate hardware', result['source_context_scope'])
        for arm in ('source', 'candidate'):
            rows = result['arms'][arm]['actual_arm']['episodes']['reference']
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(v['failed'] for v in rows))
            self.assertEqual(rows[4]['valid_ticks'], 61)
            self.assertEqual(rows[4]['terminal_tick'], 60)
            self.assertGreater(result['arms'][arm]['raw_telemetry']['reference']['samples'], 0)
            self.assertFalse(result['arms'][arm]['temporary_arm_bytes_independently_received'])
        self.assertEqual(len(result['metrics']['paired']['rows']), 8)
        self.assertEqual(result['metrics']['paired']['rows'][4]['common_valid_ticks'], 0)
        self.assertIsNone(result['metrics']['paired']['rows'][4]['source']['near48s'])

    def test_fixed_windows_missing_window_not_zero_whole60_coverage_truthful(self):
        rows = self.result['metrics']['source']
        self.assertFalse(rows[0]['complete60s'])
        self.assertFalse(rows[0]['spans']['whole60s']['quality']['complete_window'])
        self.assertTrue(rows[0]['spans']['near5s']['quality']['complete_window'])
        self.assertIsNone(rows[0]['spans']['near29to30s']['quality'])
        self.assertIsNone(rows[0]['spans']['near29to30s']['physics'])
        self.assertEqual(rows[0]['spans']['near29to30s']['coverage']['observed_ticks'], 0)
        paired = self.result['metrics']['paired']
        self.assertFalse(paired['rows'][0]['source']['near5s']['quality']['complete_window'])
        self.assertNotIn(4, paired['means']['near5s']['vx_mean_m_s']['contributing_envs'])
        self.assertIsNone(paired['means']['near48s']['substep_accel_rms']['relative_change'])

    def test_path_difference_only_registered_index_suffix_after_independent_sha(self):
        local = copy.deepcopy(self.options['source_proof'])
        local['derived_index']['path'] = 'F:\\repo\\resources\\amp_inputs\\source110_exclusions_v1.npz'
        sealed_audit.source_proof_equal(self.options['source_proof'], local)
        for mutation in ('path', 'sha', 'source'):
            wrong = copy.deepcopy(local)
            if mutation == 'path': wrong['derived_index']['path'] = '/tmp/other.npz'
            elif mutation == 'sha': wrong['derived_index']['sha256'] = 'e'*64
            else: wrong['sha256'] = 'e'*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                sealed_audit.source_proof_equal(self.options['source_proof'], wrong)

    def test_actual_shapes_dtype_full_initial_and_extra_arrays_rejected(self):
        contract = sealed_recorder_contract('formal', 706, 8, 60.)
        original = {k[len('source_'):]: v for k, v in self.arrays.items() if k.startswith('source_')}
        for mutation in ('raw9', 'fifteen', 'intmask', 'full3101', 'extra'):
            arrays = dict(original)
            if mutation == 'raw9': arrays['reference_physics_raw_velocity'] = arrays['reference_physics_raw_velocity'][:, :9]
            elif mutation == 'fifteen': arrays['reference_key_positions_w'] = np.zeros((620, 8, 15, 3), np.float32)
            elif mutation == 'intmask': arrays['reference_valid'] = arrays['reference_valid'].astype(np.int64)
            elif mutation == 'full3101': arrays['reference_initial_full_obs'] = arrays['reference_initial_full_obs'][:, :-1]
            else: arrays['reference_fake_state'] = np.zeros(1)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                sealed_audit.validate_arm_arrays(arrays, contract, 4)

    def test_initial_state_input_reset_archive_or_candidate_substitution_rejected(self):
        for mutation in ('first_input', 'initial_q', 'reset', 'head_archive', 'head_sha', 'native_reset', 'drop_arm'):
            manifest = copy.deepcopy(self.manifest)
            arrays = dict(self.arrays)
            if mutation in ('first_input', 'initial_q', 'native_reset'):
                key = {'first_input': 'candidate_reference_initial_full_obs',
                       'initial_q': 'candidate_reference_initial_dof_pos',
                       'native_reset': 'candidate_reference_native_episode'}[mutation]
                arrays[key] = arrays[key].copy()
                arrays[key].reshape(-1)[0] += 1
            elif mutation == 'reset': manifest['arms']['candidate']['manifest']['initialization']['reset_inputs']['gait_start'][0] = 1
            elif mutation == 'head_archive': manifest['arms']['candidate']['manifest']['head_archive_audit']['fixture'] = 'wrong'
            elif mutation == 'head_sha': manifest['head_artifact_sha256'] = 'e'*64
            else: manifest['arms'].pop('source')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                sealed_audit.audit_sealed_artifacts(manifest, arrays, **self.options)

    def test_raw_and_aggregate_telemetry_tampering_rejected(self):
        for key in ('source_reference_physics_raw_velocity', 'candidate_reference_physics_accel_squared'):
            arrays = dict(self.arrays)
            arrays[key] = arrays[key].copy()
            arrays[key][40].reshape(-1)[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                sealed_audit.audit_sealed_artifacts(self.manifest, arrays, **self.options)

    def test_source_cohort_native_context_required_typed_and_all_splits_exact(self):
        cases = [('missing', None), ('device_type', 'cpu'), ('cudnn_enabled', 1),
            ('cuda_matmul_allow_tf32', True), ('torch_cuda_version', '12.2'),
            ('capture_extra', True)]
        for field, value in cases:
            options = dict(self.options)
            options['cohorts'] = {role: dict(item) for role, item in options['cohorts'].items()}
            item = options['cohorts']['sealed']
            item['manifest'] = copy.deepcopy(item['manifest'])
            capture = item['manifest']['inference_capture']
            if field == 'missing': capture.pop('numerical_context')
            elif field == 'capture_extra': capture['unregistered'] = value
            else: capture['numerical_context'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                sealed_audit.audit_sealed_artifacts(self.manifest, self.arrays, **options)

    def test_duplicate_files_metadata_boolean_and_dr_cannot_bypass(self):
        for mutation in ('head_smoke', 'same_cohort', 'pair_effect', 'dr', 'pid_bool', 'same_arm_sha'):
            manifest, options = copy.deepcopy(self.manifest), dict(self.options)
            if mutation == 'head_smoke':
                options['artifact'] = copy.deepcopy(options['artifact'])
                options['artifact']['provenance']['mode'] = 'smoke'
            elif mutation == 'same_cohort':
                options['cohorts'] = {k: dict(v) for k, v in options['cohorts'].items()}
                options['cohorts']['sealed']['sha256'] = options['cohorts']['train']['sha256']
            elif mutation == 'pair_effect': manifest['effectiveness_verified'] = 0
            elif mutation == 'dr': manifest['arms']['source']['manifest']['environment']['domain_rand']['randomize_friction'] = True
            elif mutation == 'pid_bool': manifest['arms']['source']['manifest']['process_id'] = True
            else: manifest['arms']['candidate']['arm_file_sha256'] = manifest['arms']['source']['arm_file_sha256']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                sealed_audit.audit_sealed_artifacts(manifest, self.arrays, **options)

    def test_legacy_sealed_seed_and_pair_evaluation_protocol_rejected(self):
        for field, value in (('seed', 705),
                             ('evaluation_protocol', 'new_reference8_seed705_source_head_paired60')):
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                sealed_audit.audit_sealed_artifacts(manifest, self.arrays, **self.options)
        options = dict(self.options)
        options['sealed_reference'] = dict(options['sealed_reference'], seed=705)
        with self.assertRaises(ValueError):
            sealed_audit.audit_sealed_artifacts(self.manifest, self.arrays, **options)

    def test_seven_file_log_actual_chronology_all_receipts(self):
        report, audits, policy, text = self.log_fixture()
        result = stage.validate_log(text, report, audits, TASK, 'formal',
            dict(escaped_invalid_utf8=False, control_bytes=0), policy=policy, sealed=self.result)
        self.assertTrue(result['native_sealed_pair_verified'])
        self.assertEqual(result['cohort_completions'], 3)
        self.assertEqual(len(result['upload_receipts']), 7)
        self.assertTrue(result['logs_clean'])

    def log_fixture(self):
        from test_head_stage_audit import hardware_fixture
        report = dict(start=dict(fixture='SYNTHETIC_LOG_ONLY',
                                native_hardware=hardware_fixture()))
        audits = self.result['cohort_inputs']['audits']
        policy = dict(start=dict(fixture='synthetic_original32_start'), complete=dict(fixture='synthetic_original32_end'))
        text = stage.HARDWARE+json.dumps(report['start']['native_hardware'])+'\n'
        text += stage.START+json.dumps(report['start'])+'\n'
        def cohort(role):
            return stage.COHORT+json.dumps(dict(split=role, mode='formal',
                fit_eligible=audits[role]['fit_eligible'], episodes=audits[role]['episodes'],
                deduplication_passed=audits[role]['deduplication']['passed']))+'\n'
        text += cohort('train')+cohort('validation')
        text += stage.POLICY_START+json.dumps(policy['start'])+'\n'+stage.POLICY_COMPLETE+json.dumps(policy['complete'])+'\n'
        text += cohort('sealed')
        for arm in ('source', 'candidate'):
            text += sealed_audit.ARM_START+json.dumps(self.result['log_records'][arm]['start'])+'\n'
            text += sealed_audit.ARM_COMPLETE+json.dumps(self.result['log_records'][arm]['complete'])+'\n'
        text += sealed_audit.PAIR_COMPLETE+json.dumps(self.result['log_records']['pair_complete'])+'\n'
        text += stage.COMPLETE+json.dumps(report)+'\n'
        text += ''.join('[SDK][INFO] 2099-01-01 00:00:00 PT file (model_710000%d.pt) uploaded successfully\n'
                        % i for i in range(1, 8))
        text += '【SDK】训练进程状态码：0\n[SDK][INFO] 2099-01-01 00:00:00 Task('+TASK+') status updated to: Completed , ret:True\n'
        return report, audits, policy, text

    def test_log_missing_wrong_order_duplicate_pair_and_seventh_receipt_rejected(self):
        report, audits, policy, log = self.log_fixture()
        start = sealed_audit.ARM_START+json.dumps(self.result['log_records']['source']['start'])+'\n'
        mutations = [log.replace(sealed_audit.PAIR_COMPLETE, '[wrong-pair] '),
            log+sealed_audit.PAIR_COMPLETE+json.dumps(self.result['log_records']['pair_complete'])+'\n',
            log.replace('model_7100007.pt) uploaded successfully', 'model_7100009.pt) uploaded successfully'),
            log.replace(start, '').replace(stage.POLICY_START, start+stage.POLICY_START),
            log.replace('【SDK】训练进程状态码：0', '【SDK】训练进程状态码：1')]
        for text in mutations:
            with self.subTest(tail=text[-120:]), self.assertRaises(ValueError):
                stage.validate_log(text, report, audits, TASK, 'formal',
                    dict(escaped_invalid_utf8=False, control_bytes=0), policy=policy, sealed=self.result)

    def test_readonly_no_fit_native_or_platform_calls_and_facts_hash_no_selfref(self):
        for file in ('audit_head_stage.py', 'audit_head_sealed.py'):
            tree = ast.parse((ROOT/'tools/amp'/file).read_text(encoding='utf-8'))
            calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertNotIn('fit_head_smoothing', calls)
            self.assertNotIn('_solve', calls)
            self.assertNotIn('make_env', calls)
        digest = stage.canonical_sha(dict(effectiveness_verified=False, sealed=self.result))
        self.assertEqual(len(digest), 64)
        self.assertNotIn('smoke_certificate', self.result)


if __name__ == '__main__':
    unittest.main()
