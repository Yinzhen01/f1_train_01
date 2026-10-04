"""Synthetic sealed protocol tests, NOT native collection/fit/cloud evidence."""
import argparse
import ast
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from humanoid.amp.head_workflow import exclusion_add_cohort
from humanoid.scripts.collect_amp_head_cohort import (
    INITIAL_FIELDS, PHYSICS_FIELDS, SEEDS, SOURCE_SHA, SOURCE_MODEL_SHA,
    SOURCE_ENDPOINT_SHA, audit_arrays, cohort_contract, empty_exclusion_index,
    fingerprint_cohort, numerical_fingerprint, validate_source_runtime, _read_bundle)
from humanoid.scripts.record_amp_head_policy import audit_rollout_arrays
from humanoid.scripts.record_amp_head_sealed import (
    SCHEMA, ARM_SCHEMA, ARMS, arm_command, assemble_sealed_pair,
    audit_sealed_arm, audit_sealed_inputs, initial_sealed_matches,
    sealed_recorder_contract, _write_bundle)
from test_head_workflow import source_fixture


ROOT = Path(__file__).resolve().parents[1]
COMMIT, IMPLEMENTATION = '1'*40, '2'*64
IDENTITY = {'fixture': 'synthetic_sealed_recorder_not_a_real_source'}
HEAD_SHA, CANDIDATE_SHA = '8'*64, '9'*64
SOURCE_PROOF = dict(sha256=SOURCE_ENDPOINT_SHA,
                    derived_index=dict(sha256='7'*64, schema='synthetic_test_only'))


def synthetic_cohort(role, offset, index, earlier):
    """Broadcast constant histories save RAM; never represent native source data."""
    contract = cohort_contract(role, 'formal', SEEDS[role], 8, 20.)
    t, n = (100 if role == 'sealed' else 2000), 8
    widths = dict(root_state=(13,), dof_pos=(12,), dof_vel=(12,), action=(12,),
        torque=(12,), base_lin_vel=(3,), base_ang_vel=(3,), foot_state=(2, 13),
        foot_force=(2, 3), key_positions_w=(4, 3))
    arrays = {}
    for key, width in widths.items():
        initial = np.zeros((n,)+width, dtype=np.float32)
        if key != 'action':
            initial.reshape(n, -1)[:, 0] = offset+np.arange(n, dtype=np.float32)/8
        arrays['reference_initial_'+key] = initial
        arrays['reference_'+key] = np.broadcast_to(initial, (t,)+initial.shape)
    single = np.arange(47, dtype=np.float32)[None, :]/100+offset+np.arange(n, dtype=np.float32)[:, None]/8
    initial_full = np.tile(single[:, None, :], (1, 66, 1)).reshape(n, 3102)
    initial_native = np.arange(n, dtype=np.int64)+2
    valid = np.ones((t, n), dtype=bool)
    failure = np.zeros((t, n), dtype=bool)
    done = failure.copy()
    if role == 'sealed':
        failure[-1] = True
        done[-1] = True  # all fail at1s: still retain all8 held-out initials.
    arrays.update(reference_valid=valid, reference_failure=failure, reference_done=done,
        reference_initial_valid=np.ones(n, dtype=bool),
        reference_initial_failure=np.zeros(n, dtype=bool),
        reference_initial_full_obs=initial_full,
        reference_initial_native_episode=initial_native,
        reference_native_episode=np.broadcast_to(initial_native, (t, n)),
        reference_full_obs=np.broadcast_to(initial_full, (t, n, 3102)),
        reference_single_obs=np.broadcast_to(single, (t, n, 47)),
        reference_actor_hidden=np.zeros((t, n, 128), dtype=np.float32),
        reference_raw_mu=np.zeros((t, n, 12), dtype=np.float32),
        reference_command=np.broadcast_to(np.array([.45, 0., 0., 0.], dtype=np.float32), (t, n, 4)),
        reference_tick=np.broadcast_to(np.arange(t, dtype=np.int64)[:, None], (t, n)),
        reference_episode_id=np.zeros((t, n), dtype=np.int64),
        reference_time=(np.arange(t)+1)*.01)
    for key in PHYSICS_FIELDS:
        shape = (t, n) if key == 'physics_valid' else (t, n, 2 if key == 'physics_foot_force_peak' else 12)
        arrays['reference_'+key] = np.zeros(shape, dtype=bool if key == 'physics_valid' else np.float32)
    ready, real, history, episodes = audit_arrays(arrays, contract)
    fingerprints, dedupe = fingerprint_cohort(arrays, contract, ready, index)
    cfg, runtime = copy.deepcopy(source_fixture()['environment']), copy.deepcopy(source_fixture()['runtime'])
    cfg['seed'] = SEEDS[role]
    cfg['env'].update(num_envs=8, episode_length_s=20.1)
    cfg['normalization'] = dict(clip_actions=100.)
    source = source_runtime_fixture()
    runtime['physics_sim_parameters'] = copy.deepcopy(source['environment']['sim'])
    manifest = dict(contract, source_checkpoint_sha256=SOURCE_SHA,
        source_model_state_sha256=SOURCE_MODEL_SHA, identity=IDENTITY,
        source_completed_updates=2500, code_commit=COMMIT,
        implementation_fingerprint=IMPLEMENTATION, policy_deterministic=True,
        parent_frozen=True, no_training=True, effectiveness_verified=False,
        dr_unlocked=False, source_regression=copy.deepcopy(SOURCE_PROOF),
        previous_cohorts=copy.deepcopy(earlier), environment=cfg, runtime=runtime,
        source_runtime_proof=validate_source_runtime(cfg, runtime, source, contract),
        fingerprints=fingerprints, history_proof=history, episodes=episodes,
        deduplication=dedupe, fit_eligible=role != 'sealed')
    proof = dict(sha256={'train': '3'*64, 'validation': '4'*64, 'sealed': '5'*64}[role],
        split=role, seed=SEEDS[role], num_envs=8, duration_s=20., episode_ids=contract['episode_ids'])
    computed = dict(fingerprints=fingerprints)
    exclusion_add_cohort(index, computed)
    return dict(manifest=manifest, arrays=arrays, sha256=proof['sha256']), proof


def source_runtime_fixture():
    result = source_fixture()
    result['environment']['normalization'] = dict(clip_actions=100.)
    return result


def input_fixture():
    index, earlier, cohorts = empty_exclusion_index(), [], {}
    for role, offset in (('train', 10.), ('validation', 20.), ('sealed', 30.)):
        cohorts[role], proof = synthetic_cohort(role, offset, index, earlier)
        earlier.append(proof)
    artifact = dict(artifact_kind='actor_head_offline_v1', amp_identity=IDENTITY,
        provenance=dict(mode='formal', offline_admitted=True, code_commit=COMMIT,
            implementation_fingerprint=IMPLEMENTATION,
            cohorts={role: {key: value for key, value in earlier[i].items() if key != 'split'}
                     for i, role in enumerate(('train', 'validation'))}),
        effectiveness_verified=False, dr_unlocked=False)
    return cohorts, artifact


def input_audit(cohorts, artifact, index=None):
    return audit_sealed_inputs(cohorts, artifact=artifact, identity=IDENTITY,
        code_commit=COMMIT, implementation_fingerprint=IMPLEMENTATION,
        exclusions=empty_exclusion_index() if index is None else index,
        source_proof=SOURCE_PROOF, source_manifest=source_runtime_fixture())


def arm_fixture(arm, inputs, sealed_arrays, ticks=100):
    contract = sealed_recorder_contract('formal', 705, 8, 60.)
    arrays = {}
    for key in INITIAL_FIELDS:
        initial = sealed_arrays['reference_initial_'+key].copy()
        arrays['reference_initial_'+key] = initial
        arrays['reference_'+key] = np.broadcast_to(initial, (ticks,)+initial.shape).copy()
    arrays['reference_valid'][:] = True
    arrays['reference_failure'][:] = False
    arrays['reference_done'] = np.zeros((ticks, 8), dtype=bool)
    arrays['reference_failure'][-1] = True
    arrays['reference_done'][-1] = True  # early natural all-failed termination, not a crop.
    # Keep a failed env4's last native interval and every invalid tail row.
    if ticks > 61:
        arrays['reference_valid'][61:, 4] = False
        arrays['reference_failure'][60, 4] = True
        arrays['reference_failure'][-1, 4] = False
        arrays['reference_done'][60, 4] = True
        arrays['reference_done'][-1, 4] = False
    arrays['reference_time'] = (np.arange(ticks)+1)*.01
    arrays['reference_policy_raw_mu'] = arrays['reference_action'].copy()
    arrays['reference_initial_full_obs'] = sealed_arrays['reference_initial_full_obs'].copy()
    native = sealed_arrays['reference_initial_native_episode'].copy()
    arrays['reference_initial_native_episode'] = native
    arrays['reference_native_episode'] = np.broadcast_to(native, (ticks, 8)).copy()
    for key in PHYSICS_FIELDS:
        shape = (ticks, 8) if key == 'physics_valid' else (ticks, 8, 2 if key == 'physics_foot_force_peak' else 12)
        arrays['reference_'+key] = np.zeros(shape, dtype=bool if key == 'physics_valid' else np.float32)
    for key, shape in dict(physics_raw_velocity=(10, 12), physics_raw_torque=(10, 12),
            physics_raw_body_force=(10, 3, 3), physics_raw_body_state=(10, 3, 13),
            physics_interval_initial_velocity=(12,), physics_interval_initial_torque=(12,)).items():
        arrays['reference_'+key] = np.zeros((ticks,)+shape, dtype=np.float32)
    source = source_runtime_fixture()
    cfg, runtime = copy.deepcopy(source['environment']), copy.deepcopy(source['runtime'])
    cfg['seed'], cfg['env']['num_envs'] = 705, 8
    runtime['physics_sim_parameters'] = copy.deepcopy(source['environment']['sim'])
    proof = validate_source_runtime(cfg, runtime, source, contract)
    manifest = dict(contract, type=ARM_SCHEMA, arm=arm, code_commit=COMMIT,
        implementation_fingerprint=IMPLEMENTATION, source_identity=IDENTITY,
        parent_checkpoint_sha256=SOURCE_SHA, parent_model_state_sha256=SOURCE_MODEL_SHA,
        head_artifact_sha256=HEAD_SHA,
        policy_model_state_sha256=SOURCE_MODEL_SHA if arm == 'source' else CANDIDATE_SHA,
        parent_frozen=True, policy_deterministic=True, cohorts=inputs['cohorts'],
        source_regression=inputs['source_regression'], environment=cfg, runtime=runtime,
        source_runtime_proof=proof,
        initial_sealed_matches=initial_sealed_matches(arrays, sealed_arrays, contract['episode_ids']),
        episodes=audit_rollout_arrays(arrays, contract),
        inference_capture=dict(actual_forwards=ticks, extra_forward=False, model_state_unchanged=True))
    return manifest, arrays


def arm_audit(manifest, arrays, inputs, sealed_arrays, arm):
    return audit_sealed_arm(manifest, arrays, arm=arm,
        contract=sealed_recorder_contract('formal', 705, 8, 60.), identity=IDENTITY,
        code_commit=COMMIT, implementation_fingerprint=IMPLEMENTATION,
        head_sha256=HEAD_SHA, candidate_model_sha256=CANDIDATE_SHA,
        inputs=inputs, sealed_arrays=sealed_arrays, source_manifest=source_runtime_fixture())


class SealedRecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohorts, cls.artifact = input_fixture()
        cls.inputs = input_audit(cls.cohorts, cls.artifact)
        cls.sealed = cls.cohorts['sealed']['arrays']

    def test_exact_formal_only_budget_without_ppo_counters(self):
        contract = sealed_recorder_contract('formal', 705, 8, 60.)
        self.assertEqual(contract['type'], SCHEMA)
        self.assertEqual(contract['modes'], ['reference'])
        self.assertEqual(len(contract['episode_ids']), 8)
        self.assertFalse(contract['effectiveness_verified'])
        self.assertNotIn('completed_updates', contract)
        self.assertNotIn('iter', contract)
        for args in (('smoke', 705, 8, 60.), ('formal', 105, 8, 60.),
                     ('formal', 705, 16, 60.), ('formal', 705, 8, 20.),
                     ('formal', True, 8, 60.), ('formal', 705, True, 60.)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                sealed_recorder_contract(*args)

    def test_all_actual_priors_recomputed_failed_sealed_not_fitted_or_filtered(self):
        self.assertEqual(set(self.inputs['cohorts']), {'train', 'validation', 'sealed'})
        self.assertFalse(self.inputs['sealed_fit_eligible'])
        self.assertTrue(self.inputs['all_actual_arrays_checked'])
        self.assertTrue(self.inputs['no_survivor_selection'])
        rows = self.inputs['audits']['sealed']['episodes']
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row['failed'] and not row['complete'] for row in rows))
        self.assertTrue(all(row['observed_ticks'] == 100 for row in rows))
        for role in ('train', 'validation'):
            self.assertTrue(self.inputs['audits'][role]['fit_eligible'])
        # Each later split really compares against actual earlier ready inputs.
        compared = self.inputs['audits']['sealed']['deduplication']['compared_counts']
        self.assertEqual(compared, dict(initial_state=16, initial_full_observation=16,
                                       ready_full_history=16))

    def test_head_actual_file_binding_role_and_missing_cohort_rejected(self):
        for kind in ('sha', 'swap', 'missing', 'head_commit', 'head_fp', 'head_smoke', 'previous'):
            artifact = copy.deepcopy(self.artifact)
            cohorts = {role: dict(item, manifest=copy.deepcopy(item['manifest']))
                       for role, item in self.cohorts.items()}
            if kind == 'sha': cohorts['train']['sha256'] = 'a'*64
            elif kind == 'swap': cohorts['train'], cohorts['validation'] = cohorts['validation'], cohorts['train']
            elif kind == 'missing': cohorts.pop('validation')
            elif kind == 'head_commit': artifact['provenance']['code_commit'] = 'a'*40
            elif kind == 'head_fp': artifact['provenance']['implementation_fingerprint'] = 'b'*64
            elif kind == 'head_smoke': artifact['provenance']['mode'] = 'smoke'
            else: cohorts['sealed']['manifest']['previous_cohorts'].pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError): input_audit(cohorts, artifact)

    def test_old32_and_actual_prior_array_overlap_cannot_be_hidden_by_new_seed(self):
        index = empty_exclusion_index()
        actual = self.cohorts['train']['arrays']
        digest = numerical_fingerprint({key: actual['reference_initial_'+key][3] for key in INITIAL_FIELDS})
        index['initial_state'][digest] = ['source110/reference/env-3/episode-0']
        with self.assertRaises(ValueError): input_audit(self.cohorts, self.artifact, index)
        # Tamper sealed actual fields WITHOUT tampering its old claimed proof.
        cohorts = {role: dict(item) for role, item in self.cohorts.items()}
        cohorts['sealed']['arrays'] = dict(self.sealed)
        for key in INITIAL_FIELDS:
            value = self.sealed['reference_initial_'+key].copy()
            value[0] = self.cohorts['validation']['arrays']['reference_initial_'+key][2]
            cohorts['sealed']['arrays']['reference_initial_'+key] = value
        with self.assertRaises(ValueError): input_audit(cohorts, self.artifact)

    def test_all12_initial_fields_and_complete_first_input_exact_all8(self):
        _, arrays = arm_fixture('source', self.inputs, self.sealed)
        expected = sealed_recorder_contract('formal', 705, 8, 60.)['episode_ids']
        proof = initial_sealed_matches(arrays, self.sealed, expected)
        self.assertEqual(len(proof['rows']), 8)
        self.assertTrue(proof['all_exact'])
        for key in INITIAL_FIELDS+('full_obs',):
            value = dict(arrays)
            name = 'reference_initial_'+key
            value[name] = arrays[name].copy()
            flat = value[name].reshape(8, -1)
            if flat.dtype == np.bool_: flat[7, 0] = not flat[7, 0]
            else: flat[7, 0] = np.nextafter(flat[7, 0], np.float32(np.inf))
            with self.subTest(key=key), self.assertRaises(ValueError):
                initial_sealed_matches(value, self.sealed, expected)
        value = dict(arrays)
        value['reference_initial_full_obs'] = value['reference_initial_full_obs'].astype(np.float64)
        with self.assertRaises(ValueError): initial_sealed_matches(value, self.sealed, expected)
        with self.assertRaises(ValueError): initial_sealed_matches(arrays, self.sealed, list(reversed(expected)))

    def test_signed_zero_equivalence_does_not_mutate_input_or_round_nonzero(self):
        _, arrays = arm_fixture('source', self.inputs, self.sealed)
        arrays['reference_initial_action'][:] = -0.
        before = arrays['reference_initial_action'].tobytes()
        proof = initial_sealed_matches(arrays, self.sealed,
            sealed_recorder_contract('formal', 705, 8, 60.)['episode_ids'])
        self.assertTrue(proof['all_exact'])
        self.assertEqual(before, arrays['reference_initial_action'].tobytes())

    def test_actual_arm_1khz_fail_prefixes_and_no_native_restart(self):
        manifest, arrays = arm_fixture('candidate', self.inputs, self.sealed)
        result = arm_audit(manifest, arrays, self.inputs, self.sealed, 'candidate')
        failed = result['episodes']['reference'][4]
        self.assertEqual(failed['valid_ticks'], 61)
        self.assertTrue(failed['failed'])
        self.assertFalse(failed['complete'])
        self.assertEqual(arrays['reference_physics_raw_torque'].shape, (100, 10, 12))
        for kind in ('reset', 'raw_missing', 'raw_short', 'raw_nan', 'action', 'forward',
                     'ppo', 'dr', 'pd', 'crop', 'lost_terminal', 'state_shape'):
            modified_manifest = copy.deepcopy(manifest)
            value = {key: item.copy() for key, item in arrays.items()}
            if kind == 'reset': value['reference_native_episode'][10, 1] += 1
            elif kind == 'raw_missing': value.pop('reference_physics_raw_torque')
            elif kind == 'raw_short': value['reference_physics_raw_torque'] = value['reference_physics_raw_torque'][:, :9]
            elif kind == 'raw_nan': value['reference_physics_raw_torque'][1, 1, 1] = np.nan
            elif kind == 'action': value['reference_action'][1, 1, 1] = .1
            elif kind == 'forward': modified_manifest['inference_capture']['actual_forwards'] += 1
            elif kind == 'ppo': modified_manifest['completed_updates'] = 2500
            elif kind == 'dr': modified_manifest['environment']['domain_rand']['randomize_friction'] = True
            elif kind == 'pd': modified_manifest['runtime']['p_gains'][0] += 1
            elif kind == 'crop':
                value['reference_failure'][-1, 0] = False
                value['reference_done'][-1, 0] = False
                modified_manifest['episodes'] = audit_rollout_arrays(value, sealed_recorder_contract('formal', 705, 8, 60.))
            elif kind == 'lost_terminal':
                value['reference_done'][-1, 0] = False
                modified_manifest['episodes'] = audit_rollout_arrays(value, sealed_recorder_contract('formal', 705, 8, 60.))
            else: value['reference_dof_pos'] = value['reference_dof_pos'][:, :, :11]
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)):
                arm_audit(modified_manifest, value, self.inputs, self.sealed, 'candidate')

    def test_pair_retains_different_lengths_all8_each_and_cannot_swap_source(self):
        source_manifest, source = arm_fixture('source', self.inputs, self.sealed, ticks=100)
        candidate_manifest, candidate = arm_fixture('candidate', self.inputs, self.sealed, ticks=80)
        options = dict(contract=sealed_recorder_contract('formal', 705, 8, 60.),
            identity=IDENTITY, code_commit=COMMIT, implementation_fingerprint=IMPLEMENTATION,
            head_sha256=HEAD_SHA, candidate_model_sha256=CANDIDATE_SHA, inputs=self.inputs,
            sealed_arrays=self.sealed, source_runtime_manifest=source_runtime_fixture(),
            arm_file_shas={'source': 'a'*64, 'candidate': 'b'*64})
        manifest, arrays = assemble_sealed_pair(source_manifest, source, candidate_manifest, candidate, **options)
        self.assertEqual(manifest['type'], SCHEMA)
        self.assertEqual(manifest['parent_checkpoint_sha256'], SOURCE_SHA)
        self.assertEqual(manifest['head_artifact_sha256'], HEAD_SHA)
        self.assertEqual(arrays['source_reference_valid'].shape, (100, 8))
        self.assertEqual(arrays['candidate_reference_valid'].shape, (80, 8))
        self.assertEqual(len(manifest['arms']['candidate']['actual_audit']['episodes']['reference']), 8)
        self.assertIs(arrays['candidate_reference_physics_raw_torque'], candidate['reference_physics_raw_torque'])
        self.assertFalse(manifest['effectiveness_verified'])
        self.assertFalse(manifest['dr_unlocked'])
        self.assertNotIn('completed_updates', manifest)
        with self.assertRaises(ValueError):
            assemble_sealed_pair(candidate_manifest, candidate, source_manifest, source, **options)
        options['arm_file_shas']['candidate'] = options['arm_file_shas']['source']
        with self.assertRaises(ValueError):
            assemble_sealed_pair(source_manifest, source, candidate_manifest, candidate, **options)

    def test_packed_archive_exact_readback_exclusive_and_no_extra_model_route(self):
        source_manifest, source = arm_fixture('source', self.inputs, self.sealed, ticks=100)
        candidate_manifest, candidate = arm_fixture('candidate', self.inputs, self.sealed, ticks=80)
        manifest, arrays = assemble_sealed_pair(source_manifest, source, candidate_manifest, candidate,
            contract=sealed_recorder_contract('formal', 705, 8, 60.), identity=IDENTITY,
            code_commit=COMMIT, implementation_fingerprint=IMPLEMENTATION,
            head_sha256=HEAD_SHA, candidate_model_sha256=CANDIDATE_SHA, inputs=self.inputs,
            sealed_arrays=self.sealed, source_runtime_manifest=source_runtime_fixture(),
            arm_file_shas={'source': 'a'*64, 'candidate': 'b'*64})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model_7100007.pt'
            _write_bundle(path, manifest, arrays, torch)
            actual_manifest, actual_arrays = _read_bundle(path, torch)
            self.assertEqual(actual_manifest, manifest)
            self.assertEqual(set(actual_arrays), set(arrays))
            for key in arrays:
                self.assertEqual(actual_arrays[key].dtype, arrays[key].dtype)
                self.assertEqual(actual_arrays[key].shape, arrays[key].shape)
                self.assertEqual(actual_arrays[key].tobytes(), arrays[key].tobytes())
            self.assertEqual(len(list(Path(temporary).glob('model_*.pt'))), 1)
            with self.assertRaises(FileExistsError): _write_bundle(path, manifest, arrays, torch)

    def test_two_fresh_process_command_fixed_bindings_non_sdk_temp_names(self):
        extra = argparse.Namespace(source_checkpoint=Path('source.pt'), head_artifact=Path('head.pt'),
            head_sha256=HEAD_SHA, expected_commit=COMMIT, sealed_cohort=Path('sealed.pt'),
            train_cohort=Path('train.pt'), validation_cohort=Path('validation.pt'))
        remaining = ['--task', 'original', '--seed', '705', '--num_envs', '8', '--headless']
        commands = [arm_command(extra, remaining, arm, Path('temp')/(arm+'-arm.pt')) for arm in ARMS]
        for arm, command in zip(ARMS, commands):
            self.assertEqual(command[command.index('--arm')+1], arm)
            self.assertEqual(command[command.index('--duration')+1], '60')
            self.assertEqual(command[command.index('--head-sha256')+1], HEAD_SHA)
            self.assertNotIn('model_', command[command.index('--output')+1])
            self.assertEqual(command[-len(remaining):], remaining)
        with self.assertRaises(ValueError): arm_command(extra, remaining, 'source', Path('model_7100007.pt'))

    def test_native_abi_one_forward_per_tick_and_original_physics_no_fit(self):
        path = ROOT/'humanoid/scripts/record_amp_head_sealed.py'
        code = path.read_text(encoding='utf-8')
        tree = ast.parse(code)
        for name in ('_native_main', '_collect_arm'):
            native = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            imports = [node for node in native.body if isinstance(node, (ast.Import, ast.ImportFrom))]
            self.assertEqual(imports[0].module, 'isaacgym')
            self.assertEqual(imports[1].names[0].name, 'torch')
        self.assertEqual(code.count('policy.act_inference(obs)'), 1)
        self.assertNotIn('policy.act(', code)
        self.assertNotIn('optimizer.step', code)
        self.assertNotIn('make_alg_runner', code)
        self.assertNotIn('forward_parity(', code)
        self.assertIn("apply_diagnostic_config(cfg, 'original')", code)
        self.assertIn('validate_head_artifact(artifact, extra.source_checkpoint)', code)
        self.assertIn('subprocess.check_call(arm_command(', code)
        self.assertIn("with output.open('xb')", code)
        self.assertIn('original_termination()', code)
        self.assertIn('capture()  # terminal native state BEFORE auto reset.', code)


if __name__ == '__main__':
    unittest.main()
