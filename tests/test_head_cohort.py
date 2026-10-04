"""CPU synthetic protocol/native-source inference tests, not cloud collection."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from humanoid.scripts.collect_amp_head_cohort import (
    ACTIONS, HISTORY, OBS, INITIAL_FIELDS, PHYSICS_FIELDS, SCHEMA, SOURCE_SHA, SOURCE_ENDPOINT_SHA,
    SOURCE_MODEL_SHA, FrozenActorCapture, audit_arrays, canonical_split,
    cohort_contract, empty_exclusion_index, episode_id, file_sha, fit_admission,
    fingerprint_cohort, full_history_fingerprint, numerical_fingerprint,
    registered_exclusion_path, registered_source_exclusions, source_exclusions,
    validate_history, validate_source_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'f1-amp-sustain'/'outputs'/'amp-sustain'/\
         'TASK_20260926_110'/'model_8802500.pt'


def fixture():
    """4x2 smoke-shaped synthetic arrays; no simulator or fabricated provenance."""
    contract = cohort_contract('train', 'smoke', 306, 4, 2.)
    t, n = 200, 4
    singles = np.empty((t, n, OBS), dtype=np.float32)
    for tick in range(t):
        for i in range(n):
            singles[tick, i] = np.arange(OBS, dtype=np.float32)/100 + tick + i/8
    full = np.zeros((t, n, HISTORY*OBS), dtype=np.float32)
    for tick in range(t):
        start = max(0, tick-HISTORY+1)
        values = singles[start:tick+1].transpose(1, 0, 2).reshape(n, -1)
        full[tick, :, -values.shape[1]:] = values
    widths = dict(root_state=(13,), dof_pos=(12,), dof_vel=(12,), action=(12,),
        torque=(12,), base_lin_vel=(3,), base_ang_vel=(3,), foot_state=(2, 13),
        foot_force=(2, 3), key_positions_w=(15, 3))
    arrays = {}
    for key, width in widths.items():
        value = np.zeros((t, n)+width, dtype=np.float32)
        value[:, :, 0] = np.arange(n, dtype=np.float32).reshape((1, n)+(1,)*(len(width)-1))
        arrays['reference_'+key] = value
        arrays['reference_initial_'+key] = value[0].copy()
    arrays.update(reference_valid=np.ones((t, n), dtype=bool),
        reference_failure=np.zeros((t, n), dtype=bool),
        reference_done=np.zeros((t, n), dtype=bool),
        reference_initial_valid=np.ones(n, dtype=bool),
        reference_initial_failure=np.zeros(n, dtype=bool),
        reference_initial_native_episode=np.arange(n, dtype=np.int64)+2,
        reference_native_episode=np.broadcast_to(np.arange(n, dtype=np.int64)+2, (t, n)).copy(),
        reference_full_obs=full, reference_single_obs=singles,
        reference_actor_hidden=np.zeros((t, n, 128), dtype=np.float32),
        reference_raw_mu=arrays['reference_action'].copy(),
        reference_initial_full_obs=full[0].copy(),
        reference_tick=np.broadcast_to(np.arange(t, dtype=np.int64)[:, None], (t, n)).copy(),
        reference_episode_id=np.zeros((t, n), dtype=np.int64),
        reference_time=(np.arange(t)+1)*.01,
        reference_command=np.broadcast_to(np.array([.45, 0., 0., 0.], dtype=np.float32), (t, n, 4)).copy())
    for key in PHYSICS_FIELDS:
        shape = (t, n) if key == 'physics_valid' else (t, n, 2 if key == 'physics_foot_force_peak' else 12)
        arrays['reference_'+key] = np.zeros(shape, dtype=bool if key == 'physics_valid' else np.float32)
    return contract, arrays


def bytes_snapshot(value):
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), value.detach().cpu().contiguous().numpy().tobytes())
    if isinstance(value, dict):
        return tuple((key, bytes_snapshot(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(bytes_snapshot(item) for item in value)
    return value


class CohortProtocolTests(unittest.TestCase):
    def test_v2_protocol_rejects_all_legacy_seeds_not_claiming_novelty(self):
        self.assertEqual(SCHEMA, 'head_cohort_v2')
        for split, old_seed, new_seed in (('train', 305, 306),
                                         ('validation', 505, 506), ('sealed', 705, 706)):
            for mode, n, duration in (('smoke', 4, 2.), ('formal', 8, 20.)):
                with self.subTest(split=split, mode=mode), self.assertRaises(ValueError):
                    cohort_contract(split, mode, old_seed, n, duration)
            with self.assertRaises(ValueError):
                episode_id(split, old_seed, 0)
            self.assertEqual(episode_id(split, new_seed, 0),
                'head_cohort_v2/%s/seed-%d/env-0/episode-0' % (split, new_seed))

    def test_only_registered_reference_seeds_and_exact_budgets(self):
        self.assertEqual(canonical_split('val'), 'validation')
        for split, seed in (('train', 306), ('validation', 506), ('sealed', 706)):
            for mode, n, duration in (('smoke', 4, 2.), ('formal', 8, 20.)):
                contract = cohort_contract(split, mode, seed, n, duration)
                self.assertEqual(contract['type'], SCHEMA)
                self.assertEqual(contract['expected_ticks'], int(duration*100))
                self.assertEqual(contract['initialization_mode'], 'reference_only')
                self.assertNotIn('initialization', contract)
                self.assertEqual(len(set(contract['episode_ids'])), n)
                self.assertEqual(contract['episode_ids'][0], episode_id(split, seed, 0))
        for args in (('train', 'smoke', 5, 4, 2.), ('train', 'smoke', 306, 8, 2.),
                     ('train', 'formal', 306, 8, 60.), ('standing', 'smoke', 306, 4, 2.),
                     ('train', 'smoke', True, 4, 2.), ('train', 'smoke', 306, True, 2.)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                cohort_contract(*args)

    def test_hash_normalizes_signed_zero_without_mutating_or_rounding(self):
        left = np.array([-0., 1.], dtype=np.float32)
        right = np.array([0., 1.], dtype=np.float32)
        original = left.tobytes()
        self.assertEqual(numerical_fingerprint({'value': left}), numerical_fingerprint({'value': right}))
        self.assertEqual(left.tobytes(), original)
        right[1] = np.nextafter(right[1], np.float32(2.))
        self.assertNotEqual(numerical_fingerprint({'value': left}), numerical_fingerprint({'value': right}))
        with self.assertRaises(ValueError): numerical_fingerprint({'value': np.array([np.nan])})
        with self.assertRaises(ValueError): full_history_fingerprint(np.zeros(3102, dtype=np.float64))

    def test_actual_history_becomes_ready_only_after66_recorded_decisions(self):
        contract, arrays = fixture()
        ready, frames, proof, episodes = audit_arrays(arrays, contract)
        self.assertFalse(ready[:66].any())
        self.assertTrue(ready[66:].all())
        self.assertEqual(proof['checked_ready_inputs'], 134*4)
        self.assertTrue(proof['no_cross_episode'])
        self.assertTrue(all(row['complete'] for row in episodes))
        self.assertTrue(all(not row['fit_eligible'] for row in episodes))
        self.assertEqual(frames[0, 0], 1)
        self.assertEqual(frames[65, 0], 66)
        arrays['reference_history_ready'] = ready.copy()
        arrays['reference_history_real_frames'] = frames.copy()
        audit_arrays(arrays, contract)
        arrays['reference_history_ready'][0, 0] = True
        with self.assertRaisesRegex(ValueError, 'readiness'):
            audit_arrays(arrays, contract)

    def test_missing_reordered_and_cross_episode_histories_are_rejected(self):
        for mutation in ('reordered', 'missing_tick', 'new_episode', 'bad_last_single', 'bad_initial_obs'):
            contract, arrays = fixture()
            if mutation == 'reordered': arrays['reference_full_obs'][100, 0, 0] += 1
            elif mutation == 'missing_tick': arrays['reference_tick'][100, 0] += 1
            elif mutation == 'new_episode': arrays['reference_native_episode'][100, 0] += 1
            elif mutation == 'bad_last_single': arrays['reference_single_obs'][0, 0, 0] += 1
            else: arrays['reference_initial_full_obs'][0, -1] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit_arrays(arrays, contract)

    def test_failures_keep_exact_terminal_prefix_and_cannot_fit(self):
        contract, arrays = fixture()
        original_full = arrays['reference_full_obs'].copy()
        arrays['reference_valid'][174:, 1] = False
        arrays['reference_failure'][173, 1] = True
        arrays['reference_done'][173, 1] = True
        arrays['reference_native_episode'][174:, 1] += 1
        ready, _, proof, episodes = audit_arrays(arrays, contract)
        row = episodes[1]
        self.assertEqual(row['observed_ticks'], 174)
        self.assertEqual(row['terminal_tick'], 173)
        self.assertTrue(row['failed'])
        self.assertFalse(row['complete'])
        self.assertFalse(row['fit_eligible'])
        self.assertTrue(arrays['reference_failure'][row['terminal_tick'], 1])
        self.assertFalse(ready[174:, 1].any())
        np.testing.assert_array_equal(arrays['reference_full_obs'], original_full)
        self.assertFalse(fit_admission(contract, episodes, proof, {'passed': True}))
        arrays['reference_valid'][190, 1] = True
        with self.assertRaises(ValueError): audit_arrays(arrays, contract)

    def test_early_timeout_and_initial_failure_are_not_complete_episodes(self):
        contract, arrays = fixture()
        arrays['reference_done'][-1, 0] = True
        arrays['reference_initial_failure'][1] = True
        arrays['reference_valid'][:, 1] = False
        _, _, _, rows = audit_arrays(arrays, contract)
        self.assertFalse(rows[0]['complete'])
        self.assertFalse(rows[0]['failed'])
        self.assertTrue(rows[1]['failed'])
        self.assertEqual(rows[1]['observed_ticks'], 0)

    def test_seed_labels_cannot_hide_actual_duplicate_inputs(self):
        contract, arrays = fixture()
        ready, _, _, _ = audit_arrays(arrays, contract)
        own, proof = fingerprint_cohort(arrays, contract, ready, empty_exclusion_index())
        self.assertTrue(proof['passed'])
        exclusions = empty_exclusion_index()
        exclusions['initial_state'][own[0]['initial_state']] = ['source110/reference/env-9/episode-0']
        _, proof = fingerprint_cohort(arrays, contract, ready, exclusions)
        self.assertFalse(proof['passed'])
        self.assertEqual(proof['overlaps'][0]['kind'], 'initial_state')
        exclusions = empty_exclusion_index()
        exclusions['ready_full_history'][own[0]['ready_full_history'][0]['sha256']] = [
            'head_cohort_v2/validation/seed-506/env-0/episode-0']
        _, proof = fingerprint_cohort(arrays, contract, ready, exclusions)
        self.assertFalse(proof['passed'])
        self.assertEqual(proof['overlaps'][0]['kind'], 'ready_full_history')

    def test_cross_environment_duplicates_reject_but_same_episode_repeats_are_legal(self):
        contract, arrays = fixture()
        ready, _, _, _ = audit_arrays(arrays, contract)
        arrays['reference_full_obs'][67, 0] = arrays['reference_full_obs'][66, 0]
        _, proof = fingerprint_cohort(arrays, contract, ready, empty_exclusion_index())
        self.assertTrue(proof['passed'])  # fingerprint helper; history gate is separate.
        for key in INITIAL_FIELDS:
            arrays['reference_initial_'+key][1] = arrays['reference_initial_'+key][0]
        arrays['reference_initial_full_obs'][1] = arrays['reference_initial_full_obs'][0]
        _, proof = fingerprint_cohort(arrays, contract, ready, empty_exclusion_index())
        self.assertFalse(proof['passed'])
        self.assertTrue(any(row['kind'] == 'initial_state' for row in proof['overlaps']))

    def test_formal_admission_cannot_drop_a_failed_environment_or_fit_sealed_smoke(self):
        contract = cohort_contract('train', 'formal', 306, 8, 20.)
        rows = [dict(fit_eligible=True) for _ in range(8)]
        self.assertTrue(fit_admission(contract, rows, {'exact': True}, {'passed': True}))
        bad = copy.deepcopy(rows); bad[3]['fit_eligible'] = False
        self.assertFalse(fit_admission(contract, bad, {'exact': True}, {'passed': True}))
        self.assertFalse(fit_admission(contract, rows[:-1], {'exact': True}, {'passed': True}))
        self.assertFalse(fit_admission(contract, rows, {'exact': True}, {'passed': False}))
        self.assertFalse(fit_admission(contract, rows, {'exact': False}, {'passed': True}))
        for split, mode, seed, n, duration in (('sealed', 'formal', 706, 8, 20.),
                                               ('train', 'smoke', 306, 4, 2.)):
            current = cohort_contract(split, mode, seed, n, duration)
            self.assertFalse(fit_admission(current, [dict(fit_eligible=True)]*n,
                {'exact': True}, {'passed': True}))

    def test_wrong_source_endpoint_rejected_before_any_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wrong.pt'
            path.write_bytes(b'not the source regression archive')
            self.assertEqual(file_sha(path), hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'immutable original110'):
                source_exclusions(path, {}, None)

    def test_native_config_pd_and_physx_readback_only_allow_registered_collection_changes(self):
        contract = cohort_contract('train', 'smoke', 306, 4, 2.)
        physx = dict(solver_type=1, num_position_iterations=4, num_velocity_iterations=0,
            contact_offset=.01, rest_offset=0., max_depenetration_velocity=1., contact_collection=2)
        environment = dict(seed=5, env=dict(num_envs=16, episode_length_s=60.1, frame_stack=66),
            control=dict(action_scale=.5, stiffness={'ankle_pitch': 35.}),
            sim=dict(dt=.001, physx=physx))
        native_dt = float(np.float32(.001))
        runtime = dict(dof_properties={'armature': [0.1]}, p_gains=[35.], d_gains=[1.5],
            control_dt=10*native_dt, physics_dt=native_dt)
        source = dict(environment=environment, runtime=runtime)
        current = copy.deepcopy(environment)
        current.update(seed=306)
        current['env'].update(num_envs=4, episode_length_s=2.1)
        actual_runtime = copy.deepcopy(runtime)
        read_physx = copy.deepcopy(physx)
        for key in ('contact_offset', 'rest_offset', 'max_depenetration_velocity'):
            read_physx[key] = float(np.float32(read_physx[key]))
        actual_runtime['physics_sim_parameters'] = dict(dt=native_dt, physx=read_physx)
        proof = validate_source_runtime(current, actual_runtime, source, contract)
        self.assertTrue(proof['dof_pd_timing_exact'])
        for mutation in ('scale', 'stiffness', 'history', 'dof', 'p_gain', 'sim', 'missing_sim', 'seed'):
            cfg, readback = copy.deepcopy(current), copy.deepcopy(actual_runtime)
            if mutation == 'scale': cfg['control']['action_scale'] = .8
            elif mutation == 'stiffness': cfg['control']['stiffness']['ankle_pitch'] = 36.
            elif mutation == 'history': cfg['env']['frame_stack'] = 65
            elif mutation == 'dof': readback['dof_properties']['armature'] = [.2]
            elif mutation == 'p_gain': readback['p_gains'][0] = 36.
            elif mutation == 'sim': readback['physics_sim_parameters']['physx']['num_velocity_iterations'] = 1
            elif mutation == 'missing_sim': readback['physics_sim_parameters']['physx']['num_velocity_iterations'] = None
            else: cfg['seed'] = 5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_source_runtime(cfg, readback, source, contract)

    def test_derived_index_path_and_hash_owned_by_fixed_config_not_argv(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixed = root/'resources/amp_inputs/source110_exclusions_v1.npz'
            config_path = root/'configs/amp/head_smooth_v2.json'
            fixed.parent.mkdir(parents=True); config_path.parent.mkdir(parents=True)
            fixed.write_bytes(b'synthetic immutable-index protocol fixture, not actual source')
            spec = dict(path='resources/amp_inputs/source110_exclusions_v1.npz', sha256=file_sha(fixed))
            config_path.write_text(json.dumps(dict(source_exclusions=spec)), encoding='utf-8')
            path, digest = registered_exclusion_path(root, fixed)
            self.assertEqual(path, fixed.resolve())
            self.assertEqual(digest, spec['sha256'])
            wrong = fixed.parent/'arbitrary.npz'; wrong.write_bytes(fixed.read_bytes())
            with self.assertRaises(ValueError): registered_exclusion_path(root, wrong)
            fixed.write_bytes(b'different bytes')
            with self.assertRaises(ValueError): registered_exclusion_path(root, fixed)
            config_path.write_text(json.dumps(dict(source_exclusions=dict(spec, sha256='a'*64))), encoding='utf-8')
            with self.assertRaises(ValueError): registered_exclusion_path(root, fixed)
            config_path.write_text(json.dumps(dict(source_exclusions=dict(spec, path='arbitrary.npz'))), encoding='utf-8')
            with self.assertRaises(ValueError): registered_exclusion_path(root, wrong)

    def test_derived_wrapper_exposes_same_original_sha_as_raw_endpoint_and_separate_file_sha(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixed = root/'resources/amp_inputs/source110_exclusions_v1.npz'
            config_path = root/'configs/amp/head_smooth_v2.json'
            fixed.parent.mkdir(parents=True); config_path.parent.mkdir(parents=True)
            fixed.write_bytes(b'index interface fixture, not actual source provenance')
            digest = file_sha(fixed)
            config_path.write_text(json.dumps(dict(source_exclusions=dict(
                path='resources/amp_inputs/source110_exclusions_v1.npz', sha256=digest))), encoding='utf-8')
            original = dict(sha256=SOURCE_ENDPOINT_SHA, source_task='TASK_20260926_110',
                            reconstruction_not_raw_history=True)
            wrapper = dict(source_regression=copy.deepcopy(original),
                source_environment={'source': 'readback fixture'},
                source_runtime={'control_dt': .01, 'physics_dt': .001},
                derived_index=dict(sha256=digest, hash_bits=256))
            before = copy.deepcopy(wrapper)
            index, identity = empty_exclusion_index(), {'fixture_identity': True}
            with patch('humanoid.amp.head_exclusions.load_exclusion_index', return_value=(index, wrapper)) as load:
                actual_index, proof, source = registered_source_exclusions(root, fixed, identity)
                load.assert_called_once_with(fixed.resolve(), expected_file_sha=digest, identity=identity)
            self.assertIs(actual_index, index)
            self.assertEqual(proof['sha256'], SOURCE_ENDPOINT_SHA)
            self.assertEqual(proof['derived_index']['sha256'], digest)
            self.assertNotIn('source_regression', proof)
            self.assertNotIn('source_environment', proof)
            self.assertNotIn('source_runtime', proof)
            self.assertEqual(source, dict(environment=wrapper['source_environment'], runtime=wrapper['source_runtime']))
            self.assertEqual(wrapper, before)
            for changed in ('endpoint_sha', 'index_sha', 'missing_runtime', 'flat_old_wrapper'):
                bad = copy.deepcopy(wrapper)
                if changed == 'endpoint_sha': bad['source_regression']['sha256'] = digest
                elif changed == 'index_sha': bad['derived_index']['sha256'] = SOURCE_ENDPOINT_SHA
                elif changed == 'missing_runtime': bad.pop('source_runtime')
                else: bad = dict(original, derived_index=wrapper['derived_index'])
                with patch('humanoid.amp.head_exclusions.load_exclusion_index', return_value=(index, bad)):
                    with self.subTest(changed=changed), self.assertRaises(ValueError):
                        registered_source_exclusions(root, fixed, identity)
    def test_native_entry_imports_isaacgym_before_torch_and_preserves_legacy_recorder(self):
        path = ROOT/'humanoid/scripts/collect_amp_head_cohort.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        native = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_native_main')
        imports = [node for node in native.body if isinstance(node, (ast.ImportFrom, ast.Import))]
        self.assertIsInstance(imports[0], ast.ImportFrom)
        self.assertEqual(imports[0].module, 'isaacgym')
        self.assertIsInstance(imports[1], ast.Import)
        self.assertEqual(imports[1].names[0].name, 'torch')
        self.assertFalse(any(isinstance(node, ast.Import) and any(name.name == 'torch' for name in node.names)
                             for node in tree.body))
        code = path.read_text(encoding='utf-8')
        self.assertNotIn('policy.act(', code)
        self.assertNotIn('make_alg_runner', code)
        self.assertNotIn('optimizer.step', code)
        self.assertIn("apply_diagnostic_config(cfg, 'original')", code)
        self.assertIn("with extra.output.open('xb')", code)
        self.assertIn('add_mutually_exclusive_group(required=True)', code)
        self.assertIn("'--source-exclusion-index'", code)


class ActualArchitectureCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('head_cohort_actual_actor',
            ROOT/'humanoid/algo/ppo/actor_critic_dh.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cls.actor_class = module.ActorCriticDH

    def model(self):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = self.actor_class(235, 47, 219, 12, actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[768, 256, 128], state_estimator_hidden_dims=[256, 128, 64],
                in_channels=66)
        policy.eval()
        return policy

    def check_capture(self, model):
        torch.manual_seed(62)
        obs = torch.randn(4, 3102)
        with torch.no_grad(): expected = model.act_inference(obs)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        for parameter in model.parameters(): parameter.grad = torch.full_like(parameter, .25)
        state = bytes_snapshot(model.state_dict())
        adam = bytes_snapshot(optimizer.state_dict())
        gradients = [(parameter.grad, bytes_snapshot(parameter.grad)) for parameter in model.parameters()]
        rng = torch.get_rng_state().clone()
        module_calls = dict(actor=0, es=0, cnn=0)
        handles = []
        for name, module in (('actor', model.actor), ('es', model.state_estimator), ('cnn', model.long_history)):
            def count(module, inputs, output, name=name): module_calls[name] += 1
            handles.append(module.register_forward_hook(count))
        capture = FrozenActorCapture(model)
        try:
            with torch.no_grad(): result, hidden = capture.inference(obs)
            self.assertTrue(torch.equal(result, expected))
            self.assertEqual(tuple(hidden.shape), (4, 128))
            self.assertEqual(capture.calls, 1)
            self.assertEqual(module_calls, dict(actor=1, es=1, cnn=1))
            self.assertEqual(bytes_snapshot(model.state_dict()), state)
            self.assertEqual(bytes_snapshot(optimizer.state_dict()), adam)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for parameter, (pointer, value) in zip(model.parameters(), gradients):
                self.assertIs(parameter.grad, pointer)
                self.assertEqual(bytes_snapshot(parameter.grad), value)
        finally:
            capture.close()
            for handle in handles: handle.remove()
        self.assertFalse(capture.handles)

    def test_one_original_forward_keeps_action_parameters_gradients_adam_rng(self):
        self.check_capture(self.model())

    @unittest.skipUnless(SOURCE.is_file(), 'Actual original110 source required; never fabricated')
    def test_bound_original110_source_capture_is_read_only(self):
        self.assertEqual(file_sha(SOURCE), SOURCE_SHA)
        state = torch.load(str(SOURCE), weights_only=True, map_location='cpu')
        from humanoid.amp.mu_temporal import state_fingerprint
        self.assertEqual(state_fingerprint(state['model_state_dict']), SOURCE_MODEL_SHA)
        policy = self.model(); policy.load_state_dict(state['model_state_dict'], strict=True)
        self.check_capture(policy)
        self.assertEqual(state_fingerprint(policy.state_dict()), SOURCE_MODEL_SHA)

    def test_changed_final_head_architecture_and_extra_forward_are_rejected(self):
        policy = self.model()
        policy.actor[6] = torch.nn.Linear(128, 11)
        with self.assertRaises(ValueError): FrozenActorCapture(policy)
        policy = self.model()
        capture = FrozenActorCapture(policy)
        try:
            original = policy.act_inference
            def repeated(obs):
                original(obs)
                return original(obs)
            policy.act_inference = repeated
            with torch.no_grad(), self.assertRaisesRegex(ValueError, 'exactly one'):
                capture.inference(torch.zeros(4, 3102))
        finally: capture.close()


if __name__ == '__main__':
    unittest.main()
