"""Synthetic recorder protocols, NOT native rollout/candidate effectiveness."""
import ast
import copy
from pathlib import Path
import unittest

import numpy as np

from humanoid.scripts.record_amp_head_policy import (
    SCHEMA, BUDGETS, MODE_SEEDS, audit_rollout_arrays, initial_source_matches,
    recorder_contract, require_candidate_admission)
from humanoid.scripts.collect_amp_head_cohort import (
    INITIAL_FIELDS, PHYSICS_FIELDS, empty_exclusion_index, numerical_fingerprint)
from test_head_cohort import fixture


ROOT = Path(__file__).resolve().parents[1]


def rollout_fixture():
    """Smoke-shaped CPU arrays only, not a captured native physics interval."""
    _, source = fixture()
    contract = recorder_contract('smoke', 5, 2, 1.)
    arrays = {}
    for mode in contract['modes']:
        for key in INITIAL_FIELDS:
            arrays[mode+'_'+key] = source['reference_'+key][:100, :2].copy()
            arrays[mode+'_initial_'+key] = source['reference_initial_'+key][:2].copy()
        for key in PHYSICS_FIELDS:
            arrays[mode+'_'+key] = source['reference_'+key][:100, :2].copy()
        arrays[mode+'_time'] = (np.arange(100)+1)*.01
        arrays[mode+'_done'] = np.zeros((100, 2), dtype=bool)
        arrays[mode+'_policy_raw_mu'] = source['reference_raw_mu'][:100, :2].copy()
        raw_shapes = dict(physics_raw_velocity=(10, 12), physics_raw_torque=(10, 12),
            physics_raw_body_force=(10, 3, 3), physics_raw_body_state=(10, 3, 13),
            physics_interval_initial_velocity=(12,), physics_interval_initial_torque=(12,))
        for key, shape in raw_shapes.items():
            arrays[mode+'_'+key] = np.zeros((100,)+shape, dtype=np.float32)
    return contract, arrays


def initial16(mode):
    _, fixture_arrays = fixture()
    initial = {}
    for key in INITIAL_FIELDS:
        first = fixture_arrays['reference_initial_'+key]
        value = np.concatenate([first.copy() for _ in range(4)], axis=0)
        if value.dtype == np.float32:
            flat = value.reshape(16, -1)
            flat[:, 0] += np.arange(16, dtype=np.float32)
        initial[key] = value
    index = empty_exclusion_index()
    for i in range(16):
        digest = numerical_fingerprint({key: value[i] for key, value in initial.items()})
        index['initial_state'][digest] = ['source110/%s/env-%d/episode-0' % (mode, i)]
    return initial, index


class HeadRecorderProtocolTests(unittest.TestCase):
    def test_exact_old32_and_new_source_only_smoke_budgets(self):
        self.assertEqual(BUDGETS, dict(smoke=(2, 1.), formal=(16, 60.)))
        self.assertEqual(MODE_SEEDS, dict(standing=5, reference=105))
        for mode, n, duration in (('smoke', 2, 1.), ('formal', 16, 60.)):
            contract = recorder_contract(mode, 5, n, duration)
            self.assertEqual(contract['type'], SCHEMA)
            self.assertEqual(contract['parent_completed_updates'], 2500)
            self.assertFalse(contract['ppo_continuation'])
            self.assertFalse(contract['effectiveness_verified'])
            self.assertNotIn('completed_updates', contract)
            self.assertNotIn('iter', contract)
        for args in (('formal', 5, 8, 60.), ('formal', 105, 16, 60.),
                     ('formal', 5, 16, 20.), ('smoke', 5, 4, 2.),
                     ('smoke', True, 2, 1.), ('smoke', 5, True, 1.)):
            with self.subTest(args=args), self.assertRaises(ValueError): recorder_contract(*args)

    def test_candidate_requires_formal_and_true_offline_admission_after_strict_archive_gate(self):
        # Deliberately minimal synthetic admission fixture, NOT a valid archive.
        artifact = dict(artifact_kind='actor_head_offline_v1', provenance=dict(
            mode='formal', offline_admitted=True), effectiveness_verified=False, dr_unlocked=False)
        self.assertTrue(require_candidate_admission(artifact, 'formal'))
        self.assertTrue(require_candidate_admission(None, 'smoke'))
        for changed in ('missing', 'kind', 'mode', 'failed', 'int_flag', 'effectiveness', 'dr'):
            value = copy.deepcopy(artifact)
            if changed == 'missing': value = None
            elif changed == 'kind': value['artifact_kind'] = 'native_ppo_checkpoint'
            elif changed == 'mode': value['provenance']['mode'] = 'smoke'
            elif changed == 'failed': value['provenance']['offline_admitted'] = False
            elif changed == 'int_flag': value['provenance']['offline_admitted'] = 1
            elif changed == 'effectiveness': value['effectiveness_verified'] = True
            else: value['dr_unlocked'] = True
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                require_candidate_admission(value, 'formal')
        with self.assertRaises(ValueError): require_candidate_admission(artifact, 'smoke')
        with self.assertRaises(ValueError): require_candidate_admission(None, 'sealed')

    def test_formal_exact12_initial_fields_must_match_same_source_mode_and_env(self):
        for mode in ('standing', 'reference'):
            initial, exclusions = initial16(mode)
            before = {key: value.copy() for key, value in initial.items()}
            proof = initial_source_matches(initial, mode, exclusions, enforce=True)
            self.assertTrue(proof['all_exact'])
            self.assertEqual(len(proof['rows']), 16)
            self.assertEqual(proof['fields'], list(INITIAL_FIELDS))
            for key in initial: np.testing.assert_array_equal(initial[key], before[key])
            initial['foot_force'][8, 0, 2] += .00001
            with self.assertRaisesRegex(ValueError, 'immutable original32'):
                initial_source_matches(initial, mode, exclusions, enforce=True)

    def test_different_source_label_and_missing_initial_fields_cannot_fake_matching(self):
        initial, exclusions = initial16('standing')
        wrong = initial_source_matches(initial, 'reference', exclusions, enforce=False)
        self.assertFalse(wrong['all_exact'])
        with self.assertRaises(ValueError): initial_source_matches(initial, 'reference', exclusions, enforce=True)
        for mutation in ('missing', 'extra', 'count'):
            value = copy.deepcopy(initial)
            if mutation == 'missing': value.pop('foot_force')
            elif mutation == 'extra': value['fake'] = np.zeros(16)
            else: value['dof_vel'] = value['dof_vel'][:15]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                initial_source_matches(value, 'standing', exclusions, enforce=True)

    def test_smoke_does_not_claim_matched32_quality_protocol(self):
        initial, exclusions = initial16('standing')
        short = {key: value[:2].copy() for key, value in initial.items()}
        proof = initial_source_matches(short, 'standing', exclusions, enforce=False)
        self.assertTrue(proof['all_exact'])
        self.assertFalse(proof['formal_match_required'])
        with self.assertRaises(ValueError): initial_source_matches(short, 'standing', exclusions, enforce=True)

    def test_both_modes_all_environments_and_raw10_substeps_preserved(self):
        contract, arrays = rollout_fixture()
        rows = audit_rollout_arrays(arrays, contract)
        self.assertEqual(set(rows), {'standing', 'reference'})
        self.assertEqual(sum(len(value) for value in rows.values()), 4)
        self.assertTrue(all(row['complete'] for values in rows.values() for row in values))
        self.assertEqual(arrays['reference_physics_raw_velocity'].shape, (100, 10, 12))

    def test_failed_terminal_prefix_and_invalid_tail_remain_unchanged(self):
        contract, arrays = rollout_fixture()
        arrays['reference_valid'][62:, 1] = False
        arrays['reference_failure'][61, 1] = True
        arrays['reference_done'][61, 1] = True
        before = {key: value.copy() for key, value in arrays.items()}
        rows = audit_rollout_arrays(arrays, contract)
        failed = rows['reference'][1]
        self.assertTrue(failed['failed'])
        self.assertFalse(failed['complete'])
        self.assertEqual(failed['valid_ticks'], 62)
        self.assertEqual(failed['terminal_tick'], 61)
        self.assertEqual(len(rows['reference']), 2)
        for key in arrays: np.testing.assert_array_equal(arrays[key], before[key])
        arrays['reference_valid'][80, 1] = True
        with self.assertRaises(ValueError): audit_rollout_arrays(arrays, contract)

    def test_missing_raw_substep_nonfinite_missing_ticks_and_initials_rejected(self):
        for changed in ('raw_short', 'raw_nan', 'tick', 'initial', 'mask', 'cross_terminal', 'int_mask'):
            contract, arrays = rollout_fixture()
            if changed == 'raw_short': arrays['standing_physics_raw_velocity'] = arrays['standing_physics_raw_velocity'][:, :9]
            elif changed == 'raw_nan': arrays['standing_physics_raw_torque'][0, 0, 0] = np.nan
            elif changed == 'tick': arrays['standing_time'][33] += .01
            elif changed == 'initial': arrays['standing_initial_dof_vel'] = arrays['standing_initial_dof_vel'][:1]
            elif changed == 'mask': arrays['standing_valid'][3, 0] = False
            elif changed == 'int_mask': arrays['standing_failure'] = arrays['standing_failure'].astype(np.int64)
            else: arrays['standing_done'][4, 0] = True
            with self.subTest(changed=changed), self.assertRaises(ValueError): audit_rollout_arrays(arrays, contract)

    def test_native_abi_source_guards_and_no_extra_policy_forward_or_optimizer(self):
        path = ROOT/'humanoid/scripts/record_amp_head_policy.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        native = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_native_main')
        imports = [node for node in native.body if isinstance(node, (ast.ImportFrom, ast.Import))]
        self.assertEqual(imports[0].module, 'isaacgym')
        self.assertEqual(imports[1].names[0].name, 'torch')
        code = path.read_text(encoding='utf-8')
        self.assertEqual(code.count('policy.act_inference(obs)'), 1)
        self.assertNotIn('policy.act(', code)
        self.assertNotIn('optimizer.step', code)
        self.assertNotIn('make_alg_runner', code)
        self.assertIn('validate_head_artifact(artifact, extra.source_checkpoint)', code)
        native_code = ast.get_source_segment(code, native)
        self.assertLess(native_code.index('validate_head_artifact(artifact,'),
                        native_code.index('require_candidate_admission(artifact,'))
        self.assertIn("apply_diagnostic_config(cfg, 'original')", code)
        self.assertIn('registered_source_exclusions(', code)
        self.assertIn("with extra.output.open('xb')", code)
        self.assertIn('initial_source_matches(initial_fields, mode, exclusions,', code)
        self.assertNotIn("'completed_updates':", code)
        self.assertNotIn("'iter':", code)


if __name__ == '__main__':
    unittest.main()
