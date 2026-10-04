"""Native float ABI protocol regression; CPU-only, no PhysX run or real fit.

Constants match the retained old source/077 readbacks, but these test fixtures
are synthetic and cannot admit a new cloud task or prove physical improvement.
"""
import copy
import unittest

import numpy as np

from humanoid.scripts.collect_amp_head_cohort import cohort_contract, validate_source_runtime
from test_head_workflow import native_sim_fixture, source_fixture


class NativeSourceReadbackTests(unittest.TestCase):
    def fixture(self):
        source = source_fixture()
        environment = copy.deepcopy(source['environment'])
        environment['seed'] = 305
        environment['env'].update(num_envs=4, episode_length_s=2.1)
        runtime = copy.deepcopy(source['runtime'])
        runtime['physics_sim_parameters'] = native_sim_fixture(source['environment']['sim'])
        return source, environment, runtime, cohort_contract('train', 'smoke', 305, 4, 2.)

    def test_real_float32_abi_representation_passes_without_rewriting_config(self):
        source, environment, runtime, contract = self.fixture()
        before = copy.deepcopy((source, environment, runtime))
        self.assertEqual(source['environment']['sim']['dt'], .001)
        self.assertEqual(runtime['physics_sim_parameters']['dt'], .0010000000474974513)
        self.assertEqual(runtime['physics_sim_parameters']['physx']['contact_offset'],
                         .009999999776482582)
        result = validate_source_runtime(environment, runtime, source, contract)
        self.assertTrue(result['native_physx_parameters_exact'])
        self.assertIn('exact float32', result['native_float_abi'])
        self.assertEqual((source, environment, runtime), before)

    def test_one_native_float32_ulp_change_still_fails_exact_readback_gate(self):
        source, environment, runtime, contract = self.fixture()
        for name in ('dt', 'contact_offset', 'rest_offset', 'max_depenetration_velocity'):
            changed = copy.deepcopy(runtime)
            row = changed['physics_sim_parameters'] if name == 'dt' else changed['physics_sim_parameters']['physx']
            row[name] = float(np.nextafter(np.float32(row[name]), np.float32(float('inf'))))
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_source_runtime(environment, changed, source, contract)

    def test_config_and_native_must_not_jointly_self_certify_material_change(self):
        source, environment, runtime, contract = self.fixture()
        for name in ('dt', 'contact_offset', 'num_velocity_iterations'):
            cfg, changed = copy.deepcopy(environment), copy.deepcopy(runtime)
            if name == 'dt': cfg['sim'][name] = .002
            elif name == 'contact_offset': cfg['sim']['physx'][name] = .02
            else: cfg['sim']['physx'][name] = 1
            changed['physics_sim_parameters'] = native_sim_fixture(cfg['sim'])
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_source_runtime(cfg, changed, source, contract)

    def test_integer_and_float_readbacks_have_true_abi_types_not_equal_booleans(self):
        source, environment, runtime, contract = self.fixture()
        for name, value in (('solver_type', True), ('num_position_iterations', 4.),
                            ('num_velocity_iterations', False), ('contact_collection', 2.),
                            ('rest_offset', False), ('max_depenetration_velocity', 1)):
            changed = copy.deepcopy(runtime)
            changed['physics_sim_parameters']['physx'][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_source_runtime(environment, changed, source, contract)
        changed = copy.deepcopy(runtime)
        changed['physics_sim_parameters']['dt'] = False
        with self.assertRaises(ValueError):
            validate_source_runtime(environment, changed, source, contract)


if __name__ == '__main__':
    unittest.main()
