import unittest

from humanoid.gmr_inference_identity import PROFILES, validate_identity


class InferenceIdentityTest(unittest.TestCase):
    def check(self, task, **overrides):
        values = dict(PROFILES[task])
        values.update(overrides)
        return validate_identity(task, values['source_task'], values['checkpoint_sha256'],
                                 values['motion_sha256'])

    def test_verified_profiles(self):
        for task, profile in PROFILES.items():
            self.assertEqual(self.check(task)['training_commit'], profile['training_commit'])

    def test_reject_old_checkpoint_for_upright(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            self.check('x1_gmr_upright', checkpoint_sha256=PROFILES['x1_gmr_clip']['checkpoint_sha256'])

    def test_reject_old_reference_for_upright(self):
        with self.assertRaisesRegex(ValueError, 'motion_sha256'):
            self.check('x1_gmr_upright', motion_sha256=PROFILES['x1_gmr_clip']['motion_sha256'])

    def test_reject_wrong_source_task(self):
        with self.assertRaisesRegex(ValueError, 'source_task'):
            self.check('x1_gmr_upright', source_task='TASK_20260910_187')

    def test_reject_unknown_task(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            validate_identity('x1_dh_stand', '', '', '')

    def test_result_does_not_mutate_profile(self):
        result = self.check('x1_gmr_upright')
        result['source_task'] = 'wrong'
        self.assertEqual(self.check('x1_gmr_upright')['source_task'], 'TASK_20260911_008')

    def test_final_checkpoint_number_is_bound_to_profile(self):
        p = PROFILES['x1_gmr_smooth']
        result = validate_identity('x1_gmr_smooth', p['source_task'], p['checkpoint_sha256'],
                                   p['motion_sha256'], checkpoint=7000)
        self.assertEqual(result['checkpoint'], 7000)
        with self.assertRaisesRegex(ValueError, 'checkpoint number'):
            validate_identity('x1_gmr_smooth', p['source_task'], p['checkpoint_sha256'],
                              p['motion_sha256'], checkpoint=6900)

    def test_final_checkpoint_cannot_use_previous_upright_weights(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            self.check('x1_gmr_smooth', checkpoint_sha256=PROFILES['x1_gmr_upright']['checkpoint_sha256'])

    def test_swing_final_checkpoint_identity(self):
        p = PROFILES['x1_gmr_swing']
        result = validate_identity('x1_gmr_swing', p['source_task'], p['checkpoint_sha256'],
                                   p['motion_sha256'], checkpoint=8000)
        self.assertEqual(result['num_actions'], 12)
        self.assertEqual(result['source_task'], 'TASK_20260911_171')
        self.assertEqual(result['training_commit'], 'a92dfe6346bcd1e853a1c923fec33667e9b601bb')
        self.assertEqual(result['urdf_lf_sha256'], PROFILES['x1_gmr_smooth']['urdf_lf_sha256'])

    def test_swing_rejects_source_policy_or_smoke_iteration(self):
        p = PROFILES['x1_gmr_swing']
        for checkpoint in (7000, 7020, 7900):
            with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, 'checkpoint number'):
                validate_identity('x1_gmr_swing', p['source_task'], p['checkpoint_sha256'],
                                  p['motion_sha256'], checkpoint=checkpoint)

    def test_swing_rejects_previous_model_and_source(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            self.check('x1_gmr_swing', checkpoint_sha256=PROFILES['x1_gmr_smooth']['checkpoint_sha256'])
        with self.assertRaisesRegex(ValueError, 'source_task'):
            self.check('x1_gmr_swing', source_task='TASK_20260911_116')

    def test_acceleration_final_profiles(self):
        for group, suffix in (('1x', '033'), ('3x', '034'), ('10x', '035')):
            task = 'x1_gmr_accel_' + group
            p = PROFILES[task]
            result = validate_identity(task, p['source_task'], p['checkpoint_sha256'], p['motion_sha256'], 9000)
            self.assertEqual(result['source_task'], 'TASK_20260912_' + suffix)
            self.assertEqual(result['training_commit'], 'eccd14eed21ffe980c90abd7dc740d043e01fcaf')
            self.assertEqual(result['num_actions'], 12)
            self.assertEqual(result['urdf_lf_sha256'], PROFILES['x1_gmr_swing']['urdf_lf_sha256'])
            for checkpoint in (8000, 8020, 8900):
                with self.assertRaisesRegex(ValueError, 'checkpoint number'):
                    validate_identity(task, p['source_task'], p['checkpoint_sha256'], p['motion_sha256'], checkpoint)

    def test_acceleration_cross_group_rejected(self):
        tasks = ['x1_gmr_accel_' + group for group in ('1x', '3x', '10x')]
        for task in tasks:
            for other in tasks + ['x1_gmr_swing']:
                if task == other:
                    continue
                with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
                    self.check(task, checkpoint_sha256=PROFILES[other]['checkpoint_sha256'])
                with self.assertRaisesRegex(ValueError, 'source_task'):
                    self.check(task, source_task=PROFILES[other]['source_task'])

    def test_phase_formal_identity_and_checkpoint(self):
        task = 'x1_gmr_phase_accel'
        p = PROFILES[task]
        result = validate_identity(task, p['source_task'], p['checkpoint_sha256'], p['motion_sha256'], 9000)
        self.assertEqual(result['source_task'], 'TASK_20260912_109')
        self.assertEqual(result['training_commit'], '9eb084ed63c1c4d10fc70f4a50e3bf0cfbec10bd')
        self.assertEqual(result['checkpoint_sha256'], '391ec534f54c73c8272ebc7160b4dec28cf18599f6330e6baca9481b87fc0631')
        self.assertEqual(result['num_actions'], 12)
        self.assertEqual(result['urdf_lf_sha256'], PROFILES['x1_gmr_swing']['urdf_lf_sha256'])
        for checkpoint in (8000, 8020, 8900):
            with self.assertRaisesRegex(ValueError, 'checkpoint number'):
                validate_identity(task, p['source_task'], p['checkpoint_sha256'], p['motion_sha256'], checkpoint)

    def test_phase_rejects_previous_sweep_models_and_smoke_source(self):
        for other in ('x1_gmr_swing', 'x1_gmr_accel_1x', 'x1_gmr_accel_3x', 'x1_gmr_accel_10x'):
            with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
                self.check('x1_gmr_phase_accel', checkpoint_sha256=PROFILES[other]['checkpoint_sha256'])
            with self.assertRaisesRegex(ValueError, 'source_task'):
                self.check('x1_gmr_phase_accel', source_task=PROFILES[other]['source_task'])
        with self.assertRaisesRegex(ValueError, 'source_task'):
            self.check('x1_gmr_phase_accel', source_task='TASK_20260912_076')


if __name__ == '__main__':
    unittest.main()
