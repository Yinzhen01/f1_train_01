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
        p = PROFILES['x1_gmr_29dof']
        result = validate_identity('x1_gmr_29dof', p['source_task'], p['checkpoint_sha256'],
                                   p['motion_sha256'], checkpoint=5000)
        self.assertEqual(result['checkpoint'], 5000)
        with self.assertRaisesRegex(ValueError, 'checkpoint number'):
            validate_identity('x1_gmr_29dof', p['source_task'], p['checkpoint_sha256'],
                              p['motion_sha256'], checkpoint=4900)

    def test_final_checkpoint_cannot_use_previous_upright_weights(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            self.check('x1_gmr_29dof', checkpoint_sha256=PROFILES['x1_gmr_upright']['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
