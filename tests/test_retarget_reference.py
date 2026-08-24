import unittest
from pathlib import Path

import numpy as np

from humanoid.retarget_reference import (
    X1_12DOF_JOINT_NAMES,
    interpolate_reference,
    load_retarget_reference,
)


class RetargetReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        motion_dir = (
            Path(__file__).resolve().parents[1] / "resources" / "motions" / "x1"
        )
        cls.reference = load_retarget_reference(
            motion_dir / "walk_12dof.csv",
            motion_dir / "walk_12dof_contact_consistent.csv",
        )

    def test_reference_is_exactly_twelve_joint_data(self):
        self.assertEqual(self.reference.joint_names, X1_12DOF_JOINT_NAMES)
        self.assertEqual(self.reference.joint_positions.shape, (415, 12))
        self.assertAlmostEqual(self.reference.duration, 13.8, places=9)

    def test_auxiliary_root_reference_has_matching_samples(self):
        self.assertEqual(self.reference.root_positions.shape, (415, 3))
        self.assertEqual(self.reference.root_quaternions_xyzw.shape, (415, 4))
        self.assertEqual(self.reference.contacts.shape, (415, 2))
        np.testing.assert_allclose(
            np.linalg.norm(self.reference.root_quaternions_xyzw, axis=1), 1.0
        )

    def test_interpolation_keeps_twelve_joint_shape(self):
        joints, root_pos, root_quat, contacts = interpolate_reference(
            self.reference, 1.234
        )
        self.assertEqual(joints.shape, (12,))
        self.assertEqual(root_pos.shape, (3,))
        self.assertEqual(root_quat.shape, (4,))
        self.assertEqual(contacts.shape, (2,))
        self.assertAlmostEqual(np.linalg.norm(root_quat), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
