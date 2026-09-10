import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

from humanoid.gmr_motion import GMRMotion


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "resources/motions/gmr_kit317_foot_flat_12dof.npz"
URDF = ROOT / "resources/robots/x1/urdf/X1_12DOF.urdf"
NAMES = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]


class GMRMotionTest(unittest.TestCase):
    def test_shape_units_and_duration(self):
        clip = GMRMotion(REFERENCE, NAMES)
        self.assertEqual((139, 12), clip.data["dof_pos"].shape)
        self.assertEqual(30., clip.fps)
        self.assertAlmostEqual(4.6, clip.duration)
        self.assertFalse(clip.metadata["cyclic"])

    def test_by_name_not_index(self):
        normal = GMRMotion(REFERENCE, NAMES)
        reverse = GMRMotion(REFERENCE, NAMES[::-1])
        torch.testing.assert_close(normal.data["dof_pos"].flip(1), reverse.data["dof_pos"])
        torch.testing.assert_close(normal.data["dof_vel"].flip(1), reverse.data["dof_vel"])

    def test_missing_duplicate_joint_rejected(self):
        with self.assertRaises(ValueError):
            GMRMotion(REFERENCE, NAMES[:-1])
        with self.assertRaises(ValueError):
            GMRMotion(REFERENCE, NAMES[:-1] + [NAMES[0]])

    def test_hash_guard(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            GMRMotion(REFERENCE, NAMES, expected_sha256="wrong")
        digest = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
        GMRMotion(REFERENCE, NAMES, expected_sha256=digest)

    def test_clamps_instead_of_wrapping(self):
        clip = GMRMotion(REFERENCE, NAMES)
        values = clip.sample(torch.tensor([-1., 0., 4.6, 5., 9.2]))
        torch.testing.assert_close(values["dof_pos"][0], values["dof_pos"][1])
        torch.testing.assert_close(values["root_pos"][2], values["root_pos"][3])
        torch.testing.assert_close(values["root_pos"][2], values["root_pos"][4])
        self.assertGreater(float(torch.norm(values["root_pos"][2] - values["root_pos"][0])), 1.)

    def test_interpolation_and_quaternion_norm(self):
        clip = GMRMotion(REFERENCE, NAMES)
        half = clip.sample(torch.tensor([.5 / clip.fps]))
        torch.testing.assert_close(half["dof_pos"][0], clip.data["dof_pos"][:2].mean(0))
        sample = clip.sample(torch.linspace(0., 4.6, 461))
        for key in ("root_quat", "foot_quat_base"):
            torch.testing.assert_close(torch.linalg.vector_norm(sample[key], dim=-1), torch.ones(sample[key].shape[:-1]))

    def test_phase_distinguishes_start_and_end(self):
        # Same half-circle encoding as the environment's inherited 47-D observation.
        phase = .5 * torch.tensor([0., 1.])
        self.assertAlmostEqual(float(torch.cos(2 * torch.pi * phase)[0]), 1.)
        self.assertAlmostEqual(float(torch.cos(2 * torch.pi * phase)[1]), -1.)

    def test_reference_within_hard_position_and_velocity_limits(self):
        clip = GMRMotion(REFERENCE, NAMES)
        sample = clip.sample(torch.linspace(0., 4.6, 461))
        self.assertTrue(torch.all(sample["dof_pos"] >= clip.lower - 1e-6))
        self.assertTrue(torch.all(sample["dof_pos"] <= clip.upper + 1e-6))
        limits = [float(j.find("limit").get("velocity")) for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]
        self.assertTrue(torch.all(torch.abs(sample["dof_vel"]) <= torch.tensor(limits) + 1e-6))

    def test_urdf_identity_and_kinematic_quality(self):
        clip = GMRMotion(REFERENCE, NAMES)
        digest = hashlib.sha256(URDF.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        self.assertEqual(digest, clip.metadata["training_urdf_lf_sha256"])
        self.assertLess(clip.metadata["standing_max_sole_tilt_deg"], .01)
        self.assertGreaterEqual(clip.metadata["minimum_foot_mesh_z_m"], .00199)
        self.assertLess(clip.metadata["limit_fit"]["max_ankle_position_change_m"], .001)

    def test_bad_numeric_input_rejected(self):
        with np.load(REFERENCE, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["root_pos"][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.npz"
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "root_pos"):
                GMRMotion(path, NAMES)


if __name__ == "__main__":
    unittest.main()
