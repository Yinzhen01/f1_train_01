"""Actual-asset geometry and CPU reward integration tests; no physics claim."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from humanoid.gmr_motion import GMRMotion
from humanoid.gmr_posture import tilt_tracking_reward, tilt_error_squared
from test_gmr_environment_math import environment_class, quat_inverse_apply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'humanoid/scripts'))
from prepare_gmr_reference import fk, JOINT_NAMES, origin, stl_vertices
from prepare_gmr_upright import walking_weight


class UprightReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'resources/motions/gmr_kit317_upright_12dof.npz'
        cls.old_path = ROOT / 'resources/motions/gmr_kit317_foot_flat_12dof.npz'
        cls.clip = GMRMotion(cls.path, list(JOINT_NAMES))
        cls.old = GMRMotion(cls.old_path, list(JOINT_NAMES))
        cls.joints = ET.parse(ROOT / 'resources/robots/x1/urdf/X1_12DOF.urdf').getroot().findall('joint')

    def test_old_reference_unchanged_and_new_variant(self):
        self.assertEqual(hashlib.sha256(self.old_path.read_bytes()).hexdigest(), 'a3ae41e54909455d477cefa662bdcbbee8a1852564ef547ebeb455f275a0be02')
        self.assertEqual(self.clip.metadata['reference_variant'], 'upright_chest_v1')
        self.assertEqual(self.clip.frames, 139)
        self.assertAlmostEqual(self.clip.duration, 4.6)
        torch.testing.assert_close(self.clip.data['foot_contact'], self.old.data['foot_contact'])

    def test_standing_upright_and_walking_forward(self):
        q = self.clip.data['root_quat'].numpy()
        euler = Rotation.from_quat(q).as_euler('xyz', degrees=True)
        standing = np.r_[0:31, 126:139]
        self.assertLess(np.abs(euler[standing, :2]).max(), .001)
        self.assertGreater(euler[:, 1].max(), 5.)
        self.assertLess(euler[:, 1].max(), 13.)
        self.assertGreater(euler[:, 1].min(), -.1)
        old_yaw = Rotation.from_quat(self.old.data['root_quat'].numpy()).as_euler('xyz')[:, 2]
        yaw_error = np.angle(np.exp(1j * (np.deg2rad(euler[:, 2]) - old_yaw)))
        self.assertLess(np.abs(yaw_error).max(), 1e-6)

    def test_position_and_actual_interpolation_speed_limits(self):
        clip = self.clip
        self.assertTrue(torch.all(clip.data['dof_pos'] >= clip.lower - 1e-6))
        self.assertTrue(torch.all(clip.data['dof_pos'] <= clip.upper + 1e-6))
        limits = torch.tensor([float(next(j for j in self.joints if j.get('name') == n).find('limit').get('velocity')) for n in JOINT_NAMES])
        slopes = torch.diff(clip.data['dof_pos'], dim=0) * clip.fps
        self.assertTrue(torch.all(slopes.abs() <= limits + 1e-5))
        self.assertTrue(torch.all(clip.data['dof_vel'].abs() <= limits + 1e-5))
        acceleration = np.gradient(clip.data['root_vel'].numpy(), 1. / clip.fps, axis=0)
        self.assertLess(np.linalg.norm(acceleration, axis=1).max(), 12.)

    def test_exported_local_feet_match_independent_fk(self):
        errors = []
        for i, q in enumerate(self.clip.data['dof_pos'].numpy()):
            transforms = fk(self.joints, dict(zip(JOINT_NAMES, q)))
            for f, name in enumerate(self.clip.metadata['foot_names']):
                T = transforms[name]
                actual = T[:3, 3] + T[:3, :3] @ self.clip.sole_local[f].numpy()
                errors.append(np.linalg.norm(actual - self.clip.data['foot_pos_base'][i, f].numpy()))
        self.assertLess(max(errors), 1e-6)

    def test_report_and_config_hash(self):
        report = json.loads(self.path.with_suffix('.json').read_text())
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(report['output_sha256'], digest)
        config = (ROOT / 'humanoid/envs/x1/x1_gmr_upright_config.py').read_text()
        self.assertIn(digest, config)
        self.assertGreater(report['minimum_foot_mesh_z_m'], 0.)
        self.assertLess(report['standing_max_sole_tilt_deg'], .1)
        self.assertLess(report['max_dense_sole_position_change_m'], .002)

    def test_phase_blend_is_flat_at_standing_boundaries(self):
        t = np.array([0., .5, 1., 1.5, 3.7, 4.2, 4.6])
        np.testing.assert_allclose(walking_weight(t), [0, 0, 0, 1, 1, 0, 0], atol=1e-12)

    def test_dense_foot_clearance_using_independent_urdf_fk(self):
        urdf = ROOT / 'resources/robots/x1/urdf/X1_12DOF.urdf'
        robot = ET.parse(urdf).getroot()
        meshes = []
        for name in self.clip.metadata['foot_names']:
            collision = robot.find("link[@name='%s']/collision" % name)
            mesh = collision.find('geometry/mesh')
            T = origin(collision.find('origin'))
            v = stl_vertices((urdf.parent / mesh.get('filename')).resolve())
            v *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            meshes.append(v @ T[:3, :3].T + T[:3, 3])
        samples = self.clip.sample(torch.linspace(0., self.clip.duration, 461))
        minimum = float('inf')
        for i, q in enumerate(samples['dof_pos'].numpy()):
            transforms = fk(self.joints, dict(zip(JOINT_NAMES, q)))
            R = Rotation.from_quat(samples['root_quat'][i].numpy()).as_matrix()
            p = samples['root_pos'][i].numpy()
            for name, mesh in zip(self.clip.metadata['foot_names'], meshes):
                T = transforms[name]
                vertices = (mesh @ T[:3, :3].T + T[:3, 3]) @ R.T + p
                minimum = min(minimum, vertices[:, 2].min())
        self.assertGreater(minimum, .0004)

    def test_new_scale_inherits_baseline_without_enabling_legacy_upright_reward(self):
        def scale_node(path):
            return next(n for n in ast.walk(ast.parse(path.read_text()))
                        if isinstance(n, ast.ClassDef) and n.name == 'scales')
        scope = {}
        baseline = scale_node(ROOT / 'humanoid/envs/x1/x1_gmr_clip_config.py')
        exec(compile(ast.Module(body=[baseline], type_ignores=[]), '<old scales>', 'exec'), scope)
        old = scope['scales']
        scope['X1GMRClipCfg'] = SimpleNamespace(rewards=SimpleNamespace(scales=old))
        new = scale_node(ROOT / 'humanoid/envs/x1/x1_gmr_upright_config.py')
        exec(compile(ast.Module(body=[new], type_ignores=[]), '<new scales>', 'exec'), scope)
        self.assertEqual(scope['scales'].gmr_trunk_tilt, 1.)
        for key in vars(old):
            if not key.startswith('_'):
                self.assertEqual(getattr(scope['scales'], key), getattr(old, key))
        self.assertFalse(hasattr(scope['scales'], 'orientation'))
        registry = (ROOT / 'humanoid/envs/__init__.py').read_text()
        self.assertIn('register("x1_gmr_upright", X1GMRUprightEnv', registry)


class UprightEnvironmentMathTest(unittest.TestCase):
    def environment_class(self):
        path = ROOT / 'humanoid/envs/x1/x1_gmr_upright_env.py'
        definition = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        scope = dict(X1GMRClipEnv=environment_class(), torch=torch,
                     quat_rotate_inverse=quat_inverse_apply,
                     tilt_tracking_reward=tilt_tracking_reward, tilt_error_squared=tilt_error_squared)
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), scope)
        return scope['X1GMRUprightEnv']

    def test_wrong_reference_rejected(self):
        cls = self.environment_class()
        env = cls.__new__(cls)
        env.motion = SimpleNamespace(metadata={})
        with patch.object(cls.__bases__[0], '_init_buffers', lambda self: None):
            with self.assertRaisesRegex(ValueError, 'rebuilt reference'):
                env._init_buffers()

    def test_actual_environment_reward_and_yaw_independence(self):
        cls = self.environment_class()
        env = cls.__new__(cls)
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(trunk_tilt_sigma=.1))
        env.gravity_vec = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
        ref = torch.tensor(Rotation.from_euler('xyz', [[0., 5., 0.]] * 3, degrees=True).as_quat(), dtype=torch.float32)
        actual = torch.tensor(Rotation.from_euler('xyz', [[0., 5., 0.], [0., 5., 90.], [0., -5., 0.]], degrees=True).as_quat(), dtype=torch.float32)
        env.reference = {'root_quat': ref}
        env.projected_gravity = quat_inverse_apply(actual, env.gravity_vec)
        reward = env._reward_gmr_trunk_tilt()
        torch.testing.assert_close(reward[:2], torch.ones(2))
        self.assertLess(float(reward[2]), .06)


if __name__ == '__main__':
    unittest.main()
