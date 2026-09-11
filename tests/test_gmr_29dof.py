"""CPU integration/geometry gates for the 29-DOF experiment; not PhysX validation."""
import ast
from collections import deque
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from humanoid.gmr_motion import GMRMotion
from humanoid.joint_dynamics import load_joint_armature_config, armature_values_for_dof_order
from test_gmr_environment_math import quat_apply, quat_inverse_apply, quat_mul


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "resources/motions/gmr_kit317_foot_flat_29dof.npz"
URDF = ROOT / "resources/robots/f1_v1_4/urdf/F1_V1_4_29DOF.urdf"
JOINTS = [j for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]
NAMES = [j.get("name") for j in JOINTS]
sys.path.insert(0, str(ROOT / "humanoid/scripts"))
from prepare_gmr_reference import fk, origin, stl_vertices


def config_scope():
    scope = dict(inspect=inspect, json=json, Path=Path, LEGGED_GYM_ROOT_DIR=str(ROOT))
    for filename in ("base/base_config.py", "base/legged_robot_config.py", "x1/x1_dh_stand_config.py",
                     "x1/x1_dh_stand_no_dr_config.py", "x1/x1_gmr_clip_config.py", "x1/x1_gmr_29dof_config.py"):
        path = ROOT / "humanoid/envs" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), "exec"), scope)
    return scope


def environment_class():
    scope = dict(torch=torch, quat_apply=quat_apply, quat_mul=quat_mul,
                 quat_rotate_inverse=quat_inverse_apply)
    path = ROOT / "humanoid/envs/x1/x1_dh_stand_env.py"
    base = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef))
    observe = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "compute_observations")
    exec(compile(ast.Module(body=[observe], type_ignores=[]), str(path), "exec"), scope)
    scope["X1DHStandEnv"] = type("Base", (), {"compute_observations": scope["compute_observations"]})
    for filename in ("x1_gmr_clip_env.py", "x1_gmr_29dof_env.py"):
        path = ROOT / "humanoid/envs/x1" / filename
        definition = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef))
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), scope)
    return scope["X1GMR29DOFEnv"]


class WholeBodyReferenceTest(unittest.TestCase):
    def setUp(self):
        self.clip = GMRMotion(PATH, NAMES, expected_dofs=29)

    def test_name_mapping_and_default_12_guard(self):
        self.assertEqual(self.clip.data["dof_pos"].shape, (139, 29))
        reverse = GMRMotion(PATH, NAMES[::-1], expected_dofs=29)
        torch.testing.assert_close(self.clip.data["dof_pos"].flip(1), reverse.data["dof_pos"])
        with self.assertRaises(ValueError):
            GMRMotion(PATH, NAMES)
        with self.assertRaises(ValueError):
            GMRMotion(PATH, NAMES[:-1] + [NAMES[0]], expected_dofs=29)

    def test_data_hash_and_model_identity(self):
        meta = json.loads(PATH.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(meta["output_sha256"], hashlib.sha256(PATH.read_bytes()).hexdigest())
        self.assertEqual(meta["training_urdf_lf_sha256"], hashlib.sha256(URDF.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
        self.assertLess(meta["source_fk_position_error_m"], 1e-6)
        self.assertLess(meta["source_fk_rotation_error_rad"], 1e-5)
        self.assertLess(meta["standing_max_sole_tilt_deg"], .01)

    def test_position_and_interpolation_velocity_limits(self):
        clip = self.clip
        limits = torch.tensor([float(j.find("limit").get("velocity")) for j in JOINTS])
        self.assertTrue(torch.all(clip.data["dof_pos"] >= clip.lower - 1e-6))
        self.assertTrue(torch.all(clip.data["dof_pos"] <= clip.upper + 1e-6))
        self.assertTrue(torch.all(torch.diff(clip.data["dof_pos"], dim=0).abs() * clip.fps <= limits + 1e-5))
        self.assertTrue(torch.all(clip.data["dof_vel"].abs() <= limits + 1e-5))

    def test_body_interpolation_and_clamped_time(self):
        states = self.clip.sample(torch.tensor([0., .017, 4.6, 6.]))
        self.assertEqual(states["body_pos_base"].shape, (4, 3, 3))
        torch.testing.assert_close(torch.linalg.vector_norm(states["body_quat_base"], dim=-1), torch.ones(4, 3))
        torch.testing.assert_close(states["body_pos_base"][2], states["body_pos_base"][3])

    def test_all_exported_body_targets_match_independent_fk(self):
        joints = ET.parse(URDF).getroot().findall("joint")
        for frame, q in enumerate(self.clip.data["dof_pos"].numpy()):
            transforms = fk(joints, dict(zip(NAMES, q)))
            for body, name in enumerate(self.clip.tracking_body_names):
                actual = transforms[name]
                target_p = self.clip.data["body_pos_base"][frame, body].numpy()
                target_R = Rotation.from_quat(self.clip.data["body_quat_base"][frame, body].numpy()).as_matrix()
                self.assertLess(np.linalg.norm(actual[:3, 3] - target_p), 1e-6)
                self.assertLess(np.linalg.norm(actual[:3, :3] - target_R), 1e-6)

    def test_dense_100hz_sole_mesh_clearance(self):
        robot = ET.parse(URDF).getroot()
        meshes = []
        for name in self.clip.metadata["foot_names"]:
            collision = robot.find("link[@name='%s']/collision" % name)
            mesh = collision.find("geometry/mesh")
            local = origin(collision.find("origin"))
            vertices = stl_vertices(URDF.parent / mesh.get("filename"))
            vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            meshes.append(vertices @ local[:3, :3].T + local[:3, 3])
        sample = self.clip.sample(torch.linspace(0., 4.6, 461))
        minimum = np.inf
        for i, q in enumerate(sample["dof_pos"].numpy()):
            transforms = fk(robot.findall("joint"), dict(zip(NAMES, q)))
            rotation = Rotation.from_quat(sample["root_quat"][i].numpy()).as_matrix()
            for name, vertices in zip(self.clip.metadata["foot_names"], meshes):
                T = transforms[name]
                world = (vertices @ T[:3, :3].T + T[:3, 3]) @ rotation.T + sample["root_pos"][i].numpy()
                minimum = min(minimum, world[:, 2].min())
        self.assertGreater(minimum, 0., "Interpolated reference penetrates the floor")

    def test_armature_and_pd_cover_every_joint_exactly(self):
        scope = config_scope()
        cfg = scope["X1GMR29DOFCfg"]()
        armature = load_joint_armature_config(cfg.domain_rand.joint_armature_config_file)
        values, _ = armature_values_for_dof_order(armature, NAMES)
        self.assertEqual(len(values), 29)
        self.assertTrue(all(v > 0 for v in values))
        self.assertEqual(set(cfg.init_state.default_joint_angles), set(NAMES))
        for name in NAMES:
            keys = [key for key in cfg.control.stiffness if key in name]
            self.assertEqual(len(keys), 1, name)
            self.assertGreater(cfg.control.stiffness[keys[0]], 0)
            self.assertGreater(cfg.control.damping[keys[0]], 0)
        self.assertFalse(cfg.noise.add_noise)
        self.assertFalse(cfg.domain_rand.randomize_joint_armature)
        self.assertFalse(cfg.domain_rand.add_lag)
        self.assertFalse(scope["X1GMR29DOFCfgPPO"]().runner.resume)

    def test_dimensions_and_real_network_forward_backward(self):
        scope = config_scope()
        cfg, ppo = scope["X1GMR29DOFCfg"](), scope["X1GMR29DOFCfgPPO"]()
        self.assertEqual((cfg.env.num_single_obs, cfg.env.single_num_privileged_obs), (98, 141))
        self.assertEqual(ppo.algorithm.lin_vel_idx, 403)
        spec = importlib.util.spec_from_file_location("whole_body_actor", ROOT / "humanoid/algo/ppo/actor_critic_dh.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy_args = {key: getattr(ppo.policy, key) for key in dir(ppo.policy) if not key.startswith("_")}
        with contextlib.redirect_stdout(io.StringIO()):
            net = module.ActorCriticDH(490, 98, 423, 29, **policy_args)
        obs, critic = torch.randn(2, 6468), torch.randn(2, 423)
        actions, value = net.act_inference(obs), net.evaluate(critic)
        self.assertEqual(actions.shape, (2, 29))
        self.assertEqual(value.shape, (2, 1))
        (actions.square().mean() + value.square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
        self.assertEqual(net.actor[-1].weight.shape[0], 29)


class WholeBodyRewardTest(unittest.TestCase):
    def setUp(self):
        cls = environment_class()
        self.e = e = cls.__new__(cls)
        e.num_envs, e.num_actions, e.dt = 3, 29, .01
        e.episode_length_buf = torch.tensor([1, 2, 3])
        e.actions = torch.ones(3, 29)
        e.last_actions = torch.zeros(3, 29)
        e.last_last_actions = torch.zeros(3, 29)
        e.torques = torch.ones(3, 29) * 2.
        e.last_torques = torch.ones(3, 29)
        e.torque_limits = torch.ones(29) * 10.
        e.dof_pos = torch.zeros(3, 29)
        e.ref_dof_pos = torch.zeros(3, 29)
        e.joint_groups = {"leg": list(range(12)), "upper": list(range(12, 24)), "waist": [24, 25, 26], "neck": [27, 28]}

    def test_action_difference_and_reset_history_gate(self):
        torch.testing.assert_close(self.e._reward_action_smoothness(), torch.tensor([0., 1., 2.]))
        self.e.last_actions[:] = 1.
        self.e.last_last_actions[:] = 1.
        torch.testing.assert_close(self.e._reward_action_smoothness(), torch.zeros(3))

    def test_torque_rate_units_and_reset_history_gate(self):
        torch.testing.assert_close(self.e._reward_gmr_torque_rate(), torch.tensor([0., 100., 100.]))
        self.e.last_torques[:] = self.e.torques
        torch.testing.assert_close(self.e._reward_gmr_torque_rate(), torch.zeros(3))

    def test_upper_body_does_not_dilute_leg_reward(self):
        for group in self.e.joint_groups:
            torch.testing.assert_close(self.e._joint_group_reward(group), torch.ones(3))
        self.e.dof_pos[:, :12] = .2
        torch.testing.assert_close(self.e._reward_gmr_joint_pos(), torch.full((3,), np.exp(-1.)))
        torch.testing.assert_close(self.e._reward_gmr_upper_joint_pos(), torch.ones(3))

    def test_chest_rotation_is_not_pelvis_rotation(self):
        e = self.e
        e.reference = {"root_quat": torch.tensor([[0., 0., 0., 1.]]).repeat(3, 1),
                       "body_quat_base": torch.tensor([[[0., 0., 0., 1.]]]).repeat(3, 3, 1)}
        e.tracking_body_ids = torch.tensor([0, 1, 2])
        e.rigid_state = torch.zeros(3, 3, 13)
        e.rigid_state[:, :, 6] = 1.
        torch.testing.assert_close(e._reward_gmr_chest_rotation(), torch.ones(3))
        e.rigid_state[:, 0, 3:7] = torch.tensor([0., .1, 0., np.sqrt(.99)])
        self.assertTrue(torch.all(e._reward_gmr_chest_rotation() < .38))

    def test_actual_observation_builder_and_velocity_target_slice(self):
        e = self.e
        scope = config_scope()
        e.cfg = scope["X1GMR29DOFCfg"]()
        e.motion = GMRMotion(PATH, NAMES, expected_dofs=29)
        e.reference_start = torch.zeros(3)
        e.default_dof_pos = torch.tensor([[e.cfg.init_state.default_joint_angles[n] for n in NAMES]])
        e.default_joint_pd_target = e.default_dof_pos.clone()
        e.dof_vel = torch.zeros(3, 29)
        e.commands = torch.zeros(3, 4)
        e.commands_scale = torch.ones(3)
        e.obs_scales = SimpleNamespace(dof_pos=1., dof_vel=1., lin_vel=1., ang_vel=1., quat=1.)
        e.base_lin_vel = torch.tensor([[1., 2., 3.]]).repeat(3, 1)
        e.base_ang_vel = torch.zeros(3, 3)
        e.base_euler_xyz = torch.zeros(3, 3)
        e.feet_indices = torch.tensor([0, 1])
        e.contact_forces = torch.zeros(3, 2, 3)
        e.rand_push_force = torch.zeros(3, 3)
        e.rand_push_torque = torch.zeros(3, 3)
        e.env_frictions = torch.ones(3, 1) * .6
        e.body_mass = torch.ones(3, 1) * 5.
        e.add_noise = False
        e.obs_history = deque([torch.zeros(3, 98) for _ in range(66)], maxlen=66)
        e.critic_history = deque([torch.zeros(3, 141) for _ in range(3)], maxlen=3)
        e.compute_observations()
        self.assertEqual(e.obs_buf.shape, (3, 6468))
        self.assertEqual(e.privileged_obs_buf.shape, (3, 423))
        torch.testing.assert_close(e.privileged_obs_buf[:, 403:406], e.base_lin_vel)


if __name__ == "__main__":
    unittest.main()
