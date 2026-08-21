import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace

from humanoid.joint_dynamics import (
    armature_values_for_dof_order,
    configure_inference_armature,
    configure_nominal_inference_environment,
    load_joint_armature_config,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "resources", "robots", "x1", "config", "joint_dynamics.json"
)
URDF_PATH = os.path.join(
    PROJECT_ROOT, "resources", "robots", "x1", "urdf", "X1_12DOF.urdf"
)
MJCF_ROBOT_PATH = os.path.join(
    PROJECT_ROOT,
    "resources",
    "robots",
    "x1",
    "mjcf",
    "robot",
    "xyber_x1",
    "X1_12DOF.xml",
)


class JointDynamicsConfigTest(unittest.TestCase):
    @staticmethod
    def _env_cfg(randomize=True, use_nominal=True, config_path=CONFIG_PATH):
        return SimpleNamespace(
            domain_rand=SimpleNamespace(
                randomize_joint_armature=randomize,
                use_nominal_joint_armature=use_nominal,
                joint_armature_config_file=config_path,
            ),
            asset=SimpleNamespace(armature=0.75),
        )

    def test_x1_config_loads_and_nominal_is_inside_training_range(self):
        config = load_joint_armature_config(CONFIG_PATH)
        self.assertEqual(12, len(config["joint_order"]))
        for entry in config["joints"].values():
            low, high = entry["train_range"]
            self.assertLessEqual(low, entry["nominal"])
            self.assertLessEqual(entry["nominal"], high)

    def test_values_match_all_parameter_branch_centers_and_ranges(self):
        joints = load_joint_armature_config(CONFIG_PATH)["joints"]
        expected = {
            "hip_pitch_joint": (0.208, (0.1664, 0.2496)),
            "hip_roll_joint": (0.025, (0.0001, 0.05)),
            "hip_yaw_joint": (0.0148, (0.01184, 0.01776)),
            "knee_pitch_joint": (0.2728, (0.21824, 0.32736)),
            "ankle_pitch_joint": (0.15, (0.12, 0.18)),
            "ankle_roll_joint": (0.035, (0.028, 0.042)),
        }
        for side in ("left", "right"):
            for suffix, (nominal, train_range) in expected.items():
                entry = joints[f"{side}_{suffix}"]
                self.assertEqual(nominal, entry["nominal"])
                self.assertEqual(train_range, entry["train_range"])

    def test_values_are_mapped_by_joint_name_not_json_position(self):
        config = load_joint_armature_config(CONFIG_PATH)
        reversed_names = tuple(reversed(config["joint_order"]))
        nominal, train_ranges = armature_values_for_dof_order(config, reversed_names)
        for index, joint_name in enumerate(reversed_names):
            self.assertEqual(config["joints"][joint_name]["nominal"], nominal[index])
            self.assertEqual(config["joints"][joint_name]["train_range"], train_ranges[index])

    def test_config_matches_x1_urdf_actuated_joints(self):
        config = load_joint_armature_config(CONFIG_PATH)
        urdf_root = ET.parse(URDF_PATH).getroot()
        urdf_joint_names = {
            joint.attrib["name"]
            for joint in urdf_root.findall("joint")
            if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}
        }
        self.assertEqual(set(config["joint_order"]), urdf_joint_names)

    def test_config_matches_x1_mujoco_joints(self):
        config = load_joint_armature_config(CONFIG_PATH)
        mjcf_root = ET.parse(MJCF_ROBOT_PATH).getroot()
        mujoco_joint_names = {
            joint.attrib["name"]
            for joint in mjcf_root.findall(".//joint")
            if "name" in joint.attrib
        }
        self.assertEqual(set(config["joint_order"]), mujoco_joint_names)

    def test_mismatched_joint_set_is_rejected(self):
        config = load_joint_armature_config(CONFIG_PATH)
        with self.assertRaisesRegex(ValueError, "does not match simulator DOFs"):
            armature_values_for_dof_order(config, config["joint_order"][:-1])

    def test_nominal_outside_range_is_rejected(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            bad_config = json.load(config_file)
        bad_config["armature"]["joints"]["left_ankle_pitch_joint"]["nominal"] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = os.path.join(temp_dir, "bad.json")
            with open(bad_path, "w", encoding="utf-8") as config_file:
                json.dump(bad_config, config_file)
            with self.assertRaisesRegex(ValueError, "outside its train_range"):
                load_joint_armature_config(bad_path)

    def test_nominal_inference_uses_shared_config_without_randomization(self):
        env_cfg = self._env_cfg(randomize=True, use_nominal=False)
        self.assertEqual(
            "nominal", configure_inference_armature(env_cfg, "nominal")
        )
        self.assertTrue(env_cfg.domain_rand.use_nominal_joint_armature)
        self.assertFalse(env_cfg.domain_rand.randomize_joint_armature)

    def test_non_nominal_inference_modes_are_rejected(self):
        for mode in ("zero", "training"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "unsupported armature"):
                    configure_inference_armature(self._env_cfg(), mode)

    def test_standard_inference_uses_deterministic_nominal_environment(self):
        env_cfg = self._env_cfg(randomize=True, use_nominal=False)
        env_cfg.terrain = SimpleNamespace(
            mesh_type="trimesh",
            curriculum=True,
            measure_heights=True,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.5,
        )
        env_cfg.noise = SimpleNamespace(add_noise=True, curriculum=True)
        env_cfg.domain_rand.push_robots = True
        env_cfg.domain_rand.randomize_base_mass = True
        env_cfg.domain_rand.randomize_coulomb_friction = True
        env_cfg.domain_rand.add_lag = True
        env_cfg.domain_rand.enable_delivery = True

        self.assertEqual(
            "nominal", configure_nominal_inference_environment(env_cfg)
        )
        self.assertEqual("plane", env_cfg.terrain.mesh_type)
        self.assertEqual(0.6, env_cfg.terrain.static_friction)
        self.assertEqual(0.6, env_cfg.terrain.dynamic_friction)
        self.assertEqual(0.0, env_cfg.terrain.restitution)
        self.assertFalse(env_cfg.noise.add_noise)
        self.assertFalse(env_cfg.domain_rand.push_robots)
        self.assertFalse(env_cfg.domain_rand.randomize_base_mass)
        self.assertFalse(env_cfg.domain_rand.randomize_coulomb_friction)
        self.assertFalse(env_cfg.domain_rand.add_lag)
        self.assertFalse(env_cfg.domain_rand.enable_delivery)
        self.assertTrue(env_cfg.domain_rand.use_nominal_joint_armature)
        self.assertFalse(env_cfg.domain_rand.randomize_joint_armature)


if __name__ == "__main__":
    unittest.main()
