import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from humanoid.joint_dynamics import (
    armature_values_for_dof_order,
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
    def test_x1_config_loads_and_nominal_is_inside_training_range(self):
        config = load_joint_armature_config(CONFIG_PATH)
        self.assertEqual(12, len(config["joint_order"]))
        for entry in config["joints"].values():
            low, high = entry["train_range"]
            self.assertLessEqual(low, entry["nominal"])
            self.assertLessEqual(entry["nominal"], high)

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


if __name__ == "__main__":
    unittest.main()
