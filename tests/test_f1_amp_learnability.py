"""No-DR and update-count contracts; not simulator or gait acceptance."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from humanoid.amp.learnability import (
    assert_no_domain_randomization, evaluation_checkpoint_name, validate_learnability_budget,
)
from humanoid.amp.scaled_experiment import ScaledExperiment

REPO = Path(__file__).resolve().parents[1]


def load_config():
    # Execute real config class bodies without eager Isaac Gym imports.
    scope = dict(inspect=inspect)
    for name in ("base/base_config.py", "base/legged_robot_config.py",
                 "x1/x1_dh_stand_config.py", "x1/x1_dh_stand_no_dr_config.py", "x1/x1_amp_config.py"):
        path = REPO/"humanoid/envs"/name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), "exec"), scope)
    return scope["X1AMPCfg"]()


def plain(obj):
    if isinstance(obj, (str, int, float, bool, dict, list, tuple)) or obj is None:
        return obj
    return {key: plain(getattr(obj, key)) for key in dir(obj)
            if not key.startswith("_") and not callable(getattr(obj, key))}


class LearnabilityTests(unittest.TestCase):
    def test_actual_amp_config_is_no_dr_and_never_unlocked(self):
        cfg = load_config()
        result = assert_no_domain_randomization(cfg)
        self.assertTrue(result["nominal_armature"])
        self.assertFalse(result["dr_unlocked"])
        self.assertEqual(result, assert_no_domain_randomization(plain(cfg)))

    def test_every_inherited_boolean_toggle_rejected(self):
        cfg = plain(load_config())
        for key, value in cfg["domain_rand"].items():
            if type(value) is bool and key != "use_nominal_joint_armature":
                with self.subTest(flag=key):
                    bad = copy.deepcopy(cfg); bad["domain_rand"][key] = True
                    with self.assertRaises(ValueError):
                        assert_no_domain_randomization(bad)

    def test_bad_types_missing_switch_and_future_toggle_rejected(self):
        for key, value in (("randomize_friction", 0), ("push_robots", 1),
                           ("future_randomization", True), ("randomize_new_parameter", "false"),
                           ("use_nominal_joint_armature", False)):
            cfg = plain(load_config()); cfg["domain_rand"][key] = value
            with self.assertRaises(ValueError):
                assert_no_domain_randomization(cfg)
        cfg = plain(load_config()); del cfg["domain_rand"]["push_robots"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            assert_no_domain_randomization(cfg)

    def test_noise_and_terrain_rejected(self):
        for group, key, value in (("noise", "add_noise", True), ("terrain", "mesh_type", "trimesh"),
                                   ("terrain", "curriculum", True)):
            cfg = plain(load_config()); cfg[group][key] = value
            with self.assertRaises(ValueError):
                assert_no_domain_randomization(cfg)

    def test_recovery_budget_and_original_preserved(self):
        original = ScaledExperiment(REPO)
        recovery = ScaledExperiment(REPO, REPO/"configs/amp/lafan_walk02_recovery.json")
        self.assertTrue(validate_learnability_budget(recovery.cfg))
        self.assertEqual(original.cfg["formal"]["updates"], 3000)
        self.assertEqual(recovery.cfg["formal"]["updates"], 1000)
        self.assertEqual(original.identity()["motion_sha256"], recovery.identity()["motion_sha256"])
        self.assertEqual(original.identity()["feature_fingerprint"], recovery.identity()["feature_fingerprint"])
        self.assertNotEqual(original.identity(), recovery.identity())

    def test_implicit_extension_and_skipping_checkpoint_rejected(self):
        cfg = json.loads((REPO/"configs/amp/lafan_walk02_recovery.json").read_text())
        for group, key, value in (("formal", "updates", 3000), ("smoke", "updates", 5)):
            bad = copy.deepcopy(cfg); bad[group][key] = value
            with self.assertRaises(ValueError):
                validate_learnability_budget(bad)
        for key, value in (("evaluation_updates", [1000]), ("learnability_only", False),
                           ("max_experiment_rounds", 21)):
            bad = copy.deepcopy(cfg); bad[key] = value
            with self.assertRaises(ValueError):
                validate_learnability_budget(bad)

    def test_runner_checkpoint_hook_exact_completed_updates(self):
        path = REPO/"humanoid/amp/runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        class StubParent:
            def log(self, *args):
                pass
        scope = dict(DHOnPolicyRunner=StubParent, Path=Path,
                     evaluation_checkpoint_name=evaluation_checkpoint_name)
        exec(compile(tree, str(path), "exec"), scope)
        cls = scope["AMPOnPolicyRunner"]
        runner = object.__new__(cls)
        runner.log_dir = "test-only"
        runner.alg = SimpleNamespace(latest={})
        runner.amp_experiment = SimpleNamespace(cfg={"evaluation_updates": [500, 1000]})
        saved = []
        runner.save = lambda path, infos: saved.append((Path(path).name, infos))
        for index in range(1000):
            runner.log({"it": index})
        self.assertEqual([row[0] for row in saved], ["model_eval_0500.pt", "model_eval_1000.pt"])
        self.assertTrue(all(row[1]["effectiveness_verified"] is False for row in saved))
        self.assertTrue(all(row[1]["dr_unlocked"] is False for row in saved))
        self.assertIsNone(evaluation_checkpoint_name(500, []))


if __name__ == "__main__":
    unittest.main()
