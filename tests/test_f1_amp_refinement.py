"""Matched continuation and regularizers; CPU tests are not gait acceptance."""
import ast
import copy
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial import ConvexHull
import torch

from humanoid.amp.refinement import validate_refinement, robust_acceleration_cost, ground_force_weight, warm_start
from humanoid.amp.recovery import PolicyReplay, foot_collision_vertices
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from humanoid.amp.learnability import assert_no_domain_randomization
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/("configs/amp/lafan_walk02_refine_%s.json" % g))
                           for g in ("control", "smooth", "noamp")}

    def test_budget_no_dr_and_unchanged_observations_assets_pd(self):
        scope = config_scope()
        path = ROOT/"humanoid/envs/x1/x1_amp_refine_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), "exec"), scope)
        a, b = scope["X1AMPRecoveryCfg"](), scope["X1AMPRefineCfg"]()
        assert_no_domain_randomization(b)
        pa, pb = plain(a), plain(b)
        pa.pop("rewards"); pb.pop("rewards")
        self.assertEqual(pa, pb)
        for e in self.experiments.values():
            self.assertTrue(validate_refinement(e.cfg))
            bad = copy.deepcopy(e.cfg); bad["formal"]["updates"] = 1000
            with self.assertRaises(ValueError):
                validate_refinement(bad)

    def test_smooth_noamp_only_style_weight_differs(self):
        a, b = (copy.deepcopy(self.experiments[g].cfg) for g in ("smooth", "noamp"))
        self.assertEqual(a["reward"].pop("style_weight"), 1.)
        self.assertEqual(b["reward"].pop("style_weight"), 0.)
        for item in (a, b):
            item.pop("experiment"); item.pop("scope")
            item["refinement"].pop("group"); item["refinement"].pop("round")
        self.assertEqual(a, b)

    def test_acceleration_quadratic_near_zero_linear_tails(self):
        x = torch.tensor([[0.], [1.], [100.], [10000.]], requires_grad=True)
        cost = robust_acceleration_cost(x)
        self.assertEqual(float(cost[0]), 0.)
        self.assertAlmostEqual(float(cost[1]), .0001, delta=1e-6)
        self.assertTrue(bool((cost[1:] > cost[:-1]).all()))
        self.assertLess(float(cost[-1]), 201.)
        cost.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_force_has_no_light_contact_dead_band(self):
        force = torch.tensor([-1., 0., .1, 1., 5., 100.])
        w = ground_force_weight(force, torch.zeros_like(force))
        self.assertEqual(float(w[0]), 0.); self.assertEqual(float(w[1]), 0.)
        self.assertTrue(bool((w[2:] > 0).all()))
        self.assertTrue(bool((w[2:] < 1).all()))
        self.assertTrue(bool((ground_force_weight(force, torch.full_like(force, .03)) == 0).all()))

    def test_hull_preserves_all_direction_mesh_minimum(self):
        vertices = foot_collision_vertices(self.experiments["smooth"].kinematics.path)
        normals = np.random.default_rng(5).normal(size=(100, 3))
        for mesh in vertices.values():
            hull = mesh[ConvexHull(mesh).vertices]
            np.testing.assert_allclose((normals@mesh.T).min(1), (normals@hull.T).min(1), atol=1e-10)

    def test_replay_roundtrip_and_reject_wrong_capacity(self):
        replay = PolicyReplay(10)
        replay.add(torch.arange(12*10*39).reshape(12, 10, 39).float())
        restored = PolicyReplay(10); restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.cursor, replay.cursor)
        torch.testing.assert_close(restored.buffer, replay.buffer)
        with self.assertRaises(ValueError):
            PolicyReplay(20).load_state_dict(replay.state_dict())

    def test_readonly_actual_model1000_restore(self):
        checkpoint = ROOT.parent/"f1-amp-style-recovery/outputs/amp-recovery/TASK_20260924_060/model_1000.pt"
        if not checkpoint.is_file():
            self.skipTest("Local source artifact not installed; cloud smoke must prove warm start")
        e = self.experiments["smooth"]
        cfg = config_scope()["X1AMPRecoveryCfgPPO"]()
        actor_type, ppo_type = cpu_ppo_classes(ROOT)
        policy_cfg = plain(cfg.policy)
        with contextlib.redirect_stdout(io.StringIO()):
            actor = actor_type(235, 47, 219, 12, **policy_cfg)
        ppo = ppo_type(actor, device="cpu", **plain(cfg.algorithm))
        trainer = DiscriminatorTrainer(AMPDiscriminator(e.spec, e.mean, e.std))
        bridge = AMPBridge(e.spec, trainer, 2, e.cfg["reward"], "cpu", e.cfg["recovery"])
        alg = SimpleNamespace(actor_critic=actor, optimizer=ppo.optimizer,
            state_estimator_optimizer=ppo.state_estimator_optimizer, ppo=ppo, bridge=bridge)
        runner = SimpleNamespace(device="cpu", alg=alg, amp_trainer=trainer)
        result = warm_start(runner, e, checkpoint)
        self.assertEqual(result["replay_windows"], 50000)
        self.assertEqual(runner.current_learning_iteration, 1000)
        self.assertEqual(ppo.optimizer.param_groups[0]["lr"], 5e-5)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, state["model_state_dict"][name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
