"""Recovery math and data flow; physical learnability still needs cloud evidence."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.recovery import (centered_velocity_reward, reference_initial_states,
                                  static_negative_windows, PolicyReplay, foot_collision_vertices)
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def config_scope():
    scope = dict(inspect=inspect)
    for name in ("base/base_config.py", "base/legged_robot_config.py", "x1/x1_dh_stand_config.py",
                 "x1/x1_dh_stand_no_dr_config.py", "x1/x1_amp_config.py", "x1/x1_amp_recovery_config.py"):
        path = ROOT/"humanoid/envs"/name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), "exec"), scope)
    return scope


def plain(obj):
    if isinstance(obj, (str, int, float, bool, dict, list, tuple)) or obj is None:
        return obj
    return {key: plain(getattr(obj, key)) for key in dir(obj)
            if not key.startswith("_") and not callable(getattr(obj, key))}


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = ScaledExperiment(ROOT, ROOT/"configs/amp/lafan_walk02_recovery.json")

    def test_centered_velocity_zero_at_rest_one_at_target(self):
        cmd = torch.tensor([[.45, 0.]]).repeat(6, 1)
        velocities = torch.tensor([[0., 0.], [.225, 0.], [.45, 0.], [.8, 0.], [-.1, 0.], [.45, .3]])
        r = centered_velocity_reward(velocities, cmd)
        self.assertAlmostEqual(float(r[0]), 0.)
        self.assertAlmostEqual(float(r[2]), 1.)
        self.assertGreater(r[1], r[0]); self.assertLess(r[3], r[2]); self.assertLess(r[4], 0.)
        self.assertLess(r[5], r[2])
        with self.assertRaises(ValueError):
            centered_velocity_reward(velocities, torch.zeros_like(cmd))

    def test_no_dr_nominal_assets_pd_actor_shapes_unchanged(self):
        scope = config_scope()
        before, after = scope["X1AMPCfg"](), scope["X1AMPRecoveryCfg"]()
        assert_no_domain_randomization(after)
        a, b = plain(before), plain(after)
        a.pop("rewards"); b.pop("rewards")
        self.assertEqual(a, b)
        scales = plain(after.rewards.scales)
        for key in ("tracking_lin_vel", "tracking_ang_vel", "orientation", "vel_mismatch_exp", "action_smoothness"):
            self.assertEqual(scales[key], 0.)
        self.assertFalse(after.rewards.only_positive_rewards)
        pa, pb = plain(scope["X1AMPCfgPPO"]()), plain(scope["X1AMPRecoveryCfgPPO"]())
        pb["policy"]["init_noise_std"] = pa["policy"]["init_noise_std"]
        for key in ("experiment_name", "max_iterations"):
            pb["runner"][key] = pa["runner"][key]
        self.assertEqual(pa, pb)

    def test_two_groups_differ_only_by_static_negative_fraction_and_identity(self):
        a = copy.deepcopy(self.e.cfg)
        b = json.loads((ROOT/"configs/amp/lafan_walk02_recovery_static.json").read_text())
        self.assertEqual(a["recovery"].pop("static_negative_fraction"), 0.)
        self.assertEqual(b["recovery"].pop("static_negative_fraction"), .25)
        for key in ("experiment", "scope"):
            a.pop(key); b.pop(key)
        self.assertEqual(a, b)

    def test_rsi_does_not_change_data_features_or_vertical_lift_velocity(self):
        before = self.e.clip.features.clone()
        result = reference_initial_states(self.e)
        self.assertEqual(result["root"].shape, (599, 13))
        self.assertEqual(result["frame_ids"].tolist(), list(range(1, 600)))
        self.assertTrue(torch.equal(before, self.e.clip.features))
        self.assertTrue(torch.equal(result["qd"], before[1:, 15:27]))
        self.assertTrue(torch.equal(result["q"], before[1:, 3:15]))
        self.assertTrue(torch.equal(result["root"][:, 9], torch.zeros(599)))
        self.assertTrue(torch.equal(result["root"][:, :2], torch.zeros(599, 2)))
        np.testing.assert_allclose(result["root"][:, 7:9], np.diff(self.e.clip.arrays["qpos"][:, :2], axis=0)*100.)

    def test_reset_collision_clearance_with_independent_mujoco_fk(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(self.e.kinematics.path)
        data = mujoco.MjData(model)
        state = reference_initial_states(self.e)
        mesh = foot_collision_vertices(self.e.kinematics.path)
        addresses = [model.joint(name).qposadr[0] for name in self.e.spec.joint_names]
        for frame in (0, 23, 151, 345, 598):
            data.qpos[addresses] = state["q"][frame].numpy()
            mujoco.mj_forward(model, data)
            r = Rotation.from_quat(state["root"][frame, 3:7]).as_matrix()
            height = float(state["root"][frame, 2])
            all_min = []
            for name, vertices in mesh.items():
                body = model.body(name).id
                local = vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
                all_min.append(float((local @ r.T)[:, 2].min()+height))
            self.assertAlmostEqual(min(all_min), .005, places=6)

    def test_static_negative_preserves_pose_but_zeros_motion(self):
        demo = self.e.windows[:8].clone()
        held = static_negative_windows(demo)
        self.assertTrue(torch.equal(demo, self.e.windows[:8]))
        self.assertTrue(torch.equal(held[:, 0], held[:, -1]))
        self.assertTrue(torch.equal(held[:, :, :3], torch.zeros_like(held[:, :, :3])))
        self.assertTrue(torch.equal(held[:, :, 15:27], torch.zeros_like(held[:, :, 15:27])))
        torch.testing.assert_close(held[:, -1, 3:15], demo[:, -1, 3:15])
        torch.testing.assert_close(held[:, -1, 27:], demo[:, -1, 27:])

    def test_replay_boundaries_detach_and_no_nan(self):
        replay = PolicyReplay(5)
        generator = torch.Generator().manual_seed(5)
        with self.assertRaises(ValueError):
            replay.sample(2, generator)
        for start in (0, 3, 6):
            x = torch.arange(start, start+3, dtype=torch.float32)[:, None, None].expand(3, 10, 39)
            replay.add(x)
        self.assertEqual(replay.count, 5)
        self.assertEqual(set(replay.buffer[:, 0, 0].tolist()), set([4., 5., 6., 7., 8.]))
        sample = replay.sample(20, generator); sample.zero_()
        self.assertGreater(float(replay.buffer.sum()), 0.)
        with self.assertRaises(ValueError):
            replay.add(torch.full((1, 10, 39), float("nan")))
        self.assertEqual(replay.state_dict()["count"], 5)

    def test_actual_discriminator_with_replay_and_auxiliary_negatives(self):
        for fraction in (0., .25):
            d = AMPDiscriminator(self.e.spec, self.e.mean, self.e.std, [16])
            trainer = DiscriminatorTrainer(d)
            cfg = dict(self.e.cfg["recovery"], replay_capacity=64, replay_insert=32,
                       static_negative_fraction=fraction)
            bridge = AMPBridge(self.e.spec, trainer, 4, self.e.cfg["reward"], recovery_config=cfg)
            before = [p.detach().clone() for p in d.parameters()]
            for step in range(2):
                with torch.inference_mode():
                    bridge.policy_batches.append(self.e.windows[:40].clone()+.01)
                result = bridge.update_discriminator(self.e, 32, step)
                self.assertEqual(result["static_negative_fraction"], fraction)
                self.assertEqual(len(bridge.policy_batches), 0)
                self.assertTrue(all(np.isfinite(v) for v in result.values()))
            self.assertEqual(bridge.replay.count, 64)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, d.parameters())))
            # Every stored value is a real input window, not a held-pose negative.
            self.assertTrue((bridge.replay.buffer[:, :, 15:27].abs().sum((1, 2)) > 0).all())

    def test_recovery_smoke_requires_rsi_replay_and_no_dr(self):
        cert = dict(identity=self.e.identity(), implementation_fingerprint="cpu-only", num_envs=32,
                    updates=10, valid_windows=1, rsi_reset_count=32, replay_window_count=1,
                    no_dr_configuration_verified=True, smoke_evaluation_verified=True, **{k: True for k in (
                    "physics_dt_verified", "body_frames_verified", "reset_history_verified", "nonzero_style_reward",
                    "actor_updated", "discriminator_updated", "all_finite", "complete")})
        self.assertTrue(validate_cloud_smoke(self.e, cert, "cpu-only"))
        for key, value in (("rsi_reset_count", 0), ("replay_window_count", 0),
                           ("no_dr_configuration_verified", False), ("smoke_evaluation_verified", False)):
            bad = dict(cert); bad[key] = value
            with self.assertRaises(ValueError):
                validate_cloud_smoke(self.e, bad, "cpu-only")

    def test_evaluation_excludes_reset_restarts_and_initial_velocity(self):
        path = ROOT/"humanoid/scripts/record_amp_policy.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "summarize"]
        scope = dict(np=np)
        exec(compile(tree, str(path), "exec"), scope)
        valid = np.ones((400, 2), dtype=bool); valid[50:, 1] = False
        velocity = np.zeros((400, 2, 3)); velocity[:150, :, 0] = .45
        data = dict(valid=valid, failure=np.zeros_like(valid), time=(np.arange(400)+1)*.01,
                    base_lin_vel=velocity, dof_pos=np.zeros((400, 2, 12)), initial_failure=np.zeros(2, dtype=bool))
        data["failure"][49, 1] = True
        summary = scope["summarize"](data, 4.)
        self.assertTrue(summary["episodes"][0]["survived"])
        self.assertEqual(summary["episodes"][0]["post_2s_vx_mean"], 0.)
        self.assertFalse(summary["episodes"][1]["survived"])
        self.assertIsNone(summary["episodes"][1]["post_2s_vx_mean"])
        self.assertFalse(summary["dr_unlocked"])

    def test_evaluator_buffers_remain_resettable_between_modes(self):
        path = ROOT/"humanoid/scripts/record_amp_policy.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        contexts = [node.items[0].context_expr for node in ast.walk(main) if isinstance(node, ast.With)]
        torch_context = next(node for node in contexts if isinstance(node, ast.Call) and
                             isinstance(node.func, ast.Attribute) and node.func.attr in ("no_grad", "inference_mode"))
        context = eval(compile(ast.Expression(torch_context), str(path), "eval"), {"torch": torch})
        with context:
            buffer = torch.ones(2)+1
        buffer[:] = 0.  # same reset boundary that failed in the real evaluator
        self.assertTrue(torch.equal(buffer, torch.zeros(2)))

    def test_reset_hooks_select_named_reference_and_preserve_velocity_history(self):
        path = ROOT/"humanoid/envs/x1/x1_amp_recovery_env.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        class FakeParent:
            def reset_idx(self, ids):
                self._reset_dofs(ids); self._reset_root_states(ids)
                self.last_dof_vel[ids] = 0.; self.last_root_vel[ids] = 0.
        scope = dict(X1AMPEnv=FakeParent, torch=torch, gymtorch=SimpleNamespace(unwrap_tensor=lambda x: x),
                     reference_initial_states=reference_initial_states)
        exec(compile(tree, str(path), "exec"), scope)
        env = scope["X1AMPRecoveryEnv"]()
        env.num_envs, env.device, env.dof_names = 4, "cpu", self.e.spec.joint_names
        env.dof_state = torch.zeros(4, 12, 2)
        env.dof_pos, env.dof_vel = env.dof_state[:, :, 0], env.dof_state[:, :, 1]
        env.root_states, env.env_origins = torch.zeros(4, 13), torch.tensor([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [6., 0., 0.]])
        env.last_dof_vel, env.last_root_vel = torch.zeros(4, 12), torch.zeros(4, 6)
        env.sim = None
        calls = []
        env.gym = SimpleNamespace(set_dof_state_tensor_indexed=lambda *args: calls.append(args),
                                  set_actor_root_state_tensor_indexed=lambda *args: calls.append(args))
        env.configure_reference_initialization(self.e)
        ids = torch.tensor([1, 3]); env.reset_idx(ids)
        torch.testing.assert_close(env.dof_pos[ids], env.rsi["q"][env.rsi_indices[ids]])
        torch.testing.assert_close(env.last_dof_vel[ids], env.dof_vel[ids])
        torch.testing.assert_close(env.last_root_vel[ids], env.root_states[ids, 7:13])
        torch.testing.assert_close(env.root_states[ids, :2], env.env_origins[ids, :2])
        self.assertEqual(env.rsi_reset_count, 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(env.dof_pos[0], torch.zeros(12)))


if __name__ == "__main__":
    unittest.main()
