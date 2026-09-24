"""CPU contracts for the single-clip candidate, not a simulator acceptance test."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from humanoid.amp.scaled_experiment import ScaledExperiment, canonical_sha, validate_cloud_smoke
from humanoid.amp.adapter import AMPAlgorithmAdapter
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes, state_from_clip

REPO = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ScaledAMPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = ScaledExperiment(REPO)

    def test_single_clip_and_only_one_changed_column(self):
        e = self.experiment; z = e.clip.arrays
        changed = np.flatnonzero(np.any(z["qpos"] != z["parent_qpos"], axis=0))
        self.assertEqual(changed.tolist(), [7+e.clip.all_joint_names.index("left_hip_roll_joint")])
        self.assertEqual(e.diagnostics["angle_violations"], 0)
        self.assertEqual(e.diagnostics["velocity_violations"], 0)
        self.assertFalse(e.clip.metadata["training_ready"])

    def test_windows_and_normalizer_only_scaled02(self):
        e = self.experiment
        self.assertEqual(e.windows.shape, (590, 10, 39))
        torch.testing.assert_close(e.windows[0], e.clip.window(1), rtol=0, atol=0)
        torch.testing.assert_close(e.windows[-1], e.clip.window(590), rtol=0, atol=0)
        torch.testing.assert_close(e.mean, e.clip.features[1:].double().mean(0).float())
        batch, provenance = e.sample(512)
        self.assertEqual(batch.shape, (512, 10, 39))
        self.assertEqual({x["clip_id"] for x in provenance}, {e.clip.id})
        self.assertTrue(all(1 <= x["start_frame"] <= 590 and x["end_frame"] <= 599 for x in provenance))

    def test_wrong_identity_hash_and_holdout_rejected(self):
        for field, value in (("source_clip_id", "LAFAN_WALK_15"), ("candidate_id", "LAFAN_WALK_10"),
                             ("motion_sha256", "0"*64), ("training_ready", True)):
            with tempfile.TemporaryDirectory() as directory:
                cfg = copy.deepcopy(self.experiment.cfg); cfg[field] = value
                p = Path(directory)/"cfg.json"; p.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ScaledExperiment(REPO, p)

    def test_sampler_seed_reproducible(self):
        a, b = ScaledExperiment(REPO), ScaledExperiment(REPO)
        x, xp = a.sample(16); y, yp = b.sample(16)
        torch.testing.assert_close(x, y, rtol=0, atol=0); self.assertEqual(xp, yp)

    def test_cross_platform_text_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/"a", Path(directory)/"b"
            a.write_bytes(b"one\ntwo\n"); b.write_bytes(b"one\r\ntwo\r\n")
            self.assertEqual(canonical_sha(a), canonical_sha(b))

    def test_formal_gate_requires_real_smoke_identity_and_metrics(self):
        e = self.experiment
        with self.assertRaises(ValueError):
            validate_cloud_smoke(e, {}, "test-only")
        certificate = dict(identity=e.identity(), implementation_fingerprint="test-only", num_envs=32, updates=10,
            valid_windows=1, **{k: True for k in ("physics_dt_verified", "body_frames_verified", "reset_history_verified",
                "nonzero_style_reward", "actor_updated", "discriminator_updated", "all_finite", "complete")})
        self.assertTrue(validate_cloud_smoke(e, certificate, "test-only"))
        for key in ("complete", "body_frames_verified", "actor_updated", "reset_history_verified"):
            bad = dict(certificate); bad[key] = False
            with self.assertRaises(ValueError):
                validate_cloud_smoke(e, bad, "test-only")
        with self.assertRaises(ValueError):
            validate_cloud_smoke(e, certificate, "changed-code")

    def test_real_ppo_adapter_inference_mode_and_terminal_reset(self):
        torch.manual_seed(5)
        e = self.experiment; n = 4
        actor_class, ppo_class = cpu_ppo_classes(REPO)
        with contextlib.redirect_stdout(io.StringIO()):
            actor = actor_class(235, 47, 219, 12, in_channels=66, actor_hidden_dims=[32, 32],
                critic_hidden_dims=[32, 32], state_estimator_hidden_dims=[32, 16])
        ppo = ppo_class(actor, device="cpu", num_learning_epochs=1, num_mini_batches=1, lin_vel_idx=199)
        ppo.init_storage(n, 24, [3102], [219], [12])
        d = AMPDiscriminator(e.spec, e.mean, e.std, [16])
        trainer = DiscriminatorTrainer(d)
        bridge = AMPBridge(e.spec, trainer, n, e.cfg["reward"])
        episodes = torch.zeros(n, dtype=torch.long); ticks = torch.zeros(n, dtype=torch.long)
        frames = np.arange(n)*50
        def prime(ids):
            bridge.prime(state_from_clip(e.clip, frames), episodes, ticks, ids)
        env = SimpleNamespace(amp_prime=prime)
        adapter = AMPAlgorithmAdapter(ppo, bridge, env, e)
        prime(None)
        actor_before = [p.detach().clone() for p in actor.parameters()]
        d_before = [p.detach().clone() for p in d.parameters()]
        with torch.inference_mode():
            for k in range(1, 25):
                ppo.act(torch.randn(n, 3102), torch.randn(n, 219))
                frames += 1; ticks += 1
                bridge.capture_pre_reset(state_from_clip(e.clip, frames), episodes, ticks)
                rewards = torch.ones(n)*.01; dones = torch.zeros(n, dtype=torch.bool)
                if k == 13:
                    dones[0] = True; episodes[0] += 1; frames[0] = 300
                adapter.process_env_step(rewards, dones, {})
                if 13 <= k <= 22:
                    self.assertLess(bridge.stream.history.count[0], 10)
            ppo.compute_returns(torch.randn(n, 219))
        with contextlib.redirect_stdout(io.StringIO()):
            losses = adapter.update()
        self.assertTrue(all(np.isfinite(losses)))
        self.assertEqual(adapter.reset_checks, 1)
        self.assertGreater(adapter.valid_windows, 0)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(actor_before, actor.parameters())))
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(d_before, d.parameters())))
        self.assertTrue(torch.equal(d.mean, e.mean))


if __name__ == "__main__":
    unittest.main()
