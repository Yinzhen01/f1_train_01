"""CPU-only AMP contract tests, including all ten immutable local source clips."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from humanoid.amp.dataset import AMPDataset, Clip, TRAIN_IDS, load_config, sha256
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.dry_run import run_cpu_dry_run, state_from_clip
from humanoid.amp.features import (FeatureSpec, angular_velocity_body, build_features,
                                   from_world_state, name_indices, normalize_quat,
                                   quat_mul, quat_rotate, wxyz_to_xyzw, xyzw_to_wxyz)
from humanoid.amp.history import AMPHistory, PolicyAMPStream
from humanoid.amp.integration import AMPBridge, assert_formal_training_allowed
from humanoid.amp.quality import SourceModelAudit, derivative_metrics

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("F1_AMP_DATA_ROOT", r"F:\robot_f1\outputs\motion_dataset_visual_review_20260923\f1_retarget_100hz_20260923"))
CONFIG = REPO / "configs/amp/f1_100hz.json"
CFG = load_config(CONFIG)
SPEC = FeatureSpec(tuple(CFG["joint_names"]), tuple(CFG["body_names"]))
torch.set_num_threads(1)


class FeatureTests(unittest.TestCase):
    def test_named_joint_and_body_permutations(self):
        torch.manual_seed(1)
        w, q, dq, b = torch.randn(3, 3), torch.randn(3, 12), torch.randn(3, 12), torch.randn(3, 4, 3)
        ji, bi = torch.randperm(12), torch.randperm(4)
        expected = build_features(SPEC, w, q, dq, SPEC.joint_names, b, SPEC.body_names)
        actual = build_features(SPEC, w, q[:, ji], dq[:, ji], tuple(SPEC.joint_names[i] for i in ji), b[:, bi], tuple(SPEC.body_names[i] for i in bi))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (3, 39))

    def test_bad_names_and_shapes_rejected(self):
        with self.assertRaises(ValueError):
            name_indices(["a", "a"], ["a"])
        with self.assertRaises(ValueError):
            name_indices(["a"], ["b"])
        with self.assertRaises(ValueError):
            FeatureSpec(SPEC.joint_names, SPEC.body_names, fps=30)

    def test_quaternion_order_normalization_and_invalid(self):
        q = torch.tensor([[2., 0, 0, 0]])
        xyzw = wxyz_to_xyzw(q)
        torch.testing.assert_close(xyzw, torch.tensor([[0., 0, 0, 1.]]))
        torch.testing.assert_close(xyzw_to_wxyz(xyzw), q / 2)
        for bad in (torch.zeros(1, 4), torch.full((1, 4), float("nan"))):
            with self.assertRaises(ValueError):
                normalize_quat(bad)

    def test_quaternion_rate_sign_and_pi_boundary(self):
        from scipy.spatial.transform import Rotation
        q = torch.tensor(Rotation.from_euler("z", [179., 181.], degrees=True).as_quat())
        rate = angular_velocity_body(q[:1], q[1:])
        torch.testing.assert_close(rate, torch.tensor([[0., 0., np.deg2rad(2) / .01]]), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(angular_velocity_body(q[:1], -q[1:]), rate)
        torch.testing.assert_close(angular_velocity_body(q, -q), torch.zeros_like(q[:, :3]), rtol=0, atol=1e-10)

    def test_no_world_position_root_height_or_heading_leakage(self):
        from scipy.spatial.transform import Rotation
        root = torch.randn(2, 3)
        quat = normalize_quat(torch.randn(2, 4))
        kb, w, q, dq = torch.randn(2, 4, 3), torch.randn(2, 3), torch.randn(2, 12), torch.randn(2, 12)
        kw = root[:, None] + quat_rotate(quat[:, None], kb)
        a = from_world_state(SPEC, quat, root, w, q, dq, SPEC.joint_names, kw, SPEC.body_names)
        yaw = torch.tensor(Rotation.from_euler("z", [1.2, -2.]).as_quat(), dtype=torch.float32)
        shift = torch.tensor([[4., -3., 15.], [-2., 3., 10.]])
        b = from_world_state(SPEC, quat_mul(yaw, quat), quat_rotate(yaw, root) + shift,
                             w, q, dq, SPEC.joint_names, quat_rotate(yaw[:, None], kw) + shift[:, None], SPEC.body_names)
        torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-5)

    def test_history_mask_no_padding(self):
        history = AMPHistory(2, SPEC)
        episodes = torch.zeros(2, dtype=torch.long)
        for k in range(1, 11):
            windows, valid = history.append(torch.full((2, 39), float(k)), episodes, torch.full((2,), k))
            self.assertEqual(len(windows), 0 if k < 10 else 2)
            self.assertEqual(bool(valid.all()), k == 10)
        torch.testing.assert_close(windows[0, :, 0], torch.arange(1, 11).float())
        self.assertEqual(windows.shape, (2, 10, 39))

    def test_history_reset_and_gap_never_cross(self):
        h = AMPHistory(2, SPEC)
        for k in range(10):
            h.append(torch.ones(2, 39) * k, torch.zeros(2, dtype=torch.long), torch.full((2,), k))
        windows, valid = h.append(torch.ones(2, 39), torch.tensor([1, 0]), torch.tensor([0, 10]))
        self.assertEqual(valid.tolist(), [False, True])
        self.assertEqual(len(windows), 1)
        windows, valid = h.append(torch.ones(2, 39), torch.tensor([1, 0]), torch.tensor([1, 12]))
        self.assertFalse(valid.any())
        self.assertEqual(len(windows), 0)

    def test_continuous_lsgan_no_sigmoid(self):
        d = AMPDiscriminator(SPEC, torch.zeros(39), torch.ones(39), [8])
        for p in d.parameters():
            p.data.zero_()
        x = torch.randn(4, 10, 39)
        self.assertEqual(d(x).shape, (4, 1))
        loss = d.losses(x, x)
        self.assertAlmostEqual(float(loss["lsgan"]), 1.)
        self.assertAlmostEqual(float(loss["gradient_penalty"]), 0.)
        torch.testing.assert_close(d.style_reward(x), torch.full((4,), .0075))
        self.assertFalse(any(isinstance(m, torch.nn.Sigmoid) for m in d.modules()))
        with torch.no_grad():
            d.network[-1].bias.fill_(2)
        self.assertTrue((d(x) > 1).all())

    def test_discriminator_gradient_isolation_and_frozen_norm(self):
        d = AMPDiscriminator(SPEC, torch.zeros(39), torch.ones(39), [8])
        trainer = DiscriminatorTrainer(d)
        policy = torch.randn(3, 10, 39, requires_grad=True)
        mean, std = d.mean.clone(), d.std.clone()
        losses = trainer.step(torch.randn_like(policy), policy)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        self.assertIsNone(policy.grad)
        self.assertFalse(d.style_reward(policy).requires_grad)
        torch.testing.assert_close(mean, d.mean, rtol=0, atol=0)
        torch.testing.assert_close(std, d.std, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            d(torch.full((1, 10, 39), float("nan")))

    def test_discrete_derivative_metrics(self):
        t = np.arange(50) * .01
        result = derivative_metrics(t ** 3)
        self.assertAlmostEqual(result["jerk"]["abs_peak"], 6., places=7)

    def test_config_split_and_hold_guards(self):
        for key, value in (("allowed_train_ids", list(TRAIN_IDS) + ["LAFAN_WALK_15"]),
                           ("permanent_hold_ids", []), ("training_ready", True)):
            cfg = copy.deepcopy(CFG)
            cfg[key] = value
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory) / "invalid.json"
                p.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(p)


@unittest.skipUnless(DATA.is_dir(), "Set F1_AMP_DATA_ROOT to the approved 10-clip dataset")
class ActualDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = AMPDataset(DATA, REPO, CONFIG)

    def test_all_ten_actual_npz_fields_shapes_metadata_hashes(self):
        self.assertEqual(len(self.dataset.clips), 10)
        self.assertEqual(sum(c.frames for c in self.dataset.clips.values()), 5602)
        for cid, c in self.dataset.clips.items():
            self.assertEqual(c.arrays["qpos"].shape, (401 if cid.startswith("R2") else 600, 36))
            self.assertEqual(c.features.shape, (c.frames, 39))
            self.assertEqual(float(c.arrays["fps"]), 100)
            self.assertEqual(c.metadata["quaternion_order"], "wxyz")
            self.assertFalse(c.metadata["cyclic"])
            self.assertFalse(c.metadata["training_ready"])
            self.assertEqual(sha256(c.path), c.record["npz_sha256"])
        self.dataset.verify_unchanged()

    def test_named_mapping_independent_of_npz_joint_order(self):
        c = self.dataset.clips[TRAIN_IDS[0]]
        z = {k: v.copy() for k, v in c.arrays.items()}
        order = np.random.default_rng(7).permutation(29)
        z["joint_names"] = z["joint_names"][order]
        for k in ("qpos", "raw_qpos", "pre_lift_qpos", "clearance_only_qpos"):
            z[k] = np.concatenate((z[k][:, :7], z[k][:, 7:][:, order]), axis=1)
        leg_order = np.arange(12)[::-1]
        z["leg_joint_names"] = z["leg_joint_names"][leg_order]
        z["leg_dof_pos"] = z["leg_dof_pos"][:, leg_order]
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "motion.npz"
            np.savez(p, **z)
            record = dict(c.record, npz_sha256=sha256(p))
            other = Clip(p, record, SPEC, self.dataset.kinematics)
            torch.testing.assert_close(other.features, c.features, rtol=0, atol=0)

    def test_windows_span_bounds_and_no_loop(self):
        for c in self.dataset.clips.values():
            for start in (c.first_start, c.last_start):
                self.assertEqual(c.window(start).shape, (10, 39))
                self.assertAlmostEqual(float(c.arrays["time_s"][start + 9] - c.arrays["time_s"][start]), .09)
            for bad in (-1, 0, c.last_start + 1, c.frames):
                with self.assertRaises(ValueError):
                    c.window(bad)

    def test_reproducible_sampler_and_provenance_no_cross_clip(self):
        a, ma = self.dataset.sampler(seed=8).sample(256)
        b, mb = self.dataset.sampler(seed=8).sample(256)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(ma, mb)
        for i, record in enumerate(ma):
            self.assertIn(record["clip_id"], TRAIN_IDS)
            torch.testing.assert_close(a[i], self.dataset.clips[record["clip_id"]].window(record["start_frame"]), rtol=0, atol=0)
            self.assertEqual(record["end_frame"] - record["start_frame"], 9)

    def test_holdout_hold_reserve_and_stand_rejected(self):
        for cid in ("LAFAN_WALK_15", "LAFAN_WALK_10", "LAFAN_WALK_08", "LAFAN_WALK_09", "R2_009", "R2_010"):
            with self.assertRaises(ValueError):
                self.dataset.sampler(ids=TRAIN_IDS + (cid,))
            with self.assertRaises(ValueError):
                self.dataset.fit_normalization(ids=TRAIN_IDS + (cid,))
        with self.assertRaises(ValueError):
            self.dataset.sampler(purpose="training")
        with self.assertRaises(RuntimeError):
            assert_formal_training_allowed(self.dataset)

    def test_frozen_normalizer_train_only(self):
        mean, std = self.dataset.fit_normalization()
        x = torch.cat([self.dataset.clips[c].features[1:] for c in TRAIN_IDS]).double()
        torch.testing.assert_close(mean, x.mean(0).float())
        torch.testing.assert_close(std, x.std(0, unbiased=False).clamp_min(.0001).float())
        original = self.dataset.clips["LAFAN_WALK_15"].features
        try:
            self.dataset.clips["LAFAN_WALK_15"].features = original + 1e5
            mean2, std2 = self.dataset.fit_normalization()
            torch.testing.assert_close(mean, mean2, rtol=0, atol=0)
            torch.testing.assert_close(std, std2, rtol=0, atol=0)
        finally:
            self.dataset.clips["LAFAN_WALK_15"].features = original

    def test_explicit_duration_weighting(self):
        for mode in ("clip", "duration"):
            sampler = self.dataset.sampler(weighting=mode)
            np.testing.assert_allclose(sampler.probabilities, [.25] * 4)
        with self.assertRaises(ValueError):
            self.dataset.sampler(weighting="frame_count")

    def test_policy_shared_features_reset_and_missing_tick(self):
        c = self.dataset.clips[TRAIN_IDS[0]]
        stream = PolicyAMPStream(2, SPEC)
        ids = torch.zeros(2, dtype=torch.long)
        steps = torch.zeros(2, dtype=torch.long)
        with self.assertRaises(ValueError):
            stream.observe_pre_reset(state_from_clip(c, [1, 1]), ids, steps + 1)
        stream.prime(state_from_clip(c, [0, 0]), ids, steps)
        for k in range(1, 11):
            windows, valid = stream.observe_pre_reset(state_from_clip(c, [k, k]), ids, steps + k)
            self.assertEqual(bool(valid.all()), k == 10)
        torch.testing.assert_close(windows[0], c.window(1), atol=2e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            stream.observe_pre_reset(state_from_clip(c, [12, 12]), ids, steps + 12)
        ids[0] = 1
        reset_ticks = torch.tensor([0, 10])
        stream.prime(state_from_clip(c, [100, 10]), ids, reset_ticks, torch.tensor([0]))
        windows, valid = stream.observe_pre_reset(state_from_clip(c, [101, 11]), ids, reset_ticks + 1)
        self.assertEqual(valid.tolist(), [False, True])

    def test_terminal_capture_not_post_reset(self):
        class Recorder:
            def process_env_step(self, rewards, dones, infos):
                self.reward = rewards
        c = self.dataset.clips[TRAIN_IDS[0]]
        mean, std = self.dataset.fit_normalization()
        bridge = AMPBridge(SPEC, DiscriminatorTrainer(AMPDiscriminator(SPEC, mean, std, [8])), 1, CFG["reward"])
        recorder = Recorder()
        ep = torch.zeros(1, dtype=torch.long)
        bridge.prime(state_from_clip(c, [0]), ep, ep)
        with self.assertRaises(RuntimeError):
            bridge.process_env_step(recorder, torch.ones(1), torch.zeros(1, dtype=torch.bool), {})
        for k in range(1, 11):
            bridge.capture_pre_reset(state_from_clip(c, [k]), ep, torch.tensor([k]))
            r = bridge.process_env_step(recorder, torch.ones(1), torch.tensor([k == 10]), {})
        self.assertTrue(r["valid_mask"].item())
        self.assertFalse(bridge.stream.primed.any())
        with self.assertRaises(ValueError):
            bridge.capture_pre_reset(state_from_clip(c, [0]), ep + 1, ep)

    def test_urdf_fk_independent_mujoco_agreement(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(self.dataset.kinematics.path)
        data = mujoco.MjData(model)
        c = self.dataset.clips[TRAIN_IDS[0]]
        ids = [model.body(n).id for n in SPEC.body_names]
        root_id = model.body(SPEC.root_body).id
        for frame in (0, 37, 125, 599):
            for i, name in enumerate(SPEC.joint_names):
                data.qpos[model.joint(name).qposadr] = c.joint_pos[frame, i]
            mujoco.mj_forward(model, data)
            points = (data.xpos[ids] - data.xpos[root_id]) @ data.xmat[root_id].reshape(3, 3)
            np.testing.assert_allclose(points, c.key_positions_b[frame], atol=1e-10)

    def test_source_training_model_limit_mismatch_detected(self):
        for cid in TRAIN_IDS:
            q = self.dataset.clips[cid].joint_pos
            lo, hi = self.dataset.kinematics.limits[:, 0], self.dataset.kinematics.limits[:, 1]
            self.assertTrue(((q < lo - 1e-6) | (q > hi + 1e-6)).any())

    def test_actual_ppo_and_discriminator_cpu_dry_run(self):
        result = run_cpu_dry_run(self.dataset)
        self.assertEqual(result["input_shape"], [16, 10, 39])
        self.assertEqual(result["output_shape"], [16, 1])
        self.assertEqual(result["policy_valid_env_counts_per_step"][:9], [0] * 9)
        self.assertTrue(all(result["gradient_isolation"].values()))
        self.assertFalse(result["formal_amp_training_success"])


if __name__ == "__main__":
    unittest.main()
