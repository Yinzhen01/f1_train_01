"""Regression math + mocked REAL physics-loop ordering, not Isaac Gym smoke."""
import ast
import contextlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid import gmr_phase_accel as phase
from humanoid import gmr_swing as swing
from humanoid import gmr_accel_finetune as warm
from test_gmr_swing import settings_scope
from test_gmr_smooth import plain

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [side + "_" + joint + "_joint" for side in ("left", "right")
          for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")]


def phase_scope():
    scope = settings_scope()
    path = ROOT / "humanoid/envs/x1/x1_gmr_phase_accel_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), "exec"), scope)
    return scope


def class_node(path):
    return next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef))


class PhaseConfigTest(unittest.TestCase):
    def test_only_opt_in_gate_and_one_reward_added(self):
        scope = phase_scope()
        old, new = plain(scope["X1GMRSwingCfg"]()), plain(scope["X1GMRPhaseAccelCfg"]())
        self.assertEqual(new.pop("phase_accel"), dict(scale_rad_s2=100.))
        self.assertTrue(new["swing"].pop("contact_phase_only"))
        self.assertEqual(new["rewards"]["scales"].pop("gmr_phase_acc"), -.02)
        self.assertEqual(old, new)  # Every old reward, reference, PD, action, observation and DR field.
        old, new = plain(scope["X1GMRSwingCfgPPO"]()), plain(scope["X1GMRPhaseAccelCfgPPO"]())
        new["runner"]["experiment_name"] = old["runner"]["experiment_name"]
        self.assertEqual(old, new)
        self.assertIn('register("x1_gmr_phase_accel", X1GMRPhaseAccelEnv',
                      (ROOT / "humanoid/envs/__init__.py").read_text())

    def test_name_mapping_handles_permutation_and_rejects_wrong_asset(self):
        names = list(reversed(JOINTS))
        ids = phase.leg_joint_indices(names)
        self.assertEqual([[names[i] for i in leg] for leg in ids], [JOINTS[:6], JOINTS[6:]])
        for invalid in (JOINTS[:-1], JOINTS[:11] + [JOINTS[0]], JOINTS + ["waist"]):
            with self.assertRaises(ValueError):
                phase.leg_joint_indices(invalid)


class SmokeEntryTest(unittest.TestCase):
    def setUp(self):
        path = ROOT / "humanoid/scripts/train_gmr_phase_smoke.py"
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                 and n.name in ("validate_request", "validate_config")]
        scope = dict(SOURCE_TASK=warm.SOURCE_TASK, SOURCE_SHA256=warm.SOURCE_SHA256, class_to_dict=plain)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
        self.validate_request = scope["validate_request"]
        self.validate_config = scope["validate_config"]
        self.args = SimpleNamespace(task="x1_gmr_phase_accel", resume=True, checkpoint=8000,
            training_profile=None, load_run=None, seed=5, num_envs=512, max_iterations=20)
        self.extra = SimpleNamespace(expected_commit="exact", source_task=warm.SOURCE_TASK,
                                     checkpoint_sha256=warm.SOURCE_SHA256)

    def test_only_exact_smoke_budget_and_source_allowed(self):
        self.validate_request(self.args, self.extra, "exact")
        for key, value in (("num_envs", 4096), ("max_iterations", 1000), ("checkpoint", 9000),
                           ("seed", 6), ("resume", False), ("task", "x1_gmr_accel_1x"),
                           ("load_run", "unverified"), ("training_profile", "scratch")):
            args = SimpleNamespace(**vars(self.args))
            setattr(args, key, value)
            with self.assertRaises(ValueError):
                self.validate_request(args, self.extra, "exact")
        for extra, commit in ((self.extra, "wrong"), (SimpleNamespace(expected_commit="exact",
                source_task="TASK_20260912_033", checkpoint_sha256=warm.SOURCE_SHA256), "exact")):
            with self.assertRaises(ValueError):
                self.validate_request(self.args, extra, commit)

    def test_old_config_change_rejected_before_simulator_creation(self):
        scope = phase_scope()
        baseline, cfg = scope["X1GMRSwingCfg"](), scope["X1GMRPhaseAccelCfg"]()
        baseline.seed = cfg.seed = scope["X1GMRPhaseAccelCfgPPO"]().seed
        self.validate_config(cfg, baseline)
        cfg.control.action_scale += .01
        with self.assertRaises(ValueError):
            self.validate_config(cfg, baseline)

    def test_entry_preserves_import_and_mandatory_warm_start_order(self):
        entry = (ROOT / "humanoid/scripts/train_gmr_phase_smoke.py").read_text()
        self.assertLess(entry.index("from isaacgym"), entry.index("import torch"))
        self.assertLess(entry.index("identity = warm_start_runner"), entry.index("runner.learn("))
        self.assertIn("num_learning_iterations=20", entry)


class ContactGapTest(unittest.TestCase):
    def setUp(self):
        scope = phase_scope()
        self.old = scope["X1GMRSwingCfg"]().swing
        self.new = scope["X1GMRPhaseAccelCfg"]().swing

    def terms(self, cfg, gate=None, valid=None):
        # Recorded 1x rollout at t=2.25s: left is reference swing, but its
        # target mesh minimum is only 2.238 mm; actual force is 6528.997 N.
        return swing.swing_terms(torch.zeros(1, 2), torch.tensor([[.0022380091, .013053963]]),
            torch.tensor([[1., 0.]]) if gate is None else gate,
            torch.tensor([[6528.997, 0.]]), torch.tensor([True]) if valid is None else valid, cfg)

    def test_low_reference_height_no_longer_disables_swing_contact(self):
        old, new = self.terms(self.old), self.terms(self.new)
        self.assertEqual(old["contact_cost"].item(), 0.)
        self.assertEqual(new["contact_cost"].item(), 1.)
        torch.testing.assert_close(new["clearance_cost"], old["clearance_cost"])
        torch.testing.assert_close(new["contact_gate"], torch.tensor([[1., 0.]]))

    def test_reference_boundary_and_reset_still_disable_extra_cost(self):
        for extra in (dict(gate=torch.zeros(1, 2)), dict(valid=torch.tensor([False]))):
            self.assertEqual(self.terms(self.new, **extra)["contact_cost"].item(), 0.)

    def test_reference_low_height_and_actual_contact_cannot_switch_phase_off(self):
        for force in (0., 5., 5.01, 6500.):
            out = swing.swing_terms(torch.zeros(1, 2), torch.full((1, 2), -.01),
                torch.ones(1, 2), torch.full((1, 2), force), torch.tensor([True]), self.new)
            torch.testing.assert_close(out["contact_gate"], torch.ones(1, 2))
            self.assertEqual(out["contact_cost"].item(), 2. if force > 5. else 0.)

    def test_clearance_stays_identical_for_old_and_new_settings(self):
        for target in (-.01, .004, .01, .05):
            args = (torch.zeros(3, 2), torch.full((3, 2), target), torch.rand(3, 2),
                    torch.ones(3, 2) * 100., torch.tensor([True, False, True]))
            old = swing.swing_terms(*args, self.old)
            new = swing.swing_terms(*args, self.new)
            torch.testing.assert_close(old["clearance_cost"], new["clearance_cost"])
            torch.testing.assert_close(old["gate"], new["gate"])


class PhaseMathTest(unittest.TestCase):
    def test_actual_reference_t225_gate_and_stance(self):
        import numpy as np
        with np.load(ROOT / "resources/motions/gmr_kit317_upright_12dof.npz", allow_pickle=False) as data:
            envelope = swing.swing_envelope(torch.tensor(data["foot_contact"], dtype=torch.float32), 30., .04, .06)
        gates = phase.substep_phase_gates(envelope, torch.tensor([2.24, 0., 4.59]),
                                         30., .001, 10, torch.tensor([True, True, True]))
        torch.testing.assert_close(gates[0], torch.tensor([[1., 0.]]).expand(10, -1))
        torch.testing.assert_close(gates[1:], torch.zeros(2, 10, 2))

    def test_interval_endpoints_reset_mask_and_random_start(self):
        envelope = torch.tensor([[0., 1.], [1., 0.], [0., 0.]])
        gates = phase.substep_phase_gates(envelope, torch.tensor([0., .001, 0.]),
                                         1000., .0005, 2, torch.tensor([True, True, False]))
        torch.testing.assert_close(gates[0], torch.tensor([[0., .5], [.5, 0.]]))
        torch.testing.assert_close(gates[1], torch.tensor([[.5, 0.], [0., 0.]]))
        torch.testing.assert_close(gates[2], torch.zeros(2, 2))

    def test_short_swing_and_both_contact_boundaries_remain_excluded(self):
        contact = torch.ones(101, 2)
        contact[10:91, 0] = 0.
        contact[40:44, 1] = 0.
        envelope = swing.swing_envelope(contact, 100., .04, .06)
        self.assertEqual(envelope[:, 1].sum(), 0.)
        self.assertEqual(envelope[10:15, 0].sum(), 0.)
        self.assertEqual(envelope[86:91, 0].sum(), 0.)
        self.assertEqual(envelope[50, 0].item(), 1.)


def fake_environment():
    """Use the actual base step body, stubbing only simulator/external state."""
    node = class_node(ROOT / "humanoid/envs/base/legged_robot.py")
    selected = [n for n in node.body if isinstance(n, ast.FunctionDef)
                and n.name in ("step", "_begin_physics_step", "_after_physics_substep")]
    base_node = ast.ClassDef(name="Base", bases=[], keywords=[], body=selected, decorator_list=[])
    ast.fix_missing_locations(base_node)
    scope = dict(torch=torch, gymtorch=SimpleNamespace(unwrap_tensor=lambda value: value))
    exec(compile(ast.Module(body=[base_node], type_ignores=[]), "base-loop", "exec"), scope)
    base = scope["Base"]
    base._init_buffers = lambda self: None
    base._prepare_reward_function = lambda self: None
    base.compute_reward = lambda self: setattr(self, "sum_seen", self._reward_gmr_phase_acc().clone())
    base.get_command_tracking_debug = lambda self: {"existing": torch.tensor(1.)}
    env_node = class_node(ROOT / "humanoid/envs/x1/x1_gmr_phase_accel_env.py")
    scope.update(X1GMRAccelEnv=base, json=json, math=math,
                 leg_joint_indices=phase.leg_joint_indices, substep_phase_gates=phase.substep_phase_gates,
                 leg_acceleration_square=phase.leg_acceleration_square)
    exec(compile(ast.Module(body=[env_node], type_ignores=[]), "phase-env", "exec"), scope)
    cls = scope["X1GMRPhaseAccelEnv"]
    return cls, base


class PhysicsLoopTest(unittest.TestCase):
    def setUp(self):
        cls, self.base = fake_environment()
        e = self.e = cls.__new__(cls)
        e.cfg = phase_scope()["X1GMRPhaseAccelCfg"]()
        e.sim_params = SimpleNamespace(dt=.001, substeps=1)
        e.dt = .01
        e.device = "cpu"
        e.num_envs, e.num_actions = 3, 12
        e.dof_names = JOINTS
        e.dof_vel = torch.zeros(3, 12)
        e.episode_length_buf = torch.tensor([3, 4, 0])
        e.motion = SimpleNamespace(fps=30.)
        e.motion_time = lambda: torch.tensor([.2, .5, 0.])
        e.swing_envelope = torch.ones(31, 2)
        e.swing_envelope[:, 1] = 0.  # Only LEFT leg in stable reference swing.
        e.reward_scales = {"gmr_phase_acc": -.02 * e.dt}
        with contextlib.redirect_stdout(io.StringIO()):
            e._init_buffers()
        e._prepare_reward_function()
        e.sim = object()
        e.torques = torch.zeros_like(e.dof_vel)
        e._compute_torques = lambda action: action * 2.
        e.render = lambda: None
        e.obs_buf = torch.zeros(3, 1)
        e.privileged_obs_buf = None
        e.rew_buf, e.reset_buf, e.extras = torch.zeros(3), torch.zeros(3), {}
        self.samples = [torch.full_like(e.dof_vel, .1 if i % 2 == 0 else 0.) for i in range(10)]
        self.refreshes, self.applied = [], []
        e.gym = SimpleNamespace(
            set_dof_actuation_force_tensor=lambda sim, tau: self.applied.append(tau.clone()),
            simulate=lambda sim: None, fetch_results=lambda sim, block: None,
            refresh_dof_state_tensor=self.refresh)
        e.post_physics_step = self.post

    def refresh(self, sim):
        self.e.dof_vel.copy_(self.samples[len(self.refreshes) % 10])
        self.refreshes.append(self.e.dof_vel.clone())

    def post(self):
        self.e.episode_length_buf += 1
        self.e.compute_reward()

    def test_opposite_substep_accelerations_do_not_cancel(self):
        e = self.e
        e.step(torch.ones(3, 12))
        self.assertEqual(len(self.refreshes), 10)
        torch.testing.assert_close(e.dof_vel, torch.zeros_like(e.dof_vel))  # 100 Hz endpoints cancel.
        torch.testing.assert_close(e.sum_seen, torch.tensor([1., 1., 0.]))  # 1 kHz squares do not.
        torch.testing.assert_close(e.phase_gate_sum, torch.tensor([[10., 0.], [10., 0.], [0., 0.]]))
        self.assertEqual(e.phase_substep_count, 10)
        self.assertAlmostEqual(e.phase_acc_diagnostics["gmr_phase_acc_rms_rad_s2"].item(), 100.)
        self.assertAlmostEqual(e.phase_acc_diagnostics["gmr_physics_acc_rms_rad_s2"].item(), 100.)
        self.assertEqual(e.phase_acc_diagnostics["gmr_phase_acc_right_rms_rad_s2"].item(), 0.)
        self.assertIn("existing", e.get_command_tracking_debug())

    def test_stance_leg_alone_does_not_enter_additional_phase_cost(self):
        for value in self.samples:
            value[:, :6] = 0.
        self.e.step(torch.zeros(3, 12))
        torch.testing.assert_close(self.e.sum_seen, torch.zeros(3))
        self.assertGreater(self.e.phase_acc_diagnostics["gmr_physics_acc_rms_rad_s2"].item(), 0.)

    def test_each_control_step_restarts_accumulation_and_reset_has_no_stale_cost(self):
        e = self.e
        e.step(torch.ones(3, 12))
        e.episode_length_buf.zero_()
        e.dof_vel[:] = 123.  # Artificial reset discontinuity must not leak to penalty.
        self.samples = [torch.full_like(e.dof_vel, 123.) for _ in range(10)]
        e.step(torch.ones(3, 12))
        torch.testing.assert_close(e.sum_seen, torch.zeros(3))
        self.assertTrue(all(torch.isfinite(v) for v in e.phase_acc_diagnostics.values()))
        self.assertTrue(all(v.item() == 0. for v in e.phase_acc_diagnostics.values()))
        e.step(torch.ones(3, 12))
        torch.testing.assert_close(e.sum_seen, torch.zeros(3))  # Constant velocity, now valid.

    def test_missing_duplicate_samples_fail_closed(self):
        e = self.e
        e._begin_physics_step()
        with self.assertRaises(RuntimeError):
            e.compute_reward()
        e._after_physics_substep(0)
        with self.assertRaises(RuntimeError):
            e._after_physics_substep(0)

    def test_sampling_does_not_change_actions_torques_or_final_state(self):
        e = self.e
        e.step(torch.ones(3, 12))
        observed = e.dof_vel.clone(), e.actions.clone(), e.torques.clone()
        self.refreshes.clear()
        self.applied.clear()
        e._begin_physics_step = lambda: None
        e._after_physics_substep = lambda substep: None
        e.post_physics_step = lambda: None
        self.base.step(e, torch.ones(3, 12))
        for before, after in zip(observed, (e.dof_vel, e.actions, e.torques)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(len(self.applied), 10)
        self.assertTrue(all(torch.equal(tau, torch.full((3, 12), 2.)) for tau in self.applied))

    def test_invalid_runtime_and_normalization_rejected(self):
        e = self.e
        e.sim_params.substeps = 2
        with self.assertRaisesRegex(ValueError, "unchanged"):
            e._init_buffers()
        e.sim_params.substeps = 1
        for scale in (0., -1., float("nan"), float("inf")):
            e.cfg.phase_accel.scale_rad_s2 = scale
            with self.assertRaisesRegex(ValueError, "normalization"):
                e._init_buffers()


if __name__ == "__main__":
    unittest.main()
