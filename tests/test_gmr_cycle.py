"""Pure math and real-loop stubs; these do not replace the Isaac Gym smoke."""
import ast
import contextlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import tempfile
import torch
from humanoid import gmr_cycle as cycle
from humanoid import gmr_phase_accel as phase
from humanoid.gmr_swing import swing_envelope
from humanoid import gmr_accel_finetune as warm
from test_gmr_smooth import plain
from test_gmr_swing import settings_scope
from test_gmr_phase_accel import JOINTS, class_node, fake_environment

ROOT = Path(__file__).resolve().parents[1]


def cycle_scope():
    scope = settings_scope()
    path = ROOT / 'humanoid/envs/x1/x1_gmr_cycle_config.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


def validators():
    path = ROOT / 'humanoid/scripts/train_gmr_cycle.py'
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
             and n.name.startswith('validate_')]
    scope = dict(class_to_dict=plain, GROUPS=cycle.GROUPS,
                 SOURCE_TASK=warm.SOURCE_TASK, SOURCE_SHA256=warm.SOURCE_SHA256)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class CycleConfigTest(unittest.TestCase):
    def test_three_ablations_change_only_declared_factors(self):
        scope, validate = cycle_scope(), validators()['validate_config']
        names = ('X1GMRCycleContactCfg', 'X1GMRCycleSmoothCfg', 'X1GMRCycleBothCfg')
        for group, name in zip(cycle.GROUPS, names):
            validate(scope[name](), scope['X1GMRSwingCfg'](), group)
            cfg = scope[name]()
            cfg.control.action_scale += .01
            with self.assertRaises(ValueError):
                validate(cfg, scope['X1GMRSwingCfg'](), group)

    def test_ppo_unchanged_except_experiment_directory(self):
        scope = cycle_scope()
        old, new = plain(scope['X1GMRSwingCfgPPO']()), plain(scope['X1GMRCycleCfgPPO']())
        new['runner']['experiment_name'] = old['runner']['experiment_name']
        self.assertEqual(old, new)

    def test_mandatory_source_budget_commit_and_group(self):
        validate = validators()['validate_request']
        args = dict(task='x1_gmr_cycle_both', resume=True, checkpoint=8000,
                    training_profile=None, load_run=None, seed=5, num_envs=512, max_iterations=20)
        extra = dict(expected_commit='exact', source_task=warm.SOURCE_TASK,
                     checkpoint_sha256=warm.SOURCE_SHA256, mode='smoke')
        validate(SimpleNamespace(**args), SimpleNamespace(**extra), 'exact')
        for key, value in (('task', 'x1_gmr_phase_accel'), ('resume', False), ('checkpoint', 9000),
                           ('seed', 6), ('num_envs', 4096), ('max_iterations', 1000),
                           ('training_profile', 'default'), ('load_run', 'latest')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(SimpleNamespace(**dict(args, **{key: value})), SimpleNamespace(**extra), 'exact')
        for key, value in (('expected_commit', 'wrong'), ('source_task', 'TASK_other'),
                           ('checkpoint_sha256', 'wrong'), ('mode', 'unbounded')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(SimpleNamespace(**args), SimpleNamespace(**dict(extra, **{key: value})), 'exact')
        validate(SimpleNamespace(**dict(args, num_envs=4096, max_iterations=1000)),
                 SimpleNamespace(**dict(extra, mode='formal')), 'exact')


class CycleMathTest(unittest.TestCase):
    def test_reference_gate_has_nonzero_transition_floor(self):
        swing, stance = torch.tensor([[0., 1.]]), torch.tensor([[1., 0.]])
        torch.testing.assert_close(cycle.acceleration_weights(swing, stance, .25), torch.ones(1, 2))
        torch.testing.assert_close(cycle.acceleration_weights(swing * 0, stance * 0, .25), torch.full((1, 2), .25))
        for floor in (0., -1., 2.):
            with self.assertRaises(ValueError):
                cycle.acceleration_weights(swing, stance, floor)

    def test_named_ankle_emphasis_and_squared_cost(self):
        ids = phase.leg_joint_indices(list(reversed(JOINTS)))
        acc = torch.zeros(1, 12)
        acc[0, ids[0][-1]] = 100.
        cost = cycle.weighted_acceleration_cost(acc, ids, torch.ones(1, 2), torch.tensor([1.,1.,1.,1.,1.5,2.]), 100.)
        self.assertAlmostEqual(cost.item(), 2. / 6.)
        double = cycle.weighted_acceleration_cost(acc * 2, ids, torch.ones(1, 2), torch.tensor([1.,1.,1.,1.,1.5,2.]), 100.)
        torch.testing.assert_close(double, 4 * cost)

    def test_force_cost_distinguishes_light_contact_from_hard_impact(self):
        gate = torch.ones(1, 2)
        values = [cycle.impact_cost(torch.full((1, 2), f), gate, 300.).item() for f in (0., 6., 100., 6000.)]
        self.assertEqual(values[0], 0.)
        self.assertGreater(values[-1], values[-2])
        self.assertGreater(values[-2], values[1])
        self.assertGreater(cycle.impact_cost(torch.full((1, 2), 6000.), gate * 0, 300.).item(), 0.)

    def test_body_weight_scaling_and_linear_tail(self):
        force, gate = torch.tensor([[900., 3000.]]), torch.ones(1, 2)
        torch.testing.assert_close(cycle.impact_cost(force, gate, 300.), cycle.impact_cost(force * 2, gate, 600.))
        f = lambda x: cycle.impact_cost(torch.tensor([[x, 0.]]), gate, 300.)
        torch.testing.assert_close(f(9000.) - f(6000.), f(6000.) - f(3000.))


class CycleProbeTest(unittest.TestCase):
    def test_training_inference_tensors_can_be_reset_and_probed(self):
        path = ROOT / 'humanoid/scripts/train_gmr_cycle.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'paired_probe')
        # Match the persistent inference tensors produced by PPO rollout.
        with torch.inference_mode():
            actions = torch.ones(1, 12)
        self.assertTrue(actions.is_inference())
        with self.assertRaises(RuntimeError):
            actions.zero_()
        e = SimpleNamespace(num_envs=1, device='cpu', dt=.01, tick=0,
            cfg=SimpleNamespace(motion_reference=SimpleNamespace(random_start=True)),
            motion=SimpleNamespace(duration=.12, metadata={'foot_names':['L','R']},
                sole_local=torch.zeros(2,3), sole_normal_local=torch.zeros(2,3)),
            dof_names=JOINTS, torques=torch.zeros(1,12), clip_finished=torch.zeros(1),
            cycle_trace={})
        e.check_termination = lambda: None
        original = e.check_termination
        resets = []
        def reset(ids):
            actions.zero_()  # Regression: used to fail immediately here.
            e.tick = 0
            resets.append(True)
        e.reset_idx = reset
        e.compute_observations = lambda: None
        e.get_observations = lambda: torch.zeros(1,1)
        def step(action):
            e.tick += 1
            e.cycle_trace = dict(time=(torch.arange(10)[:,None] + 1) * .001 + (e.tick-1)*.01,
                valid=torch.ones(10,1), acc=torch.zeros(10,12), force_z=torch.zeros(10,2))
            e.check_termination()
            return torch.zeros(1,1), None, None, torch.tensor([e.tick >= 12]), {}
        e.step = step
        loaded = []
        policy = SimpleNamespace(eval=lambda: None, act_inference=lambda obs: actions,
                                 load_state_dict=lambda state, strict: loaded.append(state))
        scope = dict(torch=torch, np=np, io=io, json=json,
            capture=lambda env, valid=True: dict(time=env.tick*.01, dof_pos=np.zeros(12)),
            summarize_rollout=lambda data, duration: dict(completed_clip=True))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        with tempfile.TemporaryDirectory() as output, contextlib.redirect_stdout(io.StringIO()):
            artifact = Path(output) / 'probe.pt'
            reports = scope['paired_probe'](e, policy, {'source':1}, {'final':1}, {}, artifact)
            self.assertTrue(artifact.is_file())
            self.assertEqual(set(reports), {'source8000','trained'})
        self.assertEqual(len(resets), 2)
        self.assertEqual(loaded[-1], {'final':1})
        self.assertIs(e.check_termination, original)
        self.assertTrue(e.cfg.motion_reference.random_start)


class CycleLoopTest(unittest.TestCase):
    def setUp(self):
        _, base = fake_environment()
        base.compute_reward = lambda e: setattr(e, 'sum_seen', (e._reward_gmr_cycle_acc().clone(), e._reward_gmr_cycle_impact().clone()))
        scope = dict(X1GMRAccelEnv=base, torch=torch, json=json, math=math,
                     leg_joint_indices=phase.leg_joint_indices, substep_phase_gates=phase.substep_phase_gates,
                     swing_envelope=swing_envelope, acceleration_weights=cycle.acceleration_weights,
                     weighted_acceleration_cost=cycle.weighted_acceleration_cost, impact_cost=cycle.impact_cost)
        node = class_node(ROOT / 'humanoid/envs/x1/x1_gmr_cycle_env.py')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'cycle-env', 'exec'), scope)
        e = self.e = scope['X1GMRCycleEnv'].__new__(scope['X1GMRCycleEnv'])
        e.cfg = cycle_scope()['X1GMRCycleBothCfg']()
        e.sim_params = SimpleNamespace(dt=.001, substeps=1, gravity=SimpleNamespace(z=-9.81))
        e.dt, e.device, e.num_envs, e.num_actions = .01, 'cpu', 2, 12
        e.dof_names, e.dof_vel = JOINTS, torch.zeros(2, 12)
        e.contact_forces, e.feet_indices = torch.zeros(2, 2, 3), [0, 1]
        e.episode_length_buf = torch.tensor([4, 0])
        e.motion = SimpleNamespace(fps=30., data={'foot_contact': torch.zeros(31, 2)})
        e.motion_time = lambda: torch.tensor([.4, 0.])
        e.envs, e.actor_handles, e.sim = [0], [0], object()
        self.refreshes = []
        self.forces = [torch.full((2, 2), 3000. if k == 2 else 0.) for k in range(10)]
        self.vels = [torch.full_like(e.dof_vel, .1 if k % 2 == 0 else 0.) for k in range(10)]
        e.gym = SimpleNamespace(get_actor_rigid_body_properties=lambda *a: [SimpleNamespace(mass=30.)],
            refresh_net_contact_force_tensor=self.refresh_force, refresh_dof_state_tensor=self.refresh_vel,
            set_dof_actuation_force_tensor=lambda *a: None, simulate=lambda *a: None, fetch_results=lambda *a: None)
        e.render = lambda: None
        e.torques = torch.zeros_like(e.dof_vel)
        e._compute_torques = lambda actions: actions * 2
        e.obs_buf, e.privileged_obs_buf = torch.zeros(2, 1), None
        e.rew_buf, e.reset_buf, e.extras = torch.zeros(2), torch.zeros(2), {}
        e.post_physics_step = lambda: e.compute_reward()
        with contextlib.redirect_stdout(io.StringIO()):
            e._init_buffers()

    def refresh_vel(self, *args):
        self.e.dof_vel.copy_(self.vels[self.e.cycle_substep_count])

    def refresh_force(self, *args):
        self.refreshes.append(self.e.cycle_substep_count)
        self.e.contact_forces[:, :, 2] = self.forces[self.e.cycle_substep_count]

    def test_substep_squares_and_force_pulse_do_not_disappear_at_endpoints(self):
        e = self.e
        e.step(torch.ones_like(e.dof_vel))
        self.assertEqual(self.refreshes, list(range(10)))
        self.assertEqual(e.dof_vel.sum().item(), 0.)
        self.assertEqual(e.contact_forces.sum().item(), 0.)
        self.assertGreater(e.sum_seen[0][0].item(), 0.)
        self.assertGreater(e.sum_seen[1][0].item(), 0.)
        self.assertEqual(e.sum_seen[0][1].item(), 0.)
        self.assertEqual(e.sum_seen[1][1].item(), 0.)
        self.assertEqual(e.cycle_trace['force_z'][2, 0].item(), 3000.)
        self.assertAlmostEqual(e.cycle_diagnostics['gmr_cycle_acc_rms_1khz'].item(), 100.)
        self.assertTrue(torch.equal(e.torques, torch.full_like(e.dof_vel, 2.)))

    def test_transition_and_stance_are_not_unpenalized(self):
        for gate in (0., 1.):
            self.e.cycle_swing.zero_()
            self.e.cycle_stance.fill_(gate)
            self.e.step(torch.ones_like(self.e.dof_vel))
            self.assertGreater(self.e.sum_seen[0][0].item(), 0.)
            self.assertGreater(self.e.sum_seen[1][0].item(), 0.)

    def test_reset_does_not_reuse_old_cost_or_velocity(self):
        e = self.e
        e.step(torch.ones_like(e.dof_vel))
        e.episode_length_buf.zero_()
        e.dof_vel.fill_(100.)
        self.vels = [torch.full_like(e.dof_vel, 100.) for _ in range(10)]
        e.step(torch.ones_like(e.dof_vel))
        for cost in e.sum_seen:
            torch.testing.assert_close(cost, torch.zeros(2))
        self.assertTrue(all(v.item() == 0. for v in e.cycle_diagnostics.values()))

    def test_invalid_or_incomplete_sampling_fails_closed(self):
        e = self.e
        e._begin_physics_step()
        with self.assertRaises(RuntimeError):
            e.compute_reward()
        e._after_physics_substep(0)
        with self.assertRaises(RuntimeError):
            e._after_physics_substep(0)
        e.sim_params.substeps = 2
        with self.assertRaises(ValueError):
            e._init_buffers()


if __name__ == '__main__':
    unittest.main()
