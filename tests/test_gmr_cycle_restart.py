"""Exact resume identity, Adam state restoration, and periodic off-by-one tests."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from humanoid import gmr_cycle_restart as restart
from test_gmr_smooth import network

ROOT = Path(__file__).resolve().parents[1]


def runner():
    net = network()
    return SimpleNamespace(device='cpu', current_learning_iteration=0, it=0,
        alg=SimpleNamespace(actor_critic=net, learning_rate=1e-5, schedule='fixed',
            optimizer=torch.optim.Adam(net.parameters(), lr=1e-5),
            state_estimator_optimizer=torch.optim.Adam(net.state_estimator.parameters(), lr=1e-5)))


class RestartRequestTest(unittest.TestCase):
    def request(self, group='x1_gmr_cycle_contact', mode='formal'):
        s = restart.SOURCES[group]
        args = SimpleNamespace(task=group, resume=True, checkpoint=s['checkpoint'],
            load_run=None, training_profile=None, seed=5, num_envs=4096 if mode == 'formal' else 512,
            max_iterations=8999 - s['checkpoint'] if mode == 'formal' else 20)
        extra = SimpleNamespace(expected_commit='exact', source_task=s['task'], checkpoint_sha256=s['sha256'], mode=mode)
        return args, extra

    def test_exact_group_source_and_remaining_budget(self):
        for group in restart.SOURCES:
            for mode in ('smoke', 'formal'):
                args, extra = self.request(group, mode)
                self.assertEqual(restart.validate_restart_request(args, extra, 'exact'), restart.SOURCES[group])
                if mode == 'formal':
                    self.assertEqual(args.checkpoint + 1 + args.max_iterations, 9000)

    def test_reject_cross_group_wrong_commit_or_extra_updates(self):
        args, extra = self.request()
        for field, value in (('task', 'x1_gmr_cycle_smooth'), ('task', 'x1_gmr_swing_touch'),
            ('checkpoint', 8000), ('resume', False), ('seed', 6), ('num_envs', 512),
            ('max_iterations', 500), ('training_profile', 'default'), ('load_run', 'latest')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                restart.validate_restart_request(SimpleNamespace(**dict(vars(args), **{field: value})), extra, 'exact')
        for field, value in (('source_task', 'TASK_other'), ('checkpoint_sha256', 'wrong'),
                             ('mode', 'unbounded'), ('expected_commit', 'wrong')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                restart.validate_restart_request(args, SimpleNamespace(**dict(vars(extra), **{field: value})), 'exact')

    def test_entry_and_safe_loading_are_opt_in(self):
        entry = (ROOT / 'humanoid/scripts/train_gmr_cycle_restart.py').read_text()
        self.assertIn('main(restart=True)', entry)
        main = (ROOT / 'humanoid/scripts/train_gmr_cycle.py').read_text()
        self.assertIn('def main(restart=False):', main)
        self.assertLess(main.index('restart_source = validate_restart_request'), main.index('task_registry.make_env'))
        self.assertIn("identity['next_iteration'] + args.max_iterations", main)
        self.assertIn("'restart%d' % restart_source['checkpoint']", main)
        module = (ROOT / 'humanoid/gmr_cycle_restart.py').read_text()
        self.assertIn('weights_only=True', module)


class RestartRestoreTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'model_8500.pt'
        trained = runner()
        for p in trained.alg.actor_critic.parameters():
            p.grad = torch.ones_like(p) * .01
        trained.alg.optimizer.step()
        self.saved = dict(model_state_dict=trained.alg.actor_critic.state_dict(),
            optimizer_state_dict=trained.alg.optimizer.state_dict(),
            es_optimizer_state_dict=trained.alg.state_estimator_optimizer.state_dict(), iter=8500, infos=None)
        self.source = dict(restart.SOURCES['x1_gmr_cycle_contact'])
        self.write()

    def write(self):
        torch.save(self.saved, self.path)
        self.source['sha256'] = restart.sha256(self.path)

    def assert_state_equal(self, left, right):
        self.assertEqual(left['param_groups'], right['param_groups'])
        self.assertEqual(set(left['state']), set(right['state']))
        for key, state in left['state'].items():
            self.assertEqual(set(state), set(right['state'][key]))
            for name, value in state.items():
                torch.testing.assert_close(value, right['state'][key][name], rtol=0, atol=0)

    def test_all_weights_adam_moments_and_empty_es_state_restore_exactly(self):
        target = runner()
        identity = restart.restore_cycle_runner(target, self.path, self.source)
        self.assertFalse(identity['optimizer_reset'])
        self.assertFalse(identity['simulation_and_rng_restored'])
        self.assertEqual(target.current_learning_iteration, 8501)
        self.assertEqual(target.it, 8500)
        self.assertEqual(identity['loaded_tensors'], 33)
        self.assertEqual(identity['restored_optimizer_state_counts'],
                         dict(optimizer_state_dict=33, es_optimizer_state_dict=0))
        for name, value in target.alg.actor_critic.state_dict().items():
            torch.testing.assert_close(value, self.saved['model_state_dict'][name], rtol=0, atol=0)
        self.assert_state_equal(target.alg.optimizer.state_dict(), self.saved['optimizer_state_dict'])
        self.assert_state_equal(target.alg.state_estimator_optimizer.state_dict(), self.saved['es_optimizer_state_dict'])
        # The next real Adam update advances the existing counter, not a reset.
        for p in target.alg.actor_critic.parameters():
            p.grad = torch.ones_like(p) * .01
        target.alg.optimizer.step()
        self.assertTrue(all(s['step'].item() == 2 for s in target.alg.optimizer.state.values()))

    def test_missing_or_wrong_explicit_file_does_not_fall_back(self):
        self.assertEqual(restart.locate_restart_checkpoint(self.path, [], self.source), self.path)
        self.assertEqual(restart.locate_restart_checkpoint(self.path.parent / 'missing.pt', [self.path.parent], self.source), self.path)
        with self.assertRaises(ValueError):
            restart.locate_restart_checkpoint(self.path, [], dict(self.source, sha256='wrong'))
        with self.assertRaises(FileNotFoundError):
            restart.locate_restart_checkpoint(self.path.parent / 'missing.pt', [], self.source)

    def test_wrong_iteration_missing_optimizer_or_bad_moments_rejected(self):
        original = copy.deepcopy(self.saved)
        for mutation in ('iteration', 'missing_es', 'nan_moment', 'lr'):
            self.saved = copy.deepcopy(original)
            if mutation == 'iteration':
                self.saved['iter'] = 8499
            elif mutation == 'missing_es':
                del self.saved['es_optimizer_state_dict']
            elif mutation == 'nan_moment':
                self.saved['optimizer_state_dict']['state'][0]['exp_avg'].fill_(float('nan'))
            else:
                self.saved['optimizer_state_dict']['param_groups'][0]['lr'] = 1e-3
            self.write()
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                restart.restore_cycle_runner(runner(), self.path, self.source)

    def test_wrong_hash_or_schedule_rejected(self):
        with self.assertRaises(ValueError):
            restart.restore_cycle_runner(runner(), self.path, dict(self.source, sha256='wrong'))
        target = runner()
        target.alg.schedule = 'adaptive'
        with self.assertRaises(ValueError):
            restart.restore_cycle_runner(target, self.path, self.source)


if __name__ == '__main__':
    unittest.main()
