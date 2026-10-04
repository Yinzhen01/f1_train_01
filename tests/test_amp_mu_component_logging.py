"""Run the real runner log body with scalar-only writer; no simulator evidence."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from humanoid.amp.learnability import evaluation_checkpoint_name
from test_amp_mu_component_audit import fixture


ROOT = Path(__file__).resolve().parents[1]


class ScalarWriter:
    def __init__(self):
        self.rows = []

    def add_scalar(self, key, value, iteration):
        if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
            raise TypeError('Scalar writer received a non-scalar')
        self.rows.append((key, value, iteration))


class ParentRunner:
    def log(self, locs, width, pad):
        self.parent_called += 1


def log_runner():
    """Compile exactly the checked-in method, without optional Gym/wandb imports."""
    tree = ast.parse((ROOT/'humanoid/amp/runner.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'AMPOnPolicyRunner')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'log')
    selected = ast.ClassDef(name='ActualLogRunner', bases=[ast.Name(id='ParentRunner', ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    namespace = dict(ParentRunner=ParentRunner, Path=Path,
        evaluation_checkpoint_name=evaluation_checkpoint_name, __package__='humanoid.amp')
    exec(compile(module, str(ROOT/'humanoid/amp/runner.py'), 'exec'), namespace)
    runner = namespace['ActualLogRunner']()
    runner.parent_called = 0
    runner.writer = ScalarWriter()
    return runner


def populated_runner():
    spec, rows, _, _ = fixture()
    runner = log_runner()
    latest = copy.deepcopy(rows[0])
    latest.pop('iteration')
    runner.alg = SimpleNamespace(latest=latest, updates=2501)
    runner.amp_experiment = SimpleNamespace(cfg={'mu_temporal': spec, 'evaluation_updates': []})
    return runner


class ComponentRunnerLoggingTests(unittest.TestCase):
    def test_actual_log_keeps_all_scalars_but_not_the_two_registered_non_scalars(self):
        runner = populated_runner()
        before = copy.deepcopy(runner.alg.latest)
        runner.log({'it': 2500})
        excluded = {'mu_component_gradient_schema', 'mu_component_gradient_records'}
        self.assertEqual({name.removeprefix('AMP/') if hasattr(name, 'removeprefix') else name[4:]
                          for name, _, _ in runner.writer.rows}, set(before)-excluded)
        self.assertEqual(before, runner.alg.latest)
        self.assertEqual(runner.parent_called, 1)

    def test_missing_or_changed_component_packet_refused_before_writer(self):
        for key in ('mu_component_gradient_schema', 'mu_component_gradient_records'):
            runner = populated_runner()
            runner.alg.latest.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.log({'it': 2500})
            self.assertEqual(runner.writer.rows, [])
            self.assertEqual(runner.parent_called, 0)

    def test_legacy_does_not_silently_discard_unregistered_new_fields(self):
        runner = populated_runner()
        runner.amp_experiment.cfg['mu_temporal'].pop('component_gradient_schema')
        with self.assertRaises(ValueError):
            runner.log({'it': 2500})

    def test_legacy_scalar_logging_unchanged_and_unknown_list_not_masked(self):
        runner = log_runner()
        runner.alg = SimpleNamespace(latest={'reward': .5, 'count': 12}, updates=2501)
        runner.amp_experiment = SimpleNamespace(cfg={'evaluation_updates': []})
        runner.log({'it': 2500})
        self.assertEqual(runner.writer.rows, [('AMP/reward', .5, 2500), ('AMP/count', 12, 2500)])
        runner.alg.latest['unexpected_packet'] = []
        with self.assertRaises(TypeError):
            runner.log({'it': 2500})


if __name__ == '__main__':
    unittest.main()
