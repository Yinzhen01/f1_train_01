"""Execute native import prefixes in fresh Python processes with ABI stubs.

The stubs preserve Python's actual package initialization/circular-import
semantics and the Isaac-before-torch ABI guard. No Isaac Gym simulation, real
policy fitting, cloud request or native compatibility is claimed by these CPU
tests. Removing the bootstrap reproduces task076's actual import failure.
"""
import ast
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = (
    (ROOT/'humanoid/scripts/run_amp_head_smooth.py', 'main'),
    (ROOT/'humanoid/scripts/record_amp_head_sealed.py', '_native_main'),
)


def native_import_prefix(path, name):
    """Use actual entry statements, not a rewritten list of string imports."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    imports = []
    for node in function.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif imports:
            break
    if not imports:
        raise AssertionError('No executable native import prefix')
    return imports


def probe(path, name, mutation):
    """Fresh process: real import machinery and partially initialized modules."""
    nodes = native_import_prefix(path, name)
    if mutation == 'without_env_bootstrap':
        nodes = [node for node in nodes if not (
            isinstance(node, ast.Import)
            and any(alias.name == 'humanoid.envs' for alias in node.names))]
    elif mutation == 'torch_first':
        torch = next(node for node in nodes if isinstance(node, ast.Import)
                     and any(alias.name == 'torch' for alias in node.names))
        nodes = [torch]+[node for node in nodes if node is not torch]
    elif mutation != 'as_written':
        raise ValueError('Unknown test-only mutation')
    sources = {}
    packages = set()
    for node in nodes:
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
        for module in names:
            parts = module.split('.')
            packages.update('.'.join(parts[:i]) for i in range(1, len(parts)))
            sources.setdefault(module, '')
        if isinstance(node, ast.ImportFrom):
            # Non-registry heavy modules expose only requested inert names.
            sources[node.module] += '\n'.join(alias.name+' = object()' for alias in node.names)+'\n'
    packages.update(('isaacgym', 'humanoid', 'humanoid.envs',
                     'humanoid.envs.base', 'humanoid.utils'))
    for package in packages:
        sources.setdefault(package, '')
    sources.update({
        'isaacgym': "event('isaac-package')\n",
        'isaacgym.gymapi': "event('isaac-ready')\nSIM_PHYSX=1\n",
        'torch': "if 'isaac-ready' not in events:\n    raise ImportError('PyTorch was imported before isaacgym modules')\nevent('torch-ready')\n",
        'humanoid': "LEGGED_GYM_ROOT_DIR='synthetic-root'\n",
        'humanoid.envs': "event('envs-enter')\nfrom humanoid.envs.base.legged_robot_config import LeggedRobotCfg\nfrom humanoid.utils.task_registry import task_registry\nevent('envs-ready')\n",
        'humanoid.envs.base': '',
        'humanoid.envs.base.legged_robot_config': "class LeggedRobotCfg: pass\nevent('config-ready')\n",
        'humanoid.utils': "event('utils-enter')\nfrom .helpers import class_to_dict, get_args\nfrom .task_registry import task_registry\nevent('utils-ready')\n",
        'humanoid.utils.helpers': "class_to_dict=object()\nget_args=object()\nevent('helpers-ready')\n",
        'humanoid.utils.task_registry': "event('registry-enter')\nfrom humanoid.envs.base.legged_robot_config import LeggedRobotCfg\ntask_registry=object()\nevent('registry-ready')\n",
    })
    # From-import of gymapi invokes a real submodule load, not a preset object.
    events = []
    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__dict__.update(event=events.append, events=events)
            exec(compile(sources[module.__name__], '<native-import-stub:'+module.__name__+'>', 'exec'),
                 module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in sources:
                spec = importlib.util.spec_from_loader(fullname, Loader(), is_package=fullname in packages)
                # Location metadata lets CPython report the authentic
                # "partially initialized module" cause, not unknown location.
                spec.origin = '<native-import-stub:'+fullname+'>'
                spec.has_location = True
                return spec
            # Any accidental unstubbed heavyweight import must fail closed.
            if fullname.split('.')[0] in ('isaacgym', 'torch', 'numpy', 'humanoid'):
                raise ImportError('Unregistered heavyweight module in isolated test: '+fullname)
            return None

    sys.meta_path.insert(0, Finder())
    try:
        selected = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(selected, str(path), 'exec'), {})
    except ImportError as error:
        print(json.dumps(dict(status='import_error', message=str(error), events=events)))
        return 3
    print(json.dumps(dict(status='loaded', events=events)))
    return 0


class NativeHeadImportOrderTests(unittest.TestCase):
    def run_probe(self, path, name, mutation):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            '--probe', str(path), name, mutation], cwd=str(ROOT),
            capture_output=True, text=True, timeout=20)
        self.assertFalse(result.stderr, result.stderr)
        return result.returncode, json.loads(result.stdout)

    def test_actual_native_prefixes_load_after_environment_bootstrap(self):
        for path, name in ENTRIES:
            with self.subTest(entry=name):
                code, result = self.run_probe(path, name, 'as_written')
                self.assertEqual(code, 0, result)
                self.assertEqual(result['status'], 'loaded')
                events = result['events']
                self.assertLess(events.index('isaac-ready'), events.index('torch-ready'))
                self.assertLess(events.index('envs-enter'), events.index('utils-enter'))
                self.assertIn('registry-ready', events)
                self.assertIn('envs-ready', events)

    def test_removing_bootstrap_reproduces_actual_partially_initialized_registry(self):
        for path, name in ENTRIES:
            with self.subTest(entry=name):
                code, result = self.run_probe(path, name, 'without_env_bootstrap')
                self.assertEqual(code, 3, result)
                self.assertIn("partially initialized module 'humanoid.utils.task_registry'", result['message'])
                self.assertNotIn('registry-ready', result['events'])
                self.assertNotIn('envs-ready', result['events'])

    def test_torch_first_really_triggers_isolated_isaac_abi_guard(self):
        for path, name in ENTRIES:
            with self.subTest(entry=name):
                code, result = self.run_probe(path, name, 'torch_first')
                self.assertEqual(code, 3, result)
                self.assertIn('PyTorch was imported before isaacgym modules', result['message'])
                self.assertNotIn('torch-ready', result['events'])


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--probe':
        sys.exit(probe(Path(sys.argv[2]), sys.argv[3], sys.argv[4]))
    unittest.main()
