import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.amp import probe_native_hardware as probe
from tools.amp.probe_native_hardware import read_hardware


class NativeHardwareProbeTests(unittest.TestCase):
    def fake_torch(self, available=True, count=1, name='NVIDIA GeForce RTX 4090 D'):
        return SimpleNamespace(__version__='fixture-only', version=SimpleNamespace(cuda='12.1'),
            cuda=SimpleNamespace(is_available=Mock(return_value=available),
                device_count=Mock(return_value=count), get_device_name=Mock(return_value=name),
                get_device_properties=Mock(return_value=SimpleNamespace(
                    total_memory=24*1024**3, major=8, minor=9))))

    def test_records_all_actual_fields_without_weakening_driver_name_guard(self):
        fake = self.fake_torch()
        actual = read_hardware(fake, 'linux')
        self.assertTrue(actual['current_driver_guard_would_pass'])
        self.assertEqual(actual['devices'][0], dict(index=0,
            name='NVIDIA GeForce RTX 4090 D', total_memory_bytes=24*1024**3,
            compute_capability=[8, 9]))
        changed = read_hardware(self.fake_torch(name='NVIDIA GeForce RTX 4090'), 'linux')
        self.assertFalse(changed['driver_guard_checks']['name_contains_4090d'])
        self.assertFalse(changed['current_driver_guard_would_pass'])

    def test_unavailable_cuda_does_not_query_nonexistent_device_properties(self):
        fake = self.fake_torch(available=False, count=0)
        actual = read_hardware(fake, 'linux')
        self.assertFalse(actual['current_driver_guard_would_pass'])
        self.assertEqual(actual['devices'], [])
        fake.cuda.get_device_name.assert_not_called()
        fake.cuda.get_device_properties.assert_not_called()

    def test_multiple_devices_and_nonlinux_remain_rejected(self):
        actual = read_hardware(self.fake_torch(count=2), 'linux')
        self.assertEqual(len(actual['devices']), 2)
        self.assertFalse(actual['current_driver_guard_would_pass'])
        self.assertFalse(read_hardware(self.fake_torch(), 'win32')['current_driver_guard_would_pass'])

    def test_import_safe_probe_cannot_invoke_native_fit_or_create_simulator(self):
        source = Path(__file__).resolve().parents[1]/'tools/amp/probe_native_hardware.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        imports = [n for n in main.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports[0].module, 'isaacgym')
        self.assertEqual(imports[1].names[0].name, 'torch')
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = [n.attr if isinstance(n, ast.Attribute) else n.id
                 for n in calls if isinstance(n, (ast.Name, ast.Attribute))]
        self.assertFalse(any(n in ('create_sim', 'fit_head_smoothing', 'load_state_dict',
                                  'load', 'step') for n in names))
        for n in tree.body:
            if isinstance(n, ast.Import):
                self.assertFalse(any(a.name in ('torch', 'isaacgym') for a in n.names))

    def test_main_stub_executes_published_probe_without_granting_fit_or_smoke_admission(self):
        # Explicit CPU-only fixture, not native hardware evidence.
        commit = '1'*40
        output = io.StringIO()
        with patch.object(probe.sys, 'argv', ['probe', '--expected-commit', commit]), \
                patch.object(probe.subprocess, 'check_output', side_effect=[commit.encode(), b'', b'']), \
                patch.object(probe.subprocess, 'check_call') as tracked, \
                patch.dict('sys.modules', {'isaacgym': SimpleNamespace(gymapi=object()),
                                          'torch': self.fake_torch()}), \
                contextlib.redirect_stdout(output):
            probe.main()
        tracked.assert_called_once()
        self.assertIn('--error-unmatch', tracked.call_args[0][0])
        report = json.loads(output.getvalue().split('] ', 1)[1])
        self.assertEqual(report['code_commit'], commit)
        self.assertEqual(len(report['probe_source_sha256']), 64)
        for key in ('simulation_created', 'policy_executed', 'fitting_executed',
                    'smoke_admission', 'effectiveness_verified', 'dr_unlocked'):
            self.assertIs(report[key], False)

    def test_main_rejects_wrong_commit_dirty_or_untracked_native_code(self):
        commit = '1'*40
        for results in ([b'2'*40, b''], [commit.encode(), b' M tracked.py'],
                        [commit.encode(), b'', b'tools/amp/injected.py']):
            with self.subTest(results=results), \
                    patch.object(probe.sys, 'argv', ['probe', '--expected-commit', commit]), \
                    patch.object(probe.subprocess, 'check_output', side_effect=results), \
                    patch.object(probe.subprocess, 'check_call'), \
                    patch.object(probe, 'read_hardware') as readback:
                with self.assertRaises(ValueError):
                    probe.main()
                readback.assert_not_called()


if __name__ == '__main__':
    unittest.main()
