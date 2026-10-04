"""CPU protocol fixtures; none are native hardware or training evidence."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from humanoid.amp.native_hardware import (
    SCHEMA, MIN_MEMORY_BYTES, MAX_MEMORY_BYTES, read_hardware,
    validate_hardware_record)


def fixture(name='NVIDIA GeForce RTX 4090', memory=25393692672):
    return dict(schema=SCHEMA, platform='linux', cuda_available=True,
        cuda_device_count=1, devices=[dict(index=0, name=name,
            total_memory_bytes=memory, compute_capability=[8, 9])],
        torch_version='2.4.1', torch_cuda_version='12.1')


def fake_torch(available=True, count=1, name='NVIDIA GeForce RTX 4090',
               memory=25393692672, major=8, minor=9):
    return SimpleNamespace(__version__='fixture-only',
        version=SimpleNamespace(cuda='12.1'),
        cuda=SimpleNamespace(is_available=Mock(return_value=available),
            device_count=Mock(return_value=count),
            get_device_name=Mock(return_value=name),
            get_device_properties=Mock(return_value=SimpleNamespace(
                total_memory=memory, major=major, minor=minor))))


class ReadHardwareTests(unittest.TestCase):
    def test_actual_values_retained_and_no_platform_or_quality_claims_added(self):
        torch = fake_torch()
        actual = read_hardware(torch, 'linux')
        expected = fixture()
        expected['torch_version'] = 'fixture-only'
        self.assertEqual(actual, expected)
        self.assertTrue(validate_hardware_record(actual))
        torch.cuda.get_device_name.assert_called_once_with(0)
        torch.cuda.get_device_properties.assert_called_once_with(0)
        self.assertEqual(set(actual), {'schema', 'platform', 'cuda_available',
            'cuda_device_count', 'devices', 'torch_version', 'torch_cuda_version'})

    def test_unavailable_cuda_never_queries_device_name_or_properties(self):
        for count in (0, 1):
            torch = fake_torch(available=False, count=count)
            torch.version.cuda = None
            actual = read_hardware(torch, 'linux')
            self.assertIs(actual['cuda_available'], False)
            self.assertEqual(actual['cuda_device_count'], count)
            self.assertEqual(actual['devices'], [])
            self.assertIsNone(actual['torch_cuda_version'])
            torch.cuda.get_device_name.assert_not_called()
            torch.cuda.get_device_properties.assert_not_called()
            with self.assertRaises(ValueError):
                validate_hardware_record(actual)

    def test_multiple_visible_devices_are_not_hidden(self):
        torch = fake_torch(count=2)
        actual = read_hardware(torch, 'linux')
        self.assertEqual(actual['cuda_device_count'], 2)
        self.assertEqual([row['index'] for row in actual['devices']], [0, 1])
        self.assertEqual(torch.cuda.get_device_name.call_count, 2)
        with self.assertRaises(ValueError):
            validate_hardware_record(actual)

    def test_bad_flag_or_count_not_coerced_into_admission(self):
        for available, count in ((1, 1), (True, True), (True, 1.0), (True, -1)):
            torch = fake_torch(available=available, count=count)
            actual = read_hardware(torch, 'linux')
            self.assertIs(type(actual['cuda_available']), type(available))
            self.assertIs(type(actual['cuda_device_count']), type(count))
            torch.cuda.get_device_name.assert_not_called()
            torch.cuda.get_device_properties.assert_not_called()
            with self.assertRaises(ValueError):
                validate_hardware_record(actual)

    def test_property_types_and_version_values_not_fabricated(self):
        torch = fake_torch(memory=True, major=8.0)
        actual = read_hardware(torch, 'linux')
        self.assertIs(actual['devices'][0]['total_memory_bytes'], True)
        self.assertIs(type(actual['devices'][0]['compute_capability'][0]), float)
        with self.assertRaises(ValueError):
            validate_hardware_record(actual)
        torch = fake_torch()
        torch.__version__ = None
        actual = read_hardware(torch, 'linux')
        self.assertIsNone(actual['torch_version'])
        with self.assertRaises(ValueError):
            validate_hardware_record(actual)


class ValidateHardwareTests(unittest.TestCase):
    def test_only_exact_4090_and_4090d_names_with_space_case_normalization(self):
        for name in ('NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090D',
                     'NVIDIA GeForce RTX 4090 D', 'nvidia geforce rtx 4090',
                     '  NVIDIA  GeForce RTX 4090 D  '):
            with self.subTest(name=name):
                record = fixture(name)
                before = copy.deepcopy(record)
                self.assertIs(validate_hardware_record(record), True)
                self.assertEqual(record, before)

    def test_other_cards_laptop_and_prefixed_or_suffixed_names_rejected(self):
        for name in ('NVIDIA A100', 'NVIDIA L20', 'NVIDIA GeForce RTX 5090',
                     'NVIDIA GeForce RTX 4090 Laptop GPU',
                     'NVIDIA GeForce RTX 4090 Ti', 'NVIDIA GeForce RTX 4090DD',
                     'prefix NVIDIA GeForce RTX 4090',
                     'NVIDIA GeForce RTX 4090 suffix', '4090D', 'RTX4090',
                     'NVIDIA\tGeForce RTX 4090', 'NVIDIA GeForce RTX 4090\n',
                     'NVID\u0131A GeForce RTX 4090', '', None, True):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_hardware_record(fixture(name))

    def test_cpu_unavailable_cuda_and_multiple_devices_rejected(self):
        for platform in ('win32', 'darwin', 'cpu', None, True):
            record = fixture()
            record['platform'] = platform
            with self.subTest(platform=platform), self.assertRaises(ValueError):
                validate_hardware_record(record)
        for available in (False, 0, 1, 'true', None):
            record = fixture()
            record['cuda_available'] = available
            with self.subTest(available=available), self.assertRaises(ValueError):
                validate_hardware_record(record)
        record = fixture()
        record['cuda_device_count'] = 2
        record['devices'].append(dict(record['devices'][0], index=1))
        with self.assertRaises(ValueError):
            validate_hardware_record(record)

    def test_count_index_and_memory_require_true_integer_types(self):
        for key, values in (('cuda_device_count', (True, False, 1.0, '1', None, 0, 2)),
                            ('index', (True, False, 0.0, '0', None, -1, 1)),
                            ('total_memory_bytes', (True, float(24*1024**3),
                                                    '25393692672', None))):
            for value in values:
                record = fixture()
                target = record if key == 'cuda_device_count' else record['devices'][0]
                target[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_hardware_record(record)

    def test_memory_inclusive_byte_boundaries_and_one_byte_outside(self):
        for memory in (MIN_MEMORY_BYTES, MIN_MEMORY_BYTES+1, MAX_MEMORY_BYTES-1,
                       MAX_MEMORY_BYTES, 25393692672):
            with self.subTest(memory=memory):
                self.assertIs(validate_hardware_record(fixture(memory=memory)), True)
        for memory in (MIN_MEMORY_BYTES-1, MAX_MEMORY_BYTES+1, 0, -1):
            with self.subTest(memory=memory), self.assertRaises(ValueError):
                validate_hardware_record(fixture(memory=memory))

    def test_capability_exact_typed_integer_list(self):
        for capability in ([8, 0], [9, 0], [8, 9, 0], [8], [], (8, 9),
                           [8.0, 9], [8, 9.0], [True, 9], [8, True],
                           ['8', 9], None):
            record = fixture()
            record['devices'][0]['compute_capability'] = capability
            with self.subTest(capability=capability), self.assertRaises(ValueError):
                validate_hardware_record(record)

    def test_missing_extra_fields_and_container_types_rejected(self):
        for key in fixture():
            record = fixture()
            del record[key]
            with self.subTest(top_missing=key), self.assertRaises(ValueError):
                validate_hardware_record(record)
        for key in fixture()['devices'][0]:
            record = fixture()
            del record['devices'][0][key]
            with self.subTest(device_missing=key), self.assertRaises(ValueError):
                validate_hardware_record(record)
        for scope in ('top', 'device'):
            record = fixture()
            target = record if scope == 'top' else record['devices'][0]
            target['unexpected'] = True
            with self.subTest(extra=scope), self.assertRaises(ValueError):
                validate_hardware_record(record)
        for devices in ((), None, {}, [], [None], [fixture()['devices'][0]]*2):
            record = fixture()
            record['devices'] = devices
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                validate_hardware_record(record)
        for record in (None, [], tuple(fixture().items())):
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_hardware_record(record)

    def test_schema_and_versions_require_nonempty_plain_strings(self):
        for schema in ('amp_native_hardware_readback_v1', True, None, ''):
            record = fixture()
            record['schema'] = schema
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                validate_hardware_record(record)
        for key in ('torch_version', 'torch_cuda_version'):
            for value in ('', '  ', None, True, 2.4):
                record = fixture()
                record[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_hardware_record(record)

    def test_module_import_is_pure_python38_and_has_no_simulator_or_policy_calls(self):
        source = Path(__file__).resolve().parents[1]/'humanoid/amp/native_hardware.py'
        tree = ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 8))
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom))
                             for node in ast.walk(tree)))
        names = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Attribute, ast.Name))]
        self.assertFalse(any(name in names for name in
            ('create_sim', 'act_inference', 'fit_head_smoothing', 'load',
             'load_state_dict', 'step', 'save', 'read_bytes', 'getenv', 'environ')))


if __name__ == '__main__':
    unittest.main()
