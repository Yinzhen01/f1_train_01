"""CPU synthetic/AST head-driver gates; no real fit, Isaac Gym, or cloud call.

Certificate hashes/TaskIds here are deliberately synthetic protocol values.
Subprocess ancestry is mocked, and only isolated guard/control-flow statements
are executed with spies. These tests cannot establish cloud admission evidence.
"""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT/'humanoid/scripts/run_amp_head_smooth.py'
SOURCE_SHA = '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31'
SOURCE_MODEL_SHA = 'c06fbd97e32a483d46a2738014215de2721ad7db4b2b86547645ab72d4c555bc'
IDENTITY = dict(experiment='synthetic_only', config_sha256='a'*64, motion_sha256='b'*64,
    urdf_lf_sha256='c'*64, feature_fingerprint='d'*64)
FINGERPRINT, COMMIT = 'e'*64, 'f'*40


def module():
    spec = importlib.util.spec_from_file_location('head_driver_cpu_protocol', DRIVER)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def tree():
    return ast.parse(DRIVER.read_text(encoding='utf-8'))


def main_node():
    return next(node for node in tree().body if isinstance(node, ast.FunctionDef) and node.name == 'main')


def invoke_nodes(nodes, namespace):
    selected = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(selected, str(DRIVER), 'exec'), namespace)
    return namespace


def certificate():
    return dict(schema='head_smooth_native_smoke_audit_v1', mode='smoke',
        identity=copy.deepcopy(IDENTITY), implementation_fingerprint=FINGERPRINT,
        native_verified=True, cohort_arrays_verified=True, source_forward_verified=True,
        head_archive_verified=True, native_recorder_verified=True, train_seed=305, validation_seed=505,
        num_envs=4, duration_s=2, solver_runs=1, formal_admission=False,
        effectiveness_verified=False, dr_unlocked=False, code_commit=COMMIT,
        cloud_log_sha256='1'*64, train_sha256='2'*64, validation_sha256='3'*64,
        head_sha256='4'*64, audit_report_sha256='5'*64,
        policy_rollout_sha256='6'*64,
        source_checkpoint_sha256=SOURCE_SHA, source_model_state_sha256=SOURCE_MODEL_SHA,
        source_task='TASK_20260926_110', platform_task_id='TASK_20261004_999',
        platform_terminal_status='5')


class NewSmokeCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = module()

    def validate(self, value, **overrides):
        args = dict(identity=IDENTITY, fingerprint=FINGERPRINT, repo=ROOT)
        args.update(overrides)
        return self.driver.validate_smoke_certificate(value, **args)

    def test_only_new_schema_and_exact_parent_implementation_identity_are_admitted(self):
        with patch.object(self.driver.subprocess, 'check_call', return_value=0) as ancestor:
            self.assertIsNone(self.validate(certificate()))
            ancestor.assert_called_once_with(['git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD'], cwd=str(ROOT))
        for changed in (dict(schema='mu_cloud_smoke_audit_v1'), dict(schema='amp_native_smoke_v1'),
                dict(mode='formal'), dict(implementation_fingerprint='6'*64),
                dict(identity={'old_ppo_source': True}), dict(train_seed=105),
                dict(validation_seed=305), dict(num_envs=32), dict(duration_s=60),
                dict(solver_runs=2), dict(source_checkpoint_sha256='7'*64),
                dict(source_model_state_sha256='8'*64), dict(platform_task_id='old_ppo_task'),
                dict(platform_terminal_status='6')):
            value = certificate()
            value.update(changed)
            with self.subTest(changed=changed), patch.object(self.driver.subprocess, 'check_call', return_value=0), \
                    self.assertRaises(ValueError):
                self.validate(value)

    def test_every_actual_artifact_hash_and_required_new_flag_is_mandatory(self):
        keys = ('schema', 'native_verified', 'cohort_arrays_verified', 'source_forward_verified',
            'head_archive_verified', 'native_recorder_verified', 'policy_rollout_sha256',
            'cloud_log_sha256', 'train_sha256', 'validation_sha256',
            'head_sha256', 'audit_report_sha256', 'code_commit', 'source_checkpoint_sha256',
            'source_model_state_sha256', 'platform_task_id', 'platform_terminal_status')
        for key in keys:
            value = certificate()
            value.pop(key)
            with self.subTest(key=key), patch.object(self.driver.subprocess, 'check_call', return_value=0), \
                    self.assertRaises(ValueError):
                self.validate(value)
        for key in ('cloud_log_sha256', 'train_sha256', 'validation_sha256', 'head_sha256',
                    'policy_rollout_sha256', 'audit_report_sha256'):
            value = certificate()
            value[key] = 'not-an-artifact-sha'
            with self.subTest(key=key), patch.object(self.driver.subprocess, 'check_call', return_value=0), \
                    self.assertRaises(ValueError):
                self.validate(value)

    def test_booleans_are_not_equal_integers_and_one_solve_is_not_true(self):
        for key in ('native_verified', 'cohort_arrays_verified', 'source_forward_verified',
                'head_archive_verified', 'native_recorder_verified', 'formal_admission',
                'effectiveness_verified', 'dr_unlocked'):
            value = certificate()
            value[key] = int(value[key])
            with self.subTest(key=key), patch.object(self.driver.subprocess, 'check_call', return_value=0), \
                    self.assertRaises(ValueError):
                self.validate(value)
        value = certificate()
        value['solver_runs'] = True
        with patch.object(self.driver.subprocess, 'check_call', return_value=0), self.assertRaises(ValueError):
            self.validate(value)

    def test_non_ancestor_native_code_and_malformed_commit_are_rejected(self):
        call = ['git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD']
        with patch.object(self.driver.subprocess, 'check_call', side_effect=subprocess.CalledProcessError(1, call)) as ancestor:
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                self.validate(certificate())
            ancestor.assert_called_once_with(call, cwd=str(ROOT))
        value = certificate()
        value['code_commit'] = 'HEAD'
        with patch.object(self.driver.subprocess, 'check_call') as ancestor, self.assertRaises(ValueError):
            self.validate(value)
        ancestor.assert_not_called()

    def test_zero_placeholder_hash_and_identical_split_archive_rejected(self):
        for key in ('cloud_log_sha256', 'train_sha256', 'validation_sha256', 'head_sha256',
                    'policy_rollout_sha256', 'audit_report_sha256'):
            value = certificate()
            value[key] = '0'*64
            with self.subTest(key=key), patch.object(self.driver.subprocess, 'check_call', return_value=0), \
                    self.assertRaises(ValueError):
                self.validate(value)
        value = certificate()
        value['validation_sha256'] = value['train_sha256']
        with patch.object(self.driver.subprocess, 'check_call', return_value=0), self.assertRaises(ValueError):
            self.validate(value)


class NativeRecorderRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = module()

    def command(self, mode, **changes):
        args = dict(mode=mode, source=Path('synthetic_source.pt'), head=None,
            head_sha=None, commit=COMMIT, output=Path('synthetic_native.pt'), task='source_task')
        args.update(changes)
        return self.driver.policy_recording_command(**args)

    def test_smoke_probes_new_recorder_without_candidate_or_ppo_resume(self):
        command = self.command('smoke')
        self.assertEqual(command[1:3], ['-m', 'humanoid.scripts.record_amp_head_policy'])
        for key, value in (('--num_envs', '2'), ('--seed', '5'), ('--duration', '1'),
                           ('--sim_device', 'cuda:0'), ('--rl_device', 'cuda:0')):
            self.assertEqual(command[command.index(key)+1], value)
        self.assertIn('--use_gpu_pipeline', command)
        self.assertNotIn('--head-artifact', command)
        self.assertNotIn('--resume', command)
        with self.assertRaises(ValueError):
            self.command('smoke', head=Path('not_permitted.pt'))

    def test_formal_requires_actual_candidate_sha_and_all32_physical_protocol(self):
        command = self.command('formal', head=Path('candidate.pt'), head_sha='9'*64)
        self.assertEqual(command[command.index('--num_envs')+1], '16')
        self.assertEqual(command[command.index('--duration')+1], '60')
        self.assertEqual(command[command.index('--head-sha256')+1], '9'*64)
        for args in (dict(head=None, head_sha='9'*64), dict(head=Path('x.pt'), head_sha=None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.command('formal', **args)
        with self.assertRaises(ValueError): self.command('unregistered')

    def test_sealed_pair_is_fixed_new_seed_full8x60_and_bound_to_actual_splits(self):
        command = self.driver.sealed_recording_command(source=Path('source.pt'),
            head=Path('head.pt'), head_sha='9'*64, commit=COMMIT, output=Path('pair.pt'),
            task='source_task', sealed=Path('sealed.pt'), train=Path('train.pt'),
            validation=Path('validation.pt'))
        self.assertEqual(command[1:3], ['-m', 'humanoid.scripts.record_amp_head_sealed'])
        for key, value in (('--mode', 'formal'), ('--num_envs', '8'), ('--seed', '705'),
                ('--duration', '60'), ('--sealed-cohort', 'sealed.pt'),
                ('--train-cohort', 'train.pt'), ('--validation-cohort', 'validation.pt')):
            self.assertEqual(command[command.index(key)+1], value)
        self.assertNotIn('--resume', command)

    def test_sealed_evidence_is_never_loaded_until_after_the_single_fit(self):
        body = main_node().body
        fit_gate = next(i for i, node in enumerate(body) if isinstance(node, ast.If)
            and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                    and item.func.id == 'fit_head_smoothing' for item in ast.walk(node)))
        sealed_gate = next(i for i, node in enumerate(body) if isinstance(node, ast.If)
            and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                    and item.func.id == 'sealed_recording_command' for item in ast.walk(node)))
        self.assertGreater(sealed_gate, fit_gate)
        self.assertFalse(any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
            and item.func.id in ('fit_episodes', 'fit_head_smoothing', 'evaluate_head_candidate')
            for item in ast.walk(body[sealed_gate])))


class DriverGuardStructureTests(unittest.TestCase):
    def test_isaacgym_precedes_all_torch_imports_and_module_import_has_no_native_side_effects(self):
        body = main_node().body
        imports = [node for node in body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertIsInstance(imports[0], ast.ImportFrom)
        self.assertEqual(imports[0].module, 'isaacgym')
        torch_index = next(i for i, node in enumerate(imports) if isinstance(node, ast.Import)
            and any(alias.name == 'torch' for alias in node.names))
        self.assertGreater(torch_index, 0)
        for node in tree().body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name in ('torch', 'isaacgym') for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith(('humanoid.', 'isaacgym', 'torch')))

    def test_exact_native_platform_single_gpu_class_guard_executes_with_cpu_stubs(self):
        guard = next(node for node in main_node().body if isinstance(node, ast.If)
            and any(isinstance(child, ast.Constant) and isinstance(child.value, str)
                    and 'Real fitting is Gradmotion' in child.value for child in ast.walk(node)))
        def namespace(platform='linux', available=True, count=1, gpu='NVIDIA GeForce RTX 4090 D'):
            cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: count,
                get_device_name=lambda index: gpu)
            return dict(sys=SimpleNamespace(platform=platform), torch=SimpleNamespace(cuda=cuda))
        invoke_nodes([guard], namespace())
        for changed in (dict(platform='win32'), dict(available=False), dict(count=0), dict(count=2),
                dict(gpu='NVIDIA GeForce RTX 4090'), dict(gpu='NVIDIA A100')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                invoke_nodes([guard], namespace(**changed))

    def test_formal_missing_smoke_refused_and_smoke_cannot_reuse_any_certificate(self):
        guard = next(node for node in main_node().body if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Attribute)
            and node.test.left.attr == 'mode' and node.test.comparators[0].value == 'formal')
        verifier = Mock()
        namespace = dict(extra=SimpleNamespace(mode='formal', smoke_certificate=None),
            validate_smoke_certificate=verifier, identity=IDENTITY, fingerprint=FINGERPRINT,
            repo=ROOT, json=json)
        with self.assertRaises(ValueError): invoke_nodes([guard], namespace)
        verifier.assert_not_called()
        namespace['extra'] = SimpleNamespace(mode='smoke', smoke_certificate=Path('old_ppo_certificate.json'))
        with self.assertRaises(ValueError): invoke_nodes([guard], namespace)
        verifier.assert_not_called()
        namespace['extra'] = SimpleNamespace(mode='smoke', smoke_certificate=None)
        invoke_nodes([guard], namespace)
        verifier.assert_not_called()

    def test_original_source_hash_gate_is_before_source_decode_and_solver(self):
        body = main_node().body
        gate = next(node for node in body if isinstance(node, ast.If)
            and any(isinstance(child, ast.Constant) and child.value == 'Wrong original110 parent checkpoint bytes'
                    for child in ast.walk(node)))
        index = body.index(gate)
        source_decode = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'source_state' for target in node.targets))
        self.assertLess(index, source_decode)
        good = dict(source=Path('synthetic_original110.pt'), file_sha=lambda path: SOURCE_SHA,
            ORIGINAL110_PARENT={'file_sha256': SOURCE_SHA})
        invoke_nodes([gate], good)
        bad = dict(good, file_sha=lambda path: '0'*64)
        with self.assertRaises(ValueError): invoke_nodes([gate], bad)

    def test_exact_commit_and_dirty_or_untracked_code_refused(self):
        guard = next(node for node in main_node().body if isinstance(node, ast.If)
            and any(isinstance(child, ast.Constant) and child.value == 'Exact published clean checkout is required'
                    for child in ast.walk(node)))
        def namespace(expected=COMMIT, actual=COMMIT, responses=(b'', b'')):
            return dict(extra=SimpleNamespace(expected_commit=expected), commit=actual,
                repo=ROOT, re=__import__('re'),
                subprocess=SimpleNamespace(check_output=Mock(side_effect=responses)))
        invoke_nodes([guard], namespace())
        for args in (dict(expected='HEAD'), dict(actual='1'*40),
                dict(responses=(b' M humanoid/amp/head_smoothing.py',)),
                dict(responses=(b'', b'humanoid/local_override.py'))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                invoke_nodes([guard], namespace(**args))


class MemoryTensor:
    """Tiny ndarray proxy ONLY for isolated AST serialization-order spies."""
    def __init__(self, value):
        self.value = np.array(value, copy=True)

    def float(self):
        return MemoryTensor(self.value.astype(np.float32))

    def numpy(self):
        return self.value.copy()


class MemoryTorch:
    def __init__(self):
        self.saved = None
        self.events = []

    def from_numpy(self, value):
        return MemoryTensor(value)

    def save(self, value, buffer):
        self.events.append('save_f32')
        self.saved = {key: MemoryTensor(item.value) for key, item in value.items()}
        buffer.write(b'only-synthetic-memory-packet')

    def load(self, buffer, *, map_location, weights_only):
        self.events.append('read_back')
        if map_location != 'cpu' or weights_only is not True or buffer.tell() != 0:
            raise AssertionError('Actual head read-back route changed')
        return {key: MemoryTensor(item.value) for key, item in self.saved.items()}


class DriverOneSolveReadbackTests(unittest.TestCase):
    def isolated_fit(self, *, mode='formal', f64_pass=True, f32_pass=True):
        gate = next(node for node in main_node().body if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.operand, ast.Call)
            and isinstance(node.test.operand.func, ast.Name) and node.test.operand.func.id == 'all')
        # Stop BEFORE provenance/artifact assembly; no real model or disk packet.
        statements = []
        for node in gate.orelse:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                    target.id == 'math_bytes' for target in node.targets):
                break
            statements.append(node)
        fitted = SimpleNamespace(candidate_weight=np.full((12, 128), .100000002, dtype=np.float64),
            candidate_bias=np.full(12, .030000002, dtype=np.float64),
            report={'admitted': f64_pass, 'direction_scale': .5})
        solver = Mock(return_value=fitted)
        fake_torch = MemoryTorch()
        def verify(*args, **kwargs):
            self.assertEqual(fake_torch.events, ['save_f32', 'read_back'])
            self.assertEqual(args[4].dtype, np.float32)
            self.assertEqual(args[5].dtype, np.float32)
            np.testing.assert_array_equal(args[4], fitted.candidate_weight.astype(np.float32))
            np.testing.assert_array_equal(args[5], fitted.candidate_bias.astype(np.float32))
            self.assertEqual(kwargs['source_identity'], kwargs['candidate_identity'])
            return {'admitted': f32_pass}
        exporter = Mock(side_effect=verify)
        ep_builder = Mock(side_effect=[['whole_train_episode'], ['whole_validation_episode']])
        packet = dict(model_state_dict={'actor.6.weight': MemoryTensor(np.zeros((12, 128), np.float32)),
            'actor.6.bias': MemoryTensor(np.zeros(12, np.float32))})
        namespace = dict(extra=SimpleNamespace(mode=mode), manifests={'train': {}, 'validation': {}},
            arrays={'train': {}, 'validation': {}}, audits={'train': {}, 'validation': {}},
            fit_episodes=ep_builder, identity=IDENTITY,
            exclusions={'initial_state': {'x': ['source110/reference/env-0/episode-0',
                'head_cohort_v1/train/seed-305/env-0/episode-0', 'head_cohort_v1/sealed/seed-705/env-0/episode-0']}},
            source_state=packet, fit_head_smoothing=solver, evaluate_head_candidate=exporter,
            env_cfg=SimpleNamespace(normalization=SimpleNamespace(clip_actions=100.)),
            identity_digest=lambda value: 'a'*64, io=io, torch=fake_torch, report={})
        invoke_nodes(statements, namespace)
        return namespace, solver, exporter, ep_builder

    def test_exactly_one_solver_then_f32_readback_gate_and_all_episodes_preserved(self):
        namespace, solver, exporter, builder = self.isolated_fit()
        solver.assert_called_once()
        exporter.assert_called_once()
        self.assertEqual(builder.call_count, 2)
        self.assertEqual(solver.call_args.args[:2], (['whole_train_episode'], ['whole_validation_episode']))
        self.assertEqual(solver.call_args.kwargs['forbidden_cohort_ids'], [
            'source110/reference/env-0/episode-0', 'head_cohort_v1/sealed/seed-705/env-0/episode-0'])
        self.assertEqual(namespace['report']['solver_runs'], 1)
        self.assertEqual(namespace['report']['status'], 'offline_pass')
        self.assertTrue(namespace['report']['admitted_to_physical_test'])

    def test_float32_failure_or_float64_failure_cannot_be_promoted_to_physical_admission(self):
        for f64, f32 in ((True, False), (False, True), (False, False)):
            namespace, solver, exporter, _ = self.isolated_fit(f64_pass=f64, f32_pass=f32)
            solver.assert_called_once()
            exporter.assert_called_once()
            self.assertEqual(namespace['report']['status'], 'offline_rejected')
            self.assertFalse(namespace['report']['admitted_to_physical_test'])
        namespace, _, _, _ = self.isolated_fit(mode='smoke')
        self.assertEqual(namespace['report']['status'], 'offline_pass')
        self.assertFalse(namespace['report']['admitted_to_physical_test'])

    def test_one_failed_cohort_preserves_queue_evidence_and_never_invokes_any_solver(self):
        gate = next(node for node in main_node().body if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.operand, ast.Call)
            and isinstance(node.test.operand.func, ast.Name) and node.test.operand.func.id == 'all')
        solver = Mock(side_effect=AssertionError('cannot fit a rejected cohort'))
        for mode, ready_key in (('smoke', 'smoke_eligible'), ('formal', 'fit_eligible')):
            episodes = [dict(env=i, complete=i != 2, failed=i == 2) for i in range(4)]
            audits = dict(train={ready_key: False, 'episodes': episodes, 'deduplication': {'passed': True}},
                validation={ready_key: True, 'episodes': [{'complete': True}], 'deduplication': {'passed': True}})
            namespace = dict(ready_key=ready_key, audits=audits,
                extra=SimpleNamespace(mode=mode), fit_head_smoothing=solver,
                report=dict(solver_runs=0, admitted_to_physical_test=False))
            invoke_nodes([gate], namespace)
            self.assertEqual(namespace['report']['status'], 'cohort_rejected')
            self.assertEqual(namespace['report']['solver_runs'], 0)
            self.assertFalse(namespace['report']['admitted_to_physical_test'])
            self.assertEqual(namespace['report']['rejection_reasons']['train']['episodes'], episodes)
            self.assertEqual(len(namespace['report']['rejection_reasons']['train']['episodes']), 4)
        solver.assert_not_called()


if __name__ == '__main__':
    unittest.main()
