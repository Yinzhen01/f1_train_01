"""Exercise frozen-source, command, output and actual-trajectory entry gates."""
import ast
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('amp_physics_diagnostic_entry',
    ROOT/'humanoid/scripts/diagnose_amp_physics.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def fixture(mode='smoke', frames=None):
    num_envs, duration, extended = entry.diagnostic_budget(mode)
    frames = round(duration*100) if frames is None else frames
    contract, identity = dict(group='original'), dict(experiment='control')
    manifest = dict(physics_diagnostic=contract, identity=identity, checkpoint_sha256='digest',
        code_commit='commit', duration_s=duration, num_envs=num_envs, fps=100,
        modes=['standing', 'reference'], policy_deterministic=True, diagnostic_only=True,
        dr_unlocked=False, effectiveness_verified=False,
        no_dr=dict(configuration_verified=True, domain_randomization=False, observation_noise=False))
    manifest.update(evaluation_protocol='fixed60_independent_mode_seeds' if extended else 'physics_diagnostic_smoke',
                    mode_seeds=dict(standing=5, reference=105))
    reset_shapes = dict(root_states=(13,), commands=(4,), gait_start=(),
        episode_length_buf=(), phase_length_buf=(), rsi_indices=())
    reset_shapes.update({name: (12,) for name in
        ('dof_pos', 'dof_vel', 'actions', 'last_actions', 'last_last_actions')})
    snapshots = {}
    for name in ('standing', 'reference'):
        snapshots[name] = {key: np.zeros((num_envs,)+shape).tolist() for key, shape in reset_shapes.items()}
        for root in snapshots[name]['root_states']:
            root[2], root[6] = .62, 1.  # Gym quaternion order is xyzw.
        for command in snapshots[name]['commands']:
            command[0] = .45
        snapshots[name].update(obs_history_sha256='a'*64, critic_history_sha256='b'*64)
    manifest['physics_initialization'] = dict(protocol='identical_reset_inputs_then_native_10ms_warmup',
        warmup_control_dt=.01, reset_inputs=snapshots)
    arrays = {}
    shapes = dict(physics_raw_velocity=(10, 12), physics_raw_torque=(10, 12),
        physics_raw_body_force=(10, 3, 3), physics_raw_body_state=(10, 3, 13),
        physics_interval_initial_velocity=(12,), physics_interval_initial_torque=(12,))
    for name in ('standing', 'reference'):
        arrays[name+'_time'] = (np.arange(frames)+1)*.01
        arrays[name+'_valid'] = np.ones((frames, num_envs), dtype=bool)
        arrays[name+'_failure'] = np.zeros((frames, num_envs), dtype=bool)
        for key, shape in shapes.items():
            arrays[name+'_'+key] = np.zeros((frames,)+shape)
        arrays[name+'_physics_raw_body_state'][..., 6] = 1.
        for key, shape in (('dof_pos', (12,)), ('dof_vel', (12,)), ('action', (12,)),
                           ('torque', (12,)), ('root_state', (13,))):
            arrays[name+'_'+key] = np.zeros((frames, num_envs)+shape)
            arrays[name+'_initial_'+key] = np.zeros((num_envs,)+shape)
        arrays[name+'_root_state'][..., 2] = .62
        arrays[name+'_root_state'][..., 6] = 1.
        arrays[name+'_initial_root_state'][..., 2] = .62
        arrays[name+'_initial_root_state'][..., 6] = 1.
    return manifest, arrays, contract, identity


class PhysicsDiagnosticEntryTests(unittest.TestCase):
    def test_full_entry_rejects_invalid_smoke_evidence_before_source_or_process(self):
        import humanoid
        import humanoid.amp.physics_diagnostic as physics
        import humanoid.amp.refinement as refinement
        fake_isaac = ModuleType('isaacgym')
        fake_isaac.gymapi, fake_isaac.gymtorch = object(), object()
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            certificate = repo/'smoke.json'
            certificate.write_text('{}', encoding='utf-8')
            with patch.dict(sys.modules, {'isaacgym': fake_isaac}), \
                    patch.object(sys, 'path', list(sys.path)), \
                    patch.object(sys, 'argv', ['diagnose', '--mode=full', '--expected-commit=commit',
                        '--smoke-certificate', str(certificate)]), \
                    patch.object(entry, 'ROOT', repo), \
                    patch.object(humanoid, 'LEGGED_GYM_ROOT_DIR', str(repo)), \
                    patch.object(entry, 'validate_checkout', return_value='commit'), \
                    patch.object(physics, 'validate_smoke_certificate', create=True,
                                 side_effect=ValueError('Real raw smoke evidence missing')) as validate, \
                    patch.object(refinement, 'locate_source') as locate, \
                    patch.object(entry, 'prepare_output_folder') as output, \
                    patch.object(entry.subprocess, 'run') as processes:
                with self.assertRaisesRegex(ValueError, 'raw smoke evidence'):
                    entry.main()
            validate.assert_called_once_with({}, 'commit', ('original', 'velocity1', 'selfoff'), physics.SOURCE_SHA)
            locate.assert_not_called()
            output.assert_not_called()
            processes.assert_not_called()

    def test_main_runs_independent_bounded_processes_and_upload_summary(self):
        import humanoid
        import humanoid.amp.physics_diagnostic as physics
        import humanoid.amp.refinement as refinement
        import humanoid.amp.scaled_experiment as scaled
        import tools.amp.inspect_rollout as inspector
        fake_isaac = ModuleType('isaacgym')
        fake_isaac.gymapi, fake_isaac.gymtorch = object(), object()
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            source = repo/'mounted/model_8802500.pt'
            source.parent.mkdir()
            source.write_bytes(b'exact_source')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            identity = {'experiment': 'control'}
            state = dict(amp_identity=identity, completed_updates=2500, iter=2499,
                         model_state_dict={'weight': torch.ones(2)})
            def run(command, **kwargs):
                output = Path(command[command.index('--output')+1])
                output.write_bytes(output.name.encode())
            def load(output):
                group = next(name for name, number in entry.ARTIFACT_IDS.items()
                             if output.name == 'model_%d.pt' % number)
                manifest, arrays, _, _ = fixture()
                manifest.update(physics_diagnostic=physics.diagnostic_contract(group),
                                checkpoint_sha256=digest)
                return manifest, arrays
            with patch.dict(sys.modules, {'isaacgym': fake_isaac}), \
                    patch.object(sys, 'path', list(sys.path)), \
                    patch.object(sys, 'argv', ['diagnose', '--mode=smoke', '--expected-commit=commit']), \
                    patch.object(entry, 'ROOT', repo), \
                    patch.object(humanoid, 'LEGGED_GYM_ROOT_DIR', str(repo)), \
                    patch.object(physics, 'SOURCE_SHA', digest), \
                    patch.object(entry, 'validate_checkout', return_value='commit'), \
                    patch.object(refinement, 'locate_source', return_value=source), \
                    patch.object(scaled, 'ScaledExperiment', return_value=SimpleNamespace(identity=lambda: identity)), \
                    patch.object(torch, 'load', return_value=state), \
                    patch.object(torch, 'save') as save, \
                    patch.object(entry.subprocess, 'run', side_effect=run) as processes, \
                    patch.object(inspector, 'load_bundle', side_effect=load), \
                    patch.object(entry.time, 'sleep') as sleep, \
                    patch('builtins.print'):
                entry.main()
            self.assertEqual(processes.call_count, 3)
            for call in processes.call_args_list:
                self.assertEqual(call[1], dict(cwd=str(repo), check=True, timeout=900))
                self.assertIn('record_amp_policy.py', call[0][0][1])
            packed, output = save.call_args[0]
            manifest, certificate = json.loads(packed['manifest_json']), json.loads(packed['certificate_json'])
            self.assertEqual(output.name, 'model_amp_manifest.pt')
            self.assertEqual(manifest['sdk_task'], 'f1_amp_physics_diagnostic')
            self.assertEqual(set(manifest['group_artifacts']), set(entry.ARTIFACT_IDS))
            for key in ('policy_updates', 'discriminator_updates', 'estimator_updates'):
                self.assertEqual(manifest[key], 0)
                self.assertEqual(certificate[key], 0)
            self.assertTrue(certificate['complete'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            sleep.assert_called_once_with(60)

    def test_sdk_task_and_real_smoke_certificate_are_explicit(self):
        args = entry.parse_request(['--mode=smoke', '--expected-commit=commit',
                                    '--task=f1_amp_physics_diagnostic'])
        self.assertEqual(args.task, 'f1_amp_physics_diagnostic')
        with self.assertRaisesRegex(ValueError, 'real-cloud smoke'):
            entry.parse_request(['--mode=full', '--expected-commit=commit'])
        with self.assertRaisesRegex(ValueError, 'cannot consume'):
            entry.parse_request(['--mode=smoke', '--expected-commit=commit',
                                 '--smoke-certificate=/evidence/smoke.json'])
        args = entry.parse_request(['--mode=full', '--expected-commit=commit',
                                    '--smoke-certificate=/evidence/smoke.json'])
        self.assertEqual(args.smoke_certificate, Path('/evidence/smoke.json'))
        tree = ast.parse((ROOT/'humanoid/scripts/diagnose_amp_physics.py').read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        isaac_line = next(node.lineno for node in main.body
                          if isinstance(node, ast.ImportFrom) and node.module == 'isaacgym')
        torch_line = next(node.lineno for node in main.body
                          if isinstance(node, ast.Import) and any(name.name == 'torch' for name in node.names))
        self.assertLess(isaac_line, torch_line)

    def test_registered_unique_groups_and_exact_commands(self):
        registered = ('original', 'velocity1', 'selfoff')
        self.assertEqual(entry.parse_groups('selfoff,original', registered), ('selfoff', 'original'))
        for value in ('', 'original,original', 'original,', 'original,velocity2', ' original'):
            with self.assertRaises(ValueError): entry.parse_groups(value, registered)
        for mode, envs, duration, extended in (('smoke', 2, 1, False), ('full', 16, 60, True)):
            for group in registered:
                output = Path('/out/model_%d.pt' % entry.ARTIFACT_IDS[group])
                certificate = Path('/evidence/smoke.json') if extended else None
                command = entry.build_command('/repo', '/source.pt', 'sha', 'commit', output, group, mode, 'python',
                                              smoke_certificate=certificate)
                self.assertEqual(command[:2], ['python', str(Path('/repo/humanoid/scripts/record_amp_policy.py'))])
                self.assertIn('--physics-diagnostic='+group, command)
                self.assertIn('--experiment=sustain_control', command)
                self.assertIn('--task=f1_amp_walk02_sustain_control', command)
                self.assertIn('--num_envs=%d' % envs, command)
                self.assertIn('--duration=%d' % duration, command)
                self.assertEqual('--extended-validation' in command, extended)
                self.assertEqual('--smoke-certificate='+str(certificate) in command, extended)
                self.assertEqual(command[command.index('--checkpoint-sha256')+1], 'sha')
                self.assertEqual(command[command.index('--output')+1], str(output))
        with self.assertRaises(ValueError):
            entry.build_command('/repo', '/source.pt', 'sha', 'commit', '/out/model_8802500.pt', 'velocity1', 'smoke')
        with self.assertRaises(ValueError): entry.diagnostic_budget('formal')
        for mode, certificate in (('full', None), ('smoke', Path('/smoke.json'))):
            with self.assertRaisesRegex(ValueError, 'Full recorder'):
                entry.build_command('/repo', '/source.pt', 'sha', 'commit', '/out/model_8802500.pt',
                                    'original', mode, smoke_certificate=certificate)

    def test_source_requires_completed_exact_identity_and_checkout(self):
        identity = dict(experiment='source', motion_sha256='motion')
        state = dict(amp_identity=identity, completed_updates=2500, iter=2499, model_state_dict={'weight': 1})
        self.assertTrue(entry.validate_source_state(state, identity))
        for key, value in (('amp_identity', {}), ('completed_updates', 2750), ('iter', 2500),
                           ('model_state_dict', {})):
            with self.assertRaises(ValueError): entry.validate_source_state(dict(state, **{key: value}), identity)
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder).resolve()
            tracked = '\n'.join(entry.REQUIRED_TRACKED_PATHS).encode()
            with patch.object(entry.subprocess, 'check_output', side_effect=[str(repo).encode(), b'expected', b'', b'', tracked]):
                self.assertEqual(entry.validate_checkout(repo, 'expected'), 'expected')
            with patch.object(entry.subprocess, 'check_output', side_effect=[str(repo).encode(), b'other']):
                with self.assertRaises(ValueError): entry.validate_checkout(repo, 'expected')
            with patch.object(entry.subprocess, 'check_output', side_effect=[str(repo.parent).encode(), b'expected']):
                with self.assertRaises(ValueError): entry.validate_checkout(repo, 'expected')
            with patch.object(entry.subprocess, 'check_output', side_effect=[str(repo).encode(), b'expected', b' M humanoid/scripts/record_amp_policy.py']):
                with self.assertRaisesRegex(ValueError, 'clean committed'):
                    entry.validate_checkout(repo, 'expected')
            with patch.object(entry.subprocess, 'check_output', side_effect=[str(repo).encode(), b'expected', b'', b'humanoid/new_source.py']):
                with self.assertRaisesRegex(ValueError, 'Untracked diagnostic source'):
                    entry.validate_checkout(repo, 'expected')
            with patch.object(entry.subprocess, 'check_output', side_effect=[str(repo).encode(), b'expected', b'', b'', b'']):
                with self.assertRaisesRegex(ValueError, 'fully tracked'):
                    entry.validate_checkout(repo, 'expected')

    def test_artifacts_never_reuse_output_directory(self):
        with tempfile.TemporaryDirectory() as repo:
            folder = entry.prepare_output_folder(repo, 'smoke', 'unique-test')
            self.assertEqual(folder.name, 'unique-test')
            self.assertTrue(folder.is_dir())
            with self.assertRaises(FileExistsError):
                entry.prepare_output_folder(repo, 'full', 'unique-test')
            for timestamp in ('..', '../escape'):
                with self.assertRaises(ValueError): entry.prepare_output_folder(repo, 'smoke', timestamp)

    def test_actual_arrays_identity_budget_finite_and_seed_gates(self):
        for mode in ('smoke', 'full'):
            manifest, arrays, contract, identity = fixture(mode)
            self.assertTrue(entry.validate_group_artifact(manifest, arrays, contract, identity, 'commit', 'digest', mode))
            for key, value in (('checkpoint_sha256', 'wrong'), ('identity', {}), ('num_envs', 4096),
                               ('physics_diagnostic', {}), ('policy_deterministic', False), ('dr_unlocked', True)):
                with self.assertRaises(ValueError):
                    entry.validate_group_artifact(dict(manifest, **{key: value}), arrays, contract, identity, 'commit', 'digest', mode)
            broken = dict(arrays); broken.pop('reference_physics_raw_velocity')
            with self.assertRaises(ValueError): entry.validate_group_artifact(manifest, broken, contract, identity, 'commit', 'digest', mode)
            broken = dict(arrays); broken['standing_action'] = arrays['standing_action'].copy()
            broken['standing_action'][0, 0, 0] = np.nan
            with self.assertRaises(ValueError): entry.validate_group_artifact(manifest, broken, contract, identity, 'commit', 'digest', mode)
            for change in ('protocol', 'missing_reset', 'bad_shape', 'nonfinite_reset', 'bad_history'):
                bad = copy.deepcopy(manifest)
                if change == 'protocol':
                    bad['evaluation_protocol'] = 'other'
                elif change == 'missing_reset':
                    bad['physics_initialization']['reset_inputs'].pop('reference')
                elif change == 'bad_shape':
                    bad['physics_initialization']['reset_inputs']['reference']['dof_pos'] = np.zeros((bad['num_envs'], 11)).tolist()
                elif change == 'nonfinite_reset':
                    bad['physics_initialization']['reset_inputs']['reference']['root_states'][0][0] = float('nan')
                else:
                    bad['physics_initialization']['reset_inputs']['reference']['obs_history_sha256'] = ''
                with self.assertRaises(ValueError):
                    entry.validate_group_artifact(bad, arrays, contract, identity, 'commit', 'digest', mode)
            if mode == 'full':
                with self.assertRaises(ValueError):
                    entry.validate_group_artifact(dict(manifest, mode_seeds=dict(standing=5, reference=5)), arrays,
                                                  contract, identity, 'commit', 'digest', mode)

    def test_failed_prefix_allowed_but_unexplained_early_stop_rejected(self):
        manifest, arrays, contract, identity = fixture('full', frames=20)
        with self.assertRaisesRegex(ValueError, 'surviving'):
            entry.validate_group_artifact(manifest, arrays, contract, identity, 'commit', 'digest', 'full')
        for name in ('standing', 'reference'):
            arrays[name+'_failure'][-1] = True
        self.assertTrue(entry.validate_group_artifact(manifest, arrays, contract, identity, 'commit', 'digest', 'full'))
        arrays['reference_valid'][5] = False  # Restarted episodes are never admitted.
        with self.assertRaisesRegex(ValueError, 'validity'):
            entry.validate_group_artifact(manifest, arrays, contract, identity, 'commit', 'digest', 'full')


if __name__ == '__main__':
    unittest.main()
