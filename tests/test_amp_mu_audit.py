"""Pure CPU tamper checks; these do not claim Isaac execution or improvement."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from humanoid.amp.mu_temporal import mu_contract, state_fingerprint
from tools.amp.jitter_audit import validate_substep_arrays, validate_jitter_updates
from tools.amp.mu_audit import (compare_environment, compare_recording_environment, validate_mu_source,
    validate_mu_updates, validate_mu_loss_report, validate_mu_checkpoint_report,
    validate_mu_log_identity)


def fixture(group='temporal', formal=False):
    spec = mu_contract(group)
    count = 250 if formal else 10
    rows, diagnostics = [], []
    for index in range(count):
        triplets = 0 if index < 2 else 100
        temporal = .2 if triplets else 0.
        rows.append(dict(iteration=2501+index, style_reward=.01, weighted_style_reward=.01,
            style_negative_fraction=0., discriminator_bridge_gradient_penalty=.1,
            mu_updates=1, mu_anchor_coef=1., mu_temporal_coef=spec['temporal_coef'],
            mu_anchor_loss=.001, mu_temporal_loss=temporal, mu_weighted_anchor_loss=.001,
            mu_weighted_temporal_loss=temporal*spec['temporal_coef'], mu_aux_grad_norm=.03 if triplets else 0.,
            mu_teacher_unchanged=True, mu_metadata_observed=True, mu_all_finite=True,
            mu_valid_triplets=triplets))
        diagnostics.append(dict(total_samples=768, candidate_triplets=704,
            valid_triplets=triplets, invalid_triplets=768-triplets,
            invalid_reasons=dict(rollout_boundary=64, previous_done=0, episode_changed=0,
                tick_discontinuity=0, command_changed=0, startup=704-triplets),
            invalid_reason_counts_overlap=True, metadata_observed=True, startup_steps=66))
    report = {key: sum(row['mu_'+key] for row in rows)/count for key in
        ('anchor_loss', 'temporal_loss', 'weighted_anchor_loss', 'weighted_temporal_loss', 'aux_grad_norm')}
    report.update(updates=count, valid_triplets=sum(row['mu_valid_triplets'] for row in rows),
        teacher_unchanged=True, metadata_observed=True, all_finite=True,
        anchor_coef=1., temporal_coef=spec['temporal_coef'], rollout_diagnostics=diagnostics,
        teacher_final_model_sha256='a'*64)
    continuation = dict(teacher_source_model_sha256='a'*64, student_initial_model_sha256='a'*64)
    return spec, rows, report, continuation


class MuAuditTests(unittest.TestCase):
    def test_exact_smoke_and_formal_rows_allow_startup_without_triplets(self):
        for group in ('anchor', 'temporal'):
            for formal in (False, True):
                spec, rows, report, continuation = fixture(group, formal)
                self.assertTrue(validate_mu_updates(rows, group, formal))
                self.assertTrue(validate_mu_loss_report(report, spec, len(rows), continuation, rows,
                    num_envs=32, rollout_steps=24))
                self.assertEqual(rows[0]['mu_valid_triplets'], 0)
                self.assertFalse(report.get('effectiveness_verified', False))

    def test_log_range_amp_mechanics_and_every_mu_scalar_are_required(self):
        _, rows, _, _ = fixture()
        for bad in (rows[:-1], rows+[rows[-1]], [dict(r, iteration=r['iteration']-250) for r in rows]):
            with self.assertRaises(ValueError): validate_mu_updates(bad, 'temporal')
        mutations = dict(mu_anchor_coef=2., mu_temporal_coef=0., mu_weighted_anchor_loss=.2,
            mu_weighted_temporal_loss=.4, mu_anchor_loss=float('nan'), mu_temporal_loss=float('inf'),
            mu_aux_grad_norm=-1., mu_valid_triplets=-1, mu_updates=2,
            mu_teacher_unchanged=False, mu_metadata_observed=False, mu_all_finite=False,
            weighted_style_reward=0., discriminator_bridge_gradient_penalty=0., style_negative_fraction=.1)
        for key, value in mutations.items():
            bad = copy.deepcopy(rows); bad[3][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_mu_updates(bad, 'temporal')
        for key in rows[3]:
            if key.startswith('mu_'):
                bad = copy.deepcopy(rows); bad[3].pop(key)
                with self.subTest(missing=key), self.assertRaises(ValueError): validate_mu_updates(bad, 'temporal')
        for value in (True, .5, float('nan'), float('inf')):
            bad = copy.deepcopy(rows); bad[3]['mu_valid_triplets'] = value
            with self.assertRaises(ValueError): validate_mu_updates(bad, 'temporal')

    def test_no_triplets_no_gradient_and_anchor_temporal_leak_rejected(self):
        for group in ('anchor', 'temporal'):
            _, rows, _, _ = fixture(group)
            bad = [dict(row, mu_valid_triplets=0, mu_temporal_loss=0., mu_weighted_temporal_loss=0.) for row in rows]
            with self.assertRaises(ValueError): validate_mu_updates(bad, group)
        _, rows, _, _ = fixture()
        bad = [dict(row, mu_aux_grad_norm=0.) for row in rows]
        with self.assertRaises(ValueError): validate_mu_updates(bad, 'temporal')
        bad = copy.deepcopy(rows); bad[0].update(mu_temporal_loss=.2, mu_weighted_temporal_loss=.002)
        with self.assertRaises(ValueError): validate_mu_updates(bad, 'temporal')
        _, rows, _, _ = fixture('anchor')
        bad = copy.deepcopy(rows); bad[3]['mu_weighted_temporal_loss'] = .002
        with self.assertRaises(ValueError): validate_mu_updates(bad, 'anchor')

    def test_report_is_bound_to_rows_metadata_contract_and_teacher(self):
        spec, rows, report, continuation = fixture()
        mutations = (lambda r: r.update(teacher_final_model_sha256='b'*64),
            lambda r: r.update(anchor_coef=2.), lambda r: r.update(weighted_temporal_loss=.8),
            lambda r: r.update(metadata_observed=False), lambda r: r.update(teacher_unchanged=False),
            lambda r: r.update(valid_triplets=801), lambda r: r['rollout_diagnostics'].pop(),
            lambda r: r['rollout_diagnostics'][3].update(startup_steps=3),
            lambda r: r['rollout_diagnostics'][3].update(metadata_observed=False),
            lambda r: r['rollout_diagnostics'][3].update(candidate_triplets=769),
            lambda r: r['rollout_diagnostics'][3]['invalid_reasons'].update(previous_done=900),
            lambda r: r['rollout_diagnostics'][3]['invalid_reasons'].pop('command_changed'))
        for mutate in mutations:
            bad = copy.deepcopy(report); mutate(bad)
            with self.assertRaises(ValueError): validate_mu_loss_report(bad, spec, 10, continuation, rows)
        bad = dict(spec, action_scale=.6)
        with self.assertRaises(ValueError): validate_mu_loss_report(report, bad, 10, continuation, rows)
        with self.assertRaises(ValueError):
            validate_mu_loss_report(report, spec, 10, dict(continuation, student_initial_model_sha256='b'*64), rows)
        with self.assertRaises(ValueError): validate_mu_loss_report(report, spec, 10, continuation, rows, num_envs=4096, rollout_steps=24)
        bad = [dict(r, mu_anchor_loss=.002, mu_weighted_anchor_loss=.002) for r in rows]
        with self.assertRaises(ValueError): validate_mu_loss_report(report, spec, 10, continuation, bad)

    def test_checkpoint_report_matches_certificate_except_added_final_hash(self):
        spec, _, report, continuation = fixture()
        saved = dict(report); saved.pop('teacher_final_model_sha256')
        self.assertTrue(validate_mu_checkpoint_report(saved, report, spec, 10, continuation))
        for bad in (dict(saved, weighted_anchor_loss=.1), dict(saved, teacher_unchanged=False),
                    dict(saved, unexpected_field=1)):
            with self.assertRaises(ValueError): validate_mu_checkpoint_report(bad, report, spec, 10, continuation)

    def test_environment_must_remain_control_including_rewards_pd_and_timing(self):
        source = dict(env=dict(num_envs=4096, episode_length_s=60.),
            rewards=dict(scales=dict(recovery_smoothness=-.02, recovery_progress=2.)),
            control=dict(decimation=10, stiffness={'joint':20.}, damping={'joint':1.}, action_scale=.5))
        for group in ('anchor', 'temporal'):
            self.assertTrue(compare_environment(source, source, group))
            smoke = copy.deepcopy(source); smoke['env']['num_envs'] = 32
            self.assertTrue(compare_environment(source, smoke, group, smoke=True))
            with self.assertRaises(ValueError): compare_environment(source, smoke, group)
            for mutate in (lambda r: r['rewards']['scales'].update(recovery_smoothness=-.2),
                    lambda r: r['rewards']['scales'].update(recovery_progress=3.),
                    lambda r: r['control'].update(decimation=5),
                    lambda r: r['control']['stiffness'].update(joint=21.),
                    lambda r: r['control']['damping'].update(joint=2.),
                    lambda r: r.update(target_filter={'cutoff_hz':8.})):
                bad = copy.deepcopy(source); mutate(bad)
                with self.assertRaises(ValueError): compare_environment(source, bad, group, smoke=True)
        for group in ('control', 'mu_anchor', 'unknown'):
            with self.assertRaises(ValueError): compare_environment(source, source, group)

    def test_source_bytes_identity_updates_and_full_model_tensors_are_bound(self):
        model = {'actor':torch.tensor([0.,1.]), 'cnn':torch.ones(2), 'state_estimator':torch.ones(3)}
        identity = {'experiment':'original'}
        digest = state_fingerprint(model)
        continuation = dict(teacher_source_model_sha256=digest, student_initial_model_sha256=digest)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'source.pt'
            torch.save(dict(model_state_dict=model, amp_identity=identity, completed_updates=2500, iter=2499), checkpoint)
            experiment = SimpleNamespace(cfg={'mu_temporal':dict(source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())})
            source = SimpleNamespace(identity=lambda: identity)
            with patch('tools.amp.mu_audit.validate_mu', return_value=source), patch('tools.amp.mu_audit.validate_mu_certificate'):
                self.assertEqual(validate_mu_source(experiment, continuation, checkpoint), digest)
                with self.assertRaises(ValueError): validate_mu_source(experiment, dict(continuation, teacher_source_model_sha256='b'*64), checkpoint)
                for change in (dict(completed_updates=2510), dict(iter=2500), dict(amp_identity={'experiment':'wrong'}),
                               dict(model_state_dict=dict(model, cnn=torch.full((2,), float('nan'))))):
                    torch.save(dict(model_state_dict=model, amp_identity=identity, completed_updates=2500, iter=2499), checkpoint)
                    state = torch.load(checkpoint, weights_only=True); state.update(change); torch.save(state, checkpoint)
                    experiment.cfg['mu_temporal']['source_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    with self.assertRaises(ValueError): validate_mu_source(experiment, continuation, checkpoint)
                experiment.cfg['mu_temporal']['source_checkpoint_sha256'] = '0'*64
                with self.assertRaises(ValueError): validate_mu_source(experiment, continuation, checkpoint)

    def test_only_exact_recorder_horizon_and_size_may_differ_from_training(self):
        for duration, train_size, eval_size in ((1.,32,2), (60.,4096,16)):
            source = dict(env=dict(num_envs=train_size, episode_length_s=60.),
                rewards=dict(scales=dict(recovery_smoothness=-.02, recovery_progress=2.)),
                control=dict(decimation=10, stiffness={'joint':20.}, damping={'joint':1.}))
            recorded = copy.deepcopy(source)
            recorded['env'].update(num_envs=eval_size, episode_length_s=duration+.1)
            for group in ('anchor', 'temporal'):
                self.assertTrue(compare_recording_environment(source, recorded, group, duration, eval_size))
                for mutate in (lambda r: r['env'].update(episode_length_s=duration),
                        lambda r: r['env'].update(num_envs=1),
                        lambda r: r['control'].update(decimation=5),
                        lambda r: r['control']['stiffness'].update(joint=30.),
                        lambda r: r['rewards']['scales'].update(recovery_progress=3.)):
                    bad = copy.deepcopy(recorded); mutate(bad)
                    with self.assertRaises(ValueError): compare_recording_environment(source, bad, group, duration, eval_size)
                for field, value in (('episode_length_s',60.1), ('num_envs',16)):
                    bad = copy.deepcopy(source); bad['env'][field] = value
                    with self.assertRaises(ValueError): compare_recording_environment(bad, recorded, group, duration, eval_size)
            with self.assertRaises(ValueError): compare_recording_environment(source, recorded, 'temporal', duration, eval_size+1)

    def test_start_and_completion_log_records_equal_uploaded_artifacts(self):
        certificate = dict(complete=True, all_finite=True)
        manifest = dict(identity={'experiment':'mu'}, completion=certificate)
        start = dict(manifest); start.pop('completion')
        logs = '[f1-amp-start] '+json.dumps(start)+'\n[f1-amp-complete] '+json.dumps(certificate)+'\n'
        self.assertTrue(validate_mu_log_identity(logs, manifest, certificate))
        for bad in (logs+logs, logs.replace('"mu"', '"wrong"'),
                    logs.replace('"complete": true', '"complete": false'), logs.splitlines()[1],
                    '[f1-amp-update] {}\n'+logs, logs+'[f1-amp-update] {}\n'):
            with self.assertRaises(ValueError): validate_mu_log_identity(bad, manifest, certificate)

    def test_mu_substep_arrays_require_readonly_native_1khz_measurements(self):
        manifest = dict(identity={'experiment':'f1_amp_walk02_mu_temporal'}, modes=['standing'],
            substep_telemetry=dict(physics_hz=1000, samples_per_control=10, reward_used=False))
        arrays = dict(standing_dof_vel=np.zeros((3,2,12)), standing_initial_dof_vel=np.zeros((2,12)),
            standing_valid=np.ones((3,2), dtype=bool), standing_physics_valid=np.ones((3,2), dtype=bool))
        for key, width in (('accel_squared',12), ('accel_peak',12), ('torque_delta_squared',12), ('foot_force_peak',2)):
            arrays['standing_physics_'+key] = np.zeros((3,2,width))
        self.assertTrue(validate_substep_arrays(manifest, arrays))
        for field, value in (('physics_hz',100), ('samples_per_control',1), ('reward_used',True)):
            bad = copy.deepcopy(manifest); bad['substep_telemetry'][field] = value
            with self.assertRaises(ValueError): validate_substep_arrays(bad, arrays)
        bad = copy.deepcopy(arrays); bad.pop('standing_physics_accel_squared')
        with self.assertRaises(KeyError): validate_substep_arrays(manifest, bad)
        bad = copy.deepcopy(arrays); bad['standing_physics_accel_peak'][0,0,0] = float('nan')
        with self.assertRaises(ValueError): validate_substep_arrays(manifest, bad)

    def test_old_jitter_rows_still_do_not_require_mu_fields(self):
        _, rows, _, _ = fixture()
        old = [{key:value for key,value in row.items() if not key.startswith('mu_')} for row in rows]
        self.assertTrue(validate_jitter_updates(old))

    def test_both_cli_mu_routes_require_original_source_and_known_group(self):
        from tools.amp import verify_direction_smoke, verify_direction_formal
        for module in (verify_direction_smoke, verify_direction_formal):
            args = ['audit', '--family','mu','--group','temporal','--task-id','fixture',
                '--expected-commit','fixture','--folder','unused','--status','unused',
                '--logs','unused','--source-manifest','unused']
            if module is verify_direction_formal: args.extend(['--source-evaluation','unused'])
            with patch('sys.argv', args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                module.main()
            self.assertEqual(error.exception.code, 2)
            args.extend(['--source-checkpoint','unused']); args[args.index('--group')+1] = 'control'
            with patch('sys.argv', args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                module.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__': unittest.main()
