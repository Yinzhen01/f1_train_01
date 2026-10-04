"""CPU contract/admission checks, never Isaac execution or effectiveness proof.

Synthetic positive certificates below intentionally carry an impossible test
identity and a non-cloud fingerprint. They test validator semantics only, are
never written to docs/validation, and cannot stand in for independently audited
checkpoint/log/native-substep evidence from an actual Gradmotion task.
"""
import argparse
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from humanoid.amp.feature_freeze import GRADIENT_FIELDS, _same
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.mu_temporal import (MU_GROUPS, MU_SUBGROUPS, apply_mu_config,
    mu_contract, mu_source, validate_feature_smoke_admission, validate_mu,
    validate_mu_certificate, validate_mu_loss_report, warm_start_mu)
from humanoid.amp.scaled_experiment import (ScaledExperiment, canonical_sha,
    implementation_fingerprint, validate_cloud_smoke)
from humanoid.amp.sustain import apply_sustain_config
from test_f1_amp_jitter import config_type
from test_f1_amp_recovery import config_scope, plain


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('freeze_split_temporal01', 'freeze_split_temporal05')
SCHEMA = 'weighted_actor_split_v1'
COMPONENT_FIELDS = (
    'actor_anchor_grad_norm', 'actor_temporal_grad_norm',
    'actor_main_anchor_cosine', 'actor_main_temporal_cosine',
    'actor_anchor_temporal_cosine',
    'actor_main_anchor_cosine_valid_fraction',
    'actor_main_temporal_cosine_valid_fraction',
    'actor_anchor_temporal_cosine_valid_fraction',
)
LEGACY_CONFIG_SHA = {
    'anchor': '04f828a530ff84c4cdb78ca578f44346e14ba1ea05a542514273bc4ef65ddf0a',
    'temporal': '48b7ae2b69ee335a9792dab0f665d1fd0ef9a687c35a2350e653e32c4c89cd48',
    'freeze_anchor': '638f0b79441aa0db02f18507eedc819221e0ae2c297506f2c0cc66636ee6b53d',
    'freeze_temporal': '2bbfffa2638f63f856d38c5c0c9d4a9bd7edeefd1ccf74ce78c230c0ee22abc7',
}
INDEPENDENT_FLAGS = (
    'independent_artifacts_verified', 'frozen_feature_checkpoint_verified',
    'mu_optimization_log_verified', 'frozen_teacher_state_verified',
    'source_environment_equivalence_verified', 'candidate_native_substeps_verified',
)
SOURCE = ROOT.parent/'f1-amp-sustain/outputs/amp-sustain/TASK_20260926_110/model_8802500.pt'


def config_path(group):
    return ROOT/('configs/amp/lafan_walk02_mu_'+group+'.json')


def cli_parser(script):
    """Execute just the actual parser statements; no simulator imports/main."""
    tree = ast.parse(script.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    statements = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Tuple) for target in node.targets):
            break  # parse_known_args, before checkpoint/env/GPU operations.
        statements.append(copy.deepcopy(node))
    scope = dict(argparse=argparse, Path=Path, MU_GROUPS=MU_GROUPS)
    # Unrelated family constants are irrelevant to the new-family membership
    # check; existing family CLI coverage remains in its own unchanged tests.
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if isinstance(node, ast.Name) and node.id.endswith('GROUPS') and node.id != 'MU_GROUPS':
            scope[node.id] = ()
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(script), 'exec'), scope)
    return scope['parser']


class MuComponentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ScaledExperiment(ROOT, ROOT/'configs/amp/lafan_walk02_sustain_control.json')
        cls.experiments = {group: ScaledExperiment(ROOT, config_path(group)) for group in GROUPS}
        cls.old = {group: json.loads((ROOT/('docs/validation/mu_'+group+'_cloud_smoke.json')).read_text(encoding='utf-8'))
                   for group in LEGACY_CONFIG_SHA}

    def setUp(self):
        # Still compare against the real on-disk source configuration; only
        # cache its motion/URDF loading during dozens of tamper tests.
        loader = patch('humanoid.amp.mu_temporal.ScaledExperiment', return_value=self.source)
        loader.start()
        self.addCleanup(loader.stop)

    def certificate(self, group=GROUPS[0]):
        """Coherent CPU fixture with 10 x 8 indexed minibatches; not cloud proof."""
        experiment = self.experiments[group]
        spec = experiment.cfg['mu_temporal']
        cert = copy.deepcopy(self.old['freeze_temporal'])
        cert.update(identity=experiment.identity(),
            implementation_fingerprint='CPU-CONTRACT-FIXTURE-NOT-CLOUD-EVIDENCE',
            code_commit='CPU-CONTRACT-FIXTURE-NOT-A-CLOUD-COMMIT',
            source_task_id='TASK_20990101_999', checkpoint_sha256='f'*64,
            bundle_sha256='e'*64, cpu_test_fixture=True)
        cert['continuation'].update(mu_source(experiment.cfg))
        report = cert['mu_loss_report']
        report['temporal_coef'] = spec['temporal_coef']
        report['weighted_temporal_loss'] = report['temporal_loss']*spec['temporal_coef']
        records = []
        for update, diagnostics in enumerate(report['rollout_diagnostics'], start=2501):
            # PPO has two epochs/four minibatches; these are evaluation counts,
            # not a replacement for unique valid triplets in each rollout.
            evaluated = 2*diagnostics['valid_triplets']
            for minibatch in range(1, 9):
                valid = evaluated//8+int(minibatch <= evaluated % 8)
                temporal_norm = .3*(spec['temporal_coef']/.01) if valid else 0.
                aux_norm = .2+temporal_norm
                row = dict(completed_update=update, minibatch_index=minibatch,
                    batch_size=192, valid_triplets=valid, main_grad_norm=2.,
                    aux_grad_norm_all_batches=aux_norm, es_grad_norm=0.,
                    actor_main_grad_norm=2., actor_aux_grad_norm=aux_norm,
                    actor_main_aux_cosine=1., actor_cosine_valid_fraction=1.,
                    combined_grad_norm_preclip=2.+aux_norm,
                    global_clip_factor=1./(2.+aux_norm),
                    actor_anchor_grad_norm=.2, actor_temporal_grad_norm=temporal_norm,
                    actor_main_anchor_cosine=1., actor_main_temporal_cosine=float(valid > 0),
                    actor_anchor_temporal_cosine=float(valid > 0),
                    actor_main_anchor_cosine_valid_fraction=1.,
                    actor_main_temporal_cosine_valid_fraction=float(valid > 0),
                    actor_anchor_temporal_cosine_valid_fraction=float(valid > 0))
                records.append(row)
        report.update({key: sum(record[key] for record in records)/80 for key in GRADIENT_FIELDS+COMPONENT_FIELDS})
        report.update(component_gradient_schema=SCHEMA, component_gradient_records=records,
            gradient_minibatches=80, triplet_evaluations=sum(record['valid_triplets'] for record in records),
            evaluated_samples=sum(record['batch_size'] for record in records))
        return experiment, cert

    def test_only_registered_groups_add_split_schema_and_fixed_coefficients(self):
        old = mu_contract('freeze_temporal')
        for group, coefficient in zip(GROUPS, (.01, .05)):
            spec = mu_contract(group)
            self.assertIn(group, MU_SUBGROUPS)
            self.assertIn('mu_'+group, MU_GROUPS)
            self.assertEqual(spec['component_gradient_schema'], SCHEMA)
            self.assertEqual(spec['temporal_coef'], coefficient)
            comparable = {key: value for key, value in spec.items()
                          if key not in ('group', 'temporal_coef', 'component_gradient_schema')}
            self.assertEqual(comparable, {key: value for key, value in old.items()
                                        if key not in ('group', 'temporal_coef')})
        for unknown in ('freeze_split_temporal', 'freeze_split_temporal10', 'mu_'+GROUPS[0]):
            with self.assertRaises(ValueError): mu_contract(unknown)

    def test_new_configs_differ_only_in_identity_description_and_temporal_coefficient(self):
        configs = [copy.deepcopy(self.experiments[group].cfg) for group in GROUPS]
        for config in configs:
            config.pop('experiment')
            config.pop('scope')
            config['mu_temporal'].pop('group')
            config['mu_temporal'].pop('temporal_coef')
        self.assertEqual(configs[0], configs[1])
        for group, experiment in self.experiments.items():
            self.assertEqual(validate_mu(experiment), self.source)
            self.assertEqual(experiment.cfg['reward'], self.source.cfg['reward'])
            self.assertEqual(experiment.cfg['smoke'], dict(num_envs=32, updates=10))
            self.assertEqual(experiment.cfg['formal'], dict(num_envs=4096, updates=250))
            self.assertEqual(experiment.cfg['evaluation_updates'], [2750])
            self.assertEqual(experiment.cfg['evaluation_duration_s'], 60)
            self.assertIs(experiment.cfg['training_ready'], False)
            self.assertEqual(experiment.identity()['motion_sha256'], self.source.identity()['motion_sha256'])
            self.assertEqual(experiment.identity()['urdf_lf_sha256'], self.source.identity()['urdf_lf_sha256'])
            spec = experiment.cfg['mu_temporal']
            self.assertEqual(spec['source_task'], 'TASK_20260926_110')
            self.assertEqual(spec['source_completed_updates'], 2500)
            self.assertEqual(spec['source_checkpoint_sha256'], '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31')
            self.assertEqual(spec['learning_rate'], 5e-5)
            self.assertEqual(spec['anchor_coef'], 1.)
            self.assertEqual(spec['frozen_modules'], ['long_history', 'state_estimator'])
            self.assertEqual(spec['trainable_modules'], ['actor', 'critic', 'std'])

    def test_other_config_contract_or_source_changes_fail_closed(self):
        mutations = (
            lambda c: c['mu_temporal'].update(component_gradient_schema='raw_unweighted'),
            lambda c: c['mu_temporal'].update(temporal_coef=.02),
            lambda c: c['mu_temporal'].update(anchor_coef=.5),
            lambda c: c['mu_temporal'].update(source_completed_updates=2750),
            lambda c: c['mu_temporal'].update(source_checkpoint_sha256='a'*64),
            lambda c: c['mu_temporal'].update(source_model_state_sha256='a'*64),
            lambda c: c['mu_temporal'].update(frozen_modules=['state_estimator']),
            lambda c: c['mu_temporal'].update(startup_steps=3),
            lambda c: c['formal'].update(updates=1000),
            lambda c: c['smoke'].update(num_envs=64),
            lambda c: c['reward'].update(style_weight=0),
            lambda c: c['recovery'].update(replay_capacity=1000),
            lambda c: c.update(evaluation_duration_s=5),
            lambda c: c.update(domain_randomization=True),
        )
        for group, experiment in self.experiments.items():
            for index, mutate in enumerate(mutations):
                bad = copy.copy(experiment)
                bad.cfg = copy.deepcopy(experiment.cfg)
                mutate(bad.cfg)
                with self.subTest(group=group, mutation=index), self.assertRaises(ValueError): validate_mu(bad)

    def test_source_controller_physics_rewards_and_no_dr_noise_are_exact(self):
        cls = config_type()
        expected = plain(apply_sustain_config(cls(), 'control'))
        for group in GROUPS:
            cfg = apply_mu_config(cls(), group)
            self.assertEqual(plain(cfg), expected)
            self.assertTrue(assert_no_domain_randomization(cfg))
            self.assertFalse(cfg.noise.add_noise)
            self.assertEqual(cfg.control.action_scale, .5)
            self.assertEqual(cfg.control.decimation, 10)
            self.assertEqual(cfg.sim.dt, .001)
            self.assertEqual(cfg.sim.physx.num_velocity_iterations, 0)
            self.assertEqual(cfg.sim.physx.num_position_iterations, 4)
            self.assertEqual(cfg.asset.self_collisions, 0)
            self.assertEqual(cfg.rewards.scales.recovery_smoothness, -.02)
            for field in ('target_filter', 'ankle_slew', 'substep_penalty'):
                self.assertFalse(hasattr(cfg, field))

    def test_actual_train_and_record_parsers_route_both_new_registered_groups(self):
        for filename in ('train_f1_amp.py', 'record_amp_policy.py'):
            path = ROOT/'humanoid/scripts'/filename
            parser = cli_parser(path)
            for group in GROUPS:
                args = ['--experiment', 'mu_'+group, '--expected-commit', 'CPU-FIXTURE']
                if filename.startswith('train'):
                    args += ['--mode', 'smoke']
                else:
                    args += ['--checkpoint-file', 'unused.pt', '--checkpoint-sha256', 'a'*64,
                             '--output', 'unused-output.pt']
                parsed, remaining = parser.parse_known_args(args)
                self.assertEqual(parsed.experiment, 'mu_'+group)
                self.assertEqual(remaining, [])
            tree = ast.parse(path.read_text(encoding='utf-8'))
            self.assertTrue(any(isinstance(node, ast.Compare) and any(isinstance(value, ast.Name) and value.id == 'MU_GROUPS'
                for value in node.comparators) for node in ast.walk(tree)))
            self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'select_environment'
                for node in ast.walk(tree)))
        # Execute the actual mu branch with only imports removed. CPU class
        # markers are not evidence that the native Isaac environment ran.
        path = ROOT/'humanoid/amp/refinement.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'select_environment')
        branch = next(node for node in function.body if isinstance(node, ast.If) and
                      isinstance(node.test, ast.Compare) and isinstance(node.test.comparators[0], ast.Name)
                      and node.test.comparators[0].id == 'MU_GROUPS')
        function = copy.deepcopy(function)
        branch = copy.deepcopy(branch)
        branch.body = [node for node in branch.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        function.body = [branch]
        scope = dict(MU_GROUPS=MU_GROUPS, apply_mu_config=apply_mu_config,
                     X1AMPRefineCfg=config_type(), X1AMPRecoveryCfgPPO=lambda: 'CPU-PPO-MARKER',
                     X1AMPJitterEnv='CPU-ENV-MARKER')
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        for group in GROUPS:
            cfg, ppo, env = scope['select_environment']('mu_'+group)
            self.assertEqual(plain(cfg), plain(apply_sustain_config(config_type()(), 'control')))
            self.assertEqual((ppo, env), ('CPU-PPO-MARKER', 'CPU-ENV-MARKER'))

    def test_all_four_legacy_configs_and_specs_remain_unchanged(self):
        for group, digest in LEGACY_CONFIG_SHA.items():
            config = json.loads(config_path(group).read_text(encoding='utf-8'))
            self.assertEqual(canonical_sha(config_path(group)), digest)
            self.assertEqual(mu_contract(group), config['mu_temporal'])
            self.assertNotIn('component_gradient_schema', mu_contract(group))

    def test_legacy_certificates_keep_old_fingerprint_contract_not_new_branch_admission(self):
        current = implementation_fingerprint(ROOT)
        for group, cert in self.old.items():
            config = json.loads(config_path(group).read_text(encoding='utf-8'))
            experiment = SimpleNamespace(repo=ROOT, cfg=config, identity=lambda cert=cert: cert['identity'])
            self.assertTrue(validate_cloud_smoke(experiment, cert, cert['implementation_fingerprint']))
            self.assertNotEqual(current, cert['implementation_fingerprint'])
            with self.assertRaises(ValueError): validate_cloud_smoke(experiment, cert, current)
            if group.startswith('freeze_'):
                self.assertTrue(validate_feature_smoke_admission(experiment, cert))
            else:
                with self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, cert)

    def test_synthetic_contract_fixture_is_complete_but_is_not_cloud_evidence(self):
        for group in GROUPS:
            experiment, cert = self.certificate(group)
            self.assertTrue(cert['cpu_test_fixture'])
            self.assertTrue(validate_mu_certificate(experiment, cert['continuation']))
            self.assertTrue(validate_cloud_smoke(experiment, cert, cert['implementation_fingerprint']))
            self.assertTrue(validate_feature_smoke_admission(experiment, cert))
            self.assertEqual(len(cert['mu_loss_report']['component_gradient_records']), 80)
            self.assertEqual(cert['mu_loss_report']['triplet_evaluations'], 2*cert['mu_loss_report']['valid_triplets'])
            raw = copy.deepcopy(cert)
            for key in INDEPENDENT_FLAGS: raw.pop(key)
            # Raw self-check is not the independent formal-admission gate.
            self.assertTrue(validate_cloud_smoke(experiment, raw, raw['implementation_fingerprint']))
            with self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, raw)

    def test_old_certificates_or_adapted_aggregate_only_reports_cannot_admit_new_groups(self):
        for group in GROUPS:
            experiment, cert = self.certificate(group)
            for old in self.old.values():
                with self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, old)
                with self.assertRaises(ValueError): validate_cloud_smoke(experiment, old, old['implementation_fingerprint'])
            aggregate_only = copy.deepcopy(cert)
            aggregate_only['mu_loss_report'].pop('component_gradient_records')
            with self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, aggregate_only)

    def test_every_component_scalar_schema_and_record_list_are_required(self):
        experiment, cert = self.certificate()
        for key in COMPONENT_FIELDS+('component_gradient_schema', 'component_gradient_records'):
            bad = copy.deepcopy(cert)
            bad['mu_loss_report'].pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)
        bad = copy.deepcopy(cert)
        bad['mu_loss_report']['component_gradient_schema'] = 'combined_aux_only'
        with self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)

    def test_records_require_exact_length_order_update_and_minibatch_indices(self):
        experiment, cert = self.certificate()
        records = cert['mu_loss_report']['component_gradient_records']
        variants = (None, [], records[:-1], records+[records[-1]], tuple(records), records[::-1])
        for index, variant in enumerate(variants):
            bad = copy.deepcopy(cert)
            bad['mu_loss_report']['component_gradient_records'] = copy.deepcopy(variant)
            with self.subTest(variant=index), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)
        for field, value in (('completed_update', 2500), ('completed_update', 2511),
                             ('completed_update', 2501.5), ('minibatch_index', 0),
                             ('minibatch_index', 9), ('minibatch_index', True)):
            bad = copy.deepcopy(cert)
            bad['mu_loss_report']['component_gradient_records'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)

    def test_every_record_field_is_required_and_aggregate_must_match_records(self):
        experiment, cert = self.certificate()
        record_fields = GRADIENT_FIELDS+COMPONENT_FIELDS+('completed_update', 'minibatch_index', 'batch_size', 'valid_triplets')
        for key in record_fields:
            bad = copy.deepcopy(cert)
            bad['mu_loss_report']['component_gradient_records'][0].pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)
        for key in GRADIENT_FIELDS+COMPONENT_FIELDS:
            bad = copy.deepcopy(cert)
            bad['mu_loss_report'][key] += .001
            with self.subTest(aggregate=key), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)
        for key, value in (('gradient_minibatches', 79), ('triplet_evaluations', 1), ('evaluated_samples', 1)):
            bad = copy.deepcopy(cert)
            bad['mu_loss_report'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)

    def test_undefined_cosines_nonfinite_norms_and_bad_record_counts_are_rejected(self):
        experiment, cert = self.certificate()
        mutations = (('actor_temporal_grad_norm', float('nan')), ('actor_anchor_grad_norm', -1.),
            ('actor_main_anchor_cosine', 1.1), ('actor_main_temporal_cosine', .5),
            ('actor_main_temporal_cosine_valid_fraction', 1.), ('batch_size', 0),
            ('batch_size', True), ('valid_triplets', 193), ('valid_triplets', .5))
        # Record0 is startup: temporal norm/validity/cosine must be exact zero.
        for field, value in mutations:
            bad = copy.deepcopy(cert)
            bad['mu_loss_report']['component_gradient_records'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)

    def test_teacher_source_and_frozen_state_hashes_remain_exactly_bound(self):
        experiment, cert = self.certificate()
        fields = (('continuation', 'teacher_source_model_sha256'),
            ('continuation', 'student_initial_model_sha256'),
            ('continuation', 'source_sha256'), ('continuation', 'frozen_features_initial_sha256'),
            ('mu_loss_report', 'teacher_final_model_sha256'),
            ('mu_loss_report', 'frozen_features_final_sha256'))
        for section, key in fields:
            for value in (None, 'b'*64):
                bad = copy.deepcopy(cert)
                if value is None: bad[section].pop(key)
                else: bad[section][key] = value
                with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                    validate_feature_smoke_admission(experiment, bad)
        bad = copy.deepcopy(cert)
        bad['independent_source_model_sha256'] = 'b'*64
        bad['continuation'].update(teacher_source_model_sha256='b'*64, student_initial_model_sha256='b'*64)
        bad['mu_loss_report']['teacher_final_model_sha256'] = 'b'*64
        with self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)

    def test_independent_audit_flags_and_source_artifact_hashes_cannot_be_omitted(self):
        experiment, cert = self.certificate()
        for key in INDEPENDENT_FLAGS:
            for value in (None, False, 1):
                bad = copy.deepcopy(cert)
                if value is None: bad.pop(key)
                else: bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)
        for key in ('checkpoint_sha256', 'bundle_sha256', 'independent_source_model_sha256',
                    'independent_source_features_sha256'):
            bad = copy.deepcopy(cert)
            bad.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError): validate_feature_smoke_admission(experiment, bad)

    def test_smoke_eighty_records_cannot_masquerade_as_two_thousand_formal_batches(self):
        experiment, cert = self.certificate()
        report = copy.deepcopy(cert['mu_loss_report'])
        report.update(updates=250, gradient_minibatches=2000)
        with self.assertRaises(ValueError): validate_mu_loss_report(report, experiment.cfg['mu_temporal'], 250)

    def test_positive_combined_aux_cannot_hide_an_inactive_temporal_actor_gradient(self):
        from humanoid.amp.feature_freeze import validate_component_gradient_records
        for group in GROUPS:
            experiment, cert = self.certificate(group)
            report = cert['mu_loss_report']
            records = report['component_gradient_records']
            for record in records:
                record.update(actor_temporal_grad_norm=0., actor_main_temporal_cosine=0.,
                    actor_anchor_temporal_cosine=0., actor_main_temporal_cosine_valid_fraction=0.,
                    actor_anchor_temporal_cosine_valid_fraction=0., aux_grad_norm_all_batches=.2,
                    actor_aux_grad_norm=.2, combined_grad_norm_preclip=2.2, global_clip_factor=1./2.2)
            report.update({key: sum(record[key] for record in records)/len(records)
                           for key in GRADIENT_FIELDS+COMPONENT_FIELDS})
            self.assertGreater(report['weighted_temporal_loss'], 0.)
            self.assertGreater(report['aux_grad_norm_all_batches'], 0.)
            self.assertTrue(validate_component_gradient_records(records, 10, aggregate=report))
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, 'no actual actor component gradient'):
                validate_mu_loss_report(report, experiment.cfg['mu_temporal'], 10)
            with self.subTest(group=group), self.assertRaises(ValueError):
                validate_feature_smoke_admission(experiment, cert)

    def test_real_source_restores_and_freezes_new_schema_without_changing_learning_state(self):
        if not SOURCE.is_file(): self.skipTest('Actual immutable control2500 absent')
        from humanoid.amp.adapter import AMPAlgorithmAdapter
        from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
        from humanoid.amp.dry_run import cpu_ppo_classes
        from humanoid.amp.integration import AMPBridge
        from humanoid.amp.mu_temporal import state_fingerprint
        actor_type, ppo_type = cpu_ppo_classes(ROOT)
        ppo_cfg = config_scope()['X1AMPRecoveryCfgPPO']()
        original = torch.load(SOURCE, weights_only=True, map_location='cpu')
        for group in GROUPS:
            with self.subTest(group=group), contextlib.redirect_stdout(io.StringIO()):
                experiment = self.experiments[group]
                actor = actor_type(235, 47, 219, 12, **plain(ppo_cfg.policy))
                ppo = ppo_type(actor, device='cpu', **plain(ppo_cfg.algorithm))
                ppo.init_storage(2, 24, [3102], [219], [12])
                discriminator = AMPDiscriminator(experiment.spec, experiment.mean, experiment.std)
                trainer = DiscriminatorTrainer(discriminator, bridge_gradient_penalty=1.)
                bridge = AMPBridge(experiment.spec, trainer, 2, experiment.cfg['reward'], 'cpu', experiment.cfg['recovery'])
                adapter = AMPAlgorithmAdapter(ppo, bridge, None, experiment)
                runner = SimpleNamespace(device='cpu', alg=adapter, amp_trainer=trainer)
                proof = warm_start_mu(runner, experiment, SOURCE)
                self.assertTrue(validate_mu_certificate(experiment, proof))
                self.assertEqual(runner.current_learning_iteration, 2500)
                self.assertEqual(adapter.updates, 2500)
                self.assertEqual(proof['replay_windows'], 50000)
                self.assertEqual(proof['frozen_parameter_count'], 16)
                self.assertEqual(proof['trainable_parameter_count'], 17)
                self.assertEqual(ppo.mu_regularizer.component_gradient_schema, SCHEMA)
                self.assertEqual(ppo.mu_regularizer.temporal_coef, experiment.cfg['mu_temporal']['temporal_coef'])
                self.assertEqual(state_fingerprint(actor.state_dict()), proof['student_initial_model_sha256'])
                self.assertEqual(state_fingerprint(ppo.mu_regularizer.teacher.state_dict()), proof['teacher_source_model_sha256'])
                self.assertEqual(len(ppo.optimizer.param_groups[0]['params']), 33)
                self.assertTrue(_same(ppo.optimizer.state_dict(), original['optimizer_state_dict']))
                self.assertTrue(_same(ppo.state_estimator_optimizer.state_dict(), original['es_optimizer_state_dict']))
                self.assertTrue(_same(discriminator.state_dict(), original['amp_discriminator_state_dict']))
                self.assertTrue(_same(trainer.optimizer.state_dict(), original['amp_optimizer_state_dict']))
                self.assertTrue(_same(bridge.replay.state_dict(), original['amp_replay_state']))
                for parameter, slot in zip(actor.parameters(), ppo.optimizer.state_dict()['param_groups'][0]['params']):
                    saved = original['optimizer_state_dict']['state'][slot]
                    restored = ppo.optimizer.state[parameter]
                    for key in ('step', 'exp_avg', 'exp_avg_sq'):
                        self.assertTrue(torch.equal(saved[key], restored[key]))
                self.assertFalse(any(parameter.requires_grad for name, parameter in actor.named_parameters()
                    if name.startswith(('long_history.', 'state_estimator.'))))
                self.assertTrue(ppo.mu_regularizer.teacher_unchanged())


if __name__ == '__main__': unittest.main()
