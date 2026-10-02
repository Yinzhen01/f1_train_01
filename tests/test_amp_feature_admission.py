"""CPU admission tests using adapted old mu certificates, never new cloud evidence.

Positive cases are deliberately synthetic: they test the audited-certificate
contract, and must not be saved or presented as a feature-freeze smoke result.
The real immutable source is read when available to derive both model hashes.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from humanoid.amp.feature_freeze import GRADIENT_FIELDS
from humanoid.amp.mu_temporal import (
    mu_source, state_fingerprint, validate_feature_smoke_admission,
)
from humanoid.amp.scaled_experiment import validate_cloud_smoke


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'f1-amp-sustain/outputs/amp-sustain/TASK_20260926_110/model_8802500.pt'
INDEPENDENT_FLAGS = (
    'independent_artifacts_verified', 'frozen_feature_checkpoint_verified',
    'mu_optimization_log_verified', 'frozen_teacher_state_verified',
    'source_environment_equivalence_verified', 'candidate_native_substeps_verified',
)
ARTIFACT_HASHES = ('checkpoint_sha256', 'bundle_sha256', 'independent_source_model_sha256')
FEATURE_HASHES = ('frozen_features_initial_sha256', 'frozen_features_final_sha256')
torch.set_num_threads(1)


class FeatureAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = {group: json.loads((ROOT/('docs/validation/mu_'+group+'_cloud_smoke.json')).read_text(encoding='utf-8'))
                   for group in ('anchor', 'temporal')}
        cls.source_config = SimpleNamespace(cfg=json.loads(
            (ROOT/'configs/amp/lafan_walk02_sustain_control.json').read_text(encoding='utf-8')))
        freeze_config = json.loads((ROOT/'configs/amp/lafan_walk02_mu_freeze_anchor.json').read_text(encoding='utf-8'))
        cls.feature_sha = freeze_config['mu_temporal']['source_feature_state_sha256']
        cls.model_sha = cls.old['anchor']['independent_source_model_sha256']
        if SOURCE.is_file():
            if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != freeze_config['mu_temporal']['source_checkpoint_sha256']:
                raise AssertionError('Actual source checkpoint is not the immutable approved file')
            source = torch.load(SOURCE, weights_only=True, map_location='cpu')
            model = source['model_state_dict']
            cls.model_sha = state_fingerprint(model)
            features = {name: value for name, value in model.items()
                        if name.startswith(('long_history.', 'state_estimator.'))}
            cls.feature_sha = state_fingerprint(features)
            if len(features) != 16:
                raise AssertionError('Immutable source does not have the expected sixteen feature tensors')
            if cls.model_sha != cls.old['anchor']['independent_source_model_sha256']:
                raise AssertionError('Saved old certificate does not match actual immutable model tensors')
            if cls.feature_sha != freeze_config['mu_temporal']['source_feature_state_sha256']:
                raise AssertionError('Frozen source contract does not match actual immutable feature tensors')

    def setUp(self):
        # validate_mu still compares the full on-disk source configuration. Cache
        # only its loader so dozens of negative cases do not rebuild motion data.
        loader = patch('humanoid.amp.mu_temporal.ScaledExperiment', return_value=self.source_config)
        loader.start()
        self.addCleanup(loader.stop)

    def certificate(self, group='anchor', audited=True):
        path = ROOT/('configs/amp/lafan_walk02_mu_freeze_'+group+'.json')
        config = json.loads(path.read_text(encoding='utf-8'))
        cert = copy.deepcopy(self.old[group])
        identity = dict(cert['identity'], experiment=config['experiment'],
                        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        experiment = SimpleNamespace(repo=ROOT, cfg=config, identity=lambda: identity)
        cert['identity'] = identity
        cert['continuation'].update(mu_source(config))
        proof = dict(frozen_features_unchanged=True, frozen_optimizer_unchanged=True,
                     frozen_parameter_count=16, trainable_parameter_count=17,
                     frozen_features_initial_sha256=self.feature_sha,
                     frozen_features_final_sha256=self.feature_sha)
        cert['continuation'].update(proof)
        cert['continuation'].update(teacher_source_model_sha256=self.model_sha,
                                    student_initial_model_sha256=self.model_sha)
        report = cert['mu_loss_report']
        report.update(proof)
        report.update({key: .01 for key in GRADIENT_FIELDS})
        report.update(es_grad_norm=0., actor_main_aux_cosine=0.,
                      actor_cosine_valid_fraction=1., global_clip_factor=.5,
                      gradient_minibatches=80, teacher_final_model_sha256=self.model_sha)
        for key in INDEPENDENT_FLAGS:
            cert.pop(key, None)
        cert.pop('independent_source_model_sha256', None)
        if audited:
            cert.update({key: True for key in INDEPENDENT_FLAGS})
            cert['independent_source_model_sha256'] = self.model_sha
            cert['independent_source_features_sha256'] = self.feature_sha
        return experiment, cert

    def test_old_mu_certificates_cannot_admit_either_new_freeze_group(self):
        for group in self.old:
            experiment, _ = self.certificate(group)
            with self.subTest(group=group), self.assertRaises(ValueError):
                validate_feature_smoke_admission(experiment, self.old[group])

    def test_legacy_mu_smoke_selfcheck_keeps_its_original_contract(self):
        for group, certificate in self.old.items():
            config = json.loads((ROOT/('configs/amp/lafan_walk02_mu_'+group+'.json')).read_text(encoding='utf-8'))
            experiment = SimpleNamespace(repo=ROOT, cfg=config, identity=lambda: certificate['identity'])
            with self.subTest(group=group):
                self.assertTrue(validate_cloud_smoke(experiment, certificate, certificate['implementation_fingerprint']))
                with self.assertRaises(ValueError):
                    validate_feature_smoke_admission(experiment, certificate)

    def test_raw_smoke_selfcheck_stays_available_but_cannot_admit_formal(self):
        for group in self.old:
            experiment, cert = self.certificate(group, audited=False)
            with self.subTest(group=group):
                self.assertTrue(validate_cloud_smoke(experiment, cert, cert['implementation_fingerprint']))
                with self.assertRaises(ValueError):
                    validate_feature_smoke_admission(experiment, cert)

    def test_synthetic_audited_certificates_accept_both_registered_freeze_groups(self):
        for group in self.old:
            experiment, cert = self.certificate(group)
            with self.subTest(group=group):
                self.assertTrue(validate_feature_smoke_admission(experiment, cert))

    def test_every_independent_flag_is_required_and_must_be_literal_true(self):
        experiment, cert = self.certificate()
        for key in INDEPENDENT_FLAGS:
            for value in (None, False, 1):
                bad = copy.deepcopy(cert)
                if value is None:
                    bad.pop(key)
                else:
                    bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_feature_smoke_admission(experiment, bad)

    def test_every_inspected_artifact_hash_is_required_and_well_formed(self):
        experiment, cert = self.certificate()
        for key in ARTIFACT_HASHES + ('independent_source_features_sha256',):
            for value in (None, '', '0'*63, 'G'*64, True):
                bad = copy.deepcopy(cert)
                if value is None:
                    bad.pop(key)
                else:
                    bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_feature_smoke_admission(experiment, bad)

    def test_smoke_task_identity_is_required(self):
        experiment, cert = self.certificate()
        for value in (None, '', 'TASK_20261002', '20261002_081', 'TASK_bad_081', True):
            bad = copy.deepcopy(cert)
            if value is None:
                bad.pop('source_task_id')
            else:
                bad['source_task_id'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_feature_smoke_admission(experiment, bad)

    def test_frozen_hashes_cannot_disagree_with_independently_inspected_source(self):
        experiment, cert = self.certificate()
        for section in ('continuation', 'mu_loss_report'):
            for key in FEATURE_HASHES:
                for value in (None, 'b'*64):
                    bad = copy.deepcopy(cert)
                    if value is None:
                        bad[section].pop(key)
                    else:
                        bad[section][key] = value
                    with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                        validate_feature_smoke_admission(experiment, bad)
        # Changing each internal initial/final pair together must still fail.
        for section in ('continuation', 'mu_loss_report'):
            bad = copy.deepcopy(cert)
            bad[section].update({key: 'b'*64 for key in FEATURE_HASHES})
            with self.subTest(section=section), self.assertRaises(ValueError):
                validate_feature_smoke_admission(experiment, bad)
        # Self-consistent replacement everywhere cannot replace the static,
        # hash-bound original source digest either.
        bad = copy.deepcopy(cert)
        for section in ('continuation', 'mu_loss_report'):
            bad[section].update({key: 'b'*64 for key in FEATURE_HASHES})
        bad['independent_source_features_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            validate_feature_smoke_admission(experiment, bad)

    def test_teacher_hashes_must_equal_independently_inspected_full_source(self):
        experiment, cert = self.certificate()
        fields = (('continuation', 'teacher_source_model_sha256'),
                  ('continuation', 'student_initial_model_sha256'),
                  ('mu_loss_report', 'teacher_final_model_sha256'))
        for section, key in fields:
            for value in (None, 'b'*64):
                bad = copy.deepcopy(cert)
                if value is None:
                    bad[section].pop(key)
                else:
                    bad[section][key] = value
                with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                    validate_feature_smoke_admission(experiment, bad)
        bad = copy.deepcopy(cert)
        bad['independent_source_model_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            validate_feature_smoke_admission(experiment, bad)

    def test_old_or_incomplete_gradient_reports_cannot_admit_formal(self):
        experiment, cert = self.certificate('temporal')
        for key in GRADIENT_FIELDS + ('gradient_minibatches',):
            bad = copy.deepcopy(cert)
            bad['mu_loss_report'].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_feature_smoke_admission(experiment, bad)
        for key, value in (('gradient_minibatches', 79), ('es_grad_norm', .1),
                           ('main_grad_norm', 0.), ('combined_grad_norm_preclip', 0.)):
            bad = copy.deepcopy(cert)
            bad['mu_loss_report'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_feature_smoke_admission(experiment, bad)


if __name__ == '__main__':
    unittest.main()
