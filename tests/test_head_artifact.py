"""Synthetic offline archive contract tests; NO real-model/training evidence."""
import copy
import hashlib
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch

from humanoid.amp.head_artifact import (
    ARTIFACT_KIND, HEAD_SHAPES, ORIGINAL110_PARENT, assemble_head_artifact,
    validate_head_artifact,
)
from humanoid.amp.mu_temporal import state_fingerprint


def synthetic_source():
    """33 named fake tensors and populated fake learning slots, not a policy."""
    model = OrderedDict(std=torch.ones(12))
    for module in ('actor', 'critic', 'long_history', 'state_estimator'):
        indices = (0, 2, 5, 7) if module == 'long_history' else (0, 2, 4, 6)
        for index in indices:
            for suffix in ('weight', 'bias'):
                name = '%s.%d.%s' % (module, index, suffix)
                shape = HEAD_SHAPES.get(name, (2, 3) if suffix == 'weight' else (2,))
                model[name] = torch.arange(torch.tensor(shape).prod().item(),
                                           dtype=torch.float32).reshape(shape)/100
    model._metadata = OrderedDict((name, {'version': 1}) for name in ('', 'actor', 'long_history'))

    def adam():
        return dict(state={0: dict(step=torch.tensor(2500.),
                                  exp_avg=torch.tensor([-.0, .2]),
                                  exp_avg_sq=torch.tensor([.1, .3]))},
                    param_groups=[dict(params=[0], lr=5e-5, betas=(.9, .999),
                                       eps=1e-8, weight_decay=0, amsgrad=False)])

    return dict(model_state_dict=model, optimizer_state_dict=adam(),
                es_optimizer_state_dict=adam(), amp_optimizer_state_dict=adam(),
                amp_discriminator_state_dict=OrderedDict(weight=torch.tensor([.4, .5])),
                amp_replay_state=dict(capacity=4, count=4, cursor=0,
                                      windows=torch.arange(1560.).reshape(4, 10, 39)),
                amp_identity=dict(experiment='synthetic_not_a_training_run',
                                  config_sha256='9'*64),
                iter=2499, completed_updates=2500, mu_loss_report={},
                infos=dict(purpose='synthetic_only', signed_zero=-.0,
                           unknown_metadata=[None, True, (3, 'archive')],
                           legacy_certificate=dict(native_status='synthetic')))


def synthetic_provenance(mode='formal', admitted=True):
    n, duration = (4, 2) if mode == 'smoke' else (8, 20)

    def cohort(split, seed, sha):
        return dict(sha256=sha*64, seed=seed, num_envs=n, duration_s=duration,
                    episode_ids=['head_cohort_v2/%s/seed-%d/env-%d/episode-0' % (split, seed, i)
                                 for i in range(n)])

    return dict(code_commit='a'*40, implementation_fingerprint='b'*64, mode=mode,
                solver=dict(kind='quadratic_direction_per_axis_v2', temporal_weight=1.,
                            ridge=1e-6, filter_window=21, action_scale=.5, dtype='float64',
                            solve_count=1, direction_scope='per_output_axis_train_only',
                            direction_scale_per_joint=[.25]*12),
                budget=dict(num_envs=n, duration_s=duration, fit_episode_count=n),
                cohorts=dict(train=cohort('train', 306, 'c'),
                             validation=cohort('validation', 506, 'd')),
                source_parity=dict(full_obs_verified=True, hidden_verified=True,
                                   action_verified=True, max_abs_error=.0001),
                report_sha256='e'*64, offline_admitted=admitted,
                effectiveness_verified=False, dr_unlocked=False)


class OfflineHeadArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'synthetic_parent.pt'
        self.source = synthetic_source()
        torch.save(self.source, self.path)
        self.parent = dict(ORIGINAL110_PARENT)
        self.parent.update(source_task='SYNTHETIC_ONLY', source_config='synthetic_only.json',
                           checkpoint_name=self.path.name,
                           file_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                           model_state_sha256=state_fingerprint(self.source['model_state_dict']))
        self.head = {key: (self.source['model_state_dict'][key]+.05).requires_grad_()
                     for key in HEAD_SHAPES}
        self.provenance = synthetic_provenance()

    def assemble(self, provenance=None):
        return assemble_head_artifact(self.path, self.head,
            provenance=self.provenance if provenance is None else provenance,
            expected_parent=self.parent)

    def validate(self, artifact):
        return validate_head_artifact(artifact, self.path, expected_parent=self.parent)

    def reject(self, artifact):
        with self.assertRaises(ValueError):
            self.validate(artifact)

    def test_smoke_formal_and_nonadmitted_roundtrips_only_two_head_tensors(self):
        for mode in ('smoke', 'formal'):
            for admitted in (False, True):
                with self.subTest(mode=mode, admitted=admitted):
                    artifact = self.assemble(synthetic_provenance(mode, admitted))
                    audit = self.validate(artifact)
                    self.assertEqual(artifact['artifact_kind'], ARTIFACT_KIND)
                    self.assertEqual(audit['unchanged_policy_tensor_count'], 31)
                    self.assertFalse(audit['learning_states_resumed'])
                    self.assertTrue(audit['provenance_is_metadata_not_independent_forward_evidence'])
                    self.assertNotIn('completed_updates', artifact)
                    self.assertNotIn('iter', artifact)
                    self.assertEqual(artifact['source_archive']['completed_updates'], 2500)
                    self.assertEqual(artifact['source_archive']['iter'], 2499)
                    self.assertEqual(artifact['amp_identity'], self.source['amp_identity'])
                    changed = [name for name, value in artifact['model_state_dict'].items()
                               if not torch.equal(value, self.source['model_state_dict'][name])]
                    self.assertEqual(set(changed), set(HEAD_SHAPES))
                    out = Path(self.temp.name)/'synthetic_offline.pt'
                    torch.save(artifact, out)
                    reloaded = torch.load(out, map_location='cpu', weights_only=True)
                    self.assertEqual(self.validate(reloaded), audit)

    def test_no_forward_autograd_rng_grad_or_input_file_mutation(self):
        for tensor in self.head.values():
            tensor.grad = torch.full_like(tensor, .7)
        rng = torch.get_rng_state().clone()
        grads = {key: value.grad.clone() for key, value in self.head.items()}
        original_bytes = self.path.read_bytes()
        with mock.patch('torch.autograd.grad', side_effect=AssertionError('no autograd')):
            artifact = self.assemble()
            self.validate(artifact)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(self.path.read_bytes(), original_bytes)
        for key, tensor in self.head.items():
            self.assertTrue(torch.equal(grads[key], tensor.grad))
            self.assertFalse(artifact['model_state_dict'][key].requires_grad)
            self.assertNotEqual(tensor.data_ptr(), artifact['model_state_dict'][key].data_ptr())
        # Caller mutation cannot silently modify any copied provenance/archive.
        self.provenance['solver']['direction_scale_per_joint'][0] = 1.
        self.head['actor.6.weight'].data.fill_(2.)
        self.assertEqual(artifact['provenance']['solver']['direction_scale_per_joint'], [.25]*12)
        self.validate(artifact)

    def test_every_nonhead_policy_tensor_mutation_rejected_even_with_rehashed_model(self):
        pristine = self.assemble()
        others = set(pristine['model_state_dict'])-set(HEAD_SHAPES)
        self.assertEqual(len(others), 31)
        for name in others:
            with self.subTest(name=name):
                artifact = copy.deepcopy(pristine)
                artifact['model_state_dict'][name].reshape(-1)[0] += .1
                artifact['candidate_model_state_sha256'] = state_fingerprint(artifact['model_state_dict'])
                self.reject(artifact)

    def test_all_learning_archives_and_unknown_metadata_are_immutable(self):
        pristine = self.assemble()
        for key in ('optimizer_state_dict', 'es_optimizer_state_dict', 'amp_optimizer_state_dict'):
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                with self.subTest(key=key, field=field):
                    artifact = copy.deepcopy(pristine)
                    artifact['source_archive'][key]['state'][0][field].reshape(-1)[0] += 1
                    self.reject(artifact)
            artifact = copy.deepcopy(pristine)
            artifact['source_archive'][key]['param_groups'][0]['lr'] = .01
            self.reject(artifact)
        for key in ('amp_discriminator_state_dict', 'amp_replay_state'):
            artifact = copy.deepcopy(pristine)
            name = 'weight' if key == 'amp_discriminator_state_dict' else 'windows'
            artifact['source_archive'][key][name].reshape(-1)[0] += 1
            self.reject(artifact)
        for change in ('count', 'cursor', 'capacity'):
            artifact = copy.deepcopy(pristine)
            artifact['source_archive']['amp_replay_state'][change] += 1
            self.reject(artifact)
        artifact = copy.deepcopy(pristine)
        artifact['source_archive']['infos']['unknown_metadata'][2] = (3, 'changed')
        self.reject(artifact)
        artifact = copy.deepcopy(pristine)
        artifact['source_archive']['infos']['legacy_certificate']['native_status'] = 'forged'
        self.reject(artifact)
        artifact = copy.deepcopy(pristine)
        artifact['source_archive']['infos']['signed_zero'] = .0
        self.reject(artifact)

    def test_source_archive_and_candidate_version_metadata_preserved(self):
        pristine = self.assemble()
        for location in ('source_archive', 'model_state_dict'):
            artifact = copy.deepcopy(pristine)
            state = (artifact[location]['model_state_dict'] if location == 'source_archive'
                     else artifact[location])
            state._metadata['actor']['version'] = 2
            self.reject(artifact)
        artifact = copy.deepcopy(pristine)
        artifact['source_archive']['model_state_dict']['actor.6.bias'][0] += 1
        self.reject(artifact)

    def test_nonhead_dtype_shape_missing_keys_and_candidate_mapping_rejected(self):
        pristine = self.assemble()
        for change in ('dtype', 'shape', 'missing', 'rename', 'mapping'):
            artifact = copy.deepcopy(pristine)
            model = artifact['model_state_dict']
            if change == 'dtype':
                model['std'] = model['std'].double()
            elif change == 'shape':
                model['std'] = model['std'].reshape(2, 6)
            elif change == 'missing':
                del model['std']
            elif change == 'rename':
                model['unbound_std'] = model.pop('std')
            else:
                artifact['model_state_dict'] = dict(model)
            artifact['candidate_model_state_sha256'] = state_fingerprint(artifact['model_state_dict'])
            with self.subTest(change=change):
                self.reject(artifact)

    def test_dtype_shape_signed_zero_and_archive_container_type_rejected(self):
        pristine = self.assemble()
        for kind in ('dtype', 'shape', 'signed_zero'):
            artifact = copy.deepcopy(pristine)
            slot = artifact['source_archive']['optimizer_state_dict']['state'][0]
            if kind == 'dtype':
                slot['exp_avg'] = slot['exp_avg'].double()
            elif kind == 'shape':
                slot['exp_avg'] = slot['exp_avg'].reshape(1, 2)
            else:
                slot['exp_avg'][0] = .0
            self.reject(artifact)
        artifact = copy.deepcopy(pristine)
        artifact['source_archive']['optimizer_state_dict']['param_groups'][0]['betas'] = [.9, .999]
        self.reject(artifact)
        artifact = copy.deepcopy(pristine)
        artifact['source_archive']['amp_discriminator_state_dict'] = dict(
            artifact['source_archive']['amp_discriminator_state_dict'])
        self.reject(artifact)

    def test_missing_any_learning_archive_rejected_and_parent_checks_source(self):
        for key in ('optimizer_state_dict', 'es_optimizer_state_dict', 'amp_optimizer_state_dict',
                    'amp_discriminator_state_dict', 'amp_replay_state', 'amp_identity'):
            with self.subTest(key=key):
                modified = copy.deepcopy(self.source)
                del modified[key]
                torch.save(modified, self.path)
                parent = dict(self.parent, file_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
                with self.assertRaises(ValueError):
                    assemble_head_artifact(self.path, self.head, provenance=self.provenance,
                                           expected_parent=parent)

    def test_actual_parent_file_hash_and_model_fingerprint_strictly_bound(self):
        artifact = self.assemble()
        with self.assertRaisesRegex(ValueError, 'file SHA256'):
            validate_head_artifact(artifact, self.path)  # Synthetic is NEVER original110.
        for key, value in (('file_sha256', '0'*64), ('model_state_sha256', '0'*64)):
            parent = dict(self.parent, **{key: value})
            with self.assertRaises(ValueError):
                validate_head_artifact(artifact, self.path, expected_parent=parent)
        for key, value in (('source_task', 'wrong'), ('source_config', 'wrong.json'),
                           ('checkpoint_name', 'wrong.pt'), ('iter', 2500),
                           ('completed_updates', 2750), ('policy_tensor_count', 32)):
            modified = copy.deepcopy(artifact)
            modified['parent'][key] = value
            self.reject(modified)
        # Re-serializing a changed source and changing ONLY its expected file hash
        # still cannot bypass the independently pinned source policy fingerprint.
        altered = copy.deepcopy(self.source)
        altered['model_state_dict']['std'][0] = 2
        torch.save(altered, self.path)
        parent = dict(self.parent, file_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'model-state fingerprint'):
            assemble_head_artifact(self.path, self.head, provenance=self.provenance, expected_parent=parent)

    def test_native_2750_counters_old_certificates_or_optimizer_top_level_rejected(self):
        pristine = self.assemble()
        for key, value in (('completed_updates', 2750), ('iter', 2749),
                           ('optimizer_state_dict', {}), ('mu_loss_report', {}),
                           ('smoke_certificate', {'status': 'PASS'}),
                           ('formal_certificate', {'status': 'PASS'})):
            artifact = copy.deepcopy(pristine)
            artifact[key] = value
            self.reject(artifact)
        for key, value in (('completed_updates', 2750), ('iter', 2749)):
            artifact = copy.deepcopy(pristine)
            artifact['source_archive'][key] = value
            self.reject(artifact)
        for key in ('artifact_kind', 'archive_role'):
            artifact = copy.deepcopy(pristine)
            artifact[key] = 'native_ppo_2750'
            self.reject(artifact)

    def test_head_shape_dtype_nonfinite_missing_extra_and_fingerprint_rejected(self):
        for change in ('shape', 'dtype', 'nan', 'inf', 'missing', 'extra'):
            head = copy.deepcopy(self.head)
            if change == 'shape':
                head['actor.6.weight'] = torch.zeros(12, 127)
            elif change == 'dtype':
                head['actor.6.bias'] = head['actor.6.bias'].double()
            elif change in ('nan', 'inf'):
                head['actor.6.bias'].data[0] = float(change)
            elif change == 'missing':
                del head['actor.6.bias']
            else:
                head['critic.6.bias'] = torch.zeros(1)
            with self.subTest(change=change), self.assertRaises(ValueError):
                assemble_head_artifact(self.path, head, provenance=self.provenance,
                                       expected_parent=self.parent)
        artifact = self.assemble()
        artifact['model_state_dict']['actor.6.bias'][0] += .01
        self.reject(artifact)
        for key in ('candidate_model_state_sha256', 'source_archive_sha256'):
            artifact = self.assemble()
            artifact[key] = '0'*64
            self.reject(artifact)

    def test_missing_provenance_cohort_split_budget_or_parity_rejected(self):
        for key in self.provenance:
            artifact = self.assemble()
            del artifact['provenance'][key]
            self.reject(artifact)
        for container, key in (('solver', 'ridge'), ('budget', 'fit_episode_count'),
                               ('cohorts', 'validation'), ('source_parity', 'hidden_verified')):
            artifact = self.assemble()
            del artifact['provenance'][container][key]
            self.reject(artifact)
        for key in self.provenance['cohorts']['train']:
            artifact = self.assemble()
            del artifact['provenance']['cohorts']['train'][key]
            self.reject(artifact)

    def test_malformed_budget_and_solver_bounds_rejected(self):
        for key, bad in (('code_commit', 'a'*39), ('implementation_fingerprint', 'bad'),
                         ('report_sha256', 'bad'), ('mode', 'PPO'), ('offline_admitted', 1)):
            artifact = self.assemble()
            artifact['provenance'][key] = bad
            self.reject(artifact)
        for key, bad in (('kind', 'native_ppo'), ('dtype', 'float32'), ('ridge', 0.),
                         ('temporal_weight', .05), ('filter_window', 20),
                         ('solve_count', 2), ('solve_count', True), ('action_scale', 1.),
                         ('direction_scope', 'global_train_only'),
                         ('direction_scale_per_joint', [-1.]*12),
                         ('direction_scale_per_joint', [1.01]*12),
                         ('direction_scale_per_joint', [float('nan')]*12),
                         ('direction_scale_per_joint', [True]*12)):
            artifact = self.assemble()
            artifact['provenance']['solver'][key] = bad
            self.reject(artifact)
        for key, bad in (('num_envs', 4), ('duration_s', 2.), ('fit_episode_count', 7),
                         ('fit_episode_count', True)):
            artifact = self.assemble()
            artifact['provenance']['budget'][key] = bad
            self.reject(artifact)

    def test_axis_v2_vector_has_exact_float12_shape_and_train_only_scope(self):
        provenance = copy.deepcopy(self.provenance)
        provenance['solver']['direction_scale_per_joint'] = [i/11. for i in range(12)]
        artifact = self.assemble(provenance)
        self.assertEqual(artifact['artifact_kind'], 'actor_head_offline_v2')
        self.assertEqual(artifact['provenance']['solver']['direction_scale_per_joint'],
                         [i/11. for i in range(12)])
        self.validate(artifact)
        for bad in (.25, None, tuple([.25]*12), [.25]*11, [.25]*13,
                    [[.25]]*12, [0]*12, [True]*12, [float('inf')]*12,
                    [float('nan')]*12, [torch.tensor(.25)]*12):
            current = copy.deepcopy(artifact)
            current['provenance']['solver']['direction_scale_per_joint'] = bad
            with self.subTest(value_type=type(bad).__name__, value_repr=str(bad)[:40]):
                self.reject(current)
        for scope in ('global_train_only', 'per_output_axis_train_validation',
                      'per_output_axis_sealed', None, True):
            current = copy.deepcopy(artifact)
            current['provenance']['solver']['direction_scope'] = scope
            with self.subTest(scope=scope):
                self.reject(current)

    def test_legacy_head_scalar_and_cohort_cannot_be_auto_converted(self):
        pristine = self.assemble()
        for change in ('artifact_kind', 'solver_kind', 'scalar_extra', 'scalar_only',
                       'cohort_prefix', 'train_seed', 'validation_seed'):
            current = copy.deepcopy(pristine)
            solver = current['provenance']['solver']
            if change == 'artifact_kind':
                current['artifact_kind'] = 'actor_head_offline_v1'
            elif change == 'solver_kind':
                solver['kind'] = 'quadratic_direction_scaled_v1'
            elif change == 'scalar_extra':
                solver['direction_scale'] = .25
            elif change == 'scalar_only':
                del solver['direction_scale_per_joint']
                del solver['direction_scope']
                solver['direction_scale'] = .25
            elif change == 'cohort_prefix':
                for cohort in current['provenance']['cohorts'].values():
                    cohort['episode_ids'] = [v.replace('head_cohort_v2/', 'head_cohort_v1/')
                                            for v in cohort['episode_ids']]
            else:
                split, seed = ('train', 305) if change == 'train_seed' else ('validation', 505)
                current['provenance']['cohorts'][split]['seed'] = seed
            with self.subTest(change=change):
                self.reject(current)

    def test_train_validation_sealed_and_source_regression_ids_cannot_overlap(self):
        for change in ('same_hash', 'same_ids', 'duplicate', 'source', 'sealed', 'wrong_seed',
                       'wrong_envs', 'wrong_duration', 'bad_hash'):
            artifact = self.assemble()
            cohorts = artifact['provenance']['cohorts']
            train, val = cohorts['train'], cohorts['validation']
            if change == 'same_hash':
                val['sha256'] = train['sha256']
            elif change == 'same_ids':
                val['episode_ids'] = train['episode_ids'][:]
            elif change == 'duplicate':
                train['episode_ids'][1] = train['episode_ids'][0]
            elif change == 'source':
                train['episode_ids'][0] = 'source110/reference/env-0/episode-0'
            elif change == 'sealed':
                val['episode_ids'][0] = 'head_cohort_v2/sealed/seed-706/env-0/episode-0'
            elif change == 'wrong_seed':
                val['seed'] = 306
            elif change == 'wrong_envs':
                train['num_envs'] = 4
            elif change == 'wrong_duration':
                val['duration_s'] = 2
            else:
                val['sha256'] = None
            with self.subTest(change=change):
                self.reject(artifact)

    def test_parity_metadata_tolerance_and_false_effectiveness_dr_mandatory(self):
        for key in ('full_obs_verified', 'hidden_verified', 'action_verified'):
            for bad in (False, 1, 'true'):
                artifact = self.assemble()
                artifact['provenance']['source_parity'][key] = bad
                self.reject(artifact)
        for error in (-.1, .001, float('inf'), float('nan'), True):
            artifact = self.assemble()
            artifact['provenance']['source_parity']['max_abs_error'] = error
            self.reject(artifact)
        for key in ('effectiveness_verified', 'dr_unlocked'):
            for location in ('top', 'provenance'):
                for value in (True, 0, None):
                    artifact = self.assemble()
                    (artifact if location == 'top' else artifact['provenance'])[key] = value
                    self.reject(artifact)

    def test_parent_source_counters_boolean_and_empty_archives_cannot_be_relabeled(self):
        for key, bad in (('iter', 2749), ('completed_updates', 2750), ('iter', True),
                         ('completed_updates', True), ('optimizer_state_dict', {}),
                         ('amp_replay_state', None)):
            source = copy.deepcopy(self.source)
            source[key] = bad
            torch.save(source, self.path)
            parent = dict(self.parent, file_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
            with self.subTest(key=key, bad=type(bad).__name__), self.assertRaises(ValueError):
                assemble_head_artifact(self.path, self.head, provenance=self.provenance,
                                       expected_parent=parent)


if __name__ == '__main__':
    unittest.main()
