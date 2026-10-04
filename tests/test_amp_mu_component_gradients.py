"""Actual CPU graph/Adam checks, not cloud execution or motion-quality evidence."""
import contextlib
import copy
import io
import json
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from humanoid.amp.adapter import AMPAlgorithmAdapter
from humanoid.amp.feature_freeze import (
    COMPONENT_GRADIENT_FIELDS, COMPONENT_GRADIENT_SCHEMA, GRADIENT_FIELDS,
    gradient_diagnostics, validate_component_gradient_records,
    validate_component_gradient_report,
)
from humanoid.amp.mu_loss import DeterministicMuLoss
from test_amp_feature_freeze import byte_snapshot, populated_ppo
from test_amp_feature_gradients import TinyModel, actor_loss, complete_report
from test_amp_mu_loss import LinearInferenceModel
from test_amp_mu_storage import ppo_rollout


torch.set_num_threads(1)


def regularizer(model, coefficient=.01, enabled=True):
    return DeterministicMuLoss(model, [1000., 1000.], 1., coefficient,
        component_gradient_schema=COMPONENT_GRADIENT_SCHEMA if enabled else None)


def configured_ppo(enabled=True, coefficient=.01):
    ppo = populated_ppo()
    ppo.configure_mu_temporal(regularizer(ppo.actor_critic, coefficient, enabled))
    ppo.configure_feature_freeze()
    return ppo


def actual_split(model, anchor=(6., 8.), temporal=(-4., 3.)):
    main = actor_loss(model, (3., 4.))+12*model.critic.weight.sum()
    pieces = dict(weighted_anchor=actor_loss(model, anchor),
                  weighted_temporal=actor_loss(model, temporal))
    auxiliary = pieces['weighted_anchor']+pieces['weighted_temporal']
    return gradient_diagnostics(model, main, auxiliary, torch.tensor(1.), components=pieces)


def record_fixture(updates=2, batches=3, start=2501):
    diagnostic = complete_report(actual_split(TinyModel()), math.sqrt(394.))
    records = [dict(diagnostic, completed_update=start+update, minibatch_index=batch+1,
                    batch_size=4, valid_triplets=2)
               for update in range(updates) for batch in range(batches)]
    aggregate = {field: sum(record[field] for record in records)/len(records)
                 for field in GRADIENT_FIELDS+COMPONENT_GRADIENT_FIELDS}
    aggregate.update(component_gradient_schema=COMPONENT_GRADIENT_SCHEMA,
                     gradient_minibatches=len(records), evaluated_samples=4*len(records),
                     triplet_evaluations=2*len(records))
    return records, aggregate


class ComponentLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = LinearInferenceModel()
        self.now = torch.tensor([[3., 1.], [20., 20.]])
        self.previous = torch.tensor([[1., 0.], [0., 0.]])
        self.older = torch.tensor([[0., 0.], [99., 99.]])
        self.mask = torch.tensor([True, False])

    def test_constructor_and_optional_loss_return_are_explicit(self):
        old = regularizer(self.model, enabled=False)
        self.assertIsNone(old.component_gradient_schema)
        self.assertEqual(len(old.loss(self.now, self.previous, self.older, self.mask)), 2)
        with self.assertRaises(ValueError):
            old.loss(self.now, self.previous, self.older, self.mask, return_components=True)
        for schema in ('unknown', '', False, 1):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                DeterministicMuLoss(self.model, [1000., 1000.], 1., .01,
                                    component_gradient_schema=schema)
        enabled = regularizer(self.model)
        self.assertEqual(enabled.component_gradient_schema, COMPONENT_GRADIENT_SCHEMA)
        self.assertEqual(len(enabled.loss(self.now, self.previous, self.older, self.mask)), 2)
        with self.assertRaises(ValueError):
            enabled.loss(self.now, self.previous, self.older, self.mask, return_components=1)

    def test_same_forward_exposes_weighted_graphs_without_changing_legacy_stats(self):
        loss = regularizer(self.model)
        baseline, baseline_stats = loss.loss(self.now, self.previous, self.older, self.mask)
        before_rng = torch.get_rng_state().clone()
        with mock.patch.object(self.model, 'act_inference', wraps=self.model.act_inference) as student, \
                mock.patch.object(loss.teacher, 'act_inference', wraps=loss.teacher.act_inference) as teacher:
            auxiliary, stats, pieces = loss.loss(self.now, self.previous, self.older, self.mask,
                                                 return_components=True)
            self.assertEqual(student.call_count, 3)
            self.assertEqual(teacher.call_count, 1)
        self.assertEqual(stats, baseline_stats)
        self.assertEqual(set(stats), set(loss.STAT_FIELDS))
        self.assertEqual(set(pieces), {'weighted_anchor', 'weighted_temporal'})
        torch.testing.assert_close(auxiliary, baseline, rtol=0, atol=0)
        torch.testing.assert_close(auxiliary, pieces['weighted_anchor']+pieces['weighted_temporal'], rtol=0, atol=0)
        for key, piece in pieces.items():
            self.assertTrue(piece.requires_grad)
            self.assertEqual(float(piece.detach()), stats[key+'_loss'])
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        self.assertTrue(loss.teacher_unchanged())
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        # Components are local return values; report/latest_stats remain JSON only.
        json.dumps(loss.report(), allow_nan=False)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in loss.report().values()))

    def test_five_times_coefficient_scales_actual_weighted_gradient_not_the_graph(self):
        first, second = copy.deepcopy(self.model), copy.deepcopy(self.model)
        low, high = regularizer(first, .01), regularizer(second, .05)
        for model in (first, second):
            for module in (model.long_history, model.state_estimator):
                for parameter in module.parameters(): parameter.requires_grad_(False)
            with torch.no_grad(): model.actor.weight.add_(.1)
        outputs = [loss.loss(self.now, self.previous, self.older, self.mask, return_components=True)
                   for loss in (low, high)]
        gradients = [torch.autograd.grad(output[2]['weighted_temporal'], tuple(model.actor.parameters()),
                                         retain_graph=True)
                     for model, output in zip((first, second), outputs)]
        self.assertTrue(math.isclose(outputs[1][1]['weighted_temporal_loss'],
                                    5*outputs[0][1]['weighted_temporal_loss'], rel_tol=1e-6, abs_tol=1e-7))
        for a, b in zip(*gradients):
            torch.testing.assert_close(b, 5*a, rtol=1e-6, atol=1e-7)
        self.assertEqual(outputs[0][1]['anchor_loss'], outputs[1][1]['anchor_loss'])
        self.assertEqual(outputs[0][1]['temporal_loss'], outputs[1][1]['temporal_loss'])
        self.assertTrue(all(parameter.grad is None for model in (first, second) for parameter in model.parameters()))

    def test_empty_mask_has_differentiable_zero_temporal_component(self):
        loss = regularizer(self.model)
        mask = torch.zeros_like(self.mask)
        with mock.patch.object(self.model, 'act_inference', wraps=self.model.act_inference) as student:
            _, stats, pieces = loss.loss(self.now, self.previous, self.older, mask, return_components=True)
            self.assertEqual(student.call_count, 1)
        self.assertEqual(stats['valid_triplets'], 0)
        gradients = torch.autograd.grad(pieces['weighted_temporal'], tuple(self.model.actor.parameters()))
        self.assertTrue(all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in gradients))

    def test_empty_batch_preserves_both_return_contracts_and_zero_gradients(self):
        loss = regularizer(self.model)
        empty, mask = self.now[:0], self.mask[:0]
        old = loss.loss(empty, empty, empty, mask)
        auxiliary, stats, pieces = loss.loss(empty, empty, empty, mask, return_components=True)
        self.assertEqual(stats, old[1])
        self.assertEqual(float(auxiliary), 0.)
        self.assertTrue(all(piece.requires_grad for piece in pieces.values()))
        gradients = torch.autograd.grad(auxiliary, tuple(self.model.actor.parameters()), allow_unused=True)
        self.assertTrue(all(gradient is None or torch.equal(gradient, torch.zeros_like(gradient)) for gradient in gradients))


class ComponentAutogradTests(unittest.TestCase):
    def test_weighted_actor_pairs_have_hand_calculated_norms_and_cosines(self):
        diagnostic = actual_split(TinyModel())
        self.assertEqual(set(diagnostic), set(GRADIENT_FIELDS+COMPONENT_GRADIENT_FIELDS)-
                         {'combined_grad_norm_preclip', 'global_clip_factor'})
        self.assertEqual(diagnostic['actor_main_grad_norm'], 5.)
        self.assertEqual(diagnostic['main_grad_norm'], 13.)
        self.assertEqual(diagnostic['actor_anchor_grad_norm'], 10.)
        self.assertEqual(diagnostic['actor_temporal_grad_norm'], 5.)
        self.assertEqual(diagnostic['actor_main_anchor_cosine'], 1.)
        self.assertEqual(diagnostic['actor_main_temporal_cosine'], 0.)
        self.assertEqual(diagnostic['actor_anchor_temporal_cosine'], 0.)
        self.assertAlmostEqual(diagnostic['actor_aux_grad_norm'], math.sqrt(125.))
        self.assertTrue(all(diagnostic[key] == 1. for key in COMPONENT_GRADIENT_FIELDS if key.endswith('_valid_fraction')))

    def test_opposite_components_can_cancel_aux_without_becoming_undefined_individually(self):
        diagnostic = actual_split(TinyModel(), (3., 4.), (-3., -4.))
        self.assertEqual(diagnostic['actor_aux_grad_norm'], 0.)
        self.assertEqual(diagnostic['actor_cosine_valid_fraction'], 0.)
        self.assertEqual(diagnostic['actor_main_anchor_cosine'], 1.)
        self.assertEqual(diagnostic['actor_main_temporal_cosine'], -1.)
        self.assertEqual(diagnostic['actor_anchor_temporal_cosine'], -1.)

    def test_zero_component_has_separate_undefined_cosine_flags(self):
        diagnostic = actual_split(TinyModel(), (0., 0.), (-4., 3.))
        self.assertEqual(diagnostic['actor_anchor_grad_norm'], 0.)
        for pair in ('main_anchor', 'anchor_temporal'):
            self.assertEqual(diagnostic['actor_'+pair+'_cosine'], 0.)
            self.assertEqual(diagnostic['actor_'+pair+'_cosine_valid_fraction'], 0.)
        self.assertEqual(diagnostic['actor_main_temporal_cosine_valid_fraction'], 1.)

    def test_reading_split_gradients_preserves_state_grads_rng_and_real_backward(self):
        model = TinyModel()
        model.actor.weight.grad = torch.tensor([9., -0.], dtype=torch.float64)
        model.std.grad = torch.tensor([-.7], dtype=torch.float64)
        before = byte_snapshot(model.state_dict())
        saved_grads = {name: (parameter.grad, byte_snapshot(parameter.grad))
                       for name, parameter in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        main = actor_loss(model, (3., 4.))+12*model.critic.weight.sum()
        pieces = dict(weighted_anchor=actor_loss(model, (6., 8.)),
                      weighted_temporal=actor_loss(model, (-4., 3.)))
        auxiliary = pieces['weighted_anchor']+pieces['weighted_temporal']
        gradient_diagnostics(model, main, auxiliary, torch.tensor(1.), components=pieces)
        self.assertEqual(before, byte_snapshot(model.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, parameter in model.named_parameters():
            self.assertIs(parameter.grad, saved_grads[name][0])
            self.assertEqual(byte_snapshot(parameter.grad), saved_grads[name][1])
        model.zero_grad(set_to_none=True)
        (main+auxiliary).backward()
        torch.testing.assert_close(model.actor.weight.grad, torch.tensor([5., 15.], dtype=torch.float64))
        torch.testing.assert_close(model.critic.weight.grad, torch.tensor([12.], dtype=torch.float64))

    def test_equal_scalar_value_cannot_hide_wrong_component_graph(self):
        model = TinyModel()
        anchor = actor_loss(model, (6., 8.))
        temporal = actor_loss(model, (-4., 3.))
        auxiliary = anchor+temporal
        wrong_anchor = -anchor+2*anchor.detach()
        self.assertEqual(float(wrong_anchor), float(anchor))
        with self.assertRaisesRegex(ValueError, 'sum to the actual auxiliary gradient'):
            gradient_diagnostics(model, actor_loss(model, (3., 4.)), auxiliary, torch.tensor(1.),
                                 components=dict(weighted_anchor=wrong_anchor, weighted_temporal=temporal))

    def test_detached_changed_or_nonactor_components_are_refused(self):
        for change in ('detach', 'value', 'nonactor', 'missing'):
            with self.subTest(change=change):
                model = TinyModel()
                anchor, temporal = actor_loss(model, (6., 8.)), actor_loss(model, (-4., 3.))
                if change == 'nonactor': anchor = anchor+model.critic.weight.sum()
                auxiliary = anchor+temporal
                pieces = dict(weighted_anchor=anchor, weighted_temporal=temporal)
                if change == 'detach': pieces['weighted_anchor'] = anchor.detach()
                if change == 'value': pieces['weighted_anchor'] = anchor+1
                if change == 'missing': pieces.pop('weighted_anchor')
                with self.assertRaises(ValueError):
                    gradient_diagnostics(model, actor_loss(model, (3., 4.)), auxiliary, torch.tensor(1.),
                                         components=pieces)


class ComponentRecordTests(unittest.TestCase):
    def test_exact_ordered_records_and_prefixed_aggregates_are_checked(self):
        records, aggregate = record_fixture()
        self.assertTrue(validate_component_gradient_report(aggregate))
        self.assertTrue(validate_component_gradient_records(records, 2, 3, aggregate=aggregate))
        prefixed = {'mu_'+key: value for key, value in aggregate.items()}
        self.assertTrue(validate_component_gradient_records(records, 2, 3, aggregate=prefixed, prefix='mu_'))
        json.dumps(records, allow_nan=False)
        self.assertTrue(all(type(record[field]) is float for record in records
                            for field in GRADIENT_FIELDS+COMPONENT_GRADIENT_FIELDS))

    def test_omitted_extra_duplicate_reordered_and_wrong_update_records_are_refused(self):
        records, aggregate = record_fixture()
        broken_lists = [records[:-1], records+[records[-1]], records[:1]+records[:1]+records[2:],
                        list(reversed(records))]
        altered = copy.deepcopy(records); altered[3]['completed_update'] += 1
        broken_lists.append(altered)
        for values in broken_lists:
            with self.subTest(length=len(values)), self.assertRaises(ValueError):
                validate_component_gradient_records(values, 2, 3, aggregate=aggregate)

    def test_every_scalar_and_metadata_field_is_required_and_typed(self):
        records, _ = record_fixture()
        for key in records[0]:
            with self.subTest(key=key):
                broken = copy.deepcopy(records); broken[0].pop(key)
                with self.assertRaises(ValueError): validate_component_gradient_records(broken, 2, 3)
        for key, value in (('actor_anchor_grad_norm', 10), ('global_clip_factor', True),
                           ('minibatch_index', 1.), ('completed_update', True), ('batch_size', 0),
                           ('valid_triplets', 5), ('actor_main_anchor_cosine', float('nan'))):
            with self.subTest(key=key, value=value):
                broken = copy.deepcopy(records); broken[0][key] = value
                with self.assertRaises(ValueError): validate_component_gradient_records(broken, 2, 3)

    def test_invalid_undefined_cosines_and_absent_triplet_gradient_are_refused(self):
        records, _ = record_fixture()
        mutations = (('actor_main_anchor_cosine_valid_fraction', .5),
                     ('actor_anchor_grad_norm', 0.), ('actor_temporal_grad_norm', 0.),
                     ('valid_triplets', 0), ('actor_cosine_valid_fraction', .5))
        for key, value in mutations:
            with self.subTest(key=key):
                broken = copy.deepcopy(records); broken[0][key] = value
                with self.assertRaises(ValueError): validate_component_gradient_records(broken, 2, 3)

    def test_all_field_means_counts_and_sample_totals_are_reconciled(self):
        records, aggregate = record_fixture()
        for key in GRADIENT_FIELDS+COMPONENT_GRADIENT_FIELDS:
            with self.subTest(key=key):
                broken = dict(aggregate); broken[key] += .001
                with self.assertRaises(ValueError): validate_component_gradient_records(records, 2, 3, aggregate=broken)
        for key in ('gradient_minibatches', 'triplet_evaluations', 'evaluated_samples'):
            with self.subTest(key=key):
                broken = dict(aggregate); broken[key] += 1
                with self.assertRaises(ValueError): validate_component_gradient_records(records, 2, 3, aggregate=broken)
        for schema in (None, 'legacy', False):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                validate_component_gradient_records(records, 2, 3,
                    aggregate=dict(aggregate, component_gradient_schema=schema))

    def test_empty_run_is_not_an_actual_empty_minibatch(self):
        aggregate = {field: 0. for field in GRADIENT_FIELDS+COMPONENT_GRADIENT_FIELDS}
        aggregate.update(component_gradient_schema=COMPONENT_GRADIENT_SCHEMA, gradient_minibatches=0)
        self.assertTrue(validate_component_gradient_records([], 0, aggregate=aggregate))
        for arguments in ((True, 8, 2501), (1, 0, 2501), (1, 8, False)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_component_gradient_records([], *arguments)


class ComponentPPOTests(unittest.TestCase):
    def test_opt_in_telemetry_keeps_real_adam_updates_rng_and_legacy_fields_byte_exact(self):
        torch.manual_seed(22); old = configured_ppo(enabled=False)
        torch.manual_seed(22); enabled = configured_ppo()
        for update in range(2):
            for ppo in (old, enabled):
                torch.manual_seed(321+update)
                with torch.inference_mode(): ppo_rollout(ppo)
            torch.manual_seed(91+update); old_losses = old.update(); old_rng = torch.get_rng_state().clone()
            torch.manual_seed(91+update); enabled_losses = enabled.update()
            self.assertEqual(old_losses, enabled_losses)
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            self.assertEqual(byte_snapshot(old.actor_critic.state_dict()), byte_snapshot(enabled.actor_critic.state_dict()))
            self.assertEqual(byte_snapshot(old.optimizer.state_dict()), byte_snapshot(enabled.optimizer.state_dict()))
            self.assertEqual(byte_snapshot(old.state_estimator_optimizer.state_dict()),
                             byte_snapshot(enabled.state_estimator_optimizer.state_dict()))
            for (name, a), (other, b) in zip(old.actor_critic.named_parameters(), enabled.actor_critic.named_parameters()):
                self.assertEqual(name, other)
                self.assertEqual(byte_snapshot(a.grad), byte_snapshot(b.grad))
            for key, value in old.mu_latest.items(): self.assertEqual(enabled.mu_latest[key], value, key)
            self.assertNotIn('component_gradient_records', old.mu_latest)
            records = enabled.mu_latest['component_gradient_records']
            self.assertEqual(len(records), 6)
            self.assertEqual([record['minibatch_index'] for record in records], list(range(1, 7)))
            self.assertTrue(all('completed_update' not in record for record in records))
            self.assertTrue(enabled.feature_freeze_guard.validate())
            self.assertTrue(enabled.mu_regularizer.teacher_unchanged())
            json.dumps(enabled.mu_latest, allow_nan=False)

    def test_opt_in_cannot_run_without_feature_freeze(self):
        torch.manual_seed(22); ppo = populated_ppo()
        ppo.configure_mu_temporal(regularizer(ppo.actor_critic))
        ppo_rollout(ppo)
        before = byte_snapshot(ppo.actor_critic.state_dict())
        rng = torch.get_rng_state().clone()
        with mock.patch.object(ppo.optimizer, 'step', wraps=ppo.optimizer.step) as step:
            with self.assertRaises(ValueError): ppo.update()
        step.assert_not_called()
        self.assertEqual(before, byte_snapshot(ppo.actor_critic.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_adapter_labels_actual_restored_updates_logs_and_final_cumulative_records(self):
        torch.manual_seed(22); ppo = configured_ppo()
        bridge = SimpleNamespace(update_discriminator=lambda *args: {'loss': 0., 'skipped': False})
        experiment = SimpleNamespace(cfg={'discriminator_batch_size': 4, 'seed': 5})
        adapter = AMPAlgorithmAdapter(ppo, bridge, None, experiment)
        adapter.updates = 2500  # Synthetic CPU restoration marker, not a real source artifact.
        for update in range(2):
            torch.manual_seed(50+update)
            ppo_rollout(ppo)
            adapter.rollout_steps = 1
            adapter.rollout_sums = {'task_reward': 1.}
            with contextlib.redirect_stdout(io.StringIO()) as stream: adapter.update()
            row = json.loads(stream.getvalue().split('[f1-amp-update] ', 1)[1])
            self.assertEqual(row['iteration'], 2501+update)
            records = row['mu_component_gradient_records']
            self.assertTrue(validate_component_gradient_records(records, 1, 6, 2501+update,
                                                                 aggregate=row, prefix='mu_'))
            self.assertEqual(row['mu_component_gradient_schema'], COMPONENT_GRADIENT_SCHEMA)
        report = adapter.mu_loss_report()
        self.assertTrue(validate_component_gradient_records(report['component_gradient_records'], 2, 6,
                                                             aggregate=report))
        self.assertEqual(report['gradient_minibatches'], 12)
        self.assertEqual(len(report['component_gradient_records']), 12)
        self.assertEqual([record['completed_update'] for record in report['component_gradient_records']],
                         [2501]*6+[2502]*6)
        report['component_gradient_records'][0]['actor_anchor_grad_norm'] = 999.
        self.assertNotEqual(adapter.mu_component_records[0]['actor_anchor_grad_norm'], 999.)
        json.dumps(adapter.mu_loss_report(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
