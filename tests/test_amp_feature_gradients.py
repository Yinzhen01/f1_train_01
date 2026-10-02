"""Exact CPU autograd checks; no simulator or optimization steps are used."""
import math
import unittest

import torch
from torch import nn

from humanoid.amp.feature_freeze import (
    GRADIENT_FIELDS, gradient_diagnostics, validate_gradient_report,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Module()
        self.actor.weight=nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        self.critic=nn.Module()
        self.critic.weight=nn.Parameter(torch.tensor([.25], dtype=torch.float64))
        self.state_estimator=nn.Module()
        self.state_estimator.weight=nn.Parameter(torch.tensor([.5, -.3], dtype=torch.float64),
                                                requires_grad=False)
        self.std=nn.Parameter(torch.tensor([.1], dtype=torch.float64))
        self.register_buffer('audit_marker', torch.tensor([17]))


def actor_loss(model, coefficients):
    return (model.actor.weight*model.actor.weight.new_tensor(coefficients)).sum()


def complete_report(diagnostic, combined_norm):
    result=dict(diagnostic)
    result.update(combined_grad_norm_preclip=combined_norm,
                  global_clip_factor=min(1., 1./(combined_norm+1e-6)))
    return result


class GradientDiagnosticTests(unittest.TestCase):
    def test_same_opposite_and_orthogonal_actor_gradients_have_hand_calculated_cosines(self):
        # Main derivative is (3,4), norm5. The auxiliary derivatives are explicit.
        for coefficients, expected_cosine, expected_aux, expected_combined in (
                ((6., 8.), 1., 10., 15.),
                ((-6., -8.), -1., 10., 5.),
                ((-4., 3.), 0., 5., math.sqrt(50.))):
            with self.subTest(coefficients=coefficients):
                model=TinyModel()
                diagnostic=gradient_diagnostics(model, actor_loss(model, (3., 4.)),
                    actor_loss(model, coefficients), torch.tensor(2.))
                self.assertAlmostEqual(diagnostic['main_grad_norm'], 5.)
                self.assertAlmostEqual(diagnostic['aux_grad_norm_all_batches'], expected_aux)
                self.assertAlmostEqual(diagnostic['actor_main_grad_norm'], 5.)
                self.assertAlmostEqual(diagnostic['actor_aux_grad_norm'], expected_aux)
                self.assertAlmostEqual(diagnostic['actor_main_aux_cosine'], expected_cosine)
                self.assertEqual(diagnostic['actor_cosine_valid_fraction'], 1.)
                self.assertEqual(diagnostic['es_grad_norm'], 0.)
                self.assertTrue(validate_gradient_report(complete_report(diagnostic, expected_combined)))

    def test_all_parameter_norm_is_distinct_from_actor_norm_and_unused_parameters_are_allowed(self):
        model=TinyModel()
        # Actor derivative norm5 and critic derivative12 give total norm13.
        main=actor_loss(model, (3., 4.))+12*model.critic.weight.sum()
        diagnostic=gradient_diagnostics(model, main, actor_loss(model, (6., 8.)), torch.tensor(1.))
        self.assertAlmostEqual(diagnostic['main_grad_norm'], 13.)
        self.assertAlmostEqual(diagnostic['actor_main_grad_norm'], 5.)
        self.assertAlmostEqual(diagnostic['aux_grad_norm_all_batches'], 10.)
        self.assertAlmostEqual(diagnostic['actor_main_aux_cosine'], 1.)
        # Combined actor derivative (9,12), critic derivative12: sqrt(369).
        self.assertTrue(validate_gradient_report(complete_report(diagnostic, math.sqrt(369.))))

    def test_constant_and_differentiable_zero_auxiliary_have_no_valid_actor_cosine(self):
        for differentiable in (False, True):
            with self.subTest(differentiable=differentiable):
                model=TinyModel()
                aux=actor_loss(model, (0., 0.)) if differentiable else torch.tensor(0.)
                diagnostic=gradient_diagnostics(model, actor_loss(model, (3., 4.)), aux, torch.tensor(1.))
                self.assertEqual(diagnostic['aux_grad_norm_all_batches'], 0.)
                self.assertEqual(diagnostic['actor_aux_grad_norm'], 0.)
                self.assertEqual(diagnostic['actor_cosine_valid_fraction'], 0.)
                self.assertEqual(diagnostic['actor_main_aux_cosine'], 0.)
                self.assertTrue(validate_gradient_report(complete_report(diagnostic, 5.)))

    def test_frozen_estimator_loss_is_constant_and_has_zero_supervised_gradient(self):
        model=TinyModel()
        es_loss=(model.state_estimator.weight-model.state_estimator.weight.new_tensor([2., 3.])).square().mean()
        self.assertFalse(es_loss.requires_grad)
        diagnostic=gradient_diagnostics(model, actor_loss(model, (3., 4.))+es_loss,
            actor_loss(model, (-4., 3.)), es_loss)
        self.assertEqual(diagnostic['es_grad_norm'], 0.)
        self.assertEqual(diagnostic['main_grad_norm'], 5.)
        self.assertIsNone(model.state_estimator.weight.grad)
        self.assertTrue(validate_gradient_report(complete_report(diagnostic, math.sqrt(50.))))

    def test_autograd_diagnostics_preserve_parameters_buffers_existing_grads_and_rng(self):
        model=TinyModel()
        model.actor.weight.grad=torch.tensor([9., -0.], dtype=torch.float64)
        model.std.grad=torch.tensor([-.7], dtype=torch.float64)
        params=dict(model.named_parameters())
        saved_state={name: value.detach().clone() for name, value in model.state_dict().items()}
        saved_grads={name: (parameter.grad, None if parameter.grad is None else parameter.grad.clone())
                     for name, parameter in params.items()}
        rng=torch.random.get_rng_state().clone()
        main=actor_loss(model, (3., 4.))+12*model.critic.weight.sum()
        aux=actor_loss(model, (-4., 3.))
        diagnostic=gradient_diagnostics(model, main, aux, torch.tensor(1.))
        self.assertEqual(diagnostic['actor_main_aux_cosine'], 0.)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.reshape(-1).view(torch.uint8),
                                        saved_state[name].reshape(-1).view(torch.uint8)), name)
        for name, parameter in params.items():
            original, snapshot=saved_grads[name]
            self.assertIs(parameter.grad, original, name)
            if snapshot is not None:
                self.assertTrue(torch.equal(parameter.grad.reshape(-1).view(torch.uint8),
                                            snapshot.reshape(-1).view(torch.uint8)), name)
        # The graph is still usable by the caller, and this second read also avoids .grad writes.
        gradients=torch.autograd.grad(main+aux, (model.actor.weight, model.critic.weight))
        torch.testing.assert_close(gradients[0], torch.tensor([-1., 7.], dtype=torch.float64))
        torch.testing.assert_close(gradients[1], torch.tensor([12.], dtype=torch.float64))
        self.assertIsNone(model.critic.weight.grad)

    def test_nonfinite_gradients_are_rejected(self):
        for position in ('main', 'aux', 'es'):
            for invalid in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(position=position, invalid=invalid):
                    model=TinyModel()
                    model.state_estimator.weight.requires_grad_(True)
                    losses=dict(main=actor_loss(model, (3., 4.)), aux=actor_loss(model, (6., 8.)),
                                es=model.state_estimator.weight.square().sum())
                    losses[position]=losses[position]*invalid
                    with self.assertRaises(ValueError):
                        gradient_diagnostics(model, losses['main'], losses['aux'], losses['es'])

    def test_nonfinite_constant_losses_are_rejected_even_without_a_gradient(self):
        for position in ('main', 'aux', 'es'):
            for invalid in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(position=position, invalid=invalid):
                    model=TinyModel()
                    losses=dict(main=actor_loss(model, (3., 4.)), aux=actor_loss(model, (6., 8.)), es=torch.tensor(1.))
                    losses[position]=torch.tensor(invalid)
                    with self.assertRaises(ValueError):
                        gradient_diagnostics(model, losses['main'], losses['aux'], losses['es'])

    def test_no_trainable_parameters_are_rejected(self):
        model=TinyModel()
        model.requires_grad_(False)
        with self.assertRaises(ValueError):
            gradient_diagnostics(model, torch.tensor(1.), torch.tensor(0.), torch.tensor(2.))

    def test_report_requires_all_fields_and_supports_a_prefix(self):
        model=TinyModel()
        diagnostic=gradient_diagnostics(model, actor_loss(model, (3., 4.)), actor_loss(model, (6., 8.)), torch.tensor(1.))
        report=complete_report(diagnostic, 15.)
        self.assertEqual(set(report), set(GRADIENT_FIELDS))
        self.assertTrue(validate_gradient_report(report))
        self.assertTrue(validate_gradient_report({'mu_'+key: value for key, value in report.items()}, prefix='mu_'))
        for missing in GRADIENT_FIELDS:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                validate_gradient_report({key: value for key, value in report.items() if key != missing})

    def test_report_rejects_nonfinite_bool_ranges_and_nonzero_estimator_gradient(self):
        model=TinyModel()
        report=complete_report(gradient_diagnostics(model, actor_loss(model, (3., 4.)),
            actor_loss(model, (6., 8.)), torch.tensor(1.)), 15.)
        for field in GRADIENT_FIELDS:
            for invalid in (float('nan'), float('inf'), True):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    validate_gradient_report(dict(report, **{field: invalid}))
        for field, invalid in (('main_grad_norm', -.1), ('aux_grad_norm_all_batches', -.1),
                ('actor_main_grad_norm', -.1), ('actor_aux_grad_norm', -.1),
                ('combined_grad_norm_preclip', -.1), ('actor_main_aux_cosine', 1.01),
                ('actor_main_aux_cosine', -1.01), ('actor_cosine_valid_fraction', -.01),
                ('actor_cosine_valid_fraction', 1.01), ('global_clip_factor', -.01),
                ('global_clip_factor', 1.01), ('es_grad_norm', .01)):
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                validate_gradient_report(dict(report, **{field: invalid}))


if __name__ == '__main__':
    unittest.main()
