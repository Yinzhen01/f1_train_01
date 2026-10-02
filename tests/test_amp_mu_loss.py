"""CPU loss/gradient invariants; these are not native motion acceptance."""
import contextlib
import copy
import io
import json
import unittest

import torch
from torch import nn
from torch.distributions import Normal

from humanoid.amp.dry_run import cpu_ppo_classes
from humanoid.amp.mu_loss import DeterministicMuLoss
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class LinearInferenceModel(nn.Module):
    """Small complete independent-feature path for exact arithmetic tests."""
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(4, 2, bias=False)
        self.long_history = nn.Linear(2, 1, bias=False)
        self.state_estimator = nn.Linear(2, 1, bias=False)
        self.critic = nn.Linear(2, 1)
        self.std = nn.Parameter(torch.ones(2))
        self.distribution = None
        with torch.no_grad():
            self.long_history.weight.copy_(torch.tensor([[1., 0.]]))
            self.state_estimator.weight.copy_(torch.tensor([[0., 1.]]))
            self.actor.weight.copy_(torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]]))

    def act_inference(self, observations):
        return self.actor(torch.cat((observations, self.state_estimator(observations),
                                     self.long_history(observations)), dim=-1))


class DeterministicMuLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.student = LinearInferenceModel()
        self.observations = torch.tensor([[3., 1.], [20., 20.]])
        self.previous = torch.tensor([[1., 0.], [0., 0.]])
        self.older = torch.tensor([[0., 0.], [99., 99.]])
        self.mask = torch.tensor([True, False])

    def make_loss(self, **kwargs):
        values = dict(acceleration_scale=[5000., 10000.], anchor_coef=1., temporal_coef=1.)
        values.update(kwargs)
        return DeterministicMuLoss(self.student, **values)

    def test_exact_masked_formula_and_fixed_json_stats(self):
        loss = self.make_loss()
        auxiliary, stats = loss.loss(self.observations, self.previous, self.older, self.mask)
        # mu=(x0+x1,x0+x1), selected delta2=(2,2); scaled acceleration=10000.
        self.assertEqual(stats['anchor_loss'], 0.)
        self.assertAlmostEqual(stats['temporal_loss'], 2.5)
        self.assertAlmostEqual(float(auxiliary), 2.5)
        self.assertEqual(stats['valid_triplets'], 1)
        self.assertEqual(stats['batch_size'], 2)
        self.assertEqual(set(stats), set(loss.STAT_FIELDS))
        json.dumps(stats, allow_nan=False)
        stats['anchor_loss'] = 99.
        self.assertEqual(loss.latest_stats['anchor_loss'], 0.)
        report = loss.report(); report['batch_size'] = -1
        self.assertEqual(loss.report()['batch_size'], 2)

    def test_all_three_student_branches_have_exact_input_gradients(self):
        loss = self.make_loss(anchor_coef=0.)
        now, previous, older = (value.clone().requires_grad_() for value in
                                 (self.observations, self.previous, self.older))
        auxiliary, _ = loss.loss(now, previous, older, self.mask)
        gradients = torch.autograd.grad(auxiliary, (now, previous, older))
        # d mean([(.5*d2/1e-4/5000)^2,(.5*d2/1e-4/10000)^2])/dx = 2.5.
        for gradient, factor in zip(gradients, (1., -2., 1.)):
            torch.testing.assert_close(gradient[0], torch.tensor([2.5, 2.5])*factor)
            torch.testing.assert_close(gradient[1], torch.zeros(2), atol=0, rtol=0)

    def test_actual_dh_cnn_es_actor_gradients_and_no_teacher_gradients(self):
        actor_type, _ = cpu_ppo_classes(ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            student = actor_type(8, 8, 6, 3, actor_hidden_dims=[16], critic_hidden_dims=[8],
                state_estimator_hidden_dims=[12], in_channels=4, kernel_size=[3],
                filter_size=[4], stride_size=[1], lh_output_dim=4)
        with torch.no_grad():
            for name in ('actor', 'long_history', 'state_estimator'):
                for parameter in getattr(student, name).parameters():
                    parameter.fill_(.02)
        student.act(torch.ones(3, 32))
        distribution = student.distribution
        distribution_mean = distribution.mean.detach().clone()
        rng_before = torch.get_rng_state().clone()
        loss = DeterministicMuLoss(student, [1000., 1200., 1400.], 0., 1.)
        inputs = [torch.full((3, 32), value, requires_grad=True) for value in (3., 1., 0.)]
        auxiliary, stats = loss.loss(*inputs, torch.tensor([True, True, False]))
        input_gradients = torch.autograd.grad(auxiliary, inputs, retain_graph=True)
        for gradient in input_gradients:
            self.assertGreater(float(gradient[:2].abs().sum()), 0.)
            self.assertEqual(float(gradient[2].abs().sum()), 0.)
        auxiliary.backward()
        for name in ('actor', 'long_history', 'state_estimator'):
            parameters = list(getattr(student, name).parameters())
            self.assertTrue(all(parameter.grad is not None and
                bool(torch.isfinite(parameter.grad).all()) for parameter in parameters))
            # The second difference cancels constant output biases exactly.
            self.assertGreater(sum(float(parameter.grad.abs().sum())
                                   for parameter in parameters), 0.)
        self.assertIsNone(student.std.grad)
        self.assertTrue(all(parameter.grad is None for parameter in student.critic.parameters()))
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in loss.teacher.parameters()))
        self.assertFalse(loss.teacher.training)
        self.assertTrue(stats['teacher_unchanged'])
        self.assertIs(student.distribution, distribution)
        torch.testing.assert_close(distribution.mean, distribution_mean, atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_teacher_is_full_independent_function_and_cannot_enter_optimizer(self):
        loss = self.make_loss()
        expected = copy.deepcopy(self.student).act_inference(self.observations).detach()
        torch.testing.assert_close(loss.teacher.act_inference(self.observations), expected, atol=0, rtol=0)
        for name in ('actor', 'long_history', 'state_estimator'):
            self.assertIsNot(getattr(loss.teacher, name), getattr(self.student, name))
            for left, right in zip(getattr(loss.teacher, name).parameters(),
                                   getattr(self.student, name).parameters()):
                self.assertNotEqual(left.data_ptr(), right.data_ptr())
        optimizer = torch.optim.SGD(self.student.parameters(), lr=.001)
        with torch.no_grad():
            self.student.long_history.weight.add_(.2)
            self.student.state_estimator.weight.add_(.3)
            self.student.actor.weight.add_(.1)
        auxiliary, stats = loss.loss(self.observations, self.previous, self.older, self.mask)
        self.assertGreater(stats['anchor_loss'], 0.)
        auxiliary.backward(); optimizer.step()
        torch.testing.assert_close(loss.teacher.act_inference(self.observations), expected, atol=0, rtol=0)
        self.assertTrue(loss.teacher_unchanged())
        teacher_ids = {id(parameter) for parameter in loss.teacher.parameters()}
        self.assertTrue(teacher_ids.isdisjoint(id(parameter) for group in optimizer.param_groups
                                              for parameter in group['params']))
        with self.assertRaises(AttributeError):
            loss.teacher = self.student

    def test_nonleaf_distribution_and_rng_are_not_changed_even_zero_temporal(self):
        mean = self.student.act_inference(self.observations)
        self.student.distribution = Normal(mean, mean*0+self.student.std)
        distribution = self.student.distribution
        original_mean = distribution.mean.detach().clone()
        rng_before = torch.get_rng_state().clone()
        loss = self.make_loss(temporal_coef=0.)
        auxiliary, stats = loss.loss(self.observations, self.previous, self.older, self.mask)
        self.assertIs(self.student.distribution, distribution)
        torch.testing.assert_close(distribution.mean, original_mean, atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertGreater(stats['temporal_loss'], 0.)
        self.assertEqual(stats['weighted_temporal_loss'], 0.)
        self.assertTrue(auxiliary.requires_grad)

    def test_empty_mask_and_empty_batch_are_differentiable_and_finite(self):
        loss = self.make_loss(anchor_coef=0.)
        auxiliary, stats = loss.loss(self.observations, self.previous, self.older,
                                     torch.zeros(2, dtype=torch.bool))
        self.assertEqual(float(auxiliary), 0.)
        self.assertEqual(stats['temporal_loss'], 0.)
        self.assertEqual(stats['valid_triplets'], 0)
        self.assertTrue(auxiliary.requires_grad); auxiliary.backward()
        self.student.zero_grad(set_to_none=True)
        empty = torch.empty((0, 2))
        auxiliary, stats = loss.loss(empty, empty, empty, torch.empty(0, dtype=torch.bool))
        self.assertEqual(float(auxiliary), 0.)
        self.assertTrue(auxiliary.requires_grad); auxiliary.backward()
        self.assertEqual(stats['batch_size'], 0)
        json.dumps(stats, allow_nan=False)

    def test_teacher_bitwise_detects_signed_zero_buffer_mutation_and_eval_change(self):
        self.student.register_buffer('zero_probe', torch.zeros(1))
        loss = self.make_loss()
        with torch.no_grad():
            loss.teacher.zero_probe.fill_(-0.)
        self.assertFalse(loss.teacher_unchanged())
        self.assertFalse(loss.report()['teacher_unchanged'])
        with self.assertRaises(ValueError):
            loss.loss(self.observations, self.previous, self.older, self.mask)
        other = self.make_loss(); other.teacher.train()
        self.assertFalse(other.teacher_unchanged())

    def test_constructor_rejects_bad_scale_coefficients_dt_action_scale_and_state(self):
        bad_values = [dict(acceleration_scale=[]), dict(acceleration_scale=[1.]),
            dict(acceleration_scale=[[1., 2.]]), dict(acceleration_scale=[0., 1.]),
            dict(acceleration_scale=[-1., 1.]), dict(acceleration_scale=[float('nan'), 1.]),
            dict(acceleration_scale=[float('inf'), 1.]), dict(acceleration_scale=[True, False]),
            dict(acceleration_scale=[True, 1.]), dict(acceleration_scale=[1.+0j, 2.]),
            dict(acceleration_scale=torch.ones(2, requires_grad=True)),
            dict(anchor_coef=-1.), dict(anchor_coef=float('nan')), dict(anchor_coef=True),
            dict(temporal_coef=-1.), dict(temporal_coef=float('inf')), dict(dt=0.),
            dict(dt=-.01), dict(dt=float('inf')), dict(dt=True), dict(dt=1e-300),
            dict(dt=1e300), dict(action_scale=0.), dict(action_scale=-.5),
            dict(action_scale=float('nan')), dict(action_scale=True)]
        for values in bad_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.make_loss(**values)
        with torch.no_grad(): self.student.actor.weight[0, 0] = float('inf')
        with self.assertRaises(ValueError): self.make_loss()

    def test_exact_float64_scale_snapshot_and_readonly_loss_constants(self):
        baseline = [2111.67649924755, 1670.7300096750228]
        loss = self.make_loss(acceleration_scale=baseline)
        self.assertEqual(loss.acceleration_scale.dtype, torch.float64)
        self.assertEqual(loss.acceleration_scale.tolist(), baseline)
        baseline[0] = 1.
        self.assertEqual(loss.acceleration_scale[0].item(), 2111.67649924755)
        tensor_scale = torch.tensor([5000., 10000.], dtype=torch.float64)
        tensor_loss = self.make_loss(acceleration_scale=tensor_scale)
        before, _ = tensor_loss.loss(self.observations, self.previous, self.older, self.mask)
        tensor_scale.zero_()
        exposed_copy = tensor_loss.acceleration_scale
        exposed_copy.zero_()
        after, _ = tensor_loss.loss(self.observations, self.previous, self.older, self.mask)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        self.assertEqual(tensor_loss.acceleration_scale.tolist(), [5000., 10000.])
        for name, value in (('anchor_coef', 2.), ('temporal_coef', .01), ('dt', .02),
                            ('action_scale', 1.), ('acceleration_scale', torch.ones(2))):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(tensor_loss, name, value)
        self.assertIsInstance(tensor_loss.anchor_coef, float)
        self.assertIsInstance(tensor_loss.temporal_coef, float)

    def test_rejects_observation_shape_dtype_nonfinite_mask_and_loss_overflow(self):
        loss = self.make_loss()
        bad_inputs = [([1., 2.], self.previous, self.older, self.mask),
            (self.observations[0], self.previous, self.older, self.mask),
            (self.observations, self.previous[:1], self.older, self.mask),
            (self.observations, self.previous.double(), self.older, self.mask),
            (self.observations.long(), self.previous.long(), self.older.long(), self.mask),
            (self.observations, self.previous, self.older, self.mask[:, None]),
            (self.observations, self.previous, self.older, self.mask.float()),
            (self.observations, self.previous, self.older, torch.tensor([True]))]
        for values in bad_inputs:
            with self.subTest(shapes=[getattr(v, 'shape', None) for v in values]), self.assertRaises(ValueError):
                loss.loss(*values)
        for index in range(3):
            inputs = [self.observations.clone(), self.previous.clone(), self.older.clone()]
            inputs[index][1, 0] = float('nan')  # Invalid-mask rows must still be finite.
            with self.assertRaises(ValueError): loss.loss(*inputs, self.mask)
        with self.assertRaises(ValueError):
            self.make_loss(acceleration_scale=[1e-300, 1.]).loss(
                self.observations, self.previous, self.older, self.mask)
        with self.assertRaises(ValueError):
            self.make_loss(temporal_coef=1e300).loss(
                self.observations, self.previous, self.older, self.mask)
        for values in (dict(anchor_coef=1e-300), dict(action_scale=1e-300)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.make_loss(**values).loss(self.observations, self.previous,
                                               self.older, self.mask)
        huge = torch.full_like(self.observations, 1e30)
        with self.assertRaises(ValueError): loss.loss(huge, self.previous, self.older, self.mask)

    def test_rejects_invalid_or_detached_mean_and_teacher_inference_mutation(self):
        class WrongWidth(LinearInferenceModel):
            def act_inference(self, observations):
                return super().act_inference(observations)[:, :1]
        class Detached(LinearInferenceModel):
            def act_inference(self, observations):
                return super().act_inference(observations).detach()
        class MutatingBuffer(LinearInferenceModel):
            def __init__(self):
                super().__init__()
                self.register_buffer('calls', torch.zeros(1))
            def act_inference(self, observations):
                self.calls.add_(1.)
                return super().act_inference(observations)
        for model_type in (WrongWidth, Detached, MutatingBuffer):
            with self.subTest(model=model_type.__name__), self.assertRaises(ValueError):
                DeterministicMuLoss(model_type(), [5000., 10000.], 1., 1.).loss(
                    self.observations, self.previous, self.older, self.mask)


if __name__ == '__main__':
    unittest.main()
