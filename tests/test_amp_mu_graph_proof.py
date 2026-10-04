"""Exact native sum graph proof; local CUDA/CPU tests are not cloud acceptance."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn

from humanoid.amp.feature_freeze import (
    COMPONENT_GRADIENT_SCHEMA, _validate_component_sum_graph, gradient_diagnostics,
)
from humanoid.amp.mu_loss import DeterministicMuLoss
from humanoid.amp.mu_temporal import ACCELERATION_SCALES
from test_amp_feature_freeze import byte_snapshot
from test_amp_feature_gradients import TinyModel, actor_loss
from test_amp_mu_loss import LinearInferenceModel


class ScalarLeafModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Module()
        self.actor.left = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
        self.actor.right = nn.Parameter(torch.tensor(.4, dtype=torch.float64))


class NotNativeSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        return left+right

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class NativeComponentGraphTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()
        self.anchor = actor_loss(self.model, (6., 8.))
        self.temporal = actor_loss(self.model, (-4., 3.))
        self.auxiliary = self.anchor+self.temporal
        self.main = actor_loss(self.model, (3., 4.))

    def measure(self, auxiliary=None, anchor=None, temporal=None):
        return gradient_diagnostics(self.model, self.main,
            self.auxiliary if auxiliary is None else auxiliary, torch.tensor(1.),
            components=dict(weighted_anchor=self.anchor if anchor is None else anchor,
                            weighted_temporal=self.temporal if temporal is None else temporal))

    def test_native_direct_sum_is_accepted_without_gradient_value_matching(self):
        with mock.patch.object(torch, 'allclose', side_effect=AssertionError('No numerical graph substitute')):
            result = self.measure()
        self.assertEqual(result['actor_anchor_grad_norm'], 10.)
        self.assertEqual(result['actor_temporal_grad_norm'], 5.)

    def test_float32_cancellation_native_graph_is_not_confused_with_separate_reduction_order(self):
        torch.manual_seed(123)
        model = nn.Module()
        model.actor = nn.Linear(16, 4, bias=False)
        x = torch.randn(192, 16)
        y = model.actor(x)
        coefficient = torch.randn_like(y)
        anchor = (y*coefficient).sum()*.01
        temporal = (y*(-coefficient+torch.randn_like(coefficient)*1e-5)).sum()*.01
        auxiliary = anchor+temporal
        actual, left, right = [torch.autograd.grad(value, model.actor.weight, retain_graph=True)[0]
                              for value in (auxiliary, anchor, temporal)]
        # Independent small reproduction needs no checkpoint or simulator.
        self.assertFalse(torch.allclose(actual, left+right, rtol=1e-5, atol=1e-7))
        result = gradient_diagnostics(model, model.actor.weight.square().mean(), auxiliary,
            auxiliary.new_zeros(()), components=dict(weighted_anchor=anchor, weighted_temporal=temporal))
        self.assertGreater(result['actor_anchor_grad_norm'], 0.)
        self.assertGreater(result['actor_temporal_grad_norm'], 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_equal_value_opposite_or_tiny_tangent_graph_is_rejected(self):
        for wrong in (-self.anchor+2*self.anchor.detach(),
                      self.anchor+1e-14*(self.model.actor.weight.sum()-
                                        self.model.actor.weight.sum().detach())):
            with self.subTest(node=type(wrong.grad_fn).__name__):
                self.assertEqual(float(wrong), float(self.anchor))
                with self.assertRaisesRegex(ValueError, 'exact auxiliary graph edges'):
                    self.measure(anchor=wrong)

    def test_rebuilt_and_cloned_components_are_not_the_original_edges(self):
        for wrong in (self.anchor.clone(), self.anchor+0,
                      actor_loss(self.model, (6., 8.))):
            with self.subTest(node=type(wrong.grad_fn).__name__):
                self.assertEqual(float(wrong), float(self.anchor))
                with self.assertRaisesRegex(ValueError, 'exact auxiliary graph edges'):
                    self.measure(anchor=wrong)

    def test_clone_wrapper_and_custom_equal_sum_root_are_not_native_direct_add(self):
        for wrong in (self.auxiliary.clone(), NotNativeSum.apply(self.anchor, self.temporal),
                      self.auxiliary+0):
            with self.subTest(node=type(wrong.grad_fn).__name__):
                self.assertEqual(float(wrong), float(self.auxiliary))
                with self.assertRaises(ValueError): self.measure(auxiliary=wrong)

    def test_custom_sum_spoofed_native_name_alpha_and_edges_still_fails_type_identity(self):
        wrong = NotNativeSum.apply(self.anchor, self.temporal)
        node = wrong.grad_fn
        native_name = type(node).__name__
        node._saved_alpha = 1
        type(node).__name__ = 'AddBackward0'
        try:
            self.assertEqual(type(node).__name__, 'AddBackward0')
            self.assertEqual(node._saved_alpha, 1)
            self.assertIs(node.next_functions[0][0], self.anchor.grad_fn)
            with self.assertRaisesRegex(ValueError, 'native component sum graph'):
                self.measure(auxiliary=wrong)
        finally:
            type(node).__name__ = native_name

    def test_nonunit_alpha_is_rejected_even_when_both_component_values_are_zero(self):
        with torch.no_grad(): self.model.actor.weight.zero_()
        anchor = actor_loss(self.model, (6., 8.))
        temporal = actor_loss(self.model, (-4., 3.))
        for alpha in (-1, 0, 2):
            with self.subTest(alpha=alpha):
                wrong = torch.add(anchor, temporal, alpha=alpha)
                self.assertEqual(float(wrong), float(anchor+temporal))
                with self.assertRaisesRegex(ValueError, 'native component sum graph'):
                    self.measure(auxiliary=wrong, anchor=anchor, temporal=temporal)

    def test_actual_leaf_accumulate_grad_variable_identity_is_supported(self):
        model = ScalarLeafModel()
        left, right = model.actor.left, model.actor.right
        auxiliary = left+right
        edge = torch.autograd.graph.get_gradient_edge(left)
        self.assertIs(edge.node.variable, left)
        result = gradient_diagnostics(model, left-right, auxiliary, left.new_zeros(()),
                                      components=dict(weighted_anchor=left, weighted_temporal=right))
        self.assertEqual(result['actor_anchor_grad_norm'], 1.)
        self.assertEqual(result['actor_temporal_grad_norm'], 1.)
        with self.assertRaises(ValueError):
            _validate_component_sum_graph(auxiliary, (left.detach().clone().requires_grad_(), right))
        self.assertTrue(_validate_component_sum_graph(left+left, (left, left)))

    def test_same_node_wrong_multioutput_edge_is_rejected(self):
        with torch.no_grad(): self.model.actor.weight.zero_()
        left, wrong = self.model.actor.weight.unbind()
        left_edge, wrong_edge = [torch.autograd.graph.get_gradient_edge(x) for x in (left, wrong)]
        self.assertIs(left_edge.node, wrong_edge.node)
        self.assertNotEqual(left_edge.output_nr, wrong_edge.output_nr)
        auxiliary = left+self.temporal
        self.measure(auxiliary=auxiliary, anchor=left)
        self.assertEqual(float(left), float(wrong))
        with self.assertRaisesRegex(ValueError, 'exact auxiliary graph edges'):
            self.measure(auxiliary=auxiliary, anchor=wrong)

    def test_invalid_graph_fails_before_any_backward_and_preserves_state(self):
        self.model.actor.weight.grad = torch.tensor([9., -0.], dtype=torch.float64)
        saved = byte_snapshot(self.model.state_dict())
        pointer = self.model.actor.weight.grad
        grad = byte_snapshot(pointer)
        rng = torch.get_rng_state().clone()
        with mock.patch.object(torch.autograd, 'grad', side_effect=AssertionError('Invalid graph traversed')):
            with self.assertRaises(ValueError): self.measure(anchor=self.anchor.clone())
        self.assertEqual(byte_snapshot(self.model.state_dict()), saved)
        self.assertIs(self.model.actor.weight.grad, pointer)
        self.assertEqual(byte_snapshot(pointer), grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_missing_edge_api_fails_closed_instead_of_approximate_fallback(self):
        with mock.patch.object(torch.autograd.graph, 'get_gradient_edge', None):
            with self.assertRaisesRegex(ValueError, 'native component sum graph'): self.measure()

    def test_finite_and_actor_only_gradient_checks_remain_active(self):
        anchor = self.anchor+self.model.critic.weight.sum()
        with self.assertRaisesRegex(ValueError, 'non-actor'):
            self.measure(auxiliary=anchor+self.temporal, anchor=anchor)
        model = ScalarLeafModel()
        with torch.no_grad(): model.actor.left.zero_()
        anchor = model.actor.left.sqrt()
        temporal = model.actor.right*0
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            gradient_diagnostics(model, model.actor.right, anchor+temporal, torch.tensor(0.),
                                 components=dict(weighted_anchor=anchor, weighted_temporal=temporal))

    def test_empty_optin_loss_has_native_sum_and_legacy_zero_is_unchanged(self):
        model = LinearInferenceModel()
        regularizer = DeterministicMuLoss(model, [1000., 1000.], 1., .05,
                                         component_gradient_schema=COMPONENT_GRADIENT_SCHEMA)
        empty = torch.empty(0, 2)
        mask = torch.empty(0, dtype=torch.bool)
        legacy, legacy_stats = regularizer.loss(empty, empty, empty, mask)
        auxiliary, stats, pieces = regularizer.loss(empty, empty, empty, mask, return_components=True)
        self.assertEqual(stats, legacy_stats)
        self.assertEqual(float(auxiliary), float(legacy))
        self.assertEqual(stats['valid_triplets'], 0)
        self.assertEqual(stats['batch_size'], 0)
        self.assertTrue(_validate_component_sum_graph(auxiliary, tuple(pieces.values())))
        diagnostic = gradient_diagnostics(model, legacy, auxiliary, auxiliary.new_zeros(()), components=pieces)
        self.assertEqual(diagnostic['actor_temporal_grad_norm'], 0.)
        self.assertEqual(diagnostic['actor_anchor_grad_norm'], 0.)
        gradients = [torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True,
                                         allow_unused=True) for loss in (legacy, auxiliary)]
        self.assertEqual(byte_snapshot(gradients[0]), byte_snapshot(gradients[1]))


SOURCE = Path(__file__).resolve().parents[1].parent/'f1-amp-sustain'/'outputs'/'amp-sustain'/\
         'TASK_20260926_110'/'model_8802500.pt'


@unittest.skipUnless(SOURCE.is_file(), 'Original110 local artifact required; no synthetic source proof')
class ActualSourceGraphTests(unittest.TestCase):
    """Actual architecture/source, synthetic inputs; not the failed cloud tensors."""
    @classmethod
    def setUpClass(cls):
        expected_sha = '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31'
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != expected_sha:
            raise AssertionError('Local original110 checkpoint bytes do not match the bound source SHA')
        cls.source = torch.load(str(SOURCE), weights_only=True, map_location='cpu')
        path = Path(__file__).resolve().parents[1]/'humanoid'/'algo'/'ppo'/'actor_critic_dh.py'
        spec = importlib.util.spec_from_file_location('graph_proof_original_actor', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.model_class = module.ActorCriticDH

    def source_case(self, device):
        torch.manual_seed(17)
        with contextlib.redirect_stdout(io.StringIO()):
            model = self.model_class(235, 47, 219, 12, actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[768, 256, 128], state_estimator_hidden_dims=[256, 128, 64],
                in_channels=66).to(device)
        model.load_state_dict(self.source['model_state_dict'])
        regularizer = DeterministicMuLoss(model, list(ACCELERATION_SCALES.values()), 1., .05,
                                         component_gradient_schema=COMPONENT_GRADIENT_SCHEMA)
        for module in (model.long_history, model.state_estimator):
            for parameter in module.parameters(): parameter.requires_grad_(False)
        with torch.no_grad():
            for parameter in model.actor.parameters(): parameter.add_(1e-4*torch.randn_like(parameter))
        torch.manual_seed(8)
        now = torch.randn(192, 3102, device=device)
        previous = now+torch.randn_like(now)
        older = previous+torch.randn_like(now)
        auxiliary, _, pieces = regularizer.loss(now, previous, older,
            torch.ones(192, dtype=torch.bool, device=device), return_components=True)
        return model, regularizer, auxiliary, pieces

    def check_source_graph(self, device):
        original_threads = torch.get_num_threads()
        settings = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.set_num_threads(2)
            model, regularizer, auxiliary, pieces = self.source_case(device)
            parameters = list(model.actor.parameters())
            actual, left, right = [torch.autograd.grad(value, parameters, retain_graph=True)
                for value in (auxiliary, pieces['weighted_anchor'], pieces['weighted_temporal'])]
            differences = [(a-(l+r)).double() for a, l, r in zip(actual, left, right)]
            # Match the old check's actual expected summation order exactly.
            legacy_pass = all(torch.allclose(a, l+r, rtol=1e-5, atol=1e-7)
                              for a, l, r in zip(actual, left, right))
            residual = math.sqrt(sum(float(d.square().sum()) for d in differences)/
                                 sum(float(a.double().square().sum()) for a in actual))
            if device == 'cpu' and sys.platform == 'win32' and torch.__version__.startswith('2.4.1'):
                self.assertFalse(legacy_pass, 'Known original-architecture float32 false-rejection regression')
            self.assertTrue(math.isfinite(residual))
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
            optimizer.load_state_dict(copy.deepcopy(self.source['optimizer_state_dict']))
            es_optimizer = torch.optim.Adam(model.state_estimator.parameters(), lr=5e-5)
            es_optimizer.load_state_dict(copy.deepcopy(self.source['es_optimizer_state_dict']))
            for parameter in model.parameters():
                if parameter.requires_grad: parameter.grad = torch.full_like(parameter, .25)
            state = byte_snapshot(model.state_dict())
            adam = byte_snapshot(optimizer.state_dict())
            es_adam = byte_snapshot(es_optimizer.state_dict())
            gradients = [(parameter.grad, byte_snapshot(parameter.grad)) for parameter in model.parameters()]
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
            main = model.actor[0].weight.square().mean()+model.critic[0].weight.square().mean()+model.std.square().mean()
            diagnostic = gradient_diagnostics(model, main, auxiliary, auxiliary.new_zeros(()), components=pieces)
            self.assertGreater(diagnostic['actor_temporal_grad_norm'], 0.)
            self.assertEqual(byte_snapshot(model.state_dict()), state)
            self.assertEqual(byte_snapshot(optimizer.state_dict()), adam)
            self.assertEqual(byte_snapshot(es_optimizer.state_dict()), es_adam)
            for parameter, (pointer, value) in zip(model.parameters(), gradients):
                self.assertIs(parameter.grad, pointer)
                self.assertEqual(byte_snapshot(parameter.grad), value)
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
            self.assertEqual(byte_snapshot(torch.cuda.get_rng_state_all() if cuda_rng else []), byte_snapshot(cuda_rng))
            self.assertTrue(regularizer.teacher_unchanged())
            (main+auxiliary).backward()  # Original real gradient path remains usable.
        finally:
            torch.set_num_threads(original_threads)
        self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), settings)

    def test_original_architecture_cpu_false_rejection_now_passes_exact_graph(self):
        self.check_source_graph('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'Local CUDA unavailable; not cloud proof')
    def test_original_architecture_cuda_proof_keeps_all_training_state_unchanged(self):
        self.check_source_graph('cuda')


if __name__ == '__main__':
    unittest.main()
