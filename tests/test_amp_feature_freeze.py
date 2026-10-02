"""Real Adam/CPU feature-freeze checks; no simulator or physical improvement claim."""
import json
import unittest
from unittest import mock

import torch
from torch import nn

from humanoid.amp.feature_freeze import feature_fingerprint, validate_gradient_report
from humanoid.amp.mu_loss import DeterministicMuLoss
from test_amp_mu_storage import TinyActor, make_ppo, ppo_rollout


class BufferedTinyActor(TinyActor):
    """Inference feature buffers must be protected along with their parameters."""
    def __init__(self):
        super().__init__()
        self.state_estimator.register_buffer('calibration', torch.tensor([0., 1., -2.]))
        self.long_history.register_buffer('calibration', torch.tensor([0., .25]))


def byte_snapshot(value):
    """Compare dtype, shape and raw bytes, including signed zero and Adam step."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return (str(tensor.dtype), tuple(tensor.shape),
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    if isinstance(value, dict):
        return tuple((key, byte_snapshot(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(byte_snapshot(item) for item in value))
    return (type(value), value)


def feature_bytes(model):
    return byte_snapshot({name: value for name, value in model.state_dict().items()
                          if name.startswith(('long_history.', 'state_estimator.'))})


def optimizer_slots(optimizer):
    return tuple(tuple(id(parameter) for parameter in group['params'])
                 for group in optimizer.param_groups)


def optimizer_bytes(optimizer):
    # Inspect raw state: deliberately corrupting group membership makes Adam's
    # state_dict() raise before the feature guard itself can be exercised.
    states = tuple((id(parameter), byte_snapshot(state))
                   for parameter, state in optimizer.state.items())
    groups = tuple(tuple((key, tuple(id(parameter) for parameter in value)
                          if key == 'params' else byte_snapshot(value))
                         for key, value in group.items()) for group in optimizer.param_groups)
    return states, groups


def frozen_adam_bytes(ppo):
    frozen = {name: parameter for name, parameter in ppo.actor_critic.named_parameters()
              if name.startswith(('long_history.', 'state_estimator.'))}
    return {label: {name: (parameter in optimizer.state,
                          byte_snapshot(optimizer.state.get(parameter)))
                    for name, parameter in frozen.items()}
            for label, optimizer in (('ppo', ppo.optimizer),
                                     ('es', ppo.state_estimator_optimizer))}


def optimizer_for(ppo, label):
    return ppo.optimizer if label == 'ppo' else ppo.state_estimator_optimizer


def populated_ppo():
    """Populate both Adams by real differentiation/steps before configuring freeze."""
    ppo = make_ppo(BufferedTinyActor())
    ppo_rollout(ppo)
    ppo.update()
    # The regular PPO update leaves gradients on every parameter. Clear those
    # before the separate ES Adam step so its state comes only from this loss.
    ppo.optimizer.zero_grad(set_to_none=True)
    ppo.state_estimator_optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([[.2, -.3], [-.1, .4], [.6, .1]])
    targets = torch.tensor([[.1, -.2, .3], [.2, .1, -.1], [-.2, .3, .1]])
    nn.functional.mse_loss(ppo.actor_critic.state_estimator(inputs), targets).backward()
    ppo.state_estimator_optimizer.step()
    ppo.optimizer.zero_grad(set_to_none=True)
    ppo.state_estimator_optimizer.zero_grad(set_to_none=True)
    return ppo


def configure_mu(ppo):
    regularizer = DeterministicMuLoss(ppo.actor_critic, [1000., 1000.], 1., .01)
    ppo.configure_mu_temporal(regularizer)


def frozen_ppo():
    ppo = populated_ppo()
    configure_mu(ppo)
    ppo.configure_feature_freeze()
    return ppo


class FeatureFreezeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22)

    def assert_refused_before_step(self, mutate):
        ppo = frozen_ppo()
        ppo_rollout(ppo)
        mutate(ppo)
        before_model = byte_snapshot(ppo.actor_critic.state_dict())
        before_optimizers = [optimizer_bytes(optimizer) for optimizer in
                             (ppo.optimizer, ppo.state_estimator_optimizer)]
        before_rng = torch.get_rng_state().clone()
        with mock.patch.object(ppo.optimizer, 'step', wraps=ppo.optimizer.step) as main_step, \
                mock.patch.object(ppo.state_estimator_optimizer, 'step',
                                  wraps=ppo.state_estimator_optimizer.step) as es_step:
            with self.assertRaises(RuntimeError):
                ppo.update()
        main_step.assert_not_called()
        es_step.assert_not_called()
        self.assertEqual(before_model, byte_snapshot(ppo.actor_critic.state_dict()))
        self.assertEqual(before_optimizers,
                         [optimizer_bytes(optimizer) for optimizer in
                          (ppo.optimizer, ppo.state_estimator_optimizer)])
        torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(ppo.storage.step, 6)

    def test_mu_must_be_configured_before_one_time_freeze(self):
        ppo = populated_ppo()
        with self.assertRaises(RuntimeError):
            ppo.configure_feature_freeze()
        self.assertIsNone(ppo.feature_freeze_guard)
        self.assertTrue(all(parameter.requires_grad for parameter in ppo.actor_critic.parameters()))
        configure_mu(ppo)
        report = ppo.configure_feature_freeze()
        self.assertEqual(report['frozen_parameter_count'], 4)
        self.assertEqual(report['trainable_parameter_count'], 5)
        self.assertTrue(ppo.feature_freeze_guard.validate())
        with self.assertRaises(RuntimeError):
            ppo.configure_feature_freeze()

    def test_two_actual_updates_preserve_features_buffers_and_both_adams_bytewise(self):
        ppo = frozen_ppo()
        model = ppo.actor_critic
        initial_features, initial_adams = feature_bytes(model), frozen_adam_bytes(ppo)
        initial_sha = feature_fingerprint(model)
        initial_slots = [optimizer_slots(optimizer) for optimizer in
                         (ppo.optimizer, ppo.state_estimator_optimizer)]
        frozen = ppo.feature_freeze_guard.frozen
        for parameter in frozen.values():
            state = ppo.optimizer.state[parameter]
            self.assertEqual(set(state), {'step', 'exp_avg', 'exp_avg_sq'})
            self.assertGreater(float(state['step']), 0.)
            self.assertGreater(float(state['exp_avg'].abs().sum()), 0.)
            self.assertGreater(float(state['exp_avg_sq'].sum()), 0.)
        for parameter in model.state_estimator.parameters():
            state = ppo.state_estimator_optimizer.state[parameter]
            self.assertGreater(float(state['step']), 0.)
            self.assertGreater(float(state['exp_avg'].abs().sum()), 0.)
            self.assertGreater(float(state['exp_avg_sq'].sum()), 0.)
        self.assertTrue(all(parameter not in ppo.state_estimator_optimizer.state
                            for parameter in model.long_history.parameters()))
        for iteration in range(2):
            with self.subTest(update=iteration + 1):
                before = {name: parameter.detach().clone() for name, parameter in
                          ppo.feature_freeze_guard.trainable.items()}
                trainable_steps = {name: float(ppo.optimizer.state[parameter]['step'])
                                   for name, parameter in ppo.feature_freeze_guard.trainable.items()}
                with torch.inference_mode():
                    ppo_rollout(ppo)
                with mock.patch.object(ppo.state_estimator_optimizer, 'step',
                                       wraps=ppo.state_estimator_optimizer.step) as es_step:
                    losses = ppo.update()
                es_step.assert_not_called()
                self.assertEqual(len(losses), 3)
                self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses))
                self.assertEqual(initial_features, feature_bytes(model))
                self.assertEqual(initial_adams, frozen_adam_bytes(ppo))
                self.assertEqual(initial_sha, feature_fingerprint(model))
                self.assertEqual(initial_slots, [optimizer_slots(optimizer) for optimizer in
                                                (ppo.optimizer, ppo.state_estimator_optimizer)])
                self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                    for parameter in frozen.values()))
                # Actor, critic and std each continue learning through the real PPO path.
                for group in ('actor.', 'critic.', 'std'):
                    names = [name for name in before if name.startswith(group)]
                    self.assertTrue(any(not torch.equal(before[name],
                                       ppo.feature_freeze_guard.trainable[name]) for name in names), group)
                for name, parameter in ppo.feature_freeze_guard.trainable.items():
                    self.assertEqual(float(ppo.optimizer.state[parameter]['step']),
                                     trainable_steps[name] + 6)
                report = ppo.mu_latest
                self.assertTrue(report['frozen_features_unchanged'])
                self.assertTrue(report['frozen_optimizer_unchanged'])
                self.assertEqual(report['frozen_features_initial_sha256'], initial_sha)
                self.assertEqual(report['frozen_features_final_sha256'], initial_sha)
                self.assertEqual(report['gradient_minibatches'], 6)
                self.assertEqual(report['es_grad_norm'], 0.)
                self.assertGreater(report['main_grad_norm'], 0.)
                self.assertGreater(report['actor_aux_grad_norm'], 0.)
                self.assertTrue(validate_gradient_report(report))
                self.assertTrue(ppo.mu_regularizer.teacher_unchanged())
                json.dumps(report, allow_nan=False)
                self.assertEqual(ppo.storage.step, 0)

    def test_freeze_clears_stale_frozen_grads_without_changing_adam_state(self):
        ppo = populated_ppo()
        configure_mu(ppo)
        initial_adams = frozen_adam_bytes(ppo)
        for name, parameter in ppo.actor_critic.named_parameters():
            if name.startswith(('long_history.', 'state_estimator.')):
                parameter.grad = torch.ones_like(parameter)
        ppo.configure_feature_freeze()
        self.assertEqual(initial_adams, frozen_adam_bytes(ppo))
        self.assertTrue(all(parameter.grad is None for parameter in ppo.feature_freeze_guard.frozen.values()))
        # Adam also skips these existing ES slots if a caller invokes its step.
        ppo.state_estimator_optimizer.step()
        self.assertEqual(initial_adams, frozen_adam_bytes(ppo))
        self.assertTrue(ppo.feature_freeze_guard.validate())

    def test_changed_requires_grad_or_frozen_grad_is_refused_before_step(self):
        mutations = {
            'frozen_requires_grad': lambda ppo: ppo.actor_critic.state_estimator.weight.requires_grad_(True),
            'trainable_requires_grad': lambda ppo: ppo.actor_critic.actor.weight.requires_grad_(False),
            'frozen_grad': lambda ppo: setattr(ppo.actor_critic.long_history.weight, 'grad',
                                             torch.zeros_like(ppo.actor_critic.long_history.weight)),
        }
        for label, mutate in mutations.items():
            with self.subTest(mutation=label):
                self.assert_refused_before_step(mutate)

    def test_changed_parameter_buffer_or_parameter_identity_is_refused_before_step(self):
        def change_parameter(ppo):
            with torch.no_grad():
                ppo.actor_critic.long_history.weight[0, 0] += .001
        def change_buffer(ppo):
            ppo.actor_critic.state_estimator.calibration[1] += .001
        def change_buffer_signed_zero(ppo):
            buffer = ppo.actor_critic.long_history.calibration
            original = buffer.clone()
            buffer[0] = -0.
            self.assertTrue(torch.equal(original, buffer))
            self.assertNotEqual(byte_snapshot(original), byte_snapshot(buffer))
        def replace_parameter(ppo):
            old = ppo.actor_critic.state_estimator.weight
            ppo.actor_critic.state_estimator.weight = nn.Parameter(old.detach().clone(), requires_grad=False)
        for label, mutate in (('parameter', change_parameter), ('buffer', change_buffer),
                              ('buffer_signed_zero', change_buffer_signed_zero),
                              ('parameter_identity', replace_parameter)):
            with self.subTest(mutation=label):
                self.assert_refused_before_step(mutate)

    def test_changed_main_or_es_adam_moment_step_or_presence_is_refused_before_step(self):
        for label in ('ppo', 'es'):
            for key in ('exp_avg', 'exp_avg_sq', 'step', 'state_presence'):
                def mutate(ppo, label=label, key=key):
                    optimizer = optimizer_for(ppo, label)
                    parameter = ppo.actor_critic.state_estimator.weight
                    if key == 'state_presence':
                        del optimizer.state[parameter]
                    else:
                        optimizer.state[parameter][key].add_(1.)
                with self.subTest(optimizer=label, mutation=key):
                    self.assert_refused_before_step(mutate)
        def create_previously_absent_state(ppo):
            parameter = ppo.actor_critic.long_history.weight
            ppo.state_estimator_optimizer.state[parameter] = {
                'step': torch.tensor(1.), 'exp_avg': torch.zeros_like(parameter),
                'exp_avg_sq': torch.zeros_like(parameter)}
        self.assert_refused_before_step(create_previously_absent_state)

    def test_changed_main_or_es_optimizer_group_slots_are_refused_before_step(self):
        for label in ('ppo', 'es'):
            for mutation in ('reorder', 'replace', 'split_group'):
                def mutate(ppo, label=label, mutation=mutation):
                    optimizer = optimizer_for(ppo, label)
                    parameters = optimizer.param_groups[0]['params']
                    if mutation == 'reorder':
                        parameters[0], parameters[1] = parameters[1], parameters[0]
                    elif mutation == 'replace':
                        parameters[0] = nn.Parameter(parameters[0].detach().clone())
                    else:
                        second = dict(optimizer.param_groups[0])
                        second['params'] = parameters[1:]
                        optimizer.param_groups[0]['params'] = parameters[:1]
                        optimizer.param_groups.append(second)
                with self.subTest(optimizer=label, mutation=mutation):
                    self.assert_refused_before_step(mutate)

    def test_constructor_rejects_misordered_main_and_es_adam_slots_before_freezing(self):
        for label in ('ppo', 'es'):
            with self.subTest(optimizer=label):
                ppo = populated_ppo()
                configure_mu(ppo)
                parameters = optimizer_for(ppo, label).param_groups[0]['params']
                parameters[0], parameters[1] = parameters[1], parameters[0]
                before = byte_snapshot(ppo.actor_critic.state_dict())
                with self.assertRaises(ValueError):
                    ppo.configure_feature_freeze()
                self.assertIsNone(ppo.feature_freeze_guard)
                self.assertTrue(all(parameter.requires_grad for parameter in ppo.actor_critic.parameters()))
                self.assertEqual(before, byte_snapshot(ppo.actor_critic.state_dict()))


if __name__ == '__main__':
    unittest.main()
