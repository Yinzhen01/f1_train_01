"""CPU checks of actual storage/PPO temporal indexing, not physics or training QA."""
import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn
from torch.distributions import Normal

from humanoid.amp.dry_run import cpu_ppo_classes
from humanoid.amp.mu_loss import DeterministicMuLoss


REPO = Path(__file__).resolve().parents[1]
_, DHPPO = cpu_ppo_classes(REPO)
RolloutStorage = importlib.import_module(DHPPO.__module__.rsplit('.', 1)[0]+'.rollout_storage').RolloutStorage
torch.set_num_threads(1)


def metadata(n, tick, command_width=3):
    return dict(episode_ids=torch.arange(n), control_ticks=torch.full((n,), tick, dtype=torch.long),
                episode_lengths=torch.full((n,), 70+tick, dtype=torch.long), commands=torch.zeros(n, command_width))


def transition(storage, tick, meta=None, dones=None):
    n = storage.num_envs
    item = storage.Transition()
    item.observations = torch.stack((torch.arange(n)+100*tick, torch.arange(n)), -1).float()
    item.critic_observations = item.observations.clone()
    item.actions = torch.zeros(n, 1)
    item.rewards = torch.zeros(n)
    item.dones = torch.zeros(n, dtype=torch.bool) if dones is None else dones
    item.values = torch.zeros(n, 1)
    item.actions_log_prob = torch.zeros(n)
    item.action_mean = torch.zeros(n, 1)
    item.action_sigma = torch.ones(n, 1)
    item.mu_metadata = meta
    if storage.num_single_obs is not None:
        item.next_proprio_obs = item.observations.clone()
    return item


def filled_storage(t=7, n=3, enabled=True, mutate=None, startup=66, command_width=3, proprio=None):
    storage = RolloutStorage(n, t, [2], [2], [1], num_single_obs=proprio)
    if enabled:
        storage.configure_mu_temporal(startup)
    for step in range(t):
        meta, done = metadata(n, step, command_width), torch.zeros(n, dtype=torch.bool)
        if mutate is not None:
            mutate(step, meta, done)
        storage.add_transitions(transition(storage, step, meta, done))
    return storage


class TinyActor(nn.Module):
    """Normal sampling PPO model with differentiable actor/CNN/ES inference paths."""
    def __init__(self):
        super().__init__()
        self.num_short_obs = 2
        self.state_estimator = nn.Linear(2, 3)
        self.long_history = nn.Linear(4, 2)
        self.actor = nn.Linear(9, 2)
        self.critic = nn.Linear(5, 1)
        self.std = nn.Parameter(torch.ones(2)*.7)
        self.distribution = None

    def act_inference(self, obs):
        return self.actor(torch.cat((obs, self.state_estimator(obs[:, -2:]), self.long_history(obs)), -1))

    def act(self, obs, **kwargs):
        mean = self.act_inference(obs)
        self.distribution = Normal(mean, self.std.expand_as(mean))
        return self.distribution.sample()

    def evaluate(self, obs, **kwargs):
        return self.critic(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def reset(self, dones):
        pass


def ppo_rollout(ppo, steps=6, n=2, timeout=False):
    for tick in range(steps):
        if ppo.mu_regularizer is not None:
            ppo.set_mu_metadata(**metadata(n, tick))
        # Nonlinear-in-time observations give nonzero deterministic second differences.
        obs = torch.stack([torch.tensor([tick*tick*.03, e*.1, tick*.02, .1]) for e in range(n)])
        critic = torch.stack([torch.tensor([.2, .1, -.1, tick*.01, e*.2]) for e in range(n)])
        with torch.no_grad():
            ppo.act(obs, critic)
        dones = torch.zeros(n, dtype=torch.bool)
        infos = {}
        if timeout and tick == 2:
            dones[0] = True
            infos['time_outs'] = dones.clone()
        ppo.process_env_step(torch.ones(n)*.05, dones, infos)
    with torch.no_grad():
        ppo.compute_returns(torch.zeros(n, 5))


def make_ppo(actor=None, epochs=2, mini=3):
    ppo = DHPPO(TinyActor() if actor is None else actor, num_learning_epochs=epochs,
                num_mini_batches=mini, learning_rate=1e-4, lin_vel_idx=0, max_grad_norm=1.)
    ppo.init_storage(2, 6, [4], [5], [2])
    return ppo


class MuStorageTests(unittest.TestCase):
    def test_shuffled_batches_reference_same_env_real_time_neighbours(self):
        storage = filled_storage()
        torch.manual_seed(71)
        expected = torch.randperm(21)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(71)
        batches = list(storage.mini_batch_generator(3, 2, with_temporal=True))
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        self.assertEqual(len(batches), 6)
        torch.testing.assert_close(torch.cat([b[-1]['batch_indices'] for b in batches[:3]]), expected)
        self.assertFalse(torch.equal(expected, torch.arange(21)))
        for batch in batches:
            self.assertEqual(len(batch), 12)
            data = batch[-1]
            idx, valid = data['batch_indices'], data['valid_mask']
            torch.testing.assert_close(data['prev_indices'][valid], idx[valid]-3)
            torch.testing.assert_close(data['older_indices'][valid], idx[valid]-6)
            self.assertTrue((data['prev_indices'][valid] % 3 == idx[valid] % 3).all())
            torch.testing.assert_close(data['prev_obs'][valid, 0], batch[0][valid, 0]-100)
            torch.testing.assert_close(data['older_obs'][valid, 0], batch[0][valid, 0]-200)
            self.assertEqual(data['rollout_diagnostics']['valid_triplets'], 15)

    def test_default_11_tuple_rng_and_proprio_13_tuple_unchanged(self):
        storage = filled_storage(enabled=False)
        torch.manual_seed(9)
        expected = torch.randperm(20)  # Original drops the non-divisible last sample.
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(9)
        batches = list(storage.mini_batch_generator(4, 3))
        self.assertTrue(all(len(b) == 11 for b in batches))
        observations = storage.observations.flatten(0, 1)
        torch.testing.assert_close(torch.cat([b[0] for b in batches[:4]]), observations[expected])
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        proprio = filled_storage(enabled=False, proprio=2)
        self.assertEqual(len(next(proprio.mini_batch_generator(1))), 13)
        with self.assertRaises(RuntimeError):
            next(storage.mini_batch_generator(1, with_temporal=True))

    def test_current_done_is_valid_prior_failure_timeout_done_cut_two_triplets(self):
        def change(step, meta, done):
            if step == 2:
                done[0] = True
            if step >= 3:
                meta['episode_ids'][0] += 10
                meta['control_ticks'][0] = step-3
        storage = filled_storage(t=6, n=2, mutate=change)
        valid = storage.mu_temporal_data()['valid_mask'].reshape(6, 2)
        self.assertEqual(valid[:, 0].tolist(), [False, False, True, False, False, True])
        self.assertEqual(valid[:, 1].tolist(), [False, False, True, True, True, True])
        diagnostics = storage.mu_temporal_data()['rollout_diagnostics']
        self.assertEqual(diagnostics['invalid_reasons']['previous_done'], 2)
        self.assertEqual(diagnostics['invalid_reasons']['episode_changed'], 2)
        self.assertEqual(diagnostics['valid_triplets'], 6)

    def test_exact_commands_all_three_frames_and_fourth_heading_command(self):
        def change(step, meta, done):
            if step >= 3:
                meta['commands'][0, 3] = 1e-9  # No tolerance: even tiny real changes cut.
        storage = filled_storage(t=6, n=2, mutate=change, command_width=4)
        valid = storage.mu_temporal_data()['valid_mask'].reshape(6, 2)
        self.assertEqual(valid[:, 0].tolist(), [False, False, True, False, False, True])
        self.assertEqual(storage.mu_temporal_data()['rollout_diagnostics']['invalid_reasons']['command_changed'], 2)

    def test_tick_gap_and_episode_change_without_done_are_rejected(self):
        def change(step, meta, done):
            if step >= 3:
                meta['control_ticks'][0] += 1
                meta['episode_ids'][1] += 20
        storage = filled_storage(t=6, n=2, mutate=change)
        valid = storage.mu_temporal_data()['valid_mask'].reshape(6, 2)
        self.assertEqual(valid[:, 0].tolist(), [False, False, True, False, False, True])
        self.assertEqual(valid[:, 1].tolist(), [False, False, True, False, False, True])
        reasons = storage.mu_temporal_data()['rollout_diagnostics']['invalid_reasons']
        self.assertEqual(reasons['tick_discontinuity'], 2)
        self.assertEqual(reasons['episode_changed'], 2)

    def test_startup_requires_all_three_episode_lengths_at_least_66(self):
        def change(step, meta, done):
            meta['episode_lengths'].fill_(64+step)
        storage = filled_storage(t=6, n=2, mutate=change)
        self.assertEqual(storage.mu_temporal_data()['valid_mask'].reshape(6, 2)[:, 0].tolist(),
                         [False, False, False, False, True, True])
        self.assertEqual(storage.mu_temporal_data()['rollout_diagnostics']['invalid_reasons']['startup'], 4)

    def test_short_rollout_empty_mask_and_complete_read_only_diagnostics(self):
        for steps in (1, 2):
            storage = filled_storage(t=steps, n=1)
            before = {key: value.clone() for key, value in storage.mu_metadata.items()}
            dones_before = storage.dones.clone()
            result = storage.mu_temporal_data()
            self.assertFalse(result['valid_mask'].any())
            self.assertEqual(result['rollout_diagnostics']['candidate_triplets'], 0)
            self.assertEqual(result['rollout_diagnostics']['invalid_triplets'], steps)
            self.assertEqual(result['rollout_diagnostics']['invalid_reasons']['rollout_boundary'], steps)
            for key, value in before.items():
                torch.testing.assert_close(value, storage.mu_metadata[key], rtol=0, atol=0)
            torch.testing.assert_close(dones_before, storage.dones, rtol=0, atol=0)
        storage = filled_storage(t=3, n=1)
        storage.mu_metadata_recorded[1] = False
        with self.assertRaises(RuntimeError):
            storage.mu_temporal_data()

    def test_clear_never_borrows_last_rollout_even_contiguous_same_episode(self):
        storage = filled_storage(t=5, n=2)
        storage.clear()
        with self.assertRaises(RuntimeError):
            storage.mu_temporal_data()
        self.assertFalse(storage.mu_metadata_recorded.any())
        for tick in range(5, 10):
            storage.add_transitions(transition(storage, tick, metadata(2, tick)))
        self.assertEqual(storage.mu_temporal_data()['valid_mask'].reshape(5, 2)[:, 0].tolist(),
                         [False, False, True, True, True])
        self.assertEqual(storage.mu_temporal_data()['rollout_diagnostics']['valid_triplets'], 6)

    def test_snapshot_clones_and_storage_requires_all_metadata_before_recording(self):
        storage = RolloutStorage(2, 3, [2], [2], [1])
        storage.configure_mu_temporal()
        values = metadata(2, 0)
        frozen = storage.clone_mu_metadata(**values)
        for key in values:
            self.assertNotEqual(values[key].data_ptr(), frozen[key].data_ptr())
            values[key].fill_(9)
        self.assertEqual(frozen['control_ticks'].tolist(), [0, 0])
        storage.add_transitions(transition(storage, 0, frozen))
        frozen['commands'].fill_(3)
        self.assertFalse(storage.mu_metadata['commands'][0].any())
        for broken in (None, {'episode_ids': torch.zeros(2, dtype=torch.long)}):
            with self.assertRaises(ValueError):
                storage.add_transitions(transition(storage, 1, broken))
            self.assertEqual(storage.step, 1)
            self.assertFalse(storage.mu_metadata_recorded[1])
        with self.assertRaises(RuntimeError):
            storage.configure_mu_temporal()

    def test_metadata_shape_dtype_finiteness_and_device_guards(self):
        storage = RolloutStorage(2, 3, [2], [2], [1])
        storage.configure_mu_temporal()
        invalid = [('episode_ids', torch.zeros(2).float()), ('control_ticks', torch.tensor([-1, 1])),
                   ('episode_lengths', torch.zeros(2, 1).long()), ('commands', torch.zeros(2, 2)),
                   ('commands', torch.zeros(3, 3)), ('commands', torch.zeros(2, 3).long()),
                   ('commands', torch.full((2, 3), float('nan'))),
                   ('commands', torch.full((2, 3), float('inf'))),
                   ('commands', torch.empty(2, 3, device='meta'))]
        for key, value in invalid:
            with self.subTest(key=key, shape=value.shape, dtype=value.dtype):
                values = metadata(2, 0)
                values[key] = value
                with self.assertRaises(ValueError):
                    storage.clone_mu_metadata(**values)
        for startup in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                storage.configure_mu_temporal(startup)


class MuPPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22)

    def test_preaction_metadata_is_required_fresh_cloned_and_before_sampling(self):
        ppo = make_ppo()
        regularizer = DeterministicMuLoss(ppo.actor_critic, [1000., 1000.], 1., .01)
        ppo.configure_mu_temporal(regularizer)
        before_rng = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            ppo.act(torch.zeros(2, 4), torch.zeros(2, 5))
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        values = metadata(2, 0)
        ppo.set_mu_metadata(**values)
        for value in values.values():
            value.fill_(999)
        with self.assertRaises(RuntimeError):
            ppo.set_mu_metadata(**metadata(2, 0))
        with torch.no_grad():
            ppo.act(torch.zeros(2, 4), torch.zeros(2, 5))
        ppo.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {})
        self.assertEqual(ppo.storage.mu_metadata['control_ticks'][0].tolist(), [0, 0])
        self.assertFalse(ppo.storage.mu_metadata['commands'][0].any())
        with self.assertRaises(RuntimeError):
            ppo.act(torch.zeros(2, 4), torch.zeros(2, 5))

    def test_real_regularizer_update_preserves_return_es_optimizer_teacher_and_counts(self):
        ppo = make_ppo()
        regularizer = DeterministicMuLoss(ppo.actor_critic, [1000., 1000.], 1., .01)
        ppo.configure_mu_temporal(regularizer)
        optimizer_ids = {id(p) for group in ppo.optimizer.param_groups for p in group['params']}
        self.assertFalse(optimizer_ids & {id(p) for p in regularizer.teacher.parameters()})
        es_optimizer = copy.deepcopy(ppo.state_estimator_optimizer.state_dict())
        teacher = {k: v.clone() for k, v in regularizer.teacher.state_dict().items()}
        student = {k: v.clone() for k, v in ppo.actor_critic.state_dict().items()}
        ppo_rollout(ppo, timeout=True)
        # Timeout bootstrapping affects rewards, not historical done exclusion.
        diagnostics = ppo.storage.mu_temporal_data()['rollout_diagnostics']
        self.assertEqual(diagnostics['valid_triplets'], 6)
        with mock.patch('torch.autograd.grad', wraps=torch.autograd.grad) as grad:
            result = ppo.update()
        self.assertEqual(grad.call_count, 1)
        self.assertTrue(grad.call_args.kwargs['retain_graph'])
        self.assertEqual(len(result), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result))
        self.assertEqual(ppo.state_estimator_optimizer.state_dict(), es_optimizer)
        self.assertTrue(any(not torch.equal(student[k], v) for k, v in ppo.actor_critic.state_dict().items()))
        self.assertTrue(any(not torch.equal(student[k], v) for k, v in ppo.actor_critic.state_dict().items() if k.startswith('state_estimator.')))
        self.assertTrue(all(torch.equal(teacher[k], v) for k, v in regularizer.teacher.state_dict().items()))
        self.assertTrue(regularizer.teacher_unchanged())
        self.assertEqual(ppo.mu_latest['valid_triplets'], 6)
        self.assertEqual(ppo.mu_latest['triplet_evaluations'], 12)
        self.assertEqual(ppo.mu_latest['evaluated_samples'], 24)
        self.assertGreater(ppo.mu_latest['aux_grad_norm'], 0.)
        self.assertEqual(ppo.mu_latest['anchor_coef'], 1.)
        self.assertEqual(ppo.mu_latest['temporal_coef'], .01)
        self.assertEqual(ppo.mu_latest['updates'], 1)
        self.assertEqual(ppo.mu_latest['rollout_diagnostics'], diagnostics)
        self.assertAlmostEqual(ppo.mu_latest['aux_loss'],
                               ppo.mu_latest['weighted_anchor_loss']+ppo.mu_latest['weighted_temporal_loss'], places=7)
        json.dumps(ppo.mu_latest, allow_nan=False)
        self.assertEqual(ppo.storage.step, 0)
        self.assertFalse(ppo.storage.mu_metadata_recorded.any())

    def test_zero_aux_keeps_disabled_update_rng_parameters_and_original_11_interface(self):
        first, second = TinyActor(), TinyActor()
        second.load_state_dict(first.state_dict())
        baseline, enabled = make_ppo(first), make_ppo(second)
        enabled.configure_mu_temporal(DeterministicMuLoss(second, [1000., 1000.], 0., 0.))
        self.assertIsNone(baseline.mu_regularizer)
        for ppo in (baseline, enabled):
            torch.manual_seed(12)
            ppo_rollout(ppo)
        baseline_generator = baseline.storage.mini_batch_generator
        with mock.patch.object(baseline.storage, 'mini_batch_generator', wraps=baseline_generator) as generator:
            torch.manual_seed(89)
            baseline_losses = baseline.update()
            baseline_rng = torch.get_rng_state().clone()
        self.assertEqual(generator.call_args.args, (3, 2))
        self.assertEqual(generator.call_args.kwargs, {})
        torch.manual_seed(89)
        enabled_losses = enabled.update()
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        self.assertEqual(baseline_losses, enabled_losses)
        self.assertEqual(baseline.mu_latest, {})
        self.assertEqual(enabled.mu_latest['aux_grad_norm'], 0.)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    def test_configure_before_init_and_reject_unfrozen_or_mutated_teacher(self):
        actor = TinyActor()
        ppo = DHPPO(actor, lin_vel_idx=0)
        regularizer = DeterministicMuLoss(actor, [1000., 1000.], 1., 0.)
        ppo.configure_mu_temporal(regularizer)
        ppo.init_storage(2, 6, [4], [5], [2])
        self.assertEqual(ppo.storage.mu_startup_steps, 66)
        regularizer.teacher.actor.weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            ppo.configure_mu_temporal(regularizer)
        regularizer.teacher.actor.weight.requires_grad_(False)
        with torch.no_grad():
            regularizer.teacher.actor.weight[0, 0] += 1
        with self.assertRaises(RuntimeError):
            ppo.update()

    def test_empty_valid_mask_still_runs_anchor_and_retains_unique_zero_counts(self):
        ppo = make_ppo()
        regularizer = DeterministicMuLoss(ppo.actor_critic, [1000., 1000.], 1., .01)
        ppo.configure_mu_temporal(regularizer, startup_steps=1000)
        ppo_rollout(ppo)
        result = ppo.update()
        self.assertEqual(len(result), 3)
        self.assertEqual(ppo.mu_latest['valid_triplets'], 0)
        self.assertEqual(ppo.mu_latest['triplet_evaluations'], 0)
        self.assertEqual(ppo.mu_latest['weighted_temporal_loss'], 0.)
        self.assertGreaterEqual(ppo.mu_latest['anchor_loss'], 0.)
        self.assertTrue(ppo.mu_latest['all_finite'])
        self.assertTrue(regularizer.teacher_unchanged())

    def test_native_inference_mode_rollout_then_normal_gradient_update(self):
        ppo = make_ppo()
        regularizer = DeterministicMuLoss(ppo.actor_critic, [1000., 1000.], 1., .01)
        ppo.configure_mu_temporal(regularizer)
        teacher_before = {key: value.clone() for key, value in regularizer.teacher.state_dict().items()}
        student_before = ppo.actor_critic.actor.weight.detach().clone()
        # Native runner wraps metadata + act + env-step recording + returns in
        # inference_mode, not merely no_grad. Mutable metadata is cloned there.
        with torch.inference_mode():
            ppo_rollout(ppo)
            self.assertTrue(torch.is_inference(ppo.storage.advantages))
            self.assertFalse(torch.is_inference(ppo.storage.observations))
            # Commands are lazily allocated on the first recorded transition,
            # so this read-only continuity buffer really is an inference tensor.
            self.assertTrue(torch.is_inference(ppo.storage.mu_metadata['commands']))
        self.assertFalse(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_inference(ppo.storage.mu_temporal_data()['valid_mask']))
        result = ppo.update()
        self.assertEqual(len(result), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result))
        self.assertGreater(ppo.mu_latest['aux_grad_norm'], 0.)
        self.assertGreater(ppo.mu_latest['weighted_temporal_loss'], 0.)
        self.assertGreater(ppo.mu_latest['valid_triplets'], 0)
        self.assertFalse(torch.equal(student_before, ppo.actor_critic.actor.weight))
        self.assertIsNotNone(ppo.actor_critic.actor.weight.grad)
        self.assertTrue(torch.isfinite(ppo.actor_critic.actor.weight.grad).all())
        self.assertTrue(ppo.mu_latest['all_finite'])
        self.assertTrue(regularizer.teacher_unchanged())
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in regularizer.teacher.parameters()))
        self.assertTrue(all(torch.equal(teacher_before[key], value)
                            for key, value in regularizer.teacher.state_dict().items()))

    def test_nonfinite_aux_or_non_json_stats_refused_before_optimizer_step(self):
        for bad_loss, bad_stats in ((float('nan'), {}), (0., {'bad': torch.ones(1)})):
            ppo = make_ppo()
            regularizer = DeterministicMuLoss(ppo.actor_critic, [1000., 1000.], 1., .01)
            ppo.configure_mu_temporal(regularizer)
            ppo_rollout(ppo)
            before = {key: value.clone() for key, value in ppo.actor_critic.state_dict().items()}
            with mock.patch.object(regularizer, 'loss', return_value=(torch.tensor(bad_loss), bad_stats)):
                with self.assertRaises((ValueError, TypeError)):
                    ppo.update()
            for key, value in before.items():
                torch.testing.assert_close(value, ppo.actor_critic.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
