"""Synthetic workflow admission/actual-architecture forwards, never cloud proof.

No original checkpoint/archive is opened, no policy is fitted, and no simulator
or Gradmotion request is made. Source hashes in protocol fixtures are public
contract constants, not evidence that these synthetic arrays came from source.
"""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
import torch

from humanoid.amp.head_workflow import (audit_cohort, exclusion_add_cohort,
    fit_episodes, forward_parity, identity_digest)
from humanoid.amp.learnability import NO_DR_SWITCHES
from humanoid.amp.mu_temporal import state_fingerprint
from humanoid.scripts.collect_amp_head_cohort import (BUDGETS, SEEDS, SCHEMA,
    SOURCE_SHA, SOURCE_MODEL_SHA, SOURCE_ENDPOINT_SHA, FrozenActorCapture,
    audit_arrays, cohort_contract, empty_exclusion_index, fingerprint_cohort,
    validate_source_runtime)
from test_head_cohort import fixture as array_fixture


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {'fixture': 'synthetic_head_workflow', 'source': 'not_a_real_checkpoint'}
COMMIT, IMPLEMENTATION = '1'*40, '2'*64


def environment():
    dr = {key: False for key in NO_DR_SWITCHES}
    dr['use_nominal_joint_armature'] = True
    return dict(domain_rand=dr, noise=dict(add_noise=False),
        terrain=dict(mesh_type='plane', curriculum=False))


def source_fixture():
    """Independent synthetic source readback, never loaded from the cohort."""
    cfg = environment()
    physx = dict(solver_type=1, num_position_iterations=4, num_velocity_iterations=0,
        contact_offset=.01, rest_offset=0., max_depenetration_velocity=1., contact_collection=2)
    cfg.update(seed=5, env=dict(num_envs=16, episode_length_s=60.1, frame_stack=66),
        control=dict(action_scale=.5, stiffness={'ankle_pitch': 35.}),
        sim=dict(dt=.001, physx=physx))
    runtime = dict(dof_properties={'armature': [.1]}, p_gains=[35.], d_gains=[1.5],
        control_dt=.01, physics_dt=.001)
    return dict(environment=cfg, runtime=runtime)


def cohort_fixture(split='train', *, failed_env=None, exclusions=None):
    """Real fixed smoke dimensions, but only synthetic complete native-like rows."""
    _, arrays = array_fixture()
    n, seconds = BUDGETS['smoke']
    contract = cohort_contract(split, 'smoke', SEEDS[split], n, seconds)
    # Distinct feature episodes keep the downstream Episode designs meaningful.
    arrays['reference_actor_hidden'][:, :, 0] = arrays['reference_single_obs'][:, :, 0]
    if failed_env is not None:
        arrays['reference_valid'][90:, failed_env] = False
        arrays['reference_failure'][89, failed_env] = True
        arrays['reference_done'][89, failed_env] = True
    ready, _, history, episodes = audit_arrays(arrays, contract)
    exclusions = empty_exclusion_index() if exclusions is None else exclusions
    fingerprints, dedupe = fingerprint_cohort(arrays, contract, ready, exclusions)
    source = source_fixture()
    cfg, runtime = copy.deepcopy(source['environment']), copy.deepcopy(source['runtime'])
    cfg['seed'] = SEEDS[split]
    cfg['env'].update(num_envs=n, episode_length_s=seconds+.1)
    runtime['physics_sim_parameters'] = copy.deepcopy(source['environment']['sim'])
    runtime_proof = validate_source_runtime(cfg, runtime, source, contract)
    manifest = dict(contract)
    manifest.update(type=SCHEMA, source_checkpoint_sha256=SOURCE_SHA,
        source_model_state_sha256=SOURCE_MODEL_SHA, identity=copy.deepcopy(IDENTITY),
        source_completed_updates=2500, code_commit=COMMIT,
        implementation_fingerprint=IMPLEMENTATION, policy_deterministic=True,
        parent_frozen=True, no_training=True, effectiveness_verified=False,
        dr_unlocked=False, source_regression=dict(sha256=SOURCE_ENDPOINT_SHA),
        environment=cfg, runtime=runtime, source_runtime_proof=runtime_proof,
        fingerprints=fingerprints, history_proof=history, episodes=episodes,
        deduplication=dedupe, fit_eligible=False)
    return manifest, arrays, exclusions


def audit(manifest, arrays, exclusions, *, split='train', **overrides):
    options = dict(split=split, mode='smoke', identity=IDENTITY, code_commit=COMMIT,
        implementation_fingerprint=IMPLEMENTATION, exclusions=exclusions,
        source_manifest=source_fixture())
    options.update(overrides)
    return audit_cohort(manifest, arrays, **options)


class CohortWorkflowAdmissionTests(unittest.TestCase):
    def test_complete_canonical_cohort_preserves_all_ready_ticks_hidden_seed_and_env(self):
        manifest, arrays, index = cohort_fixture()
        before = arrays['reference_actor_hidden'].copy()
        result = audit(manifest, arrays, index)
        self.assertTrue(result['smoke_eligible'])
        self.assertFalse(result['fit_eligible'])
        self.assertEqual(len(result['episodes']), 4)
        self.assertTrue(all(row['complete'] for row in result['episodes']))
        episodes = fit_episodes(manifest, arrays, result, identity=IDENTITY)
        self.assertEqual(len(episodes), 4)
        for i, episode in enumerate(episodes):
            self.assertEqual(episode.episode_id, manifest['episode_ids'][i])
            self.assertEqual(episode.seed, 305)
            self.assertEqual(episode.env_id, i)
            self.assertEqual(episode.source_identity, identity_digest(IDENTITY))
            self.assertEqual(episode.history_frames, 66)
            self.assertEqual(episode.dt, .01)
            self.assertTrue(episode.complete)
            self.assertEqual(episode.design.shape, (134, 129))
            np.testing.assert_array_equal(episode.ticks, np.arange(66, 200))
            np.testing.assert_array_equal(episode.design[:, :128], before[66:, i])
            np.testing.assert_array_equal(episode.design[:, -1], np.ones(134))
        np.testing.assert_array_equal(arrays['reference_actor_hidden'], before)

    def test_canonical_train_validation_swap_and_wrong_mode_seed_budget_ids_rejected(self):
        for split, other in (('train', 'validation'), ('validation', 'train')):
            manifest, arrays, index = cohort_fixture(split)
            with self.subTest(split=split), self.assertRaises(ValueError):
                audit(manifest, arrays, index, split=other)
        manifest, arrays, index = cohort_fixture()
        changes = [dict(seed=505), dict(mode='formal'), dict(num_envs=8), dict(duration_s=20.),
            dict(fps=50), dict(expected_ticks=199), dict(initialization='standing'),
            dict(episode_ids=list(reversed(manifest['episode_ids']))), dict(split='sealed')]
        for changed in changes:
            current = copy.deepcopy(manifest)
            current.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit(current, arrays, index)
        with self.assertRaises(ValueError):
            audit(manifest, arrays, index, mode='formal')

    def test_source_code_hash_and_dr_timing_mismatch_fail_closed(self):
        manifest, arrays, index = cohort_fixture()
        changes = [dict(source_checkpoint_sha256='3'*64), dict(source_model_state_sha256='4'*64),
            dict(source_completed_updates=2501), dict(identity={'unrelated': True}),
            dict(code_commit='5'*40), dict(implementation_fingerprint='6'*64),
            dict(source_regression={}), dict(source_regression=dict(sha256='7'*64)),
            dict(parent_frozen=False), dict(no_training=False), dict(policy_deterministic=False),
            dict(effectiveness_verified=True), dict(dr_unlocked=True)]
        for changed in changes:
            current = copy.deepcopy(manifest)
            current.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit(current, arrays, index)
        for kind in ('dr', 'noise', 'plane', 'control_dt', 'physics_dt'):
            current = copy.deepcopy(manifest)
            if kind == 'dr': current['environment']['domain_rand']['randomize_friction'] = True
            elif kind == 'noise': current['environment']['noise']['add_noise'] = True
            elif kind == 'plane': current['environment']['terrain']['mesh_type'] = 'heightfield'
            elif kind == 'control_dt': current['runtime']['control_dt'] = .02
            else: current['runtime']['physics_dt'] = .002
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit(current, arrays, index)

    def test_evidence_booleans_cannot_be_replaced_with_equal_integers(self):
        manifest, arrays, index = cohort_fixture()
        for key in ('policy_deterministic', 'parent_frozen', 'no_training',
                'effectiveness_verified', 'dr_unlocked', 'fit_eligible'):
            current = copy.deepcopy(manifest)
            current[key] = int(current[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit(current, arrays, index)

    def test_cohort_runtime_cannot_self_certify_changed_source_physics_or_controller(self):
        manifest, arrays, index = cohort_fixture()
        self.assertTrue(audit(manifest, arrays, index)['smoke_eligible'])
        for kind in ('scale', 'stiffness', 'history', 'armature', 'p_gain', 'd_gain',
                'physx', 'missing_physx', 'missing_proof', 'tampered_proof'):
            current = copy.deepcopy(manifest)
            if kind == 'scale': current['environment']['control']['action_scale'] = .8
            elif kind == 'stiffness': current['environment']['control']['stiffness']['ankle_pitch'] = 36.
            elif kind == 'history': current['environment']['env']['frame_stack'] = 65
            elif kind == 'armature': current['runtime']['dof_properties']['armature'] = [.2]
            elif kind == 'p_gain': current['runtime']['p_gains'] = [36.]
            elif kind == 'd_gain': current['runtime']['d_gains'] = [1.6]
            elif kind == 'physx': current['runtime']['physics_sim_parameters']['physx']['num_velocity_iterations'] = 1
            elif kind == 'missing_physx': current['runtime']['physics_sim_parameters']['physx'].pop('contact_collection')
            elif kind == 'missing_proof': current.pop('source_runtime_proof')
            else: current['source_runtime_proof']['native_physx_parameters_exact'] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit(current, arrays, index)

    def test_manifest_claims_are_recomputed_not_trusted(self):
        manifest, arrays, index = cohort_fixture()
        for kind in ('history', 'episode', 'fingerprint', 'dedupe', 'admission'):
            current = copy.deepcopy(manifest)
            if kind == 'history': current['history_proof']['checked_ready_inputs'] -= 1
            elif kind == 'episode': current['episodes'][0]['observed_ticks'] -= 1
            elif kind == 'fingerprint': current['fingerprints'][0]['initial_state'] = 'f'*64
            elif kind == 'dedupe': current['deduplication']['passed'] = False
            else: current['fit_eligible'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit(current, arrays, index)
        actual = {key: value.copy() for key, value in arrays.items()}
        actual['reference_initial_dof_pos'][0, 0] += .5
        with self.assertRaisesRegex(ValueError, 'actual arrays'):
            audit(manifest, actual, index)
        actual = {key: value.copy() for key, value in arrays.items()}
        actual['reference_full_obs'][80, 0, 100] += .5
        with self.assertRaisesRegex(ValueError, 'history'):
            audit(manifest, actual, index)

    def test_one_failed_episode_rejects_whole_queue_without_selecting_survivors(self):
        manifest, arrays, index = cohort_fixture(failed_env=2)
        result = audit(manifest, arrays, index)
        self.assertEqual(len(result['episodes']), 4)
        self.assertEqual(sum(row['complete'] for row in result['episodes']), 3)
        self.assertTrue(result['episodes'][2]['failed'])
        self.assertEqual(result['episodes'][2]['terminal_tick'], 89)
        self.assertEqual(result['episodes'][2]['observed_ticks'], 90)
        self.assertFalse(result['fit_eligible'])
        self.assertFalse(result['smoke_eligible'])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            fit_episodes(manifest, arrays, result, identity=IDENTITY)
        fake = copy.deepcopy(manifest)
        fake['episodes'][2]['complete'] = True
        with self.assertRaises(ValueError):
            audit(fake, arrays, index)

    def test_sealed_role_never_fits_even_after_manifest_or_admission_substitution(self):
        manifest, arrays, index = cohort_fixture('sealed')
        result = audit(manifest, arrays, index, split='sealed')
        self.assertFalse(result['fit_eligible'])
        self.assertFalse(result['smoke_eligible'])
        with self.assertRaises(ValueError):
            fit_episodes(manifest, arrays, result, identity=IDENTITY)
        fake = dict(result, smoke_eligible=True)
        with self.assertRaisesRegex(ValueError, 'sealed'):
            fit_episodes(manifest, arrays, fake, identity=IDENTITY)
        manifest, arrays, index = cohort_fixture()
        result = audit(manifest, arrays, index)
        with self.assertRaises(ValueError):
            fit_episodes(manifest, arrays, result, identity={'wrong': True})

    def test_exclusion_index_contains_actual_states_and_every_ready_history_and_rejects_reuse(self):
        manifest, arrays, index = cohort_fixture()
        result = audit(manifest, arrays, index)
        added = exclusion_add_cohort(index, result)
        self.assertIs(added, index)
        self.assertEqual(len(index['initial_state']), 4)
        self.assertEqual(len(index['initial_full_observation']), 4)
        self.assertEqual(len(index['ready_full_history']), 4*134)
        for row in result['fingerprints']:
            self.assertIn(row['episode_id'], index['initial_state'][row['initial_state']])
            for ready_row in row['ready_full_history']:
                self.assertGreaterEqual(ready_row['tick'], 66)
                self.assertIn(row['episode_id'], index['ready_full_history'][ready_row['sha256']])
        reused_manifest, reused_arrays, _ = cohort_fixture('validation')
        with self.assertRaisesRegex(ValueError, 'actual arrays'):
            audit(reused_manifest, reused_arrays, index, split='validation')
        excluded_manifest, excluded_arrays, _ = cohort_fixture('validation', exclusions=index)
        excluded = audit(excluded_manifest, excluded_arrays, index, split='validation')
        self.assertFalse(excluded['deduplication']['passed'])
        self.assertFalse(excluded['smoke_eligible'])
        with self.assertRaises(ValueError):
            fit_episodes(excluded_manifest, excluded_arrays, excluded, identity=IDENTITY)

    def test_identity_hash_is_canonical_but_empty_nonjson_nonfinite_identity_is_refused(self):
        self.assertEqual(identity_digest({'a': 1, 'b': 2}), identity_digest({'b': 2, 'a': 1}))
        for value in ({}, None, [], {'nan': float('nan')}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                identity_digest(value)


class ActualArchitectureForwardParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('synthetic_workflow_actor',
            ROOT/'humanoid/algo/ppo/actor_critic_dh.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.actor_class = module.ActorCriticDH

    def fixture(self, *, identical_inputs=False):
        torch.manual_seed(571)
        with contextlib.redirect_stdout(io.StringIO()):
            policy = self.actor_class(235, 47, 219, 12, actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[768, 256, 128], state_estimator_hidden_dims=[256, 128, 64],
                in_channels=66)
        policy.eval()
        values = np.zeros((4, 2, 3102), dtype=np.float32) if identical_inputs else \
            np.random.RandomState(719).normal(size=(4, 2, 3102)).astype(np.float32)
        capture = FrozenActorCapture(policy)
        try:
            with torch.no_grad():
                mu, hidden = capture.inference(torch.from_numpy(values.reshape(-1, 3102).copy()))
        finally:
            capture.close()
        raw = mu.numpy().reshape(4, 2, 12).copy()
        clip = .01
        arrays = dict(reference_full_obs=values, reference_actor_hidden=hidden.numpy().reshape(4, 2, 128).copy(),
            reference_raw_mu=raw, reference_action=np.clip(raw, -clip, clip))
        return policy, arrays, clip, state_fingerprint(policy.state_dict())

    def parity(self, policy, arrays, clip, fingerprint, **overrides):
        options = dict(action_clip=clip, batch_size=3, expected_model_fingerprint=fingerprint)
        options.update(overrides)
        return forward_parity(policy, arrays, **options)

    def test_real_full_history_cnn_es_actor_hidden_and_mean_parity_is_read_only(self):
        policy, arrays, clip, fingerprint = self.fixture()
        for parameter in policy.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        gradients = [(p.grad, p.grad.clone()) for p in policy.parameters()]
        rng = torch.get_rng_state().clone()
        calls, handles = dict(actor=0, es=0, cnn=0), []
        for name, module in (('actor', policy.actor), ('es', policy.state_estimator), ('cnn', policy.long_history)):
            def count(module, arguments, result, name=name):
                calls[name] += 1
            handles.append(module.register_forward_hook(count))
        try:
            result = self.parity(policy, arrays, clip, fingerprint)
        finally:
            for handle in handles: handle.remove()
        self.assertTrue(result['full_obs_verified'] and result['hidden_verified'] and result['action_verified'])
        self.assertEqual(result['actual_rows'], 8)
        self.assertEqual(result['forward_batches'], 3)
        self.assertEqual(calls, dict(actor=3, es=3, cnn=3))
        self.assertLess(result['max_abs_error'], .001)
        self.assertEqual(state_fingerprint(policy.state_dict()), fingerprint)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for parameter, (pointer, expected) in zip(policy.parameters(), gradients):
            self.assertIs(parameter.grad, pointer)
            self.assertTrue(torch.equal(parameter.grad, expected))
        self.assertFalse(policy.actor[6]._forward_pre_hooks)
        self.assertFalse(policy.actor[6]._forward_hooks)

    def test_hidden_mu_full_input_and_applied_action_tampering_is_rejected(self):
        policy, arrays, clip, fingerprint = self.fixture()
        for kind in ('hidden', 'mu', 'full', 'applied', 'nonfinite'):
            current = {key: value.copy() for key, value in arrays.items()}
            if kind == 'hidden': current['reference_actor_hidden'][0, 0, 0] += .1
            elif kind == 'mu':
                current['reference_raw_mu'][0, 0, 0] += .1
                current['reference_action'] = np.clip(current['reference_raw_mu'], -clip, clip)
            elif kind == 'full': current['reference_full_obs'][:] = 10.
            elif kind == 'applied': current['reference_action'][0, 0, 0] += .1
            else: current['reference_full_obs'][0, 0, 0] = np.nan
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.parity(policy, current, clip, fingerprint)
            self.assertEqual(state_fingerprint(policy.state_dict()), fingerprint)
            self.assertFalse(policy.actor[6]._forward_pre_hooks)
            self.assertFalse(policy.actor[6]._forward_hooks)

    def test_each_full_input_must_have_one_archived_hidden_mu_and_applied_row_no_broadcast(self):
        policy, arrays, clip, fingerprint = self.fixture(identical_inputs=True)
        groups = [('reference_actor_hidden',), ('reference_raw_mu', 'reference_action'),
                  ('reference_actor_hidden', 'reference_raw_mu', 'reference_action')]
        for fields in groups:
            current = {key: value.copy() for key, value in arrays.items()}
            # All three rows shortened together would otherwise broadcast and
            # mislabel one archived forward output as proof for eight inputs.
            for key in fields:
                current[key] = current[key][:1, :1].copy()
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.parity(policy, current, clip, fingerprint)

    def test_forward_leading_axes_dtype_and_empty_rows_are_strict(self):
        policy, arrays, clip, fingerprint = self.fixture()
        for kind in ('flat_full', 'wrong_leading', 'float64_hidden', 'empty'):
            current = {key: value.copy() for key, value in arrays.items()}
            if kind == 'flat_full': current['reference_full_obs'] = current['reference_full_obs'].reshape(8, 3102)
            elif kind == 'wrong_leading': current['reference_actor_hidden'] = current['reference_actor_hidden'].reshape(2, 4, 128)
            elif kind == 'float64_hidden': current['reference_actor_hidden'] = current['reference_actor_hidden'].astype(np.float64)
            else: current = {key: value[:0].copy() for key, value in current.items()}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.parity(policy, current, clip, fingerprint)

    def test_source_fingerprint_eval_mode_and_forward_state_mutation_rejected(self):
        policy, arrays, clip, fingerprint = self.fixture()
        with self.assertRaises(ValueError):
            self.parity(policy, arrays, clip, fingerprint, expected_model_fingerprint='f'*64)
        policy.train()
        with self.assertRaises(ValueError):
            self.parity(policy, arrays, clip, fingerprint)
        policy.eval()
        original = policy.act_inference
        def mutating_forward(observations):
            result = original(observations)
            with torch.no_grad(): policy.critic[0].bias.add_(.001)
            return result
        policy.act_inference = mutating_forward
        with self.assertRaisesRegex(ValueError, 'changed the policy'):
            self.parity(policy, arrays, clip, fingerprint)
        self.assertFalse(policy.actor[6]._forward_pre_hooks)
        self.assertFalse(policy.actor[6]._forward_hooks)

    def test_invalid_parity_bounds_are_rejected(self):
        policy, arrays, clip, fingerprint = self.fixture()
        for overrides in (dict(batch_size=0), dict(batch_size=True), dict(batch_size=2.),
                dict(action_clip=0.), dict(action_clip=True), dict(action_clip=np.nan)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.parity(policy, arrays, clip, fingerprint, **overrides)


if __name__ == '__main__':
    unittest.main()
