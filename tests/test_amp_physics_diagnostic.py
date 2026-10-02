import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from humanoid.amp.physics_diagnostic import (GROUPS, SOURCE_SHA, apply_diagnostic_config,
    diagnostic_contract, physical_readback, reset_input_snapshot, validate_request,
    validate_smoke_certificate, validate_source_state)


def config():
    return NS(sim=NS(dt=.001, physx=NS(num_velocity_iterations=0)),
              control=NS(decimation=10), asset=NS(self_collisions=0))


class DiagnosticContractTests(unittest.TestCase):
    def test_each_group_changes_only_registered_physics_switch(self):
        for group in GROUPS:
            cfg = config()
            apply_diagnostic_config(cfg, group)
            self.assertEqual(cfg.sim.dt, .001)
            self.assertEqual(cfg.control.decimation, 10)
            self.assertEqual(cfg.sim.physx.num_velocity_iterations, int(group == 'velocity1'))
            self.assertEqual(cfg.asset.self_collisions, int(group == 'selfoff'))
            self.assertTrue(diagnostic_contract(group)['no_training'])
        with self.assertRaises(ValueError):
            apply_diagnostic_config(config(), 'unregistered')
        cfg = config(); cfg.sim.physx.num_velocity_iterations = 1
        with self.assertRaises(ValueError):
            apply_diagnostic_config(cfg, 'original')

    def test_request_rejects_unmatched_source_and_budget(self):
        for group in GROUPS:
            self.assertTrue(validate_request(group, 'sustain_control', SOURCE_SHA, 5, 2, 1., False))
            self.assertTrue(validate_request(group, 'sustain_control', SOURCE_SHA, 5, 16, 60., True))
        for args in [('original', 'jitter_control', SOURCE_SHA, 5, 16, 60., True),
                     ('original', 'sustain_control', 'wrong', 5, 16, 60., True),
                     ('original', 'sustain_control', SOURCE_SHA, 6, 16, 60., True),
                     ('original', 'sustain_control', SOURCE_SHA, 5, 16, 20., False)]:
            with self.assertRaises(ValueError): validate_request(*args)
        identity = {'test': 1}
        state = dict(amp_identity=identity, completed_updates=2500, iter=2499)
        self.assertTrue(validate_source_state(state, identity))
        for key, bad in [('amp_identity', {}), ('completed_updates', 2750), ('iter', 2500)]:
            changed = dict(state); changed[key] = bad
            with self.assertRaises(ValueError): validate_source_state(changed, identity)

    def test_readback_preserves_unknowns_and_does_not_mutate_properties(self):
        props = [NS(friction=.6, restitution=0., filter=1)]
        before = copy.deepcopy(vars(props[0]))
        gym = NS(get_actor_rigid_shape_properties=lambda *args: props,
                 get_actor_rigid_body_names=lambda *args: ['base_link'],
                 get_actor_rigid_body_shape_indices=lambda *args: [NS(start=0, count=1)])
        env = NS(gym=gym, envs=[0], actor_handles=[0], sim=object(),
                 sim_params=NS(dt=.001, physx=NS(num_velocity_iterations=0)))
        gym.get_sim_params = lambda sim: NS(dt=.001, physx=NS(num_velocity_iterations=1))
        report = physical_readback(env)
        shape = report['physics_shape_properties']
        self.assertEqual(shape['properties'][0]['friction'], .6)
        self.assertIsNone(shape['properties'][0]['contact_offset'])
        self.assertIsNone(report['physics_sim_parameters']['physx']['rest_offset'])
        self.assertEqual(report['physics_sim_parameters']['physx']['num_velocity_iterations'], 1)
        self.assertEqual(shape['body_shape_ranges'], [dict(start=0, count=1)])
        self.assertEqual(vars(props[0]), before)
        del gym.get_actor_rigid_body_shape_indices
        self.assertIsNone(physical_readback(env)['physics_shape_properties']['body_shape_ranges'])
        props[0].friction = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): physical_readback(env)

    def test_reset_snapshot_has_prephysics_state_and_history_without_alias(self):
        names = ('dof_pos', 'dof_vel', 'actions', 'last_actions', 'last_last_actions',
                 'commands', 'gait_start', 'episode_length_buf', 'phase_length_buf', 'rsi_indices')
        env = NS(**{name: torch.zeros(2, 12) for name in names})
        env.env_origins = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        env.root_states = torch.zeros(2, 13); env.root_states[:, :3] = env.env_origins
        env.obs_history = [torch.zeros(2, 5)]; env.critic_history = [torch.ones(2, 7)]
        before = env.root_states.clone()
        snap = reset_input_snapshot(env)
        self.assertEqual(snap['root_states'][0][:3], [0., 0., 0.])
        torch.testing.assert_close(env.root_states, before)
        env.dof_vel.fill_(1.); env.obs_history[0].fill_(2.)
        changed = reset_input_snapshot(env)
        self.assertEqual(snap['dof_vel'][0], [0.]*12)
        self.assertNotEqual(snap['obs_history_sha256'], changed['obs_history_sha256'])
        self.assertEqual(snap['critic_history_sha256'], changed['critic_history_sha256'])

    def test_full_requires_real_smoke_flags_fingerprint_and_traceable_commit(self):
        cert = dict(verified=True, no_training=True, all_finite=True, raw_verified=True,
            readback_verified=True, reset_inputs_exact=True, cloud_execution_verified=True,
            source_checkpoint_sha256=SOURCE_SHA, num_envs=2, duration_s=1.,
            effectiveness_verified=False, dr_unlocked=False, implementation_fingerprint='fingerprint',
            code_commit='a'*40, cloud_task_id='TASK_20261002_999', groups=list(GROUPS),
            group_artifacts={group: dict(sha256='b'*64) for group in GROUPS})
        with patch('humanoid.amp.scaled_experiment.implementation_fingerprint', return_value='fingerprint'), \
                patch('humanoid.amp.physics_diagnostic.subprocess.run', return_value=NS(returncode=0)):
            self.assertTrue(validate_smoke_certificate(cert, 'c'*40, GROUPS, SOURCE_SHA))
            for key, value in [('cloud_execution_verified', False), ('raw_verified', False),
                               ('reset_inputs_exact', False), ('implementation_fingerprint', 'different'),
                               ('code_commit', ''), ('cloud_task_id', ''), ('num_envs', 16)]:
                bad = copy.deepcopy(cert); bad[key] = value
                with self.assertRaises(ValueError): validate_smoke_certificate(bad, 'c'*40, GROUPS, SOURCE_SHA)
            bad = copy.deepcopy(cert); bad['group_artifacts']['original']['sha256'] = ''
            with self.assertRaises(ValueError): validate_smoke_certificate(bad, 'c'*40, GROUPS, SOURCE_SHA)
        with patch('humanoid.amp.scaled_experiment.implementation_fingerprint', return_value='fingerprint'), \
                patch('humanoid.amp.physics_diagnostic.subprocess.run', return_value=NS(returncode=1)):
            with self.assertRaisesRegex(ValueError, 'ancestor'):
                validate_smoke_certificate(cert, 'c'*40, GROUPS, SOURCE_SHA)


if __name__ == '__main__': unittest.main()
