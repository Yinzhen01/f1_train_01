"""Reject misleading frozen-policy physics comparisons and broken raw timing."""
import copy
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.amp.audit_physics_diagnostic import (validate_environment, validate_raw,
    worst_second, common_prefix_samples, validate_manifest, validate_runtime,
    compare_runtime, fast_geometry, fast_signals, validate_telemetry,
    compare_initialization, observed_initial_comparison, smoke_certificate)
from humanoid.amp.physics_diagnostic import diagnostic_contract
from humanoid.amp.jitter import SOURCE_SHA


class PhysicsDiagnosticAuditTests(unittest.TestCase):
    def test_single_variable_intervention_and_unrelated_pd_rejection(self):
        source = dict(sim=dict(physx=dict(num_velocity_iterations=0, contact_offset=.01)),
            asset=dict(self_collisions=0), control=dict(damping={'ankle': 1.5}))
        self.assertTrue(validate_environment(source, source, 'original'))
        for group, change in (('velocity1', lambda target: target['sim']['physx'].update(num_velocity_iterations=1)),
                              ('selfoff', lambda target: target['asset'].update(self_collisions=1))):
            target = copy.deepcopy(source); change(target)
            self.assertTrue(validate_environment(source, target, group))
            target['control']['damping']['ankle'] = 3.
            with self.assertRaises(ValueError): validate_environment(source, target, group)
        with self.assertRaises(ValueError): validate_environment(source, source, 'velocity1')

    def raw(self):
        velocity = np.zeros((4, 10, 12), dtype=np.float64)
        velocity[:, ::2] = 1.  # Oscillation cancels at endpoint; raw must see it.
        torque = 2*velocity
        force = np.zeros((4, 10, 3, 3)); force[:, 3, 2, 2] = 4000.
        state = np.zeros((4, 10, 3, 13)); state[..., 6] = 1.
        initial = np.zeros((4, 12)); valid = np.ones(4, dtype=bool)
        acceleration = np.diff(np.concatenate((initial[:, None], velocity), axis=1), axis=1)/.001
        delta = np.diff(np.concatenate((initial[:, None], torque), axis=1), axis=1)
        return [velocity, torque, force, state, initial, initial.copy(), velocity[:, -1].copy(),
                torque[:, -1].copy(), state[:, -1, 1:].copy(), (acceleration**2).mean(1),
                np.abs(acceleration).max(1), (delta**2).mean(1),
                np.maximum(force[:, :, 1:, 2], 0).max(1), valid]

    def test_raw_peak_cancellation_and_force_peak_corruption(self):
        values = self.raw()
        self.assertEqual(validate_raw(*values)['samples'], 40)
        self.assertEqual(float(values[6].max()), 0.)
        self.assertEqual(float(values[9].min()), 1e6)
        values[12][0, 1] = 0.  # Endpoint zero cannot replace the measured peak.
        with self.assertRaisesRegex(ValueError, 'foot-force peak'): validate_raw(*values)

    def test_raw_velocity_and_acceleration_statistics_must_agree(self):
        values = self.raw(); values[9][0, 4] = 0.
        with self.assertRaisesRegex(ValueError, 'mean square'): validate_raw(*values)
        values = self.raw(); values[6][0, 4] = 2.
        with self.assertRaisesRegex(ValueError, 'velocity endpoint'): validate_raw(*values)

    def test_worst_sliding_second_ignores_failed_and_missing_tail(self):
        time = (np.arange(400)+1)*.01; energy = np.ones(400); energy[250:350] = 25.
        valid = np.ones(400, dtype=bool); valid[300:] = False
        worst = worst_second(time, energy, valid)
        self.assertAlmostEqual(worst['start_s'], 2.01)
        self.assertAlmostEqual(worst['rms'], np.sqrt(13))
        self.assertEqual(worst['samples'], 100)
        self.assertIsNone(worst_second(time[:199], energy[:199], valid[:199]))

    def manifest(self, group='original'):
        physx = dict(solver_type=1, num_position_iterations=4, num_velocity_iterations=0,
            contact_offset=.01, rest_offset=0., max_depenetration_velocity=1., contact_collection=2)
        shapes = dict(body_names=['base_link', 'left_ankle_roll_link', 'right_ankle_roll_link'],
            body_shape_ranges=[dict(start=i, count=1) for i in range(3)],
            properties=[dict(index=i, friction=.6, rolling_friction=0., torsion_friction=0.,
                restitution=0., compliance=None, contact_offset=None, rest_offset=None,
                filter=0, thickness=None) for i in range(3)])
        snapshot = {key:np.zeros((16, width)).tolist() for key,width in
            (('root_states', 13), ('dof_pos', 12), ('dof_vel', 12), ('actions', 12),
             ('last_actions', 12), ('last_last_actions', 12), ('commands', 5),
             ('gait_start', 1), ('episode_length_buf', 1), ('phase_length_buf', 1), ('rsi_indices', 1))}
        snapshot.update(obs_history_sha256='c'*64, critic_history_sha256='d'*64)
        for value in snapshot['root_states']: value[6] = 1.
        return dict(identity={'source': 'same'}, checkpoint_sha256=SOURCE_SHA,
            code_commit='a'*40, implementation_fingerprint='b'*64,
            diagnostic_only=True, physics_diagnostic=diagnostic_contract(group),
            duration_s=60, num_envs=16, fps=100, modes=['standing', 'reference'],
            mode_seeds={'standing': 5, 'reference': 105}, policy_deterministic=True,
            evaluation_protocol='fixed60_independent_mode_seeds', dr_unlocked=False,
            effectiveness_verified=False,
            substep_telemetry=dict(physics_hz=1000, control_hz=100, samples_per_control=10, reward_used=False),
            urdf_lf_sha256='f'*64,
            environment=dict(sim=dict(dt=.001, physx=physx), control=dict(decimation=10), asset=dict(self_collisions=0)),
            physics_initialization=dict(protocol='identical_reset_inputs_then_native_10ms_warmup',
                warmup_control_dt=.01, reset_inputs={mode:copy.deepcopy(snapshot) for mode in ('standing', 'reference')}),
            runtime=dict(control_dt=.01, physics_dt=.001,
                physics_sim_parameters=dict(dt=.001, physx=copy.deepcopy(physx)),
                physics_shape_properties=shapes, p_gains=[35.]*12))

    def test_long_manifest_rejects_smoke_and_foreign_or_trainable_policy(self):
        source = self.manifest()
        self.assertTrue(validate_manifest(source, 'original', {'source': 'same'}))
        for mutate in (lambda m: m.update(duration_s=1., num_envs=2),
            lambda m: m.update(checkpoint_sha256='wrong'),
            lambda m: m['physics_diagnostic'].update(no_training=False),
            lambda m: m['mode_seeds'].update(reference=106)):
            bad = copy.deepcopy(source); mutate(bad)
            with self.assertRaises(ValueError): validate_manifest(bad, 'original', {'source': 'same'})

    def test_runtime_unknowns_are_retained_and_extra_contact_changes_rejected(self):
        source = self.manifest()
        audit = validate_runtime(source)
        self.assertEqual(audit['unknown_shape_fields']['contact_offset'], [0, 1, 2])
        self.assertIsNone(audit['actual_shape_properties']['properties'][0]['contact_offset'])
        changed = copy.deepcopy(source)
        changed['runtime']['physics_sim_parameters']['physx']['num_velocity_iterations'] = 1
        changed['environment']['sim']['physx']['num_velocity_iterations'] = 1
        validate_runtime(changed)
        self.assertTrue(compare_runtime(source['runtime'], changed['runtime'], 'velocity1')['runtime_single_intervention_verified'])
        changed['runtime']['physics_shape_properties']['properties'][0]['friction'] = .7
        with self.assertRaisesRegex(ValueError, 'shape property'): compare_runtime(source['runtime'], changed['runtime'], 'velocity1')
        changed = copy.deepcopy(source)
        changed['runtime']['physics_sim_parameters']['physx']['contact_offset'] = .002
        with self.assertRaisesRegex(ValueError, 'differs from configuration'): validate_runtime(changed)

    def test_runtime_selfoff_filter_allows_only_known_requested_mask(self):
        source = self.manifest()['runtime']; changed = copy.deepcopy(source)
        for prop in changed['physics_shape_properties']['properties']: prop['filter'] = 1
        result = compare_runtime(source, changed, 'selfoff')
        self.assertEqual(len(result['changed_shape_filters']), 3)
        changed['physics_shape_properties']['properties'][0]['filter'] = 2
        with self.assertRaisesRegex(ValueError, 'filter readback'): compare_runtime(source, changed, 'selfoff')

    def test_common_prefix_retains_failed_status_and_excludes_last_second(self):
        def data(count, failed):
            failure = np.zeros(count, dtype=bool)
            if failed and count: failure[-1] = True
            return dict(time=(np.arange(count)+1)*.01, initial={'failure': False}, failure=failure)
        self.assertEqual(common_prefix_samples(data(6000, False), data(330, True)), 230)
        self.assertEqual(common_prefix_samples(data(400, True), data(330, True)), 230)
        self.assertEqual(common_prefix_samples(data(6000, False), data(6000, False)), 6000)
        self.assertEqual(common_prefix_samples(data(6000, False), data(90, True)), 0)
        left, right = data(200, True), data(200, True); right['time'][10] += .001
        with self.assertRaisesRegex(ValueError, 'different physical time'): common_prefix_samples(left, right)

    def test_fast_mesh_extrema_and_original_sole_centroid_match(self):
        from scipy.spatial.transform import Rotation
        from tools.amp.inspect_rollout import foot_geometry
        cube = np.array([[x, y, z] for x in (-.1, .1) for y in (-.05, .05) for z in (-.03, .03)])
        mesh = np.concatenate((cube, np.array([[.02, .01, -.03], [0., 0., 0.]])))
        vertices = {'left': mesh, 'right': mesh}; geometry = fast_geometry(vertices)
        state = np.zeros((4, 2, 13)); state[..., 2] = .05
        state[:, 0, 3:7] = Rotation.from_euler('xyz', [[.1*i, .07*i, .2*i] for i in range(4)]).as_quat()
        state[:, 1, 3:7] = Rotation.from_euler('xyz', [[-.1*i, .04*i, -.1*i] for i in range(4)]).as_quat()
        state[..., 7:10] = .1; state[..., 10:13] = .3
        data = dict(foot_state=state, dof_pos=np.zeros((4, 12)), dof_vel=np.zeros((4, 12)),
            action=np.zeros((4, 12)), torque=np.zeros((4, 12)),
            initial=dict(dof_pos=np.zeros(12), dof_vel=np.zeros(12), action=np.zeros(12), torque=np.zeros(12)))
        manifest = dict(foot_names=['left', 'right'], dof_names=['j%d'%i for i in range(12)],
            environment=dict(init_state=dict(default_joint_angles={'j%d'%i: 0. for i in range(12)}), control=dict(action_scale=.5)),
            runtime=dict(dof_properties=dict(lower=[-1.]*12, upper=[1.]*12), p_gains=[35.]*12))
        expected_height, expected_speed = foot_geometry(data, vertices, manifest['foot_names'])
        signals = fast_signals(data, manifest, geometry)
        np.testing.assert_allclose(signals['sole_height'], expected_height, atol=1e-12)
        np.testing.assert_allclose(signals['sole_speed'], expected_speed, atol=1e-12)

    def test_all_environment_jensen_and_raw_dimension_not_env_sliced(self):
        raw = self.raw(); n = len(raw[0]); envs = 16
        arrays = dict(standing_dof_vel=np.zeros((n, envs, 12)), standing_torque=np.zeros((n, envs, 12)),
            standing_foot_state=np.repeat(raw[8][:, None], envs, 1),
            standing_valid=np.ones((n, envs), dtype=bool), standing_physics_valid=np.ones((n, envs), dtype=bool),
            standing_initial_dof_vel=np.zeros((envs, 12)), standing_initial_torque=np.zeros((envs, 12)))
        for key, val in zip(('raw_velocity', 'raw_torque', 'raw_body_force', 'raw_body_state',
                            'interval_initial_velocity', 'interval_initial_torque'), raw[:6]):
            arrays['standing_physics_'+key] = val.copy()
        for key, val in zip(('accel_squared', 'accel_peak', 'torque_delta_squared', 'foot_force_peak'), raw[9:13]):
            arrays['standing_physics_'+key] = np.repeat(val[:, None], envs, 1)
        manifest = dict(modes=['standing'], num_envs=16)
        self.assertEqual(validate_telemetry(manifest, arrays)['standing']['samples'], 40)
        broken = copy.deepcopy(arrays); broken['standing_dof_vel'][1, 7, 4] = 100.
        with self.assertRaisesRegex(ValueError, 'Jensen'): validate_telemetry(manifest, broken)
        broken = copy.deepcopy(arrays); broken['standing_physics_raw_velocity'] = broken['standing_physics_raw_velocity'][:, 0]
        with self.assertRaisesRegex(ValueError, 'raw substep shape'): validate_telemetry(manifest, broken)

    def test_early_env0_failure_is_not_forged_raw_success(self):
        values = self.raw(); values[-1][:] = False
        result = validate_raw(*values)
        self.assertEqual(result['samples'], 0)
        self.assertTrue(result['unavailable_due_to_early_failure'])
        self.assertIsNone(result['max_abs_errors'])

    def smoke_manifest(self, group='original'):
        result = self.manifest(group)
        result.update(num_envs=2, duration_s=1., evaluation_protocol='physics_diagnostic_smoke')
        for snapshot in result['physics_initialization']['reset_inputs'].values():
            for key,value in snapshot.items():
                if not key.endswith('_sha256'): snapshot[key] = value[:2]
        return result

    def test_smoke_does_not_satisfy_long_quality_and_fingerprint_is_bound(self):
        source = self.smoke_manifest()
        self.assertTrue(validate_manifest(source, 'original', source['identity'], smoke=True,
                                         fingerprint='b'*64, expected_commit='a'*40))
        with self.assertRaises(ValueError): validate_manifest(source, 'original', source['identity'])
        with self.assertRaisesRegex(ValueError, 'implementation'): validate_manifest(
            source, 'original', source['identity'], smoke=True, fingerprint='e'*64)

    def test_same_injected_reset_can_have_different_native_warmup_response(self):
        left = self.smoke_manifest(); right = self.smoke_manifest('velocity1')
        self.assertTrue(compare_initialization(left, right)['reset_inputs_exact'])
        a = dict(standing_initial_dof_vel=np.zeros((2, 12)), reference_initial_dof_vel=np.zeros((2, 12)))
        b = copy.deepcopy(a); b['standing_initial_dof_vel'][0, 4] = .1
        # Interventional initial response is reported, not rejected or made
        # artificially equal; raw interval initials are not mode reset fields.
        a['standing_physics_interval_initial_velocity'] = np.zeros((100, 12))
        b['standing_physics_interval_initial_velocity'] = np.ones((100, 12))
        self.assertFalse(observed_initial_comparison(a, b)['all_captured_fields_exact_equal'])
        right['physics_initialization']['reset_inputs']['standing']['dof_pos'][0][4] = .001
        with self.assertRaisesRegex(ValueError, 'not exactly matched'): compare_initialization(left, right)
        right = self.smoke_manifest('velocity1')
        right['physics_initialization']['reset_inputs']['reference']['obs_history_sha256'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'not exactly matched'): compare_initialization(left, right)

    def smoke_arrays(self):
        n, envs = 100, 2; arrays = {}
        for mode in ('standing', 'reference'):
            for name, trailing in (('root_state', (13,)), ('dof_pos', (12,)), ('dof_vel', (12,)),
                ('action', (12,)), ('torque', (12,)), ('base_lin_vel', (3,)), ('base_ang_vel', (3,)),
                ('foot_state', (2, 13)), ('foot_force', (2, 3)), ('key_positions_w', (4, 3))):
                value = np.zeros((n, envs)+trailing)
                if name in ('root_state', 'foot_state'): value[..., 6] = 1.
                arrays[mode+'_'+name] = value
                arrays[mode+'_initial_'+name] = value[0].copy()
            arrays[mode+'_valid'] = np.ones((n, envs), dtype=bool)
            arrays[mode+'_failure'] = np.zeros((n, envs), dtype=bool)
            arrays[mode+'_initial_failure'] = np.zeros(envs, dtype=bool)
            arrays[mode+'_time'] = (np.arange(n)+1)*.01
            valid = np.ones((n, envs), dtype=bool); valid[:3] = False
            arrays[mode+'_physics_valid'] = valid
            for name, width in (('accel_squared', 12), ('accel_peak', 12), ('torque_delta_squared', 12), ('foot_force_peak', 2)):
                arrays[mode+'_physics_'+name] = np.zeros((n, envs, width))
            for name in ('raw_velocity', 'raw_torque'): arrays[mode+'_physics_'+name] = np.zeros((n, 10, 12))
            for name in ('interval_initial_velocity', 'interval_initial_torque'): arrays[mode+'_physics_'+name] = np.zeros((n, 12))
            arrays[mode+'_physics_raw_body_force'] = np.zeros((n, 10, 3, 3))
            state = np.zeros((n, 10, 3, 13)); state[..., 6] = 1.
            arrays[mode+'_physics_raw_body_state'] = state
        return arrays

    def test_smoke_cli_writes_bound_certificate_without_cloud_or_quality_claim(self):
        import tools.amp.audit_physics_diagnostic as audit
        manifests = [self.smoke_manifest(group) for group in audit.GROUPS]
        manifests[1]['environment']['sim']['physx']['num_velocity_iterations'] = 1
        manifests[1]['runtime']['physics_sim_parameters']['physx']['num_velocity_iterations'] = 1
        manifests[2]['environment']['asset']['self_collisions'] = 1
        arrays = self.smoke_arrays()
        with TemporaryDirectory() as folder:
            paths = [Path(folder)/(group+'.pt') for group in audit.GROUPS]
            for path in paths: path.write_bytes(b'test fixture only')
            output = Path(folder)/'audit'
            argv = ['audit', '--smoke', '--expected-commit', 'a'*40, '--output', str(output)]
            for group,path in zip(audit.GROUPS, paths): argv += ['--'+group, str(path)]
            with patch.object(audit, 'ScaledExperiment', return_value=SimpleNamespace(identity=lambda: {'source': 'same'})), \
                    patch.object(audit, 'implementation_fingerprint', return_value='b'*64), \
                    patch.object(audit, 'load_bundle', side_effect=[(m, copy.deepcopy(arrays)) for m in manifests]), \
                    patch.object(audit.sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                audit.main()
            certificate = json.loads((output/'certificate.json').read_text())
            self.assertTrue(certificate['verified'])
            self.assertFalse(certificate['effectiveness_verified'])
            self.assertFalse(certificate['cloud_execution_verified'])
            self.assertEqual(certificate['groups'], list(audit.GROUPS))
            self.assertEqual(certificate['code_commit'], 'a'*40)
            self.assertEqual(certificate['group_evidence']['original']['raw_audit']['standing']['samples'], 970)
            with patch.object(audit.sys, 'argv', argv):
                with self.assertRaises(FileExistsError): audit.main()


if __name__ == '__main__': unittest.main()
