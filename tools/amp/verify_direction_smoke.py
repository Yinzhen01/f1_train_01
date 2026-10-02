"""Audit real direction smoke artifacts before accepting any formal run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from functools import partial

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint, validate_cloud_smoke, validate_runtime_timing
from humanoid.amp.direction import validate_direction, validate_direction_diagnostics
from humanoid.amp.learnability import assert_no_domain_randomization
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.verify_refinement_smoke import parse_updates
from tools.amp.direction_audit import validate_direction_updates, compare_environment
from tools.amp.verify_horizon_formal import read_cloud_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=('direction', 'sustain', 'jitter', 'mu'), default='direction')
    p.add_argument('--group', required=True)
    p.add_argument('--task-id', required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--status', type=Path, required=True)
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-checkpoint', type=Path, help='Required original control2500 checkpoint for mu only')
    p.add_argument('--expected-commit', required=True)
    a = p.parse_args()
    if a.family == 'mu' and a.source_checkpoint is None:
        p.error('--family mu requires --source-checkpoint')
    if a.family == 'mu':
        from humanoid.amp.mu_temporal import MU_SUBGROUPS
        if a.group not in MU_SUBGROUPS:
            p.error('--family mu requires an explicitly registered mu subgroup')
    torch.set_num_threads(2)
    validate_experiment, validate_diagnostics = validate_direction, validate_direction_diagnostics
    compare_env, validate_updates = compare_environment, validate_direction_updates
    end = 2010
    if a.family == 'sustain':
        from humanoid.amp.sustain import validate_sustain, validate_sustain_diagnostics
        from tools.amp.sustain_audit import compare_environment as compare_env, validate_sustain_updates as validate_updates
        validate_experiment, validate_diagnostics, end = validate_sustain, validate_sustain_diagnostics, 2260
    if a.family == 'jitter':
        from humanoid.amp.jitter import validate_jitter, validate_jitter_diagnostics
        from tools.amp.jitter_audit import compare_environment as compare_env, validate_jitter_updates as validate_updates
        validate_experiment, validate_diagnostics, end = validate_jitter, validate_jitter_diagnostics, 2510
    if a.family == 'mu':
        from humanoid.amp.mu_temporal import validate_mu, validate_mu_diagnostics
        from tools.amp.mu_audit import (compare_environment as compare_env, validate_mu_updates,
            validate_mu_source, validate_mu_loss_report, validate_mu_checkpoint_report, validate_mu_log_identity,
            compare_recording_environment)
        validate_experiment, validate_diagnostics, end = validate_mu, validate_mu_diagnostics, 2510
        validate_updates = partial(validate_mu_updates, group=a.group)
    e = ScaledExperiment(ROOT, ROOT/'configs/amp'/('lafan_walk02_'+a.family+'_'+a.group+'.json'))
    source_experiment = validate_experiment(e)
    status = json.loads(a.status.read_text(encoding='utf-8'))['data']['taskBaseInfo']
    assert status['taskId'] == a.task_id and str(status['taskStatus']) == '5'
    assert str(status['userId']) == '4409' and status['goodsId'] == 'ESKU000001' and status['imageVersion'] == 'V000124'
    packed = torch.load(a.folder/'model_amp_manifest.pt', weights_only=True, map_location='cpu')
    manifest, cert = json.loads(packed['manifest_json']), json.loads(packed['certificate_json'])
    assert manifest['completion'] == cert
    validate_cloud_smoke(e, cert, implementation_fingerprint(ROOT))
    if a.family == 'mu':
        assert manifest['identity'] == cert['identity'] == e.identity()
        assert manifest['implementation_fingerprint'] == cert['implementation_fingerprint'] == implementation_fingerprint(ROOT)
        teacher_hash = validate_mu_source(e, cert['continuation'], a.source_checkpoint)
        validate_mu_loss_report(cert['mu_loss_report'], e.cfg['mu_temporal'], 10, cert['continuation'],
            num_envs=32, rollout_steps=manifest['ppo_config']['runner']['num_steps_per_env'])
    assert cert['code_commit'] == manifest['code_commit'] == a.expected_commit
    assert manifest['mode'] == 'smoke' and manifest['num_envs'] == 32 and manifest['updates'] == 10
    probe = cert['horizon_diagnostics']
    validate_diagnostics(cert['direction_diagnostics'], a.group, probe['control_steps'], 32)
    assert probe['control_steps'] == 10*manifest['ppo_config']['runner']['num_steps_per_env']
    assert probe['captured_transitions'] == 32*probe['control_steps']
    assert manifest['amp_reward'] == e.cfg['reward'] and manifest['continuation'] == cert['continuation']
    assert cert['effectiveness_verified'] is False and cert['dr_unlocked'] is False
    assert_no_domain_randomization(manifest['env_config'])
    validate_runtime_timing(manifest['runtime']['control_dt'], manifest['runtime']['physics_dt'], manifest['env_config']['control']['decimation'])
    source = json.loads(torch.load(a.source_manifest, weights_only=True, map_location='cpu')['manifest_json'])
    assert source['identity'] == source_experiment.identity()
    compare_env(source['env_config'], manifest['env_config'], a.group, smoke=True)
    for field in ('pd_p', 'pd_d', 'dof_properties', 'dof_names', 'physics_dt', 'control_dt', 'actor_history', 'critic_history'):
        assert manifest['runtime'][field] == source['runtime'][field], field
    if a.family == 'mu':
        assert manifest['runtime'] == source['runtime'], 'Changed runtime physics or PD'
    bundle_path = a.folder/('model_%d.pt' % (9900000+end))
    bundle, arrays = load_bundle(bundle_path)
    if a.family in ('jitter', 'mu'):
        from tools.amp.jitter_audit import validate_substep_arrays
        validate_substep_arrays(bundle, arrays)
    assert bundle['identity'] == e.identity() and bundle['code_commit'] == a.expected_commit
    assert bundle['num_envs'] == 2 and bundle['duration_s'] == 1
    if a.family == 'mu':
        compare_recording_environment(manifest['env_config'], bundle['environment'], a.group, 1., 2)
        assert bundle['substep_telemetry']['reward_used'] is False and bundle['substep_telemetry']['control_hz'] == 100
        assert bundle['dof_names'] == manifest['runtime']['dof_names']
        for training, recording in (('pd_p', 'p_gains'), ('pd_d', 'd_gains'),
                ('physics_dt', 'physics_dt'), ('control_dt', 'control_dt'), ('dof_properties', 'dof_properties')):
            assert manifest['runtime'][training] == bundle['runtime'][recording], training
        assert bundle['policy_deterministic'] is True and bundle['effectiveness_verified'] is False
    checkpoint = a.folder/('model_%d.pt' % end)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == bundle['checkpoint_sha256']
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert state['completed_updates'] == end and state['iter'] == end-1 and state['amp_identity'] == e.identity()
    if a.family == 'mu':
        validate_mu_checkpoint_report(state['mu_loss_report'], cert['mu_loss_report'],
            e.cfg['mu_temporal'], 10, cert['continuation'])
        if e.cfg['mu_temporal'].get('frozen_modules'):
            from tools.amp.mu_audit import validate_frozen_checkpoint
            validate_frozen_checkpoint(state, a.source_checkpoint, cert['mu_loss_report'], updates=10)
    for key in ('model_state_dict', 'amp_discriminator_state_dict'):
        assert all(bool(torch.isfinite(v).all()) for v in state[key].values())
    assert set(bundle['modes']) == {'standing', 'reference'}
    for mode in bundle['modes']:
        for index in range(2): episode(arrays, mode, index)
    log_decode = {}
    logs = read_cloud_log(a.logs, allow_post_completion_binary=True, diagnostics=log_decode)
    rows = parse_updates(logs)
    validate_updates(rows)
    if a.family == 'mu':
        validate_mu_loss_report(cert['mu_loss_report'], e.cfg['mu_temporal'], 10, cert['continuation'], rows)
        validate_mu_log_identity(logs, manifest, cert)
    if any(s in logs for s in ('Traceback (most recent call last)', 'CUDA out of memory', 'Nonfinite')):
        raise ValueError('Log error requires inspection before importing certificate')
    assert '[f1-amp-complete]' in logs
    cert.update(source_task_id=a.task_id, independent_artifacts_verified=True,
        checkpoint_sha256=digest, bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        weighted_style_log_verified=True, intervention_log_verified=True, source_environment_equivalence_verified=True,
        raw_log_diagnostics=log_decode)
    if a.family == 'mu':
        cert.update(independent_source_model_sha256=teacher_hash,
            mu_optimization_log_verified=True, frozen_teacher_state_verified=True,
            loss_improvement_is_effectiveness=False, candidate_native_substeps_verified=True)
        if e.cfg['mu_temporal'].get('frozen_modules'):
            cert['frozen_feature_checkpoint_verified'] = True
            cert['independent_source_features_sha256'] = cert['mu_loss_report']['frozen_features_initial_sha256']
            from humanoid.amp.mu_temporal import validate_feature_smoke_admission
            validate_feature_smoke_admission(e, cert)
    target = ROOT/'docs/validation'/(a.family+'_'+a.group+'_cloud_smoke.json')
    with target.open('x', encoding='utf-8') as stream: json.dump(cert, stream, indent=2)
    print(json.dumps(dict(group=a.group, task=a.task_id, verified=True, checkpoint_sha256=digest,
        implementation_fingerprint=cert['implementation_fingerprint'], direction=cert['direction_diagnostics'], contact=cert['contact_diagnostics'], dr_unlocked=False)))


if __name__ == '__main__': main()
