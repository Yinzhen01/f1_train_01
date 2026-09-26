"""Verify bounded direction training and matched independent60 evaluation artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.horizon import validate_horizon_probe
from humanoid.amp.contact import validate_contact_diagnostics
from humanoid.amp.direction import validate_direction_certificate, validate_direction_diagnostics
from tools.amp.direction_audit import validate_direction_updates, compare_environment
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint, validate_runtime_timing
from humanoid.amp.learnability import assert_no_domain_randomization
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.verify_long_pair import initial_comparison
from tools.amp.verify_refinement_smoke import parse_updates
from tools.amp.verify_signal_formal import assert_equal, inspect_tracebacks


from tools.amp.verify_horizon_formal import read_cloud_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=('direction', 'sustain'), default='direction')
    p.add_argument('--output', type=Path, help='Optional new audit path; never overwrites an existing report')
    for name in ('group', 'task-id', 'expected-commit'):
        p.add_argument('--'+name, required=True)
    for name in ('folder', 'status', 'logs', 'source-manifest', 'source-evaluation'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    validate_certificate, validate_diagnostics = validate_direction_certificate, validate_direction_diagnostics
    compare_env, validate_updates = compare_environment, validate_direction_updates
    end = 2250
    if a.family == 'sustain':
        from humanoid.amp.sustain import validate_sustain_certificate, validate_sustain_diagnostics
        from tools.amp.sustain_audit import compare_environment as compare_env, validate_sustain_updates as validate_updates
        validate_certificate, validate_diagnostics, end = validate_sustain_certificate, validate_sustain_diagnostics, 2500
    torch.set_num_threads(2)
    e = ScaledExperiment(ROOT, ROOT/'configs/amp'/('lafan_walk02_'+a.family+'_'+a.group+'.json'))
    status = json.loads(a.status.read_text(encoding='utf-8'))['data']['taskBaseInfo']
    assert status['taskId'] == a.task_id and str(status['taskStatus']) == '5'
    assert str(status['userId']) == '4409' and status['goodsId'] == 'ESKU000001' and status['imageVersion'] == 'V000124'
    packed = torch.load(a.folder/'model_amp_manifest.pt', weights_only=True, map_location='cpu')
    m, c = json.loads(packed['manifest_json']), json.loads(packed['certificate_json'])
    assert m['completion'] == c and m['identity'] == c['identity'] == e.identity()
    assert m['code_commit'] == c['code_commit'] == a.expected_commit
    assert m['implementation_fingerprint'] == c['implementation_fingerprint'] == implementation_fingerprint(ROOT)
    assert m['mode'] == 'formal' and m['num_envs'] == c['num_envs'] == 4096
    assert m['updates'] == c['updates'] == 250 and m['amp_reward'] == e.cfg['reward']
    assert_no_domain_randomization(m['env_config'])
    validate_runtime_timing(m['runtime']['control_dt'], m['runtime']['physics_dt'], m['env_config']['control']['decimation'])
    validate_certificate(e, m['continuation'])
    assert m['continuation'] == c['continuation']
    validate_horizon_probe(c['horizon_diagnostics'], 60., False)
    probe = c['horizon_diagnostics']
    assert probe['control_steps'] == 250*m['ppo_config']['runner']['num_steps_per_env']
    assert probe['captured_transitions'] == probe['control_steps']*4096
    for field in ('all_finite', 'complete', 'actor_updated', 'discriminator_updated', 'no_dr_configuration_verified',
                  'physics_dt_verified', 'body_frames_verified', 'reset_history_verified', 'nonzero_style_reward'):
        assert c[field] is True, field
    assert c['dr_unlocked'] is False and c['effectiveness_verified'] is False
    source = json.loads(torch.load(a.source_manifest, weights_only=True, map_location='cpu')['manifest_json'])
    assert source['identity'] == ScaledExperiment(ROOT, ROOT/e.cfg[a.family]['source_config']).identity()
    compare_env(source['env_config'], m['env_config'], a.group)
    validate_contact_diagnostics(c['contact_diagnostics'], 'tail', probe['control_steps'], 4096)
    validate_diagnostics(c['direction_diagnostics'], a.group, probe['control_steps'], 4096)
    assert m['runtime'] == source['runtime'], 'Changed runtime physics or PD'
    checkpoint = a.folder/('model_%d.pt' % (8800000+end))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bundle_path = a.folder/('model_%d.pt' % (9900000+end))
    bundle, arrays = load_bundle(bundle_path)
    assert bundle['checkpoint_sha256'] == digest and bundle['identity'] == e.identity()
    assert bundle['code_commit'] == a.expected_commit and bundle['num_envs'] == 16 and bundle['duration_s'] == 60
    assert bundle['evaluation_protocol'] == 'fixed60_independent_mode_seeds'
    assert bundle['mode_seeds'] == {'standing': 5, 'reference': 105} and bundle['policy_deterministic'] is True
    assert bundle['modes'] == ['standing', 'reference']
    baseline, base_arrays = load_bundle(a.source_evaluation)
    assert baseline['checkpoint_sha256'] == e.cfg[a.family]['source_checkpoint_sha256']
    assert baseline['duration_s'] == 60 and baseline['num_envs'] == 16
    assert baseline['mode_seeds'] == bundle['mode_seeds']
    assert baseline['evaluation_protocol'] == bundle['evaluation_protocol']
    assert baseline['runtime'] == bundle['runtime']
    compare_env(baseline['environment'], bundle['environment'], a.group)
    initial = initial_comparison(base_arrays, arrays)
    assert initial['all_captured_fields_exact_equal'], 'Initial state mismatch; do not call matched comparison'
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    final = torch.load(a.folder/('model_%d.pt' % end), weights_only=True, map_location='cpu')
    assert state['completed_updates'] == final['completed_updates'] == end
    assert state['iter'] == final['iter'] == end-1
    assert state['amp_identity'] == final['amp_identity'] == e.identity()
    for key in ('model_state_dict', 'optimizer_state_dict', 'es_optimizer_state_dict',
                'amp_discriminator_state_dict', 'amp_optimizer_state_dict', 'amp_replay_state'):
        assert_equal(state[key], final[key])
    for key in ('model_state_dict', 'amp_discriminator_state_dict'):
        assert all(bool(torch.isfinite(v).all()) for v in state[key].values())
    log_diagnostics = {}
    logs = read_cloud_log(a.logs, allow_post_completion_binary=a.family == 'sustain', diagnostics=log_diagnostics)
    validate_updates(parse_updates(logs), formal=True)
    assert '[f1-amp-complete]' in logs and 'CUDA out of memory' not in logs and 'Nonfinite' not in logs
    complete_marker = '[f1-amp-complete] '
    assert logs.count(complete_marker) == 1
    certificate_start = logs.index(complete_marker)+len(complete_marker)
    logged_certificate, certificate_size = json.JSONDecoder().raw_decode(logs[certificate_start:])
    assert logged_certificate == c, 'Log completion differs from uploaded manifest'
    wrapper_failed = 'RL task is failed.' in logs
    if wrapper_failed:
        assert logs.index('RL task is failed.') > certificate_start+certificate_size
    log_diagnostics.update(wrapper_failure_banner=wrapper_failed,
        wrapper_success_banner='RL task is successful.' in logs,
        platform_terminal_status=str(status['taskStatus']))
    traces = inspect_tracebacks(logs)
    modes = {}
    for mode in bundle['modes']:
        rows = []
        for index in range(16):
            data = episode(arrays, mode, index)
            failure = bool(data['initial']['failure'] or data['failure'].any())
            if data['failure'].any(): assert data['failure'][-1] and data['failure'].sum() == 1
            survived = len(data['time']) == 6000 and not failure
            assert survived or failure, 'Truncated nonterminal evaluation'
            rows.append(dict(env=index, survived=survived, failure=failure, observed_s=len(data['time'])*.01))
        modes[mode] = rows
    result = dict(task=a.task_id, group=a.group, verified=True, checkpoint_sha256=digest,
        bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(), additional_updates=250,
        completed_updates=end, no_dr_verified=True, sdk_connection_reset_tracebacks=traces, log_diagnostics=log_diagnostics,
        initial_comparison=initial, horizon_diagnostics=probe, contact_diagnostics=c['contact_diagnostics'], direction_diagnostics=c['direction_diagnostics'], modes=modes,
        final_eval_learning_state_equal=True, effectiveness_verified=False, dr_unlocked=False)
    with (a.output or a.folder/'formal_artifact_audit.json').open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(dict(task=a.task_id, group=a.group, verified=True, horizon=probe,
        survival={m: sum(r['survived'] for r in rows) for m, rows in modes.items()}, dr_unlocked=False)))


if __name__ == '__main__': main()
