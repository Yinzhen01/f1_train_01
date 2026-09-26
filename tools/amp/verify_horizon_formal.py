"""Verify bounded horizon training and matched independent60 evaluation artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.horizon import validate_horizon_certificate, validate_horizon_probe
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint, validate_runtime_timing
from humanoid.amp.learnability import assert_no_domain_randomization
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.verify_long_pair import initial_comparison
from tools.amp.verify_refinement_smoke import parse_updates
from tools.amp.verify_signal_formal import assert_equal, inspect_tracebacks


def decode_post_completion_log(raw, diagnostics):
    """Retain visible escapes for binary bytes only after a complete training record.

    This is not errors='ignore': ASCII errors/tracebacks remain inspectable, raw
    evidence is never edited, and corruption before/inside completion is fatal.
    """
    positions = []
    try: raw.decode('utf-8')
    except UnicodeDecodeError as error: positions.append(error.start)
    if b'\0' in raw: positions.append(raw.index(b'\0'))
    if not positions: return raw.decode('utf-8')
    first = min(positions)
    prefix = raw[:first].decode('utf-8')
    marker = '[f1-amp-complete] '
    if prefix.count(marker) != 1 or '[amp-eval-complete]' not in prefix:
        raise ValueError('Binary data before verified evaluation/training completion')
    start = prefix.index(marker)+len(marker)
    certificate, size = json.JSONDecoder().raw_decode(prefix[start:])
    if certificate.get('complete') is not True or certificate.get('all_finite') is not True:
        raise ValueError('Binary log has no complete finite training record')
    end = start+size
    if 'uploaded successfully' not in prefix[end:]:
        raise ValueError('Unexpected binary data before SDK upload tail')
    decoded = raw.decode('utf-8', errors='surrogateescape')
    invalid_count = sum(0xdc80 <= ord(c) <= 0xdcff for c in decoded)
    # Preserve malformed bytes and control bytes as explicit visible escapes.
    text = raw.decode('utf-8', errors='backslashreplace')
    text = ''.join('\\x%02x' % ord(c) if (ord(c) < 32 and c not in '\t\r\n') or ord(c) == 127 else c for c in text)
    diagnostics.update(raw_log_sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw), escaped_post_completion_binary=True,
        first_binary_byte_offset=first, completion_record_end_byte=len(prefix[:end].encode('utf-8')),
        nul_count=raw.count(b'\0'), invalid_utf8_byte_count=invalid_count,
        note='Raw file retained. Only post-completion SDK tail contains escaped bytes; errors are not removed.')
    return text


def read_cloud_log(path, *, allow_post_completion_binary=False, diagnostics=None):
    """The live API is a tail; completed argo downloads are raw full logs."""
    diagnostics = {} if diagnostics is None else diagnostics
    try:
        text = path.read_text(encoding='utf-8')
        if '\0' in text: raise ValueError('Binary NUL in log')
    except (UnicodeDecodeError, ValueError):
        if path.suffix != '.log' or not allow_post_completion_binary: raise
        text = decode_post_completion_log(path.read_bytes(), diagnostics)
    if path.suffix == '.json':
        text = json.loads(text)['data'].replace('\\n', '\n')
    elif path.suffix != '.log':
        raise ValueError('Explicit .json API envelope or .log full artifact required')
    return text


def validate_updates(rows):
    if [r['iteration'] for r in rows] != list(range(1501, 1751)):
        raise ValueError('Not exactly250 additional updates from1500')
    for row in rows:
        if (not all(math.isfinite(v) for v in row.values()) or
            abs(row['weighted_style_reward']-row['style_reward']) >= 1e-8 or
            row['discriminator_bridge_gradient_penalty'] <= 0 or row['style_negative_fraction'] != 0):
            raise ValueError('Invalid/changed bridge update mechanics')
    return True


def main():
    p = argparse.ArgumentParser()
    for name in ('group', 'task-id', 'expected-commit'):
        p.add_argument('--'+name, required=True)
    for name in ('folder', 'status', 'logs', 'source-manifest', 'source-evaluation'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.group not in ('short', 'long'): raise ValueError('Unknown horizon group')
    torch.set_num_threads(2)
    e = ScaledExperiment(ROOT, ROOT/'configs/amp'/('lafan_walk02_horizon_'+a.group+'.json'))
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
    validate_horizon_certificate(e, m['continuation'])
    assert m['continuation'] == c['continuation']
    validate_horizon_probe(c['horizon_diagnostics'], e.cfg['horizon']['episode_length_s'], False)
    probe = c['horizon_diagnostics']
    assert probe['control_steps'] == 250*m['ppo_config']['runner']['num_steps_per_env']
    assert probe['captured_transitions'] == probe['control_steps']*4096
    for field in ('all_finite', 'complete', 'actor_updated', 'discriminator_updated', 'no_dr_configuration_verified',
                  'physics_dt_verified', 'body_frames_verified', 'reset_history_verified', 'nonzero_style_reward'):
        assert c[field] is True, field
    assert c['dr_unlocked'] is False and c['effectiveness_verified'] is False
    source = json.loads(torch.load(a.source_manifest, weights_only=True, map_location='cpu')['manifest_json'])
    assert source['identity']['experiment'] == 'f1_amp_walk02_signal_bridge'
    after = json.loads(json.dumps(m['env_config']))
    assert after['env']['episode_length_s'] == e.cfg['horizon']['episode_length_s']
    after['env']['episode_length_s'] = source['env_config']['env']['episode_length_s']
    assert after == source['env_config'], 'Changed training environment beyond episode horizon'
    assert m['runtime'] == source['runtime'], 'Changed runtime physics or PD'
    checkpoint = a.folder/'model_8801750.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bundle_path = a.folder/'model_9901750.pt'
    bundle, arrays = load_bundle(bundle_path)
    assert bundle['checkpoint_sha256'] == digest and bundle['identity'] == e.identity()
    assert bundle['code_commit'] == a.expected_commit and bundle['num_envs'] == 16 and bundle['duration_s'] == 60
    assert bundle['evaluation_protocol'] == 'fixed60_independent_mode_seeds'
    assert bundle['mode_seeds'] == {'standing': 5, 'reference': 105} and bundle['policy_deterministic'] is True
    assert bundle['modes'] == ['standing', 'reference']
    baseline, base_arrays = load_bundle(a.source_evaluation)
    assert baseline['checkpoint_sha256'] == e.cfg['horizon']['source_checkpoint_sha256']
    assert baseline['duration_s'] == 60 and baseline['num_envs'] == 16
    assert baseline['mode_seeds'] == bundle['mode_seeds']
    assert baseline['evaluation_protocol'] == bundle['evaluation_protocol']
    assert baseline['runtime'] == bundle['runtime'] and baseline['environment'] == bundle['environment']
    initial = initial_comparison(base_arrays, arrays)
    assert initial['all_captured_fields_exact_equal'], 'Initial state mismatch; do not call matched comparison'
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    final = torch.load(a.folder/'model_1750.pt', weights_only=True, map_location='cpu')
    assert state['completed_updates'] == final['completed_updates'] == 1750
    assert state['iter'] == final['iter'] == 1749
    assert state['amp_identity'] == final['amp_identity'] == e.identity()
    for key in ('model_state_dict', 'optimizer_state_dict', 'es_optimizer_state_dict',
                'amp_discriminator_state_dict', 'amp_optimizer_state_dict', 'amp_replay_state'):
        assert_equal(state[key], final[key])
    for key in ('model_state_dict', 'amp_discriminator_state_dict'):
        assert all(bool(torch.isfinite(v).all()) for v in state[key].values())
    logs = read_cloud_log(a.logs)
    validate_updates(parse_updates(logs))
    assert '[f1-amp-complete]' in logs and 'CUDA out of memory' not in logs and 'Nonfinite' not in logs
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
        completed_updates=1750, no_dr_verified=True, sdk_connection_reset_tracebacks=traces,
        initial_comparison=initial, horizon_diagnostics=probe, modes=modes,
        final_eval_learning_state_equal=True, effectiveness_verified=False, dr_unlocked=False)
    with (a.folder/'formal_artifact_audit.json').open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(dict(task=a.task_id, group=a.group, verified=True, horizon=probe,
        survival={m: sum(r['survived'] for r in rows) for m, rows in modes.items()}, dr_unlocked=False)))


if __name__ == '__main__': main()
