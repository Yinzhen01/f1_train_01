"""Audit completed signal runs; completion never implies gait effectiveness."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint
from humanoid.amp.signal import validate_signal_certificate
from humanoid.amp.learnability import assert_no_domain_randomization
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.verify_refinement_smoke import parse_updates


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left: assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_equal(a, b)
    else:
        assert left == right


def inspect_tracebacks(logs):
    # Exact SDK-only allowlist; counts retained in audit, never called clean logs.
    blocks = list(re.finditer(r'Traceback \(most recent call last\):\r?\n((?:[ \t].*\r?\n)+)([^\r\n]+)', logs))
    assert len(blocks) == len(re.findall(r'Traceback \(most recent call last\):', logs))
    for block in blocks:
        frames = re.findall(r'File "([^"]+)"', block[1])
        assert frames and all('/site-packages/pika/adapters/utils/io_services_utils.py' in p for p in frames)
        assert block[2] == 'ConnectionResetError: [Errno 104] Connection reset by peer'
    return len(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('signed', 'bridge'), required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = args.folder
    experiment = ScaledExperiment(ROOT, ROOT/'configs/amp'/('lafan_walk02_signal_'+args.group+'.json'))
    status = json.loads(args.status.read_text(encoding='utf-8'))['data']['taskBaseInfo']
    assert status['taskId'] == args.task_id and str(status['taskStatus']) == '5'
    assert str(status['userId']) == '4409' and status['goodsId'] == 'ESKU000001'
    packed = torch.load(folder/'model_amp_manifest.pt', weights_only=True, map_location='cpu')
    manifest = json.loads(packed['manifest_json']); certificate = json.loads(packed['certificate_json'])
    assert manifest['completion'] == certificate
    assert manifest['identity'] == certificate['identity'] == experiment.identity()
    assert manifest['code_commit'] == certificate['code_commit'] == args.expected_commit
    assert manifest['implementation_fingerprint'] == certificate['implementation_fingerprint'] == implementation_fingerprint(ROOT)
    assert manifest['mode'] == 'formal' and manifest['num_envs'] == certificate['num_envs'] == 4096
    assert manifest['updates'] == certificate['updates'] == 250
    assert manifest['amp_reward'] == experiment.cfg['reward']
    assert_no_domain_randomization(manifest['env_config'])
    validate_signal_certificate(experiment, manifest['continuation'])
    assert manifest['continuation'] == certificate['continuation']
    for key in ('all_finite', 'complete', 'actor_updated', 'discriminator_updated', 'no_dr_configuration_verified',
                'physics_dt_verified', 'body_frames_verified', 'reset_history_verified', 'nonzero_style_reward'):
        assert certificate[key] is True, key
    assert certificate['dr_unlocked'] is False and certificate['effectiveness_verified'] is False
    checkpoint = folder/'model_8801500.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bundle_file = folder/'model_9901500.pt'
    bundle, arrays = load_bundle(bundle_file)
    assert bundle['checkpoint_sha256'] == digest and bundle['identity'] == experiment.identity()
    assert bundle['code_commit'] == manifest['code_commit'] and bundle['num_envs'] == 16 and bundle['duration_s'] == 20
    assert set(bundle['modes']) == {'standing', 'reference'}
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    final = torch.load(folder/'model_1500.pt', weights_only=True, map_location='cpu')
    assert state['completed_updates'] == final['completed_updates'] == 1500
    assert state['amp_identity'] == final['amp_identity'] == experiment.identity()
    for key in ('model_state_dict', 'optimizer_state_dict', 'es_optimizer_state_dict',
                'amp_discriminator_state_dict', 'amp_optimizer_state_dict', 'amp_replay_state'):
        assert_equal(state[key], final[key])
    logs = args.logs.read_text(encoding='utf-8')
    updates = parse_updates(logs)
    assert [r['iteration'] for r in updates] == list(range(1251, 1501))
    for row in updates:
        assert all(math.isfinite(v) for v in row.values())
        assert abs(row['weighted_style_reward']-row['style_reward']) < 1e-8
        penalty = row.get('discriminator_bridge_gradient_penalty', 0)
        assert (penalty > 0 and row['style_negative_fraction'] == 0) if args.group == 'bridge' else penalty == 0
    if args.group == 'signed':
        assert any(row['style_negative_fraction'] > 0 for row in updates)
    assert '[f1-amp-complete]' in logs
    assert 'CUDA out of memory' not in logs and 'Nonfinite' not in logs
    tracebacks = inspect_tracebacks(logs)
    survival = {}
    for mode in bundle['modes']:
        passed = 0
        for index in range(16):
            data = episode(arrays, mode, index)
            passed += int(len(data['time']) == 2000 and not data['initial']['failure'] and not data['failure'].any())
        survival[mode] = passed
    result = dict(task=args.task_id, group=args.group, verified=True, checkpoint_sha256=digest,
        final_checkpoint_sha256=hashlib.sha256((folder/'model_1500.pt').read_bytes()).hexdigest(),
        bundle_sha256=hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
        completed_updates=1500, additional_updates=250, no_dr_verified=True,
        sdk_connection_reset_tracebacks=tracebacks, final_eval_learning_state_equal=True,
        independent_evaluation_survival=survival, effectiveness_verified=False, dr_unlocked=False)
    with (folder/'formal_artifact_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
