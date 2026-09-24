"""Verify real round6/7 smoke artifacts before admitting a 250-update run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint, validate_cloud_smoke
from humanoid.amp.signal import validate_signal
from humanoid.amp.learnability import assert_no_domain_randomization
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.verify_refinement_smoke import parse_updates


def validate_signal_updates(updates, group):
    if group not in ('signed', 'bridge'):
        raise ValueError('Unknown style-signal group')
    if [row['iteration'] for row in updates] != list(range(1251, 1261)):
        raise ValueError('Missing/duplicate/out-of-order smoke update')
    for row in updates:
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError('Nonfinite training metric')
        if abs(row['weighted_style_reward']-row['style_reward']) >= 1e-8:
            raise ValueError('Unexpected style mixing weight')
        for key in ('style_negative_fraction', 'style_zero_fraction'):
            if not 0 <= row[key] <= 1:
                raise ValueError('Invalid style fraction')
        penalty = row.get('discriminator_bridge_gradient_penalty', 0)
        if group == 'bridge' and (penalty <= 0 or row['style_negative_fraction'] != 0):
            raise ValueError('Bridge intervention not active or wrong reward floor')
        if group == 'signed' and penalty != 0:
            raise ValueError('Signed group must not change D regularization')
    if group == 'signed' and not any(row['style_negative_fraction'] > 0 for row in updates):
        raise ValueError('No actual negative style reward exercised during smoke')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('signed', 'bridge'), required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    experiment = ScaledExperiment(ROOT, ROOT/'configs/amp'/('lafan_walk02_signal_'+args.group+'.json'))
    validate_signal(experiment)
    status = json.loads(args.status.read_text(encoding='utf-8'))['data']['taskBaseInfo']
    if (status['taskId'] != args.task_id or str(status['taskStatus']) != '5' or
        status['goodsId'] != 'ESKU000001' or str(status['userId']) != '4409'):
        raise ValueError('Wrong task/account/resource or not completed')
    packed = torch.load(args.folder/'model_amp_manifest.pt', weights_only=True, map_location='cpu')
    manifest, certificate = json.loads(packed['manifest_json']), json.loads(packed['certificate_json'])
    assert manifest['completion'] == certificate
    validate_cloud_smoke(experiment, certificate, implementation_fingerprint(ROOT))
    assert certificate['code_commit'] == manifest['code_commit'] == args.expected_commit
    assert manifest['mode'] == 'smoke' and manifest['num_envs'] == 32 and manifest['updates'] == 10
    assert manifest['amp_reward'] == experiment.cfg['reward']
    assert manifest['continuation'] == certificate['continuation']
    assert certificate['dr_unlocked'] is False and certificate['effectiveness_verified'] is False
    assert_no_domain_randomization(manifest['env_config'])
    bundle_file = args.folder/'model_9901260.pt'
    bundle, arrays = load_bundle(bundle_file)
    assert bundle['identity'] == experiment.identity() and bundle['code_commit'] == args.expected_commit
    assert bundle['num_envs'] == 2 and bundle['duration_s'] == 1
    assert set(bundle['modes']) == {'standing', 'reference'}
    checkpoint = args.folder/'model_1260.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == bundle['checkpoint_sha256']
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert state['completed_updates'] == 1260 and state['amp_identity'] == experiment.identity()
    for key in ('model_state_dict', 'amp_discriminator_state_dict'):
        assert all(bool(torch.isfinite(v).all()) for v in state[key].values())
    for mode in bundle['modes']:
        for index in range(2):
            episode(arrays, mode, index)
    logs = json.loads(args.logs.read_text(encoding='utf-8'))['data'].replace('\\n', '\n')
    validate_signal_updates(parse_updates(logs), args.group)
    # Do not hide training failures behind a successful-looking artifact.
    # SDK transport traces, if encountered, require separate inspection.
    if any(marker in logs for marker in ('Traceback (most recent call last)', 'CUDA out of memory', 'Nonfinite')):
        raise ValueError('Log error requires explicit inspection before certificate import')
    certificate.update(source_task_id=args.task_id, independent_artifacts_verified=True,
        bundle_sha256=hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
        checkpoint_sha256=bundle['checkpoint_sha256'],
        weighted_style_log_verified=True, intervention_log_verified=True)
    target = ROOT/'docs/validation'/('signal_'+args.group+'_cloud_smoke.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(certificate, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(group=args.group, task=args.task_id, verified=True,
        implementation_fingerprint=certificate['implementation_fingerprint'],
        checkpoint_sha256=bundle['checkpoint_sha256'], dr_unlocked=False)))


if __name__ == '__main__':
    main()
