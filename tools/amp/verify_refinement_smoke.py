"""Import a cloud smoke certificate only after checking its real artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint, validate_cloud_smoke
from humanoid.amp.learnability import assert_no_domain_randomization
from tools.amp.inspect_rollout import load_bundle, episode


def parse_updates(logs):
    # SDK uploader output may share a line with the training marker.
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(logs[match.end():])[0]
            for match in re.finditer(re.escape('[f1-amp-update] '), logs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('control', 'smooth', 'noamp'), required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--user-id', default='4252')
    parser.add_argument('--interrupted', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    experiment = ScaledExperiment(ROOT, ROOT/'configs/amp'/('lafan_walk02_refine_'+args.group+'.json'))
    status = json.loads(args.status.read_text(encoding='utf-8'))['data']['taskBaseInfo']
    if (status['taskId'] != args.task_id or str(status['taskStatus']) != '5' or
        status['goodsId'] != 'ESKU000001' or str(status['userId']) != args.user_id):
        raise ValueError('Wrong task/account/resource or not completed')
    packed = torch.load(args.folder/'model_amp_manifest.pt', weights_only=True, map_location='cpu')
    manifest, certificate = json.loads(packed['manifest_json']), json.loads(packed['certificate_json'])
    assert manifest['completion'] == certificate
    validate_cloud_smoke(experiment, certificate, implementation_fingerprint(ROOT), interrupted=args.interrupted)
    assert certificate['code_commit'] == manifest['code_commit'] == args.expected_commit
    assert manifest['mode'] == 'smoke' and manifest['num_envs'] == 32 and manifest['updates'] == 10
    assert manifest['amp_reward'] == experiment.cfg['reward']
    assert certificate['dr_unlocked'] is False and certificate['effectiveness_verified'] is False
    assert_no_domain_randomization(manifest['env_config'])
    completed = 1260 if args.interrupted else 1010
    bundle_file = args.folder/('model_%d.pt' % (9900000+completed))
    bundle, arrays = load_bundle(bundle_file)
    assert bundle['identity'] == experiment.identity() and bundle['code_commit'] == args.expected_commit
    assert bundle['num_envs'] == 2 and bundle['duration_s'] == 1
    checkpoint = args.folder/('model_%d.pt' % completed)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == bundle['checkpoint_sha256']
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert state['completed_updates'] == completed and state['amp_identity'] == experiment.identity()
    for key in ('model_state_dict', 'amp_discriminator_state_dict'):
        assert all(bool(torch.isfinite(v).all()) for v in state[key].values())
    for mode in bundle['modes']:
        for index in range(2):
            episode(arrays, mode, index)  # Reject artificial post-reset continuations.
    logs = json.loads(args.logs.read_text(encoding='utf-8'))['data'].replace('\\n', '\n')
    updates = parse_updates(logs)
    assert [row['iteration'] for row in updates] == list(range(completed-9, completed+1))
    for row in updates:
        assert abs(row['weighted_style_reward'] - row['style_reward']*experiment.cfg['reward']['style_weight']) < 1e-8
    certificate.update(source_task_id=args.task_id, independent_artifacts_verified=True,
        bundle_sha256=hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
        checkpoint_sha256=bundle['checkpoint_sha256'],
        weighted_style_log_verified=True)
    target = ROOT/'docs/validation'/('refine_'+args.group+('_restart' if args.interrupted else '')+'_cloud_smoke.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(certificate, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(group=args.group, task=args.task_id, verified=True,
        implementation_fingerprint=certificate['implementation_fingerprint'],
        checkpoint_sha256=bundle['checkpoint_sha256'], dr_unlocked=False)))


if __name__ == '__main__':
    main()
