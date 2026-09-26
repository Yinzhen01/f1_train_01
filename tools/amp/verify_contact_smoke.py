"""Audit real contact smoke artifacts before accepting any formal run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint, validate_cloud_smoke, validate_runtime_timing
from humanoid.amp.contact import validate_contact
from humanoid.amp.learnability import assert_no_domain_randomization
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.verify_refinement_smoke import parse_updates
from tools.amp.contact_audit import validate_contact_updates, compare_environment
from tools.amp.verify_horizon_formal import read_cloud_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=('control', 'slip', 'tail'), required=True)
    p.add_argument('--task-id', required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--status', type=Path, required=True)
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    e = ScaledExperiment(ROOT, ROOT/'configs/amp'/('lafan_walk02_contact_'+a.group+'.json'))
    validate_contact(e)
    status = json.loads(a.status.read_text(encoding='utf-8'))['data']['taskBaseInfo']
    assert status['taskId'] == a.task_id and str(status['taskStatus']) == '5'
    assert str(status['userId']) == '4409' and status['goodsId'] == 'ESKU000001' and status['imageVersion'] == 'V000124'
    packed = torch.load(a.folder/'model_amp_manifest.pt', weights_only=True, map_location='cpu')
    manifest, cert = json.loads(packed['manifest_json']), json.loads(packed['certificate_json'])
    assert manifest['completion'] == cert
    validate_cloud_smoke(e, cert, implementation_fingerprint(ROOT))
    assert cert['code_commit'] == manifest['code_commit'] == a.expected_commit
    assert manifest['mode'] == 'smoke' and manifest['num_envs'] == 32 and manifest['updates'] == 10
    probe = cert['horizon_diagnostics']
    assert probe['control_steps'] == 10*manifest['ppo_config']['runner']['num_steps_per_env']
    assert probe['captured_transitions'] == 32*probe['control_steps']
    assert manifest['amp_reward'] == e.cfg['reward'] and manifest['continuation'] == cert['continuation']
    assert cert['effectiveness_verified'] is False and cert['dr_unlocked'] is False
    assert_no_domain_randomization(manifest['env_config'])
    validate_runtime_timing(manifest['runtime']['control_dt'], manifest['runtime']['physics_dt'], manifest['env_config']['control']['decimation'])
    source = json.loads(torch.load(a.source_manifest, weights_only=True, map_location='cpu')['manifest_json'])
    assert source['identity']['experiment'] == 'f1_amp_walk02_horizon_long'
    compare_environment(source['env_config'], manifest['env_config'], a.group, smoke=True)
    for field in ('pd_p', 'pd_d', 'dof_properties', 'dof_names', 'physics_dt', 'control_dt', 'actor_history', 'critic_history'):
        assert manifest['runtime'][field] == source['runtime'][field], field
    bundle_path = a.folder/'model_9901760.pt'
    bundle, arrays = load_bundle(bundle_path)
    assert bundle['identity'] == e.identity() and bundle['code_commit'] == a.expected_commit
    assert bundle['num_envs'] == 2 and bundle['duration_s'] == 1
    checkpoint = a.folder/'model_1760.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == bundle['checkpoint_sha256']
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert state['completed_updates'] == 1760 and state['iter'] == 1759 and state['amp_identity'] == e.identity()
    for key in ('model_state_dict', 'amp_discriminator_state_dict'):
        assert all(bool(torch.isfinite(v).all()) for v in state[key].values())
    assert set(bundle['modes']) == {'standing', 'reference'}
    for mode in bundle['modes']:
        for index in range(2): episode(arrays, mode, index)
    logs = read_cloud_log(a.logs)
    validate_contact_updates(parse_updates(logs))
    if any(s in logs for s in ('Traceback (most recent call last)', 'CUDA out of memory', 'Nonfinite')):
        raise ValueError('Log error requires inspection before importing certificate')
    assert '[f1-amp-complete]' in logs
    cert.update(source_task_id=a.task_id, independent_artifacts_verified=True,
        checkpoint_sha256=digest, bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        weighted_style_log_verified=True, intervention_log_verified=True, source_environment_equivalence_verified=True)
    target = ROOT/'docs/validation'/('contact_'+a.group+'_cloud_smoke.json')
    with target.open('x', encoding='utf-8') as stream: json.dump(cert, stream, indent=2)
    print(json.dumps(dict(group=a.group, task=a.task_id, verified=True, checkpoint_sha256=digest,
        implementation_fingerprint=cert['implementation_fingerprint'], contact=cert['contact_diagnostics'], dr_unlocked=False)))


if __name__ == '__main__': main()
