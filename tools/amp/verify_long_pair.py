"""Read-only provenance and initial-state audit of the explicit fixed60 protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.evaluate_long_horizon import SOURCES
from tools.amp.verify_signal_formal import inspect_tracebacks
from tools.amp.verify_refinement_smoke import parse_updates
from humanoid.amp.scaled_experiment import validate_runtime_timing


def initial_comparison(left, right):
    keys = sorted(key for key in left if '_initial_' in key)
    if keys != sorted(key for key in right if '_initial_' in key):
        raise ValueError('Different captured initial fields')
    fields = {}
    for key in keys:
        a, b = left[key], right[key]
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid initial state shape or values')
        fields[key] = dict(exact_equal=bool(np.array_equal(a, b)),
                           max_abs_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))))
    return dict(all_captured_fields_exact_equal=all(row['exact_equal'] for row in fields.values()), fields=fields,
                limitation='Observed states only; no claim about hidden PhysX solver state or other training seeds.')


def audit(folder, task, source_task, checkpoint, expected_commit):
    stem = folder/task
    status = json.loads((folder/(task+'-status-03.json')).read_text(encoding='utf-8'))['data']
    base = status['taskBaseInfo']
    assert base['taskId'] == task and str(base['taskStatus']) == '5'
    assert str(base['userId']) == '4409' and base['goodsId'] == 'ESKU000001'
    assert base['imageVersion'] == 'V000124'
    source = SOURCES[source_task]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == source['sha256']
    bundle_file = stem/'model_9961500.pt'
    manifest, arrays = load_bundle(bundle_file)
    assert manifest['checkpoint_sha256'] == digest and manifest['code_commit'] == expected_commit
    assert manifest['identity']['experiment'] == source['experiment']
    assert manifest['duration_s'] == 60 and manifest['num_envs'] == 16
    assert manifest['evaluation_protocol'] == 'fixed60_independent_mode_seeds'
    assert manifest['mode_seeds'] == {'standing': 5, 'reference': 105}
    assert manifest['modes'] == ['standing', 'reference'] and manifest['policy_deterministic'] is True
    validate_runtime_timing(manifest['runtime']['control_dt'], manifest['runtime']['physics_dt'],
                            manifest['environment']['control']['decimation'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['completed_updates'] == 1500 and state['amp_identity'] == manifest['identity']
    logs = (folder/(task+'-full.log')).read_text(encoding='utf-8')
    assert not parse_updates(logs)
    decoder = json.JSONDecoder()
    for marker in ('amp-long-audit-start', 'amp-long-audit-complete'):
        matches = list(re.finditer(re.escape('['+marker+'] '), logs))
        assert len(matches) == 1
        entry = decoder.raw_decode(logs[matches[0].end():])[0]
        assert entry['source_task'] == source_task and entry['no_training'] is True
    assert '[amp-eval-complete]' in logs and 'CUDA out of memory' not in logs and 'Nonfinite' not in logs
    traces = inspect_tracebacks(logs)
    modes = {}
    for mode in manifest['modes']:
        rows = []
        for index in range(16):
            data = episode(arrays, mode, index)
            failure = bool(data['initial']['failure'] or data['failure'].any())
            if data['failure'].any():
                assert data['failure'][-1] and data['failure'].sum() == 1
            survived = len(data['time']) == 6000 and not failure
            assert survived or failure
            rows.append(dict(env=index, survived=survived, failure=failure, observed_s=len(data['time'])*.01))
        modes[mode] = rows
    return dict(task=task, source_task=source_task, checkpoint_sha256=digest,
                bundle_sha256=hashlib.sha256(bundle_file.read_bytes()).hexdigest(), code_commit=expected_commit,
                no_training=True, no_dr_verified=True, duration_s=60, sdk_connection_reset_tracebacks=traces,
                modes=modes, effectiveness_verified=False, dr_unlocked=False), manifest, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--bridge-checkpoint', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    a, ma, aa = audit(args.folder, 'TASK_20260924_097', 'TASK_20260924_084', args.baseline_checkpoint, args.expected_commit)
    b, mb, ab = audit(args.folder, 'TASK_20260924_098', 'TASK_20260924_094', args.bridge_checkpoint, args.expected_commit)
    assert ma['runtime'] == mb['runtime'] and ma['environment'] == mb['environment']
    for key in ('motion_sha256', 'urdf_lf_sha256', 'feature_fingerprint'):
        assert ma['identity'][key] == mb['identity'][key]
    result = dict(cases=[a, b], same_environment_runtime=True, initial_comparison=initial_comparison(aa, ab))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(output=str(args.output), initial_comparison=result['initial_comparison'],
        survival={r['task']: {m: sum(e['survived'] for e in rows) for m, rows in r['modes'].items()} for r in (a, b)})))


if __name__ == '__main__':
    main()
