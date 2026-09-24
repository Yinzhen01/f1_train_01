"""Read-only 60s audit of exact completed1500 policies; never trains/resumes PPO."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.evaluate_mounted import locate_checkpoint

SOURCES = {
    'TASK_20260924_084': dict(group='refine_smooth', experiment='f1_amp_walk02_refine_smooth',
        sha256='1d5997ea04250ea0a61d27fa19f1110c7bedde2821a03542986911c69f14546e'),
    'TASK_20260924_094': dict(group='signal_bridge', experiment='f1_amp_walk02_signal_bridge',
        sha256='d5f55bb7c25f6ff8d45588372db8d3feaeb3d8b92d6baaa600260f02b64f4dc8'),
}


def audit_source(task_id, digest):
    if task_id not in SOURCES or SOURCES[task_id]['sha256'] != digest:
        raise ValueError('Only approved exact source1500 checkpoints may be evaluated')
    return dict(SOURCES[task_id])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-task', choices=tuple(SOURCES), required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    source = audit_source(args.source_task, args.checkpoint_sha256)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip()
    if actual != args.expected_commit:
        raise ValueError('Unexpected audit checkout')
    checkpoint = locate_checkpoint((ROOT, Path('/workspace'), Path('/personal')), source['sha256'],
                                   pattern='model_8801500*.pt')
    folder = ROOT/'logs/f1_amp_long_audit/exported_data'/(args.source_task+'_seed5')
    output = folder/'model_9961500.pt'
    print('[amp-long-audit-start] '+json.dumps(dict(source_task=args.source_task,
        source=source, completed_updates=1500, duration=60, num_envs=16, seed=5,
        code_commit=actual, no_training=True, dr_unlocked=False)), flush=True)
    subprocess.run([sys.executable, str(ROOT/'humanoid/scripts/record_amp_policy.py'),
        '--checkpoint-file', str(checkpoint), '--checkpoint-sha256', source['sha256'],
        '--expected-commit', actual, '--experiment', source['group'], '--output', str(output),
        '--extended-validation', '--duration', '60', '--num_envs', '16', '--seed', '5',
        '--headless', '--task', source['experiment'], '--sim_device', 'cuda:0',
        '--rl_device', 'cuda:0'], check=True, cwd=str(ROOT), timeout=900)
    print('[amp-long-audit-complete] '+json.dumps(dict(source_task=args.source_task,
        artifact=output.name, no_training=True, dr_unlocked=False)), flush=True)


if __name__ == '__main__':
    main()
