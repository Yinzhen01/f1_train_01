"""Render a hash-identified captured prefix without repeating all-env statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.inspect_rollout import load_bundle, episode, render_episode
from humanoid.amp.scaled_experiment import ScaledExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('standing', 'reference'), action='append')
    parser.add_argument('--env-index', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest, arrays = load_bundle(args.bundle)
    experiment = ScaledExperiment(ROOT, args.config)
    assert manifest['identity'] == experiment.identity()
    assert manifest['dof_names'] == list(experiment.spec.joint_names)
    if not 0 <= args.env_index < manifest['num_envs']:
        raise ValueError('Invalid initial condition')
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        checkpoint_sha256=manifest['checkpoint_sha256'], env_index=args.env_index,
        render_script_sha256=hashlib.sha256((ROOT/'tools/amp/inspect_rollout.py').read_bytes()).hexdigest(), modes={})
    for mode in (args.mode or manifest['modes']):
        if mode not in manifest['modes']: raise ValueError('Mode absent from bundle')
        data = episode(arrays, mode, args.env_index)
        folder = args.output/mode; folder.mkdir()
        result['modes'][mode] = dict(failure=bool(data['initial']['failure'] or data['failure'].any()),
            observed_s=len(data['time'])*.01, render=render_episode(data, manifest, Path(experiment.kinematics.path), folder))
    with (args.output/'render_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__': main()
