"""Render an explicit real recorded prefix/window, retaining original timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.inspect_rollout import load_bundle, episode, render_episode
from humanoid.amp.scaled_experiment import ScaledExperiment


def select_interval(data, start, end):
    if not np.isfinite([start, end]).all() or not 0. <= start < end:
        raise ValueError('Invalid recorded interval')
    keep = (data['time'] >= start) & (data['time'] <= end)
    if keep.sum() < 2:
        raise ValueError('No usable recorded interval')
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in data.items()}


def main():
    p = argparse.ArgumentParser()
    for key in ('bundle', 'config', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--mode', choices=('standing', 'reference'), required=True)
    p.add_argument('--env-index', type=int, required=True)
    p.add_argument('--start', type=float, default=0.)
    p.add_argument('--end', type=float, default=60.)
    a = p.parse_args()
    m, arrays = load_bundle(a.bundle); e = ScaledExperiment(ROOT, a.config)
    if m['identity'] != e.identity() or not 0 <= a.env_index < m['num_envs']:
        raise ValueError('Wrong source identity/index')
    d = episode(arrays, a.mode, a.env_index)
    selected = select_interval(d, a.start, a.end)
    a.output.mkdir(parents=True, exist_ok=False)
    result = render_episode(selected, m, Path(e.kinematics.path), a.output)
    result.update(bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
        checkpoint_sha256=m['checkpoint_sha256'], mode=a.mode, env=a.env_index,
        requested_interval_s=[a.start, a.end], observed_interval_s=selected['time'][[0,-1]].tolist(),
        failure_in_selection=bool(selected['failure'].any()),
        original_prefix_failed=bool(d['initial']['failure'] or d['failure'].any()))
    with (a.output/'render_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__': main()
