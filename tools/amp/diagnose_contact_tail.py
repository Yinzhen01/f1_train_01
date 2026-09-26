"""Audit the existing foot-force reward cap on real pre-reset100Hz samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.inspect_rollout import load_bundle, episode


FORCE_FIELDS = ('foot_sample_fraction_above700', 'foot_sample_fraction_above1100', 'p99_foot_force_N',
    'p999_foot_force_N', 'peak_foot_force_N', 'current_capped_reward_mean',
    'extra_uncapped_tail_reward_mean', 'actual_tail_reward_mean', 'actual_total_force_reward_mean')


def force_statistics(force, tail_scale=0.):
    force = np.asarray(force)
    if force.ndim != 2 or force.shape[1] != 2 or not np.isfinite(force).all() or tail_scale not in (0., -.0025):
        raise ValueError('Expected finite two-foot force norms and audited tail scale')
    if len(force) == 0:
        return {k: None for k in FORCE_FIELDS}
    excess = np.maximum(force-700., 0.)
    penalty = -.01*.01*np.clip(excess, 0., 400.).sum(-1)
    tail_cost = np.maximum(excess-400., 0.).sum(-1)
    tail = tail_scale*.01*tail_cost
    return dict(foot_sample_fraction_above700=float(np.mean(force > 700)),
        foot_sample_fraction_above1100=float(np.mean(force > 1100)),
        p99_foot_force_N=float(np.percentile(force, 99)),
        p999_foot_force_N=float(np.percentile(force, 99.9)), peak_foot_force_N=float(force.max()),
        current_capped_reward_mean=float(penalty.mean()),
        extra_uncapped_tail_reward_mean=float((-.01*.01*tail_cost).mean()),
        actual_tail_reward_mean=float(tail.mean()), actual_total_force_reward_mean=float((penalty+tail).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', action='append', required=True, help='label=bundle.pt')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    result = {}
    for entry in a.case:
        label, path = entry.split('=', 1); path = Path(path)
        if label in result: raise ValueError('Duplicate label')
        m, arrays = load_bundle(path)
        rewards = m['environment']['rewards']
        assert rewards['max_contact_force'] == 700 and rewards['scales']['feet_contact_forces'] == -.01
        assert m['duration_s'] == 60 and m['num_envs'] == 16
        tail_scale = rewards['scales'].get('contact_force_tail', 0.)
        record = dict(bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tail_scale=tail_scale, modes={})
        for mode in m['modes']:
            rows = []
            for index in range(16):
                d = episode(arrays, mode, index); mask = d['time'] >= 2.
                force = np.linalg.norm(d['foot_force'][mask], axis=-1)
                rows.append(dict(env=index, samples=len(force), failure=bool(d['failure'].any()),
                    observed_s=len(d['time'])*.01, **force_statistics(force, tail_scale)))
            selected = {k: [r[k] for r in rows if r[k] is not None] for k in FORCE_FIELDS}
            record['modes'][mode] = dict(episodes=rows,
                equal_initial_mean={k: float(np.mean(v)) if v else None for k, v in selected.items()},
                metric_n={k: len(v) for k, v in selected.items()})
        result[label] = record
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(cases=result, source_reward='X1DHStandEnv._reward_feet_contact_forces', threshold_N=700, excess_cap_N=400,
        limitation='Only100Hz net foot forces, not1kHz impact peaks/impulse. Failed prefixes retained; no post2s samples gives null with explicit n, never zero. current_capped names original capped reward only; actual_total includes configured tail. extra_uncapped is hypothetical full-slope tail.')
    with (a.output/'contact_tail_report.json').open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2)
    print(json.dumps({k: {m: v['equal_initial_mean'] for m, v in r['modes'].items()} for k, r in result.items()}))


if __name__ == '__main__': main()
