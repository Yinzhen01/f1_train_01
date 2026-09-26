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
        record = dict(bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), modes={})
        for mode in m['modes']:
            rows = []
            for index in range(16):
                d = episode(arrays, mode, index); mask = d['time'] >= 2.
                force = np.linalg.norm(d['foot_force'][mask], axis=-1)
                if not len(force): raise ValueError('No post2s samples')
                excess = np.maximum(force-700., 0.)
                penalty = -.01*.01*np.clip(excess, 0., 400.).sum(-1)
                tail = -.01*.01*np.maximum(excess-400., 0.).sum(-1)
                rows.append(dict(env=index, samples=len(force), failure=bool(d['failure'].any()),
                    observed_s=len(d['time'])*.01,
                    foot_sample_fraction_above700=float(np.mean(force > 700)),
                    foot_sample_fraction_above1100=float(np.mean(force > 1100)),
                    p99_foot_force_N=float(np.percentile(force, 99)),
                    p999_foot_force_N=float(np.percentile(force, 99.9)),
                    peak_foot_force_N=float(force.max()),
                    current_capped_reward_mean=float(penalty.mean()),
                    extra_uncapped_tail_reward_mean=float(tail.mean())))
            fields = ('foot_sample_fraction_above700', 'foot_sample_fraction_above1100', 'p99_foot_force_N',
                      'p999_foot_force_N', 'current_capped_reward_mean', 'extra_uncapped_tail_reward_mean')
            record['modes'][mode] = dict(episodes=rows, equal_initial_mean={k: float(np.mean([r[k] for r in rows])) for k in fields})
        result[label] = record
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(cases=result, source_reward='X1DHStandEnv._reward_feet_contact_forces', threshold_N=700, excess_cap_N=400,
        limitation='Only100Hz net rigid-body foot forces, not true1kHz impact peak/impulse or contact-pair provenance. Failed prefixes retained; durations differ. Uncapped term is offline magnitude only, not an approved training change.')
    with (a.output/'contact_tail_report.json').open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2)
    print(json.dumps({k: {m: v['equal_initial_mean'] for m, v in r['modes'].items()} for k, r in result.items()}))


if __name__ == '__main__': main()
