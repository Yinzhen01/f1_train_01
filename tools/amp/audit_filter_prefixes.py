"""Paired pre-failure diagnostic, never a replacement for full-survival gates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.verify_long_pair import initial_comparison
from tools.amp.analyze_jitter_events import backward_delta, high_frequency
from tools.amp.jitter_audit import validate_substep_arrays
from humanoid.amp.jitter import SOURCE_SHA


def metrics(d, end):
    mask = (d['time'] >= 2.) & (d['time'] <= end)
    if mask.sum() < 100: return None
    values = dict(vx=float(d['base_lin_vel'][mask, 0].mean()), samples=int(mask.sum()))
    for name, key, scale in (('action_delta', 'action', 1.), ('torque_delta', 'torque', 1.), ('acceleration', 'dof_vel', .01)):
        v = backward_delta(d[key], d['initial'][key])[mask]/scale
        values[name+'_rms'] = float(np.sqrt(np.mean(v*v)))
    values['action_high_power'] = float(np.mean(high_frequency(d['action'][mask])['high_band_power']))
    return values


def main():
    p = argparse.ArgumentParser()
    for key in ('source', 'candidate', 'output'): p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    sm, sa = load_bundle(a.source); cm, ca = load_bundle(a.candidate)
    if sm['checkpoint_sha256'] != SOURCE_SHA or not initial_comparison(sa, ca)['all_captured_fields_exact_equal']:
        raise ValueError('Wrong baseline or initial states')
    validate_substep_arrays(cm, ca)
    rows = []
    for mode in cm['modes']:
        for index in range(cm['num_envs']):
            s, c = episode(sa, mode, index), episode(ca, mode, index)
            failed = bool(c['initial']['failure'] or c['failure'].any())
            end = min(float(s['time'][-1]), float(c['time'][-1])-(1. if failed else 0.)) if len(c['time']) else 0.
            rows.append(dict(mode=mode, env=index, failed=failed, paired_end_s=end,
                             source=metrics(s, end), candidate=metrics(c, end)))
    means = {}
    for label in ('source', 'candidate'):
        usable = [r[label] for r in rows if r['source'] is not None and r['candidate'] is not None]
        means[label] = {key: float(np.mean([r[key] for r in usable])) for key in usable[0] if key != 'samples'} if usable else None
    result = dict(group=cm['identity']['experiment'], means=means, rows=rows,
                  excluded_short_prefixes=sum(r['candidate'] is None for r in rows),
                  scope='Same physical-time prefix on both sides, starts2s, excludes final1s only for failed candidate; >=100 samples. Diagnostic only: failures remain failures, not acceptance.')
    with a.output.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(group=result['group'], means=means, excluded=result['excluded_short_prefixes'])))


if __name__ == '__main__': main()
