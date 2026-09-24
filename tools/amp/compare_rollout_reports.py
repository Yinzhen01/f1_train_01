"""Compare hash-bound policy reports without discarding failed initial states."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = {
    'vx': ('vx_mean',),
    'slip': ('contact_sole_proxy_speed_mean_m_s',),
    'acceleration': ('joint_accel_fd_rms_rad_s2',),
    'native_acceleration': ('joint_accel_native_rms_rad_s2',),
    'heading': ('heading_rms_to_world_x_deg',),
    'hf_velocity_power': ('joint_velocity_spectrum', 'power_fraction_above_cutoff'),
    'nearest_demo': ('demo_nearest_window_rms_zscore', 'mean'),
    'demo_joint_velocity_mse': ('demo_nearest_window_rms_zscore', 'nearest_window_group_mse', 'joint_velocity'),
    'frozen_style': ('frozen_auditor_style', 'normalized_reward_mean'),
    'action_delta': ('action_delta_rms',),
    'torque_delta': ('torque_delta_rms_nm',),
}


def metric(row, path):
    value = row.get('post_2s')
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return None if value is None else float(value)


def summarize(rows, survivors_only=False):
    selected = [r for r in rows if r['survived']] if survivors_only else rows
    out = {}
    for name, path in METRICS.items():
        values = [metric(r, path) for r in selected]
        values = [v for v in values if v is not None]
        out[name] = dict(n=len(values), mean=None if not values else float(np.mean(values)),
            min=None if not values else min(values), max=None if not values else max(values))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', action='append', required=True, help='LABEL=report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports, result = {}, {}
    auditor = None
    for case in args.case:
        label, path = case.split('=', 1)
        if label in reports:
            raise ValueError('Duplicate label')
        path = Path(path)
        report = json.loads(path.read_text(encoding='utf-8'))
        digest = (report.get('frozen_auditor') or {}).get('sha256')
        if not digest or auditor not in (None, digest):
            raise ValueError('All comparisons require the same frozen auditor')
        auditor = digest
        reports[label] = report
        result[label] = dict(report_path=str(path.resolve()), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint_sha256=report['checkpoint_sha256'], identity=report['identity'], modes={})
        for mode, record in report['modes'].items():
            rows = record['episodes']
            result[label]['modes'][mode] = dict(n=len(rows), survived=sum(r['survived'] for r in rows),
                failed_envs=[r['env'] for r in rows if not r['survived']],
                observed_duration_range_s=[min(r['observed_s'] for r in rows), max(r['observed_s'] for r in rows)],
                all_observed=summarize(rows), survivors_only=summarize(rows, True))
    args.output.mkdir(parents=True, exist_ok=False)
    output = dict(frozen_auditor_sha256=auditor, cases=result,
        note='Equal weight per initial condition. Post-2s metrics with missing/short episodes expose n. Failed initial states are retained; survivors-only is secondary, not the primary ranking. Frozen style score is not a probability. No automatic acceptance or DR unlock.')
    (args.output/'comparison.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    views = [('vx', 'Body forward speed (m/s)'), ('slip', 'Ground-contact sole proxy speed (m/s)'),
             ('acceleration', 'Joint q finite-difference acceleration RMS (rad/s²)'),
             ('heading', 'Heading RMS to world X (deg)'), ('nearest_demo', 'Nearest demonstration window (RMS zscore)'),
             ('frozen_style', 'Same frozen D style score (not probability)')]
    for mode in ('standing', 'reference'):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11))
        labels = list(reports)
        for axis, (name, title) in zip(axes.flat, views):
            for i, label in enumerate(labels):
                rows = reports[label]['modes'][mode]['episodes']
                for row in rows:
                    value = metric(row, METRICS[name])
                    if value is not None:
                        axis.scatter(i+(row['env']-(len(rows)-1)/2)*.025, value,
                                     color='tab:blue' if row['survived'] else 'tab:red', s=14, alpha=.65)
                average = result[label]['modes'][mode]['all_observed'][name]['mean']
                if average is not None:
                    axis.plot([i-.3, i+.3], [average, average], color='black', lw=2)
            axis.set_xticks(range(len(labels)))
            axis.set_xticklabels(labels, rotation=25, ha='right')
            axis.set_title(title); axis.grid(axis='y', alpha=.25)
            if name == 'vx':
                axis.axhline(.45, color='gray', ls='--')
        counts = ['%s %s/%s' % (label, result[label]['modes'][mode]['survived'], result[label]['modes'][mode]['n']) for label in labels]
        count_lines = '\n'.join(', '.join(counts[i:i+4]) for i in range(0, len(counts), 4))
        fig.suptitle(mode+' initial state | survived 20s\n'+count_lines+'\nEach dot is one initial state; red=failed, black=observed mean; post-2s metrics', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(args.output/(mode+'_comparison.png'), dpi=150); plt.close(fig)
    print(json.dumps(dict(output=str(args.output), survival={label: {m: v['survived'] for m, v in r['modes'].items()} for label, r in result.items()})))


if __name__ == '__main__':
    main()
