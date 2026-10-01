"""Audit measured substeps against endpoint differences; no simulated replay.

Foot net-force thresholds classify measurements, not gait phase or contact pairs.
Paired prefixes include failures and never certify survival or motion quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.analyze_jitter_events import backward_delta
from tools.amp.jitter_audit import validate_substep_arrays
from tools.amp.verify_long_pair import initial_comparison


def measured_summary(endpoint, squared, peaks, force_end, force_peak, mask):
    """All comparisons use exactly the same control intervals and joints."""
    if not np.any(mask):
        return None
    endpoint = np.asarray(endpoint, dtype=np.float64)[mask]
    squared = np.asarray(squared, dtype=np.float64)[mask]
    peaks = np.asarray(peaks, dtype=np.float64)[mask]
    force_end = np.maximum(np.asarray(force_end, dtype=np.float64)[mask], 0.)
    force_peak = np.asarray(force_peak, dtype=np.float64)[mask]
    end_energy = float(np.mean(endpoint**2))
    sub_energy = float(squared.mean())
    # E[a^2] >= E[a]^2. Endpoint acceleration is the substep mean.
    if np.any(endpoint**2 > squared + np.maximum(.1, squared*.0001)):
        raise ValueError('Substep/endpoint timing or averaging mismatch')
    bursts = force_peak > 1100.
    quiet = force_peak.max(axis=1) <= 1100.
    return dict(samples=int(mask.sum()), endpoint_accel_rms=float(np.sqrt(end_energy)),
        substep_accel_rms=float(np.sqrt(sub_energy)),
        endpoint_to_substep_energy_ratio=end_energy/sub_energy if sub_energy else None,
        substep_peak_accel=float(peaks.max()),
        substep_rms_per_joint=np.sqrt(squared.mean(axis=0)).tolist(),
        intervals_without_1100n_burst_fraction=float(quiet.mean()),
        no_burst_substep_accel_rms=float(np.sqrt(squared[quiet].mean())) if quiet.any() else None,
        burst_foot_intervals=int(bursts.sum()),
        burst_missed_by_endpoint=int((bursts & (force_end <= 1100.)).sum()),
        force_peak_n=float(force_peak.max()),
        force_peak_p99_n=float(np.percentile(force_peak, 99)),
        endpoint_force_peak_n=float(force_end.max()))


def failure_state(data, start, end):
    mask = (data['time'] >= start) & (data['time'] <= end)
    if not mask.any():
        return None
    angles = Rotation.from_quat(data['root_state'][mask, 3:7]).as_euler('xyz', degrees=True)
    return dict(start_s=float(data['time'][mask][0]), end_s=float(data['time'][mask][-1]),
        vx_mean=float(data['base_lin_vel'][mask, 0].mean()),
        vx_last=float(data['base_lin_vel'][mask, 0][-1]),
        base_height_min_m=float(data['root_state'][mask, 2].min()),
        pitch_min_deg=float(angles[:, 1].min()), pitch_max_deg=float(angles[:, 1].max()),
        roll_abs_max_deg=float(np.abs(angles[:, 0]).max()))


def plot_failure(data, arrays, mode, index, output, label):
    import matplotlib.pyplot as plt
    n = len(data['time']); t = data['time']
    angles = Rotation.from_quat(data['root_state'][:, 3:7]).as_euler('xyz', degrees=True)
    end_acc = backward_delta(data['dof_vel'], data['initial']['dof_vel'])/.01
    measured = arrays[mode+'_physics_accel_squared'][:n, index]
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(t, data['base_lin_vel'][:, 0]); axes[0].axhline(0., color='gray'); axes[0].set_ylabel('Body vx m/s')
    axes[1].plot(t, data['root_state'][:, 2]); axes[1].set_ylabel('Root height m')
    for i, name in enumerate(('roll', 'pitch')):
        axes[2].plot(t, angles[:, i], label=name)
    axes[2].set_ylabel('Euler deg'); axes[2].legend()
    axes[3].plot(t, np.sqrt((end_acc**2).mean(-1)), label='100Hz endpoint')
    axes[3].plot(t, np.sqrt(measured.mean(-1)), alpha=.65, label='measured 1kHz RMS')
    axes[3].set_ylabel('Acceleration rad/s2'); axes[3].legend()
    for i, name in enumerate(('L', 'R')):
        axes[4].plot(t, arrays[mode+'_physics_foot_force_peak'][:n, index, i], label=name)
    axes[4].set_ylabel('Net foot Fz peak N'); axes[4].legend()
    for ax in axes:
        ax.set_xlim(max(.05, t[-1]-5.), t[-1]); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Original physical time s (stops at physical failure)')
    fig.suptitle(label+' | recorded states; no smoothing or simulated continuation')
    fig.tight_layout(); fig.savefig(output/(label+'.png'), dpi=125); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    for key in ('control', 'smooth', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    a.output.mkdir(parents=True, exist_ok=False)
    cases = {}
    for name in ('control', 'smooth'):
        manifest, arrays = load_bundle(getattr(a, name))
        validate_substep_arrays(manifest, arrays)
        cases[name] = (manifest, arrays)
    if not initial_comparison(cases['control'][1], cases['smooth'][1])['all_captured_fields_exact_equal']:
        raise ValueError('Not matched initial states')
    result = dict(cases={}, paired_prefixes=True, effectiveness_verified=False,
        limitations='Compared new groups only; baseline has no 1kHz telemetry. Prefix length per initial condition '
        'is capped at the earlier failure of either group. Failures are retained, not counted as quiet successes. '
        '1100N is a diagnostic net-force threshold, not a gait phase/contact-pair label. '
        'No raw substep time series exists, only measured per-control-interval mean squares/maxima.')
    for name, (manifest, arrays) in cases.items():
        rows, failures = [], []
        other = cases['smooth' if name == 'control' else 'control'][1]
        for mode in manifest['modes']:
            for index in range(manifest['num_envs']):
                data = episode(arrays, mode, index)
                paired = episode(other, mode, index)
                n = min(len(data['time']), len(paired['time']))
                endpoint = backward_delta(data['dof_vel'], data['initial']['dof_vel'])/.01
                mask = arrays[mode+'_physics_valid'][:n, index] & (data['time'][:n] >= 2.)
                summary = measured_summary(endpoint[:n], arrays[mode+'_physics_accel_squared'][:n, index],
                    arrays[mode+'_physics_accel_peak'][:n, index], data['foot_force'][:n, :, 2],
                    arrays[mode+'_physics_foot_force_peak'][:n, index], mask)
                rows.append(dict(mode=mode, env=index, common_end_s=n*.01, metrics=summary))
                if data['initial']['failure'] or data['failure'].any():
                    end = float(data['time'][-1]) if len(data['time']) else 0.
                    failures.append(dict(mode=mode, env=index, failure_s=end,
                        pre_failure=failure_state(data, max(.05, end-3.), max(.05, end-1.)),
                        terminal_second=failure_state(data, max(.05, end-1.), end)))
                    if len(data['time']):
                        plot_failure(data, arrays, mode, index, a.output, '%s_%s_env%d' % (name, mode, index))
        valid = [r['metrics'] for r in rows if r['metrics'] is not None]
        means = {key:float(np.mean([m[key] for m in valid if m[key] is not None]))
                 for key in ('endpoint_accel_rms', 'substep_accel_rms',
                     'endpoint_to_substep_energy_ratio', 'no_burst_substep_accel_rms',
                     'intervals_without_1100n_burst_fraction', 'force_peak_p99_n')}
        bursts = sum(m['burst_foot_intervals'] for m in valid)
        missed = sum(m['burst_missed_by_endpoint'] for m in valid)
        means.update(burst_foot_intervals=bursts, burst_missed_by_endpoint=missed,
            missed_burst_fraction=missed/bursts if bursts else None)
        result['cases'][name] = dict(rows=rows, failures=failures, equal_prefix_mean=means,
            bundle_sha256=hashlib.sha256(getattr(a, name).read_bytes()).hexdigest())
        print(json.dumps(dict(case=name, means=means, failures=failures)), flush=True)
    with (a.output/'substep_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
