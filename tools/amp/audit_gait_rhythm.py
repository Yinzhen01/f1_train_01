"""Phase-free gait rhythm audit of immutable demonstration/recorded policy states.

Foot fore-aft motion is measured in the root frame, avoiding mirrored joint-axis
signs. Contact-force threshold flicker is not counted as an additional stride.
This is offline validation only: no phase clock/labels are supplied to training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, csd, find_peaks, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.scaled_experiment import ScaledExperiment
from tools.amp.inspect_rollout import load_bundle, episode, features_from_episode


def rhythm_metrics(foot_x, dt=.01):
    x = np.asarray(foot_x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() or dt != .01:
        raise ValueError('Expected finite100Hz left/right body-frame foot X')
    if len(x) < 400:
        return dict(valid=False, reason='Fewer than4s of observed data')
    smooth = sosfiltfilt(butter(4, 3., fs=100, output='sos'), x, axis=0)
    frequency, power = welch(x, fs=100, nperseg=min(600, len(x)), axis=0)
    band = (frequency >= .35) & (frequency <= 2.5)
    if np.min(np.ptp(smooth[50:-50], axis=0)) < .02:
        return dict(valid=False, reason='Less than2cm fore-aft movement in at least one foot')
    pooled_peak = np.flatnonzero(band)[np.argmax(power[band].sum(axis=1))]
    _, cross_power = csd(x[:, 0], x[:, 1], fs=100, nperseg=min(600, len(x)))
    phase = float(np.angle(cross_power[pooled_peak], deg=True))
    feet, events = [], []
    for side in range(2):
        prominence = max(.02, .3*np.ptp(smooth[50:-50, side]))
        peaks, _ = find_peaks(smooth[:, side], distance=40, prominence=prominence)
        peaks = peaks[(peaks >= 50) & (peaks < len(x)-50)]
        intervals = np.diff(peaks)*dt
        feet.append(dict(peak_times_s=(peaks*dt).tolist(), complete_intervals=len(intervals),
            stride_hz=None if not len(intervals) else float(1/np.median(intervals)),
            interval_cv=None if not len(intervals) else float(intervals.std()/intervals.mean()),
            spectral_peak_hz=float(frequency[band][np.argmax(power[band, side])]),
            filtered_range_m=float(np.ptp(smooth[50:-50, side]))))
        events.extend((int(p), side) for p in peaks)
    events.sort()
    pairs = list(zip(events[:-1], events[1:]))
    return dict(valid=True, observed_s=len(x)*dt, feet=feet,
        alternating_foremost_event_fraction=None if not pairs else float(np.mean([a[1] != b[1] for a, b in pairs])),
        near_simultaneous_event_fraction=None if not pairs else float(np.mean([(b[0]-a[0])*dt < .1 for a, b in pairs])),
        left_right_phase_deg=phase, error_from_antiphase_deg=180-abs(phase),
        spectral_bin_width_hz=float(frequency[1]-frequency[0]),
        pooled_spectral_peak_hz=float(frequency[pooled_peak]))


def summarize(rows):
    valid = [r['rhythm'] for r in rows if r['rhythm']['valid']]
    out = dict(n=len(rows), valid_n=len(valid))
    for key in ('alternating_foremost_event_fraction', 'near_simultaneous_event_fraction', 'error_from_antiphase_deg'):
        values = [r[key] for r in valid if r[key] is not None]
        out[key] = dict(n=len(values), mean=None if not values else float(np.mean(values)),
                        minimum=None if not values else min(values), maximum=None if not values else max(values))
    out['feet'] = []
    for side in range(2):
        values = [r['feet'][side]['stride_hz'] for r in valid if r['feet'][side]['stride_hz'] is not None]
        out['feet'].append(dict(n=len(values), stride_hz_mean=None if not values else float(np.mean(values)),
                               stride_hz_range=None if not values else [min(values), max(values)]))
    return out


def feet_from_features(features, spec):
    names = ('left_ankle_roll_link', 'right_ankle_roll_link')
    columns = [27+3*spec.body_names.index(name) for name in names]
    return np.asarray(features)[:, columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', action='append', required=True, help='LABEL=CONFIG_JSON,BUNDLE_PT')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(schema_version=1, cases={}, source_motion_sha256=None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method='Root-relative foot link X.4th-order3Hz offline zero-phase filter; peaks prominence=max(.02m,.3*range), distance.4s, exclude.5s endpoints. Same-foot peak intervals estimate stride frequency; alternating left/right foremost events do not assert ground-contact order. Welch raw data max6s windows,.35-2.5Hz; bin resolution reported, not exact frequency equality. All failed prefixes retained, post2s policy vs full6s demonstration; no training phase labels and no automatic DR unlock.')
    plots = {}
    for case in args.case:
        label, rest = case.split('=', 1)
        config, bundle = (Path(p) for p in rest.split(',', 1))
        if label in result['cases']:
            raise ValueError('Duplicate label')
        experiment = ScaledExperiment(ROOT, config)
        manifest, arrays = load_bundle(bundle)
        if manifest['identity'] != experiment.identity():
            raise ValueError('Bundle/config identity mismatch')
        if result['source_motion_sha256'] not in (None, experiment.cfg['motion_sha256']):
            raise ValueError('Different demonstrations')
        result['source_motion_sha256'] = experiment.cfg['motion_sha256']
        demo = feet_from_features(experiment.clip.features.numpy(), experiment.spec)
        result['demonstration'] = rhythm_metrics(demo)
        plots['demonstration'] = (np.arange(len(demo))*.01, demo)
        modes = {}
        for mode in manifest['modes']:
            rows = []
            for env in range(manifest['num_envs']):
                data = episode(arrays, mode, env)
                x = feet_from_features(features_from_episode(data, experiment).numpy(), experiment.spec)
                active = data['time'] >= 2.
                rows.append(dict(env=env, observed_s=len(data['time'])*.01,
                    failed=bool(data['initial']['failure'] or data['failure'].any()),
                    rhythm=rhythm_metrics(x[active])))
                if env == 0:
                    plots[label+' '+mode+' env0'] = (data['time'][active]-2., x[active])
            modes[mode] = dict(episodes=rows, summary=summarize(rows))
        result['cases'][label] = dict(config_path=str(config.resolve()), bundle_path=str(bundle.resolve()),
            bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
            checkpoint_sha256=manifest['checkpoint_sha256'], duration_s=manifest['duration_s'], modes=modes)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'rhythm_report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(plots), 1, figsize=(13, 2.4*len(plots)), squeeze=False)
    for ax, (name, (time, x)) in zip(axes.flat, plots.items()):
        keep = time <= 6.
        ax.plot(time[keep], x[keep, 0], label='Left'); ax.plot(time[keep], x[keep, 1], label='Right')
        ax.set_title(name+' | body-frame foot fore-aft motion (not force contact)')
        ax.set_ylabel('m'); ax.grid(alpha=.2); ax.legend(loc='upper right'); ax.set_xlim(0, 6)
    axes[-1, 0].set_xlabel('Seconds from demo start / policy2s (no phase alignment)')
    fig.tight_layout(); fig.savefig(args.output/'foot_rhythm.png', dpi=140); plt.close(fig)
    print(json.dumps(dict(output=str(args.output), demo=result['demonstration'],
        cases={k: {m: v['summary'] for m, v in c['modes'].items()} for k, c in result['cases'].items()})))


if __name__ == '__main__':
    main()
