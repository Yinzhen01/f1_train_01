"""Matched-time physical diagnostics for the completed frame intervention.

No new training, survivor filtering, or failure-cause inference. Event times
come from world2250; controls are measured at those same absolute times.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.inspect_rollout import load_bundle, episode, rms
from tools.amp.diagnose_progress_frame import series


def metrics(data, start, end):
    """Require every control tick of [start,end); never silently shorten a bin."""
    mask = (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
    expected = int(round((end-start)*100))
    if expected < 20 or int(mask.sum()) != expected:
        return None
    root = data['root_state'][mask]
    angles = np.rad2deg(Rotation.from_quat(root[:, 3:7]).as_euler('xyz'))
    force = data['foot_force'][mask]
    contacts = force[:, :, 2] > 5.
    reward = series({k: v[mask] for k, v in data.items() if k != 'initial'})
    return dict(samples=expected,
        body_vx=float(data['base_lin_vel'][mask, 0].mean()),
        world_vx=float(root[:, 7].mean()), world_vy_abs=float(np.abs(root[:, 8]).mean()),
        yaw_rms_deg=rms(angles[:, 2]), roll_rms_deg=rms(angles[:, 0]),
        pitch_mean_deg=float(angles[:, 1].mean()), pitch_rms_deg=rms(angles[:, 1]),
        root_height_mean=float(root[:, 2].mean()), root_height_min=float(root[:, 2].min()),
        foot_force_peak_N=float(np.linalg.norm(force, axis=-1).max()),
        double_foot_force_fraction=float(contacts.all(-1).mean()),
        no_foot_force_fraction=float((~contacts.any(-1)).mean()),
        qdd_fd_rms=rms(np.diff(data['dof_pos'][mask], n=2, axis=0)/.01**2),
        action_delta_rms=rms(np.diff(data['action'][mask], axis=0)),
        torque_delta_rms=rms(np.diff(data['torque'][mask], axis=0)),
        joint_rom_mean=float(np.ptp(data['dof_pos'][mask], axis=0).mean()),
        body_progress=float(reward['body_progress'].mean()),
        world_progress=float(reward['world_progress'].mean()))


def summarize(rows):
    selected = [r for r in rows if r is not None]
    if not selected:
        return dict(n=0, mean={})
    return dict(n=len(selected), mean={k: float(np.mean([r[k] for r in selected]))
                                      for k in selected[0] if k != 'samples'})


def matched_initials(bundles):
    source_manifest, source_arrays = bundles['source']
    keys = sorted(k for k in source_arrays if '_initial_' in k)
    if len(keys) != 24:
        raise ValueError('Missing initial-state fields')
    for label, (manifest, arrays) in bundles.items():
        if (manifest['duration_s'] != 60 or manifest['num_envs'] != 16 or
                manifest['mode_seeds'] != source_manifest['mode_seeds'] or
                manifest['dof_names'] != source_manifest['dof_names'] or
                manifest['runtime'] != source_manifest['runtime']):
            raise ValueError('Unmatched protocol, joint order or physics: '+label)
        if any(not np.array_equal(source_arrays[k], arrays[k]) for k in keys):
            raise ValueError('Unmatched recorded initial state: '+label)
    return keys


def main():
    p = argparse.ArgumentParser()
    for label in ('source', 'body', 'world'):
        p.add_argument('--'+label, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    bundles = {label: load_bundle(getattr(args, label)) for label in ('source', 'body', 'world')}
    keys = matched_initials(bundles)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, initial_fields_matched=keys, sources={}, modes={},
        limitation='Fixed2-4s retains every initial. Failed-event bins use identical absolute times, '
        'with missing controls reported. Fz>5N is a diagnostic only, not a reward change. '
        'All values are recorded-state descriptions, not a unique cause of failure. '
        'No non-foot contact forces were recorded; termination-contact cause is unobservable.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label, (manifest, _) in bundles.items():
        expected = 'body' if label in ('source', 'body') else 'world'
        if manifest['environment']['rewards'].get('progress_velocity_frame', 'body') != expected:
            raise ValueError('Unexpected actual reward frame')
        result['sources'][label] = dict(path=str(getattr(args, label).resolve()),
            bundle_sha256=hashlib.sha256(getattr(args, label).read_bytes()).hexdigest(),
            checkpoint_sha256=manifest['checkpoint_sha256'], actual_velocity_frame=expected)
    for mode in ('standing', 'reference'):
        data = {label: [episode(arrays, mode, i) for i in range(16)]
                for label, (_, arrays) in bundles.items()}
        early = {label: [metrics(d, 2., 4.) for d in episodes] for label, episodes in data.items()}
        if any(r is None for rows in early.values() for r in rows):
            raise ValueError('Fixed early comparison lost an initial state')
        events = []
        for index, world in enumerate(data['world']):
            if not (world['initial']['failure'] or world['failure'].any()):
                continue
            end = float(world['time'][-1])
            windows = {'pre_4to2s': (end-4, end-2), 'last_2s': (end-2+.01, end+.01)}
            event = dict(env=index, failure_s=end, terminal_pitch_deg=float(np.rad2deg(
                Rotation.from_quat(world['root_state'][-1, 3:7]).as_euler('xyz')[1])),
                terminal_body_vx=float(world['base_lin_vel'][-1, 0]), windows={})
            for name, (start, stop) in windows.items():
                event['windows'][name] = dict(start=start, stop_exclusive=stop,
                    cases={label: metrics(episodes[index], start, stop) for label, episodes in data.items()})
            events.append(event)
        event_summary = {name: {label: summarize([e['windows'][name]['cases'][label] for e in events])
                               for label in data} for name in ('pre_4to2s', 'last_2s')}
        result['modes'][mode] = dict(fixed_2to4s={k: dict(**summarize(v), episodes=v) for k, v in early.items()},
            world_failed_events=events, event_summary=event_summary,
            terminal_negative_body_vx_count=sum(e['terminal_body_vx'] < 0 for e in events),
            terminal_negative_pitch_count=sum(e['terminal_pitch_deg'] < 0 for e in events))
        # Fixed env0, including its failed prefix; never select the best rollout.
        end = float(data['world'][0]['time'][-1])
        fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
        for label in data:
            d = data[label][0]
            mask = (d['time'] >= max(0., end-5)) & (d['time'] <= end+.001)
            rpy = np.rad2deg(Rotation.from_quat(d['root_state'][mask, 3:7]).as_euler('xyz'))
            values = (d['base_lin_vel'][mask, 0], rpy[:, 1], rpy[:, 2], d['root_state'][mask, 2],
                      np.linalg.norm(d['foot_force'][mask], axis=-1).max(-1))
            for axis, value in zip(axes, values):
                axis.plot(d['time'][mask], value, label=label, alpha=.85, lw=1.)
        for axis, label in zip(axes, ('body vx m/s', 'pitch deg', 'yaw deg', 'root height m', 'max foot force N')):
            axis.set_ylabel(label); axis.grid(alpha=.25); axis.axvline(end, color='red', ls=':')
        axes[0].legend(); axes[-1].set_xlabel('Absolute simulation time s')
        fig.suptitle(mode+' env0: same time window, world first failure at %.2fs' % end)
        fig.tight_layout(); fig.savefig(args.output/(mode+'_matched_failure_env0.png'), dpi=140); plt.close(fig)
    with (args.output/'matched_failure_report.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({mode: {k: v for k, v in r.items() if k not in ('world_failed_events', 'fixed_2to4s')}
                      for mode, r in result['modes'].items()}))


if __name__ == '__main__':
    main()
