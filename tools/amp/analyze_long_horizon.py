"""Describe long-run drift and failed prefixes without claiming failure causality."""
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
from tools.amp.inspect_rollout import load_bundle, episode, foot_geometry, rms
from humanoid.amp.recovery import foot_collision_vertices
from humanoid.amp.scaled_experiment import ScaledExperiment


def metrics(data, height, speed, start, end):
    mask = (data['time'] >= start) & (data['time'] < end)
    if mask.sum() < 20:
        return None
    root, vel = data['root_state'][mask], data['base_lin_vel'][mask]
    rpy = Rotation.from_quat(root[:, 3:7]).as_euler('xyz')
    contact = (data['foot_force'][mask, :, 2] > 5) & (height[mask] < .02)
    return dict(samples=int(mask.sum()), vx_mean=float(vel[:, 0].mean()),
        vx_rmse_to_command=rms(vel[:, 0]-.45), vy_rms=rms(vel[:, 1]),
        yaw_rms_deg=float(np.rad2deg(rms(rpy[:, 2]))),
        tilt_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(rpy[:, :2]**2, axis=1))))),
        root_height_min_m=float(root[:, 2].min()),
        acceleration_rms=rms(np.diff(data['dof_pos'][mask], n=2, axis=0)/.01**2),
        slip_mean_m_s=float(speed[mask][contact].mean()) if contact.any() else None,
        feet_mesh_height_p95_mm=(np.percentile(height[mask], 95, axis=0)*1000).tolist(),
        force_peak_N=float(np.linalg.norm(data['foot_force'][mask], axis=-1).max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest, arrays = load_bundle(args.bundle)
    experiment = ScaledExperiment(ROOT, args.config)
    assert manifest['identity'] == experiment.identity() and manifest['duration_s'] == 60
    vertices = foot_collision_vertices(experiment.kinematics.path)
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = dict(bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        checkpoint_sha256=manifest['checkpoint_sha256'], modes={},
        note='Descriptive valid prefixes, not causal proof. Per-initial-state averages; later bins lose failed samples. Independent of D scores. No DR acceptance.')
    intervals = ((2, 6), (6, 20), (20, 40), (40, 60.01))
    for mode in manifest['modes']:
        rows = []
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for index in range(16):
            data = episode(arrays, mode, index)
            height, speed = foot_geometry(data, vertices, manifest['foot_names'])
            failed = bool(data['initial']['failure'] or data['failure'].any())
            end = float(data['time'][-1]) if len(data['time']) else 0.
            row = dict(env=index, failed=failed, end_s=end,
                intervals={('%g-%g' % (a, b)): metrics(data, height, speed, a, b) for a, b in intervals})
            if failed:
                row['last_2s'] = metrics(data, height, speed, max(0, end-2), end+.01)
                row['preceding_2s'] = metrics(data, height, speed, max(0, end-4), end-2)
            rows.append(row)
            if not len(data['time']): continue
            t = data['time']; rpy = Rotation.from_quat(data['root_state'][:, 3:7]).as_euler('xyz')
            values = (data['base_lin_vel'][:, 0], np.rad2deg(rpy[:, 2]), data['root_state'][:, 2])
            color = 'tab:red' if failed else 'tab:blue'
            for axis, value in zip(axes, values):
                axis.plot(t[::10], value[::10], color=color, alpha=.5 if failed else .22, lw=.7)
                if failed: axis.scatter([end], [value[-1]], color=color, marker='x')
            if failed:
                failure_fig, failure_axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
                mask = t >= max(0, end-5)
                for axis, value, label in zip(failure_axes[:3], values, ('vx m/s', 'yaw deg', 'root z m')):
                    axis.plot(t[mask], value[mask]); axis.set_ylabel(label)
                failure_axes[3].plot(t[mask], height[mask]*1000)
                failure_axes[3].set_ylabel('foot mesh mm')
                failure_axes[4].plot(t[mask], data['foot_force'][mask, :, 2])
                failure_axes[4].set_ylabel('foot Fz N'); failure_axes[4].set_xlabel('time s')
                for axis in failure_axes: axis.grid(alpha=.3)
                failure_fig.suptitle('%s env%d first failure %.2fs; valid pre-reset states' % (mode, index, end))
                failure_fig.tight_layout(); failure_fig.savefig(args.output/('%s_failure_env%d.png' % (mode, index)), dpi=130)
                plt.close(failure_fig)
        axes[0].axhline(.45, color='black', ls='--')
        for axis, label in zip(axes, ('vx m/s', 'yaw deg', 'root z m')):
            axis.set_ylabel(label); axis.grid(alpha=.3)
        axes[-1].set_xlabel('time s')
        fig.suptitle('%s: all 16 initial states; red = failed prefix, blue = 60s survivor' % mode)
        fig.tight_layout(); fig.savefig(args.output/(mode+'_all_initials.png'), dpi=130); plt.close(fig)
        bins = {}
        for a, b in intervals:
            name = '%g-%g' % (a, b)
            selected = [r['intervals'][name] for r in rows if r['intervals'][name] is not None]
            bins[name] = dict(n=len(selected), means={key: float(np.mean([r[key] for r in selected if r[key] is not None]))
                for key in ('vx_mean', 'vx_rmse_to_command', 'yaw_rms_deg', 'tilt_rms_deg', 'acceleration_rms', 'slip_mean_m_s')
                if any(r[key] is not None for r in selected)})
        result['modes'][mode] = dict(episodes=rows, intervals=bins)
    with (args.output/'drift_report.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), modes={k: v['intervals'] for k, v in result['modes'].items()})))


if __name__ == '__main__': main()
