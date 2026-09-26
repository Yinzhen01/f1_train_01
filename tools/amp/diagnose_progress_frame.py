"""Read-only body-vs-world progress audit; hypothetical rewards are not training."""
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
from humanoid.amp.recovery import centered_velocity_reward
from tools.amp.inspect_rollout import load_bundle, episode


FIELDS = ('body_vx_mean', 'world_vx_mean', 'world_vy_abs_mean', 'body_progress_reward_mean',
          'hypothetical_world_progress_reward_mean', 'heading_reward_mean', 'yaw_rms_deg')


def series(data):
    body = torch.as_tensor(data['base_lin_vel'][:, :2])
    world = torch.as_tensor(data['root_state'][:, 7:9])
    cmd = torch.tensor([.45, 0.], dtype=body.dtype).expand_as(body)
    yaw = Rotation.from_quat(data['root_state'][:, 3:7]).as_euler('xyz')[:, 2]
    return dict(body_progress=centered_velocity_reward(body, cmd).numpy()*.02,
        world_progress=centered_velocity_reward(world, cmd).numpy()*.02,
        heading=-.005*(1-np.cos(yaw)), yaw=yaw)


def summarize(data, values):
    mask = data['time'] >= 2.
    if not mask.any(): return {k: None for k in FIELDS}
    return dict(body_vx_mean=float(data['base_lin_vel'][mask, 0].mean()),
        world_vx_mean=float(data['root_state'][mask, 7].mean()),
        world_vy_abs_mean=float(np.abs(data['root_state'][mask, 8]).mean()),
        body_progress_reward_mean=float(values['body_progress'][mask].mean()),
        hypothetical_world_progress_reward_mean=float(values['world_progress'][mask].mean()),
        heading_reward_mean=float(values['heading'][mask].mean()),
        yaw_rms_deg=float(np.rad2deg(np.sqrt(np.mean(values['yaw'][mask]**2)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', action='append', required=True, help='label=bundle.pt')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    result = {}
    for item in a.case:
        label, path = item.split('=', 1); path = Path(path)
        if label in result: raise ValueError('Duplicate label')
        m, arrays = load_bundle(path)
        assert m['num_envs'] == 16 and m['duration_s'] == 60
        ranges = m['environment']['commands']['ranges']
        assert ranges['lin_vel_x'] == [.45, .45] and ranges['lin_vel_y'] == [0., 0.]
        assert m['environment']['rewards']['scales']['recovery_progress'] == 2.
        assert m['environment']['rewards']['scales']['refine_heading'] == -.5
        case = dict(bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), modes={})
        for mi, mode in enumerate(m['modes']):
            rows = []
            for index in range(16):
                data = episode(arrays, mode, index)
                if not len(data['time']): raise ValueError('No captured states')
                v = series(data)
                rows.append(dict(env=index, samples_post2s=int((data['time'] >= 2.).sum()),
                    failure=bool(data['initial']['failure'] or data['failure'].any()),
                    observed_s=len(data['time'])*.01, **summarize(data, v)))
                if index == 0:
                    axes[mi, 0].plot(data['root_state'][::10, 0], data['root_state'][::10, 1], label=label)
                    axes[mi, 1].plot(data['time'][::10], np.rad2deg(v['yaw'][::10]), label=label)
            selected = {k: [r[k] for r in rows if r[k] is not None] for k in FIELDS}
            case['modes'][mode] = dict(episodes=rows, metric_n={k: len(v) for k, v in selected.items()},
                equal_initial_mean={k: float(np.mean(v)) if v else None for k, v in selected.items()})
            axes[mi, 0].set_title(mode+' env0 world path'); axes[mi, 0].set_xlabel('world X m'); axes[mi, 0].set_ylabel('world Y m')
            axes[mi, 0].axis('equal')
            axes[mi, 1].set_title(mode+' env0 heading'); axes[mi, 1].set_xlabel('time s'); axes[mi, 1].set_ylabel('yaw deg')
        result[label] = case
    for axis in axes.flat: axis.legend(); axis.grid(alpha=.25)
    fig.suptitle('Fixed env0 only in plots; all16 initial states retained in report\nWorld-frame reward is an offline counterfactual, not proof of a better learned policy')
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(a.output/'world_paths.png', dpi=140); plt.close(fig)
    report = dict(cases=result, source_function='X1AMPRecoveryEnv._reward_recovery_progress',
        limitation='Current task velocity is body-frame. AMP features are root-local and do not constrain global heading. Hypothetical world reward is not a policy intervention, and does not prove causation or effectiveness.')
    with (a.output/'progress_frame_report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: {m: v['equal_initial_mean'] for m, v in r['modes'].items()} for k, r in result.items()}))


if __name__ == '__main__': main()
