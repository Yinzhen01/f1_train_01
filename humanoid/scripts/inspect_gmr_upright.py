"""Compare reference curves and static FK poses. NOT a policy/dynamics rollout."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from render_gmr_policy import make_scene, label


def inspect(baseline, reference, urdf, output):
    output.mkdir(parents=True, exist_ok=False)
    clips = []
    for path in (baseline, reference):
        with np.load(path, allow_pickle=False) as data:
            clips.append({k: data[k].copy() for k in data.files})
    names = ['Old pelvis reference', 'New fixed-waist reference']
    colors = ['#9c5660', '#168b89']
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    metrics = {}
    for clip, name, color in zip(clips, names, colors):
        t = np.arange(len(clip['root_pos'])) / float(clip['fps'])
        pitch = Rotation.from_quat(clip['root_quat']).as_euler('xyz', degrees=True)[:, 1]
        acceleration = np.linalg.norm(np.gradient(clip['root_vel'], 1./float(clip['fps']), axis=0), axis=1)
        axes[0].plot(t, pitch, label=name, color=color)
        axes[1].plot(t, clip['root_pos'][:, 2], color=color)
        axes[2].plot(t, acceleration, color=color)
        standing = (t <= 1.) | (t >= 4.2)
        metrics[name] = dict(pitch_mean_deg=float(pitch.mean()), pitch_range_deg=[float(pitch.min()), float(pitch.max())],
                             standing_pitch_abs_max_deg=float(np.abs(pitch[standing]).max()),
                             root_acceleration_max_m_s2=float(acceleration.max()))
    axes[0].axhline(0, color='black', linewidth=.6)
    axes[0].set_ylabel('Torso pitch (deg)\npositive = forward')
    axes[1].set_ylabel('Root height (m)')
    axes[2].set_ylabel('Root acceleration (m/s2)')
    axes[2].set_xlabel('Clip time (s)')
    for ax in axes:
        ax.axvspan(0, 1., alpha=.05, color='green')
        ax.axvspan(4.2, 4.6, alpha=.05, color='green')
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper right')
    fig.suptitle('KIT317 | 12-DOF reference comparison | kinematics only, not a trained policy')
    fig.tight_layout()
    fig.savefig(output / 'reference_curves.png', dpi=150)
    plt.close(fig)
    model, _ = make_scene(urdf, output)
    data = mujoco.MjData(model)
    ids = [int(model.joint(n).qposadr[0]) for n in clips[0]['joint_names'].astype(str)]
    root = int(model.joint('render_root').qposadr[0])
    option = mujoco.MjvOption()
    model.geom_group[model.geom('render_ground').id] = 1
    option.geomgroup[:] = 0
    option.geomgroup[1] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.elevation, camera.distance = -7., 2.2
    renderer = mujoco.Renderer(model, height=480, width=480)
    peak = int(np.argmax(Rotation.from_quat(clips[1]['root_quat']).as_euler('xyz')[:, 1]))
    rows = []
    try:
        for clip, name in zip(clips, names):
            panels = []
            for i in (0, peak, len(clip['root_pos']) - 1):
                data.qpos[root:root+3] = clip['root_pos'][i]
                data.qpos[root+3:root+7] = clip['root_quat'][i, [3, 0, 1, 2]]
                data.qpos[ids] = clip['dof_pos'][i]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = clips[0]['root_pos'][i]
                camera.lookat[2] = .65
                old_heading = Rotation.from_quat(clips[0]['root_quat'][i]).as_euler('xyz', degrees=True)[2]
                camera.azimuth = old_heading + 90.
                renderer.update_scene(data, camera=camera, scene_option=option)
                panel = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                pitch = Rotation.from_quat(clip['root_quat'][i]).as_euler('xyz', degrees=True)[1]
                label(panel, name, 12, 28, scale=.52)
                label(panel, 't=%.2fs | pitch=%+.2f deg' % (i / float(clip['fps']), pitch), 12, 453, scale=.52)
                panels.append(panel)
            rows.append(np.concatenate(panels, axis=1))
    finally:
        renderer.close()
    if not cv2.imwrite(str(output / 'reference_poses.png'), np.concatenate(rows, axis=0)):
        raise RuntimeError('Could not save reference comparison')
    metrics['boundary'] = 'Static kinematic poses and numerical references only; no trained policy or dynamics rollout.'
    (output / 'comparison.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'reference', 'urdf', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    inspect(args.baseline, args.reference, args.urdf, args.output)
