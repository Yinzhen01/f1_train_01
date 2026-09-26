"""Reconstruct recorded-state reward terms; not a new simulated intervention."""
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
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.discriminator import AMPDiscriminator
from humanoid.amp.recovery import centered_velocity_reward, foot_collision_vertices
from humanoid.amp.refinement import robust_acceleration_cost, ground_force_weight
from humanoid.amp.contact import force_tail_cost
from tools.amp.inspect_rollout import load_bundle, episode, foot_geometry, features_from_episode


def differences(data, name):
    sequence = np.concatenate((data['initial'][name][None], data[name]))
    return np.diff(sequence, axis=0)


def task_terms(data, manifest, vertices):
    env = manifest['environment']; rewards = env['rewards']; scales = rewards['scales']
    dt = .01
    tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    body = tensor(data['base_lin_vel'])
    command = body.new_tensor([.45, 0.]).expand(len(body), 2)
    raw = {}
    raw['recovery_progress'] = centered_velocity_reward(body[:, :2], command).numpy()
    if rewards.get('direction_world_fraction', 0.) != 0.:
        raise ValueError('This diagnostic requires the unchanged body-only progress')
    if rewards.get('progress_velocity_frame', 'body') != 'body':
        raise ValueError('This diagnostic requires body progress')
    rotation = Rotation.from_quat(data['root_state'][:, 3:7])
    angles = rotation.as_euler('xyz')
    raw['refine_heading'] = 1-np.cos(angles[:, 2])
    raw['recovery_tilt'] = np.square(rotation.as_matrix()[:, 2, :2]).sum(-1)
    raw['recovery_yaw_rate'] = data['base_ang_vel'][:, 2]**2
    raw['recovery_vertical_velocity'] = data['base_lin_vel'][:, 2]**2
    acceleration = differences(data, 'dof_vel')/dt
    first = differences(data, 'action')
    second = np.concatenate((np.zeros_like(first[:1]), np.diff(first, axis=0)))
    # First two ticks lack pre-initial action history and are never summarized.
    active = np.arange(len(first)) >= 2
    raw['recovery_smoothness'] = ((first**2+second**2).sum(-1))*active
    raw['refine_acceleration'] = robust_acceleration_cost(tensor(acceleration)).numpy()*active
    height, speed = foot_geometry(data, vertices, manifest['foot_names'])
    weight = ground_force_weight(tensor(data['foot_force'][:, :, 2]), tensor(height)).numpy()
    raw['refine_slip'] = (weight*speed**2).mean(-1)*active
    force = np.linalg.norm(data['foot_force'], axis=-1)
    raw['feet_contact_forces'] = np.clip(force-rewards['max_contact_force'], 0, 400).sum(-1)
    raw['contact_force_tail'] = force_tail_cost(tensor(data['foot_force']), rewards['max_contact_force']).numpy()
    raw['dof_acc'] = (acceleration**2).sum(-1)
    raw['dof_vel'] = (data['dof_vel']**2).sum(-1)
    raw['torques'] = (data['torque']**2).sum(-1)
    props = manifest['runtime']['dof_properties']; safety = env['safety']
    lower = np.asarray(props['lower'])*safety['pos_limit']
    upper = np.asarray(props['upper'])*safety['pos_limit']
    middle, half = (lower+upper)/2, (upper-lower)*.5*rewards['soft_dof_pos_limit']
    raw['dof_pos_limits'] = (np.maximum(middle-half-data['dof_pos'], 0)+
                             np.maximum(data['dof_pos']-middle-half, 0)).sum(-1)
    velocity_limit = np.asarray(props['velocity'])*safety['vel_limit']*rewards['soft_dof_vel_limit']
    raw['dof_vel_limits'] = np.clip(np.abs(data['dof_vel'])-velocity_limit, 0, 1).sum(-1)
    torque_limit = np.asarray(props['effort'])*safety['torque_limit']*rewards['soft_torque_limit']
    raw['dof_torque_limits'] = np.maximum(np.abs(data['torque'])-torque_limit, 0).sum(-1)
    raw['termination'] = data['failure'].astype(float)
    missing = {k: v for k, v in scales.items() if v and k not in raw}
    if set(missing) != {'collision'}:
        raise ValueError('Unexpected unreconstructed active rewards: '+str(missing))
    weighted = {k: np.asarray(v)*scales[k]*dt for k, v in raw.items() if scales.get(k, 0)}
    return weighted, missing


def style_terms(data, experiment, discriminator):
    features = features_from_episode(data, experiment)
    windows = features.unfold(0, 10, 1).permute(0, 2, 1).contiguous()
    with torch.no_grad():
        scores = torch.cat([discriminator(batch).flatten() for batch in windows.split(1024)]).numpy()
    reward = (1-.25*(scores-1)**2).clip(min=0.)
    # Nine excluded warmup ticks; use only real history, never padding.
    full = np.full(len(data['time']), np.nan)
    full[9:] = reward*experiment.cfg['reward']['style_scale']*.01*experiment.cfg['reward']['style_weight']
    return full


def summarize_window(data, terms, start, end):
    mask = (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
    expected = int(round((end-start)*100))
    if expected < 20 or int(mask.sum()) != expected: return None
    if not all(np.isfinite(x[mask]).all() for x in terms.values()):
        raise ValueError('Window includes missing history or nonfinite reward')
    values = {k: float(v[mask].mean()) for k, v in terms.items()}
    values['partial_total_no_collision'] = sum(values.values())
    return dict(samples=expected, body_vx=float(data['base_lin_vel'][mask, 0].mean()),
                qd_rms=float(np.sqrt(np.mean(data['dof_vel'][mask]**2))), rewards=values)


def main():
    p = argparse.ArgumentParser()
    for key in ('bundle', 'checkpoint', 'config', 'output'): p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(2)
    manifest, arrays = load_bundle(a.bundle)
    e = ScaledExperiment(ROOT, a.config)
    digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    assert manifest['checkpoint_sha256'] == digest and manifest['identity'] == e.identity()
    assert manifest['duration_s'] == 60 and manifest['num_envs'] == 16
    assert e.cfg['reward'] == dict(task_weight=1, style_weight=1, style_scale=5, dt=.01)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    assert state['amp_identity'] == e.identity()
    d = AMPDiscriminator(e.spec, e.mean, e.std)
    d.load_state_dict(state['amp_discriminator_state_dict'], strict=True); d.eval()
    vertices = foot_collision_vertices(e.kinematics.path)
    a.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = dict(checkpoint_sha256=digest, bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
        modes={}, limitation='100Hz recorded-state reconstruction, fixed final discriminator. Collision '
        'reward unavailable: no non-foot contact force capture. Not exact total reward, training-time D, '
        'counterfactual simulation, or causal/acceptance proof. First two action-history ticks excluded.')
    for mode in manifest['modes']:
        rows = []
        for index in range(16):
            data = episode(arrays, mode, index)
            terms, missing = task_terms(data, manifest, vertices)
            terms['amp_style'] = style_terms(data, e, d)
            failed = bool(data['initial']['failure'] or data['failure'].any())
            end = float(data['time'][-1])
            row = dict(env=index, failed=failed, observed_s=end, missing_terms=missing,
                fixed_2to4s=summarize_window(data, terms, 2., 4.),
                fixed_20to24s=summarize_window(data, terms, 20., 24.))
            if failed:
                row['pre_4to2s'] = summarize_window(data, terms, end-4, end-2)
                row['last2s'] = summarize_window(data, terms, end-2+.01, end+.01)
            rows.append(row)
            if index == 0 or failed:
                np.savez_compressed(a.output/('%s_env%d_terms.npz' % (mode, index)),
                    time=data['time'], body_vx=data['base_lin_vel'][:, 0], **terms)
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                mask = data['time'] >= (max(2., end-8) if failed else 2.)
                t = data['time'][mask]
                axes[0].plot(t, data['base_lin_vel'][mask, 0]); axes[0].axhline(.45, ls='--', color='gray')
                axes[0].set_ylabel('Body vx m/s')
                for key in ('recovery_progress', 'amp_style', 'refine_heading'):
                    axes[1].plot(t, terms[key][mask], label=key, lw=.8)
                for key in ('refine_slip', 'refine_acceleration', 'recovery_smoothness', 'feet_contact_forces', 'contact_force_tail'):
                    axes[2].plot(t, terms[key][mask], label=key, lw=.7)
                for ax in axes: ax.grid(alpha=.25)
                for ax in axes[1:]: ax.legend(); ax.set_ylabel('dt-scaled reward / step')
                axes[-1].set_xlabel('Physical time s')
                fig.suptitle('%s env%d: reconstructed components, collision omitted, %s' %
                    (mode, index, 'failed prefix' if failed else '60s survivor'))
                fig.tight_layout(); fig.savefig(a.output/('%s_env%d_rewards.png' % (mode, index)), dpi=130); plt.close(fig)
        result['modes'][mode] = rows
    with (a.output/'reward_balance.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({m: [r for r in rows if r['failed']] for m, rows in result['modes'].items()}))


if __name__ == '__main__': main()
