"""Audit user-identified timestamps and all recorded prefixes, without simulation.

The100Hz archive cannot recover within-step1kHz acceleration/contact peaks.
Position servo setpoints outside limits are not actual joint-limit violations.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import welch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.discriminator import AMPDiscriminator
from humanoid.amp.recovery import foot_collision_vertices
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.inspect_rollout import load_bundle, episode, foot_geometry, rms
from tools.amp.audit_reward_balance import task_terms, style_terms
from tools.amp.audit_recorded_estimator import replay, interval

WINDOWS = {'near5s': (4., 6.), 'near29to30s': (28., 31.), 'near48s': (47., 49.)}


def backward_delta(values, initial):
    values = np.asarray(values, dtype=np.float64)
    return np.diff(np.concatenate((np.asarray(initial)[None], values)), axis=0)


def command_geometry(actions, default, scale, lower, upper):
    target = np.asarray(default)+scale*np.asarray(actions)
    excess = np.maximum(np.asarray(lower)-target, 0)+np.maximum(target-np.asarray(upper), 0)
    return target, excess


def high_frequency(values, cutoff=8.):
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 100:
        return None
    f, power = welch(x, fs=100, nperseg=min(1000, len(x)), axis=0)
    high = f >= cutoff
    total = power.sum(axis=0)
    fractions = np.divide(power[high].sum(axis=0), total, out=np.zeros_like(total), where=total > 1e-20)
    return dict(cutoff_hz=cutoff, fraction=fractions.tolist(),
                high_band_power=(power[high].sum(axis=0)*(f[1]-f[0])).tolist(),
                total_power=(total*(f[1]-f[0])).tolist(),
                peak_in_high_band_hz=f[high][power[high].argmax(axis=0)].tolist(),
                stationary_columns=(total <= 1e-20).tolist())


def signals(data, manifest, vertices):
    cfg = manifest['environment']; props = manifest['runtime']['dof_properties']
    names = manifest['dof_names']
    default = [cfg['init_state']['default_joint_angles'][n] for n in names]
    target, excess = command_geometry(data['action'], default, cfg['control']['action_scale'],
                                      props['lower'], props['upper'])
    du = backward_delta(data['action'], data['initial']['action'])
    dq = backward_delta(data['dof_pos'], data['initial']['dof_pos'])/.01
    qacc = backward_delta(dq, dq[0])/.01  # Tick1 is not interpreted; windows start>=2s.
    acceleration = backward_delta(data['dof_vel'], data['initial']['dof_vel'])/.01
    torque_delta = backward_delta(data['torque'], data['initial']['torque'])
    height, slip = foot_geometry(data, vertices, manifest['foot_names'])
    return dict(target=target, target_excess=excess, action_delta=du, acceleration_native=acceleration,
        acceleration_qfd=qacc, torque_delta=torque_delta, sole_height=height, sole_speed=slip,
        pd_target_jump_nm=du*cfg['control']['action_scale']*np.asarray(manifest['runtime']['p_gains']))


def window_summary(data, s, manifest, start, end):
    active = (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
    if active.sum() < 20:
        return None
    props = manifest['runtime']['dof_properties']
    q = data['dof_pos'][active]; excess = s['target_excess'][active]
    force = data['foot_force'][active, :, 2]
    h = s['sole_height'][active]
    contact = (force > 5.) & (h < .02)
    below = np.asarray(props['lower']); above = np.asarray(props['upper'])
    limit_actual = (q < below-1e-4) | (q > above+1e-4)
    bound_near = (q < below+.01) | (q > above-.01)
    expected = int(round((end-start)*100))
    return dict(samples=int(active.sum()), requested_samples=expected,
        complete_window=int(active.sum()) == expected, requested_interval_s=[start, end],
        vx_mean=float(data['base_lin_vel'][active, 0].mean()),
        action_delta_rms=rms(s['action_delta'][active]), torque_delta_rms_nm=rms(s['torque_delta'][active]),
        action_delta_step_rms_p95=float(np.percentile(np.sqrt((s['action_delta'][active]**2).mean(-1)), 95)),
        torque_delta_step_rms_p95_nm=float(np.percentile(np.sqrt((s['torque_delta'][active]**2).mean(-1)), 95)),
        acceleration_native_step_rms_p95=float(np.percentile(np.sqrt((s['acceleration_native'][active]**2).mean(-1)), 95)),
        acceleration_native_rms=rms(s['acceleration_native'][active]), acceleration_qfd_rms=rms(s['acceleration_qfd'][active]),
        pd_target_jump_rms_nm=rms(s['pd_target_jump_nm'][active]),
        foot_force_max_n=force.max(axis=0).tolist(), foot_force_p99_n=np.percentile(force, 99, axis=0).tolist(),
        foot_ground_contact_fraction=contact.mean(axis=0).tolist(),
        contact_proxy_slip_mean_m_s=None if not contact.any() else float(s['sole_speed'][active][contact].mean()),
        max_penetration_mm=(np.maximum(-h, 0).max(axis=0)*1000).tolist(),
        target_outside_fraction_per_joint=(excess > 0).mean(axis=0).tolist(),
        target_excess_max_rad_per_joint=excess.max(axis=0).tolist(),
        actual_near_hard_limit_fraction_per_joint=bound_near.mean(axis=0).tolist(),
        actual_hard_limit_violation_fraction_per_joint=limit_actual.mean(axis=0).tolist(),
        action_delta_rms_per_joint=np.sqrt((s['action_delta'][active]**2).mean(axis=0)).tolist(),
        acceleration_native_rms_per_joint=np.sqrt((s['acceleration_native'][active]**2).mean(axis=0)).tolist(),
        torque_saturation_fraction_per_joint=(np.abs(data['torque'][active]) >= np.asarray(props['effort'])*manifest['environment']['safety']['torque_limit']*.99).mean(axis=0).tolist(),
        action_spectrum=high_frequency(data['action'][active]),
        velocity_spectrum=high_frequency(data['dof_vel'][active]))


def save_curves(data, s, terms, estimate, manifest, output, start, end, label):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
    active = (data['time'] >= start) & (data['time'] <= end)
    t = data['time'][active]
    axes[0].plot(t, data['base_lin_vel'][active, 0], label='actual vx')
    keep = (estimate['time'] >= start) & (estimate['time'] <= end)
    axes[0].plot(estimate['time'][keep], estimate['estimate'][keep, 0], label='reconstructed ES vx', alpha=.7)
    axes[0].axhline(.45, color='gray', ls='--'); axes[0].set_ylabel('m/s')
    for j, side in ((4, 'left ankle pitch'), (10, 'right ankle pitch')):
        axes[1].plot(t, s['target'][active, j], label=side+' target', alpha=.65)
        axes[1].plot(t, data['dof_pos'][active, j], label=side+' actual', lw=1.5)
    axes[1].axhline(-.41, color='gray', ls='--'); axes[1].axhline(.35, color='gray', ls='--')
    axes[1].set_ylabel('ankle pitch rad')
    for j in (0, 3, 4, 6, 9, 10):
        axes[2].plot(t, s['pd_target_jump_nm'][active, j], label=manifest['dof_names'][j].replace('_joint', ''), lw=.65)
    axes[2].set_ylabel('Kp*scale*delta_action Nm')
    for foot, name in enumerate(('left', 'right')):
        axes[3].plot(t, data['foot_force'][active, foot, 2], label=name+' Fn', lw=.8)
        axes[4].plot(t, s['sole_height'][active, foot]*1000, label=name+' sole min height')
    axes[3].set_ylabel('recorded endpoint N'); axes[4].set_ylabel('mm')
    axes[5].plot(t, np.sqrt((s['acceleration_native'][active]**2).mean(axis=1)), label='native dq diff RMS')
    axes[5].plot(t, np.sqrt((s['acceleration_qfd'][active]**2).mean(axis=1)), label='q diff RMS', alpha=.7)
    axes[5].set_ylabel('rad/s2')
    for key in ('amp_style', 'recovery_progress', 'recovery_smoothness', 'refine_acceleration', 'dof_pos_limits'):
        axes[6].plot(t, terms[key][active], label=key, lw=.8)
    axes[6].set_ylabel('reward/control step'); axes[6].set_xlabel('Original uncut video / physical time s')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(loc='upper right', fontsize=8, ncol=3)
    fig.suptitle(label+' | reference env0, real100Hz states; fixed final D; no1kHz peak reconstruction')
    fig.tight_layout(); fig.savefig(output/(label+'.png'), dpi=135); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'checkpoint', 'training-manifest', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(2)
    m, arrays = load_bundle(a.bundle); e = ScaledExperiment(ROOT, a.config)
    sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if m['checkpoint_sha256'] != sha or m['identity'] != e.identity():
        raise ValueError('Source identity mismatch')
    training = json.loads(torch.load(a.training_manifest, weights_only=True, map_location='cpu')['manifest_json'])
    if training['identity'] != m['identity']: raise ValueError('Training identity mismatch')
    state = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    cls, _ = cpu_ppo_classes(ROOT); cfg = m['environment']['env']
    with contextlib.redirect_stdout(io.StringIO()):
        policy = cls(cfg['num_single_obs']*cfg['short_frame_stack'], cfg['num_single_obs'],
                     cfg['num_privileged_obs'], cfg['num_actions'], **training['ppo_config']['policy'])
    policy.load_state_dict(state['model_state_dict'], strict=True); policy.eval()
    disc = AMPDiscriminator(e.spec, e.mean, e.std)
    disc.load_state_dict(state['amp_discriminator_state_dict'], strict=True); disc.eval()
    vertices = foot_collision_vertices(e.kinematics.path)
    a.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    result = dict(bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(), checkpoint_sha256=sha,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), dof_names=m['dof_names'],
        user_windows=WINDOWS, modes={},
        limitations='All32 recorded prefixes retained; pointwise100Hz endpoints cannot recover1kHz force/acceleration peaks. '
        'Force>5N and sole<2cm is only an analysis ground-contact proxy, not a reward deadband. Target-limit excess is not '
        'actual joint-limit violation; PD targets can intentionally create loads. Fixed final D is not training-time D. '
        'Reconstructed reward misses non-foot collision; temporal association is not causal intervention.')
    for mode in m['modes']:
        rows = []
        for index in range(m['num_envs']):
            data = episode(arrays, mode, index)
            if not len(data['time']):
                rows.append(dict(env=index, failure=True, observed_s=0., windows={k:None for k in WINDOWS}, post2s=None))
                continue
            s = signals(data, m, vertices)
            end = float(data['time'][-1])+.01
            row = dict(env=index, failure=bool(data['initial']['failure'] or data['failure'].any()),
                observed_s=end-.01, windows={k: window_summary(data, s, m, *v) for k, v in WINDOWS.items()},
                post2s=window_summary(data, s, m, 2., end))
            if mode == 'reference' and index == 0 and len(data['time']) > 66:
                estimate = replay(data, m, policy, stride=1)
                terms, missing = task_terms(data, m, vertices)
                terms['amp_style'] = style_terms(data, e, disc)
                row.update(action_reproduction=estimate['parity'], missing_reward_terms=missing, event_rewards={}, estimator={})
                for name, (start, stop) in dict(WINDOWS, whole=(2., end)).items():
                    keep = (data['time'] >= start-1e-8) & (data['time'] < stop-1e-8)
                    row['event_rewards'][name] = ({k: float(v[keep].mean()) for k, v in terms.items()}
                                                  if keep.any() else None)
                    row['estimator'][name] = interval(estimate, start, stop)
                    save_curves(data, s, terms, estimate, m, a.output, start, stop, name)
                row['one_second_windows'] = [dict(start_s=float(start), summary=window_summary(data, s, m, start, start+1.))
                                             for start in np.arange(2., 60., 1.)]
                np.savez_compressed(a.output/'reference_env0_signals.npz', time=data['time'], **s,
                                    foot_force=data['foot_force'], dof_pos=data['dof_pos'], action=data['action'], torque=data['torque'])
            rows.append(row)
        result['modes'][mode] = rows
    with (a.output/'jitter_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    r = result['modes']['reference'][0]
    print(json.dumps(dict(output=str(a.output), action_reproduction=r.get('action_reproduction'),
        windows={k:None if v is None else {n:v[n] for n in ('complete_window','vx_mean','action_delta_rms','acceleration_native_rms','foot_force_max_n')} for k,v in r['windows'].items()},
        estimator=r.get('estimator'), event_rewards=r.get('event_rewards'))))


if __name__ == '__main__': main()
