"""Matched fixed60 quality audit; failed prefixes remain failures, never dropped."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.jitter import SOURCE_SHA, validate_jitter, SUBSTEP_GROUPS
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.recovery import foot_collision_vertices
from tools.amp.inspect_rollout import load_bundle, episode, analyze_episode
from tools.amp.analyze_jitter_events import WINDOWS, signals, window_summary
from tools.amp.jitter_audit import compare_environment, validate_substep_arrays
from tools.amp.verify_long_pair import initial_comparison


def relative(candidate, baseline):
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return candidate/baseline


def summarize_rows(rows):
    selected = [r for r in rows if r['post2s'] is not None]
    metrics = {}
    for key in ('vx_mean', 'action_delta_rms', 'torque_delta_rms_nm', 'acceleration_native_rms',
                'action_delta_step_rms_p95', 'torque_delta_step_rms_p95_nm',
                'acceleration_native_step_rms_p95', 'contact_proxy_slip_mean_m_s'):
        values = [r['post2s'][key] for r in selected if r['post2s'][key] is not None]
        metrics[key] = float(np.mean(values)) if values else None
    for key in ('action_spectrum', 'velocity_spectrum'):
        values = [r['post2s'][key]['high_band_power'] for r in selected if r['post2s'][key] is not None]
        metrics[key+'_high_power_joint_mean'] = float(np.mean(values)) if values else None
    for key in ('heading_rms_to_world_x_deg', 'demo_nearest_window_rms_zscore'):
        values = [r['quality'][key] for r in selected if r['quality'] is not None]
        if key == 'demo_nearest_window_rms_zscore': values = [v['mean'] for v in values]
        metrics[key] = float(np.mean(values)) if values else None
    return dict(total=len(rows), survived=sum(r['survived'] for r in rows),
        contributing_prefixes=len(selected), mean_metrics=metrics,
        mean_of_prefixes_is_not_survival_acceptance=True)


def numerical_gates(candidate, reference):
    c, r = candidate['mean_metrics'], reference['mean_metrics']
    gates = dict(full_survival=candidate['survived'] == candidate['total'] == 32)
    gates['progress'] = c['vx_mean'] is not None and c['vx_mean'] >= max(.35, .9*r['vx_mean'])
    for key, bound in (('action_delta_rms', .8), ('action_spectrum_high_power_joint_mean', .8),
        ('torque_delta_rms_nm', 1.), ('acceleration_native_rms', 1.),
        ('torque_delta_step_rms_p95_nm', 1.), ('acceleration_native_step_rms_p95', 1.),
        ('contact_proxy_slip_mean_m_s', 1.1), ('heading_rms_to_world_x_deg', 1.1),
        ('demo_nearest_window_rms_zscore', 1.1)):
        ratio = relative(c[key], r[key])
        gates[key] = ratio is not None and ratio <= bound
    return gates


def plot_window(cases, output, start, end, label):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 3, figsize=(17, 11), sharex=True, sharey='row')
    for col, (name, case) in enumerate(cases.items()):
        d, s = case['env0']
        if s is None:
            axes[0, col].set_title(name+' | reference env0 failed at initialization')
            continue
        mask = (d['time'] >= start) & (d['time'] <= end)
        t = d['time'][mask]
        axes[0, col].set_title(name+' | reference env0')
        axes[0, col].plot(t, d['base_lin_vel'][mask, 0]); axes[0, col].axhline(.45, color='gray', ls='--')
        for j, color, side in ((4, 'C0', 'L'), (10, 'C1', 'R')):
            axes[1, col].plot(t, s['target'][mask, j], color=color, alpha=.5, label=side+' target')
            axes[1, col].plot(t, d['dof_pos'][mask, j], color=color, lw=2, label=side+' actual')
        axes[1, col].axhline(-.41, color='gray', ls='--'); axes[1, col].axhline(.35, color='gray', ls='--')
        axes[1, col].legend(fontsize=7, ncol=2)
        for j in range(2):
            axes[2, col].plot(t, d['foot_force'][mask, j, 2], color='C'+str(j), lw=.7)
            axes[3, col].plot(t, s['sole_height'][mask, j]*1000, color='C'+str(j))
        axes[4, col].plot(t, np.sqrt((s['acceleration_native'][mask]**2).mean(-1)), lw=.7)
        axes[4, col].set_xlabel('Original physical time (s)')
    for row, name in enumerate(('body vx m/s', 'ankle pitch rad', 'endpoint foot Fz N', 'sole mesh height mm', '100Hz acceleration RMS rad/s2')):
        axes[row, 0].set_ylabel(name)
    for ax in axes.flat:
        ax.set_xlim(start, end); ax.grid(alpha=.2)
    fig.suptitle('Same captured initial states; no video filtering; missing tail is not a successful quiet gait')
    fig.tight_layout(); fig.savefig(output/(label+'.png'), dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    for key in ('source', 'control', 'smooth', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--groups', nargs=2, default=('control', 'smooth'),
                   choices=('control', 'smooth')+SUBSTEP_GROUPS,
                   help='Actual registered group names of the two candidate paths')
    a = p.parse_args(); torch.set_num_threads(2)
    import matplotlib
    matplotlib.use('Agg')
    a.output.mkdir(parents=True, exist_ok=False)
    cases = {}
    if len(set(a.groups)) != 2: raise ValueError('Two distinct registered groups required')
    paths = dict(zip(('source',)+tuple(a.groups), (a.source, a.control, a.smooth)))
    for name, path in paths.items():
        m, arrays = load_bundle(path)
        if m['duration_s'] != 60 or m['num_envs'] != 16 or m['mode_seeds'] != {'standing': 5, 'reference': 105}:
            raise ValueError('Not the fixed matched60 protocol')
        if m['modes'] != ['standing', 'reference']: raise ValueError('Wrong initial mode set')
        cfg = 'sustain_control' if name == 'source' else 'jitter_'+name
        e = ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_'+cfg+'.json'))
        if e.identity() != m['identity']: raise ValueError('Wrong data/config identity')
        if name == 'source':
            if m['checkpoint_sha256'] != SOURCE_SHA: raise ValueError('Wrong baseline')
        else:
            validate_jitter(e); validate_substep_arrays(m, arrays)
            source = cases['source']
            compare_environment(source['manifest']['environment'], m['environment'], name)
            if source['manifest']['runtime'] != m['runtime']: raise ValueError('Runtime changed')
            proof = initial_comparison(source['arrays'], arrays)
            if not proof['all_captured_fields_exact_equal']: raise ValueError('Different initial states')
        vertices = foot_collision_vertices(e.kinematics.path)
        rows = []
        for mode in m['modes']:
            for index in range(16):
                d = episode(arrays, mode, index)
                q, _ = analyze_episode(d, e, vertices, m['foot_names'], 60.)
                s = signals(d, m, vertices) if len(d['time']) else None
                post = window_summary(d, s, m, 2., 60.01) if s is not None else None
                row = dict(mode=mode, env=index, survived=q['survived'], failure=q['failure'],
                    observed_s=q['observed_s'], post2s=post, quality=q['post_2s'],
                    windows={k: window_summary(d, s, m, *v) if s is not None else None for k,v in WINDOWS.items()},
                    one_second_windows=[dict(start_s=t, summary=window_summary(d, s, m, t, t+1.) if s is not None else None)
                                        for t in range(2, 60)])
                if name != 'source':
                    valid = arrays[mode+'_physics_valid'][:len(d['time']), index] & (d['time'] >= 2.)
                    row['substep'] = {}
                    for key in ('accel_squared', 'accel_peak', 'torque_delta_squared', 'foot_force_peak'):
                        values = arrays[mode+'_physics_'+key][:len(d['time']), index][valid]
                        row['substep'][key] = None if not values.size else dict(mean=float(values.mean()),
                            p95=float(np.percentile(values, 95)), peak=float(values.max()))
                rows.append(row)
                if mode == 'reference' and index == 0: env0 = (d, s)
        cases[name] = dict(manifest=m, arrays=arrays, env0=env0, rows=rows,
            bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            summary=summarize_rows(rows), modes={mode:summarize_rows([r for r in rows if r['mode'] == mode]) for mode in m['modes']})
        print(json.dumps(dict(case=name, summary=cases[name]['summary'])), flush=True)
    for label, (start, end) in dict(WINDOWS, whole=(0., 60.)).items():
        plot_window(cases, a.output, start, end, label)
    numerical = {group:{baseline:numerical_gates(cases[group]['summary'], cases[baseline]['summary'])
                        for baseline in ('source', a.groups[0]) if group != baseline} for group in a.groups}
    result = dict(cases={name:{k:v for k,v in case.items() if k not in ('arrays','env0')} for name,case in cases.items()},
        numerical_gates=numerical, effectiveness_verified=False, visual_verification_pending=True,
        dr_unlocked=False, limitations='Failed prefixes retained, not survivor-only. Numerical gates do not replace all three event-window and whole-video review. Baseline lacks 1kHz data; compare substep values only between the two new groups. One training seed/clip, fixed nominal simulation only.')
    with (a.output/'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(a.output), numerical_gates=numerical)), flush=True)


if __name__ == '__main__': main()
