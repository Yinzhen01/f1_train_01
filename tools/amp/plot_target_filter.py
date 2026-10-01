"""Raw versus actually applied targets, beside unfiltered physical measurements."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.inspect_rollout import load_bundle, episode
from tools.amp.jitter_audit import validate_substep_arrays
from tools.amp.analyze_jitter_events import WINDOWS, high_frequency, backward_delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m, arrays = load_bundle(a.bundle)
    if 'target_filter' not in m: raise ValueError('Not a filter experiment')
    validate_substep_arrays(m, arrays)
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for mode in m['modes']:
        for index in range(m['num_envs']):
            d = episode(arrays, mode, index); n = len(d['time']); valid = d['time'] >= 2.
            raw = arrays[mode+'_control_raw_action'][:n, index]
            applied = d['action']
            row = dict(mode=mode, env=index, captured_steps=n, post2s_steps=int(valid.sum()))
            if valid.sum() >= 100:
                for label, signal in (('raw', raw), ('applied', applied)):
                    row[label] = dict(delta_rms=float(np.sqrt(np.mean(np.diff(signal[valid], axis=0)**2))),
                                      spectrum=high_frequency(signal[valid]))
                row['residual_rms'] = float(np.sqrt(np.mean((raw-applied)[valid]**2)))
            rows.append(row)
    d = episode(arrays, 'reference', 0); n = len(d['time'])
    raw = arrays['reference_control_raw_action'][:n, 0]
    acc = backward_delta(d['dof_vel'], d['initial']['dof_vel'])/.01 if n else None
    for label, bounds in dict(WINDOWS, whole=(0., 60.)).items():
        fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
        mask = (d['time'] >= bounds[0]) & (d['time'] <= bounds[1]); t = d['time'][mask]
        for ax, j in zip(axes[:2], (4, 10)):
            ax.plot(t, raw[mask, j], alpha=.5, lw=.8, label='raw clipped command')
            ax.plot(t, d['action'][mask, j], label='applied filtered command')
            ax.set_ylabel(m['dof_names'][j]); ax.legend()
        if n:
            axes[2].plot(t, np.sqrt(np.mean(acc[mask]**2, axis=-1)), label='actual 100Hz endpoint acceleration')
            axes[2].plot(t, np.sqrt(np.mean(arrays['reference_physics_accel_squared'][:n, 0][mask], axis=-1)), alpha=.65, label='actual 1kHz acceleration RMS')
            for j in range(2):
                axes[3].plot(t, arrays['reference_physics_foot_force_peak'][:n, 0, j][mask], label=('L', 'R')[j]+' foot peak Fz')
            axes[4].plot(t, d['base_lin_vel'][mask, 0], label='body vx')
        axes[2].set_ylabel('rad/s2'); axes[3].set_ylabel('N'); axes[4].set_ylabel('m/s')
        for ax in axes:
            ax.set_xlim(*bounds); ax.grid(alpha=.2)
        for ax in axes[2:]: ax.legend()
        axes[-1].set_xlabel('Original physical time (s), missing tail is not success')
        fig.suptitle(m['identity']['experiment']+' | reference env0 | physical signals not filtered')
        fig.tight_layout(); fig.savefig(a.output/(label+'.png'), dpi=125); plt.close(fig)
    with (a.output/'raw_applied_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(target_filter=m['target_filter'], recurrence_verified=True, rows=rows,
                       physical_improvement_verified=False), stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(a.output), rows=len(rows))))


if __name__ == '__main__': main()
