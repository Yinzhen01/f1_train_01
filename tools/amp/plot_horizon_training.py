"""Full250-update diagnostics; training averages are not independent gait quality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.amp.verify_horizon_formal import read_cloud_log, validate_updates
from tools.amp.contact_audit import validate_contact_updates
from tools.amp.progress_audit import validate_progress_updates
from tools.amp.direction_audit import validate_direction_updates
from tools.amp.sustain_audit import validate_sustain_updates
from tools.amp.verify_refinement_smoke import parse_updates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', action='append', required=True, help='label=full.log')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--family', choices=('horizon', 'contact', 'progress', 'direction', 'sustain'), default='horizon')
    a = p.parse_args()
    cases, result = {}, {}
    fields = ('task_reward', 'style_reward', 'mixed_reward', 'style_zero_fraction',
              'discriminator_real_policy_score', 'discriminator_demo_score',
              'discriminator_held_pose_score', 'discriminator_total',
              'discriminator_gradient_penalty', 'discriminator_bridge_gradient_penalty')
    for entry in a.case:
        label, path = entry.split('=', 1)
        if label in cases: raise ValueError('Duplicate case label')
        path = Path(path)
        log_diagnostics = {}
        rows = parse_updates(read_cloud_log(path, allow_post_completion_binary=a.family == 'sustain', diagnostics=log_diagnostics))
        if a.family == 'contact': validate_contact_updates(rows, formal=True)
        elif a.family == 'progress': validate_progress_updates(rows, formal=True)
        elif a.family == 'direction': validate_direction_updates(rows, formal=True)
        elif a.family == 'sustain': validate_sustain_updates(rows, formal=True)
        else: validate_updates(rows)
        cases[label] = rows
        result[label] = dict(log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), updates=250, log_diagnostics=log_diagnostics,
            first50={k: float(np.mean([r[k] for r in rows[:50]])) for k in fields},
            last50={k: float(np.mean([r[k] for r in rows[-50:]])) for k in fields})
    a.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    plots = [('task_reward', 'Mean task reward / control step'),
             ('style_reward', 'Mean weighted AMP style reward / step'),
             ('style_zero_fraction', 'Fraction of valid AMP rewards clipped to zero'),
             ('discriminator_real_policy_score', 'Current D score: real policy (not probability)'),
             ('discriminator_demo_score', 'Current D score: demonstration (not probability)'),
             ('discriminator_bridge_gradient_penalty', 'Mixed-feature gradient penalty contribution')]
    for label, rows in cases.items():
        x = np.asarray([r['iteration'] for r in rows])
        for axis, (key, title) in zip(axes.flat, plots):
            y = np.asarray([r[key] for r in rows])
            line = axis.plot(x, y, alpha=.2)[0]
            axis.plot(x[24:], np.convolve(y, np.ones(25)/25, mode='valid'),
                      label=label, color=line.get_color())
            axis.set_title(title); axis.set_xlabel('Completed PPO updates'); axis.grid(alpha=.25)
    for axis in axes.flat: axis.legend()
    subtitle = ('Contact reward scales differ; task totals are not a common physical score' if a.family == 'contact'
                else 'Different episode coverage changes training samples; not a gait-acceptance plot')
    if a.family == 'progress': subtitle = 'Velocity reward frames differ; totals are not a common physical score'
    if a.family == 'direction': subtitle = 'Direction reward terms differ; totals are not a common physical score'
    if a.family == 'sustain': subtitle = 'Progress weights differ; higher task reward is not evidence of better gait'
    fig.suptitle('Identical update budget; faint=raw, solid=trailing25 mean\n'+subtitle)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(a.output/'training_curves.png', dpi=150); plt.close(fig)
    report = dict(cases=result, family=a.family, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Current D changes during learning; reward/loss alone cannot prove style or physical success. Use independent matched60s trajectories.')
    with (a.output/'training_summary.json').open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2)
    print(json.dumps(dict(output=str(a.output), cases=result)))


if __name__ == '__main__': main()
