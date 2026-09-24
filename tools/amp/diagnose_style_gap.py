"""Read-only discriminator diagnostics, not physical interpolation or acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.discriminator import AMPDiscriminator
from humanoid.amp.recovery import static_negative_windows
from humanoid.amp.scaled_experiment import ScaledExperiment
from tools.amp.inspect_rollout import load_bundle, episode, features_from_episode

GROUPS = {'root_omega': (0, 3), 'joint_position': (3, 15),
          'joint_velocity': (15, 27), 'key_position': (27, 39)}


def score_summary(scores):
    reward = (1-.25*(scores-1).square()).clamp_min(0)
    return dict(score_mean=float(scores.mean()), score_std=float(scores.std(unbiased=False)),
                reward_mean=float(reward.mean()), zero_fraction=float((reward == 0).float().mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest, arrays = load_bundle(args.bundle)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != manifest['checkpoint_sha256']:
        raise ValueError('Exact evaluated checkpoint required')
    group = manifest['identity']['experiment'].replace('f1_amp_walk02_refine_', '')
    experiment = ScaledExperiment(ROOT, ROOT/'configs/amp'/('lafan_walk02_refine_'+group+'.json'))
    if manifest['identity'] != experiment.identity():
        raise ValueError('Data/config/feature identity mismatch')
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    d = AMPDiscriminator(experiment.spec, experiment.mean, experiment.std)
    d.load_state_dict(state['amp_discriminator_state_dict'], strict=True)
    d.eval()
    windows, selection = [], []
    for env in range(manifest['num_envs']):
        data = episode(arrays, 'reference', env)
        x = features_from_episode(data, experiment)
        if len(x) < 210:
            continue
        all_windows = x.unfold(0, 10, 1).permute(0, 2, 1)
        # Fixed evenly-spaced observed windows, including failed initial states.
        indices = np.linspace(200, len(all_windows)-1, 16).round().astype(int)
        windows.append(all_windows[indices])
        selection.append(dict(env=env, starts=indices.tolist(), observed_s=len(x)/100))
    policy = torch.cat(windows)
    with torch.no_grad():
        xp, xd = d.normalized_flat(policy), d.normalized_flat(experiment.windows)
        distances = torch.cdist(xp, xd)
        nearest = experiment.windows[distances.argmin(1)]
        replacement = {}
        for name, (start, end) in GROUPS.items():
            changed = policy.clone()
            changed[:, :, start:end] = nearest[:, :, start:end]
            replacement[name] = score_summary(d(changed).flatten())
        mix = []
        for alpha in np.linspace(0, 1, 11):
            mix.append(dict(toward_demo=float(alpha), **score_summary(d(policy*(1-alpha)+nearest*alpha).flatten())))
        result = dict(checkpoint_sha256=digest, identity=manifest['identity'], samples=len(policy),
                      selection=selection, policy=score_summary(d(policy).flatten()),
                      demo=score_summary(d(experiment.windows).flatten()),
                      held_demo=score_summary(d(static_negative_windows(experiment.windows)).flatten()),
                      replace_one_group_with_nearest_demo=replacement, feature_space_interpolation=mix,
                      note='Same learned D within each case, not calibrated across models. Counterfactual feature mixtures can be physically inconsistent; diagnose sensitivity only, never training samples, target motions, causal proof, or acceptance. All observed reference initial states sampled evenly after 2s.')
    xp = xp.detach().requires_grad_(True)
    gradient = torch.autograd.grad(d.network(xp).sum(), xp)[0].reshape(-1, 10, 39)
    result['normalized_input_gradient_rms'] = {name: float(gradient[:, :, start:end].square().mean().sqrt())
                                               for name, (start, end) in GROUPS.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['toward_demo'] for r in mix], [r['reward_mean'] for r in mix], marker='o')
    axes[0].set(xlabel='Feature blend toward nearest demonstration', ylabel='Learned D normalized style reward')
    labels = ['Observed']+list(replacement)
    values = [result['policy']['reward_mean']]+[replacement[k]['reward_mean'] for k in replacement]
    axes[1].bar(labels, values)
    axes[1].tick_params(axis='x', rotation=35)
    axes[1].set_ylabel('Normalized style reward')
    for axis in axes: axis.grid(axis='y', alpha=.25)
    fig.suptitle(group+': diagnostic feature substitutions (NOT physical motions)')
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(args.output/'sensitivity.png', dpi=150); plt.close(fig)
    print(json.dumps(dict(group=group, observed=result['policy'], demo=result['demo'], replacements=replacement)))


if __name__ == '__main__':
    main()
