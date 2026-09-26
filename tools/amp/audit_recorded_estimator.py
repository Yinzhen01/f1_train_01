"""Reconstruct no-DR actor inputs, prove action parity, then inspect velocity ES.

Only sampled existing states are replayed through the exact saved network.
This is not a simulation or a causal intervention on the estimator.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.inspect_rollout import load_bundle, episode, rms


def actual_euler_function():
    path = ROOT/'humanoid/envs/base/legged_robot.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in ('copysign_new', 'get_euler_rpy', 'get_euler_xyz_tensor')]
    scope = dict(torch=torch, np=np)
    exec(compile(tree, str(path), 'exec'), scope)
    return scope['get_euler_xyz_tensor']


def single_observations(data, manifest):
    cfg = manifest['environment']; env = cfg['env']; scale = cfg['normalization']['obs_scales']
    if (env['num_single_obs'] != 47 or env['frame_stack'] != 66 or env['short_frame_stack'] != 5 or
            cfg['noise']['add_noise'] or cfg['env']['use_ref_actions']):
        raise ValueError('Unsupported observation or noise/reference-action contract')
    for flag in ('add_lag', 'add_dof_lag', 'add_dof_pos_vel_lag', 'add_imu_lag'):
        if cfg['domain_rand'].get(flag, False):
            raise ValueError('Cannot reconstruct lagged observations: '+flag)
    ranges = cfg['commands']['ranges']
    if ranges['lin_vel_x'] != [.45, .45] or ranges['lin_vel_y'] != [0., 0.] or ranges['ang_vel_yaw'] != [0., 0.]:
        raise ValueError('Commands were not captured and must be fixed')
    n = len(data['time'])
    target = torch.tensor([cfg['init_state']['default_joint_angles'][name] for name in manifest['dof_names']])
    command = torch.tensor([0., 1., .45*scale['lin_vel'], 0., 0.]).expand(n, -1)
    value = torch.cat((command, (torch.as_tensor(data['dof_pos'])-target)*scale['dof_pos'],
        torch.as_tensor(data['dof_vel'])*scale['dof_vel'], torch.as_tensor(data['action']),
        torch.as_tensor(data['base_ang_vel'])*scale['ang_vel'],
        actual_euler_function()(torch.as_tensor(data['root_state'][:, 3:7]))*scale['quat']), dim=-1)
    limit = cfg['normalization']['clip_observations']
    return value.clamp(-limit, limit)


def replay(data, manifest, policy, stride=5):
    obs = single_observations(data, manifest)
    # History ends at k; the action applied to transition k+1 uses that history.
    end = np.arange(65, len(obs)-1, stride)
    if not len(end):
        raise ValueError('Insufficient captured history')
    prediction, estimated = [], []
    with torch.no_grad():
        for batch in torch.tensor(end).split(256):
            history = obs[batch[:, None]+torch.arange(-65, 1)].reshape(len(batch), -1)
            prediction.append(policy.act_inference(history))
            estimated.append(policy.state_estimator(history[:, -policy.num_short_obs:]))
    prediction = torch.cat(prediction)
    clip = manifest['environment']['normalization']['clip_actions']
    predicted = prediction.clamp(-clip, clip).numpy()
    observed = data['action'][end+1]
    error = np.abs(predicted-observed)
    maximum = float(error.max())
    parity = dict(sampled_transitions=len(end), stride_ticks=stride, max_action_abs_error=maximum,
                  action_rmse=rms(error), action_parity_tolerance=.001, passed=maximum < .001)
    if not parity['passed']:
        raise ValueError('Reconstructed action parity failed: '+json.dumps(parity))
    estimated = torch.cat(estimated).numpy()/manifest['environment']['normalization']['obs_scales']['lin_vel']
    return dict(time=data['time'][end], estimate=estimated, actual=data['base_lin_vel'][end], parity=parity)


def interval(row, start, end):
    mask = (row['time'] >= start) & (row['time'] < end)
    if mask.sum() < 4:
        return None
    estimate, actual = row['estimate'][mask], row['actual'][mask]
    return dict(n=int(mask.sum()), estimated_vx_mean=float(estimate[:, 0].mean()),
        actual_vx_mean=float(actual[:, 0].mean()), vx_bias=float((estimate[:, 0]-actual[:, 0]).mean()),
        vx_rmse=rms(estimate[:, 0]-actual[:, 0]), vy_rmse=rms(estimate[:, 1]-actual[:, 1]),
        vz_rmse=rms(estimate[:, 2]-actual[:, 2]),
        wrong_forward_fraction=float(((estimate[:, 0] > .1) & (actual[:, 0] < -.1)).mean()))


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'checkpoint', 'training-manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(2)
    manifest, arrays = load_bundle(args.bundle)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if sha != manifest['checkpoint_sha256'] or not manifest['policy_deterministic']:
        raise ValueError('Wrong checkpoint or stochastic actions')
    training = json.loads(torch.load(args.training_manifest, weights_only=True, map_location='cpu')['manifest_json'])
    if manifest['identity'] != training['identity']:
        raise ValueError('Wrong training manifest')
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    if state['amp_identity'] != manifest['identity']:
        raise ValueError('Wrong policy identity')
    cls, _ = cpu_ppo_classes(ROOT)
    cfg = manifest['environment']['env']
    with contextlib.redirect_stdout(io.StringIO()):
        policy = cls(cfg['num_single_obs']*cfg['short_frame_stack'], cfg['num_single_obs'],
                     cfg['num_privileged_obs'], cfg['num_actions'], **training['ppo_config']['policy'])
    policy.load_state_dict(state['model_state_dict'], strict=True); policy.eval()
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = dict(checkpoint_sha256=sha, bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        modes={}, limitation='Offline20Hz samples after66 true control ticks. Action parity is checked '
        'before interpreting ES. No hypothetical actions are applied. ES velocity is divided by the '
        'critic velocity normalization scale; parity/error are not causal proof or robustness acceptance.')
    for mode in manifest['modes']:
        rows = []
        for index in range(manifest['num_envs']):
            data = episode(arrays, mode, index); row = replay(data, manifest, policy)
            failed = bool(data['initial']['failure'] or data['failure'].any())
            end = float(data['time'][-1])
            summary = dict(env=index, failed=failed, observed_s=end, parity=row['parity'],
                fixed_2to4s=interval(row, 2., 4.), post2s=interval(row, 2., end+.01))
            if failed:
                summary.update(pre_4to2s=interval(row, end-4, end-2),
                               last2s=interval(row, end-2, end+.01))
            rows.append(summary)
            if index == 0 or failed:
                fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
                for dim, axis in enumerate(axes):
                    axis.plot(row['time'], row['actual'][:, dim], label='recorded body velocity')
                    axis.plot(row['time'], row['estimate'][:, dim], label='reconstructed ES', alpha=.8)
                    axis.set_ylabel(('vx', 'vy', 'vz')[dim]+' m/s'); axis.grid(alpha=.3)
                axes[0].legend(); axes[-1].set_xlabel('time s')
                fig.suptitle('%s env%d, action parity max error %.2g, %s' %
                    (mode, index, row['parity']['max_action_abs_error'], 'failed prefix' if failed else 'survived60s'))
                fig.tight_layout(); fig.savefig(args.output/('%s_env%d.png' % (mode, index)), dpi=130); plt.close(fig)
        means = {}
        for name in ('fixed_2to4s', 'post2s', 'pre_4to2s', 'last2s'):
            selected = [r[name] for r in rows if r.get(name) is not None]
            means[name] = dict(n=len(selected), mean={k: float(np.mean([r[k] for r in selected]))
                for k in selected[0] if k != 'n'}) if selected else dict(n=0)
        result['modes'][mode] = dict(episodes=rows, equal_initial_means=means)
    with (args.output/'estimator_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({mode: r['equal_initial_means'] for mode, r in result['modes'].items()}))


if __name__ == '__main__':
    main()
