"""Record deterministic GMR policy dynamics in Isaac Gym, before automatic resets.

No training updates, fixed speed command, time rescaling, or motion wrapping.
The resulting NPZ is rendered by MuJoCo, not simulated a second time there.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

from isaacgym import gymapi  # Must precede torch imports.
import numpy as np
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.algo.ppo.actor_critic_dh import ActorCriticDH
from humanoid.gmr_rollout_metrics import summarize_rollout
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import class_to_dict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate_checkpoint(path, expected_sha, checkpoint):
    explicit = Path(path)
    if explicit.is_file():
        candidates = [explicit]
    else:
        candidates = []
        for root in (Path(LEGGED_GYM_ROOT_DIR), Path('/personal'), Path('/workspace')):
            if root.is_dir():
                candidates.extend(root.rglob('model_%d*.pt' % checkpoint))
    for candidate in sorted(set(candidates)):
        if sha(candidate) == expected_sha.lower():
            return candidate
    raise FileNotFoundError('No mounted checkpoint matches the required SHA256; candidates=%s' %
                            [str(p) for p in sorted(set(candidates))])


def capture(env, valid=True):
    def array(value):
        return value[0].detach().cpu().numpy().copy()
    root = array(env.root_states)
    root[:3] -= array(env.env_origins)
    contact_failure = torch.any(torch.linalg.vector_norm(
        env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1., dim=1)
    tilt_failure = torch.any(torch.abs(env.base_euler_xyz[:, :2]) > 1.5, dim=1)
    failure = contact_failure | tilt_failure | (env.root_states[:, 2] < .3)
    return {
        'time': float(env.motion_time()[0]), 'root_state': root,
        'dof_pos': array(env.dof_pos), 'dof_vel': array(env.dof_vel),
        'torque': array(env.torques), 'action': array(env.actions),
        'foot_state': array(env.rigid_state[:, env.feet_indices]),
        'foot_force': array(env.contact_forces[:, env.feet_indices]),
        'sole_position': array(env._sole_positions()) - array(env.env_origins),
        'sole_velocity': array(env._sole_velocities()),
        'reference_dof_pos': array(env.reference['dof_pos']),
        'reference_root_pos': array(env.reference['root_pos']),
        'reference_root_quat': array(env.reference['root_quat']),
        'reference_contact': array(env.reference['foot_contact']),
        'physical_failure': bool(failure[0]) if valid else False,
        'contact_failure': bool(contact_failure[0]) if valid else False,
        'tilt_failure': bool(tilt_failure[0]) if valid else False,
        'clip_finished': bool(env.clip_finished[0]) if valid else False,
        'foot_diagnostics_valid': bool(valid),
    }


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--checkpoint-file', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--source-task', required=True)
    parser.add_argument('--episodes', type=int, default=3)
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    if args.task != 'x1_gmr_clip' or args.num_envs != 1 or args.checkpoint != 5000:
        raise ValueError('This entry point requires x1_gmr_clip, one environment, checkpoint 5000')
    if not 1 <= extra.episodes <= 10:
        raise ValueError('episodes must be between 1 and 10')
    checkpoint = locate_checkpoint(extra.checkpoint_file, extra.checkpoint_sha256, args.checkpoint)
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    # The training task is already nominal/no-DR. Do not rewrite its dynamics.
    cfg.env.num_envs = 1
    cfg.seed = args.seed if args.seed is not None else 5
    cfg.motion_reference.random_start = False
    if cfg.noise.add_noise or cfg.env.use_ref_actions or cfg.env.episode_length_s != 4.6:
        raise ValueError('Unexpected training configuration')
    if cfg.terrain.mesh_type != 'plane' or cfg.domain_rand.randomize_joint_armature:
        raise ValueError('Nominal training dynamics required')
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    policy = ActorCriticDH(env.num_short_obs, env.num_single_obs, env.num_privileged_obs,
                          env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
    state = torch.load(str(checkpoint), map_location=env.device, weights_only=False)
    # This runner names the final file by update count, but stores self.it
    # (zero-based) inside it: model_5000.pt therefore contains iter=4999.
    stored_iteration = int(state['iter'])
    if stored_iteration != args.checkpoint - 1:
        raise ValueError('Checkpoint iteration mismatch')
    policy.load_state_dict(state['model_state_dict'], strict=True)
    policy.eval()
    del state
    print('[gmr-inference] exact checkpoint loaded; deterministic=True from_time=0; no learning', flush=True)
    output = Path(LEGGED_GYM_ROOT_DIR) / 'logs' / train_cfg.runner.experiment_name / 'exported_data' / 'gmr_inference'
    output.mkdir(parents=True, exist_ok=True)
    original_termination = env.check_termination
    samples = []

    def capture_before_reset():
        original_termination()
        samples.append(capture(env))

    # Read-only observer, invoked after physics and target refresh, before reset.
    env.check_termination = capture_before_reset
    recordings, reports = {}, []
    for episode in range(extra.episodes):
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        env.clip_finished.zero_()
        env.torques.zero_()
        env.compute_observations()
        obs = env.get_observations()
        samples = [capture(env, valid=False)]
        # The environment replaces persistent tensors during stepping and resets
        # them between episodes. no_grad permits those out-of-context mutations.
        with torch.no_grad():
            for step in range(int(round(env.motion.duration / env.dt)) + 2):
                actions = policy.act_inference(obs)
                if not torch.isfinite(actions).all():
                    raise ValueError('Nonfinite policy action')
                obs, _, _, dones, _ = env.step(actions)
                if bool(dones[0]):
                    break
            else:
                raise RuntimeError('Finite clip did not terminate')
        data = {key: np.asarray([s[key] for s in samples]) for key in samples[0]}
        for key in ('root_state', 'dof_pos', 'torque', 'action'):
            if not np.isfinite(data[key]).all():
                raise ValueError('Nonfinite recorded ' + key)
        if not np.all(np.diff(data['time']) > 0):
            raise ValueError('Recorded timeline contains a reset or wrap')
        report = summarize_rollout(data, env.motion.duration)
        reports.append(report)
        recordings.update({'episode_%d_%s' % (episode, key): value for key, value in data.items()})
        print('[gmr-inference] episode=%d %s' % (episode, json.dumps(report)), flush=True)
    recordings['joint_names'] = np.asarray(env.dof_names)
    recordings['foot_names'] = np.asarray(env.motion.metadata['foot_names'])
    recordings['sole_local'] = env.motion.sole_local.detach().cpu().numpy()
    recordings['sole_normal_local'] = env.motion.sole_normal_local.detach().cpu().numpy()
    recordings['env_origin'] = env.env_origins[0].detach().cpu().numpy()
    props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    asset_path = Path(cfg.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', LEGGED_GYM_ROOT_DIR))
    metadata = {
        'source_task': extra.source_task, 'checkpoint': args.checkpoint,
        'stored_zero_based_iteration': stored_iteration,
        'checkpoint_sha256': sha(checkpoint),
        'inference_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LEGGED_GYM_ROOT_DIR).decode().strip(),
        'training_commit': '181f8034e62105f4138b90ebb560f63c0d1640a4',
        'urdf_lf_sha256': hashlib.sha256(asset_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'motion_sha256': cfg.motion_reference.sha256, 'control_dt': env.dt,
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'dof_names': env.dof_names, 'dof_properties': {key: props[key].tolist() for key in props.dtype.names},
        'pd_p': env.p_gains.detach().cpu().tolist(), 'pd_d': env.d_gains.detach().cpu().tolist(),
        'effective_torque_limits': env.torque_limits.detach().cpu().tolist(),
        'action_scale': cfg.control.action_scale, 'config': class_to_dict(cfg),
        'episodes': reports,
        'protocol': 'Deterministic policy at t=0 with reset observation history; record post-physics before auto-reset. MuJoCo only visualizes these Isaac Gym states. No Sim2Sim claim.',
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **recordings)
    torch.save({'npz_bytes': buffer.getvalue(), 'manifest_json': json.dumps(metadata)}, output / 'model_gmr_rollout.pt')
    print('[gmr-inference] artifact ready: model_gmr_rollout.pt', flush=True)
    env.gym.destroy_sim(env.sim)
    time.sleep(30)  # Give SDK its usual artifact discovery window.
    print('[gmr-inference] inference completed', flush=True)


if __name__ == '__main__':
    main()
