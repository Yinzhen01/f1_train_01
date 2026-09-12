"""Hash-gated model8000 ablations, followed by paired 1kHz deterministic probes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # Must precede torch, including indirect imports.
import numpy as np
import torch
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.gmr_cycle import GROUPS
from humanoid.gmr_accel_finetune import (SOURCE_TASK, SOURCE_SHA256, SOURCE_REFERENCE_SHA256,
    SOURCE_URDF_SHA256, SOURCE_COMPLETED_UPDATES, locate_verified_checkpoint, sha256, warm_start_runner)
from humanoid.gmr_rollout_metrics import summarize_rollout
from humanoid.scripts.record_gmr_policy import capture
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import class_to_dict


def validate_request(args, extra, commit):
    if commit != extra.expected_commit:
        raise ValueError("Cloud checkout differs from the approved experiment commit")
    if args.task not in GROUPS or not args.resume or args.checkpoint != 8000:
        raise ValueError("Requires cycle group and explicit model8000 warm start")
    if args.training_profile or args.load_run is not None:
        raise ValueError("No implicit profile/load_run source")
    if extra.source_task != SOURCE_TASK or extra.checkpoint_sha256.lower() != SOURCE_SHA256:
        raise ValueError("Incorrect original checkpoint identity")
    budget = {"smoke": (5, 512, 20), "formal": (5, 4096, 1000)}
    if extra.mode not in budget or (args.seed, args.num_envs, args.max_iterations) != budget[extra.mode]:
        raise ValueError("Requires exact seed/environment/update budget")


def validate_config(cfg, baseline, group):
    current = class_to_dict(cfg)
    expected = dict(boundary_s=.02, ramp_s=.04, acceleration_floor=.25,
        acceleration_scale_rad_s2=100., joint_weights=[1., 1., 1., 1., 1.5, 2.],
        stance_force_body_weights=1.5, swing_force_body_weights=.05)
    if current.pop('cycle') != expected:
        raise ValueError("Unapproved cycle settings")
    acc, impact, contact = GROUPS[group]
    if current['swing'].pop('contact_phase_only') != contact:
        raise ValueError("Incorrect contact ablation")
    scales = current['rewards']['scales']
    if (scales.pop('gmr_cycle_acc'), scales.pop('gmr_cycle_impact')) != (acc, impact):
        raise ValueError("Incorrect reward ablation")
    if current != class_to_dict(baseline):
        raise ValueError("Old rewards, reference, observation or dynamics changed")


@torch.inference_mode()
def paired_probe(env, policy, source, final, manifest, output):
    """No learning. Same env0 in the training-sized simulator, two exact policies.

    This is NOT the previous single-environment evaluation protocol. Retain full
    100Hz pre-reset states and 1kHz substeps to enable a common-protocol comparison.
    Probing only happens after learning, so it cannot change the training RNG.
    The PPO rollout creates persistent inference tensors. Resets as well as
    stepping must remain inside inference_mode when reusing that simulator.
    """
    original_termination = env.check_termination
    original_random_start = env.cfg.motion_reference.random_start
    policy.eval()
    env.cfg.motion_reference.random_start = False
    all_data, reports = {}, {}
    samples, traces = [], []

    def observe():
        original_termination()
        samples.append(capture(env))
        traces.append({k: v.detach().cpu().numpy().copy() for k, v in env.cycle_trace.items()})

    env.check_termination = observe
    try:
        for label, state in (('source8000', source), ('trained', final)):
            policy.load_state_dict(state, strict=True)
            env.reset_idx(torch.arange(env.num_envs, device=env.device))
            env.clip_finished.zero_()
            env.torques.zero_()
            env.compute_observations()
            obs = env.get_observations()
            samples, traces = [capture(env, valid=False)], []
            with torch.no_grad():
                for _ in range(int(round(env.motion.duration / env.dt)) + 2):
                    actions = policy.act_inference(obs)
                    if not torch.isfinite(actions).all():
                        raise ValueError('Nonfinite probe action')
                    obs, _, _, dones, _ = env.step(actions)
                    if bool(dones[0]):
                        break
                else:
                    raise RuntimeError('Finite probe did not terminate')
            data = {k: np.asarray([s[k] for s in samples]) for k in samples[0]}
            physics = {k: np.concatenate([s[k] for s in traces], axis=0) for k in traces[0]}
            if not np.all(np.diff(data['time']) > 0) or not np.all(np.diff(physics['time'][:, 0]) > 0):
                raise ValueError('Probe timeline contains reset/wrap')
            if not all(np.isfinite(v).all() for v in list(data.values()) + list(physics.values())):
                raise ValueError('Nonfinite probe states')
            report = summarize_rollout(data, env.motion.duration)
            valid = (physics['valid'][:, 0] > .5) & (physics['time'][:, 0] >= .1)
            if not np.any(valid):
                raise ValueError('Probe has no valid post-reset samples')
            report.update(acc_rms_1khz=float(np.sqrt(np.mean(physics['acc'][valid] ** 2))),
                acc_peak_1khz=float(np.max(np.abs(physics['acc'][valid]))),
                force_peak_n_1khz=float(np.max(physics['force_z'][valid])))
            all_data.update({label + '_control_' + k: v for k, v in data.items()})
            all_data.update({label + '_physics_' + k: v for k, v in physics.items()})
            reports[label] = report
            print('[gmr-cycle-probe] ' + label + ' ' + json.dumps(report), flush=True)
    finally:
        policy.load_state_dict(final, strict=True)
        env.check_termination = original_termination
        env.cfg.motion_reference.random_start = original_random_start
    all_data['joint_names'] = np.asarray(env.dof_names)
    all_data['foot_names'] = np.asarray(env.motion.metadata['foot_names'])
    all_data['sole_local'] = env.motion.sole_local.detach().cpu().numpy()
    all_data['sole_normal_local'] = env.motion.sole_normal_local.detach().cpu().numpy()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **all_data)
    probe_manifest = dict(manifest, reports=reports,
        protocol='Post-training deterministic t0, env0 of training-sized plane simulation; source8000 and trained weights; no learning; pre-reset 100Hz states plus 1kHz refreshed DOF/contact; not single-env robustness or Sim2Sim.')
    torch.save(dict(npz_bytes=buffer.getvalue(), manifest_json=json.dumps(probe_manifest)), output)
    return reports


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--checkpoint-file', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--source-task', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--mode', required=True, choices=('smoke', 'formal'))
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LEGGED_GYM_ROOT_DIR).decode().strip()
    validate_request(args, extra, commit)
    checkpoint = locate_verified_checkpoint(extra.checkpoint_file, [LEGGED_GYM_ROOT_DIR, '/personal', '/workspace'])
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    baseline = X1GMRSwingCfg()
    baseline.seed = train_cfg.seed
    validate_config(cfg, baseline, args.task)
    reference = cfg.motion_reference.file.replace('{LEGGED_GYM_ROOT_DIR}', LEGGED_GYM_ROOT_DIR)
    urdf = Path(cfg.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', LEGGED_GYM_ROOT_DIR))
    if cfg.motion_reference.sha256 != SOURCE_REFERENCE_SHA256 or sha256(reference) != SOURCE_REFERENCE_SHA256:
        raise ValueError('Reference differs from source policy')
    if hashlib.sha256(urdf.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != SOURCE_URDF_SHA256:
        raise ValueError('Training URDF differs from source policy')
    args.resume = False
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg)
    identity = warm_start_runner(runner, checkpoint)
    if runner.alg.learning_rate != 1e-5 or runner.alg.schedule != 'fixed' or env.num_envs != args.num_envs:
        raise ValueError('Effective training budget/learning rate mismatch')
    for key, expected in zip(('gmr_cycle_acc', 'gmr_cycle_impact'), GROUPS[args.task][:2]):
        if not np.isclose(env.reward_scales.get(key, 0.) / env.dt, expected, atol=1e-12):
            raise ValueError('Effective reward differs from ablation')
    final_updates = SOURCE_COMPLETED_UPDATES + args.max_iterations
    identity.update(code_commit=commit, task=args.task, purpose=extra.mode,
        reference_sha256=SOURCE_REFERENCE_SHA256, urdf_lf_sha256=SOURCE_URDF_SHA256,
        additional_updates=args.max_iterations, final_completed_updates=final_updates,
        num_envs=env.num_envs, seed=args.seed, learning_rate=runner.alg.learning_rate,
        schedule=runner.alg.schedule, physics_dt=env.cycle_h, physics_substeps=env.sim_params.substeps,
        control_dt=env.dt, body_weight_n=env.cycle_body_weight,
        reward_scales_before_dt={k: v / env.dt for k, v in env.reward_scales.items()})
    print('[gmr-cycle-resume] ' + json.dumps(identity, sort_keys=True), flush=True)
    props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    manifest = dict(identity=identity, env_config=class_to_dict(cfg), ppo_config=class_to_dict(train_cfg),
        runtime=dict(dof_names=env.dof_names, dof_properties={k: props[k].tolist() for k in props.dtype.names},
            pd_p=env.p_gains.detach().cpu().tolist(), pd_d=env.d_gains.detach().cpu().tolist(),
            torque_limits=env.torque_limits.detach().cpu().tolist(), body_weight_n=env.cycle_body_weight))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / 'cycle_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    final_path = Path(log_dir) / ('model_%d.pt' % final_updates)
    final = torch.load(str(final_path), map_location=env.device, weights_only=True)
    if final['iter'] != final_updates - 1 or not all(torch.isfinite(v).all() for v in final['model_state_dict'].values()):
        raise ValueError('Invalid final checkpoint')
    manifest['final_checkpoint_sha256'] = sha256(final_path)
    manifest['final_cycle_diagnostics'] = {k: float(v) for k, v in env.cycle_diagnostics.items()}
    if not all(np.isfinite(v) for v in manifest['final_cycle_diagnostics'].values()):
        raise ValueError('Nonfinite cycle diagnostics')
    source = torch.load(str(checkpoint), map_location=env.device, weights_only=True)['model_state_dict']
    manifest['probe_reports'] = paired_probe(env, runner.alg.actor_critic, source, final['model_state_dict'],
        manifest, Path(log_dir) / 'model_cycle_probe.pt')
    torch.save(dict(manifest_json=json.dumps(manifest)), Path(log_dir) / 'model_cycle_manifest.pt')
    print('[gmr-cycle] ' + extra.mode + ' completed', flush=True)


if __name__ == '__main__':
    main()
