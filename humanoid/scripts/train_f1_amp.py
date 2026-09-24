"""Gradmotion AMP smoke/formal entry, single scaled WALK_02, unchanged assets."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # Must precede any indirect torch import.
import torch
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.x1.x1_amp_config import X1AMPCfg, X1AMPCfgPPO
from humanoid.envs.x1.x1_amp_env import X1AMPEnv
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint, validate_cloud_smoke
from humanoid.amp.runner import AMPOnPolicyRunner
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import class_to_dict, update_cfg_from_args


def changed(before, after):
    return any(not torch.equal(before[k], after[k].detach().cpu()) for k in before)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--smoke-certificate")
    extra, remaining = parser.parse_known_args(); sys.argv = [sys.argv[0]]+remaining
    args = get_args()
    repo = Path(LEGGED_GYM_ROOT_DIR)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo)).decode().strip()
    if commit != extra.expected_commit or args.task != "f1_amp_walk02_s092":
        raise ValueError("Unapproved cloud checkout or task")
    if args.resume or args.load_run is not None or args.training_profile:
        raise ValueError("This AMP experiment must start from scratch without old profiles")
    experiment = ScaledExperiment(repo)
    budget = experiment.cfg[extra.mode]
    if (args.num_envs, args.max_iterations, args.seed) != (budget["num_envs"], budget["updates"], 5):
        raise ValueError("Unexpected experiment budget")
    fingerprint = implementation_fingerprint(repo)
    if extra.mode == "formal":
        if not extra.smoke_certificate:
            raise ValueError("A matching real-simulator smoke certificate is required")
        validate_cloud_smoke(experiment, json.loads(Path(extra.smoke_certificate).read_text()), fingerprint)
    cfg, train_cfg = X1AMPCfg(), X1AMPCfgPPO()
    cfg.seed = train_cfg.seed = 5
    cfg, train_cfg = update_cfg_from_args(cfg, train_cfg, args)
    task_registry.register(args.task, X1AMPEnv, cfg, train_cfg)
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    for field, col in (("lower", 0), ("upper", 1), ("velocity", 2)):
        actual = torch.as_tensor(props[field].copy())
        expected = torch.as_tensor(experiment.kinematics.limits[:, col], dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    log_dir = repo/"logs"/experiment.cfg["experiment"]/"exported_data"/(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_"+extra.mode)
    log_dir.mkdir(parents=True, exist_ok=False)
    config = {**class_to_dict(train_cfg), **class_to_dict(cfg)}
    runner = AMPOnPolicyRunner(env, config, experiment, str(log_dir), args.rl_device)
    if extra.mode == "smoke":
        # Deliberately exercise one terminal/reset path, not just no-reset frames.
        env.amp_force_reset_step = env.common_step_counter+37
    actor_before = {k: v.detach().cpu().clone() for k, v in runner.alg.actor_critic.state_dict().items()}
    d_before = {k: v.detach().cpu().clone() for k, v in runner.amp_trainer.discriminator.state_dict().items()}
    manifest = dict(identity=experiment.identity(), implementation_fingerprint=fingerprint, code_commit=commit,
        mode=extra.mode, num_envs=env.num_envs, updates=args.max_iterations,
        feature_spec=experiment.spec.description(), diagnostics=experiment.diagnostics,
        env_config=class_to_dict(cfg), ppo_config=class_to_dict(train_cfg),
        amp_reward=experiment.cfg["reward"], training_ready=False, dynamics_acceptance="not claimed",
        runtime=dict(physics_dt=env.sim_params.dt, control_dt=env.dt,
            dof_names=env.dof_names, dof_properties={k: props[k].tolist() for k in props.dtype.names},
            pd_p=env.p_gains.detach().cpu().tolist(), pd_d=env.d_gains.detach().cpu().tolist(),
            actor_history=cfg.env.frame_stack, critic_history=cfg.env.c_frame_stack))
    print("[f1-amp-start] "+json.dumps(manifest), flush=True)
    runner.learn(args.max_iterations, init_at_random_ep_len=False)
    tensors = list(runner.alg.actor_critic.state_dict().values())+list(runner.amp_trainer.discriminator.state_dict().values())
    certificate = dict(identity=experiment.identity(), implementation_fingerprint=fingerprint, code_commit=commit,
        num_envs=env.num_envs, updates=args.max_iterations, physics_dt_verified=True,
        body_frames_verified=env.amp_verified_frames >= 12 and env.amp_frame_check_max_error <= 5e-4,
        body_frame_max_error_m=env.amp_frame_check_max_error,
        reset_history_verified=runner.alg.reset_checks > 0 and runner.alg.warmup_excluded >= 9*env.num_envs,
        nonzero_style_reward=runner.alg.style_sum > 0, valid_windows=runner.alg.valid_windows,
        reset_checks=runner.alg.reset_checks, warmup_excluded=runner.alg.warmup_excluded,
        actor_updated=changed(actor_before, runner.alg.actor_critic.state_dict()),
        discriminator_updated=changed(d_before, runner.amp_trainer.discriminator.state_dict()),
        all_finite=all(bool(torch.isfinite(t).all()) for t in tensors), complete=True)
    if extra.mode == "smoke":
        validate_cloud_smoke(experiment, certificate, fingerprint)
    manifest["completion"] = certificate
    (log_dir/"amp_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save(dict(manifest_json=json.dumps(manifest), certificate_json=json.dumps(certificate)), log_dir/"model_amp_manifest.pt")
    print("[f1-amp-complete] "+json.dumps(certificate), flush=True)
    if runner.writer:
        runner.writer.flush()


if __name__ == "__main__":
    main()
