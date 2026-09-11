"""Registered, bounded, single-variable acceleration sweep from verified model8000."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # Must precede torch.
import torch
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.gmr_accel_finetune import (SOURCE_TASK, SOURCE_SHA256, SOURCE_REFERENCE_SHA256,
    SOURCE_URDF_SHA256, SOURCE_COMPLETED_UPDATES, GROUP_WEIGHTS,
    locate_verified_checkpoint, sha256, warm_start_runner)
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import class_to_dict


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--expected-commit", required=True)
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LEGGED_GYM_ROOT_DIR).decode().strip()
    if commit != extra.expected_commit:
        raise ValueError("Cloud checkout differs from the approved experiment commit")
    if args.task not in GROUP_WEIGHTS or not args.resume or args.checkpoint != 8000:
        raise ValueError("Requires acceleration group --resume --checkpoint=8000")
    if args.training_profile or args.load_run is not None:
        raise ValueError("Use only explicitly verified checkpoint, not profile/load_run")
    if extra.source_task != SOURCE_TASK or extra.checkpoint_sha256.lower() != SOURCE_SHA256:
        raise ValueError("Unapproved fine-tuning source identity")
    if args.seed != 5 or args.num_envs not in (512, 4096):
        raise ValueError("Sweep requires seed 5 and 512 smoke / 4096 formal environments")
    checkpoint = locate_verified_checkpoint(extra.checkpoint_file, [LEGGED_GYM_ROOT_DIR, "/personal", "/workspace"])
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    reference = cfg.motion_reference.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    urdf = Path(cfg.asset.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR))
    if cfg.motion_reference.sha256 != SOURCE_REFERENCE_SHA256 or sha256(reference) != SOURCE_REFERENCE_SHA256:
        raise ValueError("Reference differs from source policy")
    if hashlib.sha256(urdf.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != SOURCE_URDF_SHA256:
        raise ValueError("Training URDF differs from source policy")
    if cfg.rewards.scales.dof_acc != GROUP_WEIGHTS[args.task]:
        raise ValueError("Acceleration group/weight mismatch")
    additional = args.max_iterations if args.max_iterations is not None else train_cfg.runner.max_iterations
    if (args.num_envs, additional) not in ((512, 20), (4096, 1000)):
        raise ValueError("Only 512x20 smoke or 4096x1000 formal sweep is authorized")
    args.resume = False  # Replace directory loader with mandatory strict warm start.
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg)
    identity = warm_start_runner(runner, checkpoint)
    if runner.alg.learning_rate != 1e-5 or runner.alg.schedule != "fixed":
        raise ValueError("Sweep requires identical fixed 1e-5 learning rate")
    identity.update(code_commit=commit, task=args.task, dof_acc_weight=GROUP_WEIGHTS[args.task],
                    reference_sha256=SOURCE_REFERENCE_SHA256, urdf_lf_sha256=SOURCE_URDF_SHA256,
                    additional_updates=additional, final_completed_updates=SOURCE_COMPLETED_UPDATES + additional,
                    num_envs=env.num_envs, seed=args.seed, learning_rate=runner.alg.learning_rate,
                    schedule=runner.alg.schedule, reward_scales_before_dt={k: v/env.dt for k,v in env.reward_scales.items()})
    print("[gmr-accel-resume] " + json.dumps(identity, sort_keys=True), flush=True)
    manifest = dict(identity=identity, env_config=class_to_dict(cfg), ppo_config=class_to_dict(train_cfg))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / "accel_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runner.learn(num_learning_iterations=additional, init_at_random_ep_len=False)
    torch.save({"manifest_json": json.dumps(manifest)}, Path(log_dir) / "model_accel_manifest.pt")
    print("[gmr-accel] training completed", flush=True)


if __name__ == "__main__":
    main()
