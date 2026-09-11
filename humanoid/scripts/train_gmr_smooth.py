"""Registered Gradmotion entry point; refuses a missing/wrong warm-start source."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # Must precede torch imports.
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.gmr_finetune import (SOURCE_TASK, SOURCE_SHA256, SOURCE_REFERENCE_SHA256,
                                  locate_verified_checkpoint, sha256, warm_start_runner)
from humanoid.utils import get_args, task_registry


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-task", required=True)
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    if args.task != "x1_gmr_smooth" or not args.resume or args.checkpoint != 5000:
        raise ValueError("Requires x1_gmr_smooth --resume --checkpoint=5000")
    if args.training_profile or args.load_run is not None:
        raise ValueError("Use only the explicitly verified checkpoint, not a profile/load_run")
    if extra.source_task != SOURCE_TASK or extra.checkpoint_sha256.lower() != SOURCE_SHA256:
        raise ValueError("Unapproved fine-tuning source identity")
    checkpoint = locate_verified_checkpoint(extra.checkpoint_file,
                                            [LEGGED_GYM_ROOT_DIR, "/personal", "/workspace"])
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    reference = cfg.motion_reference.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    if cfg.motion_reference.sha256 != SOURCE_REFERENCE_SHA256 or sha256(reference) != SOURCE_REFERENCE_SHA256:
        raise ValueError("Reference differs from the source policy")
    additional = args.max_iterations if args.max_iterations is not None else train_cfg.runner.max_iterations
    if not 1 <= additional <= 2000:
        raise ValueError("This continuation is bounded to 1..2000 additional PPO updates")
    # Bypass only the generic directory-based loader; warm_start_runner below is
    # mandatory and hash-gated, and learn is unreachable if it fails.
    args.resume = False
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg)
    identity = warm_start_runner(runner, checkpoint)
    identity.update(code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LEGGED_GYM_ROOT_DIR).decode().strip(),
                    reference_sha256=SOURCE_REFERENCE_SHA256, additional_updates=additional,
                    final_completed_updates=5000 + additional, num_envs=env.num_envs,
                    learning_rate=runner.alg.learning_rate, schedule=runner.alg.schedule)
    print("[gmr-smooth-resume] " + json.dumps(identity, sort_keys=True), flush=True)
    runner.learn(num_learning_iterations=additional, init_at_random_ep_len=False)
    print("[gmr-smooth] training completed", flush=True)


if __name__ == "__main__":
    main()
