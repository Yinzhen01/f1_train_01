"""Shared strict warm start; direct execution remains a 512 x 20 smoke only."""
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
    SOURCE_URDF_SHA256, SOURCE_COMPLETED_UPDATES, locate_verified_checkpoint, sha256, warm_start_runner)
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import class_to_dict


def validate_request(args, extra, commit, mode="smoke"):
    if commit != extra.expected_commit:
        raise ValueError("Cloud checkout differs from the approved experiment commit")
    if args.task != "x1_gmr_phase_accel" or not args.resume or args.checkpoint != 8000:
        raise ValueError("Requires phase experiment --resume --checkpoint=8000")
    if args.training_profile or args.load_run is not None:
        raise ValueError("Use only explicitly verified checkpoint, not profile/load_run")
    if extra.source_task != SOURCE_TASK or extra.checkpoint_sha256.lower() != SOURCE_SHA256:
        raise ValueError("Unapproved phase-training source identity")
    budgets = {"smoke": (5, 512, 20), "formal": (5, 4096, 1000)}
    if mode not in budgets or (args.seed, args.num_envs, args.max_iterations) != budgets[mode]:
        raise ValueError("Requires exact seed/environment/update budget for the selected entry")


def validate_config(cfg, baseline):
    # Fail before constructing a simulator if anything outside this experiment
    # changes. Reference/URDF bytes and effective physics are checked separately.
    current = class_to_dict(cfg)
    if current.pop("phase_accel") != {"scale_rad_s2": 100.}:
        raise ValueError("Unapproved phase acceleration normalization")
    if current["swing"].pop("contact_phase_only") is not True:
        raise ValueError("Swing contact height-gate gap must be closed")
    if current["rewards"]["scales"].pop("gmr_phase_acc") != -.02:
        raise ValueError("Unapproved phase acceleration weight")
    if current != class_to_dict(baseline):
        raise ValueError("An old reward, reference, observation or dynamics setting changed")


def main(mode="smoke"):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--expected-commit", required=True)
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LEGGED_GYM_ROOT_DIR).decode().strip()
    validate_request(args, extra, commit, mode)
    additional = args.max_iterations
    checkpoint = locate_verified_checkpoint(extra.checkpoint_file, [LEGGED_GYM_ROOT_DIR, "/personal", "/workspace"])
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    baseline = X1GMRSwingCfg()
    baseline.seed = train_cfg.seed  # TaskRegistry.get_cfgs adds this field to cfg.
    validate_config(cfg, baseline)
    reference = cfg.motion_reference.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    urdf = Path(cfg.asset.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR))
    if cfg.motion_reference.sha256 != SOURCE_REFERENCE_SHA256 or sha256(reference) != SOURCE_REFERENCE_SHA256:
        raise ValueError("Reference differs from source policy")
    if hashlib.sha256(urdf.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != SOURCE_URDF_SHA256:
        raise ValueError("Training URDF differs from source policy")
    args.resume = False  # Bypass directory loader, NOT the mandatory warm start below.
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg)
    identity = warm_start_runner(runner, checkpoint)
    if runner.alg.learning_rate != 1e-5 or runner.alg.schedule != "fixed":
        raise ValueError("Phase training requires fixed 1e-5 learning rate")
    if env.num_envs != args.num_envs:
        raise ValueError("Effective environment count differs from the approved budget")
    identity.update(code_commit=commit, task=args.task, purpose="smoke_only" if mode == "smoke" else "formal",
                    reference_sha256=SOURCE_REFERENCE_SHA256, urdf_lf_sha256=SOURCE_URDF_SHA256,
                    additional_updates=additional, final_completed_updates=SOURCE_COMPLETED_UPDATES + additional,
                    num_envs=env.num_envs, seed=args.seed, learning_rate=runner.alg.learning_rate,
                    schedule=runner.alg.schedule, physics_dt=env.phase_physics_dt,
                    physics_substeps=env.sim_params.substeps, control_dt=env.dt,
                    reward_scales_before_dt={k: v / env.dt for k, v in env.reward_scales.items()})
    print("[gmr-phase-resume] " + json.dumps(identity, sort_keys=True), flush=True)
    manifest = dict(identity=identity, env_config=class_to_dict(cfg), ppo_config=class_to_dict(train_cfg))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / ("phase_" + mode + "_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runner.learn(num_learning_iterations=additional, init_at_random_ep_len=False)
    if not all(torch.isfinite(v).all() for v in runner.alg.actor_critic.state_dict().values()):
        raise ValueError("Nonfinite final phase model")
    manifest["final_phase_diagnostics"] = {k: float(v) for k, v in env.phase_acc_diagnostics.items()}
    if not all(torch.isfinite(torch.tensor(v)) for v in manifest["final_phase_diagnostics"].values()):
        raise ValueError("Nonfinite final phase diagnostics")
    torch.save({"manifest_json": json.dumps(manifest)}, Path(log_dir) / ("model_phase_" + mode + "_manifest.pt"))
    print("[gmr-phase] " + mode + " completed", flush=True)


if __name__ == "__main__":
    main()
