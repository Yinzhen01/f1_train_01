"""Independent deterministic Isaac Gym AMP rollout, before auto-resets.

Called in a separate process: no PPO/D updates, no shared training environment,
and no evaluation samples enter the replay buffer. Full states permit local
MuJoCo rendering and more detailed motion-quality analysis.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi, gymtorch  # Must precede torch.
import numpy as np
import torch
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.algo.ppo.actor_critic_dh import ActorCriticDH
from humanoid.envs.x1.x1_amp_config import X1AMPCfg, X1AMPCfgPPO
from humanoid.envs.x1.x1_amp_env import X1AMPEnv
from humanoid.envs.x1.x1_amp_recovery_config import X1AMPRecoveryCfg, X1AMPRecoveryCfgPPO
from humanoid.envs.x1.x1_amp_recovery_env import X1AMPRecoveryEnv
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_runtime_timing
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.recovery import FOOT_NAMES
from humanoid.amp.refinement import GROUPS, select_environment
from humanoid.amp.signal import SIGNAL_GROUPS, validate_signal
from humanoid.amp.horizon import HORIZON_GROUPS, validate_horizon
from humanoid.amp.contact import CONTACT_GROUPS, validate_contact
from humanoid.amp.progress import PROGRESS_GROUPS, validate_progress
from humanoid.amp.direction import DIRECTION_GROUPS, validate_direction
from humanoid.amp.sustain import SUSTAIN_GROUPS, validate_sustain
from humanoid.amp.evaluation import validate_evaluation_budget, independent_mode_seeds
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import class_to_dict, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(data, duration):
    valid = data["valid"]
    rows = []
    for index in range(valid.shape[1]):
        selected = valid[:, index]
        count = int(selected.sum())
        failures = data["failure"][selected, index]
        # Do not count the RSI-injected initial velocity as learned locomotion.
        sustained = selected & (data["time"] >= 2.)
        vx = data["base_lin_vel"][sustained, index, 0]
        q = data["dof_pos"][sustained, index]
        rows.append(dict(env=index, observed_s=count*.01,
            survived=bool(count >= round(duration*100) and not failures.any()),
            failure=bool(failures.any() or data.get("initial_failure", np.zeros(valid.shape[1], dtype=bool))[index]),
            post_2s_vx_mean=None if not len(vx) else float(vx.mean()),
            post_2s_vx_std=None if not len(vx) else float(vx.std()),
            joint_rom_rad=None if not len(q) else np.ptp(q, axis=0).tolist()))
    return dict(episodes=rows, survival_fraction=float(np.mean([r["survived"] for r in rows])),
                acceptance="not established; inspect full trajectory, style and rendered gait",
                dr_unlocked=False)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--experiment", choices=("baseline", "recovery", "recovery_static")+GROUPS+SIGNAL_GROUPS+HORIZON_GROUPS+CONTACT_GROUPS+PROGRESS_GROUPS+DIRECTION_GROUPS+SUSTAIN_GROUPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--extended-validation", action="store_true")
    extra, remaining = parser.parse_known_args(); sys.argv = [sys.argv[0]]+remaining
    args = get_args()
    repo = Path(LEGGED_GYM_ROOT_DIR)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo)).decode().strip()
    if commit != extra.expected_commit or sha(extra.checkpoint_file) != extra.checkpoint_sha256:
        raise ValueError("Evaluation checkout/checkpoint identity mismatch")
    if extra.output.exists() or extra.output.with_suffix(".json").exists():
        raise FileExistsError(extra.output)
    validate_evaluation_budget(args.num_envs, extra.duration, extra.extended_validation)
    cfg_name = "lafan_walk02_s092.json" if extra.experiment == "baseline" else "lafan_walk02_%s.json" % extra.experiment
    experiment = ScaledExperiment(repo, repo/"configs/amp"/cfg_name)
    if extra.experiment in SIGNAL_GROUPS:
        validate_signal(experiment)
    if extra.experiment in HORIZON_GROUPS:
        validate_horizon(experiment)
    if extra.experiment in CONTACT_GROUPS:
        validate_contact(experiment)
    if extra.experiment in PROGRESS_GROUPS:
        validate_progress(experiment)
    if extra.experiment in DIRECTION_GROUPS:
        validate_direction(experiment)
    if extra.experiment in SUSTAIN_GROUPS:
        validate_sustain(experiment)
    state = torch.load(str(extra.checkpoint_file), map_location="cpu", weights_only=True)
    if state["amp_identity"] != experiment.identity() or args.task != experiment.cfg["experiment"]:
        raise ValueError("Evaluation task/data identity mismatch")
    cfg, train_cfg, cls = select_environment(extra.experiment)
    cfg.seed = args.seed
    # Longer observation horizon, never motion-time rescaling or playback.
    cfg.env.episode_length_s = extra.duration+.1
    gate = assert_no_domain_randomization(cfg)
    task_registry.register(args.task, cls, cfg, train_cfg)
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    validate_runtime_timing(env.dt, env.sim_params.dt, cfg.control.decimation)
    if tuple(env.dof_names) != experiment.spec.joint_names:
        raise ValueError("Evaluation joint order mismatch")
    if extra.experiment != "baseline":
        env.configure_reference_initialization(experiment)
    policy = ActorCriticDH(env.num_short_obs, env.num_single_obs, env.num_privileged_obs,
                          env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()
    del state
    body_ids = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
                for name in experiment.spec.body_names]
    if min(body_ids) < 0:
        raise ValueError("Missing evaluation AMP bodies")
    expected_feet = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name) for name in FOOT_NAMES]
    if env.feet_indices.cpu().tolist() != expected_feet:
        raise ValueError("Evaluation foot order mismatch")
    original_reset = env._reset_dofs
    def fixed_reset(ids):
        env.dof_pos[ids] = env.default_dof_pos
        env.dof_vel[ids] = 0.
        ids32 = ids.to(torch.int32)
        env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state),
                                            gymtorch.unwrap_tensor(ids32), len(ids))
    original_termination = env.check_termination
    frames = []
    recording = False
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    def capture():
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        root = cpu(env.root_states); root[:, :3] -= cpu(env.env_origins)
        feet = cpu(env.rigid_state[:, env.feet_indices]); feet[:, :, :3] -= cpu(env.env_origins)[:, None, :]
        return dict(root_state=root, dof_pos=cpu(env.dof_pos), dof_vel=cpu(env.dof_vel),
            action=cpu(env.actions), torque=cpu(env.torques), base_lin_vel=cpu(env.base_lin_vel),
            base_ang_vel=cpu(env.base_ang_vel), foot_state=feet,
            foot_force=cpu(env.contact_forces[:, env.feet_indices]),
            key_positions_w=cpu(env.rigid_state[:, body_ids, :3]-env.env_origins[:, None, :]),
            valid=cpu(alive), failure=cpu(env.reset_buf.bool() & ~env.time_out_buf.bool()))
    def check_and_capture():
        original_termination()
        if recording:
            frames.append(capture())  # Always before automatic reset.
    env.check_termination = check_and_capture
    arrays, summaries = {}, {}
    modes = ("standing",) if extra.experiment == "baseline" else ("standing", "reference")
    mode_seeds = independent_mode_seeds(args.seed, extra.extended_validation)
    for mode in modes:
        if mode_seeds is not None:
            set_seed(mode_seeds[mode])
        recording = False
        if extra.experiment != "baseline":
            env.rsi_enabled = mode == "reference"
        env._reset_dofs = fixed_reset if mode == "standing" else original_reset
        alive[:] = True
        obs, _ = env.reset()
        initial = capture()
        alive &= ~env.reset_buf.bool()
        frames = []
        recording = True
        # Gym replaces some state buffers during step(). Use no_grad so those
        # buffers remain writable by the next mode's reset outside this block.
        with torch.no_grad():
            for step in range(round(extra.duration*100)):
                actions = policy.act_inference(obs)
                if not torch.isfinite(actions).all():
                    raise ValueError("Nonfinite evaluation policy actions")
                obs, _, _, dones, _ = env.step(actions)
                alive &= ~dones.bool()
                if not alive.any():
                    break
                if (step+1) % 500 == 0:
                    print("[amp-eval-progress] %s %.2fs alive=%d" % (mode, (step+1)*.01, int(alive.sum())), flush=True)
        recording = False
        data = {key: np.stack([frame[key] for frame in frames]) for key in initial}
        data["time"] = (np.arange(len(frames))+1)*.01
        data["initial_failure"] = initial["failure"]
        for key, value in data.items():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite captured rollout "+key)
            arrays[mode+"_"+key] = value
        for key, value in initial.items():
            arrays[mode+"_initial_"+key] = value
        summaries[mode] = summarize(data, extra.duration)
        print("[amp-eval-summary] "+json.dumps(dict(mode=mode, **summaries[mode])), flush=True)
    props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    manifest = dict(checkpoint_sha256=extra.checkpoint_sha256, code_commit=commit,
        identity=experiment.identity(), no_dr=gate, duration_s=extra.duration,
        num_envs=env.num_envs, fps=100, modes=list(modes), summaries=summaries,
        dof_names=env.dof_names, body_names=list(experiment.spec.body_names),
        foot_names=list(FOOT_NAMES),
        urdf_lf_sha256=experiment.kinematics.lf_sha256, environment=class_to_dict(cfg),
        runtime=dict(dof_properties={k: props[k].tolist() for k in props.dtype.names},
            p_gains=env.p_gains.cpu().tolist(), d_gains=env.d_gains.cpu().tolist(),
            control_dt=env.dt, physics_dt=env.sim_params.dt),
        policy_deterministic=True, capture="pre-reset; valid stops after first failure",
        effectiveness_verified=False, dr_unlocked=False)
    if extra.extended_validation:
        manifest.update(evaluation_protocol="fixed60_independent_mode_seeds", mode_seeds=mode_seeds)
    packed = io.BytesIO()
    np.savez_compressed(packed, **arrays)
    extra.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(npz_bytes=packed.getvalue(), manifest_json=json.dumps(manifest)), extra.output)
    extra.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env.gym.destroy_sim(env.sim)
    print("[amp-eval-complete] "+str(extra.output), flush=True)


if __name__ == "__main__":
    main()
