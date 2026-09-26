"""Gradmotion AMP smoke/formal entry, single scaled WALK_02, unchanged assets."""
import argparse
import hashlib
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
from humanoid.envs.x1.x1_amp_recovery_config import X1AMPRecoveryCfg, X1AMPRecoveryCfgPPO
from humanoid.envs.x1.x1_amp_recovery_env import X1AMPRecoveryEnv
from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint, validate_cloud_smoke
from humanoid.amp.runner import AMPOnPolicyRunner
from humanoid.amp.learnability import assert_no_domain_randomization, validate_learnability_budget
from humanoid.amp.refinement import GROUPS, validate_refinement, select_environment, locate_source, warm_start
from humanoid.amp.interrupted import interrupted_source, restart_budget, recover_interrupted
from humanoid.amp.signal import SIGNAL_GROUPS, validate_signal, warm_start_signal
from humanoid.amp.horizon import (HORIZON_GROUPS, validate_horizon, warm_start_horizon,
                                 HorizonProbe, validate_horizon_probe)
from humanoid.amp.contact import CONTACT_GROUPS, validate_contact, warm_start_contact, validate_contact_diagnostics
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import class_to_dict, update_cfg_from_args


def changed(before, after):
    return any(not torch.equal(before[k], after[k].detach().cpu()) for k in before)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--smoke-certificate")
    parser.add_argument("--recover-interrupted", action="store_true")
    parser.add_argument("--experiment", choices=("baseline", "recovery", "recovery_static")+GROUPS+SIGNAL_GROUPS+HORIZON_GROUPS+CONTACT_GROUPS, default="recovery")
    extra, remaining = parser.parse_known_args(); sys.argv = [sys.argv[0]]+remaining
    args = get_args()
    repo = Path(LEGGED_GYM_ROOT_DIR)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo)).decode().strip()
    config_name = "lafan_walk02_s092.json" if extra.experiment == "baseline" else "lafan_walk02_%s.json" % extra.experiment
    experiment = ScaledExperiment(repo, repo/"configs/amp"/config_name)
    if commit != extra.expected_commit or args.task != experiment.cfg["experiment"]:
        raise ValueError("Unapproved cloud checkout or task")
    refining = extra.experiment in GROUPS
    signaling = extra.experiment in SIGNAL_GROUPS
    horizoning = extra.experiment in HORIZON_GROUPS
    contacting = extra.experiment in CONTACT_GROUPS
    resuming = refining or signaling or horizoning or contacting
    if extra.recover_interrupted and not refining:
        raise ValueError("Interrupted recovery is restricted to the matched refinement groups")
    if args.load_run is not None or args.training_profile or (args.resume and not resuming):
        raise ValueError("This AMP experiment must start from scratch without old profiles")
    if contacting:
        validate_contact(experiment)
        if not args.resume or args.checkpoint != 1750:
            raise ValueError('Contact experiments require exact long1750 source')
    elif horizoning:
        validate_horizon(experiment)
        if not args.resume or args.checkpoint != 1500:
            raise ValueError('Horizon experiments require explicit bridge1500 source')
    elif signaling:
        validate_signal(experiment)
        if not args.resume or args.checkpoint != 1250:
            raise ValueError("Style-signal experiments require explicit smooth1250 source")
    elif refining:
        validate_refinement(experiment.cfg)
        if not args.resume or args.checkpoint != (1250 if extra.recover_interrupted else 1000):
            raise ValueError("Matched refinement requires the approved explicit recovery checkpoint")
    elif extra.experiment != "baseline":
        validate_learnability_budget(experiment.cfg)
    budget = restart_budget(experiment.cfg, extra.mode) if extra.recover_interrupted else experiment.cfg[extra.mode]
    if (args.num_envs, args.max_iterations, args.seed) != (budget["num_envs"], budget["updates"], 5):
        raise ValueError("Unexpected experiment budget")
    fingerprint = implementation_fingerprint(repo)
    if extra.mode == "formal":
        if not extra.smoke_certificate:
            raise ValueError("A matching real-simulator smoke certificate is required")
        validate_cloud_smoke(experiment, json.loads(Path(extra.smoke_certificate).read_text()), fingerprint,
                             interrupted=extra.recover_interrupted)
    cfg, train_cfg, env_class = select_environment(extra.experiment)
    if refining:
        train_cfg.algorithm.learning_rate = experiment.cfg["refinement"]["learning_rate"]
    elif signaling:
        train_cfg.algorithm.learning_rate = experiment.cfg["signal"]["learning_rate"]
    elif horizoning:
        train_cfg.algorithm.learning_rate = experiment.cfg['horizon']['learning_rate']
    elif contacting:
        train_cfg.algorithm.learning_rate = experiment.cfg['contact_refinement']['learning_rate']
    cfg.seed = train_cfg.seed = 5
    cfg, train_cfg = update_cfg_from_args(cfg, train_cfg, args)
    train_cfg.runner.experiment_name = experiment.cfg["experiment"]
    no_dr_gate = assert_no_domain_randomization(cfg)
    task_registry.register(args.task, env_class, cfg, train_cfg)
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    assert_no_domain_randomization(cfg)
    if extra.experiment != "baseline":
        env.configure_reference_initialization(experiment)
    props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    for field, col in (("lower", 0), ("upper", 1), ("velocity", 2)):
        actual = torch.as_tensor(props[field].copy())
        expected = torch.as_tensor(experiment.kinematics.limits[:, col], dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    log_dir = repo/"logs"/experiment.cfg["experiment"]/"exported_data"/(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_"+extra.mode)
    log_dir.mkdir(parents=True, exist_ok=False)
    config = {**class_to_dict(train_cfg), **class_to_dict(cfg)}
    runner = AMPOnPolicyRunner(env, config, experiment, str(log_dir), args.rl_device)
    continuation = None
    if contacting:
        source = locate_source((repo, Path('/workspace'), Path('/personal')), experiment.cfg['contact_refinement']['source_checkpoint_sha256'])
        continuation = warm_start_contact(runner, experiment, source)
    elif horizoning:
        source = locate_source((repo, Path('/workspace'), Path('/personal')), experiment.cfg['horizon']['source_checkpoint_sha256'])
        continuation = warm_start_horizon(runner, experiment, source)
    elif signaling:
        source = locate_source((repo, Path("/workspace"), Path("/personal")), experiment.cfg["signal"]["source_checkpoint_sha256"])
        continuation = warm_start_signal(runner, experiment, source)
    elif refining:
        source_sha = (interrupted_source(experiment.cfg)["source_sha256"] if extra.recover_interrupted else
                      experiment.cfg["refinement"]["source_checkpoint_sha256"])
        source = locate_source((repo, Path("/workspace"), Path("/personal")),
                               source_sha)
        continuation = (recover_interrupted if extra.recover_interrupted else warm_start)(runner, experiment, source)
    def evaluate(checkpoint, completed_updates, smoke=False):
        output = log_dir/(("model_%d.pt" % (9900000+completed_updates)) if resuming else
                          ("model_rollout_%04d.pt" % completed_updates))
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        extended = (horizoning or contacting) and not smoke
        command = [sys.executable, str(repo/"humanoid/scripts/record_amp_policy.py"),
            "--checkpoint-file", str(checkpoint), "--checkpoint-sha256", digest,
            "--expected-commit", commit, "--experiment", extra.experiment,
            "--output", str(output), "--duration", "1" if smoke else ("60" if extended else "20"),
            "--num_envs", "2" if smoke else "16", "--seed", "5", "--headless",
            "--task", experiment.cfg["experiment"], "--sim_device", args.sim_device,
            "--rl_device", args.rl_device]
        if extended: command.append('--extended-validation')
        subprocess.run(command, cwd=str(repo), check=True, timeout=900)
        record = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        if record["checkpoint_sha256"] != digest or record["identity"] != experiment.identity():
            raise ValueError("Independent evaluation artifact mismatch")
        if extended and (record.get('evaluation_protocol') != 'fixed60_independent_mode_seeds' or
                         record.get('mode_seeds') != {'standing': 5, 'reference': 105}):
            raise ValueError('Wrong matched long-horizon evaluation protocol')
        return True
    if extra.mode == "formal" and extra.experiment != "baseline":
        runner.evaluation_callback = evaluate
    if extra.mode == "smoke":
        # Deliberately exercise one terminal/reset path, not just no-reset frames.
        env.amp_force_reset_step = env.common_step_counter+37
    horizon_probe = HorizonProbe(env, smoke=extra.mode == 'smoke') if (horizoning or contacting) else None
    if contacting: env.reset_contact_diagnostics()
    actor_before = {k: v.detach().cpu().clone() for k, v in runner.alg.actor_critic.state_dict().items()}
    d_before = {k: v.detach().cpu().clone() for k, v in runner.amp_trainer.discriminator.state_dict().items()}
    manifest = dict(identity=experiment.identity(), implementation_fingerprint=fingerprint, code_commit=commit,
        mode=extra.mode, num_envs=env.num_envs, updates=args.max_iterations,
        feature_spec=experiment.spec.description(), diagnostics=experiment.diagnostics,
        env_config=class_to_dict(cfg), ppo_config=class_to_dict(train_cfg),
        amp_reward=experiment.cfg["reward"], continuation=continuation,
        training_ready=False, dynamics_acceptance="not claimed",
        learnability_gate=no_dr_gate, evaluation_updates=experiment.cfg.get("evaluation_updates", []),
        runtime=dict(physics_dt=env.sim_params.dt, control_dt=env.dt,
            dof_names=env.dof_names, dof_properties={k: props[k].tolist() for k in props.dtype.names},
            pd_p=env.p_gains.detach().cpu().tolist(), pd_d=env.d_gains.detach().cpu().tolist(),
            actor_history=cfg.env.frame_stack, critic_history=cfg.env.c_frame_stack))
    print("[f1-amp-start] "+json.dumps(manifest), flush=True)
    runner.learn(args.max_iterations, init_at_random_ep_len=False)
    smoke_evaluation_verified = False
    if extra.mode == "smoke" and extra.experiment != "baseline":
        completed = runner.current_learning_iteration
        smoke_evaluation_verified = evaluate(log_dir/("model_%d.pt" % completed), completed, True)
    tensors = list(runner.alg.actor_critic.state_dict().values())+list(runner.amp_trainer.discriminator.state_dict().values())
    certificate = dict(identity=experiment.identity(), implementation_fingerprint=fingerprint, code_commit=commit,
        num_envs=env.num_envs, updates=args.max_iterations, physics_dt_verified=True,
        body_frames_verified=env.amp_verified_frames >= 12 and env.amp_frame_check_max_error <= 5e-4,
        body_frame_max_error_m=env.amp_frame_check_max_error,
        reset_history_verified=runner.alg.reset_checks > 0 and runner.alg.warmup_excluded >= 9*env.num_envs,
        nonzero_style_reward=(runner.alg.style_abs_sum > 0 if signaling else runner.alg.style_sum > 0), valid_windows=runner.alg.valid_windows,
        reset_checks=runner.alg.reset_checks, warmup_excluded=runner.alg.warmup_excluded,
        actor_updated=changed(actor_before, runner.alg.actor_critic.state_dict()),
        discriminator_updated=changed(d_before, runner.amp_trainer.discriminator.state_dict()),
        all_finite=all(bool(torch.isfinite(t).all()) for t in tensors), complete=True,
        no_dr_configuration_verified=assert_no_domain_randomization(cfg)["configuration_verified"],
        rsi_reset_count=getattr(env, "rsi_reset_count", 0),
        replay_window_count=0 if runner.alg.bridge.replay is None else runner.alg.bridge.replay.count,
        smoke_evaluation_verified=smoke_evaluation_verified,
        continuation=continuation, effectiveness_verified=False, dr_unlocked=False)
    if horizon_probe is not None:
        certificate['horizon_diagnostics'] = horizon_probe.report()
        validate_horizon_probe(certificate['horizon_diagnostics'], 60. if contacting else experiment.cfg['horizon']['episode_length_s'],
                               extra.mode == 'smoke')
    if contacting:
        certificate['contact_diagnostics'] = env.contact_diagnostics()
        validate_contact_diagnostics(certificate['contact_diagnostics'], experiment.cfg['contact_refinement']['group'],
                                     horizon_probe.steps, env.num_envs)
    if extra.mode == "smoke":
        validate_cloud_smoke(experiment, certificate, fingerprint, interrupted=extra.recover_interrupted)
    manifest["completion"] = certificate
    (log_dir/"amp_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save(dict(manifest_json=json.dumps(manifest), certificate_json=json.dumps(certificate)), log_dir/"model_amp_manifest.pt")
    print("[f1-amp-complete] "+json.dumps(certificate), flush=True)
    if runner.writer:
        runner.writer.flush()


if __name__ == "__main__":
    main()
