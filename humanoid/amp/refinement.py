"""Hash-bound matched AMP continuation; pure CPU validation/math helpers."""
import hashlib
from pathlib import Path

import torch

from .scaled_experiment import ScaledExperiment

GROUPS = ("refine_control", "refine_smooth", "refine_noamp")


def robust_acceleration_cost(acceleration):
    # Quadratic near zero, linear tails: rare touchdown impulses cannot dominate.
    x = acceleration / 100.
    return (2 * (torch.sqrt(1 + x.square()) - 1)).mean(-1)


def ground_force_weight(force_z, mesh_height_proxy):
    # No 0..5 N dead band. Geometry rejects distant self/contact-force artifacts.
    positive = force_z.clamp_min(0.)
    return positive/(positive+5.) * ((.02-mesh_height_proxy)/.01).clamp(0., 1.)


def validate_refinement(cfg):
    r = cfg.get("refinement", {})
    group = r.get("group")
    if group not in ("control", "smooth", "noamp") or cfg["experiment"] != "f1_amp_walk02_refine_"+group:
        raise ValueError("Unknown matched continuation group")
    if (r.get("source_task") != "TASK_20260924_060" or r.get("source_completed_updates") != 1000 or
        r.get("source_checkpoint_sha256") != "def5eea837a60bf8bc4b7c3a481e8c3e84b5da3843e9af4d60b1e3c134c5dd63"):
        raise ValueError("Unapproved warm-start source")
    if (cfg["smoke"] != {"num_envs": 32, "updates": 10} or
        cfg["formal"] != {"num_envs": 4096, "updates": 500} or
        cfg["evaluation_updates"] != [1250, 1500] or cfg["artifact_checkpoint_offset"] != 8800000 or
        cfg["max_experiment_rounds"] != 20 or cfg["learnability_only"] is not True or
        r["learning_rate"] != 5e-5 or cfg["reward"]["style_weight"] != (0. if group == "noamp" else 1.)):
        raise ValueError("Unapproved continuation budget or ablation")
    return True


def locate_source(roots, expected_sha):
    seen = {}
    for root in roots:
        if root.is_dir():
            for candidate in sorted(root.rglob("model_*.pt")):
                if candidate.resolve() in seen:
                    continue
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
                seen[candidate.resolve()] = (candidate.name, digest)
                if digest == expected_sha:
                    return candidate
    raise FileNotFoundError("No mounted source checkpoint matches the required SHA256; candidates="+str(list(seen.values())))


def warm_start(runner, experiment, checkpoint):
    validate_refinement(experiment.cfg)
    cfg = experiment.cfg["refinement"]
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != cfg["source_checkpoint_sha256"]:
        raise ValueError("Warm-start checkpoint hash mismatch")
    source = ScaledExperiment(experiment.repo, experiment.repo/cfg["source_config"])
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state["amp_identity"] != source.identity() or state["completed_updates"] != 1000:
        raise ValueError("Warm-start source identity/update mismatch")
    for key in ("motion_sha256", "urdf_lf_sha256", "feature_fingerprint"):
        if experiment.identity()[key] != source.identity()[key]:
            raise ValueError("Continuation changed source data/robot/features")
    restored = restore_learning_state(runner, experiment, state)
    return dict(source_task=cfg["source_task"], source_sha256=cfg["source_checkpoint_sha256"],
        source_completed_updates=1000, **restored)


def restore_learning_state(runner, experiment, state, learning_rate=None):
    """Used only after the caller verifies exact file hash, identity and update."""
    learning_rate = experiment.cfg["refinement"]["learning_rate"] if learning_rate is None else learning_rate
    actor = runner.alg.actor_critic
    actor.load_state_dict(state["model_state_dict"], strict=True)
    runner.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
    runner.alg.state_estimator_optimizer.load_state_dict(state["es_optimizer_state_dict"])
    runner.amp_trainer.discriminator.load_state_dict(state["amp_discriminator_state_dict"], strict=True)
    # Keep the checkpoint buffers exactly. Cross-platform quaternion arithmetic
    # differs by ~1e-8 locally; this checks compatibility, never refits statistics.
    torch.testing.assert_close(runner.amp_trainer.discriminator.mean.cpu(), experiment.mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(runner.amp_trainer.discriminator.std.cpu(), experiment.std, rtol=1e-5, atol=1e-6)
    runner.amp_trainer.optimizer.load_state_dict(state["amp_optimizer_state_dict"])
    runner.alg.bridge.replay.load_state_dict(state["amp_replay_state"], runner.device)
    for optimizer in (runner.alg.optimizer, runner.alg.state_estimator_optimizer):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    runner.alg.ppo.learning_rate = learning_rate
    runner.current_learning_iteration = state["completed_updates"]
    runner.it = state["completed_updates"]-1
    runner.alg.updates = state["completed_updates"]
    return dict(actor_and_discriminator_restored=True,
        optimizers_restored=True, replay_windows=runner.alg.bridge.replay.count,
        learning_rate=learning_rate, rng="fresh seed5 and real freshly-reset history, not source RNG continuation")


def select_environment(experiment_name):
    from humanoid.envs.x1.x1_amp_config import X1AMPCfg, X1AMPCfgPPO
    from humanoid.envs.x1.x1_amp_env import X1AMPEnv
    from humanoid.envs.x1.x1_amp_recovery_config import X1AMPRecoveryCfg, X1AMPRecoveryCfgPPO
    from humanoid.envs.x1.x1_amp_recovery_env import X1AMPRecoveryEnv
    if experiment_name in ('contact_control', 'contact_slip', 'contact_tail'):
        from .contact import apply_contact_config
        from humanoid.envs.x1.x1_amp_refine_config import X1AMPRefineCfg
        from humanoid.envs.x1.x1_amp_contact_env import X1AMPContactEnv
        cfg = apply_contact_config(X1AMPRefineCfg(), experiment_name.split('_', 1)[1])
        return cfg, X1AMPRecoveryCfgPPO(), X1AMPContactEnv
    if experiment_name in ('horizon_short', 'horizon_long'):
        from .horizon import apply_horizon_config
        from humanoid.envs.x1.x1_amp_refine_config import X1AMPRefineCfg
        from humanoid.envs.x1.x1_amp_refine_env import X1AMPRefineEnv
        cfg = X1AMPRefineCfg()
        apply_horizon_config(cfg, experiment_name.split('_', 1)[1])
        return cfg, X1AMPRecoveryCfgPPO(), X1AMPRefineEnv
    if experiment_name == "baseline":
        return X1AMPCfg(), X1AMPCfgPPO(), X1AMPEnv
    if experiment_name in ("recovery", "recovery_static", "refine_control"):
        return X1AMPRecoveryCfg(), X1AMPRecoveryCfgPPO(), X1AMPRecoveryEnv
    if experiment_name in ("refine_smooth", "refine_noamp", "signal_signed", "signal_bridge"):
        from humanoid.envs.x1.x1_amp_refine_config import X1AMPRefineCfg
        from humanoid.envs.x1.x1_amp_refine_env import X1AMPRefineEnv
        return X1AMPRefineCfg(), X1AMPRecoveryCfgPPO(), X1AMPRefineEnv
    raise ValueError("Unknown experiment environment")
