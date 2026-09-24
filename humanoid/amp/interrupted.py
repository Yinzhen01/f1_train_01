"""Recover the interrupted matched runs without changing their experiment definition."""
import hashlib
from pathlib import Path

import torch

from .refinement import validate_refinement, restore_learning_state

SOURCES = {
    "control": ("TASK_20260924_071", "bfe69b086395292e7277b75a8472f6aed451cb33ad2705cb8d336254eb9eb3e8"),
    "smooth": ("TASK_20260924_072", "e553b2e800d6c307e746e51f17877e9fd4bf27268979fdd7c3cb2c76453495df"),
    "noamp": ("TASK_20260924_073", "b2a78642e284a9c7a43b4c0754202a8a5d06baecdd57e9d42d3e13ac89e6b01a"),
}


def interrupted_source(cfg):
    validate_refinement(cfg)
    task, digest = SOURCES[cfg["refinement"]["group"]]
    return dict(source_task=task, source_sha256=digest, source_completed_updates=1250,
                kind="interrupted_matched_run", target_completed_updates=1500)


def restart_budget(cfg, mode):
    source = interrupted_source(cfg)
    if mode not in ("smoke", "formal"):
        raise ValueError("Unknown restart mode")
    return dict(num_envs=cfg[mode]["num_envs"],
                updates=10 if mode == "smoke" else source["target_completed_updates"]-source["source_completed_updates"])


def validate_recovery_certificate(cfg, continuation):
    expected = interrupted_source(cfg)
    if any(continuation.get(key) != value for key, value in expected.items()):
        raise ValueError("Interrupted run recovery source mismatch")
    if (continuation.get("actor_and_discriminator_restored") is not True or
        continuation.get("optimizers_restored") is not True or continuation.get("replay_windows") != 50000):
        raise ValueError("Incomplete interrupted training-state restoration")
    return True


def recover_interrupted(runner, experiment, checkpoint):
    source = interrupted_source(experiment.cfg)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != source["source_sha256"]:
        raise ValueError("Interrupted checkpoint SHA mismatch")
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state["amp_identity"] != experiment.identity() or state["completed_updates"] != 1250 or state["iter"] != 1249:
        raise ValueError("Interrupted checkpoint identity/update mismatch")
    if state.get("amp_replay_state") is None:
        raise ValueError("Interrupted source must include complete policy replay")
    # Existing fixed learning rate must not silently change at the restart.
    expected_lr = experiment.cfg["refinement"]["learning_rate"]
    for key in ("optimizer_state_dict", "es_optimizer_state_dict"):
        if any(group["lr"] != expected_lr for group in state[key]["param_groups"]):
            raise ValueError("Source learning rate differs from matched experiment")
    proof = dict(**source, **restore_learning_state(runner, experiment, state))
    validate_recovery_certificate(experiment.cfg, proof)
    return proof
