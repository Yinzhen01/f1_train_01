"""Fail-closed model8000 warm start shared by all acceleration experiment groups."""
import hashlib
from pathlib import Path
import torch

SOURCE_TASK = "TASK_20260911_171"
SOURCE_SHA256 = "4c865c178a12a4796013b6b138972becb875301d7198543a6909c9a76ef7fe50"
SOURCE_TRAINING_COMMIT = "a92dfe6346bcd1e853a1c923fec33667e9b601bb"
SOURCE_REFERENCE_SHA256 = "d625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb"
SOURCE_URDF_SHA256 = "9fb3d6efa623576a964aef9a3e2db2f96ca2296a85f2e0ade6bfe0ecf043ba15"
SOURCE_COMPLETED_UPDATES = 8000
GROUP_WEIGHTS = {"x1_gmr_accel_1x": -1e-7, "x1_gmr_accel_3x": -3e-7, "x1_gmr_accel_10x": -1e-6}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate_verified_checkpoint(explicit_path, roots):
    explicit = Path(explicit_path)
    if explicit.is_file():
        if sha256(explicit) != SOURCE_SHA256:
            raise ValueError("Explicit checkpoint hash mismatch; no fallback allowed")
        return explicit
    candidates = set()
    for root in roots:
        root = Path(root)
        if root.is_dir():
            candidates.update(root.rglob("model_8000*.pt"))
    for candidate in sorted(candidates):
        if sha256(candidate) == SOURCE_SHA256:
            return candidate
    raise FileNotFoundError("Approved model_8000 was not mounted; no scratch fallback")


def warm_start_runner(runner, checkpoint_path):
    if sha256(checkpoint_path) != SOURCE_SHA256:
        raise ValueError("Source checkpoint SHA256 mismatch")
    loaded = torch.load(str(checkpoint_path), map_location=runner.device, weights_only=True)
    if loaded.get("iter") != SOURCE_COMPLETED_UPDATES - 1:
        raise ValueError("Unexpected source iteration")
    state = loaded["model_state_dict"]
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite source model")
    if state["actor.6.weight"].shape[0] != 12 or not torch.all(state["std"] > 0):
        raise ValueError("Expected a valid twelve-action source policy")
    if runner.alg.optimizer.state or runner.alg.state_estimator_optimizer.state:
        raise ValueError("Fine-tuning requires fresh optimizer states in ALL groups")
    runner.alg.actor_critic.load_state_dict(state, strict=True)
    runner.current_learning_iteration = SOURCE_COMPLETED_UPDATES
    runner.it = SOURCE_COMPLETED_UPDATES - 1
    return dict(source_task=SOURCE_TASK, checkpoint_sha256=SOURCE_SHA256,
                source_training_commit=SOURCE_TRAINING_COMMIT,
                stored_iteration=loaded["iter"], completed_updates=SOURCE_COMPLETED_UPDATES,
                next_iteration=runner.current_learning_iteration, optimizer_reset=True,
                loaded_tensors=len(state), mean_noise_std=float(state["std"].mean()))
