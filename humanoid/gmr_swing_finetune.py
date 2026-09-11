"""Auditable warm start for one approved GMR fine-tuning lineage (no Gym import)."""
import hashlib
from pathlib import Path
import torch

SOURCE_TASK = "TASK_20260911_116"
SOURCE_SHA256 = "ccaf51ecb069e6ecb80ee9fe07eda4a63dab627e4592d15b4240b5d024701857"
SOURCE_REFERENCE_SHA256 = "d625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb"
SOURCE_COMPLETED_UPDATES = 7000


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
            candidates.update(root.rglob("model_7000*.pt"))
    for candidate in sorted(candidates):
        if sha256(candidate) == SOURCE_SHA256:
            return candidate
    raise FileNotFoundError("The approved model_7000 checkpoint was not mounted; no scratch fallback")


def warm_start_runner(runner, checkpoint_path):
    # Verify identity BEFORE unpickling and require an exact network match.
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
    runner.alg.actor_critic.load_state_dict(state, strict=True)
    # New reward: retain all network weights/std but deliberately use fresh Adam
    # moments. Do not inherit an old adaptive learning rate through the optimizer.
    if runner.alg.optimizer.state or runner.alg.state_estimator_optimizer.state:
        raise ValueError("Fine-tuning requires fresh optimizer states")
    runner.current_learning_iteration = SOURCE_COMPLETED_UPDATES
    runner.it = SOURCE_COMPLETED_UPDATES - 1
    return dict(source_task=SOURCE_TASK, checkpoint_sha256=SOURCE_SHA256,
                stored_iteration=loaded["iter"], completed_updates=SOURCE_COMPLETED_UPDATES,
                next_iteration=runner.current_learning_iteration, optimizer_reset=True,
                loaded_tensors=len(state), mean_noise_std=float(state["std"].mean()))
