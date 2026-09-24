"""CPU synthetic data-flow checks; not physics, training success or deployment."""
import contextlib
import importlib
import io
from pathlib import Path
import sys
import types

import numpy as np
import torch

from .discriminator import AMPDiscriminator, DiscriminatorTrainer
from .features import quat_rotate, wxyz_to_xyzw
from .history import PolicyState
from .integration import AMPBridge


def state_from_clip(clip, frames):
    """Synthetic simulator-shaped states from training-model FK for interface QA."""
    root = torch.from_numpy(clip.arrays["qpos"][frames, :3]).float()
    quat = wxyz_to_xyzw(torch.from_numpy(clip.arrays["qpos"][frames, 3:7]).float())
    key_b = torch.from_numpy(clip.key_positions_b[frames]).float()
    return PolicyState(root, quat, torch.from_numpy(clip.joint_pos[frames]).float(),
                       root[:, None, :] + quat_rotate(quat[:, None, :], key_b),
                       clip.spec.joint_names, clip.spec.body_names)


def cpu_ppo_classes(repo_root):
    # Avoid __init__.py eagerly importing optional wandb/Isaac Gym runner.
    # Load the REAL unmodified DHPPO/ActorCriticDH/storage files as one package.
    namespace = "_f1_amp_cpu_ppo"
    path = str(Path(repo_root) / "humanoid/algo/ppo")
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [path]
        sys.modules[namespace] = package
    elif sys.modules[namespace].__path__ != [path]:
        raise ValueError("CPU PPO package already bound to a different checkout")
    return (importlib.import_module(namespace + ".actor_critic_dh").ActorCriticDH,
            importlib.import_module(namespace + ".dh_ppo").DHPPO)


def run_cpu_dry_run(dataset, seed=17):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec, cfg = dataset.spec, dataset.cfg
    mean, std = dataset.fit_normalization()
    discriminator = AMPDiscriminator(spec, mean, std, cfg["discriminator"]["hidden_dims"])
    dc = cfg["discriminator"]
    trainer = DiscriminatorTrainer(discriminator, dc["learning_rate"], dc["gradient_penalty"], dc["max_grad_norm"])
    actor_class, ppo_class = cpu_ppo_classes(dataset.repo_root)
    # Same 66 x 47 actor history, 5 x 47 short history, 3 x 73 critic layout.
    # Small MLP widths are only for synthetic CPU QA; existing configs untouched.
    with contextlib.redirect_stdout(io.StringIO()):
        actor = actor_class(235, 47, 219, 12, in_channels=66,
                            actor_hidden_dims=[32, 32], critic_hidden_dims=[32, 32],
                            state_estimator_hidden_dims=[32, 16])
    ppo = ppo_class(actor, device="cpu", num_learning_epochs=1, num_mini_batches=1,
                    learning_rate=1e-4, lin_vel_idx=199)
    nenv, steps = 8, 24
    ppo.init_storage(nenv, steps, [66 * 47], [219], [12])
    # Exercise nonzero mixing only for this test; NOT a selected F1 training mix.
    reward = dict(cfg["reward"], style_weight=.25)
    bridge = AMPBridge(spec, trainer, nenv, reward)
    clip = dataset.clips["LAFAN_WALK_01"]
    episodes, ticks = torch.zeros(nenv, dtype=torch.long), torch.zeros(nenv, dtype=torch.long)
    offsets = np.arange(nenv) * 25
    bridge.prime(state_from_clip(clip, offsets), episodes, ticks)
    valid_steps, max_demo_policy_error = [], 0.
    for k in range(1, steps + 1):
        obs, critic = torch.randn(nenv, 66 * 47), torch.randn(nenv, 219)
        with torch.no_grad():
            ppo.act(obs, critic)
        bridge.capture_pre_reset(state_from_clip(clip, offsets + k), episodes, ticks + k)
        windows, valid = bridge.pending
        if k >= 10:
            expected = torch.stack([clip.window(int(o + k - 9)) for o in offsets])
            max_demo_policy_error = max(max_demo_policy_error, float((windows - expected).abs().max()))
        diagnostics = bridge.process_env_step(ppo, torch.ones(nenv), torch.zeros(nenv, dtype=torch.bool), {})
        valid_steps.append(int(diagnostics["valid_mask"].sum()))
    with torch.no_grad():
        ppo.compute_returns(torch.randn(nenv, 219))
    discriminator_before = [p.detach().clone() for p in discriminator.parameters()]
    actor_before = [p.detach().clone() for p in actor.parameters()]
    ppo_losses = ppo.update()
    ppo_did_not_change_discriminator = all(torch.equal(a, b) for a, b in zip(discriminator_before, discriminator.parameters()))
    ppo_changed_actor = any(not torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))
    actor_before = [p.detach().clone() for p in actor.parameters()]
    disc_losses = bridge.update_discriminator(dataset.sampler(seed=seed), batch_size=16, seed=seed)
    disc_did_not_change_actor = all(torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))
    disc_changed = any(not torch.equal(a, b) for a, b in zip(discriminator_before, discriminator.parameters()))
    demo, provenance = dataset.sampler(seed=seed).sample(16)
    with torch.no_grad():
        score, style = discriminator(demo), discriminator.style_reward(demo)
    result = dict(device="cpu", seed=seed, torch_version=torch.__version__,
                  data_loading_success=True, discriminator_forward_success=score.shape == (16, 1),
                  synthetic_ppo_discriminator_step_success=True,
                  isaac_gym_smoke_success=False, formal_amp_training_success=False,
                  dynamic_executable=False, hardware_deployable=False,
                  input_shape=list(demo.shape), output_shape=list(score.shape),
                  window_span_s=.09, actor_history=66, actor_short_history=5, critic_history=3,
                  policy_valid_env_counts_per_step=valid_steps,
                  max_demo_policy_feature_error=max_demo_policy_error,
                  ppo_losses=list(map(float, ppo_losses)), discriminator_losses=disc_losses,
                  style_reward_range=[float(style.min()), float(style.max())],
                  reward_mix_for_test_only=reward,
                  gradient_isolation=dict(ppo_did_not_change_discriminator=ppo_did_not_change_discriminator,
                                          discriminator_did_not_change_actor=disc_did_not_change_actor,
                                          ppo_changed_actor=ppo_changed_actor, discriminator_changed=disc_changed),
                  sampled_provenance=provenance)
    if not all(result["gradient_isolation"].values()) or valid_steps[:9] != [0] * 9:
        raise AssertionError("Gradient/history isolation failed")
    if max_demo_policy_error > 2e-5 or not np.isfinite(ppo_losses).all() or not torch.isfinite(style).all():
        raise AssertionError("CPU interface dry run failed")
    return result
