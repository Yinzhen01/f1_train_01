"""Exact intermediate-checkpoint identities for the interrupted cycle ablations."""
from pathlib import Path

import torch

from humanoid.gmr_accel_finetune import sha256


SOURCES = {
    'x1_gmr_cycle_contact': dict(task='TASK_20260912_164', checkpoint=8500,
        sha256='b2798048962554046d59e6eda2fa37b76f2eb0bc644ecdaae503120edd33b489'),
    'x1_gmr_cycle_smooth': dict(task='TASK_20260912_165', checkpoint=8500,
        sha256='2dd73d3b91470731f2d3d0c4a4b8b4b7ead8e38b1043ead88dfa8a04cb95fa8a'),
    'x1_gmr_cycle_both': dict(task='TASK_20260912_166', checkpoint=8400,
        sha256='f365810a1f82315ac7629bffa8b33973190e2d2c1387970d7600d3f17a7fb88b'),
}
SOURCE_COMMIT = 'bee194f046f276f12769bd18201b0dff89dace27'
FINAL_COMPLETED_UPDATES = 9000


def validate_restart_request(args, extra, commit):
    if commit != extra.expected_commit or args.task not in SOURCES:
        raise ValueError('Unapproved restart task or commit')
    source = SOURCES[args.task]
    if (not args.resume or args.checkpoint != source['checkpoint']
            or extra.source_task != source['task'] or extra.checkpoint_sha256 != source['sha256']
            or args.load_run is not None or args.training_profile or args.seed != 5):
        raise ValueError('Restart requires the exact same-group checkpoint identity and seed')
    remaining = FINAL_COMPLETED_UPDATES - source['checkpoint'] - 1
    budget = {'smoke': (512, 20), 'formal': (4096, remaining)}
    if extra.mode not in budget or (args.num_envs, args.max_iterations) != budget[extra.mode]:
        raise ValueError('Incorrect restart environment/update budget')
    return source


def locate_restart_checkpoint(explicit_path, roots, source):
    explicit = Path(explicit_path)
    if explicit.is_file():
        if sha256(explicit) != source['sha256']:
            raise ValueError('Explicit restart checkpoint hash mismatch; no fallback')
        return explicit
    name = 'model_%d*.pt' % source['checkpoint']
    candidates = set()
    for root in roots:
        if Path(root).is_dir():
            candidates.update(Path(root).rglob(name))
    for candidate in sorted(candidates):
        if sha256(candidate) == source['sha256']:
            return candidate
    raise FileNotFoundError('Exact restart checkpoint not mounted; no scratch fallback')


def restore_cycle_runner(runner, checkpoint, source):
    if sha256(checkpoint) != source['sha256']:
        raise ValueError('Restart checkpoint SHA256 mismatch')
    saved = torch.load(str(checkpoint), map_location=runner.device, weights_only=True)
    # Periodic saves occur AFTER the update named by iter. Unlike the final
    # model9000 (iter8999), a periodic model8500 stores iter8500 -> next8501.
    if saved.get('iter') != source['checkpoint']:
        raise ValueError('Unexpected periodic checkpoint iteration')
    state = saved['model_state_dict']
    if (len(state) != 33 or not all(torch.isfinite(v).all() for v in state.values())
            or state['actor.6.weight'].shape[0] != 12 or not torch.all(state['std'] > 0)):
        raise ValueError('Invalid 12-action restart network')
    if runner.alg.learning_rate != 1e-5 or runner.alg.schedule != 'fixed':
        raise ValueError('Restart must preserve fixed learning rate')
    optimizers = ((runner.alg.optimizer, 'optimizer_state_dict'),
                  (runner.alg.state_estimator_optimizer, 'es_optimizer_state_dict'))
    for optimizer, key in optimizers:
        incoming = saved[key]  # Missing ES optimizer must NOT silently reset it.
        current = optimizer.state_dict()
        if optimizer.state or len(incoming['param_groups']) != len(current['param_groups']):
            raise ValueError('Restart requires fresh runner and matching optimizer groups')
        for old, new in zip(incoming['param_groups'], current['param_groups']):
            if old != new:
                raise ValueError('Optimizer parameter ordering/hyperparameters differ')
        if not all(torch.isfinite(v).all() for entry in incoming['state'].values()
                   for v in entry.values() if isinstance(v, torch.Tensor)):
            raise ValueError('Nonfinite saved optimizer state')
    runner.alg.actor_critic.load_state_dict(state, strict=True)
    for optimizer, key in optimizers:
        optimizer.load_state_dict(saved[key])
    runner.it = saved['iter']
    runner.current_learning_iteration = runner.it + 1
    return dict(source_task=source['task'], checkpoint_sha256=source['sha256'],
        source_training_commit=SOURCE_COMMIT, stored_iteration=runner.it,
        completed_updates=runner.current_learning_iteration, next_iteration=runner.current_learning_iteration,
        optimizer_reset=False, loaded_tensors=len(state),
        restored_optimizer_state_counts={k: len(saved[k]['state']) for _, k in optimizers},
        restored_optimizer_steps={k: sorted({float(v['step']) for v in saved[k]['state'].values()})
                                  for _, k in optimizers},
        mean_noise_std=float(state['std'].mean()),
        simulation_and_rng_restored=False, original_target_completed_updates=FINAL_COMPLETED_UPDATES)
