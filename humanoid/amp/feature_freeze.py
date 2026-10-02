"""Opt-in learning-path isolation; no inference/controller/physics changes.

Freeze only restored CNN/ES parameters. Keep every Adam slot and its moments;
grad=None makes Adam skip the frozen parameters, including its step counter.
Gradient diagnostics use autograd.grad, not an additional optimization step.
"""
import copy
import hashlib
import math

import torch


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and
                left.shape == right.shape and torch.equal(
                    left.detach().contiguous().reshape(-1).view(torch.uint8),
                    right.detach().contiguous().reshape(-1).view(torch.uint8).to(left.device)))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def feature_state(model):
    return {name: value for name, value in model.state_dict().items()
            if name.startswith(('long_history.', 'state_estimator.'))}


def feature_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(feature_state(model).items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(value.dtype).encode()+b'\0')
        digest.update(str(tuple(value.shape)).encode()+b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FrozenInferenceFeatures:
    def __init__(self, ppo):
        self.ppo, self.model = ppo, ppo.actor_critic
        self.parameters = dict(self.model.named_parameters())
        self.frozen = {name: p for name, p in self.parameters.items()
                       if name.startswith(('long_history.', 'state_estimator.'))}
        self.trainable = {name: p for name, p in self.parameters.items()
                          if name not in self.frozen}
        if (not self.frozen or not self.trainable or
                not all(p.requires_grad for p in self.parameters.values()) or
                any(name != 'std' and not name.startswith(('actor.', 'critic.'))
                    for name in self.trainable)):
            raise ValueError('Feature freeze requires a fully restored actor/CNN/ES/critic/std model')
        optimizer_params = [p for g in ppo.optimizer.param_groups for p in g['params']]
        if (len(optimizer_params) != len(self.parameters) or
                any(left is not right for left, right in zip(optimizer_params, self.parameters.values()))):
            raise ValueError('Feature freeze must preserve all original PPO Adam parameter slots')
        es_params = [p for g in ppo.state_estimator_optimizer.param_groups for p in g['params']]
        expected_es = list(self.model.state_estimator.parameters())
        if len(es_params) != len(expected_es) or any(a is not b for a, b in zip(es_params, expected_es)):
            raise ValueError('Feature freeze must preserve original standalone ES Adam parameter slots')
        self.optimizer_ids = {label: tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups)
            for label, optimizer in (('ppo', ppo.optimizer), ('es', ppo.state_estimator_optimizer))}
        self.initial = {name: v.detach().clone() for name, v in feature_state(self.model).items()}
        self.initial_sha = feature_fingerprint(self.model)
        self.optimizer_initial = {}
        for label, optimizer in (('ppo', ppo.optimizer), ('es', ppo.state_estimator_optimizer)):
            for name, p in self.frozen.items():
                self.optimizer_initial[label, name] = (p in optimizer.state,
                                                       copy.deepcopy(optimizer.state.get(p)))
        for p in self.frozen.values():
            p.requires_grad_(False)
            p.grad = None
        self.validate()

    def validate(self):
        if (not _same(self.initial, feature_state(self.model)) or
                any(p.requires_grad or p.grad is not None for p in self.frozen.values()) or
                any(not p.requires_grad for p in self.trainable.values())):
            raise RuntimeError('Frozen inference features, gradients or learning path changed')
        current_parameters = dict(self.model.named_parameters())
        if (current_parameters.keys() != self.parameters.keys() or
                any(self.parameters[name] is not p for name, p in current_parameters.items())):
            raise RuntimeError('Original model parameter identity changed')
        for label, optimizer in (('ppo', self.ppo.optimizer), ('es', self.ppo.state_estimator_optimizer)):
            if tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups) != self.optimizer_ids[label]:
                raise RuntimeError('Original Adam parameter identity/order changed: '+label)
            for name, p in self.frozen.items():
                present, saved = self.optimizer_initial[label, name]
                if present != (p in optimizer.state) or not _same(saved, optimizer.state.get(p)):
                    raise RuntimeError('Frozen Adam moments or step counter changed')
        return True

    def report(self):
        self.validate()
        return dict(frozen_features_unchanged=True,
                    frozen_features_initial_sha256=self.initial_sha,
                    frozen_features_final_sha256=feature_fingerprint(self.model),
                    frozen_optimizer_unchanged=True,
                    frozen_parameter_count=len(self.frozen),
                    trainable_parameter_count=len(self.trainable))


GRADIENT_FIELDS = ('main_grad_norm', 'aux_grad_norm_all_batches', 'es_grad_norm',
                   'actor_main_grad_norm', 'actor_aux_grad_norm',
                   'actor_main_aux_cosine', 'actor_cosine_valid_fraction',
                   'combined_grad_norm_preclip', 'global_clip_factor')


def gradient_diagnostics(model, main_loss, aux_loss, es_loss):
    """Actual current minibatch gradients, before global clipping. No RNG/.grad writes."""
    if any(not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss))
           for loss in (main_loss, aux_loss, es_loss)):
        raise ValueError('Gradient diagnostics require three finite scalar tensor losses')
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    if not parameters:
        raise ValueError('Gradient diagnostics require trainable parameters')
    def grads(loss):
        return (torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                if loss.requires_grad else (None,)*len(parameters))
    main, aux, es = [grads(loss) for loss in (main_loss, aux_loss, es_loss)]
    zero = parameters[0].new_zeros((), dtype=torch.float64)
    def sq(values, actor=False):
        return sum((g.detach().double().square().sum() for (name, _), g in zip(named, values)
                    if g is not None and (not actor or name.startswith('actor.'))), zero)
    main_sq, aux_sq, es_sq = sq(main), sq(aux), sq(es)
    actor_main_sq, actor_aux_sq = sq(main, True), sq(aux, True)
    dot = sum((a.detach().double().mul(b.detach().double()).sum()
               for (name, _), a, b in zip(named, main, aux)
               if name.startswith('actor.') and a is not None and b is not None), zero)
    values = torch.stack((main_sq, aux_sq, es_sq, actor_main_sq, actor_aux_sq, dot)).cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite actual gradient diagnostics')
    valid = values[3] > 0 and values[4] > 0
    return dict(main_grad_norm=math.sqrt(values[0]),
                aux_grad_norm_all_batches=math.sqrt(values[1]), es_grad_norm=math.sqrt(values[2]),
                actor_main_grad_norm=math.sqrt(values[3]), actor_aux_grad_norm=math.sqrt(values[4]),
                actor_main_aux_cosine=max(-1., min(1., values[5]/math.sqrt(values[3]*values[4]))) if valid else 0.,
                actor_cosine_valid_fraction=float(valid))


def validate_gradient_report(report, prefix=''):
    for key in GRADIENT_FIELDS:
        value = report.get(prefix+key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('Missing/nonfinite gradient diagnostic: '+key)
        if key == 'actor_main_aux_cosine':
            valid = -1. <= value <= 1.
        elif key in ('actor_cosine_valid_fraction', 'global_clip_factor'):
            valid = 0. <= value <= 1.
        else:
            valid = value >= 0.
        if not valid:
            raise ValueError('Invalid actual gradient diagnostic: '+key)
    if report[prefix+'es_grad_norm'] != 0.:
        raise ValueError('Frozen state estimator acquired a supervised gradient')
    return True
