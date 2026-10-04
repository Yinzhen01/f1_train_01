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

COMPONENT_GRADIENT_SCHEMA = 'weighted_actor_split_v1'
COMPONENT_GRADIENT_FIELDS = (
    'actor_anchor_grad_norm', 'actor_temporal_grad_norm',
    'actor_main_anchor_cosine', 'actor_main_temporal_cosine',
    'actor_anchor_temporal_cosine',
    'actor_main_anchor_cosine_valid_fraction',
    'actor_main_temporal_cosine_valid_fraction',
    'actor_anchor_temporal_cosine_valid_fraction',
)
_COMPONENT_PAIRS = (
    ('main_anchor', 'actor_main_grad_norm', 'actor_anchor_grad_norm'),
    ('main_temporal', 'actor_main_grad_norm', 'actor_temporal_grad_norm'),
    ('anchor_temporal', 'actor_anchor_grad_norm', 'actor_temporal_grad_norm'),
)


def gradient_diagnostics(model, main_loss, aux_loss, es_loss, *, components=None):
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
    result = dict(main_grad_norm=math.sqrt(values[0]),
                aux_grad_norm_all_batches=math.sqrt(values[1]), es_grad_norm=math.sqrt(values[2]),
                actor_main_grad_norm=math.sqrt(values[3]), actor_aux_grad_norm=math.sqrt(values[4]),
                actor_main_aux_cosine=max(-1., min(1., values[5]/math.sqrt(values[3]*values[4]))) if valid else 0.,
                actor_cosine_valid_fraction=float(valid))
    if components is not None:
        if (not isinstance(components, dict) or
                set(components) != {'weighted_anchor', 'weighted_temporal'}):
            raise ValueError('Both original weighted component tensors are required')
        component_losses = [components[key] for key in ('weighted_anchor', 'weighted_temporal')]
        if any(not isinstance(loss, torch.Tensor) or loss.ndim != 0 or
               loss.device != aux_loss.device or loss.dtype != aux_loss.dtype or
               not bool(torch.isfinite(loss)) for loss in component_losses):
            raise ValueError('Weighted gradient components must be finite scalar tensors')
        if aux_loss.requires_grad and any(not loss.requires_grad for loss in component_losses):
            raise ValueError('Weighted loss components were detached from the actual graph')
        if not torch.equal((component_losses[0]+component_losses[1]).detach(), aux_loss.detach()):
            raise ValueError('Weighted loss components differ from the actual auxiliary loss')
        anchor, temporal = [grads(loss) for loss in component_losses]
        # Frozen features leave ONLY actor parameters in the auxiliary path.
        # Reject an accidentally supplied critic/std supervision graph rather
        # than presenting its actor projection as the entire component loss.
        for component in (anchor, temporal):
            if any(g is not None and (not bool(torch.isfinite(g).all()) or
                   (not name.startswith('actor.') and bool((g != 0).any())))
                   for (name, _), g in zip(named, component)):
                raise ValueError('Nonfinite or non-actor weighted component gradient')
        for parameter, actual, left, right in zip(parameters, aux, anchor, temporal):
            if actual is None and left is None and right is None:
                continue
            zero_gradient = torch.zeros_like(parameter)
            expected = (left if left is not None else zero_gradient)+(right if right is not None else zero_gradient)
            actual = actual if actual is not None else zero_gradient
            if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-7):
                raise ValueError('Weighted component gradients do not sum to the actual auxiliary gradient')
        anchor_sq, temporal_sq = sq(anchor, True), sq(temporal, True)
        def actor_dot(left, right):
            return sum((a.detach().double().mul(b.detach().double()).sum()
                        for (name, _), a, b in zip(named, left, right)
                        if name.startswith('actor.') and a is not None and b is not None), zero)
        split_values = torch.stack((anchor_sq, temporal_sq, actor_dot(main, anchor),
                                    actor_dot(main, temporal), actor_dot(anchor, temporal))).cpu().tolist()
        if not all(math.isfinite(value) for value in split_values):
            raise ValueError('Nonfinite actual component gradient diagnostics')
        result.update(actor_anchor_grad_norm=math.sqrt(split_values[0]),
                      actor_temporal_grad_norm=math.sqrt(split_values[1]))
        for (pair, left_key, right_key), pair_dot in zip(_COMPONENT_PAIRS, split_values[2:]):
            left_norm, right_norm = result[left_key], result[right_key]
            pair_valid = left_norm > 0 and right_norm > 0
            cosine = max(-1., min(1., pair_dot/left_norm/right_norm)) if pair_valid else 0.
            result['actor_'+pair+'_cosine'] = cosine
            result['actor_'+pair+'_cosine_valid_fraction'] = float(pair_valid)
    return result


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


def validate_component_gradient_report(report, prefix=''):
    """New-schema scalar checks only; legacy reports retain their old validator."""
    if report.get(prefix+'component_gradient_schema') != COMPONENT_GRADIENT_SCHEMA:
        raise ValueError('Missing or changed weighted component-gradient schema')
    validate_gradient_report(report, prefix)
    for key in COMPONENT_GRADIENT_FIELDS:
        value = report.get(prefix+key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('Missing/nonfinite component gradient diagnostic: '+key)
        if key.endswith('_cosine'):
            valid = -1. <= value <= 1.
        elif key.endswith('_valid_fraction'):
            valid = 0. <= value <= 1.
        else:
            valid = value >= 0.
        if not valid:
            raise ValueError('Invalid actual component gradient diagnostic: '+key)
    return True


def validate_component_gradient_records(records, expected_updates, minibatches_per_update=8,
                                        start_update=2501, aggregate=None, prefix=''):
    """Audit actual ordered minibatch scalars, never gradient-vector telemetry.

    Record fields are always unprefixed, including when an enclosing log report
    has ``mu_`` keys. Means include undefined-cosine zeroes, accompanied by a
    separate validity fraction; they are not conditional angular averages.
    Triplet counts are repeated PPO evaluations, NOT unique rollout triplets.
    """
    for value, name, minimum in ((expected_updates, 'expected_updates', 0),
                                 (minibatches_per_update, 'minibatches_per_update', 1),
                                 (start_update, 'start_update', 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError('Invalid component record budget: '+name)
    fields = GRADIENT_FIELDS+COMPONENT_GRADIENT_FIELDS
    keys = set(fields+('completed_update', 'minibatch_index', 'batch_size', 'valid_triplets'))
    if not isinstance(records, list) or len(records) != expected_updates*minibatches_per_update:
        raise ValueError('Missing actual all-minibatch component gradient records')
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != keys:
            raise ValueError('Missing or unexpected component minibatch record fields')
        if any(type(record[key]) is not float for key in fields):
            raise ValueError('Actual component minibatch diagnostics must be pure float scalars')
        for key in ('completed_update', 'minibatch_index', 'batch_size', 'valid_triplets'):
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Invalid integer component record field: '+key)
        if (record['completed_update'] != start_update+index//minibatches_per_update or
                record['minibatch_index'] != 1+index % minibatches_per_update):
            raise ValueError('Missing, duplicated or reordered actual component minibatch')
        if record['batch_size'] <= 0 or record['valid_triplets'] > record['batch_size']:
            raise ValueError('More component triplets than actual minibatch samples')
        validate_component_gradient_report(dict(record, component_gradient_schema=COMPONENT_GRADIENT_SCHEMA))
        for pair, left_key, right_key in _COMPONENT_PAIRS:
            cosine_key = 'actor_'+pair+'_cosine'
            validity_key = cosine_key+'_valid_fraction'
            valid = float(record[left_key] > 0 and record[right_key] > 0)
            if record[validity_key] != valid or (not valid and record[cosine_key] != 0.):
                raise ValueError('Inconsistent component cosine zero-norm validity: '+pair)
        valid = float(record['actor_main_grad_norm'] > 0 and record['actor_aux_grad_norm'] > 0)
        if (record['actor_cosine_valid_fraction'] != valid or
                (not valid and record['actor_main_aux_cosine'] != 0.)):
            raise ValueError('Inconsistent main/auxiliary cosine zero-norm validity')
        if not record['valid_triplets'] and record['actor_temporal_grad_norm'] != 0.:
            raise ValueError('Temporal gradient exists without a valid contiguous triplet')
    if aggregate is not None:
        validate_component_gradient_report(aggregate, prefix)
        count = aggregate.get(prefix+'gradient_minibatches')
        if isinstance(count, bool) or not isinstance(count, int) or count != len(records):
            raise ValueError('Component gradient aggregate has the wrong minibatch count')
        for key in fields:
            expected = sum(record[key] for record in records)/len(records) if records else 0.
            if not math.isclose(aggregate[prefix+key], expected, rel_tol=1e-5, abs_tol=1e-10):
                raise ValueError('Component minibatch aggregate differs from actual records: '+key)
        for key, record_key in (('triplet_evaluations', 'valid_triplets'), ('evaluated_samples', 'batch_size')):
            if prefix+key in aggregate:
                value = aggregate[prefix+key]
                if (isinstance(value, bool) or not isinstance(value, int) or
                        value != sum(record[record_key] for record in records)):
                    raise ValueError('Component minibatch sample aggregate mismatch: '+key)
    return True
