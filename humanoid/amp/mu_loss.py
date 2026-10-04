"""Learning-only deterministic-mean losses, not deployed action filtering.

The anchor freezes the complete restored actor, long-history CNN and state
estimator. The temporal term regularizes a target-command second difference,
``action_scale * delta2(mu) / dt**2``, normalized by fixed baseline per-joint
scales. It is not measured joint acceleration or a physical safety threshold.
The caller must supply a strict same-episode, contiguous three-frame mask from
rollout storage; shuffled minibatch adjacency is not temporal adjacency. CPU
gradient checks do not establish motion quality, physics or hardware benefit.
"""
import copy
import math
from numbers import Real

import torch
from torch import nn


class DeterministicMuLoss:
    """Snapshot after complete source restoration, before any student update.

    This is deliberately not an ``nn.Module`` containing the student or teacher:
    constructing it does not add the teacher to the existing PPO/ES optimizer.
    Only ``act_inference`` is called, never sampled ``act`` or distribution
    updates. ``ActorCriticDH.act_inference`` traverses each model's own complete
    CNN, state estimator and actor without sampling or changing its distribution.
    """

    STAT_FIELDS = ('anchor_loss', 'temporal_loss', 'weighted_anchor_loss',
                   'weighted_temporal_loss', 'valid_triplets', 'batch_size',
                   'teacher_unchanged')

    def __init__(self, actor_critic, acceleration_scale, anchor_coef,
                 temporal_coef, dt=.01, action_scale=.5, *, component_gradient_schema=None):
        from .feature_freeze import COMPONENT_GRADIENT_SCHEMA
        if component_gradient_schema not in (None, COMPONENT_GRADIENT_SCHEMA):
            raise ValueError('Unknown deterministic mu component-gradient schema')
        self._component_gradient_schema = component_gradient_schema
        if (not isinstance(actor_critic, nn.Module) or
                not callable(getattr(actor_critic, 'act_inference', None)) or
                any(not isinstance(getattr(actor_critic, name, None), nn.Module)
                    for name in ('actor', 'long_history', 'state_estimator'))):
            raise ValueError('A complete actor/CNN/state-estimator inference model is required')
        self._anchor_coef = self._scalar(anchor_coef, 'anchor_coef', positive=False)
        self._temporal_coef = self._scalar(temporal_coef, 'temporal_coef', positive=False)
        self._dt = self._scalar(dt, 'dt')
        self._action_scale = self._scalar(action_scale, 'action_scale')
        dt_squared = self.dt*self.dt
        if not dt_squared or not math.isfinite(dt_squared):
            raise ValueError('dt squared must be positive and finite')
        self._acceleration_factor = self.action_scale/dt_squared
        if not math.isfinite(self._acceleration_factor) or not self._acceleration_factor:
            raise ValueError('Target-command acceleration factor is not representable')
        try:
            supplied_scale = torch.as_tensor(acceleration_scale)
            # Infer the original kind first, rejecting bool/complex. Python
            # decimal floats must then enter float64 directly, not via float32.
            if (not isinstance(acceleration_scale, torch.Tensor) and
                    supplied_scale.ndim == 1 and supplied_scale.dtype != torch.bool and
                    not supplied_scale.is_complex()):
                if any(isinstance(value, bool) for value in acceleration_scale):
                    raise ValueError('Boolean per-joint scales are not allowed')
                supplied_scale = torch.as_tensor(acceleration_scale, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError):
            raise ValueError('acceleration_scale must be a positive finite per-joint vector') from None
        if (supplied_scale.ndim != 1 or supplied_scale.numel() == 0 or
                supplied_scale.dtype == torch.bool or supplied_scale.is_complex() or
                supplied_scale.requires_grad):
            raise ValueError('acceleration_scale must be a fixed nonempty real 1D vector')
        self._acceleration_scale = supplied_scale.detach().to(device='cpu', dtype=torch.float64).clone()
        if not bool((torch.isfinite(self._acceleration_scale) & (self._acceleration_scale > 0)).all()):
            raise ValueError('acceleration_scale must be positive and finite')
        std = getattr(actor_critic, 'std', None)
        if (isinstance(std, torch.Tensor) and
                (std.ndim != 1 or std.numel() != self._acceleration_scale.numel())):
            raise ValueError('acceleration_scale does not match the model action width')
        state = actor_critic.state_dict()
        if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError('Inference model must have a tensor-only nonempty state')
        finite = [torch.isfinite(value).all() for value in state.values()]
        if not bool(torch.stack([value.to(finite[0].device) for value in finite]).all()):
            raise ValueError('Source model state contains nonfinite values')
        self.actor_critic = actor_critic
        if not any(parameter.requires_grad for parameter in self._student_parameters()):
            raise ValueError('Student deterministic inference path has no trainable parameters')
        # A PPO Normal cache can contain non-leaf tensors that cannot be deepcopied.
        # Exclude only this transient cache via memo; never mutate the student.
        distribution = getattr(actor_critic, 'distribution', None)
        memo = {} if distribution is None else {id(distribution): None}
        self._teacher = copy.deepcopy(actor_critic, memo).eval()
        for parameter in self._teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._teacher_state = {name: value.detach().clone()
                               for name, value in self._teacher.state_dict().items()}
        self._latest_stats = self._stats(0., 0., 0., 0., 0, 0, True)

    @staticmethod
    def _scalar(value, name, positive=True):
        if (isinstance(value, bool) or not isinstance(value, Real) or
                not math.isfinite(float(value)) or
                (value <= 0 if positive else value < 0)):
            raise ValueError(name+' must be '+('positive' if positive else 'nonnegative')+' and finite')
        return float(value)

    @property
    def teacher(self):
        """Read-only reference for source-state provenance; do not reload its state."""
        return self._teacher

    @property
    def acceleration_scale(self):
        """Independent float64 copy; modifying it cannot change normalization."""
        return self._acceleration_scale.clone()

    @property
    def anchor_coef(self):
        return self._anchor_coef

    @property
    def temporal_coef(self):
        return self._temporal_coef

    @property
    def dt(self):
        return self._dt

    @property
    def action_scale(self):
        return self._action_scale

    @property
    def component_gradient_schema(self):
        """Opt-in measurement contract, not an optimization or controller change."""
        return self._component_gradient_schema

    def _student_parameters(self):
        seen = set()
        for name in ('actor', 'long_history', 'state_estimator'):
            for parameter in getattr(self.actor_critic, name).parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def teacher_unchanged(self):
        """Bitwise parameters/buffers check, including signed zeros, plus freeze/eval."""
        if (any(module.training for module in self._teacher.modules()) or
                any(parameter.requires_grad or parameter.grad is not None
                    for parameter in self._teacher.parameters())):
            return False
        current = self._teacher.state_dict()
        if current.keys() != self._teacher_state.keys():
            return False
        comparisons = []
        for name, value in current.items():
            saved = self._teacher_state[name]
            if value.dtype != saved.dtype or value.shape != saved.shape:
                return False
            # Numeric equality would miss +0.0 -> -0.0; compare tensor bytes.
            left = value.detach().contiguous().reshape(-1).view(torch.uint8)
            right = saved.contiguous().reshape(-1).view(torch.uint8).to(left.device)
            comparisons.append((left == right).all())
        return bool(torch.stack([value.to(comparisons[0].device)
                                 for value in comparisons]).all())

    @staticmethod
    def _stats(anchor, temporal, weighted_anchor, weighted_temporal,
               valid_triplets, batch_size, unchanged):
        return dict(anchor_loss=float(anchor), temporal_loss=float(temporal),
                    weighted_anchor_loss=float(weighted_anchor),
                    weighted_temporal_loss=float(weighted_temporal),
                    valid_triplets=int(valid_triplets), batch_size=int(batch_size),
                    teacher_unchanged=bool(unchanged))

    def report(self):
        result = dict(self._latest_stats)
        result['teacher_unchanged'] = self.teacher_unchanged()
        return result

    @property
    def latest_stats(self):
        return self.report()

    def loss(self, observations, previous_observations, older_observations, valid_mask,
             *, return_components=False):
        """Return a differentiable scalar and JSON-safe, detached fixed-field stats.

        All three observations are finite floating ``[batch, observation]``
        tensors of identical shape/device/dtype. ``valid_mask`` is bool ``[batch]``
        on that device. Empty batches and empty triplet selections are finite
        differentiable zeroes. A zero coefficient still reports its unweighted
        diagnostic term; no branch uses detached rollout means.

        The legacy two-tuple is unchanged. An explicitly registered component
        schema permits a three-tuple whose last item contains the two WEIGHTED
        scalar tensors from this same forward graph. The caller must release
        them after the real backward; no tensor/graph is retained as telemetry.
        """
        if not isinstance(return_components, bool):
            raise ValueError('return_components must be a boolean')
        if return_components and self.component_gradient_schema is None:
            raise ValueError('Differentiable loss components require the opt-in schema')
        inputs = (observations, previous_observations, older_observations)
        if any(not isinstance(value, torch.Tensor) or value.ndim != 2 or
               not value.is_floating_point() for value in inputs):
            raise ValueError('All observations must be floating 2D tensors')
        if (observations.shape[1] == 0 or any(value.shape != observations.shape or
                value.device != observations.device or value.dtype != observations.dtype
                for value in inputs)):
            raise ValueError('Observation shape/device/dtype must be identical and nonzero-width')
        if (not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool or
                valid_mask.ndim != 1 or valid_mask.shape[0] != observations.shape[0] or
                valid_mask.device != observations.device):
            raise ValueError('valid_mask must be a same-device bool [batch] tensor')
        parameter = next(self._student_parameters())
        if observations.device != parameter.device or observations.dtype != parameter.dtype:
            raise ValueError('Observation device/dtype differs from the student model')
        channels = getattr(self.actor_critic, 'in_channels', None)
        width = getattr(self.actor_critic, 'num_proprio_obs', None)
        if channels is not None and width is not None and observations.shape[1] != channels*width:
            raise ValueError('Observation width differs from the complete history input')
        scale = self._acceleration_scale.to(device=observations.device, dtype=observations.dtype)
        constants = torch.tensor((self.anchor_coef, self.temporal_coef,
                                  self._acceleration_factor),
                                 device=observations.device, dtype=observations.dtype)
        checks = [torch.isfinite(value).all() for value in inputs]
        checks.append((torch.isfinite(scale) & (scale > 0)).all())
        checks.append(torch.isfinite(constants).all() & (constants[2] > 0) &
                      (constants[0] > 0 if self.anchor_coef > 0 else constants[0] == 0) &
                      (constants[1] > 0 if self.temporal_coef > 0 else constants[1] == 0))
        if not bool(torch.stack(checks).all()):
            raise ValueError('Observations, scales or loss constants are nonfinite/unrepresentable')
        if not self.teacher_unchanged():
            raise ValueError('Frozen teacher state, eval mode or gradient isolation changed')
        batch_size = observations.shape[0]
        valid_triplets = int(valid_mask.sum())
        if batch_size == 0:
            zero = sum(parameter.reshape(-1)[:0].sum()
                       for parameter in self._student_parameters() if parameter.requires_grad)
            self._latest_stats = self._stats(0., 0., 0., 0., 0, 0, True)
            if return_components:
                return zero, dict(self._latest_stats), dict(weighted_anchor=zero,
                                                           weighted_temporal=zero)
            return zero, dict(self._latest_stats)
        current_mu = self.actor_critic.act_inference(observations)
        with torch.no_grad():
            teacher_mu = self._teacher.act_inference(observations)
        means = [current_mu, teacher_mu]
        if valid_triplets:
            previous_mu = self.actor_critic.act_inference(previous_observations[valid_mask])
            older_mu = self.actor_critic.act_inference(older_observations[valid_mask])
            means.extend((previous_mu, older_mu))
        expected_width = self._acceleration_scale.numel()
        for index, mean in enumerate(means):
            expected_batch = batch_size if index < 2 else valid_triplets
            if (not isinstance(mean, torch.Tensor) or mean.shape != (expected_batch, expected_width) or
                    mean.device != observations.device or mean.dtype != observations.dtype):
                raise ValueError('Deterministic mean has an invalid batch/action shape or dtype')
            if index != 1 and not mean.requires_grad:
                raise ValueError('Student deterministic mean was detached from autograd')
        if not bool(torch.stack([torch.isfinite(mean).all() for mean in means]).all()):
            raise ValueError('Deterministic mean contains nonfinite values')
        anchor = (current_mu-teacher_mu).square().mean()
        if valid_triplets:
            normalized = ((current_mu[valid_mask]-2*previous_mu+older_mu)*
                          constants[2])/scale
            temporal = normalized.square().mean()
        else:
            temporal = current_mu[:1, :1].sum()*0
        weighted_anchor = anchor*constants[0]
        weighted_temporal = temporal*constants[1]
        auxiliary = weighted_anchor+weighted_temporal
        terms = torch.stack((anchor, temporal, weighted_anchor, weighted_temporal, auxiliary))
        if not bool(torch.isfinite(terms).all()):
            raise ValueError('Auxiliary losses overflowed or became nonfinite')
        if not self.teacher_unchanged():
            raise ValueError('Frozen teacher changed during deterministic inference')
        values = terms[:4].detach().cpu().tolist()
        self._latest_stats = self._stats(*values, valid_triplets, batch_size, True)
        if return_components:
            return auxiliary, dict(self._latest_stats), dict(weighted_anchor=weighted_anchor,
                                                            weighted_temporal=weighted_temporal)
        return auxiliary, dict(self._latest_stats)
