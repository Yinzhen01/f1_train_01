"""Offline, final-linear-head quadratic smoothing; no policy or simulator run.

Features must be extracted from *new* complete training/validation episodes by
the source policy. This module never executes an actor, loads a checkpoint,
updates PPO/Adam, or certifies closed-loop quality. Previous-action observations
remain source-policy observations, not candidate-policy closed-loop inputs.
"""
from dataclasses import dataclass
import hashlib
import math
import re

import numpy as np


SCHEMA = 'actor_head_offline_smoothing_v1'
HEAD_WEIGHT_KEY = 'actor.6.weight'
HEAD_BIAS_KEY = 'actor.6.bias'
HEAD_KEYS = (HEAD_WEIGHT_KEY, HEAD_BIAS_KEY)
HIDDEN_WIDTH, DESIGN_WIDTH, OUTPUT_WIDTH = 128, 129, 12
HISTORY_FRAMES, CONTROL_DT, ACTION_SCALE = 66, .01, .5
FIR_TAPS, TEMPORAL_WEIGHT, RIDGE = 21, 1., 1e-6
MAX_OUTPUT_CHANGE = .05
MIN_CURVATURE_REDUCTION = .10
LOW_FREQUENCY_RMS_LIMIT, LOW_FREQUENCY_PEAK_LIMIT = .01, .025
ACTIVE_LOW_FREQUENCY_RANGE = .001
LOW_FREQUENCY_RANGE_RATIO = (.98, 1.02)
NUMERICAL_ATOL = 1e-12


@dataclass(frozen=True)
class Episode:
    """One complete history-valid episode, with genuine recorded control ticks.

    ``complete`` is a caller assertion backed by the recording manifest, not
    something that can be inferred from features alone. The driver must retain
    its original seed/env provenance and verify the full history extraction.
    ``source_identity`` is the hash-bound source identity chosen by the driver.
    """
    episode_id: str
    cohort_id: str
    design: np.ndarray
    ticks: np.ndarray
    seed: int
    env_id: int
    source_identity: str
    dt: float = CONTROL_DT
    history_frames: int = HISTORY_FRAMES
    complete: bool = True


@dataclass(frozen=True)
class HeadSmoothingResult:
    candidate_weight: np.ndarray
    candidate_bias: np.ndarray
    report: dict


def _identity(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('source identity must be a lowercase SHA256')
    return value


def _number(value, name, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float,
            np.integer, np.floating)) or not math.isfinite(float(value)) or
            (positive and float(value) <= 0)):
        raise ValueError(name+' must be finite'+(' and positive' if positive else ''))
    return float(value)


def _array(value, shape, name):
    arr = np.asarray(value)
    if (arr.shape != shape or arr.dtype.kind != 'f' or
            not np.isfinite(arr).all()):
        raise ValueError(name+' must be a finite floating array of shape '+str(shape))
    return np.array(arr, dtype=np.float64, copy=True)


def _head(weight, bias):
    w = _array(weight, (OUTPUT_WIDTH, HIDDEN_WIDTH), 'head weight')
    b = _array(bias, (OUTPUT_WIDTH,), 'head bias')
    return np.vstack((w.T, b))


def _design_digest(design):
    # No ID, cohort, tick offset, or split label: renaming an episode cannot hide
    # exact feature reuse. Partial overlap requires the driver's recording audit.
    return hashlib.sha256(np.ascontiguousarray(design, dtype='<f8').tobytes()).hexdigest()


def episode_digest(episode):
    """Content identifier for rejecting protected originals and renamed reuse."""
    if not isinstance(episode, Episode):
        raise ValueError('episode must be an Episode')
    design = np.asarray(episode.design)
    if design.ndim != 2 or design.shape[1] != DESIGN_WIDTH:
        raise ValueError('episode design must have 129 columns')
    design = _array(design, design.shape, 'episode design')
    return _design_digest(design)


def _episodes(episodes, identity, forbidden_cohorts, forbidden_digests, split):
    if not isinstance(episodes, (list, tuple)) or not episodes:
        raise ValueError(split+' must be a nonempty complete-episode list')
    rows = []
    for episode in episodes:
        if not isinstance(episode, Episode):
            raise ValueError('episode must be an Episode')
        if (not isinstance(episode.episode_id, str) or not episode.episode_id or
                not isinstance(episode.cohort_id, str) or not episode.cohort_id):
            raise ValueError('episode and cohort IDs must be nonempty strings')
        if episode.cohort_id in forbidden_cohorts:
            raise ValueError('protected audit cohort cannot enter fitting or validation')
        if _identity(episode.source_identity) != identity:
            raise ValueError('episode source identity mismatch')
        if episode.complete is not True:
            raise ValueError('partial episode is not eligible')
        if (isinstance(episode.history_frames, (bool, np.bool_)) or
                not isinstance(episode.history_frames, (int, np.integer)) or
                episode.history_frames != HISTORY_FRAMES):
            raise ValueError('complete 66-frame source history is required')
        if _number(episode.dt, 'dt', positive=True) != CONTROL_DT:
            raise ValueError('control dt must remain exactly .01 seconds')
        for value, name in ((episode.seed, 'seed'), (episode.env_id, 'env_id')):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(name+' must be a nonnegative integer')
        x = np.asarray(episode.design)
        if x.ndim != 2 or x.shape[1] != DESIGN_WIDTH or x.shape[0] < FIR_TAPS:
            raise ValueError('episode needs at least 21 rows and exactly 129 columns')
        x = _array(x, x.shape, 'episode design')
        if not np.array_equal(x[:, -1], np.ones(len(x))):
            raise ValueError('design final column must be the constant bias 1')
        ticks = np.asarray(episode.ticks)
        if (ticks.shape != (len(x),) or ticks.dtype.kind not in 'iu' or
                np.any(ticks < 0) or np.any(ticks > np.iinfo(np.int64).max)):
            raise ValueError('genuine history-valid consecutive control ticks are required')
        ticks = ticks.astype(np.int64)
        if (int(ticks[0]) < HISTORY_FRAMES-1 or
                not np.array_equal(np.diff(ticks), np.ones(len(x)-1, dtype=np.int64))):
            raise ValueError('genuine history-valid consecutive control ticks are required')
        digest = _design_digest(x)
        if digest in forbidden_digests:
            raise ValueError('protected audit episode content cannot enter this fit')
        rows.append(dict(episode_id=episode.episode_id, cohort_id=episode.cohort_id,
            seed=int(episode.seed), env_id=int(episode.env_id), design=x,
            ticks=ticks.copy(), digest=digest, split=split))
    if len({row['episode_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate episode ID within '+split)
    if len({row['digest'] for row in rows}) != len(rows):
        raise ValueError('duplicate episode content within '+split)
    return rows


def valid_hann(values):
    """21-point normalized Hann FIR, valid interior only, independently per call."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < FIR_TAPS or not np.isfinite(x).all():
        raise ValueError('FIR requires finite 2D data with at least 21 rows')
    taps = np.hanning(FIR_TAPS)
    taps /= taps.sum()
    result = np.zeros((len(x)-FIR_TAPS+1, x.shape[1]), dtype=np.float64)
    for index, tap in enumerate(taps):
        result += tap*x[index:index+len(result)]
    if not np.isfinite(result).all():
        raise ValueError('FIR computation overflowed')
    return result


def curvature(values):
    """Three recorded 100-Hz frames; rad/s² for physical output rad."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('curvature requires finite 2D data with at least three rows')
    result = (x[2:]-2*x[1:-1]+x[:-2])/(CONTROL_DT**2)
    if not np.isfinite(result).all():
        raise ValueError('curvature computation overflowed')
    return result


def _source_scales(train, source):
    low_sq, curve_sq, raw_sq = [], [], []
    for row in train:
        u = ACTION_SCALE*row['design'].dot(source)
        low_sq.append(np.mean(valid_hann(u)**2, axis=0))
        curve_sq.append(np.mean(curvature(u)**2, axis=0))
        raw_sq.append(np.mean(u**2, axis=0))
    raw_rms = np.sqrt(np.mean(raw_sq, axis=0))
    low = np.sqrt(np.mean(low_sq, axis=0))
    curve = np.sqrt(np.mean(curve_sq, axis=0))
    if not np.isfinite(low).all() or not np.isfinite(curve).all():
        raise ValueError('source scale computation overflowed')
    if float(np.max(curve)) == 0:
        raise ValueError('constant/zero-curvature training source cannot demonstrate smoothing')
    # Both floors derive solely from this training source, never from validation.
    low_floor = float(np.max(raw_rms))*1e-6
    curve_floor = float(np.max(curve))*1e-6
    if low_floor <= 0 or curve_floor <= 0:
        raise ValueError('training-derived scale floors must be positive')
    return np.maximum(low, low_floor), np.maximum(curve, curve_floor), low_floor, curve_floor


def _solve(train, source, low_scales, curve_scales):
    """Per-axis source-centered quadratic, with equal weight per episode.

    min_B mean_e mean_t ||H(.5 XB-.5 XB0)/s_low||²
          + mean_e mean_t ||D²(.5 XB)/s_curve||² + 1e-6 ||B-B0||².
    H and D² never join episodes; every axis is solved in float64.
    """
    low_gram = np.zeros((DESIGN_WIDTH, DESIGN_WIDTH), dtype=np.float64)
    curve_gram = np.zeros_like(low_gram)
    for row in train:
        low_x, curve_x = valid_hann(row['design']), curvature(row['design'])
        low_gram += low_x.T.dot(low_x)/len(low_x)/len(train)
        curve_gram += curve_x.T.dot(curve_x)/len(curve_x)/len(train)
    ridge_eye = RIDGE*np.eye(DESIGN_WIDTH)
    solved = np.empty_like(source)
    for joint in range(OUTPUT_WIDTH):
        keep = ACTION_SCALE**2*low_gram/(low_scales[joint]**2)
        smooth = ACTION_SCALE**2*TEMPORAL_WEIGHT*curve_gram/(curve_scales[joint]**2)
        normal = keep+smooth+ridge_eye
        rhs = (keep+ridge_eye).dot(source[:, joint])
        if not np.isfinite(normal).all() or not np.isfinite(rhs).all():
            raise ValueError('quadratic system overflowed')
        try:
            solved[:, joint] = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError('quadratic system is not solvable') from error
    if not np.isfinite(solved).all():
        raise ValueError('quadratic solution is nonfinite')
    return solved


def _limit_direction(train, source, solution):
    direction = solution-source
    peak = max(float(np.max(np.abs(ACTION_SCALE*row['design'].dot(direction)))) for row in train)
    if not math.isfinite(peak):
        raise ValueError('training direction is nonfinite')
    scale = min(1., MAX_OUTPUT_CHANGE/peak) if peak > 0 else 1.
    candidate = source+scale*direction
    # A global backoff only compensates floating rounding of the *training* bound.
    actual = max(float(np.max(np.abs(ACTION_SCALE*row['design'].dot(candidate-source)))) for row in train)
    if actual > MAX_OUTPUT_CHANGE:
        scale *= np.nextafter(MAX_OUTPUT_CHANGE/actual, 0.)
        candidate = source+scale*direction
    return candidate, scale, peak


def _summary(u, curve_scales, action_clip):
    low, curve = valid_hann(u), curvature(u)
    bound = ACTION_SCALE*action_clip
    low_range, raw_range = np.ptp(low, axis=0), np.ptp(u, axis=0)
    curve_rms = np.sqrt(np.mean(curve**2, axis=0))
    curve_mean = float(np.mean((curve/curve_scales)**2))
    if (not np.isfinite(low_range).all() or not np.isfinite(raw_range).all() or
            not np.isfinite(curve_rms).all() or not math.isfinite(curve_mean)):
        raise ValueError('episode metric computation overflowed')
    return dict(low_frequency_range_rad=low_range.tolist(), raw_range_rad=raw_range.tolist(),
        curvature_rms_rad_s2=curve_rms.tolist(), normalized_curvature_squared=curve_mean,
        saturation_boundary_count=np.sum(np.abs(u) >= bound, axis=0).astype(int).tolist(),
        saturation_overflow_count=np.sum(np.abs(u) > bound, axis=0).astype(int).tolist())


def _metrics(rows, source, candidate, curve_scales, action_clip):
    reports = []
    for row in rows:
        source_u, candidate_u = (ACTION_SCALE*row['design'].dot(value) for value in (source, candidate))
        delta, low_delta = candidate_u-source_u, valid_hann(candidate_u)-valid_hann(source_u)
        old, new = _summary(source_u, curve_scales, action_clip), _summary(candidate_u, curve_scales, action_clip)
        baseline = old['normalized_curvature_squared']
        reduction = 1-new['normalized_curvature_squared']/baseline if baseline > 0 else None
        old_range, new_range = np.array(old['low_frequency_range_rad']), np.array(new['low_frequency_range_rad'])
        active = old_range >= ACTIVE_LOW_FREQUENCY_RANGE
        ratios = [float(new_range[j]/old_range[j]) if active[j] else None for j in range(OUTPUT_WIDTH)]
        low_rms, low_peak = np.sqrt(np.mean(low_delta**2, axis=0)), np.max(np.abs(low_delta), axis=0)
        if (not np.isfinite(low_rms).all() or not np.isfinite(low_peak).all() or
                any(ratio is not None and not math.isfinite(ratio) for ratio in ratios) or
                (reduction is not None and not math.isfinite(reduction))):
            raise ValueError('episode deviation computation overflowed')
        report = dict(episode_id=row['episode_id'], cohort_id=row['cohort_id'], seed=row['seed'],
            env_id=row['env_id'], split=row['split'], design_sha256=row['digest'],
            samples=len(source_u), first_tick=int(row['ticks'][0]), last_tick=int(row['ticks'][-1]),
            low_frequency_samples=len(low_delta), curvature_triplets=len(source_u)-2,
            source=old, candidate=new, normalized_curvature_reduction=reduction,
            output_change_peak_rad=float(np.max(np.abs(delta))),
            output_change_peak_per_joint_rad=np.max(np.abs(delta), axis=0).tolist(),
            low_frequency_deviation_rms_per_joint_rad=low_rms.tolist(),
            low_frequency_deviation_peak_per_joint_rad=low_peak.tolist(),
            low_frequency_active_axes=active.tolist(), low_frequency_range_ratio=ratios)
        # Reject overflow in metrics instead of producing a misleading audit JSON.
        if (not math.isfinite(report['output_change_peak_rad']) or
                not math.isfinite(new['normalized_curvature_squared'])):
            raise ValueError('candidate metric computation overflowed')
        reports.append(report)
    return reports


def _admission(train_reports, validation_reports):
    reasons = []
    for row in train_reports+validation_reports:
        label = row['split']+'/'+row['episode_id']
        if row['output_change_peak_rad'] > MAX_OUTPUT_CHANGE+NUMERICAL_ATOL:
            reasons.append(label+': output_change_exceeds_0.05_rad')
        if max(row['low_frequency_deviation_rms_per_joint_rad']) > LOW_FREQUENCY_RMS_LIMIT+NUMERICAL_ATOL:
            reasons.append(label+': low_frequency_rms_exceeds_0.01_rad')
        if max(row['low_frequency_deviation_peak_per_joint_rad']) > LOW_FREQUENCY_PEAK_LIMIT+NUMERICAL_ATOL:
            reasons.append(label+': low_frequency_peak_exceeds_0.025_rad')
        if any(ratio is not None and not LOW_FREQUENCY_RANGE_RATIO[0]-NUMERICAL_ATOL <= ratio <=
                LOW_FREQUENCY_RANGE_RATIO[1]+NUMERICAL_ATOL for ratio in row['low_frequency_range_ratio']):
            reasons.append(label+': low_frequency_range_outside_fixed_2_percent')
        for field in ('saturation_boundary_count', 'saturation_overflow_count'):
            if any(new > old for new, old in zip(row['candidate'][field], row['source'][field])):
                reasons.append(label+': '+field+'_increased')
        if row['split'] == 'validation':
            reduction = row['normalized_curvature_reduction']
            if reduction is None:
                reasons.append(label+': zero_baseline_curvature_not_evaluable')
            elif reduction < MIN_CURVATURE_REDUCTION-NUMERICAL_ATOL:
                reasons.append(label+': curvature_reduction_below_10_percent')
    return reasons


def _prepare_head_audit(train_episodes, validation_episodes, source_weight, source_bias,
        source_identity, action_clip, forbidden_cohort_ids, forbidden_episode_digests):
    identity = _identity(source_identity)
    clip = _number(action_clip, 'action_clip', positive=True)
    if (not isinstance(forbidden_cohort_ids, (set, frozenset, list, tuple)) or
            not forbidden_cohort_ids or any(not isinstance(x, str) or not x for x in forbidden_cohort_ids)):
        raise ValueError('explicit protected audit cohort IDs are required')
    forbidden_cohorts = set(forbidden_cohort_ids)
    if not isinstance(forbidden_episode_digests, (set, frozenset, list, tuple)):
        raise ValueError('protected episode digests must be an explicit collection')
    forbidden_digests = {_identity(value) for value in forbidden_episode_digests}
    source = _head(source_weight, source_bias)
    train = _episodes(train_episodes, identity, forbidden_cohorts, forbidden_digests, 'train')
    validation = _episodes(validation_episodes, identity, forbidden_cohorts, forbidden_digests, 'validation')
    for field in ('episode_id', 'digest', 'cohort_id'):
        if {row[field] for row in train} & {row[field] for row in validation}:
            raise ValueError('train/validation '+field+' overlap is forbidden')
    scales = _source_scales(train, source)
    return identity, clip, source, train, validation, scales


def _candidate_report(identity, clip, source, train, validation, candidate, scales):
    low_scales, curve_scales, low_floor, curve_floor = scales
    train_reports = _metrics(train, source, candidate, curve_scales, clip)
    validation_reports = _metrics(validation, source, candidate, curve_scales, clip)
    reasons = _admission(train_reports, validation_reports)
    return dict(schema=SCHEMA, source_identity=identity, evidence='offline_mathematics_only',
        admitted=not reasons, rejection_reasons=reasons,
        parameter_scope=list(HEAD_KEYS), parameter_scalars=OUTPUT_WIDTH*DESIGN_WIDTH,
        action_scale=ACTION_SCALE, dt=CONTROL_DT, history_frames=HISTORY_FRAMES,
        fir_taps=FIR_TAPS, fir_boundary='valid_per_episode_no_padding',
        temporal_weight=TEMPORAL_WEIGHT, ridge=RIDGE, episode_weighting='equal_episode_then_mean_rows',
        scales_origin='training_source_only', low_frequency_scales_rad=low_scales.tolist(),
        curvature_scales_rad_s2=curve_scales.tolist(), low_frequency_scale_floor_rad=low_floor,
        curvature_scale_floor_rad_s2=curve_floor, action_clip=clip,
        train=train_reports, validation=validation_reports,
        train_normalized_curvature_squared={key: float(np.mean([
            row[key]['normalized_curvature_squared'] for row in train_reports])) for key in ('source', 'candidate')},
        validation_normalized_curvature_squared={key: float(np.mean([
            row[key]['normalized_curvature_squared'] for row in validation_reports])) for key in ('source', 'candidate')},
        frozen_check=dict(status='required_on_exported_complete_state', checker='validate_head_only_change'),
        ppo_updates_added=0, optimizer_state_reused=False, closed_loop_verified=False)


def fit_head_smoothing(train_episodes, validation_episodes, source_weight, source_bias,
        *, source_identity, action_clip, forbidden_cohort_ids, forbidden_episode_digests=(),
        temporal_weight=TEMPORAL_WEIGHT, ridge=RIDGE):
    """Solve once on training only, then audit untouched independent validation.

    Coefficients are fixed, not a validation-search API. Validation never affects
    scales, the normal equations, or the single source-to-solution direction.
    Check the actual exported float32 head with ``evaluate_head_candidate`` and
    its complete model state with ``validate_head_only_change``. Neither offline
    mathematical audit replaces independent original closed-loop evaluation.
    """
    if (_number(temporal_weight, 'temporal_weight') != TEMPORAL_WEIGHT or
            _number(ridge, 'ridge', positive=True) != RIDGE):
        raise ValueError('predeclared temporal_weight=1 and ridge=1e-6 cannot be retuned')
    identity, clip, source, train, validation, scales = _prepare_head_audit(
        train_episodes, validation_episodes, source_weight, source_bias, source_identity,
        action_clip, forbidden_cohort_ids, forbidden_episode_digests)
    solution = _solve(train, source, scales[0], scales[1])
    candidate, scale, unconstrained_peak = _limit_direction(train, source, solution)
    report = _candidate_report(identity, clip, source, train, validation, candidate, scales)
    report.update(audit_origin='float64_closed_form_training', candidate_head_dtype='float64',
        direction_scale=float(scale), unconstrained_training_output_change_peak_rad=unconstrained_peak)
    return HeadSmoothingResult(candidate[:-1].T.copy(), candidate[-1].copy(), report)


def evaluate_head_candidate(train_episodes, validation_episodes, source_weight, source_bias,
        candidate_weight, candidate_bias, *, source_identity, candidate_identity,
        action_clip, forbidden_cohort_ids, forbidden_episode_digests=()):
    """Read-only audit of an already exported/read-back float32 final head.

    Stored float32 values are promoted exactly to float64 for the same feature
    matrix mathematics used by ``fit_head_smoothing``. This does not simulate a
    float32 actor/GEMM, nor substitute for full-state or closed-loop verification.
    Training-source scales are recomputed with the shared immutable protocol;
    no solve, direction scaling, clipping, search, or candidate modification is
    performed. A rounding-induced breach is rejected without repair or retry.
    """
    if _identity(candidate_identity) != _identity(source_identity):
        raise ValueError('candidate/source identity mismatch')
    for value, name in ((candidate_weight, 'candidate weight'), (candidate_bias, 'candidate bias')):
        if np.asarray(value).dtype != np.dtype(np.float32):
            raise ValueError(name+' must be the read-back float32 array')
    candidate = _head(candidate_weight, candidate_bias)
    identity, clip, source, train, validation, scales = _prepare_head_audit(
        train_episodes, validation_episodes, source_weight, source_bias, source_identity,
        action_clip, forbidden_cohort_ids, forbidden_episode_digests)
    report = _candidate_report(identity, clip, source, train, validation, candidate, scales)
    report.update(audit_origin='exported_float32_head_readback', candidate_head_dtype='float32',
        candidate_identity=candidate_identity, candidate_adjusted=False, solver_run=False)
    return report


def _state_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in 'fiub' or not np.isfinite(array).all():
        raise ValueError('state must contain finite numeric tensors')
    return array


def validate_head_only_change(source_state, candidate_state, source_identity, candidate_identity):
    """Exact dtype/shape/byte preservation for every supplied non-head tensor.

    This does not prove an arbitrarily truncated dictionary is a complete model:
    the driver must supply the checkpoint's full model_state_dict, retaining its
    original fingerprint and all non-model learning states separately.
    """
    if _identity(source_identity) != _identity(candidate_identity):
        raise ValueError('candidate/source identity mismatch')
    if (not isinstance(source_state, dict) or not isinstance(candidate_state, dict) or
            source_state.keys() != candidate_state.keys() or
            any(key not in source_state for key in HEAD_KEYS) or len(source_state) <= 2 or
            any(not isinstance(key, str) for key in source_state)):
        raise ValueError('matching full state keys including both head tensors are required')
    changed, frozen = [], []
    shapes = {HEAD_WEIGHT_KEY: (OUTPUT_WIDTH, HIDDEN_WIDTH), HEAD_BIAS_KEY: (OUTPUT_WIDTH,)}
    for key in source_state:
        old, new = _state_array(source_state[key]), _state_array(candidate_state[key])
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError('state dtype/shape changed: '+key)
        if key in shapes and (old.shape != shapes[key] or old.dtype.kind != 'f'):
            raise ValueError('head tensor shape/dtype mismatch: '+key)
        same = old.tobytes(order='C') == new.tobytes(order='C')
        if key not in HEAD_KEYS:
            if not same:
                raise ValueError('frozen non-head tensor changed: '+key)
            frozen.append(key)
        elif not same:
            changed.append(key)
    return dict(schema=SCHEMA, scope='provided_complete_model_state_dict',
        allowed_keys=list(HEAD_KEYS), changed_keys=changed, frozen_keys=sorted(frozen),
        frozen_tensor_count=len(frozen), frozen_exact=True, source_identity=source_identity)
