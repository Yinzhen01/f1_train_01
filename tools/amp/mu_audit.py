"""Independent deterministic-mu artifact checks, never a motion-quality claim."""
import hashlib
import copy
import json
import math
from numbers import Real
from pathlib import Path
import re

import torch

from humanoid.amp.mu_temporal import (mu_contract, state_fingerprint,
    validate_mu, validate_mu_certificate,
    validate_mu_loss_report as validate_contract_loss_report)
from tools.amp.jitter_audit import (compare_environment as compare_control_environment,
    validate_jitter_updates)


LOSS_FIELDS = ('anchor_loss', 'temporal_loss', 'weighted_anchor_loss',
               'weighted_temporal_loss', 'aux_grad_norm')
# Native X1AMPRecoveryCfgPPO inherits these unchanged from X1DHStandCfgPPO.
# Independently read back from the actual control/freeze Gradmotion manifests;
# eight batches are TWO epochs times FOUR minibatches, not four times two.
PPO_EPOCHS, PPO_MINIBATCHES, PPO_ROLLOUT_STEPS = 2, 4, 24


def _number(value, name, count=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or
            not math.isfinite(value) or value < 0 or
            (count and (not isinstance(value, int)))):
        raise ValueError('Invalid deterministic mu evidence: '+name)
    return value


def _same_number(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-10):
        raise ValueError('Deterministic mu arithmetic mismatch: '+name)


def _loss_fields(report, spec, prefix=''):
    for key in LOSS_FIELDS:
        _number(report.get(prefix+key), prefix+key)
    for key in ('anchor_coef', 'temporal_coef'):
        _number(report.get(prefix+key), prefix+key)
        if report.get(prefix+key) != spec[key]:
            raise ValueError('Changed deterministic mu coefficient: '+key)
    _same_number(report[prefix+'weighted_anchor_loss'],
                 report[prefix+'anchor_loss']*spec['anchor_coef'], 'weighted_anchor_loss')
    _same_number(report[prefix+'weighted_temporal_loss'],
                 report[prefix+'temporal_loss']*spec['temporal_coef'], 'weighted_temporal_loss')
    if not spec['temporal_coef'] and report[prefix+'weighted_temporal_loss'] != 0:
        raise ValueError('Anchor-only run acquired a temporal optimization term')
    for key in ('teacher_unchanged', 'metadata_observed', 'all_finite'):
        if report.get(prefix+key) is not True:
            raise ValueError('Missing deterministic mu invariant: '+prefix+key)
    if spec.get('frozen_modules'):
        from humanoid.amp.feature_freeze import validate_gradient_report
        validate_gradient_report(report, prefix)
        for key in ('frozen_features_unchanged', 'frozen_optimizer_unchanged'):
            if report.get(prefix+key) is not True:
                raise ValueError('Missing frozen feature invariant: '+prefix+key)
    if spec.get('component_gradient_schema'):
        from humanoid.amp.feature_freeze import validate_component_gradient_report
        if report.get(prefix+'component_gradient_schema') != spec['component_gradient_schema']:
            raise ValueError('Changed weighted actor gradient schema')
        validate_component_gradient_report(report, prefix)
    return _number(report.get(prefix+'valid_triplets'), prefix+'valid_triplets', count=True)


def _component_records(report, spec, updates, start_update=2501, prefix='', formal=False):
    """Bind actual labeled minibatches to all scalar means and real PPO size."""
    if not spec.get('component_gradient_schema'):
        return None
    from humanoid.amp.feature_freeze import validate_component_gradient_records
    if report.get(prefix+'component_gradient_schema') != spec['component_gradient_schema']:
        raise ValueError('Changed weighted actor gradient schema')
    records = report.get(prefix+'component_gradient_records')
    validate_component_gradient_records(records, updates, minibatches_per_update=8,
        start_update=start_update, aggregate=report, prefix=prefix)
    if _number(report.get(prefix+'gradient_minibatches'),
               prefix+'gradient_minibatches', count=True) != 8*updates:
        raise ValueError('Missing actual weighted actor minibatches')
    # Registered native smoke/formal PPO: 24 steps, 2 epochs, 4 minibatches.
    # Batch size is evidence, not an arbitrary positive number or a valid-count
    # denominator that may silently change across records.
    expected_batch = (4096 if formal else 32)*PPO_ROLLOUT_STEPS//PPO_MINIBATCHES
    if any(record['batch_size'] != expected_batch for record in records):
        raise ValueError('Weighted actor minibatch differs from real PPO sample budget')
    return records


def compare_environment(source, target, group, smoke=False):
    mu_contract(group)  # Never let an unknown mu group fall through as control.
    return compare_control_environment(source, target, 'control', smoke)


def compare_recording_environment(source, target, group, duration, num_envs):
    """Normalize only the recorder's explicitly bounded horizon and batch size."""
    mu_contract(group)
    _number(duration, 'evaluation duration')
    _number(num_envs, 'evaluation num_envs', count=True)
    if ((duration, num_envs) not in ((1., 2), (60., 16)) or
            source['env']['num_envs'] != (32 if duration == 1. else 4096) or
            source['env']['episode_length_s'] != 60. or
            target['env']['num_envs'] != num_envs or
            target['env']['episode_length_s'] != duration+.1):
        raise ValueError('Changed deterministic mu training or recording protocol budget')
    after = copy.deepcopy(target)
    for key in ('num_envs', 'episode_length_s'):
        after['env'][key] = source['env'][key]
    return compare_environment(source, after, group)


def validate_mu_source(experiment, continuation, checkpoint):
    """Bind the frozen teacher to actual hash-bound original model tensors."""
    source = validate_mu(experiment)
    validate_mu_certificate(experiment, continuation)
    spec = experiment.cfg['mu_temporal']
    path = Path(checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['source_checkpoint_sha256']:
        raise ValueError('Independent deterministic mu source checkpoint SHA mismatch')
    state = torch.load(path, weights_only=True, map_location='cpu')
    if (state.get('amp_identity') != source.identity() or
            state.get('completed_updates') != 2500 or state.get('iter') != 2499):
        raise ValueError('Independent deterministic mu source identity/update mismatch')
    model = state['model_state_dict']
    if not model or any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all())
                        for v in model.values()):
        raise ValueError('Nonfinite or incomplete original source model')
    digest = state_fingerprint(model)
    if any(continuation.get(key) != digest for key in
           ('teacher_source_model_sha256', 'student_initial_model_sha256')):
        raise ValueError('Frozen teacher does not match independent original model tensors')
    if spec.get('frozen_modules'):
        feature_digest = state_fingerprint({name: value for name, value in model.items()
            if name.startswith(('long_history.', 'state_estimator.'))})
        if any(continuation.get(key) != feature_digest for key in
               ('frozen_features_initial_sha256', 'frozen_features_final_sha256')):
            raise ValueError('Initial frozen features not bound to original source tensors')
    return digest


def validate_mu_updates(rows, group, formal=False, report=None):
    spec = mu_contract(group)
    scalar_rows = rows
    if spec.get('component_gradient_schema'):
        # The original AMP validator intentionally accepts scalar rows only.
        # Remove exactly the two new non-scalar/string fields from that view;
        # audit both separately below, retaining every old scalar gate.
        scalar_rows = [{key: value for key, value in row.items()
            if key not in ('mu_component_gradient_schema', 'mu_component_gradient_records')}
            for row in rows]
        if any(not isinstance(value, Real) for row in scalar_rows for value in row.values()):
            raise ValueError('Non-numeric scalar in weighted actor optimization logs')
    validate_jitter_updates(scalar_rows, formal)  # Exact2501..2510/2750 and original AMP gates.
    triplets, component_records = 0, []
    for row in rows:
        count = _loss_fields(row, spec, 'mu_')
        if _number(row.get('mu_updates'), 'mu_updates', count=True) != 1:
            raise ValueError('Wrong per-update deterministic mu optimization count')
        if count == 0 and (row['mu_temporal_loss'] != 0 or row['mu_weighted_temporal_loss'] != 0):
            raise ValueError('Temporal loss observed without a valid three-frame input')
        records = _component_records(row, spec, 1, start_update=row['iteration'],
                                     prefix='mu_', formal=formal)
        if records is not None:
            if sum(record['valid_triplets'] for record in records) != PPO_EPOCHS*count:
                raise ValueError('Actual weighted actor batches lost or invented rollout triplets')
            component_records.extend(records)
        triplets += count
    if triplets <= 0:
        raise ValueError('No real contiguous deterministic mu triplets during the run')
    if spec['temporal_coef'] and not any(
            row['mu_valid_triplets'] > 0 and row['mu_weighted_temporal_loss'] > 0 and
            row['mu_aux_grad_norm'] > 0 for row in rows):
        raise ValueError('Enabled temporal term has no actual nonzero gradient evidence')
    if report is not None:
        if report['valid_triplets'] != triplets:
            raise ValueError('Deterministic mu triplet summary differs from update logs')
        for key in LOSS_FIELDS:
            _same_number(report[key], sum(row['mu_'+key] for row in rows)/len(rows), key)
        if spec.get('frozen_modules'):
            from humanoid.amp.feature_freeze import GRADIENT_FIELDS
            for key in GRADIENT_FIELDS:
                _same_number(report[key], sum(row['mu_'+key] for row in rows)/len(rows), key)
            if (report['gradient_minibatches'] != sum(row['mu_gradient_minibatches'] for row in rows) or
                    any(row.get('mu_gradient_minibatches') != 8 for row in rows)):
                raise ValueError('Missing actual every-minibatch gradient telemetry')
        records = _component_records(report, spec, len(rows), formal=formal)
        if records is not None and records != component_records:
            raise ValueError('Saved weighted actor records differ from actual ordered update logs')
    return True


def validate_frozen_checkpoint(actual, source_path, report, updates=None):
    """Independent end-vs-original CNN/ES tensor AND Adam-state proof, not flags."""
    from humanoid.amp.feature_freeze import _same
    from humanoid.amp.jitter import SOURCE_SHA
    path = Path(source_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Frozen-feature source SHA mismatch')
    source = torch.load(path, weights_only=True, map_location='cpu')
    old, new = source['model_state_dict'], actual['model_state_dict']
    names = list(old)
    # Current exact architecture has 33 parameter tensors and no mutable buffers.
    if len(names) != 33 or new.keys() != old.keys():
        raise ValueError('Frozen-feature model architecture changed')
    frozen = [name for name in names if name.startswith(('long_history.', 'state_estimator.'))]
    old_features, new_features = [{name: state[name] for name in frozen} for state in (old, new)]
    digest = state_fingerprint(old_features)
    if (len(frozen) != 16 or not _same(old_features, new_features) or
            report.get('frozen_features_initial_sha256') != digest or
            report.get('frozen_features_final_sha256') != digest):
        raise ValueError('Saved CNN/ES state differs from immutable original')
    before, after = source['optimizer_state_dict'], actual['optimizer_state_dict']
    old_slots = [p for g in before['param_groups'] for p in g['params']]
    new_slots = [p for g in after['param_groups'] for p in g['params']]
    if len(old_slots) != 33 or new_slots != old_slots:
        raise ValueError('Original full Adam parameter slots changed')
    if not _same(before['param_groups'], after['param_groups']):
        raise ValueError('Original Adam group hyperparameters/order changed')
    for name in frozen:
        slot = old_slots[names.index(name)]
        if (slot not in before['state'] or slot not in after['state'] or
                not _same(before['state'][slot], after['state'][slot])):
            raise ValueError('Saved frozen Adam moment/step differs from original: '+name)
    if updates is not None:
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            raise ValueError('Invalid independently checked update budget')
        for name in names:
            if name in frozen:
                continue
            slot = old_slots[names.index(name)]
            if (slot not in before['state'] or slot not in after['state'] or
                    float(after['state'][slot]['step']) != float(before['state'][slot]['step'])+8*updates):
                raise ValueError('Active Adam step count differs from actual update budget: '+name)
    if not _same(source['es_optimizer_state_dict'], actual['es_optimizer_state_dict']):
        raise ValueError('Standalone restored ES Adam changed despite feature freeze')
    return True


def validate_mu_loss_report(report, spec, updates, continuation, rows=None,
                            require_final_hash=True, num_envs=None, rollout_steps=None):
    if spec != mu_contract(spec.get('group')):
        raise ValueError('Changed deterministic mu audit contract')
    validate_contract_loss_report(report, spec, updates)
    if spec.get('frozen_modules'):
        if any(report.get(key) != continuation.get('frozen_features_initial_sha256')
               for key in ('frozen_features_initial_sha256', 'frozen_features_final_sha256')):
            raise ValueError('Final frozen-feature evidence differs from source restoration')
    count = _loss_fields(report, spec)
    source_hash = continuation.get('teacher_source_model_sha256')
    if (not isinstance(source_hash, str) or not re.fullmatch('[0-9a-f]{64}', source_hash) or
            continuation.get('student_initial_model_sha256') != source_hash or
            (require_final_hash and report.get('teacher_final_model_sha256') != source_hash)):
        raise ValueError('Missing or changed frozen teacher model-state fingerprint')
    diagnostics = report.get('rollout_diagnostics')
    if not isinstance(diagnostics, list) or len(diagnostics) != updates:
        raise ValueError('Incomplete deterministic mu rollout metadata diagnostics')
    observed_triplets = 0
    for item in diagnostics:
        if (item.get('metadata_observed') is not True or
                item.get('invalid_reason_counts_overlap') is not True or
                item.get('startup_steps') != spec['startup_steps']):
            raise ValueError('Changed deterministic mu rollout continuity definition')
        total, candidate, valid, invalid = [_number(item.get(key), key, count=True)
            for key in ('total_samples', 'candidate_triplets', 'valid_triplets', 'invalid_triplets')]
        reasons = item.get('invalid_reasons', {})
        expected = {'rollout_boundary', 'previous_done', 'episode_changed',
                    'tick_discontinuity', 'command_changed', 'startup'}
        if set(reasons) != expected or valid+invalid != total or valid > candidate or candidate > total:
            raise ValueError('Invalid deterministic mu continuity counts')
        for key, value in reasons.items():
            _number(value, key, count=True)
            if value > invalid or (key != 'rollout_boundary' and value > candidate):
                raise ValueError('Impossible deterministic mu rejection count')
        if (reasons['rollout_boundary'] != total-candidate or
                max(reasons.values()) > invalid or sum(reasons.values()) < invalid):
            raise ValueError('Inconsistent deterministic mu rejection coverage')
        if num_envs is not None and rollout_steps is not None:
            if total != num_envs*rollout_steps or candidate != num_envs*max(0, rollout_steps-2):
                raise ValueError('Wrong real deterministic mu rollout sample budget')
        observed_triplets += valid
    if observed_triplets != count:
        raise ValueError('Deterministic mu report lost or invented contiguous triplets')
    records = _component_records(report, spec, updates, formal=updates == 250)
    if records is not None:
        # Each valid rollout sample appears once per PPO epoch. Keep the
        # per-update reconciliation, not merely a pooled total that could hide
        # samples moved between updates with equal run aggregates.
        for index, item in enumerate(diagnostics):
            actual = records[index*8:(index+1)*8]
            if sum(record['valid_triplets'] for record in actual) != PPO_EPOCHS*item['valid_triplets']:
                raise ValueError('Saved weighted actor records disagree with rollout continuity')
    if rows is not None:
        validate_mu_updates(rows, spec['group'], formal=updates == 250, report=report)
        if any(row['mu_valid_triplets'] != item['valid_triplets']
               for row, item in zip(rows, diagnostics)):
            raise ValueError('Per-rollout mu continuity differs from update logs')
    return True


def validate_mu_checkpoint_report(actual, final_report, spec, updates, continuation):
    validate_mu_loss_report(actual, spec, updates, continuation, require_final_hash=False)
    expected = dict(final_report)
    expected.pop('teacher_final_model_sha256')
    if actual != expected:
        raise ValueError('Checkpoint mu optimization report differs from final certificate')
    return True


def validate_mu_log_identity(logs, manifest, certificate):
    decoder = json.JSONDecoder()
    start_manifest = dict(manifest)
    start_manifest.pop('completion')
    boundaries = {}
    for marker, expected in (('[f1-amp-start] ', start_manifest),
                             ('[f1-amp-complete] ', certificate)):
        if logs.count(marker) != 1:
            raise ValueError('Missing or duplicated deterministic mu log identity: '+marker)
        position = logs.index(marker)
        actual, size = decoder.raw_decode(logs[position+len(marker):])
        if actual != expected:
            raise ValueError('Deterministic mu log differs from uploaded artifact: '+marker)
        boundaries[marker] = (position, position+len(marker)+size)
    if logs.index('[f1-amp-start] ') >= logs.index('[f1-amp-complete] '):
        raise ValueError('Reversed deterministic mu start/completion records')
    if any(not (boundaries['[f1-amp-start] '][1] < match.start() <
                boundaries['[f1-amp-complete] '][0])
           for match in re.finditer(re.escape('[f1-amp-update] '), logs)):
        raise ValueError('Deterministic mu updates outside verified training records')
    return True
