"""Read-only cohort/forward gates for the independent final-head experiment.

These gates do not fit a real policy.  Formal fitting belongs to the registered
Gradmotion entry point.  A manifest flag alone is never an admission proof.
"""
import hashlib
import json

import numpy as np
import torch

from humanoid.scripts.collect_amp_head_cohort import (
    SCHEMA, SOURCE_SHA, SOURCE_MODEL_SHA, SOURCE_ENDPOINT_SHA, BUDGETS,
    SEEDS, FrozenActorCapture, audit_arrays, cohort_contract,
    fingerprint_cohort, validate_source_runtime)
from .head_smoothing import Episode
from .learnability import assert_no_domain_randomization
from .mu_temporal import state_fingerprint


def identity_digest(identity):
    if not isinstance(identity, dict) or not identity:
        raise ValueError('Missing original AMP identity')
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def audit_cohort(manifest, arrays, *, split, mode, identity, code_commit,
                 implementation_fingerprint, exclusions, source_manifest):
    """Recompute gates from all rows, including failures; never pick survivors."""
    if split not in ('train', 'validation', 'sealed') or mode not in BUDGETS:
        raise ValueError('Unregistered cohort role/budget')
    n, seconds = BUDGETS[mode]
    contract = cohort_contract(split, mode, SEEDS[split], n, seconds)
    if any(manifest.get(key) != value for key, value in contract.items()):
        raise ValueError('Cohort role/seed/budget/episode IDs changed')
    constants = dict(type=SCHEMA, source_checkpoint_sha256=SOURCE_SHA,
        source_model_state_sha256=SOURCE_MODEL_SHA, identity=identity,
        source_completed_updates=2500, code_commit=code_commit,
        implementation_fingerprint=implementation_fingerprint,
        policy_deterministic=True, parent_frozen=True, no_training=True,
        effectiveness_verified=False, dr_unlocked=False)
    if any(manifest.get(key) != value for key, value in constants.items()):
        raise ValueError('Cohort/source/code identity mismatch')
    for key, value in constants.items():
        if type(value) is bool and type(manifest.get(key)) is not bool:
            raise ValueError('Cohort evidence flags must be genuine booleans')
    if manifest.get('source_regression', {}).get('sha256') != SOURCE_ENDPOINT_SHA:
        raise ValueError('Cohort omitted the immutable original32 exclusions')
    assert_no_domain_randomization(manifest['environment'])
    runtime = manifest['runtime']
    runtime_proof = validate_source_runtime(manifest['environment'], runtime,
                                           source_manifest, contract)
    if runtime_proof != manifest.get('source_runtime_proof'):
        raise ValueError('Source physics/configuration proof differs from actual readback')
    if (not np.isclose(runtime['control_dt'], .01, rtol=1e-6, atol=1e-10)
            or not np.isclose(runtime['physics_dt'], .001, rtol=1e-6, atol=1e-10)):
        raise ValueError('Wrong native physical/control timing')
    ready, real_frames, history, episodes = audit_arrays(arrays, contract)
    fingerprints, dedupe = fingerprint_cohort(arrays, contract, ready, exclusions)
    if (fingerprints != manifest.get('fingerprints')
            or history != manifest.get('history_proof')
            or episodes != manifest.get('episodes')
            or dedupe != manifest.get('deduplication')):
        raise ValueError('Cohort evidence differs from recomputed actual arrays')
    complete = all(row['complete'] for row in episodes)
    fit_eligible = bool(mode == 'formal' and split != 'sealed' and complete
                        and history['exact'] and dedupe['passed'])
    if manifest.get('fit_eligible') is not fit_eligible:
        raise ValueError('Cohort admission differs from actual complete episodes')
    smoke_eligible = bool(mode == 'smoke' and split != 'sealed' and complete
                          and history['exact'] and dedupe['passed'])
    return dict(contract=contract, ready=ready, real_frames=real_frames,
                history=history, episodes=episodes, fingerprints=fingerprints,
                deduplication=dedupe, fit_eligible=fit_eligible,
                smoke_eligible=smoke_eligible)


def forward_parity(policy, arrays, *, action_clip, batch_size=256,
                   expected_model_fingerprint=SOURCE_MODEL_SHA):
    """Verify EVERY stored actual observation against the original frozen actor.

The hook observes one ordinary deterministic act_inference call per batch. It
does not inject recorded hidden vectors, actions, or reset/replay observations.
The original collector used one call per real native control decision; this
second, independent read-only forward audits its exported rows.
"""
    if (type(batch_size) is not int or batch_size < 1 or isinstance(action_clip, bool)
            or not np.isfinite(action_clip) or action_clip <= 0):
        raise ValueError('Invalid source parity bounds')
    before = state_fingerprint(policy.state_dict())
    if before != expected_model_fingerprint or policy.training:
        raise ValueError('Parity requires the frozen source in inference mode')
    actual_full = arrays['reference_full_obs']
    if not isinstance(actual_full, np.ndarray) or actual_full.ndim != 3:
        raise ValueError('Missing actual time/environment observation rows')
    leading = actual_full.shape[:2]
    for key, width in (('full_obs', 3102), ('actor_hidden', 128),
                       ('raw_mu', 12), ('action', 12)):
        value = arrays['reference_'+key]
        if (not isinstance(value, np.ndarray) or value.shape != leading+(width,)
                or value.dtype != np.float32):
            raise ValueError('Missing/misaligned actual forward rows: '+key)
    full = actual_full.reshape(-1, 3102)
    hidden = arrays['reference_actor_hidden'].reshape(-1, 128)
    mu = arrays['reference_raw_mu'].reshape(-1, 12)
    applied = arrays['reference_action'].reshape(-1, 12)
    if not len(full) or any(not np.isfinite(value).all() for value in (full, hidden, mu, applied)):
        raise ValueError('Missing/nonfinite actual source forward rows')
    if not np.array_equal(np.clip(mu, -action_clip, action_clip), applied):
        raise ValueError('Actual applied native action is not the clipped recorded mean')
    device = next(policy.parameters()).device
    max_hidden, max_mu = 0., 0.
    hook = FrozenActorCapture(policy)
    try:
        with torch.no_grad():
            for start in range(0, len(full), batch_size):
                end = min(start+batch_size, len(full))
                observation = torch.from_numpy(full[start:end].copy()).to(device)
                actual_mu, actual_hidden = hook.inference(observation)
                max_hidden = max(max_hidden, float(np.max(np.abs(
                    actual_hidden.cpu().numpy()-hidden[start:end]))))
                max_mu = max(max_mu, float(np.max(np.abs(
                    actual_mu.cpu().numpy()-mu[start:end]))))
    finally:
        hook.close()
    if state_fingerprint(policy.state_dict()) != before:
        raise ValueError('Read-only source forward changed the policy')
    error = max(max_hidden, max_mu)
    if not error < .001:
        raise ValueError('Source full-input/hidden/action forward parity failed')
    return dict(full_obs_verified=True, hidden_verified=True, action_verified=True,
                max_abs_error=error, max_hidden_error=max_hidden,
                max_raw_mu_error=max_mu, actual_rows=len(full),
                forward_batches=hook.calls, policy_unchanged=True,
                observations='actual archived 66x47 native inputs, not reconstructed')


def fit_episodes(manifest, arrays, audit, *, identity):
    """Build whole first-episode designs only AFTER all cohort gates pass."""
    if not (audit['fit_eligible'] or audit['smoke_eligible']):
        raise ValueError('Incomplete/duplicate/sealed cohort cannot be fitted')
    if manifest['identity'] != identity or manifest['split'] == 'sealed':
        raise ValueError('Cannot substitute a sealed or unrelated fitting split')
    episodes = []
    for index, label in enumerate(audit['contract']['episode_ids']):
        mask = audit['ready'][:, index]
        ticks = arrays['reference_tick'][mask, index]
        hidden = arrays['reference_actor_hidden'][mask, index]
        design = np.concatenate((hidden.astype(np.float64),
                                np.ones((len(hidden), 1))), axis=1)
        episodes.append(Episode(episode_id=label, cohort_id=label,
            design=design, ticks=ticks.copy(), seed=manifest['seed'], env_id=index,
            source_identity=identity_digest(identity), dt=.01, history_frames=66,
            complete=True))
    return episodes


def exclusion_add_cohort(index, audit):
    """Use recomputed fingerprints, never unverified recorded claims."""
    from humanoid.scripts.collect_amp_head_cohort import _index_add
    for row in audit['fingerprints']:
        for kind in ('initial_state', 'initial_full_observation'):
            _index_add(index, kind, row[kind], row['episode_id'])
        for item in row['ready_full_history']:
            _index_add(index, 'ready_full_history', item['sha256'], row['episode_id'])
    return index
