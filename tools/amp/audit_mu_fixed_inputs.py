"""Read-only deterministic policies on the original, parity-checked inputs.

No simulator, optimizer step, parameter mutation or candidate state trajectory
is used. The report describes policy functions on recorded original states;
it does not establish a cause of falls or physical acceleration improvement.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.dry_run import cpu_ppo_classes
from humanoid.amp.mu_temporal import (
    ACCELERATION_SCALES, BASELINE_BUNDLE_SHA, SOURCE_CONFIG, SOURCE_SHA,
    mu_contract, state_fingerprint, validate_mu_certificate,
)
from humanoid.amp.scaled_experiment import ScaledExperiment
from tools.amp.audit_recorded_estimator import single_observations
from tools.amp.inspect_rollout import episode, load_bundle


SPANS = dict(post2s=(2., 60.), near5s=(4., 6.),
             near29to30s=(28., 31.), near48s=(47., 49.))
RUNS = ('source', '081', '082')
FORMAL_COMMIT = 'dd24051df25867f49e6fd1b7ff5e7b6ffdff4fb3'
GROUPS = {'081': 'anchor', '082': 'temporal'}
GROUP_CHOICES = {'081': ('anchor', 'freeze_anchor'),
                 '082': ('temporal', 'freeze_temporal')}
LEGACY_GRADIENT_LIMITATION = (
    'Training auxiliary gradient logs are first-minibatch diagnostics; main PPO gradients are not recorded.')
METRICS = ('normalized_temporal_loss', 'qmu_second_difference_rms_rad_s2',
           'anchor_pd_target_rms_rad')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def action_parity(raw_mu, observed, clip, tolerance=.001):
    raw, actual = np.asarray(raw_mu), np.asarray(observed)
    if (raw.ndim != 2 or raw.shape != actual.shape or not len(raw) or
            not np.isfinite(raw).all() or not np.isfinite(actual).all() or
            not np.isfinite(clip) or clip <= 0):
        raise ValueError('Invalid source parity arrays or clipping limit')
    error = raw.clip(-clip, clip)-actual
    maximum = float(np.abs(error).max())
    result = dict(sampled_transitions=len(raw), stride_ticks=1,
        max_action_abs_error=maximum, action_rmse=float(np.sqrt((error**2).mean())),
        action_parity_tolerance=tolerance, passed=maximum < tolerance,
        source_raw_mu_clip_count=int((np.abs(raw) > clip).sum()),
        source_raw_mu_boundary_count=int((np.abs(raw) >= clip).sum()))
    if not result['passed']:
        raise ValueError('Source action parity failed: '+json.dumps(result))
    return result


def validate_fixed_source_shas(endpoint_sha, checkpoint_sha):
    if endpoint_sha != BASELINE_BUNDLE_SHA or checkpoint_sha != SOURCE_SHA:
        raise ValueError('Fixed source bundle/checkpoint SHA mismatch')


def candidate_groups(anchor_group='anchor', temporal_group='temporal', expected_commit=FORMAL_COMMIT):
    """Explicit role-preserving opt-in; unchanged legacy defaults stay hard-bound."""
    groups = {'081': anchor_group, '082': temporal_group}
    if any(group not in GROUP_CHOICES[run] for run, group in groups.items()):
        raise ValueError('Candidate groups must preserve anchor/temporal roles')
    if not isinstance(expected_commit, str) or not re.fullmatch('[0-9a-f]{40}', expected_commit):
        raise ValueError('Expected candidate formal commit must be a full lowercase Git SHA')
    return groups


def validate_candidate_binding(run, checkpoint_identity, training, expected_identity, source_model_hash,
                               group=None, expected_commit=FORMAL_COMMIT):
    """Pure guard: labels, full mu contract and formal commit cannot be swapped."""
    if run not in GROUPS:
        raise ValueError('Unknown fixed candidate label')
    group = GROUPS[run] if group is None else group
    candidate_groups(group if run == '081' else 'anchor',
                     group if run == '082' else 'temporal', expected_commit)
    proof=training.get('continuation') or {}
    expected=mu_contract(group)
    if (checkpoint_identity != expected_identity or training.get('identity') != expected_identity or
            training.get('code_commit') != expected_commit or proof.get('mu_contract') != expected or
            proof.get('source_sha256') != SOURCE_SHA or proof.get('source_completed_updates') != 2500 or
            proof.get('target_completed_updates') != 2750 or proof.get('teacher_frozen') is not True or
            proof.get('teacher_source_model_sha256') != source_model_hash or
            proof.get('student_initial_model_sha256') != source_model_hash):
        raise ValueError('Candidate full identity/contract/formal-commit binding mismatch: '+run)


def validate_freeze_candidate(checkpoint, training, source_path, experiment):
    """New groups need actual checkpoint/Adam evidence, not self-reported flags."""
    spec = experiment.cfg['mu_temporal']
    if not spec.get('frozen_modules'):
        return None
    from tools.amp.mu_audit import (validate_frozen_checkpoint,
        validate_mu_checkpoint_report, validate_mu_loss_report)
    completion = training.get('completion') or {}
    proof = training.get('continuation') or {}
    if (training.get('mode') != 'formal' or training.get('updates') != 250 or
            training.get('num_envs') != 4096 or completion.get('updates') != 250 or
            completion.get('num_envs') != 4096 or
            completion.get('identity') != training.get('identity') or
            completion.get('code_commit') != training.get('code_commit') or
            completion.get('continuation') != proof):
        raise ValueError('Freeze fixed-input audit requires the complete original formal protocol')
    report = completion.get('mu_loss_report') or {}
    validate_mu_loss_report(report, spec, 250, proof, num_envs=4096,
        rollout_steps=training['ppo_config']['runner']['num_steps_per_env'])
    validate_mu_checkpoint_report(checkpoint.get('mu_loss_report') or {}, report, spec, 250, proof)
    validate_frozen_checkpoint(checkpoint, source_path, report, updates=250)
    return dict(frozen_feature_checkpoint_verified=True, additional_updates=250,
        source_features_sha256=report['frozen_features_initial_sha256'],
        candidate_features_sha256=report['frozen_features_final_sha256'],
        teacher_final_model_sha256=report['teacher_final_model_sha256'],
        checkpoint_report_matches_completion=True,
        frozen_parameters_and_adam_unchanged=True, active_adam_steps_verified=True)


def gradient_evidence_limitations(groups):
    old = [run for run, group in groups.items() if not group.startswith('freeze_')]
    frozen = [run for run, group in groups.items() if group.startswith('freeze_')]
    result = []
    if old:
        result.append(LEGACY_GRADIENT_LIMITATION if not frozen else
            'Legacy candidates '+','.join(old)+': '+LEGACY_GRADIENT_LIMITATION)
    if frozen:
        result.append('Freeze candidates '+','.join(frozen)+
            ': main/auxiliary/ES norms and actor gradient cosine are computed every minibatch; '
            'saved all-minibatch diagnostics are minibatch/update aggregates, not individual gradient vectors '
            'or a causal proof. aux_grad_norm_all_batches is the aggregate auxiliary metric; '
            'the retained aux_grad_norm is still the legacy first-minibatch metric.')
    return result


def validate_complete_cohorts(rows):
    expected={(mode, index) for mode in ('standing', 'reference') for index in range(16)}
    if len(rows) != 32*len(SPANS)*len(RUNS):
        raise ValueError('Fixed source comparison requires exactly384 complete rows')
    for span in SPANS:
        for run in RUNS:
            selected=[row for row in rows if row['span'] == span and row['run'] == run]
            keys={(row['mode'], row['env']) for row in selected}
            if (len(selected) != 32 or keys != expected or
                    any(row['metrics']['complete_window'] is not True for row in selected)):
                raise ValueError('Incomplete, duplicate or mismatched fixed initial-state cohort: '+span+'/'+run)


def span_metrics(time, raw_mu, source_mu, scales, start, end, clip):
    """Every three-point stencil is wholly inside the half-open interval."""
    time = np.asarray(time, dtype=np.float64)
    mu, source = np.asarray(raw_mu, dtype=np.float64), np.asarray(source_mu, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if (mu.ndim != 2 or mu.shape != source.shape or len(time) != len(mu) or
            scales.shape != (mu.shape[1],) or not np.isfinite(mu).all() or
            not np.isfinite(source).all() or not np.isfinite(time).all() or
            not np.isfinite(scales).all() or (scales <= 0).any() or
            not end > start or not np.isfinite(clip) or clip <= 0):
        raise ValueError('Invalid raw-mean span inputs')
    if len(time) > 1 and not np.allclose(np.diff(time), .01, atol=1e-7, rtol=0):
        raise ValueError('Fixed inputs must be consecutive real 100Hz records')
    inside = (time >= start-1e-8) & (time < end-1e-8)
    stencil = inside[2:] & inside[1:-1] & inside[:-2]
    requested = int(round((end-start)*100))
    selected = mu[inside]
    result = dict(requested_interval_s=[start, end], decision_frames=int(inside.sum()),
        triplets=int(stencil.sum()), requested_decision_frames=requested,
        requested_triplets=max(0, requested-2),
        complete_window=int(inside.sum()) == requested and int(stencil.sum()) == max(0, requested-2),
        raw_mu_available=True, raw_mu_metrics_unclipped=True,
        raw_mu_clip_count=int((np.abs(selected) > clip).sum()),
        raw_mu_boundary_count=int((np.abs(selected) >= clip).sum()),
        raw_mu_clip_count_per_joint=(np.abs(selected) > clip).sum(axis=0).tolist(),
        raw_mu_min_per_joint=None if not len(selected) else selected.min(axis=0).tolist(),
        raw_mu_max_per_joint=None if not len(selected) else selected.max(axis=0).tolist(),
        anchor_pd_target_rms_rad=None if not len(selected) else
            float(.5*np.sqrt(((selected-source[inside])**2).mean())),
        normalized_temporal_loss=None, qmu_second_difference_rms_rad_s2=None,
        qmu_second_difference_rms_per_joint_rad_s2=None,
        qmu_second_difference_absolute_p95_per_joint_rad_s2=None,
        normalized_temporal_loss_per_joint=None)
    if stencil.any():
        values=.5*np.diff(mu, n=2, axis=0)[stencil]/.01**2
        normalized=(values/scales)**2
        result.update(normalized_temporal_loss=float(normalized.mean()),
            qmu_second_difference_rms_rad_s2=float(np.sqrt((values**2).mean())),
            qmu_second_difference_rms_per_joint_rad_s2=np.sqrt((values**2).mean(axis=0)).tolist(),
            qmu_second_difference_absolute_p95_per_joint_rad_s2=np.percentile(np.abs(values), 95, axis=0).tolist(),
            normalized_temporal_loss_per_joint=normalized.mean(axis=0).tolist())
    return result


def equal_initial_means(rows):
    result={}
    for span in SPANS:
        result[span]={}
        for run in RUNS:
            selected=[row for row in rows if row['span'] == span and row['run'] == run]
            result[span][run]={}
            for metric in METRICS:
                available=[row for row in selected if row['metrics'][metric] is not None]
                result[span][run][metric]=dict(
                    mean=float(np.mean([row['metrics'][metric] for row in available])) if available else None,
                    contributing_initial_states=len(available),
                    initial_states=[dict(mode=row['mode'], env=row['env']) for row in available])
    return result


def write_report(directory, report):
    directory=Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory/'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)


def checkpoint_statistics(states, checkpoints, joint_names):
    result={}
    source=states['source']
    for run in RUNS:
        state=states[run]
        drift={}
        for group in ('long_history', 'state_estimator', 'actor', 'critic', 'std'):
            keys=[key for key in source if key == group or key.startswith(group+'.')]
            original=torch.cat([source[key].double().reshape(-1) for key in keys])
            delta=torch.cat([(state[key]-source[key]).double().reshape(-1) for key in keys])
            drift[group]=dict(tensors=len(keys), numel=len(delta),
                relative_l2_pct=float(100*delta.norm()/original.norm()),
                delta_rms=float(delta.square().mean().sqrt()), delta_max_abs=float(delta.abs().max()))
        std=state['std'].double()
        optimizers={}
        for key in ('optimizer_state_dict', 'es_optimizer_state_dict', 'amp_optimizer_state_dict'):
            optimizer=checkpoints[run][key]
            steps=[float(value['step']) for value in optimizer['state'].values() if 'step' in value]
            optimizers[key]=dict(lr=[group['lr'] for group in optimizer['param_groups']],
                state_entries=len(optimizer['state']), step_values=sorted(set(steps)))
        result[run]=dict(parameter_drift_descriptive_only=drift,
            std_by_joint=dict(zip(joint_names, std.tolist())), std_min=float(std.min()),
            std_max=float(std.max()), std_mean=float(std.mean()), optimizers=optimizers,
            mu_loss_report=checkpoints[run].get('mu_loss_report'))
    return result


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    old=ROOT/'../f1-amp-sustain/outputs/amp-sustain/TASK_20260926_110'
    current=ROOT/'outputs/amp-mu-temporal'
    parser.add_argument('--source-bundle', type=Path, default=old/'model_9902500.pt')
    for run, directory, number in (('source', old, 2500),
            ('081', current/'TASK_20261002_081', 2750), ('082', current/'TASK_20261002_082', 2750)):
        parser.add_argument('--'+run+'-checkpoint', type=Path, default=directory/('model_880%d.pt' % number))
        parser.add_argument('--'+run+'-training-manifest', type=Path, default=directory/'model_amp_manifest.pt')
    parser.add_argument('--anchor-group', choices=GROUP_CHOICES['081'], default=GROUPS['081'])
    parser.add_argument('--temporal-group', choices=GROUP_CHOICES['082'], default=GROUPS['082'])
    parser.add_argument('--expected-commit', default=FORMAL_COMMIT,
        help='Exact formal training commit; defaults to the original 081/082 commit')
    parser.add_argument('--output', type=Path, required=True)
    return parser


def main():
    args=build_parser().parse_args()
    groups=candidate_groups(args.anchor_group, args.temporal_group, args.expected_commit)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing diagnostic directory: '+str(args.output))
    torch.set_num_threads(2)
    endpoint_sha=sha256_file(args.source_bundle)
    validate_fixed_source_shas(endpoint_sha, sha256_file(args.source_checkpoint))
    manifest, arrays=load_bundle(args.source_bundle)
    if (manifest['modes'] != ['standing', 'reference'] or manifest['num_envs'] != 16 or
            not manifest['policy_deterministic'] or manifest['fps'] != 100):
        raise ValueError('Requires the complete original deterministic 32-initial-state cohort')
    checkpoints, training, provenance={}, {}, {}
    for run in RUNS:
        checkpoint_path=getattr(args, run+'_checkpoint')
        training_path=getattr(args, run+'_training_manifest')
        checkpoint=torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        train=json.loads(torch.load(training_path, map_location='cpu', weights_only=True)['manifest_json'])
        expected=2500 if run == 'source' else 2750
        if (checkpoint['amp_identity'] != train['identity'] or
                checkpoint['completed_updates'] != expected or checkpoint['iter'] != expected-1):
            raise ValueError('Wrong checkpoint/training identity or update count: '+run)
        checkpoints[run], training[run]=checkpoint, train
        provenance[run]=dict(checkpoint_path=str(checkpoint_path.resolve()),
            checkpoint_sha256=sha256_file(checkpoint_path),
            training_manifest_path=str(training_path.resolve()), training_manifest_sha256=sha256_file(training_path),
            identity=checkpoint['amp_identity'], completed_updates=checkpoint['completed_updates'],
            full_training_manifest=train)
    source_sha=provenance['source']['checkpoint_sha256']
    validate_fixed_source_shas(endpoint_sha, source_sha)
    if manifest['checkpoint_sha256'] != source_sha or manifest['identity'] != training['source']['identity']:
        raise ValueError('Original observation endpoint is not bound to the source checkpoint')
    experiments={'source': ScaledExperiment(ROOT, ROOT/SOURCE_CONFIG)}
    for run, group in groups.items():
        experiments[run]=ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_mu_'+group+'.json'))
    expected_identities={run: experiment.identity() for run, experiment in experiments.items()}
    if checkpoints['source']['amp_identity'] != expected_identities['source']:
        raise ValueError('Original policy identity does not match its fixed ScaledExperiment')
    states={run: checkpoints[run]['model_state_dict'] for run in RUNS}
    source_hash=state_fingerprint(states['source'])
    frozen_checks={}
    for run in RUNS:
        state=states[run]
        if (state.keys() != states['source'].keys() or
                any(state[key].shape != states['source'][key].shape for key in state) or
                not all(bool(torch.isfinite(value).all()) for value in state.values()) or
                training[run]['ppo_config']['policy'] != training['source']['ppo_config']['policy']):
            raise ValueError('Incompatible or nonfinite full inference policy: '+run)
        if run != 'source':
            proof=training[run]['continuation']
            validate_candidate_binding(run, checkpoints[run]['amp_identity'], training[run],
                expected_identities[run], source_hash, groups[run], args.expected_commit)
            validate_mu_certificate(experiments[run], proof)
            if (proof['source_sha256'] != source_sha or proof['source_completed_updates'] != 2500 or
                    proof['target_completed_updates'] != 2750 or not proof['teacher_frozen'] or
                    proof['teacher_source_model_sha256'] != source_hash or
                    proof['student_initial_model_sha256'] != source_hash or
                    proof['mu_contract']['acceleration_scales_rad_s2'] != ACCELERATION_SCALES):
                raise ValueError('Candidate source/teacher/fixed-scale provenance mismatch: '+run)
            checked=validate_freeze_candidate(checkpoints[run], training[run],
                                               args.source_checkpoint, experiments[run])
            if checked is not None:
                frozen_checks[run]=checked
    cls, _=cpu_ppo_classes(ROOT)
    cfg=manifest['environment']['env']
    if manifest['environment']['control']['action_scale'] != .5:
        raise ValueError('Unexpected PD action scale')
    clip=manifest['environment']['normalization']['clip_actions']
    scales=np.array([ACCELERATION_SCALES[name] for name in manifest['dof_names']])
    policies, initial_hashes={}, {}
    for run in RUNS:
        with contextlib.redirect_stdout(io.StringIO()):
            policy=cls(cfg['num_single_obs']*cfg['short_frame_stack'], cfg['num_single_obs'],
                cfg['num_privileged_obs'], cfg['num_actions'], **training[run]['ppo_config']['policy'])
        policy.load_state_dict(states[run], strict=True)
        policy.eval()
        policies[run]=policy
        initial_hashes[run]=state_fingerprint(policy.state_dict())
    rows, parities=[], []
    for mode in manifest['modes']:
        for index in range(manifest['num_envs']):
            data=episode(arrays, mode, index)
            if (len(data['time']) != 6000 or data['initial']['failure'] or data['failure'].any()):
                raise ValueError('Original fixed-input cohort is not a complete 60s surviving prefix')
            obs=single_observations(data, manifest)
            ends=torch.arange(65, len(obs)-1)
            outputs={run: [] for run in RUNS}
            with torch.no_grad():
                for batch in ends.split(512):
                    history=obs[batch[:, None]+torch.arange(-65, 1)].reshape(len(batch), -1)
                    for run in RUNS:
                        outputs[run].append(policies[run].act_inference(history))
            outputs={run: torch.cat(values).numpy() for run, values in outputs.items()}
            parity=action_parity(outputs['source'], data['action'][ends.numpy()+1], clip)
            parities.append(dict(mode=mode, env=index, **parity))
            time=data['time'][ends.numpy()]
            for span, (start, end) in SPANS.items():
                for run in RUNS:
                    metrics=span_metrics(time, outputs[run], outputs['source'], scales, start, end, clip)
                    rows.append(dict(mode=mode, env=index, span=span, run=run, metrics=metrics))
    validate_complete_cohorts(rows)
    unchanged={run: state_fingerprint(policy.state_dict()) == initial_hashes[run] and
        all(parameter.grad is None for parameter in policy.parameters()) for run, policy in policies.items()}
    if not all(unchanged.values()) or len(parities) != 32:
        raise ValueError('Read-only policy/cohort invariant failed')
    dependencies=('tools/amp/audit_mu_fixed_inputs.py', 'tools/amp/audit_recorded_estimator.py',
        'tools/amp/inspect_rollout.py', 'humanoid/amp/dry_run.py', 'humanoid/amp/mu_temporal.py',
        'humanoid/amp/jitter.py', 'humanoid/amp/scaled_experiment.py',
        'humanoid/amp/dataset.py', 'humanoid/amp/features.py',
        'humanoid/algo/ppo/actor_critic_dh.py', 'humanoid/envs/base/legged_robot.py')
    if frozen_checks:
        dependencies+=('tools/amp/mu_audit.py', 'humanoid/amp/feature_freeze.py')
    report=dict(schema_version=2,
        purpose='Fixed original observations through unchanged complete deterministic policies',
        source_endpoint=dict(path=str(args.source_bundle.resolve()), sha256=endpoint_sha,
            full_evaluation_manifest=manifest), checkpoints=provenance,
        local_code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
        source_file_sha256={name: sha256_file(ROOT/name) for name in dependencies},
        script_sha256=sha256_file(__file__),
        fixed_source_binding=dict(expected_source_endpoint_sha256=BASELINE_BUNDLE_SHA,
            expected_source_checkpoint_sha256=SOURCE_SHA, candidate_formal_commit=args.expected_commit,
            expected_scaled_experiment_identities=expected_identities,
            source_and_candidates_verified=True, complete_unique32_per_span_run_verified=True),
        protocol=dict(initial_states=32, modes=manifest['modes'], environments_per_mode=16,
            control_dt=.01, action_scale=.5, history_frames=66, history_width=47,
            first_history_decision_time_s=.66, last_history_decision_time_s=59.99,
            source_action_comparison='History ending at recorded k predicts the action on transition k+1',
            time_labels='Decision time; all three decisions must lie in half-open[start,end)',
            fixed_normalization='Original per-joint pooled P95 scales; never refitted to candidate output',
            fixed_acceleration_scales_rad_s2=dict(zip(manifest['dof_names'], scales.tolist())),
            raw_mu='Unclipped act_inference outputs in memory; summary metrics, P95 and clip counts preserved',
            aggregation='Each available initial condition has equal weight; null remains missing',
            no_simulation=True, no_optimizer_steps=True, parameters_and_grad_unchanged=unchanged),
        source_action_parity=dict(all32_passed=True,
            max_action_abs_error=max(row['max_action_abs_error'] for row in parities), rows=parities),
        checkpoint_statistics=checkpoint_statistics(states, checkpoints, manifest['dof_names']),
        rows=rows, equal_initial_means=equal_initial_means(rows),
        limitations=[
            'Policy function inference on original observations is not a closed-loop dynamics intervention.',
            'PD command second differences are not actual joint acceleration or a safety threshold.',
            'No candidate trajectories, physical1kHz waveforms, falls or survival are measured here.',
            'Parameter L2 changes and optimizer moments are descriptive, not a proven causal mechanism.',
            *gradient_evidence_limitations(groups)])
    if frozen_checks:
        report['fixed_source_binding'].update(candidate_groups=groups,
            independent_frozen_candidate_checks=frozen_checks)
    write_report(args.output, report)
    print(json.dumps(dict(output=str((args.output/'report.json').resolve()),
        source_action_parity_max=report['source_action_parity']['max_action_abs_error'],
        fixed_source_post2s={run: {key: result['mean'] for key, result in
            report['equal_initial_means']['post2s'][run].items()} for run in RUNS})))


if __name__ == '__main__':
    main()
