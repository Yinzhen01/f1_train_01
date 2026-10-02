"""Hash-bound control2500 continuation: frozen anchor vs deterministic mu loss.

The auxiliary loss differentiates policy output on contiguous recorded inputs.
It is not a dynamics derivative, execution filter, acceleration safety limit, or
an effectiveness certificate. Simulator/controller/reward semantics stay fixed.
"""
import copy
import hashlib
import math
import re
from pathlib import Path

import torch

from .jitter import SOURCE_CONFIG, SOURCE_SHA, validate_jitter_diagnostics
from .refinement import restore_learning_state
from .scaled_experiment import ScaledExperiment
from .sustain import apply_sustain_config

MU_SUBGROUPS = ('anchor', 'temporal', 'freeze_anchor', 'freeze_temporal')
MU_GROUPS = tuple('mu_'+group for group in MU_SUBGROUPS)
BASELINE_BUNDLE_SHA = 'de77d4a0c85cd581ae9dc9418d3fbce144730749b9f746f93b9671ec122a833b'
BASELINE_ANALYSIS_SHA = 'eb9002c5c5798ddb5277e65dd90f19f95c67d81385f17af59ab99ee41ef20002'
ACCELERATION_SCALES = {
    'left_hip_pitch_joint': 2111.67649924755,
    'left_hip_roll_joint': 1670.7300096750228,
    'left_hip_yaw_joint': 923.8082990050318,
    'left_knee_pitch_joint': 900.609664618969,
    'left_ankle_pitch_joint': 2332.6044082641597,
    'left_ankle_roll_joint': 735.3416234254836,
    'right_hip_pitch_joint': 2410.4408025741564,
    'right_hip_roll_joint': 1916.0323701798916,
    'right_hip_yaw_joint': 1223.1887057423587,
    'right_knee_pitch_joint': 1150.2685584127903,
    'right_ankle_pitch_joint': 2849.940061569213,
    'right_ankle_roll_joint': 1157.8007414937006,
}


def mu_contract(group):
    if group not in MU_SUBGROUPS:
        raise ValueError('Unknown deterministic mu experiment')
    result = dict(group=group, source_task='TASK_20260926_110',
        source_config=SOURCE_CONFIG, source_checkpoint_sha256=SOURCE_SHA,
        source_completed_updates=2500, learning_rate=5e-5,
        episode_length_s=60., style_floor=0., bridge_gradient_penalty=1.,
        smoothness_scale=-.02, anchor_coef=1.,
        temporal_coef=.01 if group.endswith('temporal') else 0.,
        control_dt=.01, action_scale=.5, startup_steps=66,
        teacher='frozen_full_source_actor_cnn_state_estimator',
        temporal_definition='mean(((action_scale*(mu_t-2*mu_t1+mu_t2)/dt**2)/scale_joint)**2)',
        continuity='same_episode_ticks_commands_all_three_no_prior_done_same_rollout',
        baseline_bundle_sha256=BASELINE_BUNDLE_SHA,
        baseline_analysis_sha256=BASELINE_ANALYSIS_SHA,
        normalization='pooled_32_initial_states_post2s_absolute_P95_qmu_second_difference',
        acceleration_scales_rad_s2=copy.deepcopy(ACCELERATION_SCALES))
    if group.startswith('freeze_'):
        result.update(frozen_modules=['long_history', 'state_estimator'],
            source_model_state_sha256='c06fbd97e32a483d46a2738014215de2721ad7db4b2b86547645ab72d4c555bc',
            source_feature_state_sha256='29a49a9c8b27e839aefdfa785945f4a57b8b5635468fa81e3bc4d91f090cf9bc',
            trainable_modules=['actor', 'critic', 'std'],
            optimizer_freeze='retain_all_original_slots_and_moments_skip_grad_none',
            gradient_diagnostics='every_minibatch_main_aux_es_actor_cosine_before_global_clip',
            gradient_diagnostics_changes_update=False)
    return result


def apply_mu_config(cfg, group):
    mu_contract(group)
    # Exactly source dynamics, observations, controller and environment rewards.
    return apply_sustain_config(cfg, 'control')


def validate_mu(experiment):
    spec = mu_contract(experiment.cfg.get('mu_temporal', {}).get('group'))
    if experiment.cfg['mu_temporal'] != spec:
        raise ValueError('Deterministic mu contract changed')
    source = ScaledExperiment(experiment.repo, experiment.repo/SOURCE_CONFIG)
    expected = copy.deepcopy(source.cfg)
    expected.pop('sustain')
    expected.update(experiment='f1_amp_walk02_mu_'+spec['group'],
        mu_temporal=spec, evaluation_updates=[2750], max_experiment_rounds=None)
    actual = copy.deepcopy(experiment.cfg)
    for key in ('scope', 'known_risks'):
        expected.pop(key, None)
        actual.pop(key, None)
    if expected != actual:
        raise ValueError('Change outside bounded deterministic mu intervention')
    return source


def state_fingerprint(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode()+b'\0'+str(value.dtype).encode()+b'\0')
        digest.update(str(tuple(value.shape)).encode()+b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def mu_source(cfg):
    spec = cfg['mu_temporal']
    return dict(source_task=spec['source_task'], source_sha256=SOURCE_SHA,
        source_completed_updates=2500, target_completed_updates=2750,
        kind='bounded_deterministic_mu', mu_contract=copy.deepcopy(spec))


def validate_mu_certificate(experiment, continuation):
    validate_mu(experiment)
    if any(continuation.get(k) != v for k, v in mu_source(experiment.cfg).items()):
        raise ValueError('Wrong deterministic mu continuation proof')
    if (continuation.get('actor_and_discriminator_restored') is not True or
            continuation.get('optimizers_restored') is not True or
            continuation.get('replay_windows') != 50000 or
            continuation.get('teacher_frozen') is not True or
            continuation.get('teacher_source_model_sha256') != continuation.get('student_initial_model_sha256') or
            len(continuation.get('teacher_source_model_sha256', '')) != 64):
        raise ValueError('Incomplete deterministic mu source/teacher restoration')
    if experiment.cfg['mu_temporal'].get('frozen_modules'):
        spec = experiment.cfg['mu_temporal']
        validate_feature_proof(continuation, spec['source_feature_state_sha256'])
        if continuation['teacher_source_model_sha256'] != spec['source_model_state_sha256']:
            raise ValueError('Frozen-feature teacher differs from exact source model tensors')
    return True


def validate_feature_proof(report, expected_sha=None):
    if (report.get('frozen_features_unchanged') is not True or
            report.get('frozen_optimizer_unchanged') is not True or
            report.get('frozen_parameter_count') != 16 or
            report.get('trainable_parameter_count') != 17 or
            len(report.get('frozen_features_initial_sha256', '')) != 64 or
            report.get('frozen_features_initial_sha256') != report.get('frozen_features_final_sha256') or
            (expected_sha is not None and report.get('frozen_features_initial_sha256') != expected_sha)):
        raise ValueError('Missing exact frozen CNN/ES and Adam-state proof')
    return True


def validate_feature_smoke_admission(experiment, certificate):
    """Formal-only admission AFTER external checkpoint/log/source inspection.

    The training smoke self-check must not forge independent evidence. Its raw
    certificate is deliberately insufficient until verify_direction_smoke runs.
    """
    validate_mu(experiment)
    if not experiment.cfg['mu_temporal'].get('frozen_modules'):
        raise ValueError('Feature admission is restricted to registered freeze groups')
    required = ('independent_artifacts_verified', 'frozen_feature_checkpoint_verified',
        'mu_optimization_log_verified', 'frozen_teacher_state_verified',
        'source_environment_equivalence_verified', 'candidate_native_substeps_verified')
    if any(certificate.get(key) is not True for key in required):
        raise ValueError('Formal feature-freeze run requires independently audited real smoke')
    if not re.fullmatch(r'TASK_\d{8}_\d+', str(certificate.get('source_task_id', ''))):
        raise ValueError('Missing actual smoke task identity')
    for key in ('checkpoint_sha256', 'bundle_sha256', 'independent_source_model_sha256',
                'independent_source_features_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(certificate.get(key, ''))):
            raise ValueError('Missing independently inspected smoke artifact hash: '+key)
    if (certificate['independent_source_model_sha256'] !=
            certificate.get('continuation', {}).get('teacher_source_model_sha256') or
            certificate['independent_source_model_sha256'] != experiment.cfg['mu_temporal']['source_model_state_sha256']):
        raise ValueError('Independent smoke source does not match restored teacher')
    validate_mu_certificate(experiment, certificate.get('continuation') or {})
    validate_mu_loss_report(certificate.get('mu_loss_report') or {}, experiment.cfg['mu_temporal'], 10)
    if (certificate['independent_source_features_sha256'] != experiment.cfg['mu_temporal']['source_feature_state_sha256'] or
            certificate['mu_loss_report'].get('teacher_final_model_sha256') != certificate['independent_source_model_sha256']):
        raise ValueError('Independent feature/teacher final state differs from exact source')
    return True


def warm_start_mu(runner, experiment, checkpoint):
    from .mu_loss import DeterministicMuLoss
    source = validate_mu(experiment)
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Deterministic mu source SHA mismatch')
    state = torch.load(str(checkpoint), weights_only=True, map_location=runner.device)
    if state['amp_identity'] != source.identity() or state['completed_updates'] != 2500 or state['iter'] != 2499:
        raise ValueError('Wrong deterministic mu source identity/update')
    spec = experiment.cfg['mu_temporal']
    for key in ('optimizer_state_dict', 'es_optimizer_state_dict'):
        if any(group['lr'] != spec['learning_rate'] for group in state[key]['param_groups']):
            raise ValueError('Source learning rate changed')
    restored = restore_learning_state(runner, experiment, state, learning_rate=spec['learning_rate'])
    student = runner.alg.actor_critic
    source_hash = state_fingerprint(state['model_state_dict'])
    if state_fingerprint(student.state_dict()) != source_hash:
        raise ValueError('Student differs from full original source')
    regularizer = DeterministicMuLoss(student,
        [spec['acceleration_scales_rad_s2'][name] for name in experiment.spec.joint_names],
        spec['anchor_coef'], spec['temporal_coef'], spec['control_dt'], spec['action_scale'])
    teacher_hash = state_fingerprint(regularizer.teacher.state_dict())
    if teacher_hash != source_hash or any(p.requires_grad for p in regularizer.teacher.parameters()):
        raise ValueError('Teacher is not frozen full original source')
    runner.alg.ppo.configure_mu_temporal(regularizer, startup_steps=spec['startup_steps'])
    runner.mu_initial_teacher_hash = source_hash
    proof = dict(**mu_source(experiment.cfg), **restored, teacher_frozen=True,
        teacher_source_model_sha256=teacher_hash, student_initial_model_sha256=source_hash)
    if spec.get('frozen_modules'):
        proof.update(runner.alg.ppo.configure_feature_freeze())
    if runner.amp_trainer.discriminator.style_floor != 0. or runner.amp_trainer.bridge_gradient_penalty != 1.:
        raise ValueError('Deterministic mu intervention changed AMP mechanics')
    validate_mu_certificate(experiment, proof)
    return proof


def validate_mu_diagnostics(report, group, steps, num_envs):
    # Same reward-call/substep evidence as unchanged source jitter_control.
    validate_jitter_diagnostics(report, 'control', steps, num_envs)
    mu_contract(group)
    return True


def validate_mu_loss_report(report, spec, updates):
    if (report.get('updates') != updates or report.get('teacher_unchanged') is not True or
            report.get('metadata_observed') is not True or report.get('valid_triplets', 0) <= 0 or
            report.get('all_finite') is not True or report.get('anchor_coef') != spec['anchor_coef'] or
            report.get('temporal_coef') != spec['temporal_coef']):
        raise ValueError('Missing actual deterministic mu optimization/continuity evidence')
    for key in ('anchor_loss', 'temporal_loss', 'weighted_anchor_loss', 'weighted_temporal_loss', 'aux_grad_norm'):
        value = report.get(key, float('nan'))
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid deterministic mu loss evidence: '+key)
    if spec['temporal_coef'] and (report['temporal_loss'] <= 0 or report['weighted_temporal_loss'] <= 0 or report['aux_grad_norm'] <= 0):
        raise ValueError('Enabled temporal loss not used in actual gradient step')
    if not spec['temporal_coef'] and report['weighted_temporal_loss'] != 0:
        raise ValueError('Anchor-only control acquired temporal optimization')
    if spec.get('frozen_modules'):
        from .feature_freeze import validate_gradient_report
        validate_feature_proof(report, spec['source_feature_state_sha256'])
        validate_gradient_report(report)
        if (report.get('gradient_minibatches') != 8*updates or
                report.get('main_grad_norm', 0.) <= 0 or
                report.get('combined_grad_norm_preclip', 0.) <= 0):
            raise ValueError('Missing actual all-minibatch gradient/clip evidence')
    return True
