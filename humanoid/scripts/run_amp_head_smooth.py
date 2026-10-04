"""Registered Gradmotion-only final-head experiment; NOT a PPO continuation.

The old original110 source/AMP/CNN/ES/critic/optimizers remain immutable.  New
independent native source cohorts are exported before any fit.  An offline pass
only admits a later physical test, not adoption, deployment, or DR unlocking.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit


CONFIG = 'configs/amp/head_smooth_v1.json'
SOURCE_CONFIG = 'configs/amp/lafan_walk02_sustain_control.json'
SMOKE_SCHEMA = 'head_smooth_native_smoke_audit_v1'
SOURCE_SHA = '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31'
SOURCE_MODEL_SHA = 'c06fbd97e32a483d46a2738014215de2721ad7db4b2b86547645ab72d4c555bc'


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _marker(name, value):
    print('[head-smooth-'+name+'] '+json.dumps(value, allow_nan=False), flush=True)


def policy_recording_command(*, mode, source, head, head_sha, commit, output, task):
    """New recorder probe or admitted physical test, not a resumed PPO run."""
    if mode not in ('smoke', 'formal'):
        raise ValueError('Unknown native head recording mode')
    count, duration = (2, 1) if mode == 'smoke' else (16, 60)
    command = [sys.executable, '-m', 'humanoid.scripts.record_amp_head_policy',
        '--mode', mode, '--source-checkpoint', str(source),
        '--expected-commit', commit, '--output', str(output), '--duration', str(duration),
        '--task', task, '--headless', '--num_envs', str(count), '--seed', '5',
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu',
        '--armature_mode', 'nominal']
    if mode == 'formal':
        if head is None or not re.fullmatch('[0-9a-f]{64}', head_sha or ''):
            raise ValueError('Admitted physical head must be bound to its actual file SHA')
        command += ['--head-artifact', str(head), '--head-sha256', head_sha]
    elif head is not None or head_sha is not None:
        raise ValueError('Recorder smoke is source-only, not candidate effectiveness')
    return command


def sealed_recording_command(*, source, head, head_sha, commit, output, task,
                             sealed, train, validation):
    """One predeclared paired holdout, in separate source/candidate processes."""
    if head is None or not re.fullmatch('[0-9a-f]{64}', head_sha or ''):
        raise ValueError('Sealed physical test requires the actual admitted head file SHA')
    return [sys.executable, '-m', 'humanoid.scripts.record_amp_head_sealed',
        '--mode', 'formal', '--source-checkpoint', str(source),
        '--head-artifact', str(head), '--head-sha256', head_sha,
        '--expected-commit', commit, '--output', str(output), '--duration', '60',
        '--sealed-cohort', str(sealed), '--train-cohort', str(train),
        '--validation-cohort', str(validation), '--task', task, '--headless',
        '--num_envs', '8', '--seed', '705', '--sim_device', 'cuda:0',
        '--rl_device', 'cuda:0', '--pipeline', 'gpu', '--armature_mode', 'nominal']


def _head_history_git(repo, *arguments):
    """Capture Git transport diagnostics: they may contain credentials."""
    environment = os.environ.copy()
    environment['GIT_TERMINAL_PROMPT'] = '0'
    environment['GIT_ALLOW_PROTOCOL'] = 'https'
    try:
        result = subprocess.run(['git']+list(arguments), cwd=str(repo), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError('Head history operation unavailable or timed out') from None
    if (type(result.returncode) is not int or type(result.stdout) is not bytes
            or type(result.stderr) is not bytes):
        raise ValueError('Invalid head history Git readback')
    return result


def ensure_smoke_ancestry(repo, commit):
    """Repair only a missing commit in a verified shallow checkout, never bypass it."""
    if not isinstance(commit, str) or re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('Invalid head history required commit')
    repo = Path(repo).resolve()
    ancestry = ('merge-base', '--is-ancestor', commit, 'HEAD')
    initial = _head_history_git(repo, *ancestry)
    if initial.returncode not in (0, 128):
        raise ValueError('Head smoke ancestry is not established')
    head_record = _head_history_git(repo, 'rev-parse', 'HEAD')
    head_bytes = head_record.stdout.strip()
    if head_record.returncode != 0 or re.fullmatch(b'[0-9a-f]{40}', head_bytes) is None:
        raise ValueError('Invalid head history checkout HEAD')
    head = head_bytes.decode('ascii')
    if initial.returncode == 0:
        return dict(ancestor_verified=True, history_refreshed=False,
                    required_commit=commit, head_commit=head)
    missing = _head_history_git(repo, 'rev-parse', '--verify', '--quiet', commit+'^{commit}')
    if missing.returncode != 1 or missing.stdout.strip():
        raise ValueError('Required head smoke commit is not confirmed missing')
    shallow = _head_history_git(repo, 'rev-parse', '--is-shallow-repository')
    if shallow.returncode != 0 or shallow.stdout.strip() != b'true':
        raise ValueError('Missing head smoke commit outside a verified shallow checkout')

    def unchanged_checkout():
        root = _head_history_git(repo, 'rev-parse', '--show-toplevel')
        current = _head_history_git(repo, 'rev-parse', 'HEAD')
        tracked = _head_history_git(repo, 'status', '--porcelain', '--untracked-files=no')
        untracked = _head_history_git(repo, 'ls-files', '--others', '--exclude-standard',
                                      '--', 'humanoid', 'configs', 'resources')
        try:
            actual_root = Path(root.stdout.decode('utf-8').strip()).resolve()
        except (UnicodeError, OSError, ValueError):
            raise ValueError('Invalid head history checkout root') from None
        if (any(row.returncode != 0 for row in (root, current, tracked, untracked))
                or actual_root != repo or current.stdout.strip() != head_bytes
                or tracked.stdout.strip() or untracked.stdout.strip()):
            raise ValueError('Head history checkout changed or is not clean')

    unchanged_checkout()
    remote = _head_history_git(repo, 'remote', 'get-url', 'origin')
    try:
        remote_text = remote.stdout.decode('utf-8').strip()
        url = urlsplit(remote_text)
        trusted = (remote.returncode == 0 and remote_text
            and not any(ord(c) <= 32 or ord(c) == 127 for c in remote_text)
            and '?' not in remote_text and '#' not in remote_text and url.scheme == 'https'
            and url.hostname == 'github.com' and url.port in (None, 443)
            and url.path in ('/Yinzhen01/f1_train_01', '/Yinzhen01/f1_train_01.git')
            and not url.query and not url.fragment)
    except (UnicodeError, ValueError):
        trusted = False
    if not trusted:
        raise ValueError('Head history origin is not the registered HTTPS repository')
    # Cloud Git 2.25.1 rejects --no-write-fetch-head (actual probe091).
    # Allow only normal FETCH_HEAD metadata; keep the one-shot/checkout guards.
    fetched = _head_history_git(repo, 'fetch', '--no-tags',
                               '--deepen=64', 'origin', head)
    if fetched.returncode != 0:
        raise ValueError('Head history refresh failed; no retry or diagnostic echo')
    unchanged_checkout()
    if _head_history_git(repo, *ancestry).returncode != 0:
        raise ValueError('Head smoke ancestry remains unproved after one history refresh')
    return dict(ancestor_verified=True, history_refreshed=True,
                required_commit=commit, head_commit=head)


def validate_smoke_certificate(certificate, *, identity, fingerprint, repo):
    """A new head-specific independently audited real native probe is required."""
    expected = dict(schema=SMOKE_SCHEMA, mode='smoke', identity=identity,
        implementation_fingerprint=fingerprint, native_verified=True,
        cohort_arrays_verified=True, source_forward_verified=True,
        head_archive_verified=True, native_recorder_verified=True,
        train_seed=305, validation_seed=505,
        num_envs=4, duration_s=2, solver_runs=1, formal_admission=False,
        source_checkpoint_sha256=SOURCE_SHA, source_model_state_sha256=SOURCE_MODEL_SHA,
        platform_terminal_status='5', effectiveness_verified=False, dr_unlocked=False)
    for key, value in expected.items():
        actual = certificate.get(key)
        if actual != value or (type(value) in (bool, int) and type(actual) is not type(value)):
            raise ValueError('New head smoke certificate mismatch: '+key)
    if not re.fullmatch('TASK_[0-9]{8}_[0-9]{3}', certificate.get('platform_task_id', '')):
        raise ValueError('Missing actual independently verified Gradmotion smoke task')
    commit = certificate.get('code_commit', '')
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Missing actual new head native code commit')
    for key in ('cloud_log_sha256', 'train_sha256', 'validation_sha256',
                'head_sha256', 'policy_rollout_sha256', 'audit_report_sha256'):
        if (not re.fullmatch('[0-9a-f]{64}', certificate.get(key, ''))
                or certificate[key] == '0'*64):
            raise ValueError('Missing independently checked new cloud evidence: '+key)
    if certificate['train_sha256'] == certificate['validation_sha256']:
        raise ValueError('New training/validation cloud artifacts cannot be identical')
    ancestry = ['git', 'merge-base', '--is-ancestor', commit, 'HEAD']
    try:
        subprocess.check_call(ancestry, cwd=str(repo))
    except subprocess.CalledProcessError as error:
        if error.returncode != 128:
            raise
        proof = ensure_smoke_ancestry(repo, commit)
        subprocess.check_call(ancestry, cwd=str(repo))
        _marker('history', proof)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--headless', action='store_true', required=True)
    parser.add_argument('--checkpoint-file', type=Path)
    parser.add_argument('--source-endpoint', type=Path)
    parser.add_argument('--smoke-certificate', type=Path)
    extra = parser.parse_args()
    # gm-run may launch a file from its scripts directory. Prefer this exact
    # checkout over an unrelated editable package already installed in the image.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # Isaac Gym MUST precede all direct and transitive torch imports.
    from isaacgym import gymapi  # noqa: F401
    import numpy as np
    import torch
    from humanoid import LEGGED_GYM_ROOT_DIR
    # The environment package initializes task_registry before utils reimports
    # it. Importing utils first creates a real partially initialized cycle.
    import humanoid.envs  # noqa: F401
    from humanoid.algo.ppo.actor_critic_dh import ActorCriticDH
    from humanoid.amp.head_artifact import (
        ORIGINAL110_PARENT, assemble_head_artifact, validate_head_artifact)
    from humanoid.amp.head_smoothing import (
        fit_head_smoothing, evaluate_head_candidate, validate_head_only_change)
    from humanoid.amp.head_workflow import (
        identity_digest, audit_cohort, forward_parity, fit_episodes, exclusion_add_cohort)
    from humanoid.amp.refinement import select_environment, locate_source
    from humanoid.amp.native_hardware import read_hardware, validate_hardware_record
    from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint
    from humanoid.scripts.collect_amp_head_cohort import (
        file_sha, _read_bundle, source_exclusions, registered_source_exclusions,
        BUDGETS, SEEDS)
    from humanoid.utils.helpers import class_to_dict, set_seed

    # ESKU000001 is independently checked in the actual Gradmotion task API.
    # That SKU's native readback (TASK_20261004_080) omits D in the device name.
    # Accept only the two exact 4090-family names plus single-card CUDA/24GiB/
    # Ada 8.9 readback; do not claim this identifies the silicon as 4090D.
    hardware = read_hardware(torch, sys.platform)
    _marker('hardware', hardware)
    validate_hardware_record(hardware)
    repo = Path(LEGGED_GYM_ROOT_DIR)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    if (not re.fullmatch('[0-9a-f]{40}', extra.expected_commit) or commit != extra.expected_commit
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=str(repo)).strip()
            or subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--',
                                       'humanoid', 'configs', 'resources'], cwd=str(repo)).strip()):
        raise ValueError('Exact published clean checkout is required')
    cfg = json.loads((repo/CONFIG).read_text(encoding='utf-8'))
    if (cfg['schema'] != 'actor_head_offline_v1' or cfg['source_config'] != SOURCE_CONFIG
            or cfg['source_checkpoint_sha256'] != ORIGINAL110_PARENT['file_sha256']
            or cfg['source_model_state_sha256'] != ORIGINAL110_PARENT['model_state_sha256']
            or cfg['source_task'] != ORIGINAL110_PARENT['source_task']
            or cfg['head_keys'] != ['actor.6.weight', 'actor.6.bias']
            or cfg['physical_acceptance_required'] is not True
            or cfg['solver'] != dict(temporal_weight=1., ridge=1e-6, fir_taps=21)
            or cfg['seeds'] != SEEDS or cfg['effectiveness_verified'] is not False
            or cfg['dr_unlocked'] is not False):
        raise ValueError('Unregistered final-head experiment definition')
    source = extra.checkpoint_file or locate_source(
        (repo, Path('/workspace'), Path('/personal')), ORIGINAL110_PARENT['file_sha256'])
    if file_sha(source) != ORIGINAL110_PARENT['file_sha256']:
        raise ValueError('Wrong original110 parent checkpoint bytes')
    experiment = ScaledExperiment(repo, repo/SOURCE_CONFIG)
    identity, fingerprint = experiment.identity(), implementation_fingerprint(repo)
    if extra.mode == 'formal':
        if extra.smoke_certificate is None:
            raise ValueError('A new independently audited head native smoke is required')
        validate_smoke_certificate(json.loads(extra.smoke_certificate.read_text(encoding='utf-8')),
            identity=identity, fingerprint=fingerprint, repo=repo)
    elif extra.smoke_certificate is not None:
        raise ValueError('Smoke cannot reuse a prior certificate')
    if extra.source_endpoint:
        exclusions, source_proof, source_manifest = source_exclusions(extra.source_endpoint, identity, torch)
        exclusion_args = ['--source-endpoint', str(extra.source_endpoint)]
    else:
        index_path = repo/cfg['source_exclusions']['path']
        exclusions, source_proof, source_manifest = registered_source_exclusions(repo, index_path, identity)
        exclusion_args = ['--source-exclusion-index', str(index_path)]
    num_envs, duration = BUDGETS[extra.mode]
    if cfg[extra.mode] != dict(num_envs=num_envs, duration_s=duration):
        raise ValueError('Head source cohort budget changed')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    # Match the existing GM policy-PT discovery route; this does not establish
    # actual upload success, which must be checked in task logs/model rows.
    log_dir = repo/'logs'/cfg['experiment']/'exported_data'/(extra.mode+'_'+stamp)
    log_dir.mkdir(parents=True, exist_ok=False)
    base_number = 7000000 if extra.mode == 'smoke' else 7100000
    paths = {name: log_dir/('model_%d.pt' % (base_number+offset))
             for name, offset in (('train', 1), ('validation', 2), ('head', 3),
                                  ('report', 4), ('policy_probe', 5),
                                  ('sealed', 6), ('sealed_policy', 7))}
    start = dict(schema='head_smooth_native_stage_v1', mode=extra.mode,
        code_commit=commit, implementation_fingerprint=fingerprint,
        native_hardware=hardware,
        identity=identity, source_checkpoint_sha256=file_sha(source),
        source_exclusions=source_proof, num_envs=num_envs, duration_s=duration,
        seeds=dict(train=305, validation=505), ppo_updates_added=0,
        optimizer_state_reused=False, effectiveness_verified=False, dr_unlocked=False,
        platform_evidence='4090-family runtime readback is not SKU/silicon authentication; '
            'registered Gradmotion task/run/log/artifact audit is required separately')
    _marker('start', start)
    manifests, arrays, audits, parities = {}, {}, {}, {}
    source_state = torch.load(str(source), map_location='cpu', weights_only=True)
    env_cfg, train_cfg, _ = select_environment('sustain_control')
    # Match the source collector's existing deterministic cuDNN setup, not a
    # new numerical precision setting. Loaded source parameters remain frozen.
    set_seed(SEEDS['train'])
    policy = ActorCriticDH(env_cfg.env.short_frame_stack*env_cfg.env.num_single_obs,
        env_cfg.env.num_single_obs, env_cfg.env.num_privileged_obs, env_cfg.env.num_actions,
        **class_to_dict(train_cfg.policy)).to('cuda:0')
    policy.load_state_dict(source_state['model_state_dict'], strict=True)
    policy.eval()
    for split in ('train', 'validation'):
        command = [sys.executable, '-m', 'humanoid.scripts.collect_amp_head_cohort',
            '--split', split, '--mode', extra.mode, '--checkpoint-file', str(source),
            '--expected-commit', commit, '--output', str(paths[split]),
            '--duration', str(duration), '--task', experiment.cfg['experiment'],
            '--headless', '--num_envs', str(num_envs), '--seed', str(SEEDS[split]),
            '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu',
            '--armature_mode', 'nominal']+exclusion_args
        if split == 'validation':
            command += ['--exclude-cohort', str(paths['train'])]
        subprocess.check_call(command, cwd=str(repo))
        manifest, data = _read_bundle(paths[split], torch)
        manifests[split], arrays[split] = manifest, data
        audits[split] = audit_cohort(manifest, data, split=split, mode=extra.mode,
            identity=identity, code_commit=commit, implementation_fingerprint=fingerprint,
            exclusions=exclusions, source_manifest=source_manifest)
        parities[split] = forward_parity(policy, data,
            action_clip=env_cfg.normalization.clip_actions, batch_size=num_envs,
            expected_numerical_context=manifest['inference_capture']['numerical_context'])
        exclusion_add_cohort(exclusions, audits[split])
    report = dict(start=start, cohort_sha256={key: file_sha(paths[key]) for key in manifests},
        source_parity=parities, effectiveness_verified=False, dr_unlocked=False,
        solver_runs=0, admitted_to_physical_test=False)
    ready_key = 'fit_eligible' if extra.mode == 'formal' else 'smoke_eligible'
    if not all(audits[key][ready_key] for key in audits):
        report.update(status='cohort_rejected', rejection_reasons={key: dict(
            episodes=audits[key]['episodes'], deduplication=audits[key]['deduplication'])
            for key in audits if not audits[key][ready_key]})
    else:
        train = fit_episodes(manifests['train'], arrays['train'], audits['train'], identity=identity)
        validation = fit_episodes(manifests['validation'], arrays['validation'],
                                  audits['validation'], identity=identity)
        forbidden = [label for values in exclusions['initial_state'].values() for label in values
                     if label.startswith('source110/') or '/sealed/' in label]
        old_head = source_state['model_state_dict']
        fitted = fit_head_smoothing(train, validation,
            old_head['actor.6.weight'].numpy(), old_head['actor.6.bias'].numpy(),
            source_identity=identity_digest(identity),
            action_clip=env_cfg.normalization.clip_actions, forbidden_cohort_ids=forbidden)
        # Actual f32 storage round trip is performed BEFORE the admission audit.
        buffer = io.BytesIO()
        torch.save({'actor.6.weight': torch.from_numpy(fitted.candidate_weight).float(),
                    'actor.6.bias': torch.from_numpy(fitted.candidate_bias).float()}, buffer)
        buffer.seek(0)
        head = torch.load(buffer, map_location='cpu', weights_only=True)
        exported = evaluate_head_candidate(train, validation,
            old_head['actor.6.weight'].numpy(), old_head['actor.6.bias'].numpy(),
            head['actor.6.weight'].numpy(), head['actor.6.bias'].numpy(),
            source_identity=identity_digest(identity), candidate_identity=identity_digest(identity),
            action_clip=env_cfg.normalization.clip_actions, forbidden_cohort_ids=forbidden)
        admitted = bool(fitted.report['admitted'] and exported['admitted'])
        report.update(solver_runs=1, float64_fit=fitted.report, exported_head=exported,
                      status='offline_pass' if admitted else 'offline_rejected',
                      admitted_to_physical_test=bool(admitted and extra.mode == 'formal'))
        math_bytes = json.dumps(report, sort_keys=True, allow_nan=False).encode('utf-8')
        provenance = dict(code_commit=commit, implementation_fingerprint=fingerprint,
            mode=extra.mode, solver=dict(kind='quadratic_direction_scaled_v1',
                temporal_weight=1., ridge=1e-6, filter_window=21, action_scale=.5,
                dtype='float64', solve_count=1, direction_scale=fitted.report['direction_scale']),
            budget=dict(num_envs=num_envs, duration_s=duration, fit_episode_count=num_envs),
            cohorts={key: dict(sha256=file_sha(paths[key]), seed=SEEDS[key],
                num_envs=num_envs, duration_s=duration, episode_ids=manifests[key]['episode_ids'])
                for key in manifests}, source_parity=dict(full_obs_verified=True,
                hidden_verified=True, action_verified=True,
                max_abs_error=max(value['max_abs_error'] for value in parities.values())),
            report_sha256=hashlib.sha256(math_bytes).hexdigest(), offline_admitted=admitted,
            effectiveness_verified=False, dr_unlocked=False)
        artifact = assemble_head_artifact(source, head, provenance=provenance)
        with paths['head'].open('xb') as stream:
            torch.save(artifact, stream)
        artifact = torch.load(str(paths['head']), map_location='cpu', weights_only=True)
        archive_audit = validate_head_artifact(artifact, source)
        freeze_audit = validate_head_only_change(old_head, artifact['model_state_dict'],
                                                identity_digest(identity), identity_digest(identity))
        report.update(head_sha256=file_sha(paths['head']), archive_audit=archive_audit,
            frozen_audit=freeze_audit, provenance_report_json=math_bytes.decode('utf-8'))
    # A genuine NEW source-only recorder probe precedes formal admission. An
    # admitted formal head is then evaluated on every original32 initial state;
    # failures are retained and are NOT treated as task-execution failures or
    # removed from the later physical quality comparison.
    if extra.mode == 'smoke' or report['admitted_to_physical_test']:
        from humanoid.scripts.record_amp_head_policy import recorder_contract, audit_rollout_arrays
        candidate = paths['head'] if extra.mode == 'formal' else None
        candidate_sha = report.get('head_sha256') if candidate is not None else None
        subprocess.check_call(policy_recording_command(mode=extra.mode, source=source,
            head=candidate, head_sha=candidate_sha, commit=commit,
            output=paths['policy_probe'], task=experiment.cfg['experiment']), cwd=str(repo))
        probe, probe_arrays = _read_bundle(paths['policy_probe'], torch)
        probe_contract = recorder_contract(extra.mode, 5,
            2 if extra.mode == 'smoke' else 16, 1 if extra.mode == 'smoke' else 60)
        probe_episodes = audit_rollout_arrays(probe_arrays, probe_contract)
        if (any(probe.get(key) != value for key, value in probe_contract.items())
                or probe.get('code_commit') != commit
                or probe.get('implementation_fingerprint') != fingerprint
                or probe.get('parent_checkpoint_sha256') != SOURCE_SHA
                or probe.get('head_artifact_sha256') != candidate_sha
                or probe.get('episodes') != probe_episodes):
            raise ValueError('Actual new head native recording differs from the bound request')
        report['policy_probe'] = dict(artifact_name=paths['policy_probe'].name,
            sha256=file_sha(paths['policy_probe']), type=probe['type'], mode=extra.mode,
            num_envs=probe['num_envs'], duration_s=probe['duration_s'],
            head_artifact_sha256=candidate_sha,
            source_checkpoint_sha256=probe['parent_checkpoint_sha256'],
            policy_model_state_sha256=probe['policy_model_state_sha256'],
            evaluation_protocol=probe['evaluation_protocol'], episodes=probe_episodes)
    if extra.mode == 'formal' and report['admitted_to_physical_test']:
        # Sealed observations are first collected AFTER the sole solve/export
        # decision; they never enter fitting, scaling, tuning or candidate repair.
        subprocess.check_call([sys.executable, '-m', 'humanoid.scripts.collect_amp_head_cohort',
            '--split', 'sealed', '--mode', 'formal', '--checkpoint-file', str(source),
            '--expected-commit', commit, '--output', str(paths['sealed']),
            '--duration', '20', '--task', experiment.cfg['experiment'], '--headless',
            '--num_envs', '8', '--seed', '705', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
            '--pipeline', 'gpu', '--armature_mode', 'nominal',
            '--exclude-cohort', str(paths['train']), '--exclude-cohort', str(paths['validation'])]
            +exclusion_args, cwd=str(repo))
        sealed_manifest, sealed_arrays = _read_bundle(paths['sealed'], torch)
        sealed_audit = audit_cohort(sealed_manifest, sealed_arrays, split='sealed', mode='formal',
            identity=identity, code_commit=commit, implementation_fingerprint=fingerprint,
            exclusions=exclusions, source_manifest=source_manifest)
        if not sealed_audit['deduplication']['passed']:
            raise ValueError('Sealed native initial states/history overlap protected cohorts')
        report['sealed_cohort'] = dict(artifact_name=paths['sealed'].name,
            sha256=file_sha(paths['sealed']), seed=705, num_envs=8, duration_s=20,
            split='sealed', fit_eligible=False, episodes=sealed_audit['episodes'],
            deduplication_passed=True, holdout_not_used_for_fitting=True)
        subprocess.check_call(sealed_recording_command(source=source, head=paths['head'],
            head_sha=report['head_sha256'], commit=commit, output=paths['sealed_policy'],
            task=experiment.cfg['experiment'], sealed=paths['sealed'], train=paths['train'],
            validation=paths['validation']), cwd=str(repo))
        sealed_policy, _ = _read_bundle(paths['sealed_policy'], torch)
        # Full paired state/initial/runtime proof is independently audited from
        # the final archive. Here bind the actual child output to this request.
        if (sealed_policy.get('type') != 'head_policy_sealed_rollout_v1'
                or sealed_policy.get('code_commit') != commit
                or sealed_policy.get('implementation_fingerprint') != fingerprint
                or sealed_policy.get('head_artifact_sha256') != report['head_sha256']
                or sealed_policy.get('parent_checkpoint_sha256') != SOURCE_SHA):
            raise ValueError('Actual sealed physical pair differs from the bound native request')
        report['sealed_policy'] = dict(artifact_name=paths['sealed_policy'].name,
            sha256=file_sha(paths['sealed_policy']), type=sealed_policy['type'],
            mode='formal', seed=705, num_envs=8, duration_s=60,
            head_artifact_sha256=report['head_sha256'], source_checkpoint_sha256=SOURCE_SHA,
            sealed_cohort_sha256=report['sealed_cohort']['sha256'],
            effectiveness_verified=False, dr_unlocked=False)
    _write_json(log_dir/'head_stage_report.json', report)
    with paths['report'].open('xb') as stream:
        torch.save(dict(report_json=json.dumps(report, allow_nan=False)), stream)
    _marker('complete', report)
    # Same bounded discovery hold as the existing native diagnostic entry.
    # This is not an upload receipt: platform logs/models still must prove all
    # five/seven exact artifact bytes before any independent admission.
    time.sleep(60)


if __name__ == '__main__':
    main()
