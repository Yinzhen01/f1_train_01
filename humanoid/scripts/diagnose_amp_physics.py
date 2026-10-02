"""Hash-bound frozen-policy PhysX diagnosis; each group uses a fresh process."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_IDS = {'original': 8802500, 'velocity1': 8802501, 'selfoff': 8802502}
REQUIRED_TRACKED_PATHS = (
    'humanoid/scripts/diagnose_amp_physics.py', 'humanoid/scripts/record_amp_policy.py',
    'humanoid/amp/physics_diagnostic.py', 'humanoid/envs/x1/x1_amp_physics_diagnostic_env.py')


def parse_groups(value, registered):
    groups = tuple(value.split(','))
    if (not groups or any(not name or name.strip() != name for name in groups) or
            len(set(groups)) != len(groups) or any(name not in registered for name in groups)):
        raise ValueError('Diagnostic groups must be registered, nonempty and unique')
    return groups


def diagnostic_budget(mode):
    if mode == 'smoke':
        return 2, 1., False
    if mode == 'full':
        return 16, 60., True
    raise ValueError('Unknown physics diagnostic mode')


def build_command(repo, checkpoint, digest, commit, output, group, mode, python=None,
                  smoke_certificate=None):
    if group not in ARTIFACT_IDS:
        raise ValueError('Unregistered physics diagnostic group')
    num_envs, duration, extended = diagnostic_budget(mode)
    if (extended and smoke_certificate is None) or (not extended and smoke_certificate is not None):
        raise ValueError('Full recorder requires its smoke certificate; smoke cannot consume one')
    output = Path(output)
    if output.name != 'model_%d.pt' % ARTIFACT_IDS[group]:
        raise ValueError('Diagnostic artifact number does not match its group')
    command = [python or sys.executable, str(Path(repo)/'humanoid/scripts/record_amp_policy.py'),
        '--physics-diagnostic='+group, '--experiment=sustain_control',
        '--checkpoint-file', str(checkpoint), '--checkpoint-sha256', digest,
        '--expected-commit', commit, '--output', str(output),
        '--task=f1_amp_walk02_sustain_control', '--seed=5',
        '--num_envs=%d' % num_envs, '--duration=%g' % duration, '--headless',
        '--sim_device=cuda:0', '--rl_device=cuda:0']
    if extended:
        command.extend(('--extended-validation', '--smoke-certificate='+str(smoke_certificate)))
    return command


def validate_checkout(repo, expected_commit):
    repo = Path(repo).resolve()
    root = Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], cwd=str(repo)).decode().strip()).resolve()
    actual = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    if root != repo or actual != expected_commit:
        raise ValueError('Physics diagnostic checkout/commit mismatch')
    status = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=str(repo)).decode().strip()
    if status:
        raise ValueError('Physics diagnosis requires a clean committed checkout')
    untracked_source = subprocess.check_output(
        ['git', 'ls-files', '--others', '--exclude-standard', '--', 'humanoid', 'configs', 'resources'],
        cwd=str(repo)).decode().strip()
    if untracked_source:
        raise ValueError('Untracked diagnostic source or robot/data configuration is forbidden')
    tracked = subprocess.check_output(
        ['git', 'ls-files', '--error-unmatch', '--']+list(REQUIRED_TRACKED_PATHS),
        cwd=str(repo)).decode().splitlines()
    if set(tracked) != set(REQUIRED_TRACKED_PATHS):
        raise ValueError('Physics diagnostic implementation is not fully tracked')
    return actual


def validate_source_state(state, expected_identity):
    if (state.get('amp_identity') != expected_identity or
            state.get('completed_updates') != 2500 or state.get('iter') != 2499 or
            not isinstance(state.get('model_state_dict'), dict) or not state['model_state_dict']):
        raise ValueError('Physics diagnosis requires the exact completed control2500 source')
    return True


def prepare_output_folder(repo, mode, timestamp=None):
    diagnostic_budget(mode)
    stamp = timestamp or datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    if Path(stamp).name != stamp or stamp in ('', '.', '..'):
        raise ValueError('Invalid diagnostic output timestamp')
    folder = Path(repo)/'logs/f1_amp_physics_diagnostic/exported_data'/stamp
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def validate_group_artifact(manifest, arrays, contract, identity, commit, digest, mode):
    """Validate actual recorder output, retaining every failed episode prefix."""
    import numpy as np
    from tools.amp.audit_physics_diagnostic import validate_initialization
    num_envs, duration, extended = diagnostic_budget(mode)
    if (manifest.get('physics_diagnostic') != contract or manifest.get('identity') != identity or
            manifest.get('checkpoint_sha256') != digest or manifest.get('code_commit') != commit or
            manifest.get('duration_s') != duration or manifest.get('num_envs') != num_envs or
            manifest.get('fps') != 100 or manifest.get('modes') != ['standing', 'reference'] or
            manifest.get('policy_deterministic') is not True or manifest.get('diagnostic_only') is not True or
            manifest.get('dr_unlocked') is not False or manifest.get('effectiveness_verified') is not False):
        raise ValueError('Recorded diagnostic identity/budget/contract mismatch')
    gate = manifest.get('no_dr', {})
    if (gate.get('configuration_verified') is not True or
            gate.get('domain_randomization') is not False or gate.get('observation_noise') is not False):
        raise ValueError('Recorded diagnostic lacks no-DR/no-noise proof')
    protocol = 'fixed60_independent_mode_seeds' if extended else 'physics_diagnostic_smoke'
    if (manifest.get('evaluation_protocol') != protocol or
            manifest.get('mode_seeds') != {'standing': 5, 'reference': 105}):
        raise ValueError('Diagnosis requires its fixed protocol and independent mode seeds')
    validate_initialization(manifest)
    reset_shapes = dict(root_states=(13,), commands=(4,), gait_start=(),
        episode_length_buf=(), phase_length_buf=(), rsi_indices=())
    reset_shapes.update({name: (12,) for name in
        ('dof_pos', 'dof_vel', 'actions', 'last_actions', 'last_last_actions')})
    for snapshot in manifest['physics_initialization']['reset_inputs'].values():
        for name, shape in reset_shapes.items():
            if np.asarray(snapshot[name]).shape != (num_envs,)+shape:
                raise ValueError('Invalid reset input shape: '+name)
    if not arrays or any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Diagnostic trajectory arrays are missing or nonfinite')
    raw_shapes = {'physics_raw_velocity': (10, 12), 'physics_raw_torque': (10, 12),
        'physics_raw_body_force': (10, 3, 3), 'physics_raw_body_state': (10, 3, 13),
        'physics_interval_initial_velocity': (12,), 'physics_interval_initial_torque': (12,)}
    for name in ('standing', 'reference'):
        prefix = name+'_'
        time_values = arrays.get(prefix+'time')
        if time_values is None or time_values.ndim != 1 or not 1 <= len(time_values) <= round(duration*100):
            raise ValueError('Invalid diagnostic control-frame budget')
        frames = len(time_values)
        if not np.allclose(time_values, (np.arange(frames)+1)*.01, atol=1e-7, rtol=0):
            raise ValueError('Diagnostic control time is not contiguous 100 Hz')
        valid, failure = arrays.get(prefix+'valid'), arrays.get(prefix+'failure')
        if (valid is None or failure is None or valid.shape != (frames, num_envs) or
                failure.shape != (frames, num_envs) or valid.dtype != np.bool_ or failure.dtype != np.bool_ or
                np.any(np.diff(valid.astype(int), axis=0) > 0)):
            raise ValueError('Invalid diagnostic episode-validity masks')
        if frames < round(duration*100) and np.any(valid[-1] & ~failure[-1]):
            raise ValueError('Diagnostic recorder stopped with surviving environments')
        for key, shape in raw_shapes.items():
            value = arrays.get(prefix+key)
            if value is None or value.shape != (frames,)+shape:
                raise ValueError('Missing or invalid env0 raw substep array: '+key)
        for key, shape in (('dof_pos', (12,)), ('dof_vel', (12,)), ('action', (12,)),
                           ('torque', (12,)), ('root_state', (13,))):
            if (prefix+key not in arrays or arrays[prefix+key].shape != (frames, num_envs)+shape or
                    prefix+'initial_'+key not in arrays or
                    arrays[prefix+'initial_'+key].shape != (num_envs,)+shape):
                raise ValueError('Missing or invalid full-state diagnostic array: '+key)
    return True


def parse_request(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--groups', default='original,velocity1,selfoff')
    parser.add_argument('--task', choices=('f1_amp_physics_diagnostic',),
                        default='f1_amp_physics_diagnostic')
    parser.add_argument('--smoke-certificate', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'full' and args.smoke_certificate is None:
        raise ValueError('Full physics diagnosis requires a real-cloud smoke certificate')
    if args.mode == 'smoke' and args.smoke_certificate is not None:
        raise ValueError('Physics diagnostic smoke cannot consume a smoke certificate')
    return args


def main():
    args = parse_request()
    commit = validate_checkout(ROOT, args.expected_commit)
    # Isaac Gym must initialize before torch, including transitive AMP imports.
    from isaacgym import gymapi, gymtorch  # noqa: F401
    import torch
    sys.path.insert(0, str(ROOT))
    from humanoid import LEGGED_GYM_ROOT_DIR
    from humanoid.amp.physics_diagnostic import GROUPS, SOURCE_SHA, SOURCE_CONFIG, diagnostic_contract
    from humanoid.amp.refinement import locate_source
    from humanoid.amp.scaled_experiment import ScaledExperiment
    from tools.amp.inspect_rollout import load_bundle
    from tools.amp.audit_physics_diagnostic import compare_initialization
    if Path(LEGGED_GYM_ROOT_DIR).resolve() != ROOT or set(GROUPS) != set(ARTIFACT_IDS):
        raise ValueError('Diagnostic repository/group registration mismatch')
    groups = parse_groups(args.groups, GROUPS)
    if args.mode == 'full':
        from humanoid.amp.physics_diagnostic import validate_smoke_certificate
        certificate = json.loads(args.smoke_certificate.read_text(encoding='utf-8'))
        validate_smoke_certificate(certificate, commit, groups, SOURCE_SHA)
    source = locate_source((ROOT, Path('/workspace'), Path('/personal')), SOURCE_SHA)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Mounted control2500 checkpoint hash mismatch')
    experiment = ScaledExperiment(ROOT, ROOT/SOURCE_CONFIG)
    state = torch.load(str(source), weights_only=True, map_location='cpu')
    validate_source_state(state, experiment.identity())
    if any(not bool(torch.isfinite(value).all()) for value in state['model_state_dict'].values()):
        raise ValueError('Nonfinite source policy parameters')
    del state
    folder = prepare_output_folder(ROOT, args.mode)
    manifest = dict(source_task='TASK_20260926_110', source_config=SOURCE_CONFIG,
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=2500,
        identity=experiment.identity(), code_commit=commit, mode=args.mode, groups=list(groups),
        sdk_task=args.task,
        no_training=True, policy_updates=0, discriminator_updates=0, estimator_updates=0,
        seed=5, num_envs=diagnostic_budget(args.mode)[0], duration_s=diagnostic_budget(args.mode)[1],
        group_artifacts={}, effectiveness_verified=False, dr_unlocked=False)
    print('[amp-physics-diagnostic-start] '+json.dumps(manifest), flush=True)
    for group in groups:
        output = folder/('model_%d.pt' % ARTIFACT_IDS[group])
        if output.exists() or output.with_suffix('.json').exists():
            raise FileExistsError(output)
        subprocess.run(build_command(ROOT, source, SOURCE_SHA, commit, output, group, args.mode,
                                     smoke_certificate=args.smoke_certificate),
                       cwd=str(ROOT), check=True, timeout=900)
        record, arrays = load_bundle(output)
        validate_group_artifact(record, arrays, diagnostic_contract(group),
                                experiment.identity(), commit, SOURCE_SHA, args.mode)
        if manifest['group_artifacts']:
            first = next(iter(manifest['group_artifacts'].values()))['manifest']
            compare_initialization(first, record)
        manifest['group_artifacts'][group] = dict(file=output.name,
            sha256=hashlib.sha256(output.read_bytes()).hexdigest(), manifest=record)
        print('[amp-physics-diagnostic-group-complete] '+json.dumps(
            dict(group=group, artifact=output.name, sha256=manifest['group_artifacts'][group]['sha256'])), flush=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Source checkpoint changed during frozen-policy diagnosis')
    certificate = dict(complete=True, mode=args.mode, groups=list(groups),
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=2500,
        group_artifacts_verified=True, all_finite=True, no_training=True,
        policy_updates=0, discriminator_updates=0, estimator_updates=0,
        effectiveness_verified=False, dr_unlocked=False)
    summary = folder/'model_amp_manifest.pt'
    if summary.exists():
        raise FileExistsError(summary)
    torch.save(dict(manifest_json=json.dumps(manifest), certificate_json=json.dumps(certificate)), summary)
    print('[amp-physics-diagnostic-complete] '+json.dumps(certificate), flush=True)
    # Gradmotion discovers and uploads generated PT artifacts asynchronously.
    time.sleep(60)


if __name__ == '__main__':
    main()
