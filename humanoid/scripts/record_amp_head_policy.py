"""Independent native final-head rollout; never a PPO checkpoint continuation.

Smoke probes the NEW recorder using only immutable original110. Formal rollouts
require a strictly validated, offline-admitted formal head archive. The old
record_amp_policy.py and its source/checkpoint guards remain untouched. Capture
is before native auto-reset, including every first-episode failure prefix.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from humanoid.scripts.collect_amp_head_cohort import (
    INITIAL_FIELDS, PHYSICS_FIELDS, SOURCE_SHA, SOURCE_MODEL_SHA,
    SOURCE_CONFIG, file_sha, numerical_fingerprint, registered_source_exclusions,
    validate_source_runtime)


SCHEMA = 'head_policy_rollout_v1'
MODE_SEEDS = dict(standing=5, reference=105)
BUDGETS = dict(smoke=(2, 1.), formal=(16, 60.))


def recorder_contract(mode, seed, num_envs, duration_s):
    if (mode not in BUDGETS or type(seed) is not int or seed != 5
            or type(num_envs) is not int or isinstance(duration_s, bool)
            or (num_envs, duration_s) != BUDGETS[mode]):
        raise ValueError('Head recorder permits only source-only2x1 smoke or fixed32x60 formal')
    return dict(type=SCHEMA, mode=mode, seed=seed, num_envs=num_envs,
        duration_s=float(duration_s), fps=100, mode_seeds=dict(MODE_SEEDS),
        modes=['standing', 'reference'], parent_completed_updates=2500,
        no_training=True, ppo_continuation=False,
        effectiveness_verified=False, dr_unlocked=False)


def require_candidate_admission(artifact, mode):
    """Additional admission AFTER strict head_artifact validation, not its replacement."""
    if mode not in BUDGETS:
        raise ValueError('Unknown head recorder admission mode')
    if mode != 'formal':
        if artifact is not None:
            raise ValueError('New recorder smoke is source-only, not candidate quality evidence')
        return True
    if (not isinstance(artifact, dict) or artifact.get('artifact_kind') != 'actor_head_offline_v1'
            or artifact.get('provenance', {}).get('mode') != 'formal'
            or artifact['provenance'].get('offline_admitted') is not True
            or type(artifact['provenance'].get('offline_admitted')) is not bool
            or artifact.get('effectiveness_verified') is not False
            or artifact.get('dr_unlocked') is not False):
        raise ValueError('Physical candidate requires strictly validated formal offline admission')
    return True


def initial_source_matches(initial, mode, exclusions, *, enforce):
    if mode not in MODE_SEEDS or set(initial) != set(INITIAL_FIELDS):
        raise ValueError('Incomplete actual12-field matched initial schema')
    n = len(initial['valid'])
    if any(np.asarray(value).shape[0] != n for value in initial.values()):
        raise ValueError('Misaligned native initial states')
    if enforce and n != 16:
        raise ValueError('Formal matching requires all16 actual initial states per mode')
    rows = []
    for i in range(n):
        digest = numerical_fingerprint({key: value[i] for key, value in initial.items()})
        label = 'source110/%s/env-%d/episode-0' % (mode, i)
        exact = label in exclusions['initial_state'].get(digest, [])
        rows.append(dict(env=i, source_episode_id=label, initial_state_sha256=digest, exact=exact))
    if enforce and not all(row['exact'] for row in rows):
        raise ValueError('Actual formal12 initial fields differ from the immutable original32 source')
    return dict(all_exact=all(row['exact'] for row in rows), formal_match_required=bool(enforce),
        fields=list(INITIAL_FIELDS), rows=rows,
        limitation='Smoke has a different environment count and is not the matched32 quality protocol.')


def audit_rollout_arrays(arrays, contract):
    """All initial conditions and terminal ticks remain present; no successful subset."""
    n, target = contract['num_envs'], int(contract['duration_s']*100)
    results = {}
    for mode in contract['modes']:
        valid = arrays[mode+'_valid']
        if valid.ndim != 2 or valid.dtype != np.bool_ or valid.shape[1] != n:
            raise ValueError('Malformed native first-episode validity')
        t = len(valid)
        if not 1 <= t <= target or not np.array_equal(arrays[mode+'_time'], (np.arange(t)+1)*.01):
            raise ValueError('Missing or nonconsecutive native control ticks')
        for key in INITIAL_FIELDS + PHYSICS_FIELDS + ('done', 'policy_raw_mu'):
            value = arrays[mode+'_'+key]
            if value.shape[:2] != (t, n) or not np.isfinite(value).all():
                raise ValueError('Missing/nonfinite pre-reset native '+key)
        for key in INITIAL_FIELDS:
            value = arrays[mode+'_initial_'+key]
            if value.shape[0] != n or not np.isfinite(value).all():
                raise ValueError('Missing/nonfinite actual initial '+key)
        for key in ('valid', 'failure', 'done', 'physics_valid'):
            if arrays[mode+'_'+key].dtype != np.bool_:
                raise ValueError('Actual native masks must be boolean: '+key)
        for key in ('valid', 'failure'):
            if arrays[mode+'_initial_'+key].dtype != np.bool_:
                raise ValueError('Actual native initial masks must be boolean: '+key)
        raw_shapes = dict(physics_raw_velocity=(10, 12), physics_raw_torque=(10, 12),
            physics_raw_body_force=(10, 3, 3), physics_raw_body_state=(10, 3, 13),
            physics_interval_initial_velocity=(12,), physics_interval_initial_torque=(12,))
        for key, shape in raw_shapes.items():
            value = arrays[mode+'_'+key]
            if value.shape != (t,)+shape or not np.isfinite(value).all():
                raise ValueError('Missing/nonfinite env0 native1ms interval '+key)
        episodes = []
        for i in range(n):
            count = int(valid[:, i].sum())
            if not np.array_equal(valid[:, i], np.arange(t) < count):
                raise ValueError('A restarted/missing episode entered native valid states')
            failure = arrays[mode+'_failure'][:count, i]
            done = arrays[mode+'_done'][:count, i]
            if (failure.any() and (not failure[-1] or failure[:-1].any())) or done[:-1].any():
                raise ValueError('Native terminal state was dropped or crossed')
            failed = bool(arrays[mode+'_initial_failure'][i] or failure.any())
            episodes.append(dict(env=i, valid_ticks=count, failed=failed,
                terminal_tick=None if not done.any() else count-1,
                complete=bool(count == target and not failed and not done.any()
                              and arrays[mode+'_initial_valid'][i])))
        results[mode] = episodes
    return results


def _native_main(extra, remaining):
    from isaacgym import gymapi, gymtorch  # Must precede direct/transitive torch.
    import torch
    from humanoid import LEGGED_GYM_ROOT_DIR
    from humanoid.algo.ppo.actor_critic_dh import ActorCriticDH
    from humanoid.amp.head_artifact import validate_head_artifact
    from humanoid.amp.learnability import assert_no_domain_randomization
    from humanoid.amp.mu_temporal import state_fingerprint
    from humanoid.amp.physics_diagnostic import (apply_diagnostic_config,
        cleared_history_diagnostics, reset_input_snapshot, physical_readback, validate_source_state)
    from humanoid.amp.recovery import FOOT_NAMES
    from humanoid.amp.refinement import select_environment
    from humanoid.amp.scaled_experiment import (ScaledExperiment,
        validate_runtime_timing, implementation_fingerprint)
    from humanoid.envs.x1.x1_amp_physics_diagnostic_env import X1AMPPhysicsDiagnosticEnv
    from humanoid.scripts.record_amp_policy import summarize
    from humanoid.utils import get_args, task_registry
    from humanoid.utils.helpers import class_to_dict, set_seed

    sys.argv = [sys.argv[0]]+remaining
    args = get_args()
    if (sys.platform != 'linux' or not torch.cuda.is_available() or args.sim_device_type != 'cuda'
            or args.rl_device != 'cuda:0' or not args.use_gpu_pipeline
            or args.physics_engine != gymapi.SIM_PHYSX or not args.headless
            or args.armature_mode != 'nominal' or args.training_profile or args.resume or args.horovod):
        raise ValueError('Head native recorder is Linux/CUDA/PhysX nominal source inference only')
    contract = recorder_contract(extra.mode, args.seed, args.num_envs, extra.duration)
    repo = Path(LEGGED_GYM_ROOT_DIR)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    if (not re.fullmatch('[0-9a-f]{40}', extra.expected_commit) or commit != extra.expected_commit
            or file_sha(extra.source_checkpoint) != SOURCE_SHA
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=str(repo)).strip()
            or subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--',
                                         'humanoid', 'configs', 'resources'], cwd=str(repo)).strip()):
        raise ValueError('Published clean native recorder and immutable original110 source are required')
    subprocess.check_call(['git', 'ls-files', '--error-unmatch',
        'humanoid/scripts/record_amp_head_policy.py'], cwd=str(repo), stdout=subprocess.DEVNULL)
    if extra.output.exists() or extra.output.with_suffix('.json').exists():
        raise FileExistsError(extra.output)
    experiment = ScaledExperiment(repo, repo/SOURCE_CONFIG)
    state = torch.load(str(extra.source_checkpoint), map_location='cpu', weights_only=True)
    validate_source_state(state, experiment.identity())
    if args.task != experiment.cfg['experiment'] or state_fingerprint(state['model_state_dict']) != SOURCE_MODEL_SHA:
        raise ValueError('Wrong original110 task/model identity')
    config = json.loads((repo/'configs/amp/head_smooth_v1.json').read_text(encoding='utf-8'))
    exclusions, source_proof, source_manifest = registered_source_exclusions(
        repo, repo/config['source_exclusions']['path'], experiment.identity())
    artifact, head_sha, archive_audit = None, None, None
    if extra.mode == 'formal':
        if extra.head_artifact is None or not re.fullmatch('[0-9a-f]{64}', extra.head_sha256 or ''):
            raise ValueError('Formal candidate requires an explicitly hash-bound offline head artifact')
        blob = extra.head_artifact.read_bytes()
        head_sha = hashlib.sha256(blob).hexdigest()
        if head_sha != extra.head_sha256:
            raise ValueError('Head artifact file SHA differs from actual received bytes')
        artifact = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        archive_audit = validate_head_artifact(artifact, extra.source_checkpoint)
        require_candidate_admission(artifact, extra.mode)
        if artifact['amp_identity'] != experiment.identity():
            raise ValueError('Offline head parent identity differs from original110 source')
        model_state = artifact['model_state_dict']
        expected_policy_sha = archive_audit['candidate_model_state_sha256']
    else:
        if extra.head_artifact is not None or extra.head_sha256 is not None:
            raise ValueError('Native recorder smoke must be source-only')
        require_candidate_admission(None, extra.mode)
        model_state = state['model_state_dict']
        expected_policy_sha = SOURCE_MODEL_SHA
    cfg, train_cfg, _ = select_environment('sustain_control')
    cfg = apply_diagnostic_config(cfg, 'original')
    cfg.seed = args.seed
    cfg.env.episode_length_s = extra.duration+.1
    gate = assert_no_domain_randomization(cfg)
    task_registry.register(args.task, X1AMPPhysicsDiagnosticEnv, cfg, train_cfg)
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        validate_runtime_timing(env.dt, env.sim_params.dt, cfg.control.decimation)
        if tuple(env.dof_names) != experiment.spec.joint_names or env.num_envs != contract['num_envs']:
            raise ValueError('Head native robot/joint order/count changed')
        env.configure_reference_initialization(experiment)
        policy = ActorCriticDH(env.num_short_obs, env.num_single_obs, env.num_privileged_obs,
                              env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
        policy.load_state_dict(model_state, strict=True)
        policy.eval()
        if state_fingerprint(policy.state_dict()) != expected_policy_sha:
            raise ValueError('Actual loaded candidate policy differs from the bound head state')
        body_ids = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
                    for name in experiment.spec.body_names]
        feet = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
                for name in FOOT_NAMES]
        if min(body_ids) < 0 or env.feet_indices.cpu().tolist() != feet:
            raise ValueError('Head native source body/foot order changed')
        print('[head-policy-start] '+json.dumps(dict(type=SCHEMA, mode=extra.mode,
            code_commit=commit, parent_checkpoint_sha256=SOURCE_SHA,
            head_artifact_sha256=head_sha, policy_model_state_sha256=expected_policy_sha,
            num_envs=env.num_envs, duration_s=extra.duration,
            control_dt=env.dt, physics_dt=env.sim_params.dt,
            parent_completed_updates=2500, no_training=True,
            effectiveness_verified=False, dr_unlocked=False)), flush=True)
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        frames, recording, raw_mu = [], False, None
        reset_inputs, reset_proofs, initial_matches = {}, {}, {}
        reset_pending = False
        original_reset_idx, original_reset_dofs = env.reset_idx, env._reset_dofs
        def reset_and_prove(ids):
            nonlocal reset_pending
            original_reset_idx(ids)
            if reset_pending:
                if len(ids) != env.num_envs:
                    raise ValueError('Native mode reset did not inject every fixed initial state')
                reset_proofs[mode] = cleared_history_diagnostics(env)
                reset_inputs[mode] = reset_input_snapshot(env)
                reset_pending = False
        env.reset_idx = reset_and_prove
        def fixed_reset(ids):
            env.dof_pos[ids] = env.default_dof_pos
            env.dof_vel[ids] = 0.
            ids32 = ids.to(torch.int32)
            env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state),
                                                gymtorch.unwrap_tensor(ids32), len(ids))
        def capture():
            root = cpu(env.root_states); root[:, :3] -= cpu(env.env_origins)
            foot = cpu(env.rigid_state[:, env.feet_indices]); foot[:, :, :3] -= cpu(env.env_origins)[:, None, :]
            result = dict(root_state=root, dof_pos=cpu(env.dof_pos), dof_vel=cpu(env.dof_vel),
                action=cpu(env.actions), torque=cpu(env.torques), base_lin_vel=cpu(env.base_lin_vel),
                base_ang_vel=cpu(env.base_ang_vel), foot_state=foot,
                foot_force=cpu(env.contact_forces[:, env.feet_indices]),
                key_positions_w=cpu(env.rigid_state[:, body_ids, :3]-env.env_origins[:, None, :]),
                valid=cpu(alive), failure=cpu(env.reset_buf.bool() & ~env.time_out_buf.bool()),
                done=cpu(env.reset_buf.bool()))
            result.update({key: cpu(value) for key, value in env.jitter_capture().items()})
            return result
        original_termination = env.check_termination
        def check_and_capture():
            original_termination()
            if recording:
                frame = capture()  # before reset; keeps terminal guard data.
                frame['policy_raw_mu'] = raw_mu.copy()
                frames.append(frame)
        env.check_termination = check_and_capture
        arrays, summaries = {}, {}
        for mode in contract['modes']:
            set_seed(MODE_SEEDS[mode])
            recording = False
            env.rsi_enabled = mode == 'reference'
            env._reset_dofs = fixed_reset if mode == 'standing' else original_reset_dofs
            alive[:] = True
            reset_pending = True
            obs, _ = env.reset()
            initial = capture()
            initial_fields = {key: initial[key] for key in INITIAL_FIELDS}
            initial_matches[mode] = initial_source_matches(initial_fields, mode, exclusions,
                                                           enforce=extra.mode == 'formal')
            if reset_pending:
                raise ValueError('Missing actual native reset history proof')
            alive &= ~env.reset_buf.bool()
            frames, recording = [], True
            with torch.no_grad():
                for tick in range(int(extra.duration*100)):
                    actions = policy.act_inference(obs)  # ONE original deterministic forward.
                    if not torch.isfinite(actions).all():
                        raise ValueError('Nonfinite candidate native mean')
                    raw_mu = cpu(actions)
                    obs, _, _, dones, _ = env.step(actions)
                    alive &= ~dones.bool()
                    if not alive.any():
                        break
                    if (tick+1) % 500 == 0:
                        print('[head-policy-progress] '+json.dumps(dict(mode=mode,
                            seconds=(tick+1)*.01, alive=int(alive.sum()))), flush=True)
            recording = False
            data = {key: np.stack([row[key] for row in frames]) for key in frames[0]}
            data['time'] = (np.arange(len(frames))+1)*.01
            data['initial_failure'] = initial['failure']
            for key, value in data.items():
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite native head rollout '+key)
                arrays[mode+'_'+key] = value
            for key in INITIAL_FIELDS:
                arrays[mode+'_initial_'+key] = initial[key]
            summaries[mode] = summarize(data, extra.duration)
        episodes = audit_rollout_arrays(arrays, contract)
        for mode in contract['modes']:
            mask = arrays[mode+'_valid']
            clipped = np.clip(arrays[mode+'_policy_raw_mu'], -cfg.normalization.clip_actions,
                              cfg.normalization.clip_actions)
            if not np.array_equal(arrays[mode+'_action'][mask], clipped[mask]):
                raise ValueError('Actual candidate command differs from original clipped PD action')
        if state_fingerprint(policy.state_dict()) != expected_policy_sha or file_sha(extra.source_checkpoint) != SOURCE_SHA:
            raise ValueError('Native collection changed candidate/parent weights')
        if extra.mode == 'formal' and file_sha(extra.head_artifact) != head_sha:
            raise ValueError('Head artifact bytes changed during native collection')
        props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
        environment = class_to_dict(cfg)
        runtime = dict(dof_properties={key: props[key].tolist() for key in props.dtype.names},
            p_gains=cpu(env.p_gains).tolist(), d_gains=cpu(env.d_gains).tolist(),
            control_dt=env.dt, physics_dt=env.sim_params.dt, **physical_readback(env))
        runtime_proof = validate_source_runtime(environment, runtime, source_manifest,
            dict(seed=5, num_envs=contract['num_envs'], duration_s=contract['duration_s']))
        manifest = dict(contract, code_commit=commit, implementation_fingerprint=implementation_fingerprint(repo),
            identity=experiment.identity(), source_identity=experiment.identity(),
            parent_checkpoint_sha256=SOURCE_SHA, parent_model_state_sha256=SOURCE_MODEL_SHA,
            head_artifact_sha256=head_sha, policy_model_state_sha256=expected_policy_sha,
            policy_variant='offline_final_head_candidate' if extra.mode == 'formal' else 'source_smoke_probe',
            head_archive_audit=archive_audit, source_regression=source_proof,
            no_dr=gate, environment=environment, runtime=runtime, source_runtime_proof=runtime_proof,
            policy_deterministic=True, dof_names=env.dof_names, body_names=list(experiment.spec.body_names),
            foot_names=list(FOOT_NAMES), urdf_lf_sha256=experiment.kinematics.lf_sha256,
            initialization=dict(protocol='identical_reset_inputs_then_native_10ms_warmup',
                reset_inputs=reset_inputs, cleared_histories=reset_proofs, initial_source_matches=initial_matches),
            summaries=summaries, episodes=episodes,
            evaluation_protocol='head_source_smoke' if extra.mode == 'smoke' else 'head_fixed32_source_matched60',
            capture='pre-reset; retain full failing first-episode prefixes and invalid tails',
            action_semantics='policy_raw_mu is actual unclipped deterministic mean; action is actual clipped PD command',
            substep_telemetry=dict(physics_hz=1000, control_hz=100, samples_per_control=10,
                raw_env=0, raw_body_names=list(env.physics_raw_body_names), reward_used=False,
                reset_guard='native physics_valid excludes first three reset intervals',
                acceleration='native1ms velocity difference/.001 mean squared and peak absolute',
                torque_delta='actual1ms torque difference mean squared; not derivative',
                foot_force='max positive NET footFz, not contact-pair identity'))
        packed = io.BytesIO(); np.savez_compressed(packed, **arrays)
        extra.output.parent.mkdir(parents=True, exist_ok=True)
        with extra.output.open('xb') as stream:
            torch.save(dict(npz_bytes=packed.getvalue(), manifest_json=json.dumps(manifest)), stream)
        with extra.output.with_suffix('.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
        print('[head-policy-complete] '+json.dumps(dict(type=SCHEMA, mode=extra.mode,
            parent_checkpoint_sha256=SOURCE_SHA, head_artifact_sha256=head_sha,
            policy_model_state_sha256=expected_policy_sha, episodes=episodes,
            effectiveness_verified=False, dr_unlocked=False)), flush=True)
        return manifest
    finally:
        env.gym.destroy_sim(env.sim)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--mode', choices=tuple(BUDGETS), required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--head-artifact', type=Path)
    parser.add_argument('--head-sha256')
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    extra, remaining = parser.parse_known_args()
    return _native_main(extra, remaining)


if __name__ == '__main__':
    main()
