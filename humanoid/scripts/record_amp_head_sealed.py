"""Independent sealed-initial-state source/head pair, never PPO continuation.

The public formal-only wrapper runs source and candidate in TWO fresh native
processes. Neither arm shares a simulator/contact cache with the other. Actual
12 initial fields and the complete first 66x47 input must equal the previously
sealed source cohort for all eight environments. Failed first-episode prefixes
are retained, not filtered into a successful subset. CPU tests are protocol
tests, not evidence of native physics, improvement, or statistical independence.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

from humanoid.scripts.collect_amp_head_cohort import (
    INITIAL_FIELDS, SOURCE_SHA, SOURCE_MODEL_SHA, SOURCE_CONFIG,
    SOURCE_ENDPOINT_SHA, SEEDS, cohort_contract, file_sha,
    full_history_fingerprint, numerical_fingerprint, registered_source_exclusions,
    validate_source_runtime, _read_bundle)
from humanoid.scripts.record_amp_head_policy import (
    audit_rollout_arrays, require_candidate_admission)


SCHEMA = 'head_policy_sealed_rollout_v1'
ARM_SCHEMA = 'head_policy_sealed_arm_v1'
ARMS = ('source', 'candidate')


def sealed_recorder_contract(mode, seed, num_envs, duration_s):
    if (mode != 'formal' or type(seed) is not int or seed != 706
            or type(num_envs) is not int or num_envs != 8
            or isinstance(duration_s, bool) or duration_s != 60.):
        raise ValueError('Sealed recorder requires formal reference8 seed706 for60s')
    return dict(type=SCHEMA, mode=mode, seed=seed, num_envs=num_envs,
        duration_s=float(duration_s), fps=100, modes=['reference'],
        parent_completed_updates=2500, no_training=True, ppo_continuation=False,
        effectiveness_verified=False, dr_unlocked=False,
        episode_ids=cohort_contract('sealed', 'formal', 706, 8, 20.)['episode_ids'])


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Missing actual file/model SHA: '+label)
    return value


def _cohort_file_proof(role, item):
    contract = cohort_contract(role, 'formal', SEEDS[role], 8, 20.)
    return dict(sha256=_sha(item['sha256'], role), split=role,
        seed=contract['seed'], num_envs=8, duration_s=20.,
        episode_ids=contract['episode_ids'])


def audit_sealed_inputs(cohorts, *, artifact, identity, code_commit,
                        implementation_fingerprint, exclusions,
                        source_proof, source_manifest):
    """Recompute EVERY actual source/prior/sealed state/history exclusion.

    ``cohorts`` has train/validation/sealed entries, each containing actual
    manifest/arrays and a SHA of the received file. The caller must first run
    strict validate_head_artifact against the original checkpoint. No actor
    forward or randomness is used here. A failed sealed20 episode remains a
    held-out initial condition, never a fitting episode or a discarded row.
    """
    from humanoid.amp.head_workflow import audit_cohort, exclusion_add_cohort
    require_candidate_admission(artifact, 'formal')
    if set(cohorts) != {'train', 'validation', 'sealed'}:
        raise ValueError('Actual train, validation and sealed files are all mandatory')
    provenance = artifact['provenance']
    if (provenance.get('code_commit') != code_commit
            or provenance.get('implementation_fingerprint') != implementation_fingerprint
            or artifact.get('amp_identity') != identity
            or source_proof.get('sha256') != SOURCE_ENDPOINT_SHA):
        raise ValueError('Sealed inputs must use this published admitted head/source definition')
    proofs = {role: _cohort_file_proof(role, item) for role, item in cohorts.items()}
    if len({item['sha256'] for item in proofs.values()}) != 3:
        raise ValueError('Train/validation/sealed archives cannot reuse the same file')
    for role in ('train', 'validation'):
        actual = {key: value for key, value in proofs[role].items() if key != 'split'}
        if provenance.get('cohorts', {}).get(role) != actual:
            raise ValueError('Actual '+role+' file differs from the fitted head provenance')
    index = copy.deepcopy(exclusions)
    audits = {}
    earlier = []
    for role in ('train', 'validation', 'sealed'):
        item, expected = cohorts[role], proofs[role]
        manifest = item['manifest']
        if (manifest.get('source_regression') != source_proof
                or manifest.get('previous_cohorts') != earlier):
            raise ValueError('Missing actual original32/prior split provenance for '+role)
        result = audit_cohort(manifest, item['arrays'], split=role, mode='formal',
            identity=identity, code_commit=code_commit,
            implementation_fingerprint=implementation_fingerprint,
            exclusions=index, source_manifest=source_manifest)
        if not result['deduplication']['passed']:
            raise ValueError('Actual sealed/prior state or full history overlaps another episode')
        if role != 'sealed' and not result['fit_eligible']:
            raise ValueError('Fitted head prior cohort is not whole-episode fit eligible')
        if role == 'sealed' and result['fit_eligible']:
            raise ValueError('Sealed data must never be fit eligible')
        audits[role] = {key: result[key] for key in
            ('contract', 'history', 'episodes', 'fingerprints', 'deduplication', 'fit_eligible')}
        exclusion_add_cohort(index, result)
        earlier.append(expected)
    return dict(cohorts=proofs, audits=audits, source_regression=copy.deepcopy(source_proof),
        all_actual_arrays_checked=True, sealed_fit_eligible=False,
        no_survivor_selection=True,
        limitation='Exact state/history exclusion is not statistical independence or task novelty.')


def initial_sealed_matches(arrays, sealed_arrays, episode_ids):
    """Exact dtype/shape/value/row matches, signed zero equivalent on copies."""
    expected_ids = cohort_contract('sealed', 'formal', 706, 8, 20.)['episode_ids']
    if episode_ids != expected_ids:
        raise ValueError('All eight canonical sealed episode identities are required')
    for key in INITIAL_FIELDS+('full_obs',):
        name = 'reference_initial_'+key
        actual, expected = arrays[name], sealed_arrays[name]
        if (not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray)
                or actual.shape != expected.shape or actual.shape[0] != 8
                or actual.dtype != expected.dtype or not np.isfinite(actual).all()
                or not np.isfinite(expected).all() or not np.array_equal(actual, expected)):
            raise ValueError('Actual sealed initial field/input mismatch: '+key)
    if ('reference_initial_native_episode' in arrays
            and not np.array_equal(arrays['reference_initial_native_episode'],
                                   sealed_arrays['reference_initial_native_episode'])):
        raise ValueError('Actual sealed initial native episode differs from the sealed source')
    if (arrays['reference_initial_full_obs'].shape != (8, 3102)
            or arrays['reference_initial_full_obs'].dtype != np.float32):
        raise ValueError('Missing complete actual first66x47 input')
    rows = []
    for i, label in enumerate(expected_ids):
        state = numerical_fingerprint({key: arrays['reference_initial_'+key][i]
                                       for key in INITIAL_FIELDS})
        first = full_history_fingerprint(arrays['reference_initial_full_obs'][i])
        rows.append(dict(env=i, episode_id=label, initial_state_sha256=state,
            initial_full_observation_sha256=first, all12_exact=True, full_input_exact=True))
    return dict(all_exact=True, fields=list(INITIAL_FIELDS), full_input_shape=[8, 3102],
        rows=rows, numerical_equivalence='exact dtype/shape/values; signed zeros equivalent',
        limitation='Observed initial equivalence does not snapshot hidden PhysX contact state '
                   'or guarantee bitwise-identical future closed-loop trajectories.')


def audit_sealed_arm(manifest, arrays, *, arm, contract, identity, code_commit,
                     implementation_fingerprint, head_sha256, candidate_model_sha256,
                     inputs, sealed_arrays, source_manifest):
    """Read back actual arm arrays/runtime; no survival/effectiveness gate."""
    from humanoid.amp.learnability import assert_no_domain_randomization
    if arm not in ARMS:
        raise ValueError('Unknown physical arm')
    expected = dict(contract, type=ARM_SCHEMA, arm=arm, code_commit=code_commit,
        implementation_fingerprint=implementation_fingerprint, source_identity=identity,
        parent_checkpoint_sha256=SOURCE_SHA, parent_model_state_sha256=SOURCE_MODEL_SHA,
        head_artifact_sha256=_sha(head_sha256, 'head'),
        policy_model_state_sha256=SOURCE_MODEL_SHA if arm == 'source' else
            _sha(candidate_model_sha256, 'candidate'),
        policy_deterministic=True, parent_frozen=True)
    if any(manifest.get(key) != value or (type(value) in (bool, int)
           and type(manifest.get(key)) is not type(value)) for key, value in expected.items()):
        raise ValueError('Actual sealed arm source/head/code/protocol mismatch')
    if any(key in manifest for key in ('completed_updates', 'iter', 'optimizer_state_dict')):
        raise ValueError('Head physical arm cannot claim PPO continuation/update counters')
    if (manifest.get('cohorts') != inputs['cohorts']
            or manifest.get('source_regression') != inputs['source_regression']):
        raise ValueError('Physical arm substituted a sealed/prior/source file')
    assert_no_domain_randomization(manifest['environment'])
    runtime_proof = validate_source_runtime(manifest['environment'], manifest['runtime'],
                                            source_manifest, contract)
    if manifest.get('source_runtime_proof') != runtime_proof:
        raise ValueError('Sealed arm physics/PD proof does not match actual source readback')
    proof = initial_sealed_matches(arrays, sealed_arrays, contract['episode_ids'])
    if manifest.get('initial_sealed_matches') != proof:
        raise ValueError('Actual sealed initial/input proof differs from recorded claim')
    episodes = audit_rollout_arrays(arrays, contract)
    if manifest.get('episodes') != episodes:
        raise ValueError('Sealed arm terminal prefixes differ from actual arrays')
    valid = arrays['reference_valid']
    t = len(valid)
    for key in INITIAL_FIELDS:
        initial = arrays['reference_initial_'+key]
        if (arrays['reference_'+key].shape != (t,)+initial.shape
                or arrays['reference_'+key].dtype != initial.dtype):
            raise ValueError('Native sealed state dimensions/dtypes differ from actual initials: '+key)
    # Collection is bounded by60s OR every arm environment's real termination,
    # not an arbitrary early crop. A failure is evidence, not an admission fail.
    for row in episodes['reference']:
        i, count = row['env'], row['valid_ticks']
        initial_failed = bool(arrays['reference_initial_failure'][i])
        if initial_failed:
            if count != 0:
                raise ValueError('Warmup-failed sealed episode entered a later valid prefix')
        elif row['failed']:
            if not count or not arrays['reference_done'][count-1, i]:
                raise ValueError('Sealed failed prefix omitted its actual native terminal tick')
        elif not row['complete']:
            raise ValueError('Sealed nonfailed episode was prematurely truncated before60s')
    native_episode = arrays['reference_native_episode']
    initial_episode = arrays['reference_initial_native_episode']
    if (native_episode.shape != (t, 8) or native_episode.dtype != np.int64
            or initial_episode.shape != (8,) or initial_episode.dtype != np.int64
            or np.any(native_episode[valid] != np.broadcast_to(initial_episode, (t, 8))[valid])):
        raise ValueError('Physical sealed valid prefix crossed a native reset')
    raw, applied = arrays['reference_policy_raw_mu'], arrays['reference_action']
    if (raw.shape != (t, 8, 12) or raw.dtype != np.float32
            or applied.shape != raw.shape or applied.dtype != np.float32):
        raise ValueError('Missing actual native mean/PD action arrays')
    clip = manifest['environment']['normalization']['clip_actions']
    if not np.array_equal(applied[valid], np.clip(raw, -clip, clip)[valid]):
        raise ValueError('Native sealed applied action differs from original clipping')
    capture = manifest.get('inference_capture')
    if capture != dict(actual_forwards=t, extra_forward=False, model_state_unchanged=True):
        raise ValueError('Missing truthful one-forward-per-native-decision capture')
    return dict(episodes=episodes, initial_sealed_matches=proof,
                source_runtime_proof=runtime_proof, frames=t)


def assemble_sealed_pair(source_manifest, source_arrays, candidate_manifest,
                         candidate_arrays, *, contract, identity, code_commit,
                         implementation_fingerprint, head_sha256,
                         candidate_model_sha256, inputs, sealed_arrays,
                         source_runtime_manifest, arm_file_shas):
    """Combine two real arm archives, retaining every row and every failure."""
    if set(arm_file_shas) != set(ARMS) or len(set(arm_file_shas.values())) != 2:
        raise ValueError('Two independently received arm files are mandatory')
    manifests = dict(source=source_manifest, candidate=candidate_manifest)
    arrays_by_arm = dict(source=source_arrays, candidate=candidate_arrays)
    proofs = {}
    for arm in ARMS:
        _sha(arm_file_shas[arm], arm)
        proofs[arm] = audit_sealed_arm(manifests[arm], arrays_by_arm[arm], arm=arm,
            contract=contract, identity=identity, code_commit=code_commit,
            implementation_fingerprint=implementation_fingerprint,
            head_sha256=head_sha256, candidate_model_sha256=candidate_model_sha256,
            inputs=inputs, sealed_arrays=sealed_arrays, source_manifest=source_runtime_manifest)
    if proofs['source']['initial_sealed_matches'] != proofs['candidate']['initial_sealed_matches']:
        raise ValueError('Paired arms do not share all eight actual sealed initial states/inputs')
    # Prefixing does not pad/truncate/reorder/select any arm rows or environments.
    arrays = {arm+'_'+key: value for arm in ARMS for key, value in arrays_by_arm[arm].items()}
    manifest = dict(contract, code_commit=code_commit,
        implementation_fingerprint=implementation_fingerprint, source_identity=identity,
        parent_checkpoint_sha256=SOURCE_SHA, parent_model_state_sha256=SOURCE_MODEL_SHA,
        head_artifact_sha256=head_sha256, candidate_model_state_sha256=candidate_model_sha256,
        cohorts=copy.deepcopy(inputs['cohorts']), cohort_audit=copy.deepcopy(inputs),
        arms={arm: dict(manifest=copy.deepcopy(manifests[arm]),
            arm_file_sha256=arm_file_shas[arm], actual_audit=proofs[arm]) for arm in ARMS},
        initial_sealed_matches=proofs['source']['initial_sealed_matches'],
        evaluation_protocol='new_reference8_seed706_source_head_paired60',
        array_naming='source_reference_* and candidate_reference_* retain native reference field semantics',
        process_isolation='source and candidate run in separate fresh native processes/simulators',
        capture='all eight first-episode prefixes, terminal ticks and invalid tails; no survivor selection',
        limitation='Sealed RSI-initial-state validation is not new motion/task or hardware '
                   'generalization. Process success is not survival or improvement. '
                   'Observed exact initials do not prove bitwise future PhysX replay.')
    return manifest, arrays


def arm_command(extra, remaining, arm, output):
    """Exactly one fresh process per arm; non-model temporary artifacts retained."""
    if arm not in ARMS or Path(output).name != arm+'-arm.pt':
        raise ValueError('Unregistered arm/temporary file route')
    return [sys.executable, '-m', 'humanoid.scripts.record_amp_head_sealed',
        '--mode', 'formal', '--source-checkpoint', str(extra.source_checkpoint),
        '--head-artifact', str(extra.head_artifact), '--head-sha256', extra.head_sha256,
        '--expected-commit', extra.expected_commit, '--sealed-cohort', str(extra.sealed_cohort),
        '--train-cohort', str(extra.train_cohort), '--validation-cohort', str(extra.validation_cohort),
        '--output', str(output), '--duration', '60', '--arm', arm]+list(remaining)


def _write_bundle(output, manifest, arrays, torch):
    packed = io.BytesIO()
    np.savez_compressed(packed, **arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save(dict(npz_bytes=packed.getvalue(),
                        manifest_json=json.dumps(manifest, allow_nan=False)), stream)
    with output.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)


def _collect_arm(extra, args, context):
    from isaacgym import gymtorch  # native ABI: before direct/transitive torch.
    import torch
    from humanoid.algo.ppo.actor_critic_dh import ActorCriticDH
    from humanoid.amp.learnability import assert_no_domain_randomization
    from humanoid.amp.mu_temporal import state_fingerprint
    from humanoid.amp.physics_diagnostic import (apply_diagnostic_config,
        cleared_history_diagnostics, reset_input_snapshot, physical_readback)
    from humanoid.amp.recovery import FOOT_NAMES
    from humanoid.amp.refinement import select_environment
    from humanoid.amp.scaled_experiment import validate_runtime_timing
    from humanoid.envs.x1.x1_amp_physics_diagnostic_env import X1AMPPhysicsDiagnosticEnv
    from humanoid.scripts.record_amp_policy import summarize
    from humanoid.utils import task_registry
    from humanoid.utils.helpers import class_to_dict, set_seed

    contract, experiment = context['contract'], context['experiment']
    state = context['source_state']['model_state_dict'] if extra.arm == 'source' else context['artifact']['model_state_dict']
    policy_sha = SOURCE_MODEL_SHA if extra.arm == 'source' else context['candidate_sha']
    cfg, train_cfg, _ = select_environment('sustain_control')
    cfg = apply_diagnostic_config(cfg, 'original')  # validate only; original physics/PD.
    cfg.seed, cfg.env.episode_length_s = 706, 60.1
    gate = assert_no_domain_randomization(cfg)
    task_registry.register(args.task, X1AMPPhysicsDiagnosticEnv, cfg, train_cfg)
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        validate_runtime_timing(env.dt, env.sim_params.dt, cfg.control.decimation)
        if tuple(env.dof_names) != experiment.spec.joint_names or env.num_envs != 8:
            raise ValueError('Sealed native robot/joint order/count changed')
        env.configure_reference_initialization(experiment)
        policy = ActorCriticDH(env.num_short_obs, env.num_single_obs, env.num_privileged_obs,
                              env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
        policy.load_state_dict(state, strict=True)
        policy.eval()
        if state_fingerprint(policy.state_dict()) != policy_sha:
            raise ValueError('Loaded native sealed policy fingerprint differs from the bound state')
        body_ids = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
                    for name in experiment.spec.body_names]
        feet = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
                for name in FOOT_NAMES]
        if min(body_ids) < 0 or env.feet_indices.cpu().tolist() != feet:
            raise ValueError('Sealed native body/foot order changed')
        print('[head-sealed-arm-start] '+json.dumps(dict(type=ARM_SCHEMA, arm=extra.arm,
            code_commit=context['commit'], parent_checkpoint_sha256=SOURCE_SHA,
            head_artifact_sha256=extra.head_sha256, policy_model_state_sha256=policy_sha,
            num_envs=8, seed=706, duration_s=60., control_dt=env.dt,
            physics_dt=env.sim_params.dt, no_training=True)), flush=True)
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        alive = torch.ones(8, dtype=torch.bool, device=env.device)
        frames, recording, raw_mu = [], False, None
        reset_pending = True
        reset_inputs, reset_proofs = {}, {}
        original_reset_idx = env.reset_idx
        def reset_and_prove(ids):
            nonlocal reset_pending
            original_reset_idx(ids)
            if reset_pending:
                if len(ids) != 8:
                    raise ValueError('Sealed reset must inject all eight registered initials')
                reset_proofs.update(cleared_history_diagnostics(env))
                reset_inputs.update(reset_input_snapshot(env))
                reset_pending = False
        env.reset_idx = reset_and_prove
        def capture():
            root = cpu(env.root_states); root[:, :3] -= cpu(env.env_origins)
            foot = cpu(env.rigid_state[:, env.feet_indices]); foot[:, :, :3] -= cpu(env.env_origins)[:, None, :]
            result = dict(root_state=root, dof_pos=cpu(env.dof_pos), dof_vel=cpu(env.dof_vel),
                action=cpu(env.actions), torque=cpu(env.torques), base_lin_vel=cpu(env.base_lin_vel),
                base_ang_vel=cpu(env.base_ang_vel), foot_state=foot,
                foot_force=cpu(env.contact_forces[:, env.feet_indices]),
                key_positions_w=cpu(env.rigid_state[:, body_ids, :3]-env.env_origins[:, None, :]),
                valid=cpu(alive), failure=cpu(env.reset_buf.bool() & ~env.time_out_buf.bool()),
                done=cpu(env.reset_buf.bool()), native_episode=cpu(env.amp_episode))
            result.update({key: cpu(value) for key, value in env.jitter_capture().items()})
            return result
        original_termination = env.check_termination
        def check_and_capture():
            original_termination()
            if recording:
                value = capture()  # terminal native state BEFORE auto reset.
                value['policy_raw_mu'] = raw_mu.copy()
                frames.append(value)
        env.check_termination = check_and_capture
        # Construction consumes initialization RNG before this same registered
        # seed/reset protocol. No replay/hidden injection/pre-forward is used.
        set_seed(706)
        env.rsi_enabled = True
        obs, _ = env.reset()  # unchanged original reference zero-action10ms warmup.
        initial, initial_obs = capture(), cpu(obs)
        initial_arrays = {'reference_initial_'+key: initial[key] for key in INITIAL_FIELDS}
        initial_arrays['reference_initial_full_obs'] = initial_obs
        initial_matches = initial_sealed_matches(initial_arrays, context['sealed_arrays'], contract['episode_ids'])
        if reset_pending:
            raise ValueError('Missing actual native cleared-history/reset proof')
        # Each arm also matches the observed pre-warmup registered reset inputs.
        sealed_init = context['sealed_manifest']['initialization']
        if (reset_inputs != sealed_init['reset_inputs']
                or reset_proofs != sealed_init['cleared_histories']):
            raise ValueError('Native sealed reset inputs/history-clear proof differ from the sealed source')
        alive &= ~env.reset_buf.bool()
        recording, actual_forwards = True, 0
        with torch.no_grad():
            for tick in range(6000):
                actions = policy.act_inference(obs)  # ONE deterministic ordinary forward per tick.
                actual_forwards += 1
                if not torch.isfinite(actions).all():
                    raise ValueError('Nonfinite native sealed policy mean')
                raw_mu = cpu(actions)
                obs, _, _, dones, _ = env.step(actions)
                alive &= ~dones.bool()
                if not alive.any():
                    break
                if (tick+1) % 500 == 0:
                    print('[head-sealed-arm-progress] '+json.dumps(dict(arm=extra.arm,
                        seconds=(tick+1)*.01, alive=int(alive.sum()))), flush=True)
        recording = False
        arrays = {'reference_'+key: np.stack([row[key] for row in frames]) for key in frames[0]}
        arrays['reference_time'] = (np.arange(len(frames))+1)*.01
        arrays.update(initial_arrays)
        arrays['reference_initial_native_episode'] = initial['native_episode']
        episodes = audit_rollout_arrays(arrays, contract)
        if actual_forwards != len(frames) or state_fingerprint(policy.state_dict()) != policy_sha:
            raise ValueError('Extra native sealed forward or policy weight mutation')
        _assert_input_files(context)
        props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
        environment = class_to_dict(cfg)
        runtime = dict(dof_properties={key: props[key].tolist() for key in props.dtype.names},
            p_gains=cpu(env.p_gains).tolist(), d_gains=cpu(env.d_gains).tolist(),
            control_dt=env.dt, physics_dt=env.sim_params.dt, **physical_readback(env))
        runtime_proof = validate_source_runtime(environment, runtime, context['source_manifest'], contract)
        summary_data = {key[len('reference_'):]: value for key, value in arrays.items()
                        if key.startswith('reference_') and not key.startswith('reference_initial_')}
        summary_data['initial_failure'] = initial['failure']
        manifest = dict(contract, type=ARM_SCHEMA, arm=extra.arm,
            code_commit=context['commit'], implementation_fingerprint=context['fingerprint'],
            source_identity=experiment.identity(), parent_checkpoint_sha256=SOURCE_SHA,
            parent_model_state_sha256=SOURCE_MODEL_SHA, head_artifact_sha256=extra.head_sha256,
            policy_model_state_sha256=policy_sha, parent_frozen=True, policy_deterministic=True,
            cohorts=context['inputs']['cohorts'], source_regression=context['source_proof'],
            head_archive_audit=context['archive_audit'], no_dr=gate,
            environment=environment, runtime=runtime, source_runtime_proof=runtime_proof,
            dof_names=env.dof_names, body_names=list(experiment.spec.body_names),
            foot_names=list(FOOT_NAMES), urdf_lf_sha256=experiment.kinematics.lf_sha256,
            initialization=dict(protocol='original_reference_reset_then_native_10ms_zero_action_warmup',
                reset_inputs=reset_inputs, cleared_histories=reset_proofs),
            initial_sealed_matches=initial_matches, episodes=episodes,
            summaries=dict(reference=summarize(summary_data, 60.)),
            inference_capture=dict(actual_forwards=actual_forwards, extra_forward=False, model_state_unchanged=True),
            process_id=__import__('os').getpid(),
            capture='pre-reset first-episode terminal prefixes and invalid tails; no survivor selection',
            action_semantics='policy_raw_mu actual deterministic mean; action actual original clipped PD command',
            substep_telemetry=dict(physics_hz=1000, control_hz=100, samples_per_control=10,
                raw_env=0, raw_body_names=list(env.physics_raw_body_names), reward_used=False,
                reset_guard='native physics_valid excludes first three reset intervals',
                acceleration='native1ms dq/.001 mean squared and peak absolute',
                torque_delta='actual1ms torque difference mean squared; not derivative',
                foot_force='max positive NET footFz, not contact-pair identity'))
        audit_sealed_arm(manifest, arrays, arm=extra.arm, contract=contract,
            identity=experiment.identity(), code_commit=context['commit'],
            implementation_fingerprint=context['fingerprint'], head_sha256=extra.head_sha256,
            candidate_model_sha256=context['candidate_sha'], inputs=context['inputs'],
            sealed_arrays=context['sealed_arrays'], source_manifest=context['source_manifest'])
        _write_bundle(extra.output, manifest, arrays, torch)
        print('[head-sealed-arm-complete] '+json.dumps(dict(type=ARM_SCHEMA, arm=extra.arm,
            episodes=episodes, policy_model_state_sha256=policy_sha,
            effectiveness_verified=False, dr_unlocked=False)), flush=True)
        return manifest
    finally:
        env.gym.destroy_sim(env.sim)


def _assert_input_files(context):
    for path, expected in context['file_bindings']:
        if file_sha(path) != expected:
            raise ValueError('Actual source/head/cohort input bytes changed during sealed evaluation')


def _native_main(extra, remaining):
    from isaacgym import gymapi  # must precede torch and all transitive torch imports.
    import torch
    from humanoid import LEGGED_GYM_ROOT_DIR
    # Match the native environment-first registry bootstrap; utils-first has
    # a circular import through envs/base/legged_robot_config.
    import humanoid.envs  # noqa: F401
    from humanoid.amp.head_artifact import validate_head_artifact
    from humanoid.amp.mu_temporal import state_fingerprint
    from humanoid.amp.physics_diagnostic import validate_source_state
    from humanoid.amp.scaled_experiment import ScaledExperiment, implementation_fingerprint
    from humanoid.utils import get_args

    sys.argv = [sys.argv[0]]+remaining
    args = get_args()
    if (sys.platform != 'linux' or not torch.cuda.is_available() or args.sim_device_type != 'cuda'
            or args.rl_device != 'cuda:0' or args.sim_device != 'cuda:0' or not args.use_gpu_pipeline
            or args.physics_engine != gymapi.SIM_PHYSX or not args.headless
            or args.armature_mode != 'nominal' or args.training_profile or args.resume or args.horovod):
        raise ValueError('Sealed native recorder requires original Linux/CUDA0/PhysX GPU nominal inference')
    contract = sealed_recorder_contract(extra.mode, args.seed, args.num_envs, extra.duration)
    repo = Path(LEGGED_GYM_ROOT_DIR)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    if (not re.fullmatch('[0-9a-f]{40}', extra.expected_commit) or commit != extra.expected_commit
            or file_sha(extra.source_checkpoint) != SOURCE_SHA
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=str(repo)).strip()
            or subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--',
                                        'humanoid', 'configs', 'resources'], cwd=str(repo)).strip()):
        raise ValueError('Published clean native sealed recorder and immutable original110 source required')
    subprocess.check_call(['git', 'ls-files', '--error-unmatch',
        'humanoid/scripts/record_amp_head_sealed.py'], cwd=str(repo), stdout=subprocess.DEVNULL)
    if extra.output.exists() or extra.output.with_suffix('.json').exists():
        raise FileExistsError(extra.output)
    if extra.arm is not None and extra.output.name != extra.arm+'-arm.pt':
        raise ValueError('Temporary arm output cannot enter the SDK model_* upload route')
    experiment = ScaledExperiment(repo, repo/SOURCE_CONFIG)
    identity, fingerprint = experiment.identity(), implementation_fingerprint(repo)
    source = torch.load(str(extra.source_checkpoint), map_location='cpu', weights_only=True)
    validate_source_state(source, identity)
    if args.task != experiment.cfg['experiment'] or state_fingerprint(source['model_state_dict']) != SOURCE_MODEL_SHA:
        raise ValueError('Wrong original110 task/model identity')
    head_blob = extra.head_artifact.read_bytes()
    head_sha = hashlib.sha256(head_blob).hexdigest()
    if head_sha != _sha(extra.head_sha256, 'head'):
        raise ValueError('Actual received head file SHA differs from the admitted file')
    artifact = torch.load(io.BytesIO(head_blob), map_location='cpu', weights_only=True)
    archive_audit = validate_head_artifact(artifact, extra.source_checkpoint)
    require_candidate_admission(artifact, 'formal')
    config = json.loads((repo/'configs/amp/head_smooth_v2.json').read_text(encoding='utf-8'))
    index_path = repo/config['source_exclusions']['path']
    exclusions, source_proof, source_manifest = registered_source_exclusions(repo, index_path, identity)
    cohorts, bindings = {}, [(extra.source_checkpoint, SOURCE_SHA), (extra.head_artifact, head_sha),
                            (index_path, config['source_exclusions']['sha256'])]
    for role, path in (('train', extra.train_cohort), ('validation', extra.validation_cohort),
                       ('sealed', extra.sealed_cohort)):
        sha = file_sha(path)
        manifest, arrays = _read_bundle(path, torch)
        cohorts[role] = dict(sha256=sha, manifest=manifest, arrays=arrays)
        bindings.append((path, sha))
    inputs = audit_sealed_inputs(cohorts, artifact=artifact, identity=identity,
        code_commit=commit, implementation_fingerprint=fingerprint, exclusions=exclusions,
        source_proof=source_proof, source_manifest=source_manifest)
    context = dict(contract=contract, experiment=experiment, commit=commit,
        fingerprint=fingerprint, source_state=source, artifact=artifact, archive_audit=archive_audit,
        candidate_sha=archive_audit['candidate_model_state_sha256'],
        source_manifest=source_manifest, source_proof=source_proof, inputs=inputs,
        sealed_arrays=cohorts['sealed']['arrays'], sealed_manifest=cohorts['sealed']['manifest'],
        file_bindings=bindings)
    del cohorts  # full prior arrays are not reused as replay/simulator inputs.
    _assert_input_files(context)
    if extra.arm is not None:
        return _collect_arm(extra, args, context)
    extra.output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='.head-sealed-arms-', dir=str(extra.output.parent)))
    actual = {}
    for arm in ARMS:
        path = directory/(arm+'-arm.pt')
        subprocess.check_call(arm_command(extra, remaining, arm, path), cwd=str(repo))
        manifest, arrays = _read_bundle(path, torch)
        actual[arm] = dict(manifest=manifest, arrays=arrays, sha256=file_sha(path))
        _assert_input_files(context)
    if (type(actual['source']['manifest'].get('process_id')) is not int
            or type(actual['candidate']['manifest'].get('process_id')) is not int):
        raise ValueError('Missing actual independently executed arm process evidence')
    manifest, arrays = assemble_sealed_pair(actual['source']['manifest'], actual['source']['arrays'],
        actual['candidate']['manifest'], actual['candidate']['arrays'], contract=contract,
        identity=identity, code_commit=commit, implementation_fingerprint=fingerprint,
        head_sha256=head_sha, candidate_model_sha256=context['candidate_sha'],
        inputs=inputs, sealed_arrays=context['sealed_arrays'], source_runtime_manifest=source_manifest,
        arm_file_shas={arm: actual[arm]['sha256'] for arm in ARMS})
    _write_bundle(extra.output, manifest, arrays, torch)
    print('[head-sealed-pair-complete] '+json.dumps(dict(type=SCHEMA, code_commit=commit,
        parent_checkpoint_sha256=SOURCE_SHA, head_artifact_sha256=head_sha,
        episodes={arm: actual[arm]['manifest']['episodes'] for arm in ARMS},
        effectiveness_verified=False, dr_unlocked=False)), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--mode', choices=('formal',), required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--head-artifact', type=Path, required=True)
    parser.add_argument('--head-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--sealed-cohort', type=Path, required=True)
    parser.add_argument('--train-cohort', type=Path, required=True)
    parser.add_argument('--validation-cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    extra, remaining = parser.parse_known_args()
    return _native_main(extra, remaining)


if __name__ == '__main__':
    main()
