"""Offline matched mu quality evidence, not training/cloud/acceptance certification.

Candidates contain measured ten-substep interval statistics, not raw1ms series.
The immutable077 original supplies raw env0 validation and the original110
archive supplies the identical endpoint trajectory. Never synthesize missing
substeps, smooth the robot, or treat a failed/missing tail as a quiet success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.jitter import SOURCE_SHA
from humanoid.amp.mu_temporal import (ACCELERATION_SCALES, BASELINE_BUNDLE_SHA,
    state_fingerprint, validate_mu, validate_mu_loss_report)
from humanoid.amp.recovery import foot_collision_vertices
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_runtime_timing
from tools.amp.analyze_jitter_events import WINDOWS, high_frequency
from tools.amp.audit_physics_diagnostic import (FORCE_THRESHOLD, common_prefix_samples,
    fast_geometry, fast_signals, inspect_prefixes, observed_initial_comparison,
    physics_summary, quality_summary, source_archive_comparison, validate_manifest,
    validate_telemetry, validate_runtime, worst_second)
from tools.amp.inspect_rollout import episode, load_bundle
from tools.amp.jitter_audit import validate_substep_arrays
from tools.amp.mu_audit import compare_environment

SOURCE_PHYSICS_BUNDLE_SHA = '96d56e4219ebd5106567e80cc09ce53ceb17091c1018dda9a514d16108bc2f47'
SPANS = dict(whole=(.01, 60.01), post2s=(2., 60.01), **WINDOWS)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def target_second_difference(data, manifest, start, end):
    """Applied-target curvature; every three-point stencil stays inside the span."""
    action = np.asarray(data['action'], dtype=np.float64)
    time = np.asarray(data['time'], dtype=np.float64)
    cfg = manifest['environment']
    if (action.ndim != 2 or action.shape[1] != 12 or len(action) != len(time) or
            not np.isfinite(action).all() or not np.isfinite(time).all() or
            cfg['control']['action_scale'] != .5 or
            not np.isclose(manifest['runtime']['control_dt'], .01, rtol=1e-6, atol=1e-10) or
            not np.isfinite([start, end]).all() or end <= start):
        raise ValueError('Invalid deterministic target/time/scale inputs')
    if len(time) and not np.allclose(time, (np.arange(len(time))+1)*.01, rtol=0, atol=1e-7):
        raise ValueError('Target samples are not one contiguous100Hz prefix')
    limit = cfg['normalization']['clip_actions']
    if not np.isfinite(limit) or limit <= 0 or np.any(np.abs(action) > limit):
        raise ValueError('Recorded action violates its actual clipping contract')
    boundaries = int((np.abs(action) >= limit).sum())
    inside = (time >= start-1e-8) & (time < end-1e-8)
    eligible = inside[2:] & inside[1:-1] & inside[:-2]
    values = .5*np.diff(action, n=2, axis=0)/.01**2
    expected_records = int(round((end-start)*100))
    requested = max(0, expected_records-2)
    return dict(time=time[2:], values=values, eligible=eligible, samples=int(eligible.sum()),
        requested_samples=requested,
        complete_window=int(inside.sum()) == expected_records and int(eligible.sum()) == requested,
        clip_boundary_count=boundaries, raw_mu_equals_recorded_action=boundaries == 0,
        raw_mu_available=boundaries == 0)


def paired_masks(left, right, left_arrays, right_arrays, mode, index, terminal_guard=100):
    n = common_prefix_samples(left, right, terminal_guard)
    physical = np.ones(n, dtype=bool)
    quiet = np.ones(n, dtype=bool)
    for arrays in (left_arrays, right_arrays):
        valid = arrays[mode+'_physics_valid'][:n, index]
        force = arrays[mode+'_physics_foot_force_peak'][:n, index]
        if valid.shape != (n,) or valid.dtype != np.bool_ or force.shape != (n, 2) or not np.isfinite(force).all():
            raise ValueError('Invalid shared physical interval/force data')
        physical &= valid
        quiet &= force.max(-1) <= FORCE_THRESHOLD
    return dict(samples=n, physical=physical, quiet=quiet & physical)


def rank_worst_initial_states(rows, metric='substep_accel_rms'):
    keys = [(row['mode'], row['env']) for row in rows]
    if (len(rows) != 32 or len(set(keys)) != 32 or
            set(keys) != {(mode, index) for mode in ('standing', 'reference') for index in range(16)}):
        raise ValueError('Worst-initial-state ranking must retain all32 distinct starts')
    def order(row):
        value = row['overview'].get(metric)
        if value is not None and not np.isfinite(value):
            raise ValueError('Nonfinite worst-initial-state metric')
        return (not row['failure'], value is not None, -value if value is not None else 0., row['mode'], row['env'])
    return sorted(rows, key=order)


def checkpoint_proof(path, manifest, completed):
    digest = file_sha(path)
    if digest != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint file differs from recorded policy')
    state = torch.load(path, weights_only=True, map_location='cpu')
    if (state['amp_identity'] != manifest['identity'] or state['completed_updates'] != completed or
            state['iter'] != completed-1):
        raise ValueError('Wrong actual policy identity or completed update')
    for key in ('model_state_dict', 'amp_discriminator_state_dict'):
        if not state[key] or any(not bool(torch.isfinite(v).all()) for v in state[key].values()):
            raise ValueError('Nonfinite saved inference/discriminator model')
    return dict(checkpoint_sha256=digest, model_state_sha256=state_fingerprint(state['model_state_dict']),
                completed_updates=completed), state.get('mu_loss_report')


def target_summary(data, manifest, start, end):
    result = target_second_difference(data, manifest, start, end)
    selected = result['values'][result['eligible']]
    summary = {key:result[key] for key in ('samples', 'requested_samples', 'complete_window',
        'clip_boundary_count', 'raw_mu_equals_recorded_action', 'raw_mu_available')}
    summary.update(requested_interval_s=[start, end], units='rad/s2 of PD target second difference, not joint acceleration')
    if not len(selected):
        summary.update(rms=None, per_joint_rms=None, abs_p95=None, normalized_mean_square=None)
    else:
        scale = np.asarray([ACCELERATION_SCALES[name] for name in manifest['dof_names']])
        summary.update(rms=float(np.sqrt(np.mean(selected**2))),
            per_joint_rms=np.sqrt(np.mean(selected**2, axis=0)).tolist(),
            abs_p95=np.percentile(np.abs(selected), 95, axis=0).tolist(),
            normalized_mean_square=float(np.mean((selected/scale)**2)))
    mask = (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
    summary['target_spectrum_100hz'] = high_frequency(.5*data['action'][mask], cutoff=10.)
    return summary


def section(data, case, mode, index, geometry, start, end, physical=None, quiet=None):
    if not len(data['time']) or end <= start:
        return None
    signals = fast_signals(data, case['manifest'], geometry)
    mask = (data['time'] >= start-1e-8) & (data['time'] < end-1e-8)
    return dict(quality=quality_summary(data, signals, case['manifest'], start, end),
        target=target_summary(data, case['manifest'], start, end),
        physics=physics_summary(case['arrays'], mode, index, data, signals, start, end, physical),
        shared_quiet_physics=physics_summary(case['arrays'], mode, index, data, signals, start, end, quiet)
            if quiet is not None else None,
        endpoint_torque_rms_nm=float(np.sqrt(np.mean(data['torque'][mask].astype(np.float64)**2))) if mask.any() else None)


def overview(value):
    quality = (value or {}).get('quality') or {}
    physics = (value or {}).get('physics') or {}
    target = (value or {}).get('target') or {}
    speed = quality.get('vx_mean')
    return dict(vx_mean_m_s=speed, speed_error_abs_m_s=None if speed is None else abs(speed-.45),
        contact_proxy_slip_mean_m_s=quality.get('contact_proxy_slip_mean_m_s'),
        heading_rms_deg=quality.get('heading_rms_to_world_x_deg'),
        target_second_difference_rms_rad_s2=target.get('rms'),
        endpoint_accel_rms_rad_s2=quality.get('acceleration_native_rms'),
        substep_accel_rms=physics.get('substep_accel_rms'),
        substep_torque_delta_rms_nm=physics.get('substep_torque_delta_rms_nm'),
        endpoint_torque_rms_nm=(value or {}).get('endpoint_torque_rms_nm'),
        native_substep_foot_force_peak_n=physics.get('force_peak_n'))


def case_rows(case, geometry):
    rows = []
    for prefix in inspect_prefixes(case['manifest'], case['arrays']):
        mode, index = prefix['mode'], prefix['env']
        data = episode(case['arrays'], mode, index)
        spans = {name:section(data, case, mode, index, geometry, *span) for name, span in SPANS.items()}
        terminal = None
        if prefix['failure'] and len(data['time']):
            end = float(data['time'][-1])+.01
            terminal = section(data, case, mode, index, geometry, max(.01, end-1.), end)
        worst = {}
        if len(data['time']):
            mask = case['arrays'][mode+'_physics_valid'][:len(data['time']), index]
            energies = dict(physics_acceleration=case['arrays'][mode+'_physics_accel_squared'][:len(data['time']), index].mean(-1),
                physics_torque_delta=case['arrays'][mode+'_physics_torque_delta_squared'][:len(data['time']), index].mean(-1))
            worst = {key:worst_second(data['time'], energy, mask, minimum_time=.01) for key, energy in energies.items()}
        rows.append(dict(prefix, overview=overview(spans['post2s']), spans=spans,
            terminal_second=terminal, worst_one_second_including_terminal=worst))
    return rows


def paired_rows(left_case, right_case, geometry):
    rows = []
    for mode in ('standing', 'reference'):
        for index in range(16):
            left = episode(left_case['arrays'], mode, index)
            right = episode(right_case['arrays'], mode, index)
            masks = paired_masks(left, right, left_case['arrays'], right_case['arrays'], mode, index)
            n = masks['samples']
            end = float(left['time'][n-1])+.01 if n else 0.
            cases = []
            spans = dict(SPANS, whole=(.01, end), post2s=(2., end))
            for data, case in ((left, left_case), (right, right_case)):
                clipped = {key:value[:n] if key != 'initial' else value for key,value in data.items()}
                values = {name:section(clipped, case, mode, index, geometry, *span,
                    physical=masks['physical'], quiet=masks['quiet']) for name,span in spans.items()}
                cases.append(dict(overview=overview(values['post2s']), spans=values))
            rows.append(dict(mode=mode, env=index, common_samples=n, common_end_s=end-.01 if n else 0.,
                terminal_guard_s=1., left_failure=bool(left['initial']['failure'] or left['failure'].any()),
                right_failure=bool(right['initial']['failure'] or right['failure'].any()),
                shared_physics_samples=int(masks['physical'].sum()), shared_quiet_samples=int(masks['quiet'].sum()),
                left=cases[0], right=cases[1]))
    return rows


def means(rows, section_key='overview'):
    result = {}
    for key in overview(None):
        values = [row[section_key][key] for row in rows if row[section_key][key] is not None]
        result[key] = dict(mean=float(np.mean(values)) if values else None, contributing_initial_states=len(values),
                           minimum=min(values) if values else None, maximum=max(values) if values else None)
    return result


def paired_means(rows):
    """Use the same explicit initial-state cohort on both sides of each metric."""
    keys = [(row['mode'], row['env']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate paired initial state')
    result = dict(left={}, right={})
    for metric in overview(None):
        shared = [row for row in rows if all(row[side]['overview'][metric] is not None
                                            for side in ('left', 'right'))]
        cohort = [dict(mode=row['mode'], env=row['env']) for row in shared]
        for side in ('left', 'right'):
            values = [row[side]['overview'][metric] for row in shared]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite paired metric')
            result[side][metric] = dict(mean=float(np.mean(values)) if values else None,
                contributing_initial_states=len(values), common_initial_states=cohort,
                minimum=min(values) if values else None, maximum=max(values) if values else None)
    return result


def validate_geometry_binding(manifest, source_manifest, body_names, urdf_sha):
    """Bind mesh-derived foot signals to the original body's order and URDF."""
    for field in ('foot_names', 'body_names', 'urdf_lf_sha256'):
        if field not in manifest or field not in source_manifest or manifest[field] != source_manifest[field]:
            raise ValueError('Changed or missing candidate geometry binding: '+field)
    if (manifest['body_names'] != list(body_names) or
            not isinstance(urdf_sha, str) or not re.fullmatch('[0-9a-f]{64}', urdf_sha) or
            manifest['urdf_lf_sha256'] != urdf_sha):
        raise ValueError('Geometry does not match the actual experiment URDF/body order')
    feet = manifest['foot_names']
    if not isinstance(feet, list) or len(feet) != 2 or len(set(feet)) != 2 or not set(feet).issubset(body_names):
        raise ValueError('Invalid two-foot geometry binding')


def plot_cases(cases, geometry, output, mode, index, start, end, label):
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    fig, axes = plt.subplots(7, 3, figsize=(17, 14), sharex=True, sharey='row')
    for col, (name, case) in enumerate(cases.items()):
        data = episode(case['arrays'], mode, index)
        t = data['time']; n = len(t)
        axes[0,col].set_title(name+' | '+mode+' env'+str(index)+' | '+('failed prefix' if
            data['initial']['failure'] or data['failure'].any() else 'survived60s'))
        if not n: continue
        s = fast_signals(data, case['manifest'], geometry)
        mask = (t >= start-1e-8) & (t < end-1e-8)
        axes[0,col].plot(t[mask], data['base_lin_vel'][mask,0]); axes[0,col].axhline(.45,c='gray',ls='--')
        contact = (data['foot_force'][:,:,2] > 5.) & (s['sole_height'] < .02)
        for foot in range(2):
            axes[1,col].plot(t[mask], np.where(contact[:,foot],s['sole_speed'][:,foot],np.nan)[mask],lw=.7)
        yaw = Rotation.from_quat(data['root_state'][:,3:7]).as_euler('xyz')[:,2]
        axes[2,col].plot(t[mask],np.rad2deg(np.arctan2(np.sin(yaw),np.cos(yaw)))[mask])
        target = target_second_difference(data,case['manifest'],max(.01,start),end)
        axes[3,col].plot(target['time'][target['eligible']],np.sqrt((target['values'][target['eligible']]**2).mean(-1)))
        valid = mask & case['arrays'][mode+'_physics_valid'][:n,index]
        axes[4,col].plot(t[valid],np.sqrt(case['arrays'][mode+'_physics_accel_squared'][:n,index][valid].mean(-1)),label='native1kHz interval RMS',lw=.7)
        axes[4,col].plot(t[mask],np.sqrt((s['acceleration_native'][mask]**2).mean(-1)),label='100Hz endpoint',alpha=.5,lw=.7)
        axes[4,col].legend(fontsize=6)
        axes[5,col].plot(t[valid],np.sqrt(case['arrays'][mode+'_physics_torque_delta_squared'][:n,index][valid].mean(-1)),lw=.7)
        for foot in range(2):
            axes[6,col].plot(t[valid],case['arrays'][mode+'_physics_foot_force_peak'][:n,index,foot][valid],lw=.7)
    labels = ('Body vx m/s','Contact-proxy sole speed m/s','World-X heading deg',
        'PD target second diff rad/s2','Joint acceleration rad/s2','Native1ms torque increment RMS Nm','Net foot Fz interval peak N')
    for row, text in enumerate(labels): axes[row,0].set_ylabel(text)
    for axis in axes.flat: axis.set_xlim(start,end); axis.grid(alpha=.2)
    axes[-1,1].set_xlabel('Original capture-label physical time s; missing/failed tails are not filled')
    fig.suptitle('Recorded states only; no robot smoothing. Native1kHz interval aggregates, not candidate raw1ms waveforms.')
    fig.tight_layout(); fig.savefig(output/(label+'.png'),dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-endpoint','source-physics','source-checkpoint','anchor-bundle',
                 'anchor-checkpoint','temporal-bundle','temporal-checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if not re.fullmatch('[0-9a-f]{40}',a.expected_commit): raise ValueError('Expected full40hex candidate commit')
    torch.set_num_threads(2)
    if file_sha(a.source_endpoint) != BASELINE_BUNDLE_SHA or file_sha(a.source_physics) != SOURCE_PHYSICS_BUNDLE_SHA or file_sha(a.source_checkpoint) != SOURCE_SHA:
        raise ValueError('Immutable original110/077 source SHA mismatch')
    source_experiment = ScaledExperiment(ROOT,ROOT/'configs/amp/lafan_walk02_sustain_control.json')
    sm,sa = load_bundle(a.source_physics)
    validate_manifest(sm,'original',source_experiment.identity())
    source_runtime = validate_runtime(sm)
    source_raw = validate_telemetry(sm,sa)
    validate_substep_arrays(sm,sa)
    source = dict(manifest=sm,arrays=sa)
    archive = source_archive_comparison(source,a.source_endpoint,source_experiment.identity())
    if not archive['all_trajectory_fields_exact_equal']: raise ValueError('077 original is not the exact original110 shared trajectory')
    om,oa = load_bundle(a.source_endpoint)
    validate_geometry_binding(sm,om,source_experiment.spec.body_names,source_experiment.kinematics.lf_sha256)
    source_checkpoint,_ = checkpoint_proof(a.source_checkpoint,om,2500)
    geometry = fast_geometry(foot_collision_vertices(source_experiment.kinematics.path))
    cases = dict(original=source)
    provenance = dict(original=dict(bundle_path=str(a.source_physics.resolve()),bundle_sha256=SOURCE_PHYSICS_BUNDLE_SHA,
        endpoint_bundle_path=str(a.source_endpoint.resolve()),endpoint_bundle_sha256=BASELINE_BUNDLE_SHA,
        checkpoint=source_checkpoint,code_commit=sm['code_commit'],runtime_readback=source_runtime,env0_raw_verification=source_raw))
    for group in ('anchor','temporal'):
        path = getattr(a,group+'_bundle'); checkpoint = getattr(a,group+'_checkpoint')
        experiment = ScaledExperiment(ROOT,ROOT/('configs/amp/lafan_walk02_mu_'+group+'.json'))
        validate_mu(experiment)
        manifest,arrays = load_bundle(path)
        if (manifest['identity'] != experiment.identity() or manifest['code_commit'] != a.expected_commit or
                (manifest['duration_s'],manifest['num_envs'],manifest['fps']) != (60,16,100) or
                manifest['modes'] != ['standing','reference'] or manifest['mode_seeds'] != {'standing':5,'reference':105} or
                manifest['evaluation_protocol'] != 'fixed60_independent_mode_seeds' or manifest['policy_deterministic'] is not True or
                manifest['effectiveness_verified'] is not False or manifest['dr_unlocked'] is not False):
            raise ValueError('Wrong candidate fixed60 policy/config/commit protocol')
        if manifest['dof_names'] != list(experiment.spec.joint_names): raise ValueError('Changed candidate joint order')
        validate_geometry_binding(manifest,om,experiment.spec.body_names,experiment.kinematics.lf_sha256)
        telemetry = manifest.get('substep_telemetry',{})
        if (telemetry.get('physics_hz'),telemetry.get('control_hz'),telemetry.get('samples_per_control')) != (1000,100,10) or telemetry.get('reward_used') is not False:
            raise ValueError('Candidate lacks measured read-only native1kHz interval statistics')
        validate_substep_arrays(manifest,arrays)
        compare_environment(om['environment'],manifest['environment'],group)
        if manifest['runtime'] != om['runtime']: raise ValueError('Candidate runtime PD/DOF/timing differs from original110')
        validate_runtime_timing(manifest['runtime']['control_dt'],manifest['runtime']['physics_dt'],manifest['environment']['control']['decimation'])
        initial = observed_initial_comparison(oa,arrays)
        if not initial['all_captured_fields_exact_equal']: raise ValueError('Candidate actual post-warmup initial states changed')
        model,loss = checkpoint_proof(checkpoint,manifest,2750)
        validate_mu_loss_report(loss,experiment.cfg['mu_temporal'],250)
        cases[group] = dict(manifest=manifest,arrays=arrays)
        provenance[group] = dict(bundle_path=str(path.resolve()),bundle_sha256=file_sha(path),checkpoint=model,
            code_commit=manifest['code_commit'],actual_post_warmup_initial=initial,
            raw1ms_available=False,raw1ms_waveform_and_spectrum='not evaluable; only ten-substep interval aggregates captured',
            full_shape_and_solver_readback_available=False)
    rows = {name:case_rows(case,geometry) for name,case in cases.items()}
    rankings = {name:{key:[dict(mode=row['mode'],env=row['env'],failure=row['failure'],observed_s=row['observed_s'],overview=row['overview'])
        for row in rank_worst_initial_states(records,key)] for key in ('substep_accel_rms','speed_error_abs_m_s',
            'contact_proxy_slip_mean_m_s','heading_rms_deg','target_second_difference_rms_rad_s2')} for name,records in rows.items()}
    pairs = {}
    for left,right in (('original','anchor'),('original','temporal'),('anchor','temporal')):
        records = paired_rows(cases[left],cases[right],geometry)
        pairs[left+'_vs_'+right] = dict(rows=records,equal_initial_condition_means=paired_means(records))
    report = dict(schema_version=2,script_sha256=file_sha(__file__),expected_candidate_commit=a.expected_commit,
        evidence=provenance,original110_to_077_exact_shared_trajectory=archive,
        cases={name:dict(rows=records,survived=sum(r['survived'] for r in records),failed=sum(r['failure'] for r in records),
                        equal_initial_condition_means=means(records)) for name,records in rows.items()},
        paired=pairs,worst_initial_states=rankings,
        definitions=dict(control_hz=100,physics_hz=1000,physics_samples_per_control=10,
            windows=SPANS,intervals='half-open[start,end); whole captures0.01..60.00s; target stencils require all three records inside span',
            physical_pairing='common contiguous observed prefix; independently subtract100 ticks from every failed episode before intersecting; shared physics_valid and shared<=1100N quiet cohorts',
            aggregation='equal weight per contributing initial condition; each paired metric uses the same explicit bilateral non-null cohort; missing remains null, never zero',
            target_second_difference='0.5*delta2(recorded deterministic clipped action)/0.01^2; raw_mu only when no clip boundary; fixed original per-joint scales, never refit',
            command_spectrum='100Hz PD target only; Welch, >=10Hz high band, <=50Hz Nyquist. Not native1kHz or legacy Hann>10Hz spectrum',
            native_torque='sqrt(mean(native1ms torque-command increment squared)) in Nm, not torque derivative or raw1ms torque RMS',
            foot_slip='>5N endpoint foot Fz and source mesh min-z<0.02m contact proxy; sole-centroid XY speed, not identified ground contact/friction'),
        effectiveness_verified=False,hardware_effectiveness_verified=False,dr_unlocked=False,cloud_execution_verified=False,
        limitations=['Formal training task/log/teacher/end-state audit remains separate; this is supplied-artifact quality evidence.',
            'Candidates have no raw1ms waveform,1kHz spectrum, substep burst-energy fraction or complete simulator/shape readback.',
            'Raw env0 077 validation cannot be transferred to uncaptured candidate substeps; interval measurement/Jensen checks are distinct.',
            'All32 failures and missing windows retained; failed terminal seconds are reported separately, not counted as quiet success.',
            'Observed post-warmup equality does not certify hidden solver states or uncaptured pre-physics history.',
            'Target second difference is a PD command quantity, not physical joint acceleration; capture-label action stencils lag decision time.',
            'Same physical ticks need not imply same gait phase; net forces do not identify contact pairs; one seed/clip/nominal model only.'])
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'report.json').open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    import matplotlib
    matplotlib.use('Agg')
    for name,span in dict(whole=(.01,60.01),**WINDOWS).items():
        plot_cases(cases,geometry,a.output,'reference',0,*span,'reference_env0_'+name)
    selected = set()
    for name,ranking in rankings.items():
        row = ranking['substep_accel_rms'][0]
        selected.add((row['mode'],row['env']))
    selected.update((row['mode'],row['env']) for records in rows.values() for row in records if row['failure'])
    for mode,index in sorted(selected):
        plot_cases(cases,geometry,a.output,mode,index,.01,60.01,mode+'_env%d_worst_or_failure'%index)
    print(json.dumps(dict(output=str(a.output.resolve()),survival={name:sum(r['survived'] for r in records) for name,records in rows.items()},
        candidate_raw1ms_available=False,effectiveness_verified=False,dr_unlocked=False)))


if __name__ == '__main__': main()
