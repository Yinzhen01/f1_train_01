"""Rebuild a fixed-waist 12-DOF reference with explicit chest-tilt supervision.

The old reference defines world foot poses, contact labels, timing and heading.
The original 29-DOF chest (not pelvis) defines walking tilt. No dynamics claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from prepare_gmr_reference import JOINT_NAMES, fk, sha, stl_vertices, origin


def smoothstep5(value):
    x = np.clip(value, 0., 1.)
    return x ** 3 * (10. + x * (-15. + 6. * x))


def walking_weight(times):
    # Quiet standing: 0..1.0 and 4.2..4.6 s. C2 transitions, no clock rescaling.
    return smoothstep5((times - 1.) / .5) * (1. - smoothstep5((times - 3.7) / .5))


def make_targets(source_model, source_qpos, baseline_quat, fixed_chest_rotation, fps, tilt_scale=.6):
    data = mujoco.MjData(source_model)
    body = source_model.body('lumbar_pitch_link').id
    rotations = []
    for pose in source_qpos:
        data.qpos[:] = pose
        mujoco.mj_forward(source_model, data)
        rotations.append(data.xmat[body].reshape(3, 3) @ fixed_chest_rotation.T)
    chest_euler = Rotation.from_matrix(rotations).as_euler('xyz')
    smoothed_tilt = savgol_filter(chest_euler[:, :2], 9, 3, axis=0, mode='interp')
    times = np.arange(len(source_qpos)) / fps
    weight = walking_weight(times)
    heading = Rotation.from_quat(baseline_quat).as_euler('xyz')[:, 2]
    desired = np.column_stack((smoothed_tilt * weight[:, None] * tilt_scale, heading))
    return Rotation.from_euler('xyz', desired), chest_euler, weight


def reconstruct(model, old, rotations, lower, upper, speed_limits, fps, root_hint=None):
    data = mujoco.MjData(model)
    feet = [model.body(name).id for name in old['foot_names']]
    addresses = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    original_R = Rotation.from_quat(old['root_quat']).as_matrix()
    target_p, target_R = [], []
    for i, joints in enumerate(old['dof_pos']):
        data.qpos[addresses] = joints
        mujoco.mj_forward(model, data)
        target_p.append(data.xpos[feet] @ original_R[i].T + old['root_pos'][i])
        target_R.append(original_R[i] @ data.xmat[feet].reshape(2, 3, 3))
    target_p, target_R = np.asarray(target_p), np.asarray(target_R)
    positions, joints_out, fitted_rotations, errors_p, errors_r = [], [], [], [], []
    previous = None
    for i, angles in enumerate(rotations.as_euler('xyz')):
        q_ref = old['dof_pos'][i]
        centre = old['root_pos'][i]
        translation_hint = np.zeros(3) if root_hint is None else root_hint[i] - centre
        translation_weight = 5. if root_hint is None else 40.
        temporal = np.r_[np.zeros(3), q_ref, np.zeros(2)] if previous is None else previous + np.r_[np.zeros(3), q_ref - old['dof_pos'][i-1], np.zeros(2)]

        def orientation(x):
            return Rotation.from_euler('xyz', angles + np.r_[x[15:], 0.]).as_matrix()

        def residual(x):
            R = orientation(x)
            data.qpos[addresses] = x[3:15]
            mujoco.mj_forward(model, data)
            ep = data.xpos[feet] @ R.T + centre + x[:3] - target_p[i]
            er = Rotation.from_matrix(target_R[i].transpose(0, 2, 1) @ R @ data.xmat[feet].reshape(2, 3, 3)).as_rotvec()
            return np.r_[1000. * ep.ravel(), 300. * er.ravel(),
                         translation_weight * (x[:3] - translation_hint), .05 * (x[3:15] - q_ref), 10. * x[15:],
                         .1 * (x[:15] - temporal[:15]), 5. * (x[15:] - temporal[15:])]

        # Permit at most one degree of walking tilt adjustment to satisfy feet
        # and hard limits; quiet-standing orientation remains fixed upright.
        delta = 1e-6 if walking_weight(np.array([i / fps]))[0] == 0 else np.deg2rad(1.)
        lo = np.r_[[-.10, -.10, -.06], lower, [-delta, -delta]]
        hi = np.r_[[.10, .10, .06], upper, [delta, delta]]
        if previous is not None:
            # Enforce actual interpolation slope, not only central-difference labels.
            lo[3:15] = np.maximum(lo[3:15], previous[3:15] - .98 * speed_limits / fps)
            hi[3:15] = np.minimum(hi[3:15], previous[3:15] + .98 * speed_limits / fps)
        initial = np.clip(temporal, lo + 1e-8, hi - 1e-8)
        fit = least_squares(residual, initial, bounds=(lo, hi), max_nfev=200,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if not fit.success:
            raise ValueError('IK failed at frame %d: %s' % (i, fit.message))
        value = residual(fit.x)
        errors_p.append(np.linalg.norm(value[:6].reshape(2, 3) / 1000., axis=-1))
        errors_r.append(np.linalg.norm(value[6:12].reshape(2, 3) / 300., axis=-1))
        positions.append(centre + fit.x[:3])
        joints_out.append(fit.x[3:15])
        fitted_rotations.append(orientation(fit.x))
        previous = fit.x.copy()
    fitted_rotations = Rotation.from_matrix(fitted_rotations)
    return np.asarray(positions), np.asarray(joints_out), fitted_rotations, target_p, target_R, {
        'max_ankle_position_change_m': float(np.max(errors_p)),
        'max_ankle_rotation_change_rad': float(np.max(errors_r)),
        'max_root_translation_change_m': float(np.linalg.norm(np.asarray(positions) - old['root_pos'], axis=1).max()),
        'max_target_rotation_adjustment_deg': float(np.rad2deg((rotations.inv() * fitted_rotations).magnitude()).max()),
        'method': 'bounded root translation, walking roll/pitch adjustment <=1 degree per axis, and 12 joints with interframe hard speed bounds; preserve old world ankles',
    }


def derive_fields(model, root_pos, root_quat, dof_pos, sole_local, normal_local, fps):
    data = mujoco.MjData(model)
    feet = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    addresses = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    foot_p, foot_q = [], []
    for q in dof_pos:
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        R = data.xmat[feet].reshape(2, 3, 3)
        foot_p.append(data.xpos[feet] + np.einsum('fij,fj->fi', R, sole_local))
        foot_q.append(Rotation.from_matrix(R).as_quat())
    rot = Rotation.from_quat(root_quat)
    angular_steps = (rot[1:] * rot[:-1].inv()).as_rotvec() * fps
    omega = np.vstack([angular_steps[0], (angular_steps[:-1] + angular_steps[1:]) / 2., angular_steps[-1]])
    return dict(root_pos=root_pos, root_quat=root_quat, dof_pos=dof_pos,
                dof_vel=np.gradient(dof_pos, 1. / fps, axis=0),
                root_vel=np.gradient(root_pos, 1. / fps, axis=0), root_ang_vel=omega,
                foot_pos_base=np.asarray(foot_p), foot_quat_base=np.asarray(foot_q))


def dense_audit(model, urdf, old, new, target_p, target_R, fps):
    data = mujoco.MjData(model)
    feet = [model.body(name).id for name in old['foot_names']]
    addresses = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    robot = ET.parse(urdf).getroot()
    vertices = []
    for name in old['foot_names']:
        shape = robot.find("link[@name='%s']/collision" % name)
        mesh = shape.find('geometry/mesh')
        T = origin(shape.find('origin'))
        v = stl_vertices((urdf.parent / mesh.get('filename')).resolve())
        v *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        vertices.append(v @ T[:3, :3].T + T[:3, 3])
    samples = np.linspace(0., (len(new['dof_pos']) - 1) / fps, 461)
    min_z, tilt, sole_error = [], [], []
    for time in samples:
        x = min(time * fps, len(new['dof_pos']) - 1)
        lo, hi = int(x), min(int(x) + 1, len(new['dof_pos']) - 1)
        alpha = x - lo
        def sample(array):
            return array[lo] * (1-alpha) + array[hi] * alpha
        quat = sample(new['root_quat'])
        R = Rotation.from_quat(quat).as_matrix()
        pos = sample(new['root_pos'])
        data.qpos[addresses] = sample(new['dof_pos'])
        mujoco.mj_forward(model, data)
        foot_R = R @ data.xmat[feet].reshape(2, 3, 3)
        foot_p = data.xpos[feet] @ R.T + pos
        min_z.append([float((v @ foot_R[f].T + foot_p[f])[:, 2].min()) for f, v in enumerate(vertices)])
        normals = np.einsum('fij,fj->fi', foot_R, old['sole_normal_local'])
        tilt.append(np.rad2deg(np.arccos(np.clip(normals[:, 2], -1, 1))))
        # Compare dense FK against old dense FK, not interpolation of body poses.
        q_old = sample(old['root_quat'])
        R_old = Rotation.from_quat(q_old).as_matrix()
        data.qpos[addresses] = sample(old['dof_pos'])
        mujoco.mj_forward(model, data)
        old_R = R_old @ data.xmat[feet].reshape(2, 3, 3)
        old_p = data.xpos[feet] @ R_old.T + sample(old['root_pos'])
        new_sole = foot_p + np.einsum('fij,fj->fi', foot_R, old['sole_local'])
        old_sole = old_p + np.einsum('fij,fj->fi', old_R, old['sole_local'])
        sole_error.append(np.linalg.norm(new_sole-old_sole, axis=-1))
    standing = (samples <= 1. + 1e-8) | (samples >= 4.2 - 1e-8)
    return dict(samples=461, minimum_foot_mesh_z_m=float(np.min(min_z)),
                standing_max_sole_tilt_deg=float(np.max(np.asarray(tilt)[standing])),
                max_dense_sole_position_change_m=float(np.max(sole_error)),
                max_joint_speed_rad_s=float(np.abs(new['dof_vel']).max()),
                max_segment_joint_speed_rad_s=float(np.abs(np.diff(new['dof_pos'], axis=0) * fps).max()),
                max_joint_acceleration_rad_s2=float(np.abs(np.gradient(new['dof_vel'], 1./fps, axis=0)).max()))


def prepare(baseline, source, urdf, source_xml, output, tilt_scale=.6):
    if not 0. < tilt_scale <= 1.:
        raise ValueError('Walking tilt scale must be in (0, 1]')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Refusing to overwrite an existing reference')
    with np.load(baseline, allow_pickle=False) as archive:
        old = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(old['metadata_json']))
    if old['dof_pos'].shape != (139, 12) or float(old['fps']) != 30. or metadata.get('cyclic') is not False:
        raise ValueError('This standing-phase schedule is specific to the 139-frame 30Hz KIT317 clip')
    for key in ('root_pos', 'root_quat', 'dof_pos', 'lower', 'upper', 'sole_local', 'sole_normal_local'):
        if not np.isfinite(old[key]).all():
            raise ValueError('Nonfinite baseline field: ' + key)
    if metadata['source_sha256'] != sha(source):
        raise ValueError('Original GMR source differs from baseline provenance')
    if metadata['gmr_xml_sha256'] != sha(source_xml):
        raise ValueError('Original GMR model differs from baseline provenance')
    if metadata['training_urdf_lf_sha256'] != hashlib.sha256(urdf.read_bytes().replace(b'\r\n', b'\n')).hexdigest():
        raise ValueError('Training URDF differs from baseline')
    if old['joint_names'].astype(str).tolist() != list(JOINT_NAMES):
        raise ValueError('Unexpected baseline joint order')
    with np.load(source, allow_pickle=False) as archive:
        source_qpos = archive['qpos'].copy()
        source_names = archive['joint_names'].astype(str).tolist()
        if float(archive['fps']) != float(old['fps']) or len(source_qpos) != len(old['dof_pos']):
            raise ValueError('Source/baseline timelines differ')
    source_model = mujoco.MjModel.from_xml_path(str(source_xml))
    if source_model.nq != source_qpos.shape[1]:
        raise ValueError('Source model width mismatch')
    for i, name in enumerate(source_names):
        if int(source_model.joint(name).qposadr[0]) != 7 + i:
            raise ValueError('Source model joint order mismatch')
    training = mujoco.MjModel.from_xml_path(str(urdf))
    robot = ET.parse(urdf).getroot()
    joints = robot.findall('joint')
    for name, expected in metadata['foot_mesh_sha256'].items():
        mesh = robot.find("link[@name='%s']/collision/geometry/mesh" % name)
        if sha((urdf.parent / mesh.get('filename')).resolve()) != expected:
            raise ValueError('Foot mesh differs from baseline: ' + name)
    if {j.get('name') for j in joints if j.get('type') != 'fixed'} != set(JOINT_NAMES):
        raise ValueError('Expected a fixed-waist, 12-leg-joint training URDF')
    zero_chest = fk(joints, {})['lumbar_pitch_link'][:3, :3]
    old['foot_names'] = np.asarray(metadata['foot_names'])
    fps = float(old['fps'])
    hard_speed = np.asarray([float(next(j for j in joints if j.get('name') == name).find('limit').get('velocity')) for name in JOINT_NAMES])
    rotations, chest_euler, weight = make_targets(source_model, source_qpos, old['root_quat'], zero_chest, fps, tilt_scale)
    target_rotations = rotations
    root_pos, dof_pos, rotations, feet_p, feet_R, fit_report = reconstruct(training, old, target_rotations, old['lower'], old['upper'], hard_speed, fps)
    for _ in range(3):
        # Smooth the IK correction across the complete clip, then solve again
        # under the same foot, joint, velocity and upright constraints. Never
        # filter the final joints/poses without rechecking their foot geometry.
        correction = savgol_filter(root_pos - old['root_pos'], 21, 3, axis=0, mode='interp')
        hint = old['root_pos'] + correction
        root_pos, dof_pos, rotations, feet_p, feet_R, fit_report = reconstruct(training, old, target_rotations, old['lower'], old['upper'], hard_speed, fps, hint)
    root_quat = rotations.as_quat()
    # Preserve quaternion hemisphere continuity for the runtime's linear sampler.
    for i in range(1, len(root_quat)):
        if np.dot(root_quat[i], root_quat[i-1]) < 0:
            root_quat[i] *= -1
    new = derive_fields(training, root_pos, root_quat, dof_pos, old['sole_local'], old['sole_normal_local'], fps)
    audit = dense_audit(training, urdf, old, new, feet_p, feet_R, fps)
    clearance_lift = max(0., .0005 - audit['minimum_foot_mesh_z_m'])
    if clearance_lift > .001:
        raise ValueError('Dense clearance repair would exceed 1 mm')
    if clearance_lift:
        new['root_pos'][:, 2] += clearance_lift
        # A uniform lift preserves velocity, acceleration and relative foot poses.
        audit = dense_audit(training, urdf, old, new, feet_p, feet_R, fps)
        fit_report['max_ankle_position_change_m'] += clearance_lift  # conservative bound
    fit_report['max_root_translation_change_m'] = float(np.linalg.norm(new['root_pos'] - old['root_pos'], axis=1).max())
    audit['max_root_speed_m_s'] = float(np.linalg.norm(new['root_vel'], axis=1).max())
    audit['max_root_acceleration_m_s2'] = float(np.linalg.norm(np.gradient(new['root_vel'], 1./fps, axis=0), axis=1).max())
    errors = []
    if fit_report['max_ankle_position_change_m'] > .001: errors.append('ankle position > 1 mm')
    if fit_report['max_ankle_rotation_change_rad'] > .002: errors.append('ankle rotation > .002 rad')
    if audit['minimum_foot_mesh_z_m'] < 0.: errors.append('dense mesh penetrates ground')
    if audit['standing_max_sole_tilt_deg'] > .1: errors.append('standing feet tilt > .1 degree')
    if audit['max_dense_sole_position_change_m'] > .002: errors.append('dense sole changes > 2 mm')
    if np.any(np.abs(new['dof_vel']) > hard_speed + 1e-6): errors.append('joint velocity label exceeds hard limit')
    if np.any(np.abs(np.diff(dof_pos, axis=0)*fps) > hard_speed + 1e-6): errors.append('interpolation slope exceeds hard speed limit')
    if audit['max_root_acceleration_m_s2'] > 12.: errors.append('root acceleration > 12 m/s2 kinematic gate')
    if errors:
        raise ValueError(json.dumps(dict(failed=errors, fit=fit_report, audit=audit), indent=2))
    identity = {key: value for key, value in metadata.items() if key in (
        'source_sha256', 'training_urdf_lf_sha256', 'source_name', 'frames', 'fps',
        'duration_s', 'cyclic', 'joint_names', 'foot_names', 'quaternion_order', 'units', 'foot_mesh_sha256')}
    report = dict(identity, baseline_sha256=sha(baseline), source_model_sha256=sha(source_xml),
                  reference_variant='upright_chest_v1',
                  walking_tilt_scale=tilt_scale,
                  reconstruction_settings=dict(root_smoothing_window=21, root_smoothing_polyorder=3,
                                               root_smoothing_passes=3, root_hint_weight=40.,
                                               foot_position_weight=1000., foot_rotation_weight=300.,
                                               walking_tilt_adjustment_per_axis_deg=1.,
                                               speed_limit_fraction=.98),
                  target_definition='Quiet standing roll/pitch=0; walking scaled smoothed 29DOF chest tilt; original pelvis yaw. C2 transitions 1.0..1.5 and 3.7..4.2 s.',
                  limit_fit=fit_report, **audit)
    report['contact_frames'] = old['foot_contact'].sum(0).tolist()
    report['extra_clearance_lift_max_m'] = clearance_lift
    report['source_chest_aligned_rpy_mean_deg'] = np.rad2deg(chest_euler).mean(0).tolist()
    report['new_root_tilt_mean_deg'] = rotations.as_euler('xyz', degrees=True)[:, :2].mean(0).tolist()
    report['max_root_pitch_deg'] = float(rotations.as_euler('xyz', degrees=True)[:, 1].max())
    report['boundary'] = 'Kinematic reconstruction only; changed reference and reward require new Isaac Gym smoke and training. No policy or dynamics validation.'
    for key in ('fps', 'joint_names', 'lower', 'upper', 'sole_local', 'sole_normal_local', 'foot_contact'):
        new[key] = old[key]
    new['walking_tilt_weight'] = weight
    new['metadata_json'] = json.dumps(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **new)
    report['output_sha256'] = sha(output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'source', 'urdf', 'source-xml', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--walking-tilt-scale', type=float, default=.6)
    args = parser.parse_args()
    prepare(args.baseline, args.source, args.urdf, args.source_xml, args.output, args.walking_tilt_scale)
