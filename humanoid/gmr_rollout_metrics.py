"""Metrics for recorded finite-clip inference, without importing Isaac Gym."""
import numpy as np


def rotation_error(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def gravity_in_body(quaternion_xyzw):
    """World unit gravity expressed in body coordinates, same as tilt reward."""
    q = np.asarray(quaternion_xyzw, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack((2 * (w * y - x * z), -2 * (y * z + w * x),
                     2 * (x * x + y * y) - 1), axis=-1)


def trunk_metrics(actual_quat, target_quat):
    actual, target = gravity_in_body(actual_quat), gravity_in_body(target_quat)
    # Use atan2, not acos alone, for numerical stability at very small errors.
    error = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(actual, target), axis=-1),
                                 np.sum(actual * target, axis=-1)))
    pitch = np.rad2deg(np.arcsin(np.clip(actual[..., 0], -1, 1)))
    target_pitch = np.rad2deg(np.arcsin(np.clip(target[..., 0], -1, 1)))
    return dict(trunk_tilt_error_mean_deg=float(error.mean()),
                trunk_tilt_error_p95_deg=float(np.percentile(error, 95)),
                trunk_tilt_error_max_deg=float(error.max()),
                actual_pitch_mean_deg=float(pitch.mean()),
                actual_pitch_min_deg=float(pitch.min()),
                actual_pitch_max_deg=float(pitch.max()),
                target_pitch_mean_deg=float(target_pitch.mean()))


def summarize_rollout(data, duration):
    valid = np.asarray(data['foot_diagnostics_valid'], dtype=bool)
    contact = np.asarray(data['foot_force'])[valid, :, 2] > 5.
    speed = np.linalg.norm(np.asarray(data['sole_velocity'])[valid, :, :2], axis=-1)
    times = np.asarray(data['time'])
    physical_failure = np.asarray(data['physical_failure'], dtype=bool)
    q_error = np.asarray(data['dof_pos']) - np.asarray(data['reference_dof_pos'])
    root_error = np.asarray(data['root_state'])[:, :3] - np.asarray(data['reference_root_pos'])
    match = 1 - np.abs(contact.astype(float) - np.asarray(data['reference_contact'])[valid])
    contacted = speed[contact]
    result = {
        'duration_s': float(times[-1]),
        'samples': len(times),
        'completed_clip': bool(times[-1] >= duration - 1e-5 and not physical_failure.any()),
        'physical_failure': bool(physical_failure.any()),
        'joint_rmse_rad': float(np.sqrt(np.mean(q_error ** 2))),
        'root_position_rmse_m': float(np.sqrt(np.mean(np.sum(root_error ** 2, axis=-1)))),
        'root_rotation_mean_deg': float(np.rad2deg(rotation_error(
            np.asarray(data['root_state'])[:, 3:7], np.asarray(data['reference_root_quat']))).mean()),
        'contact_match_score': float(match.mean()) if match.size else None,
        'contact_sole_speed_mean_m_s': float(contacted.mean()) if contacted.size else None,
        'contact_sole_speed_p95_m_s': float(np.percentile(contacted, 95)) if contacted.size else None,
        'min_base_height_m': float(np.asarray(data['root_state'])[:, 2].min()),
        'max_abs_torque_Nm': float(np.max(np.abs(data['torque']))),
        'notes': 'Contact sole-center horizontal speed includes rolling; not a pure slip distance. Initial FK/contact sample excluded. Not a robustness test.',
    }
    result.update(trunk_metrics(np.asarray(data['root_state'])[:, 3:7],
                                np.asarray(data['reference_root_quat'])))
    return result
