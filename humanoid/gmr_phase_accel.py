"""Reference-only swing gating and physics-rate acceleration reward math."""
import torch

from humanoid.gmr_swing import sample_envelope


def leg_joint_indices(dof_names):
    """Map six joints of each leg by name, never by an assumed tensor order."""
    suffixes = ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
    names = [[side + "_" + joint + "_joint" for joint in suffixes]
             for side in ("left", "right")]
    if len(dof_names) != 12 or set(dof_names) != set(names[0] + names[1]):
        raise ValueError("Phase acceleration requires exactly the 12 named leg DOFs")
    return [[dof_names.index(name) for name in leg] for leg in names]


def substep_phase_gates(envelope, start_times, fps, physics_dt, decimation, valid):
    """[env, substep, left/right]; conservative gates across each interval.

    start_times includes each environment's random reference start. No actual
    contact or height enters this mask. Take the smaller endpoint gate so an
    interval straddling an excluded boundary cannot receive a strong penalty.
    """
    offsets = torch.arange(decimation + 1, device=start_times.device) * physics_dt
    times = start_times[:, None] + offsets[None, :]
    endpoints = sample_envelope(envelope, times.reshape(-1), fps).reshape(-1, decimation + 1, 2)
    return torch.minimum(endpoints[:, :-1], endpoints[:, 1:]) * valid[:, None, None]


def leg_acceleration_square(acceleration, leg_ids):
    """Mean squared actual angular acceleration for each leg, in rad^2/s^4."""
    return torch.stack([acceleration[:, ids].square().mean(dim=1) for ids in leg_ids], dim=1)
