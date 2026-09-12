"""Reference-only full-cycle weights and body-weight-normalized impact costs."""
import torch


GROUPS = {
    "x1_gmr_cycle_contact": (0., -.2, True),
    "x1_gmr_cycle_smooth": (-.04, 0., False),
    "x1_gmr_cycle_both": (-.04, -.2, True),
}


def acceleration_weights(swing, stance, floor):
    """Never remove the floor at contact transitions; no measured contact input."""
    if not 0 < floor <= 1:
        raise ValueError("Full-cycle floor must be in (0,1]")
    return floor + (1. - floor) * torch.maximum(swing, stance)


def weighted_acceleration_cost(acceleration, leg_ids, phase_weights, joint_weights, scale):
    per_leg = torch.stack([(acceleration[:, ids] / scale).square().mul(joint_weights).mean(-1)
                           for ids in leg_ids], dim=-1)
    return (per_leg * phase_weights).sum(-1)


def impact_cost(vertical_force, swing, body_weight, stance_limit=1.5, swing_limit=.05):
    """Smooth L1 excess force: quadratic nearby, linear tail; no hard cost cap.

    The allowance is per foot, in units of total robot weight. Reference swing
    lowers it, but even stance/transition impacts retain supervision. This is a
    force magnitude cost, not an impulse measurement or a hardware force limit.
    """
    allowance = stance_limit + (swing_limit - stance_limit) * swing
    excess = (vertical_force.clamp_min(0.) / body_weight - allowance).clamp_min(0.)
    return torch.where(excess < 1., .5 * excess.square(), excess - .5).sum(-1)
