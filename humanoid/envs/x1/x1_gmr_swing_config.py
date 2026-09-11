"""Stage-one swing execution fine-tuning; all previous objectives stay fixed."""
from .x1_gmr_smooth_config import X1GMRSmoothCfg, X1GMRSmoothCfgPPO


class X1GMRSwingCfg(X1GMRSmoothCfg):
    class swing:
        geometry_file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/gmr_kit317_swing_geometry.json"
        boundary_s = .04
        ramp_s = .06
        gate_height_low_m = .005
        gate_height_full_m = .015
        height_tolerance_m = .005
        height_sigma_m = .02
        max_height_cost = 9.
        contact_threshold_n = 5.  # Same instantaneous threshold as the old slip term.

    class rewards(X1GMRSmoothCfg.rewards):
        class scales(X1GMRSmoothCfg.rewards.scales):
            gmr_swing_clearance = -1.
            gmr_swing_contact = -1.


class X1GMRSwingCfgPPO(X1GMRSmoothCfgPPO):
    class runner(X1GMRSmoothCfgPPO.runner):
        experiment_name = "x1_gmr_kit317_swing_12dof"
        max_iterations = 1000  # ADDITIONAL updates: 7000..7999.
        resume = False  # Dedicated hash-gated warm start, never scratch.
