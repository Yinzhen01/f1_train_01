"""Temporal-smoothness fine-tuning of the verified 12-DOF upright policy."""
from .x1_gmr_upright_config import X1GMRUprightCfg, X1GMRUprightCfgPPO


class X1GMRSmoothCfg(X1GMRUprightCfg):
    class rewards(X1GMRUprightCfg.rewards):
        class scales(X1GMRUprightCfg.rewards.scales):
            # Mean over 12 joints. Equivalent temporal coefficient -0.1 on
            # each sum: 10x the old -0.01, without the old L1 pose bias.
            action_smoothness = -1.2
            # Mean squared normalized torque derivative, before dt scaling.
            gmr_torque_rate = -5e-5


class X1GMRSmoothCfgPPO(X1GMRUprightCfgPPO):
    class algorithm(X1GMRUprightCfgPPO.algorithm):
        learning_rate = 1e-5
        schedule = "fixed"  # Do not let the adaptive schedule raise this rate.

    class runner(X1GMRUprightCfgPPO.runner):
        experiment_name = "x1_gmr_kit317_smooth_12dof"
        max_iterations = 2000  # ADDITIONAL updates, from completed update 5000.
        # The dedicated entry point performs a hash-verified strict warm start.
        resume = False
