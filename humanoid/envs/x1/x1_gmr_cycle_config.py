"""Three bounded ablations, preserving all model8000 dynamics and references."""
from .x1_gmr_swing_config import X1GMRSwingCfg, X1GMRSwingCfgPPO


class X1GMRCycleBaseCfg(X1GMRSwingCfg):
    class cycle:
        boundary_s = .02
        ramp_s = .04
        acceleration_floor = .25
        acceleration_scale_rad_s2 = 100.
        # hip pitch/roll/yaw, knee, ankle pitch/roll; mapped by name.
        joint_weights = [1., 1., 1., 1., 1.5, 2.]
        stance_force_body_weights = 1.5
        swing_force_body_weights = .05

    class swing(X1GMRSwingCfg.swing):
        contact_phase_only = False

    class rewards(X1GMRSwingCfg.rewards):
        class scales(X1GMRSwingCfg.rewards.scales):
            gmr_cycle_acc = 0.
            gmr_cycle_impact = 0.


class X1GMRCycleContactCfg(X1GMRCycleBaseCfg):
    class swing(X1GMRCycleBaseCfg.swing):
        contact_phase_only = True

    class rewards(X1GMRCycleBaseCfg.rewards):
        class scales(X1GMRCycleBaseCfg.rewards.scales):
            gmr_cycle_impact = -.2


class X1GMRCycleSmoothCfg(X1GMRCycleBaseCfg):
    class rewards(X1GMRCycleBaseCfg.rewards):
        class scales(X1GMRCycleBaseCfg.rewards.scales):
            gmr_cycle_acc = -.04


class X1GMRCycleBothCfg(X1GMRCycleContactCfg):
    class rewards(X1GMRCycleContactCfg.rewards):
        class scales(X1GMRCycleContactCfg.rewards.scales):
            gmr_cycle_acc = -.04


class X1GMRCycleCfgPPO(X1GMRSwingCfgPPO):
    class runner(X1GMRSwingCfgPPO.runner):
        experiment_name = "x1_gmr_kit317_cycle_12dof"
