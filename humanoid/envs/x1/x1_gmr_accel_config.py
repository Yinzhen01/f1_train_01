"""Controlled acceleration-weight sweep; preserve the complete swing objective."""
from .x1_gmr_swing_config import X1GMRSwingCfg, X1GMRSwingCfgPPO


class X1GMRAccel1Cfg(X1GMRSwingCfg):
    """Same objective as model8000; controls for extra training alone."""
    pass


class X1GMRAccel3Cfg(X1GMRSwingCfg):
    class rewards(X1GMRSwingCfg.rewards):
        class scales(X1GMRSwingCfg.rewards.scales):
            dof_acc = -3e-7


class X1GMRAccel10Cfg(X1GMRSwingCfg):
    class rewards(X1GMRSwingCfg.rewards):
        class scales(X1GMRSwingCfg.rewards.scales):
            dof_acc = -1e-6


class X1GMRAccelCfgPPO(X1GMRSwingCfgPPO):
    class runner(X1GMRSwingCfgPPO.runner):
        experiment_name = "x1_gmr_kit317_accel_12dof"
        max_iterations = 1000  # Additional updates, always from ORIGINAL model8000.
