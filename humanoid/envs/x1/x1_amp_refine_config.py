"""Only task regularization changes. Actor observations, actions and dynamics stay fixed."""
from .x1_amp_recovery_config import X1AMPRecoveryCfg


class X1AMPRefineCfg(X1AMPRecoveryCfg):
    class rewards(X1AMPRecoveryCfg.rewards):
        class scales(X1AMPRecoveryCfg.rewards.scales):
            recovery_smoothness = -.02  # 10x first/second action-difference penalty
            recovery_yaw_rate = -1.    # 10x yaw-rate tracking penalty
            refine_heading = -.5      # heading command is world +X; yaw is observed
            refine_acceleration = -.5 # continuous robust cost, no unpenalized gait phase
            refine_slip = -2.          # continuous force-weighted sole velocity squared
