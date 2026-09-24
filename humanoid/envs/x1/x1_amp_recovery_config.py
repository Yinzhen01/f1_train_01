"""No-DR AMP recovery: remove stationary bonuses, preserve nominal dynamics."""
from .x1_amp_config import X1AMPCfg, X1AMPCfgPPO


class X1AMPRecoveryCfg(X1AMPCfg):
    class rewards(X1AMPCfg.rewards):
        only_positive_rewards = False

        class scales(X1AMPCfg.rewards.scales):
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            orientation = 0.
            vel_mismatch_exp = 0.
            action_smoothness = 0.
            recovery_progress = 2.
            recovery_tilt = -.5
            recovery_yaw_rate = -.1
            recovery_vertical_velocity = -.1
            recovery_smoothness = -.002
            termination = -50.  # dt-scaled = -0.5 on a physical failure, not timeout


class X1AMPRecoveryCfgPPO(X1AMPCfgPPO):
    class policy(X1AMPCfgPPO.policy):
        init_noise_std = .35

    class runner(X1AMPCfgPPO.runner):
        experiment_name = "f1_amp_walk02_recovery"
        max_iterations = 1000
