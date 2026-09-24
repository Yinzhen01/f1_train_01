"""First AMP learnability experiment, nominal unchanged dynamics, no DR.

No old checkpoint is compatible with the new reward/phase semantics. Actor
dimensions/history/PD/URDF remain unchanged; this experiment starts from scratch.
"""
from .x1_dh_stand_no_dr_config import X1DHStandNoDRCfg, X1DHStandNoDRCfgPPO


class X1AMPCfg(X1DHStandNoDRCfg):
    class env(X1DHStandNoDRCfg.env):
        episode_length_s = 6.0
        use_ref_actions = False

    class commands(X1DHStandNoDRCfg.commands):
        curriculum = False
        gait = ["walk_sagittal"]
        gait_time_range = {"walk_sagittal": [1, 1]}
        heading_command = False

        class ranges(X1DHStandNoDRCfg.commands.ranges):
            # Demo median body-forward speed is 0.4507 m/s; no lateral/yaw task.
            lin_vel_x = [0.45, 0.45]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(X1DHStandNoDRCfg.rewards):
        # No hand-authored sine gait, joint/foot matching, or proxy phase labels.
        class scales:
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            orientation = 0.5
            vel_mismatch_exp = 0.2
            action_smoothness = -0.002
            torques = -8e-9
            dof_vel = -2e-8
            dof_acc = -1e-7
            collision = -1.0
            feet_contact_forces = -0.01
            dof_vel_limits = -1.0
            dof_pos_limits = -10.0
            dof_torque_limits = -0.1


class X1AMPCfgPPO(X1DHStandNoDRCfgPPO):
    class algorithm(X1DHStandNoDRCfgPPO.algorithm):
        learning_rate = 3e-4
        schedule = "fixed"

    class runner(X1DHStandNoDRCfgPPO.runner):
        experiment_name = "f1_amp_walk02_s092"
        max_iterations = 3000
