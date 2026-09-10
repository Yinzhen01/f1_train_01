"""12-leg-DOF, no-DR, noncyclic KIT317 GMR imitation baseline.

Observation width stays 47, but the two phase channels encode a half-circle
over the entire clip. Old locomotion checkpoints are NOT semantically compatible.
"""
from .x1_dh_stand_no_dr_config import X1DHStandNoDRCfg, X1DHStandNoDRCfgPPO


class X1GMRClipCfg(X1DHStandNoDRCfg):
    class env(X1DHStandNoDRCfg.env):
        episode_length_s = 4.6
        use_ref_actions = False

    class motion_reference:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/gmr_kit317_foot_flat_12dof.npz"
        sha256 = "a3ae41e54909455d477cefa662bdcbbee8a1852564ef547ebeb455f275a0be02"
        random_start = True
        start_at_zero_probability = .25
        minimum_remaining_s = .3

    class commands(X1DHStandNoDRCfg.commands):
        curriculum = False
        heading_command = False
        sw_switch = False

    class rewards(X1DHStandNoDRCfg.rewards):
        only_positive_rewards = False

        # Independent scale class deliberately removes procedural gait rewards,
        # default-pose/stand-still bias and all-phase feet-horizontal rewards.
        class scales:
            gmr_joint_pos = 6.
            gmr_joint_vel = 0.5
            gmr_foot_pos = 2.
            gmr_foot_rotation = 1.
            gmr_contact = 1.
            gmr_root_pos = 1.
            gmr_root_rotation = 1.
            gmr_velocity = 2.
            gmr_slip = -2.
            action_smoothness = -.01
            torques = -8e-9
            dof_acc = -1e-7
            collision = -1.
            dof_vel_limits = -1.
            dof_pos_limits = -10.
            dof_torque_limits = -.1


class X1GMRClipCfgPPO(X1DHStandNoDRCfgPPO):
    class policy(X1DHStandNoDRCfgPPO.policy):
        init_noise_std = .5

    class algorithm(X1DHStandNoDRCfgPPO.algorithm):
        learning_rate = 1e-4

    class runner(X1DHStandNoDRCfgPPO.runner):
        experiment_name = "x1_gmr_kit317_foot_flat_12dof"
        max_iterations = 5000
        resume = False
