# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic X1 baseline used to verify that the task is learnable.

This configuration intentionally removes Sim2Real robustness mechanisms while
leaving command and gait sampling enabled.  It is an ablation baseline, not a
deployment configuration.
"""

from .x1_dh_stand_config import X1DHStandCfg, X1DHStandCfgPPO


class X1DHStandNoDRCfg(X1DHStandCfg):
    class terrain(X1DHStandCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        restitution = 0.0

    class noise(X1DHStandCfg.noise):
        add_noise = False

    class domain_rand(X1DHStandCfg.domain_rand):
        # Contact and external disturbances.
        randomize_friction = False
        push_robots = False
        add_ext_force = False
        continuous_push = False

        # Rigid-body and actuator parameter randomization.
        randomize_base_mass = False
        randomize_com = False
        randomize_link_com = False
        randomize_base_inertia = False
        randomize_link_inertia = False
        randomize_gains = False
        randomize_torque = False
        randomize_link_mass = False
        randomize_motor_offset = False
        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False
        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False
        randomize_joint_armature = False
        randomize_joint_armature_each_joint = False
        randomize_coulomb_friction = False

        # Action and sensor delays.
        add_lag = False
        randomize_lag_timesteps = False
        randomize_lag_timesteps_perstep = False
        add_dof_lag = False
        randomize_dof_lag_timesteps = False
        randomize_dof_lag_timesteps_perstep = False
        add_dof_pos_vel_lag = False
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        add_imu_lag = False
        randomize_imu_lag_timesteps = False
        randomize_imu_lag_timesteps_perstep = False

        # Remove the deterministic ankle delivery filter as part of the ideal
        # simulation learnability baseline.
        enable_delivery = False


class X1DHStandNoDRCfgPPO(X1DHStandCfgPPO):
    class runner(X1DHStandCfgPPO.runner):
        experiment_name = "x1_dh_stand_no_dr"

