# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-1 dynamics randomization for the learned mixed-command policy.

This stage deliberately introduces only narrow, low-impact dynamics variation.
Observation noise, pushes, delays, and detailed joint-level randomization stay
disabled so that the model can adapt without losing the learned gait.
"""

from .x1_dh_stand_no_dr_mixed_config import (
    X1DHStandNoDRMixedCfg,
    X1DHStandNoDRMixedCfgPPO,
)


class X1DHStandDRStage1Cfg(X1DHStandNoDRMixedCfg):
    class domain_rand(X1DHStandNoDRMixedCfg.domain_rand):
        # Nominal terrain friction is 0.6. Keep the first range centered near
        # that value instead of immediately restoring the full [0.2, 1.3].
        randomize_friction = True
        friction_range = [0.45, 0.80]

        # Low-amplitude rigid-body uncertainty.
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        randomize_com = True
        com_displacement_range = [
            [-0.015, 0.015],
            [-0.015, 0.015],
            [-0.015, 0.015],
        ]

        # Low-amplitude actuator uncertainty.
        randomize_gains = True
        stiffness_multiplier_range = [0.95, 1.05]
        damping_multiplier_range = [0.95, 1.05]
        randomize_torque = True
        torque_multiplier_range = [0.95, 1.05]

        # High-impact sources remain disabled for the first DR stage.
        push_robots = False
        add_ext_force = False
        continuous_push = False
        randomize_link_mass = False
        randomize_motor_offset = False
        randomize_joint_friction = False
        randomize_joint_damping = False
        # Stage-1 keeps deterministic robot armature at the shared nominal
        # values. Disabling armature DR must not fall back to zero inertia.
        use_nominal_joint_armature = True
        randomize_joint_armature = False
        randomize_coulomb_friction = False
        add_lag = False
        add_dof_lag = False
        add_dof_pos_vel_lag = False
        add_imu_lag = False
        enable_delivery = False


class X1DHStandDRStage1CfgPPO(X1DHStandNoDRMixedCfgPPO):
    class runner(X1DHStandNoDRMixedCfgPPO.runner):
        experiment_name = "x1_dh_stand_dr_stage1"
