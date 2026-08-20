# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-only X1 learnability gate without domain randomization.

This experiment deliberately removes command diversity and reduces gait-shape
shortcuts.  Its only purpose is to establish that PPO can learn actual forward
translation before stand, lateral, yaw, and robustness objectives are restored.
"""

from .x1_dh_stand_no_dr_config import X1DHStandNoDRCfg, X1DHStandNoDRCfgPPO


class X1DHStandNoDRForwardCfg(X1DHStandNoDRCfg):
    class commands(X1DHStandNoDRCfg.commands):
        curriculum = False
        gait = ["walk_sagittal"]
        gait_time_range = {"walk_sagittal": [1, 1]}
        heading_command = False
        sw_switch = True

        class ranges(X1DHStandNoDRCfg.commands.ranges):
            # A fixed command is intentional: first prove that the policy can
            # translate at one known speed, then broaden the command space.
            lin_vel_x = [0.4, 0.4]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(X1DHStandNoDRCfg.rewards):
        # Preserve negative feedback from low-speed and hard-tracking terms.
        only_positive_rewards = False

        class scales(X1DHStandNoDRCfg.rewards.scales):
            # Make forward translation dominate the learnability gate.
            tracking_lin_vel = 4.0
            low_speed = 1.0
            track_vel_hard = 1.0

            # Keep a weak gait prior while removing the profitable in-place
            # stepping shortcut observed in TASK_20260820_175.
            ref_joint_pos = 0.5
            feet_clearance = 0.25
            feet_contact_number = 0.5
            foot_slip = -0.25


class X1DHStandNoDRForwardCfgPPO(X1DHStandNoDRCfgPPO):
    class runner(X1DHStandNoDRCfgPPO.runner):
        experiment_name = "x1_dh_stand_no_dr_forward"
