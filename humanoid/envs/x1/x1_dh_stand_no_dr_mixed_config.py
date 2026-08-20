# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Old mixed-command training distribution with the new no-DR reward setup.

The command sampler is inherited unchanged from ``X1DHStandCfg``: command
curriculum, mixed gait schedule, and the original vx/vy/yaw/heading ranges.
Only the velocity-related reward scales from the successful forward gate are
applied.  Domain randomization and observation noise remain disabled through
``X1DHStandNoDRCfg``.
"""

from .x1_dh_stand_no_dr_config import X1DHStandNoDRCfg, X1DHStandNoDRCfgPPO


class X1DHStandNoDRMixedCfg(X1DHStandNoDRCfg):
    class rewards(X1DHStandNoDRCfg.rewards):
        only_positive_rewards = True

        class scales(X1DHStandNoDRCfg.rewards.scales):
            tracking_lin_vel = 4.0
            low_speed = 1.0
            track_vel_hard = 1.0


class X1DHStandNoDRMixedCfgPPO(X1DHStandNoDRCfgPPO):
    class runner(X1DHStandNoDRCfgPPO.runner):
        experiment_name = "x1_dh_stand_no_dr_mixed"
