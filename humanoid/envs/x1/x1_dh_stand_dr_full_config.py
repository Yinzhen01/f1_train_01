# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Complete domain randomization for continuation from the Stage-1 policy.

Commands, rewards, network settings, and flat terrain remain identical to the
mixed-command Stage-1 experiment. Observation noise and the complete default
dynamics randomization from ``X1DHStandCfg`` are enabled for Stage-2 training.
"""

from .x1_dh_stand_config import X1DHStandCfg
from .x1_dh_stand_no_dr_mixed_config import (
    X1DHStandNoDRMixedCfg,
    X1DHStandNoDRMixedCfgPPO,
)


class X1DHStandDRFullCfg(X1DHStandNoDRMixedCfg):
    class noise(X1DHStandCfg.noise):
        add_noise = True

    class domain_rand(X1DHStandCfg.domain_rand):
        pass


class X1DHStandDRFullCfgPPO(X1DHStandNoDRMixedCfgPPO):
    class runner(X1DHStandNoDRMixedCfgPPO.runner):
        experiment_name = "x1_dh_stand_dr_full"
