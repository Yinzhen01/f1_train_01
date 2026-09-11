"""V1.4 whole-body KIT317 imitation, separate from fixed-waist 12-DOF policies."""
import json
from pathlib import Path

from humanoid import LEGGED_GYM_ROOT_DIR
from .x1_gmr_clip_config import X1GMRClipCfg, X1GMRClipCfgPPO


_REFERENCE = json.loads((Path(LEGGED_GYM_ROOT_DIR) /
                         "resources/motions/gmr_kit317_foot_flat_29dof.json").read_text(encoding="utf-8"))


class X1GMR29DOFCfg(X1GMRClipCfg):
    class env(X1GMRClipCfg.env):
        num_actions = 29
        num_single_obs = 11 + 3 * num_actions  # 98
        single_num_privileged_obs = 25 + 4 * num_actions  # 141
        single_linvel_index = 5 + 4 * num_actions  # 121
        num_observations = X1GMRClipCfg.env.frame_stack * num_single_obs
        num_privileged_obs = X1GMRClipCfg.env.c_frame_stack * single_num_privileged_obs
        num_envs = 2048

    class asset(X1GMRClipCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/f1_v1_4/urdf/F1_V1_4_29DOF.urdf"
        # Do not silently deploy through the unrelated legacy 29-DOF MJCF.
        xml_file = ""
        name = "f1_v1_4_29dof"

    class motion_reference(X1GMRClipCfg.motion_reference):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/gmr_kit317_foot_flat_29dof.npz"
        sha256 = _REFERENCE["output_sha256"]

    class init_state(X1GMRClipCfg.init_state):
        default_joint_angles = dict(_REFERENCE["initial_joint_angles"])

    class control(X1GMRClipCfg.control):
        # Legs preserve the previous training gains. Upper-body values are
        # conservative simulation starting points, NOT identified hardware gains.
        stiffness = dict(X1GMRClipCfg.control.stiffness,
                         lumbar_yaw_joint=100., lumbar_roll_joint=100., lumbar_pitch_joint=100.,
                         shoulder_pitch_joint=40., shoulder_roll_joint=40., shoulder_yaw_joint=40.,
                         elbow_pitch_joint=40., elbow_yaw_joint=40., wrist_pitch_joint=5.,
                         neck_motor_base_pitch_joint=5., head_face_bracket_pitch_joint=5.)
        damping = dict(X1GMRClipCfg.control.damping,
                       lumbar_yaw_joint=4., lumbar_roll_joint=4., lumbar_pitch_joint=4.,
                       shoulder_pitch_joint=2., shoulder_roll_joint=2., shoulder_yaw_joint=2.,
                       elbow_pitch_joint=2., elbow_yaw_joint=2., wrist_pitch_joint=.5,
                       neck_motor_base_pitch_joint=.5, head_face_bracket_pitch_joint=.5)

    class domain_rand(X1GMRClipCfg.domain_rand):
        joint_armature_config_file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/f1_v1_4/config/joint_dynamics.json"

    class rewards(X1GMRClipCfg.rewards):
        class scales(X1GMRClipCfg.rewards.scales):
            # The six-point joint tracking term is leg-only here; give upper-body
            # groups separate budgets so quiet arms cannot hide poor leg tracking.
            gmr_upper_joint_pos = 2.
            gmr_waist_joint_pos = 1.
            gmr_neck_joint_pos = .25
            gmr_hand_pos = 1.
            gmr_chest_rotation = 1.
            # Temporal differences only, averaged over all 29 actions; no L1 pose bias.
            action_smoothness = -1.2
            # Normalized torque derivative: mean((delta_tau / torque_limit / dt)^2).
            gmr_torque_rate = -5e-5


class X1GMR29DOFCfgPPO(X1GMRClipCfgPPO):
    class algorithm(X1GMRClipCfgPPO.algorithm):
        # State estimator target is the latest critic frame's linear velocity.
        lin_vel_idx = (X1GMR29DOFCfg.env.c_frame_stack - 1) * X1GMR29DOFCfg.env.single_num_privileged_obs + X1GMR29DOFCfg.env.single_linvel_index

    class runner(X1GMRClipCfgPPO.runner):
        experiment_name = "f1_v1_4_gmr_kit317_29dof"
        max_iterations = 5000
        resume = False  # 12-DOF model_5000.pt is dimensionally incompatible.
