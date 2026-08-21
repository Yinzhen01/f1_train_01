"""No-DR X1 task that imitates the 12 controlled joints from walk.csv."""

from .x1_dh_stand_no_dr_config import X1DHStandNoDRCfg, X1DHStandNoDRCfgPPO


class X1DHStandRetargetWalkCfg(X1DHStandNoDRCfg):
    class env(X1DHStandNoDRCfg.env):
        # The policy learns the trajectory from phase and reward.  It is not a
        # residual controller that receives the reference as an added action.
        use_ref_actions = False

    class motion_reference(X1DHStandNoDRCfg.motion_reference):
        enabled = True
        file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/x1/walk_12dof.csv"
        # Consecutive left-knee swing peaks delimit a clean, nearly periodic
        # cycle in the 30 Hz source (0.600 s -> 5.533 s).
        start_time = 0.6
        end_time = 5.533333333333333
        close_loop = True
        # In the environment, left swing occupies phase [0.5, 1.0).  The CSV
        # clip starts at a left-knee swing peak, hence the half-cycle offset.
        phase_offset = 0.5

    class commands(X1DHStandNoDRCfg.commands):
        curriculum = False
        gait = ["walk_sagittal"]
        gait_time_range = {"walk_sagittal": [1, 1]}
        heading_command = False
        sw_switch = False

        class ranges:
            # The source root travels 3.52 m over 13.8 s (about 0.255 m/s).
            lin_vel_x = [0.255, 0.255]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(X1DHStandNoDRCfg.rewards):
        cycle_time = 4.933333333333333
        # The first imitation stage should clearly distinguish walking from
        # standing still.  At vx=0 the inherited sigma=5 kernel would still
        # award 72% of the maximum 0.255 m/s tracking reward.
        tracking_sigma = 30.0

        class scales(X1DHStandNoDRCfg.rewards.scales):
            # Reference-first curriculum: learn the 12-joint trajectory before
            # introducing contact labels or hand-authored swing constraints.
            ref_joint_pos = 6.0
            tracking_lin_vel = 4.0
            low_speed = 2.0

            # These inherited terms directly compete with the retargeted joint
            # targets or rely on a synthetic 50/50 stance mask.  Reintroduce
            # them only after deriving contact/clearance targets from the
            # motion itself.
            default_joint_pos = 0.0
            feet_contact_number = 0.0
            feet_air_time = 0.0
            feet_clearance = 0.0
            track_vel_hard = 0.0


class X1DHStandRetargetWalkCfgPPO(X1DHStandNoDRCfgPPO):
    class runner(X1DHStandNoDRCfgPPO.runner):
        experiment_name = "x1_dh_stand_retarget_walk"


class X1DHStandRetargetWalkVx045Cfg(X1DHStandRetargetWalkCfg):
    """Time-scaled reference whose root-path speed is 0.45 m/s."""

    class motion_reference(X1DHStandRetargetWalkCfg.motion_reference):
        # FK-derived foot-bottom clearance from the same source frames. Only
        # two scalar targets are added; the policy still controls 12 joints.
        clearance_file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/motions/x1/walk_foot_clearance.csv"
        )
        clearance_swing_threshold = 0.02
        # Raise the retargeted swing modestly: 5% plus 5 mm. Raw FK peaks are
        # about 15.5 cm (left) and 11.8 cm (right), yielding targets of about
        # 16.8 cm and 12.9 cm after lifting.
        clearance_scale = 1.05
        clearance_lift_offset = 0.005
        clearance_max = 0.18

    class commands(X1DHStandRetargetWalkCfg.commands):
        class ranges:
            lin_vel_x = [0.45, 0.45]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(X1DHStandRetargetWalkCfg.rewards):
        # The selected root path is 1.2597648717 m long. Compressing the
        # original 4.933333 s clip by 1.762234x gives 1.259765 / 0.45.
        cycle_time = 2.799477492696072

        ref_feet_clearance_sigma = 400.0
        ref_feet_clearance_low_penalty = 0.5

        class scales(X1DHStandRetargetWalkCfg.rewards.scales):
            # Use reference-derived swing timing instead of the synthetic
            # single-cycle stance mask used by the legacy feet_clearance term.
            ref_feet_clearance = 3.0


class X1DHStandRetargetWalkVx045CfgPPO(X1DHStandRetargetWalkCfgPPO):
    class runner(X1DHStandRetargetWalkCfgPPO.runner):
        experiment_name = "x1_dh_stand_retarget_walk_vx045"
