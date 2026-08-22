"""No-DR X1 task that imitates the 12 controlled joints from walk.csv."""

from .x1_dh_stand_no_dr_config import X1DHStandNoDRCfg, X1DHStandNoDRCfgPPO


class X1DHStandRetargetWalkCfg(X1DHStandNoDRCfg):
    class env(X1DHStandNoDRCfg.env):
        # The policy learns the trajectory from phase and reward.  It is not a
        # residual controller that receives the reference as an added action.
        use_ref_actions = False

    class motion_reference(X1DHStandNoDRCfg.motion_reference):
        enabled = True
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/motions/x1/"
            "walk_12dof_contact_consistent.csv"
        )
        # Full-body normalized-pose autocorrelation identifies a 146-frame
        # cycle in the 30 Hz source. The processed file retains only the root,
        # 12 controlled joints, and contact confidence columns.
        contact_columns = ("left_contact", "right_contact")
        contact_force_threshold = 40.0
        start_time = 0.6
        end_time = 5.466666666666667
        close_loop = True
        # A circular search against the reference contacts places the first
        # support transfer 26 frames into the selected 146-frame cycle. This
        # makes the environment phase mask and reference contacts agree on
        # 99.3% of foot/frame pairs; the former 0.5 offset agreed on only 36.3%.
        phase_offset = 0.1780821917808219

    class commands(X1DHStandNoDRCfg.commands):
        curriculum = False
        gait = ["walk_sagittal"]
        gait_time_range = {"walk_sagittal": [1, 1]}
        heading_command = False
        sw_switch = False

        class ranges:
            # Periodic, contact-consistent reconstruction gives a mean 0.301 m
            # forward step every 2.433 s (about 0.124 m/s). The original raw
            # root trajectory was rejected because it made the stance foot
            # slide at about 0.14 m/s RMS.
            lin_vel_x = [0.124, 0.124]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(X1DHStandNoDRCfg.rewards):
        cycle_time = 4.866666666666667
        # The first imitation stage should distinguish walking from standing.
        # At vx=0 the inherited sigma=5 kernel would award 92.6% of the maximum
        # 0.124 m/s tracking reward; sigma=30 lowers that to about 63%, while
        # the low_speed term supplies the explicit standing penalty.
        tracking_sigma = 30.0

        class scales(X1DHStandNoDRCfg.rewards.scales):
            # Reference-first curriculum: learn the 12-joint trajectory before
            # introducing contact labels or hand-authored swing constraints.
            ref_joint_pos = 6.0
            # Data-derived replacement for the legacy feet_contact_number
            # square wave. It retains the old contact reward's [-1, 1] range
            # while respecting the reference's soft double-support windows.
            ref_feet_contact = 2.0
            tracking_lin_vel = 4.0
            low_speed = 2.0

            # Keep the remaining legacy terms disabled. feet_air_time can
            # reward a scheduled stance transition even before real contact;
            # feet_clearance uses a 3-6 cm target instead of the measured
            # 13.6-16.4 cm peaks; the other terms duplicate or compete with
            # reference/velocity objectives.
            default_joint_pos = 0.0
            feet_contact_number = 0.0
            feet_air_time = 0.0
            feet_clearance = 0.0
            track_vel_hard = 0.0


class X1DHStandRetargetWalkCfgPPO(X1DHStandNoDRCfgPPO):
    class runner(X1DHStandNoDRCfgPPO.runner):
        experiment_name = "x1_dh_stand_retarget_walk_periodic_contact"


class X1DHStandRetargetWalkVx045Cfg(X1DHStandRetargetWalkCfg):
    """Time-scaled contact-consistent 0.45 m/s reference."""

    class motion_reference(X1DHStandRetargetWalkCfg.motion_reference):
        # FK-derived foot-bottom clearance from the same source frames. Only
        # two scalar targets are added; the policy still controls 12 joints.
        clearance_file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/motions/x1/walk_foot_clearance.csv"
        )
        clearance_swing_threshold = 0.02
        # Conservative baseline: track the measured retargeted clearance
        # without inflating it. Mesh-derived sole peaks remain about 16.4 cm
        # (left) and 13.6 cm (right).
        clearance_scale = 1.0
        clearance_lift_offset = 0.0
        clearance_max = 0.18

    class commands(X1DHStandRetargetWalkCfg.commands):
        class ranges:
            lin_vel_x = [0.45, 0.45]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(X1DHStandRetargetWalkCfg.rewards):
        # Two 0.3013507 m steps per cycle give 0.6027014 m. Replaying that
        # geometry in 1.339336 s yields 0.45 m/s and about 89.6 steps/min.
        cycle_time = 1.3393364741639295

        ref_feet_clearance_sigma = 400.0
        ref_feet_clearance_low_penalty = 0.5

        class scales(X1DHStandRetargetWalkCfg.rewards.scales):
            # Start with the same scale as reference contact. The reward also
            # requires both reference swing contact and clearance > 2 cm.
            ref_feet_clearance = 2.0


class X1DHStandRetargetWalkVx045CfgPPO(X1DHStandRetargetWalkCfgPPO):
    class runner(X1DHStandRetargetWalkCfgPPO.runner):
        experiment_name = (
            "x1_dh_stand_retarget_walk_periodic_contact_vx045_conservative"
        )


class X1DHStandRetargetWalkVx045GeometryCfg(X1DHStandRetargetWalkVx045Cfg):
    """Conservative clearance plus phase-aligned stance geometry."""

    class motion_reference(X1DHStandRetargetWalkVx045Cfg.motion_reference):
        stance_geometry_file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/motions/x1/"
            "walk_stance_geometry.csv"
        )
        foot_lateral_min_target = 0.12
        knee_lateral_min_target = 0.14
        foot_heading_max_target = 0.17453292519943295
        # Keep the clipped imitation target away from the physical hip-roll
        # boundary; the source left hip roll otherwise saturates at 0.20 rad.
        joint_limit_margin_by_name = {
            "left_hip_roll_joint": 0.02,
            "right_hip_roll_joint": 0.02,
        }

    class rewards(X1DHStandRetargetWalkVx045Cfg.rewards):
        ref_foot_lateral_sigma = 200.0
        ref_knee_lateral_sigma = 200.0
        ref_foot_heading_sigma = 20.0
        foot_heading_common_sigma = 20.0
        ref_hip_yaw_sigma = 20.0
        foot_lateral_safe_min = 0.10
        knee_lateral_safe_min = 0.12
        foot_lateral_shortfall_penalty = 1.0
        knee_lateral_shortfall_penalty = 1.0

        class scales(X1DHStandRetargetWalkVx045Cfg.rewards.scales):
            ref_foot_lateral_distance = 1.0
            ref_knee_lateral_distance = 0.5
            ref_foot_heading = 0.5
            ref_hip_yaw = 0.5
            # These legacy rewards use total XY distance, so fore-aft
            # separation can hide a crossed or overly narrow stance.
            feet_distance = 0.0
            knee_distance = 0.0


class X1DHStandRetargetWalkVx045GeometryCfgPPO(
    X1DHStandRetargetWalkVx045CfgPPO
):
    class runner(X1DHStandRetargetWalkVx045CfgPPO.runner):
        experiment_name = (
            "x1_dh_stand_retarget_walk_periodic_contact_vx045_geometry"
        )
