"""Fixed-waist chest-reference experiment; old GMR task remains unchanged."""
from .x1_gmr_clip_config import X1GMRClipCfg, X1GMRClipCfgPPO


class X1GMRUprightCfg(X1GMRClipCfg):
    class motion_reference(X1GMRClipCfg.motion_reference):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/gmr_kit317_upright_12dof.npz"
        sha256 = "d625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb"

    class rewards(X1GMRClipCfg.rewards):
        # Unit-gravity chord distance: sigma .1 is about 5.73 deg at reward e^-1.
        trunk_tilt_sigma = .1

        class scales(X1GMRClipCfg.rewards.scales):
            gmr_trunk_tilt = 1.


class X1GMRUprightCfgPPO(X1GMRClipCfgPPO):
    class runner(X1GMRClipCfgPPO.runner):
        experiment_name = "x1_gmr_kit317_upright_12dof"
        resume = False
