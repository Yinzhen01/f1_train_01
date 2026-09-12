"""Isolated phase-gated reward experiment; no policy or dynamics changes."""
from .x1_gmr_swing_config import X1GMRSwingCfg, X1GMRSwingCfgPPO


class X1GMRPhaseAccelCfg(X1GMRSwingCfg):
    class swing(X1GMRSwingCfg.swing):
        contact_phase_only = True

    class phase_accel:
        # Unit normalization, NOT a threshold: a=100 gives (a/scale)^2=1.
        scale_rad_s2 = 100.

    class rewards(X1GMRSwingCfg.rewards):
        class scales(X1GMRSwingCfg.rewards.scales):
            # Provisional starting weight; requires Isaac Gym reward-budget smoke.
            # Average over six joints and ten substeps, sum over gated legs.
            gmr_phase_acc = -.02


class X1GMRPhaseAccelCfgPPO(X1GMRSwingCfgPPO):
    class runner(X1GMRSwingCfgPPO.runner):
        experiment_name = "x1_gmr_kit317_phase_accel_12dof"
        # Inherited budget is a config default, not a submitted training job.
        # Approved smoke uses train_gmr_phase_smoke.py (model8000, 512 x 20).
