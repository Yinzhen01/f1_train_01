"""Read-only acceleration diagnostics; no reward/history/dynamics overrides."""
import torch
from .x1_gmr_swing_env import X1GMRSwingEnv


class X1GMRAccelEnv(X1GMRSwingEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self.accel_diagnostics = {}

    def compute_reward(self):
        super().compute_reward()
        # BEFORE reset/history overwrite. Same 100 Hz velocity difference as the
        # inherited reward, NOT instantaneous acceleration at every 1 kHz substep.
        acc = (self.dof_vel - self.last_dof_vel) / self.dt
        valid = self.episode_length_buf > 1
        count = valid.sum().clamp_min(1)
        masked = acc * valid[:, None]
        self.accel_diagnostics = {
            "gmr_joint_acc_rms_rad_s2": torch.sqrt(masked.square().sum() / count / self.num_actions),
            "gmr_ankle_acc_rms_rad_s2": torch.sqrt(masked[:, self.smooth_ankle_ids].square().sum() / count / 4),
            "gmr_joint_acc_peak_rad_s2": masked.abs().max(),
            "gmr_acc_raw_cost": masked.square().sum() / count,
            "gmr_acc_weighted_cost_before_dt": masked.square().sum() / count * self.reward_scales["dof_acc"] / self.dt,
        }

    def get_command_tracking_debug(self):
        result = super().get_command_tracking_debug()
        result.update(self.accel_diagnostics)
        return result
