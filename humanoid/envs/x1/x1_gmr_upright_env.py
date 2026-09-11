"""Independent upright-reference variant; the original GMR baseline is unchanged."""
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from humanoid.gmr_posture import tilt_error_squared, tilt_tracking_reward
from .x1_gmr_clip_env import X1GMRClipEnv


class X1GMRUprightEnv(X1GMRClipEnv):
    def _init_buffers(self):
        super()._init_buffers()
        if self.motion.metadata.get('reference_variant') != 'upright_chest_v1':
            raise ValueError('Upright task requires its rebuilt reference, not the pelvis-tilt baseline')
        if self.cfg.rewards.trunk_tilt_sigma <= 0:
            raise ValueError('Invalid trunk tilt sigma')
        print('[gmr-upright] variant=%s reference_sha256=%s tilt_weight=%g tilt_sigma=%g' %
              (self.motion.metadata['reference_variant'], self.cfg.motion_reference.sha256,
               self.cfg.rewards.scales.gmr_trunk_tilt, self.cfg.rewards.trunk_tilt_sigma), flush=True)

    def _target_gravity(self):
        # For a fixed waist the root's gravity projection defines trunk tilt.
        # Both actual and target are expressed in their corresponding body frames;
        # world heading changes therefore do not create a spurious tilt error.
        return quat_rotate_inverse(self.reference['root_quat'], self.gravity_vec)

    def _reward_gmr_trunk_tilt(self):
        return tilt_tracking_reward(self.projected_gravity, self._target_gravity(),
                                    self.cfg.rewards.trunk_tilt_sigma)

    def get_command_tracking_debug(self):
        result = super().get_command_tracking_debug()
        chord = torch.sqrt(tilt_error_squared(self.projected_gravity, self._target_gravity()))
        angle = 2. * torch.asin((chord / 2.).clamp(0., 1.))
        result['gmr_trunk_tilt_error_deg'] = torch.rad2deg(angle).mean()
        result['gmr_actual_pitch_deg'] = torch.rad2deg(self.base_euler_xyz[:, 1]).mean()
        target = self._target_gravity()
        target_pitch = torch.asin(target[:, 0].clamp(-1., 1.))
        result['gmr_target_pitch_deg'] = torch.rad2deg(target_pitch).mean()
        return result
