"""Phase-gated acceleration at 1 kHz, with reset-safe per-control-step sums."""
import json
import math

import torch

from humanoid.gmr_phase_accel import (leg_joint_indices, substep_phase_gates,
                                     leg_acceleration_square)
from .x1_gmr_accel_env import X1GMRAccelEnv


class X1GMRPhaseAccelEnv(X1GMRAccelEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self.phase_leg_ids = leg_joint_indices(self.dof_names)
        self.phase_physics_dt = float(self.sim_params.dt)
        if (not math.isclose(self.phase_physics_dt, .001, rel_tol=0., abs_tol=1e-9)
                or self.cfg.control.decimation != 10 or self.sim_params.substeps != 1
                or not math.isclose(self.dt, .01, rel_tol=0., abs_tol=1e-9)):
            raise ValueError("Phase experiment requires unchanged 1 kHz physics / 100 Hz control")
        scale = self.cfg.phase_accel.scale_rad_s2
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Acceleration normalization must be finite and positive")
        if not self.cfg.swing.contact_phase_only:
            raise ValueError("Phase experiment must close the swing contact height-gate gap")
        self.phase_previous_vel = torch.zeros_like(self.dof_vel)
        self.phase_square_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self.phase_all_square_sum = torch.zeros_like(self.phase_square_sum)
        self.phase_gate_sum = torch.zeros_like(self.phase_square_sum)
        self.phase_peak = torch.zeros(self.num_envs, device=self.device)
        self.phase_acc_cost = torch.zeros(self.num_envs, device=self.device)
        self.phase_substep_count = 0
        self.phase_acc_diagnostics = {}
        print("[gmr-phase-accel] " + json.dumps(dict(
            physics_dt=self.phase_physics_dt, control_dt=self.dt, leg_ids=self.phase_leg_ids,
            scale_rad_s2=scale, boundary_s=self.cfg.swing.boundary_s, ramp_s=self.cfg.swing.ramp_s,
            contact_gate="reference_phase_only", acceleration_gate="reference_phase_only",
            acceleration_sampling="each_refreshed_physics_step", contact_sampling="control_endpoint")), flush=True)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        if self.reward_scales.get("gmr_phase_acc", 0.) >= 0:
            raise ValueError("Missing negative phase acceleration penalty")

    def _begin_physics_step(self):
        super()._begin_physics_step()
        # Snapshot after the previous step's resets; never difference across reset.
        self.phase_previous_vel.copy_(self.dof_vel)
        self.phase_square_sum.zero_()
        self.phase_all_square_sum.zero_()
        self.phase_gate_sum.zero_()
        self.phase_peak.zero_()
        self.phase_acc_cost.zero_()
        self.phase_substep_count = 0
        # Counter increments AFTER physics. >0 here equals >1 at reward time.
        self.phase_valid = self.episode_length_buf > 0
        self.phase_substep_gates = substep_phase_gates(
            self.swing_envelope, self.motion_time(), self.motion.fps,
            self.phase_physics_dt, self.cfg.control.decimation, self.phase_valid)

    def _after_physics_substep(self, substep):
        super()._after_physics_substep(substep)
        if substep != self.phase_substep_count:
            raise RuntimeError("Missing or duplicated physics-rate reward sample")
        acceleration = (self.dof_vel - self.phase_previous_vel) / self.phase_physics_dt
        self.phase_previous_vel.copy_(self.dof_vel)
        square = leg_acceleration_square(acceleration, self.phase_leg_ids)
        gate = self.phase_substep_gates[:, substep]
        self.phase_square_sum.add_(square * gate)
        self.phase_all_square_sum.add_(square * self.phase_valid[:, None])
        self.phase_gate_sum.add_(gate)
        peak = acceleration.abs().amax(dim=1) * self.phase_valid
        self.phase_peak.copy_(torch.maximum(self.phase_peak, peak))
        self.phase_substep_count += 1

    def compute_reward(self):
        if self.phase_substep_count != self.cfg.control.decimation:
            raise RuntimeError("Phase reward requires all ten refreshed physics samples")
        self.phase_acc_cost = (self.phase_square_sum.sum(dim=1)
                               / self.cfg.control.decimation / self.cfg.phase_accel.scale_rad_s2 ** 2)
        # Cache BEFORE the inherited reward summation and all reset/history writes.
        super().compute_reward()
        gate_count = self.phase_gate_sum.sum().clamp_min(1e-8)
        valid_count = self.phase_valid.sum().clamp_min(1)
        self.phase_acc_diagnostics = {
            "gmr_phase_acc_rms_rad_s2": torch.sqrt(self.phase_square_sum.sum() / gate_count),
            "gmr_physics_acc_rms_rad_s2": torch.sqrt(self.phase_all_square_sum.sum()
                / valid_count / 2 / self.cfg.control.decimation),
            "gmr_physics_acc_peak_rad_s2": self.phase_peak.max(),
            "gmr_phase_acc_gate_mean": self.phase_gate_sum.mean() / self.cfg.control.decimation,
            "gmr_phase_acc_cost": self.phase_acc_cost.mean(),
            "gmr_phase_acc_weighted_cost_before_dt": self.phase_acc_cost.mean()
                * self.reward_scales["gmr_phase_acc"] / self.dt,
        }
        for side, foot in (("left", 0), ("right", 1)):
            self.phase_acc_diagnostics["gmr_phase_acc_" + side + "_rms_rad_s2"] = torch.sqrt(
                self.phase_square_sum[:, foot].sum() / self.phase_gate_sum[:, foot].sum().clamp_min(1e-8))

    def _reward_gmr_phase_acc(self):
        return self.phase_acc_cost

    def get_command_tracking_debug(self):
        result = super().get_command_tracking_debug()
        result.update(self.phase_acc_diagnostics)
        return result
