"""1 kHz acceleration/contact observation; no change to integration or PD."""
import json
import math

import torch

from humanoid.gmr_cycle import acceleration_weights, weighted_acceleration_cost, impact_cost
from humanoid.gmr_phase_accel import leg_joint_indices, substep_phase_gates
from humanoid.gmr_swing import swing_envelope
from .x1_gmr_accel_env import X1GMRAccelEnv


class X1GMRCycleEnv(X1GMRAccelEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self.cycle_h = float(self.sim_params.dt)
        if (not math.isclose(self.cycle_h, .001, abs_tol=1e-9)
                or self.cfg.control.decimation != 10 or self.sim_params.substeps != 1
                or not math.isclose(self.dt, .01, abs_tol=1e-9)):
            raise ValueError("Cycle experiment requires 1kHz physics / 100Hz control / substeps=1")
        c = self.cfg.cycle
        self.cycle_ids = leg_joint_indices(self.dof_names)
        self.cycle_joint_weights = torch.tensor(c.joint_weights, device=self.device)
        if (self.cycle_joint_weights.shape != (6,) or not torch.isfinite(self.cycle_joint_weights).all()
                or not torch.all(self.cycle_joint_weights > 0)
                or not math.isfinite(c.acceleration_scale_rad_s2) or c.acceleration_scale_rad_s2 <= 0):
            raise ValueError("Invalid cycle acceleration normalization")
        props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        self.cycle_body_weight = sum(p.mass for p in props) * abs(float(self.sim_params.gravity.z))
        if not math.isfinite(self.cycle_body_weight) or self.cycle_body_weight <= 0:
            raise ValueError("Invalid runtime body weight")
        if not 0 < c.swing_force_body_weights < c.stance_force_body_weights:
            raise ValueError("Invalid reference force allowance")
        self.cycle_swing = swing_envelope(self.motion.data['foot_contact'], self.motion.fps,
                                          c.boundary_s, c.ramp_s)
        self.cycle_stance = swing_envelope(1. - self.motion.data['foot_contact'], self.motion.fps,
                                           c.boundary_s, c.ramp_s)
        self.cycle_prev_vel = torch.zeros_like(self.dof_vel)
        self.cycle_acc_cost = torch.zeros(self.num_envs, device=self.device)
        self.cycle_impact_cost = torch.zeros_like(self.cycle_acc_cost)
        self.cycle_acc_squares = torch.zeros(self.num_envs, 12, device=self.device)
        self.cycle_gate_squares = torch.zeros(self.num_envs, 3, device=self.device)
        self.cycle_gate_counts = torch.zeros_like(self.cycle_gate_squares)
        self.cycle_acc_peak = torch.zeros(self.num_envs, device=self.device)
        self.cycle_force_peak = torch.zeros_like(self.cycle_acc_peak)
        self.cycle_diagnostics = {}
        # Only first environment, one control interval: retained for deterministic
        # rollout capture. GPU-to-CPU transfer is done only by the eval observer.
        self.cycle_trace = {k: torch.zeros((10, n), device=self.device) for k, n in
                            [('time', 1), ('valid', 1), ('dof_vel', 12), ('acc', 12),
                             ('torque', 12), ('force_z', 2), ('swing_gate', 2), ('acc_weight', 2)]}
        self.cycle_substep_count = 0
        print('[gmr-cycle-runtime] ' + json.dumps(dict(
            physics_dt=self.cycle_h, control_dt=self.dt, body_weight_n=self.cycle_body_weight,
            leg_ids=self.cycle_ids, contact_sampling='each_refreshed_physics_step',
            acceleration_sampling='each_refreshed_physics_step',
            settings={k: getattr(c, k) for k in dir(c) if not k.startswith('_')})), flush=True)

    def _begin_physics_step(self):
        super()._begin_physics_step()
        self.cycle_prev_vel.copy_(self.dof_vel)
        for tensor in (self.cycle_acc_cost, self.cycle_impact_cost, self.cycle_acc_squares,
                       self.cycle_gate_squares, self.cycle_gate_counts, self.cycle_acc_peak,
                       self.cycle_force_peak):
            tensor.zero_()
        self.cycle_valid = self.episode_length_buf > 0
        self.cycle_start_time = self.motion_time().clone()
        args = (self.cycle_start_time, self.motion.fps, self.cycle_h,
                self.cfg.control.decimation, self.cycle_valid)
        self.cycle_swing_steps = substep_phase_gates(self.cycle_swing, *args)
        self.cycle_stance_steps = substep_phase_gates(self.cycle_stance, *args)
        self.cycle_weights = acceleration_weights(self.cycle_swing_steps, self.cycle_stance_steps,
            self.cfg.cycle.acceleration_floor) * self.cycle_valid[:, None, None]
        self.cycle_substep_count = 0

    def _after_physics_substep(self, substep):
        super()._after_physics_substep(substep)
        if substep != self.cycle_substep_count:
            raise RuntimeError("Missing or duplicate cycle physics sample")
        # Refresh after gym.simulate, alongside the already refreshed DOF state.
        self.gym.refresh_net_contact_force_tensor(self.sim)
        acc = (self.dof_vel - self.cycle_prev_vel) / self.cycle_h
        self.cycle_prev_vel.copy_(self.dof_vel)
        force = self.contact_forces[:, self.feet_indices, 2]
        swing, stance = self.cycle_swing_steps[:, substep], self.cycle_stance_steps[:, substep]
        weights = self.cycle_weights[:, substep]
        c, d = self.cfg.cycle, self.cfg.control.decimation
        self.cycle_acc_cost.add_(weighted_acceleration_cost(acc, self.cycle_ids, weights,
            self.cycle_joint_weights, c.acceleration_scale_rad_s2) / d)
        self.cycle_impact_cost.add_(impact_cost(force, swing, self.cycle_body_weight,
            c.stance_force_body_weights, c.swing_force_body_weights) * self.cycle_valid / d)
        squares = acc.square() * self.cycle_valid[:, None]
        self.cycle_acc_squares.add_(squares)
        leg_square = torch.stack([squares[:, ids].mean(-1) for ids in self.cycle_ids], -1)
        for i, gate in enumerate((stance, swing, (1. - torch.maximum(stance, swing)) * self.cycle_valid[:, None])):
            self.cycle_gate_squares[:, i].add_((leg_square * gate).sum(-1))
            self.cycle_gate_counts[:, i].add_(gate.sum(-1))
        self.cycle_acc_peak.copy_(torch.maximum(self.cycle_acc_peak, acc.abs().amax(-1) * self.cycle_valid))
        self.cycle_force_peak.copy_(torch.maximum(self.cycle_force_peak, force.clamp_min(0.).amax(-1) * self.cycle_valid))
        trace = dict(time=(self.cycle_start_time + (substep + 1) * self.cycle_h)[:, None],
            valid=self.cycle_valid[:, None], dof_vel=self.dof_vel, acc=acc, torque=self.torques,
            force_z=force, swing_gate=swing, acc_weight=weights)
        for key, value in trace.items():
            self.cycle_trace[key][substep].copy_(value[0])
        self.cycle_substep_count += 1

    def compute_reward(self):
        if self.cycle_substep_count != self.cfg.control.decimation:
            raise RuntimeError("Cycle reward requires all ten physics samples")
        super().compute_reward()  # Costs have already been accumulated before resets.
        n = self.cycle_valid.sum().clamp_min(1) * self.cfg.control.decimation
        diag = {
            'gmr_cycle_acc_rms_1khz': torch.sqrt(self.cycle_acc_squares.sum() / n / 12),
            'gmr_cycle_acc_peak_1khz': self.cycle_acc_peak.max(),
            'gmr_cycle_force_peak_n_1khz': self.cycle_force_peak.max(),
            'gmr_cycle_acc_raw': self.cycle_acc_cost.mean(),
            'gmr_cycle_impact_raw': self.cycle_impact_cost.mean(),
        }
        for i, label in enumerate(('stance', 'swing', 'transition')):
            diag['gmr_cycle_' + label + '_acc_rms_1khz'] = torch.sqrt(
                self.cycle_gate_squares[:, i].sum() / self.cycle_gate_counts[:, i].sum().clamp_min(1e-8))
        for label, ids in zip(('left', 'right'), self.cycle_ids):
            diag['gmr_cycle_' + label + '_ankle_roll_acc_rms_1khz'] = torch.sqrt(
                self.cycle_acc_squares[:, ids[-1]].sum() / n)
        self.cycle_diagnostics = diag

    def _reward_gmr_cycle_acc(self):
        return self.cycle_acc_cost

    def _reward_gmr_cycle_impact(self):
        return self.cycle_impact_cost

    def get_command_tracking_debug(self):
        result = super().get_command_tracking_debug()
        result.update(self.cycle_diagnostics)
        return result
