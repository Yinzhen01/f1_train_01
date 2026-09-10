"""CPU tensor-math tests; deliberately not a substitute for Isaac Gym smoke."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch


def quat_apply(q, v):
    uv = torch.cross(q[:, :3], v, dim=-1)
    return v + 2 * (q[:, 3:] * uv + torch.cross(q[:, :3], uv, dim=-1))


def quat_inverse_apply(q, v):
    inverse = torch.cat((-q[:, :3], q[:, 3:]), dim=-1)
    return quat_apply(inverse, v)


def quat_mul(a, b):
    xyz = a[:, 3:] * b[:, :3] + b[:, 3:] * a[:, :3] + torch.cross(a[:, :3], b[:, :3], dim=-1)
    w = a[:, 3:] * b[:, 3:] - (a[:, :3] * b[:, :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


class StubBase:
    def get_command_tracking_debug(self):
        return {"contact/foot_slip_speed_mean": torch.tensor(-999.)}


def environment_class():
    path = Path(__file__).resolve().parents[1] / "humanoid/envs/x1/x1_gmr_clip_env.py"
    definition = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef))
    scope = {"X1DHStandEnv": StubBase, "torch": torch, "quat_apply": quat_apply,
             "quat_rotate_inverse": quat_inverse_apply, "quat_mul": quat_mul}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), scope)
    return scope["X1GMRClipEnv"]


class GMRMathTest(unittest.TestCase):
    def setUp(self):
        cls = environment_class()
        self.env = cls.__new__(cls)
        e = self.env
        e.num_envs = 2
        e.feet_indices = torch.tensor([0, 1])
        e.motion = SimpleNamespace(sole_local=torch.zeros(2, 3), duration=4.6)
        e.reference_start = torch.tensor([0., 4.])
        e.episode_length_buf = torch.tensor([0, 60])
        e.dt = .01
        e.rigid_state = torch.zeros(2, 2, 13)
        e.rigid_state[:, :, 6] = 1.
        e.contact_forces = torch.zeros(2, 2, 3)
        e.contact_forces[:, :, 2] = 20.
        e.root_states = torch.zeros(2, 13)
        e.root_states[:, 6] = 1.
        e.base_quat = e.root_states[:, 3:7]
        e.env_origins = torch.zeros(2, 3)
        e.ref_dof_pos = torch.zeros(2, 12)
        e.dof_pos = torch.zeros(2, 12)
        e.dof_vel = torch.zeros(2, 12)
        e.reference = dict(dof_vel=torch.zeros(2, 12), root_pos=torch.zeros(2, 3), root_vel=torch.zeros(2, 3),
                           root_ang_vel=torch.zeros(2, 3), root_quat=e.base_quat.clone(), foot_contact=torch.ones(2, 2),
                           foot_pos_base=torch.zeros(2, 2, 3), foot_quat_base=e.rigid_state[:, :, 3:7].clone())

    def test_exact_reference_has_unit_tracking_rewards(self):
        for term in ("joint_pos", "joint_vel", "foot_pos", "foot_rotation", "contact", "root_pos", "root_rotation", "velocity"):
            torch.testing.assert_close(getattr(self.env, "_reward_gmr_" + term)(), torch.ones(2))

    def test_slip_uses_linear_not_angular_velocity(self):
        self.env.rigid_state[:, :, 10:13] = 10.
        torch.testing.assert_close(self.env._reward_gmr_slip(), torch.zeros(2))
        self.assertEqual(float(self.env.get_command_tracking_debug()["contact/foot_slip_speed_mean"]), 0.)
        self.env.rigid_state[:, :, 7] = .5
        torch.testing.assert_close(self.env._reward_gmr_slip(), torch.full((2,), .5))
        self.assertEqual(float(self.env.get_command_tracking_debug()["contact/foot_slip_speed_mean"]), .5)

    def test_swing_slip_is_not_penalized(self):
        self.env.rigid_state[:, :, 7:10] = 1.
        self.env.contact_forces.zero_()
        torch.testing.assert_close(self.env._reward_gmr_slip(), torch.zeros(2))

    def test_phase_uses_reference_start_and_distinguishes_end(self):
        torch.testing.assert_close(self.env.motion_time(), torch.tensor([0., 4.6]))
        torch.testing.assert_close(self.env._get_phase(), torch.tensor([0., .5]))


if __name__ == "__main__":
    unittest.main()
