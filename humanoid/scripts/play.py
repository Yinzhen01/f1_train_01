# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.
#
# Adapted from F1 exp_A09 play.py for X1-12DOF (exp_010_1):
# same video/CSV sync protocol; 12 DOF, no motion-ref / lumbar-torso metrics.


import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from datetime import datetime

import csv
import pygame
from threading import Thread


def _gait_state_label(left_on, right_on):
    if left_on and right_on:
        return "double"
    if not left_on and not right_on:
        return "flight"
    return "single"


def _draw_play_hud(img, env, robot_index, play_step, target_vel, current_vel_x, avg_vel,
                   left_force, right_force, l_on, r_on):
    img_h, img_w = img.shape[:2]
    base_x = img_w - 1200
    base_y = 55
    line_height = 48

    def draw_outlined_text(image, text, pos, color, scale=0.9):
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    draw_outlined_text(
        img, f"step={play_step:4d} | CMD:{target_vel:.2f} REAL:{current_vel_x:.2f} AVG:{avg_vel:.2f}",
        (base_x, base_y), (255, 255, 0), 1.0)
    l_color = (0, 255, 0) if l_on else (0, 0, 255)
    r_color = (0, 255, 0) if r_on else (0, 0, 255)
    draw_outlined_text(
        img, f"L-FOOT: {'ON ' if l_on else 'OFF'} ({left_force:.1f} N)",
        (base_x, base_y + line_height), l_color)
    draw_outlined_text(
        img, f"R-FOOT: {'ON ' if r_on else 'OFF'} ({right_force:.1f} N)",
        (base_x, base_y + line_height * 2), r_color)
    if l_on and r_on:
        state_text, state_color = "STATE: *** DOUBLE SUPPORT ***", (0, 255, 255)
    elif not l_on and not r_on:
        state_text, state_color = "STATE: >>> FLIGHT PHASE <<<", (255, 0, 255)
    else:
        state_text, state_color = "STATE: SINGLE SUPPORT", (200, 200, 200)
    draw_outlined_text(img, state_text, (base_x, base_y + line_height * 3), state_color, 1.0)
    phase = env._get_phase()[robot_index].item()
    # X1-12DOF: L/R hip_pitch, knee (indices 0,3 / 6,9)
    lp = env.dof_pos[robot_index, 0].item() * 57.3
    lk = env.dof_pos[robot_index, 3].item() * 57.3
    rp = env.dof_pos[robot_index, 6].item() * 57.3
    rk = env.dof_pos[robot_index, 9].item() * 57.3
    draw_outlined_text(
        img,
        f"ph={phase:.3f} | L hp/kn={lp:+.1f}/{lk:+.1f}  R hp/kn={rp:+.1f}/{rk:+.1f}",
        (base_x, base_y + line_height * 4), (180, 255, 180), 0.95)


# control_dt = sim.dt * decimation = 0.001 * 10
PLAY_DT = 0.01
VIDEO_RECORD_EVERY = 2

FIXED_CMD_VX = 0.5  # X1 walk

# X1-12DOF short names (dof_names order)
JOINT_SHORT_NAMES = [
    'Ll_hp', 'Ll_hr', 'Ll_hy', 'Ll_kn', 'Ll_ap', 'Ll_ar',
    'Rl_hp', 'Rl_hr', 'Rl_hy', 'Rl_kn', 'Rl_ap', 'Rl_ar',
]

DOF_SUMMARY_GROUPS = [
    ('L_leg', [0, 1, 2, 3, 4, 5], ['hip_p', 'hip_r', 'hip_y', 'knee', 'ank_p', 'ank_r']),
    ('R_leg', [6, 7, 8, 9, 10, 11], ['hip_p', 'hip_r', 'hip_y', 'knee', 'ank_p', 'ank_r']),
]

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
joystick_use = True
joystick_opened = False

if joystick_use:
    pygame.init()
    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")
    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        while not exit_flag:
            pygame.event.get()
            x_vel_cmd = -joystick.get_axis(1) * 1
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(3) * 1
            pygame.time.delay(100)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.env.episode_length_s = 1000
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.continuous_push = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_torque = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    # play: 关闭整机 lag，避免与踝辨识 tauLPF 叠加重延迟
    env_cfg.domain_rand.add_lag = False
    env_cfg.domain_rand.add_dof_lag = False
    env_cfg.domain_rand.add_imu_lag = False
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = True  # exp_heading: play 时也开启 heading 跟踪

    # --- 踝关节阶跃辨识名义值（固定点，非随机区间）---
    # pitch: coulomb0.5 viscous0.225 arm0.15; roll: coulomb0.5 viscous0 arm0.035; tauLPF=8ms
    env_cfg.domain_rand.randomize_coulomb_friction = True
    env_cfg.domain_rand.joint_coulomb_range = [0.0, 0.0]
    env_cfg.domain_rand.joint_viscous_range = [0.0, 0.0]
    env_cfg.domain_rand.ankle_pitch_joint_coulomb_range = [0.5, 0.5]
    env_cfg.domain_rand.ankle_pitch_joint_viscous_range = [0.225, 0.225]
    env_cfg.domain_rand.ankle_roll_joint_coulomb_range = [0.5, 0.5]
    env_cfg.domain_rand.ankle_roll_joint_viscous_range = [0.0, 0.0]

    env_cfg.domain_rand.randomize_joint_armature = True
    env_cfg.domain_rand.randomize_joint_armature_each_joint = True
    for _ji in range(1, 13):
        setattr(env_cfg.domain_rand, f'joint_{_ji}_armature_range', [0.0, 0.0])
    env_cfg.domain_rand.joint_5_armature_range = [0.15, 0.15]   # left_ankle_pitch
    env_cfg.domain_rand.joint_6_armature_range = [0.035, 0.035]  # left_ankle_roll
    env_cfg.domain_rand.joint_11_armature_range = [0.15, 0.15]  # right_ankle_pitch
    env_cfg.domain_rand.joint_12_armature_range = [0.035, 0.035]  # right_ankle_roll

    env_cfg.domain_rand.enable_delivery = True
    env_cfg.domain_rand.delivery_tau_d = 0.008
    env_cfg.domain_rand.delivery_joint_ids = [4, 5, 10, 11]

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = -1
    train_cfg.runner.checkpoint = -1
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name,
            '0_exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env_cfg.sim.dt * env_cfg.control.decimation)
    robot_index = 0
    stop_state_log = 1000
    csv_log_start = 0
    csv_log_end = stop_state_log - 1
    num_dof = env_cfg.env.num_actions  # 12

    assert num_dof == 12, f"exp_010_1 expects 12 DOF, got {num_dof}"
    assert len(env.dof_names) == 12, f"dof_names len={len(env.dof_names)}"

    print(f"[play] exp_010_add_ankle_parameter  X1-12DOF  fixed cmd={FIXED_CMD_VX} m/s")
    print(f"[play] csv_log steps {csv_log_start}–{csv_log_end}")
    print("[play] ankle ID plant (nominal):")
    print(f"  pitch: Fc=0.5 B=0.225 arm=0.15 | roll: Fc=0.5 B=0 arm=0.035")
    print(f"  delivery LPF: enable={env.cfg.domain_rand.enable_delivery} "
          f"tau_d={env.cfg.domain_rand.delivery_tau_d}s ids={list(env.cfg.domain_rand.delivery_joint_ids)}")
    print(f"  coulomb_on={env.cfg.domain_rand.randomize_coulomb_friction} "
          f"armature_on={env.cfg.domain_rand.randomize_joint_armature} "
          f"add_lag={env.cfg.domain_rand.add_lag}")
    print("[dof] Isaac order:")
    for i, name in enumerate(env.dof_names):
        print(f"  [{i:2d}] {name:32s} → {JOINT_SHORT_NAMES[i]}")

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        custom_save_path = "/personal/train-more"
        run_name_str = args.run_name if args.run_name is not None else "test"
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name_str}.mp4"
        video_filepath = os.path.join(custom_save_path, file_name)
        os.makedirs(custom_save_path, exist_ok=True)
        print(f"[VIDEO] Recording to: {video_filepath}")
        video = cv2.VideoWriter(video_filepath, fourcc, 50.0, (1920, 1080))

    obs = env.get_observations()
    if FIX_COMMAND:
        env.commands[:, 0] = FIXED_CMD_VX
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0
        print(f"[play] cold start cmd={env.commands[0, 0].item():.2f} m/s")

    frame_count = 0
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    vel_sum = 0.0
    step_accum = 0

    all_dof_log = [[] for _ in range(num_dof)]  # (act_rad, des_rad, err_rad)
    body_yaw_log = []

    _csv_dir = "/personal/train-more"
    os.makedirs(_csv_dir, exist_ok=True)
    _csv_path = os.path.join(_csv_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_joints.csv")
    _csv_headers = [
        'step', 'video_frame', 'time_s', 'phase',
        'base_roll', 'base_pitch', 'base_yaw',
        'cmd_x', 'base_vel_x', 'base_vel_y', 'base_pos_x', 'base_pos_y',
        'foot_fz_l', 'foot_fz_r', 'gait_state',
    ]
    for _sn in JOINT_SHORT_NAMES:
        # X1 无 motion ref：des = PD 目标 (default + action*scale)
        _csv_headers += [f'{_sn}_des', f'{_sn}_act', f'{_sn}_err']
    _csv_file = open(_csv_path, 'w', newline='')
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(_csv_headers)
    print(f"[CSV] 开始记录到: {_csv_path}")
    print(f"[sync] play_dt={PLAY_DT}s  video_fps=50  record_every={VIDEO_RECORD_EVERY} render ticks")
    print(f"[sync] CSV step N 时间 ≈ N×{PLAY_DT}s；step=-1 为 reset 标定")

    _video_frame_idx = 0
    _last_actions = torch.zeros(env.num_envs, num_dof, device=env.device)

    def _foot_contacts():
        _fz_l = env.contact_forces[robot_index, env.feet_indices[0].item(), 2].item()
        _fz_r = env.contact_forces[robot_index, env.feet_indices[1].item(), 2].item()
        return _fz_l, _fz_r, _fz_l > 1.0, _fz_r > 1.0

    def _joint_des_act_err(actions_t):
        """des/act/err in rad for 12 DOF."""
        out = []
        for ji in range(num_dof):
            act = env.dof_pos[robot_index, ji].item()
            des = (env.default_dof_pos[robot_index, ji]
                   + actions_t[robot_index, ji] * env.cfg.control.action_scale).item()
            out.append((des, act, act - des))
        return out

    def _build_csv_row(play_step, vf_idx, actions_t):
        _phase = env._get_phase()[robot_index].item()
        _body_roll_deg = env.base_euler_xyz[robot_index, 0].item() * 57.3
        _body_pitch_deg = env.base_euler_xyz[robot_index, 1].item() * 57.3
        _body_yaw_deg = env.base_euler_xyz[robot_index, 2].item() * 57.3
        _fz_l, _fz_r, _l_on, _r_on = _foot_contacts()
        _gait = _gait_state_label(_l_on, _r_on)
        _base_vx = env.root_states[robot_index, 7].item()
        _base_vy = env.root_states[robot_index, 8].item()
        _base_px = env.root_states[robot_index, 0].item()
        _base_py = env.root_states[robot_index, 1].item()
        _cmd_x = env.commands[robot_index, 0].item()
        _time_s = max(play_step, 0) * PLAY_DT
        _row = [
            play_step,
            vf_idx if vf_idx > 0 else '',
            f"{_time_s:.4f}",
            f"{_phase:.4f}",
            f"{_body_roll_deg:.3f}", f"{_body_pitch_deg:.3f}", f"{_body_yaw_deg:.3f}",
            f"{_cmd_x:.4f}", f"{_base_vx:.4f}", f"{_base_vy:.4f}",
            f"{_base_px:.4f}", f"{_base_py:.4f}",
            f"{_fz_l:.2f}", f"{_fz_r:.2f}", _gait,
        ]
        for des, act, err in _joint_des_act_err(actions_t):
            _row += [f"{des * 57.3:.3f}", f"{act * 57.3:.3f}", f"{err * 57.3:.3f}"]
        return _row, _phase, _body_yaw_deg

    def _log_csv_step(play_step, vf_idx, actions_t):
        _row, _phase, _byaw = _build_csv_row(play_step, vf_idx, actions_t)
        _csv_writer.writerow(_row)
        _lp = env.dof_pos[robot_index, 0].item() * 57.3
        _lk = env.dof_pos[robot_index, 3].item() * 57.3
        print(f"[S{play_step:4d}|ph={_phase:.3f}|vf={vf_idx if vf_idx > 0 else '-'}] "
              f"L_hp/kn={_lp:+.1f}/{_lk:+.1f}°  base_yaw={_byaw:+.1f}°")
        if play_step < 0:
            return _row
        for ji, (des, act, err) in enumerate(_joint_des_act_err(actions_t)):
            all_dof_log[ji].append((act, des, err))
        body_yaw_log.append((play_step, _byaw))
        return _row

    def _capture_video_frame(play_step, avg_vel):
        nonlocal _video_frame_idx, frame_count
        if not RENDER:
            return -1
        frame_count += 1
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        if frame_count % VIDEO_RECORD_EVERY != 0:
            return -1
        img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (1080, 1920, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _fz_l, _fz_r, _l_on, _r_on = _foot_contacts()
        _draw_play_hud(
            img, env, robot_index, play_step,
            env.commands[0, 0].item(), env.base_lin_vel[0, 0].item(), avg_vel,
            _fz_l, _fz_r, _l_on, _r_on)
        video.write(img[..., :3])
        _video_frame_idx += 1
        return _video_frame_idx

    # reset 标定帧（A09 同协议）
    _vf_reset = _capture_video_frame(-1, 0.0)
    _log_csv_step(-1, vf_idx=_vf_reset, actions_t=_last_actions)
    if _vf_reset > 0:
        print(f"[sync] reset 标定帧 → video_frame={_vf_reset}  step=-1")

    for i in range(10 * stop_state_log):
        actions = policy(obs.detach())
        _last_actions = actions.detach()

        if FIX_COMMAND:
            env.commands[:, 0] = FIXED_CMD_VX
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0
        else:
            env.commands[:, 0] = x_vel_cmd
            env.commands[:, 1] = y_vel_cmd
            env.commands[:, 2] = yaw_vel_cmd
            env.commands[:, 3] = 0.0

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        current_vel_x = env.base_lin_vel[0, 0].item()
        vel_sum += current_vel_x
        step_accum += 1
        avg_vel = vel_sum / step_accum if step_accum > 0 else 0.0
        _vf = _capture_video_frame(i, avg_vel)

        if csv_log_start <= i <= csv_log_end:
            _log_csv_step(i, vf_idx=_vf, actions_t=_last_actions)
            logger.log_states(dict={
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'video_frame': _vf,
            })

        elif i == stop_state_log:
            logger.plot_states()
            import numpy as _np
            print("\n" + "=" * 68)
            print("  全关节跟踪误差汇总  (err = act - des, 单位: °)")
            print("=" * 68)
            for _gname, _gidx, _gnames in DOF_SUMMARY_GROUPS:
                print(f"\n{'─' * 20} {_gname} {'─' * 20}")
                print(f"  {'DOF':>3}  {'关节':8}  {'mean':>7}  {'std':>6}  {'min':>7}  {'max':>7}")
                for _di, _dn in zip(_gidx, _gnames):
                    if not all_dof_log[_di]:
                        continue
                    _errs = _np.array([x[2] for x in all_dof_log[_di]]) * 57.3
                    print(f"  {_di:3d}  {_dn:8s}  {_np.mean(_errs):+7.1f}°  {_np.std(_errs):6.1f}°  "
                          f"{_np.min(_errs):+7.1f}°  {_np.max(_errs):+7.1f}°")
            if body_yaw_log:
                _byaws = _np.array([x[1] for x in body_yaw_log])
                print(f"\n  body_yaw: mean={_np.mean(_byaws):+.1f}°  max={_np.max(_byaws):+.1f}°")
            print("=" * 68)

        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

    _csv_file.close()
    print(f"[CSV] 已保存至: {_csv_path}")
    if RENDER:
        print(f"[sync] 视频总帧数={_video_frame_idx}")
        video.release()
        print(f"[VIDEO] 已保存至: {video_filepath}")


if __name__ == '__main__':
    EXPORT_POLICY = False
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
