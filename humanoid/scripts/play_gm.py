# play_gm.py — GM 平台专用 play 脚本（无 pygame / 无头渲染 / OSS 下载 / SDK 上传）
# 基于 play.py 改造，适配 Gradmotion 云端环境。

import os
import csv
import time
import zipfile
import pickle
import urllib.request
import numpy as np
import cv2
import torch
from datetime import datetime
from isaacgym import gymapi
from isaacgym.torch_utils import *
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger

# ============================================================
# 配置：填入训练任务的 model 下载链接（flux task model list 的 policUrlDown）
# ============================================================
FALLBACK_CHECKPOINT_URL = "https://limx-gradmotion.oss-cn-beijing.aliyuncs.com/upload%2F2026%2F8%2F12%2Fmodel_5000_20260812191726A103.pt?OSSAccessKeyId=LTAI5tMec8RQN1nZuRkVMgxz&Expires=1787191043&Signature=Nr%2BWQ0RpYf0UQSLQqG809OqeMOE%3D"

# ============================================================
# 常量（与 play.py 一致）
# ============================================================
PLAY_DT = 0.01
VIDEO_RECORD_EVERY = 2
FIXED_CMD_VX = 0.5

JOINT_SHORT_NAMES = [
    'Ll_hp', 'Ll_hr', 'Ll_hy', 'Ll_kn', 'Ll_ap', 'Ll_ar',
    'Rl_hp', 'Rl_hr', 'Rl_hy', 'Rl_kn', 'Rl_ap', 'Rl_ar',
]

DOF_SUMMARY_GROUPS = [
    ('L_leg', [0, 1, 2, 3, 4, 5], ['hip_p', 'hip_r', 'hip_y', 'knee', 'ank_p', 'ank_r']),
    ('R_leg', [6, 7, 8, 9, 10, 11], ['hip_p', 'hip_r', 'hip_y', 'knee', 'ank_p', 'ank_r']),
]


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
    lp = env.dof_pos[robot_index, 0].item() * 57.3
    lk = env.dof_pos[robot_index, 3].item() * 57.3
    rp = env.dof_pos[robot_index, 6].item() * 57.3
    rk = env.dof_pos[robot_index, 9].item() * 57.3
    draw_outlined_text(
        img,
        f"ph={phase:.3f} | L hp/kn={lp:+.1f}/{lk:+.1f}  R hp/kn={rp:+.1f}/{rk:+.1f}",
        (base_x, base_y + line_height * 4), (180, 255, 180), 0.95)


def _find_checkpoint():
    """搜索本地 checkpoint，找不到则从 OSS 下载。"""
    # 1. 搜索常见路径
    search_dirs = [
        os.path.join(LEGGED_GYM_ROOT_DIR, "logs"),
        "/personal",
        "/workspace",
        os.getcwd(),
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            for f in sorted(files, reverse=True):
                if f.startswith("model_") and f.endswith(".pt") and "exported" not in root:
                    path = os.path.join(root, f)
                    print(f"[play_gm] Found checkpoint: {path}")
                    return path

    # 2. 从 OSS 下载
    if FALLBACK_CHECKPOINT_URL:
        print(f"[play_gm] No local checkpoint found, downloading from OSS...")
        dest = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "downloaded_model.pt")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(FALLBACK_CHECKPOINT_URL, dest)
        print(f"[play_gm] Downloaded to: {dest}")
        return dest

    raise FileNotFoundError("No checkpoint found and FALLBACK_CHECKPOINT_URL is empty")


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
    env_cfg.domain_rand.add_lag = False
    env_cfg.domain_rand.add_dof_lag = False
    env_cfg.domain_rand.add_imu_lag = False
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = True

    # 踝关节阶跃辨识名义值
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
    env_cfg.domain_rand.joint_5_armature_range = [0.15, 0.15]
    env_cfg.domain_rand.joint_6_armature_range = [0.035, 0.035]
    env_cfg.domain_rand.joint_11_armature_range = [0.15, 0.15]
    env_cfg.domain_rand.joint_12_armature_range = [0.035, 0.035]

    env_cfg.domain_rand.enable_delivery = True
    env_cfg.domain_rand.delivery_tau_d = 0.008
    env_cfg.domain_rand.delivery_joint_ids = [4, 5, 10, 11]

    # 无头渲染：保持 GPU 相机，不开 viewer
    env_cfg.env.enable_headless_render = True

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # 加载 checkpoint
    ckpt_path = _find_checkpoint()
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = os.path.dirname(ckpt_path)
    train_cfg.runner.checkpoint = os.path.basename(ckpt_path)
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    logger = Logger(env_cfg.sim.dt * env_cfg.control.decimation)
    robot_index = 0
    stop_state_log = 1000
    csv_log_start = 0
    csv_log_end = stop_state_log - 1
    num_dof = env_cfg.env.num_actions

    assert num_dof == 12, f"exp_010 expects 12 DOF, got {num_dof}"

    # ============================================================
    # GM SDK 路径
    # ============================================================
    exp_name = train_cfg.runner.experiment_name
    logs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", exp_name)
    os.makedirs(logs_dir, exist_ok=True)

    # 视频输出路径（SDK 自动扫描）
    video_filepath = os.path.join(logs_dir, "play_output.mp4")

    # CSV 打包路径（SDK 上传）
    gm_play_dir = os.path.join(logs_dir, "gm_play")
    os.makedirs(gm_play_dir, exist_ok=True)

    # diag 输出路径（SDK 上传）
    diag_dir = os.path.join(logs_dir, "play_output")
    os.makedirs(diag_dir, exist_ok=True)

    print(f"[play_gm] exp_010 X1-12DOF  fixed cmd={FIXED_CMD_VX} m/s")
    print(f"[play_gm] video → {video_filepath}")
    print(f"[play_gm] csv pt → {gm_play_dir}/model_isaac_csv.pt")
    print(f"[play_gm] diag pt → {diag_dir}/model_diag.pt")

    # 相机
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
    video = cv2.VideoWriter(video_filepath, fourcc, 50.0, (1920, 1080))
    print(f"[VIDEO] Recording to: {video_filepath}")

    obs = env.get_observations()
    env.commands[:, 0] = FIXED_CMD_VX
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    env.commands[:, 3] = 0.0
    print(f"[play_gm] cold start cmd={env.commands[0, 0].item():.2f} m/s")

    frame_count = 0
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    vel_sum = 0.0
    step_accum = 0

    all_dof_log = [[] for _ in range(num_dof)]
    body_yaw_log = []

    # CSV 到内存（先收集，结尾打包）
    csv_rows = []
    csv_headers = [
        'step', 'video_frame', 'time_s', 'phase',
        'base_roll', 'base_pitch', 'base_yaw',
        'cmd_x', 'base_vel_x', 'base_vel_y', 'base_pos_x', 'base_pos_y',
        'foot_fz_l', 'foot_fz_r', 'gait_state',
    ]
    for _sn in JOINT_SHORT_NAMES:
        csv_headers += [f'{_sn}_des', f'{_sn}_act', f'{_sn}_err']

    _video_frame_idx = 0
    _last_actions = torch.zeros(env.num_envs, num_dof, device=env.device)

    def _foot_contacts():
        _fz_l = env.contact_forces[robot_index, env.feet_indices[0].item(), 2].item()
        _fz_r = env.contact_forces[robot_index, env.feet_indices[1].item(), 2].item()
        return _fz_l, _fz_r, _fz_l > 1.0, _fz_r > 1.0

    def _joint_des_act_err(actions_t):
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
        csv_rows.append(_row)
        _lp = env.dof_pos[robot_index, 0].item() * 57.3
        _lk = env.dof_pos[robot_index, 3].item() * 57.3
        print(f"[S{play_step:4d}|ph={_phase:.3f}|vf={vf_idx if vf_idx > 0 else '-'}] "
              f"L_hp/kn={_lp:+.1f}/{_lk:+.1f}  base_yaw={_byaw:+.1f}")
        if play_step < 0:
            return _row
        for ji, (des, act, err) in enumerate(_joint_des_act_err(actions_t)):
            all_dof_log[ji].append((act, des, err))
        body_yaw_log.append((play_step, _byaw))
        return _row

    def _capture_video_frame(play_step, avg_vel):
        nonlocal _video_frame_idx, frame_count
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

    # reset 标定帧
    _vf_reset = _capture_video_frame(-1, 0.0)
    _log_csv_step(-1, vf_idx=_vf_reset, actions_t=_last_actions)
    if _vf_reset > 0:
        print(f"[sync] reset calibration frame -> video_frame={_vf_reset}  step=-1")

    for i in range(10 * stop_state_log):
        actions = policy(obs.detach())
        _last_actions = actions.detach()

        env.commands[:, 0] = FIXED_CMD_VX
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
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
            print("  Joint tracking error summary  (err = act - des, unit: deg)")
            print("=" * 68)
            for _gname, _gidx, _gnames in DOF_SUMMARY_GROUPS:
                print(f"\n{'--' * 10} {_gname} {'--' * 10}")
                print(f"  {'DOF':>3}  {'joint':8}  {'mean':>7}  {'std':>6}  {'min':>7}  {'max':>7}")
                for _di, _dn in zip(_gidx, _gnames):
                    if not all_dof_log[_di]:
                        continue
                    _errs = _np.array([x[2] for x in all_dof_log[_di]]) * 57.3
                    print(f"  {_di:3d}  {_dn:8s}  {_np.mean(_errs):+7.1f}  {_np.std(_errs):6.1f}  "
                          f"{_np.min(_errs):+7.1f}  {_np.max(_errs):+7.1f}")
            if body_yaw_log:
                _byaws = _np.array([x[1] for x in body_yaw_log])
                print(f"\n  body_yaw: mean={_np.mean(_byaws):+.1f}  max={_np.max(_byaws):+.1f}")
            print("=" * 68)

        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

    # ============================================================
    # 保存产物
    # ============================================================
    # 1. 释放视频
    print(f"[sync] total video frames={_video_frame_idx}")
    video.release()
    print(f"[VIDEO] Saved to: {video_filepath}")

    # 2. CSV 写入临时文件，打包为 model_isaac_csv.pt
    import io
    csv_buf = io.StringIO()
    csv_writer = csv.writer(csv_buf)
    csv_writer.writerow(csv_headers)
    for row in csv_rows:
        csv_writer.writerow(row)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
    csv_buf.close()
    print(f"[CSV] {len(csv_rows)} rows collected")

    csv_pt_path = os.path.join(gm_play_dir, "model_isaac_csv.pt")
    with zipfile.ZipFile(csv_pt_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model_isaac_csv/data.pkl", pickle.dumps({"bytes": csv_bytes}))
    print(f"[CSV] Packaged to: {csv_pt_path}")

    # 3. diag pt
    diag_data = {
        'dof_log': all_dof_log,
        'body_yaw_log': body_yaw_log,
        'joint_short_names': JOINT_SHORT_NAMES,
    }
    diag_pt_path = os.path.join(diag_dir, "model_diag.pt")
    torch.save(diag_data, diag_pt_path)
    print(f"[DIAG] Saved to: {diag_pt_path}")

    # 4. 等待 SDK 上传
    print("[play_gm] Waiting 60s for SDK file upload...")
    time.sleep(60)
    print("[play_gm] Done.")


if __name__ == '__main__':
    args = get_args()
    play(args)
