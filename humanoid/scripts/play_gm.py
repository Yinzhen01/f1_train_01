# GM cloud playback script for X1 Isaac Gym simulation
# Features:
#   - No pygame dependency (headless cloud)
#   - Auto-loads checkpoint from /personal/ or logs/
#   - Headless rendering with GPU camera sensors
#   - Packages video as model_video.pt for GM SDK upload
#   - Outputs diagnostic trajectory data

import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import cv2
import csv
from isaacgym import gymapi
import torch
from datetime import datetime

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

# Fallback: download checkpoint from OSS if not found locally
FALLBACK_CHECKPOINT_URL = "https://limx-gradmotion.oss-cn-beijing.aliyuncs.com/upload%2F2026%2F8%2F13%2Fmodel_5000_20260813121400A315.pt?OSSAccessKeyId=LTAI5tMec8RQN1nZuRkVMgxz&Expires=1787222353&Signature=rnEcXxn1dzAVReNKWFYgZ0DvlZI%3D"


def _gait_state_label(left_on, right_on):
    if left_on and right_on:
        return "double"
    if not left_on and not right_on:
        return "flight"
    return "single"


def _draw_play_hud(img, env, robot_index, play_step, target_vel, current_vel_x, avg_vel,
                   left_force, right_force, l_on, r_on):
    """视频 HUD overlay（与 play.py 一致）：速度指令/足端接触/步态相位/髋膝角度"""
    img_h, img_w = img.shape[:2]
    # 分辨率自适应：1080p 与 play.py 布局一致；低分辨率回退时压缩到左上角
    if img_w >= 1600:
        base_x, base_y, line_height = img_w - 1200, 55, 48
    else:
        base_x, base_y, line_height = 30, 45, 32

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
    try:
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
    except Exception as e:
        print(f"[play_gm] HUD phase info skipped: {e}")


def find_checkpoint():
    """Search broadly for model_*.pt checkpoint"""
    search_dirs = [
        "/personal",
        "/workspace",
        os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand"),
        os.getcwd(),
    ]
    for d in search_dirs:
        if not os.path.exists(d):
            print(f"[play_gm] Search dir does not exist: {d}")
            continue
        # List all files in the directory for debugging
        try:
            all_files = os.listdir(d)
            pt_files = [f for f in all_files if f.endswith('.pt')]
            if pt_files:
                print(f"[play_gm] .pt files in {d}: {pt_files}")
            else:
                print(f"[play_gm] No .pt files directly in {d} (total files: {len(all_files)})")
        except Exception as e:
            print(f"[play_gm] Cannot list {d}: {e}")
            continue
        # Search for model_*.pt recursively
        models = sorted(glob.glob(os.path.join(d, "**", "model_*.pt"), recursive=True))
        # Also try non-recursive
        models += sorted(glob.glob(os.path.join(d, "model_*.pt")))
        # Also try any .pt file if no model_*.pt found
        if not models:
            models = sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
            models += sorted(glob.glob(os.path.join(d, "*.pt")))
        # Exclude deploy/video/diag files
        models = [m for m in models if "deploy" not in m and "video" not in m and "diag" not in m]
        if models:
            print(f"[play_gm] Found checkpoint: {models[-1]}")
            return models[-1]  # Return latest
    # Fallback: download from OSS if not found locally
    print("[play_gm] No local checkpoint found, downloading from OSS...")
    download_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(download_dir, exist_ok=True)
    download_path = os.path.join(download_dir, "model_3000.pt")
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", download_path, FALLBACK_CHECKPOINT_URL],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(download_path):
            print(f"[play_gm] Downloaded checkpoint to {download_path} ({os.path.getsize(download_path)} bytes)")
            return download_path
        else:
            print(f"[play_gm] Download failed: {result.stderr}")
    except Exception as e:
        print(f"[play_gm] Download error: {e}")
    return None


def copy_checkpoint_to_logs(checkpoint_path, experiment_name="x1_dh_stand"):
    """Copy checkpoint to logs directory structure expected by task_registry"""
    # task_registry uses: logs/{experiment_name}/exported_data/{load_run}/model_{checkpoint}.pt
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name, "exported_data", "gm_play")
    os.makedirs(log_dir, exist_ok=True)
    dest = os.path.join(log_dir, os.path.basename(checkpoint_path))
    if not os.path.exists(dest):
        shutil.copy2(checkpoint_path, dest)
        print(f"[play_gm] Copied checkpoint: {checkpoint_path} -> {dest}")
    return log_dir


def package_video_as_pt(video_path, experiment_name="x1_dh_stand"):
    """Package mp4 video as model_isaac_video.pt for GM SDK auto-upload"""
    # Save in a subdirectory so SDK's PT directory scan discovers it
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name, "play_output")
    os.makedirs(log_dir, exist_ok=True)
    pt_path = os.path.join(log_dir, "model_isaac_video.pt")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    torch.save({"bytes": video_bytes, "filename": os.path.basename(video_path)}, pt_path)
    print(f"[play_gm] Packaged video ({len(video_bytes)} bytes) -> {pt_path}")
    return pt_path


def save_diag_data(diag_data, experiment_name="x1_dh_stand"):
    """Save diagnostic trajectory data as model_diag.pt for GM SDK upload"""
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name, "play_output")
    os.makedirs(log_dir, exist_ok=True)
    pt_path = os.path.join(log_dir, "model_diag.pt")
    torch.save(diag_data, pt_path)
    print(f"[play_gm] Saved diagnostic data -> {pt_path}")


def save_diag_csv(diag_data, experiment_name="x1_dh_stand", num_actions=12, dt=0.01):
    """Save diagnostic trajectory data as isaac_diag.csv."""
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name, "play_output")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "isaac_diag.csv")

    header = ["step", "time_s", "base_height", "base_vel_x", "base_vel_y", "base_vel_yaw",
              "base_yaw", "foot_yaw_l", "foot_yaw_r",
              "command_x", "foot_z_l", "foot_z_r", "foot_force_l", "foot_force_r"]
    header += [f"dof_pos_{i}" for i in range(num_actions)]
    header += [f"dof_vel_{i}" for i in range(num_actions)]
    header += [f"dof_torque_{i}" for i in range(num_actions)]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(diag_data["base_height"])):
            row = [i, round(i * dt, 6), diag_data["base_height"][i],
                   diag_data["base_vel_x"][i], diag_data["base_vel_y"][i],
                   diag_data["base_vel_yaw"][i], diag_data["base_yaw"][i],
                   diag_data["foot_yaw_l"][i], diag_data["foot_yaw_r"][i],
                   diag_data["command_x"][i],
                   diag_data["foot_z_l"][i], diag_data["foot_z_r"][i],
                   diag_data["foot_force_l"][i], diag_data["foot_force_r"][i]]
            row += diag_data["dof_pos"][i]
            row += diag_data["dof_vel"][i]
            row += diag_data["dof_torque"][i]
            writer.writerow(row)

    print(f"[play_gm] Saved diagnostic CSV -> {csv_path}")
    return csv_path


def package_csv_as_pt(csv_path, experiment_name="x1_dh_stand"):
    """Package diagnostic CSV as model_isaac_csv.pt for GM SDK auto-upload"""
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name, "gm_play")
    os.makedirs(log_dir, exist_ok=True)
    pt_path = os.path.join(log_dir, "model_isaac_csv.pt")
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_bytes = f.read().encode("utf-8")
    torch.save({"bytes": csv_bytes, "filename": os.path.basename(csv_path)}, pt_path)
    print(f"[play_gm] Packaged CSV ({len(csv_bytes)} bytes) -> {pt_path}")
    return pt_path


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override for playback
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.env.episode_length_s = 1000
    env_cfg.noise.add_noise = False

    # Disable all domain randomization for clean playback
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_torque = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.domain_rand.add_lag = False
    env_cfg.domain_rand.add_dof_lag = False
    env_cfg.commands.heading_command = False
    env_cfg.noise.curriculum = False

    # Enable headless rendering: no viewer but GPU camera sensors work
    env_cfg.env.enable_headless_render = True

    train_cfg.seed = 12345

    # Find and load checkpoint
    checkpoint_path = find_checkpoint()
    if checkpoint_path is None:
        print("[play_gm] ERROR: No checkpoint found in /personal/ or logs/")
        sys.exit(1)

    print(f"[play_gm] Found checkpoint: {checkpoint_path}")

    # Copy checkpoint to expected logs directory
    log_dir = copy_checkpoint_to_logs(checkpoint_path, train_cfg.runner.experiment_name)
    model_name = os.path.basename(checkpoint_path)  # e.g. model_10000.pt
    checkpoint_num = int(model_name.replace("model_", "").replace(".pt", ""))

    # Configure runner to load from our copied checkpoint
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = "gm_play"
    train_cfg.runner.checkpoint = checkpoint_num

    # Create environment (headless with rendering enabled)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 注：viewer 相机 API（set_camera/viewer_camera_look_at）在 headless 下无效，
    # camera sensor 在下方 update_camera() 中手动定位跟随

    # Create runner and load policy
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    print("[play_gm] Policy loaded successfully!")

    # 高分辨率相机：渲染管线已启用（headless GPU graphics），尝试 1920x1080；
    # 失败（handle -1）时回退到 base_task.py 的 720x480 相机
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1920
    camera_properties.height = 1080
    h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    if h1 == -1:
        h1 = env.camera_handle
        CAM_W, CAM_H = 720, 480
        print("[play_gm] WARN: high-res camera failed, fallback to base 720x480")
    else:
        CAM_W, CAM_H = 1920, 1080
    print(f"[play_gm] Using camera handle: {h1} ({CAM_W}x{CAM_H})")

    # 相机跟随（防晃动）：固定相机高度 + 水平 EMA 平滑
    # root z 随步态 ±2-3cm 颠簸，若相机 z 跟随会上下晃；水平低通滤波消除步态振荡
    cam_xy = np.zeros(2)
    cam_inited = False
    CAM_SMOOTH = 0.2  # EMA 系数（10Hz 更新，时间常数约 0.5s，2Hz 步态振荡衰减至 ~15%）
    CAM_OFF = np.array([1.4, 1.4])  # 相机相对机器人的水平偏移

    def update_camera():
        nonlocal cam_xy, cam_inited
        root_pos = env.root_states[0, :3].cpu().numpy()
        goal = root_pos[:2] + CAM_OFF
        if not cam_inited:
            cam_xy = goal.copy()
            cam_inited = True
        else:
            cam_xy = CAM_SMOOTH * goal + (1 - CAM_SMOOTH) * cam_xy
        env.gym.set_camera_location(
            h1, env.envs[0],
            gymapi.Vec3(cam_xy[0], cam_xy[1], 1.2),              # 固定高度
            gymapi.Vec3(cam_xy[0] - 1.4, cam_xy[1] - 1.4, 0.5),  # 看向固定高度躯干，视线方向恒定
        )
    update_camera()

    # Setup video writer - save to logs dir (not /personal which may not exist)
    video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "play_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, 50.0, (CAM_W, CAM_H))
    print(f"[play_gm] Recording video to: {video_path}")

    # Get foot indices for diagnostics
    left_foot_idx = env.feet_indices[0].item()
    right_foot_idx = env.feet_indices[1].item()

    obs = env.get_observations()
    total_steps = 2000  # 40 seconds at 50Hz control
    frame_count = 0

    # Diagnostic data storage
    diag = {
        "base_height": [],
        "base_vel_x": [],
        "base_vel_y": [],
        "base_vel_yaw": [],
        "base_yaw": [],
        "foot_yaw_l": [],
        "foot_yaw_r": [],
        "command_x": [],
        "foot_z_l": [],
        "foot_z_r": [],
        "foot_force_l": [],
        "foot_force_r": [],
        "dof_pos": [],
        "dof_vel": [],
        "dof_torque": [],
    }

    FIX_COMMAND = True
    fix_vel = 0.5  # Forward walking speed
    vel_sum = 0.0  # HUD 平均速度统计

    for i in range(total_steps):
        actions = policy(obs.detach())

        if FIX_COMMAND:
            env.commands[:, 0] = fix_vel
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        vel_sum += env.base_lin_vel[0, 0].item()
        avg_vel = vel_sum / (i + 1)

        # Record diagnostic data
        diag["base_height"].append(env.root_states[0, 2].item())
        diag["base_vel_x"].append(env.base_lin_vel[0, 0].item())
        diag["base_vel_y"].append(env.base_lin_vel[0, 1].item())
        diag["base_vel_yaw"].append(env.base_ang_vel[0, 2].item())
        base_quat = env.root_states[0, 3:7]
        base_yaw = torch.atan2(2.0 * (base_quat[3] * base_quat[2] + base_quat[0] * base_quat[1]),
                               1.0 - 2.0 * (base_quat[1] * base_quat[1] + base_quat[2] * base_quat[2]))
        diag["base_yaw"].append(base_yaw.item())
        # 真实脚朝向：脚局部前向轴(+z，URDF 名义位姿 FK 验证：local_z≈base+X)投影到水平面的航向角，相对 base yaw
        # 注：feet_euler_xyz[:,:,2] 因 ankle_roll_link rpy=(0,pi/2,0) 万向锁产生固定伪影(≈1.89/1.65 rad)，不可用
        # 注2：不可用 quat_rotate——GM 镜像中的实现内部 torch.cross 对 (...,4) 四元数直接叉乘会报
        #      "linalg.cross: inputs dimension -1 must have length 3. Got 4 and 3"（TASK_20260813_027 已实测崩溃）。
        #      此处手写四元数旋转局部 (0,0,1)（与 isaacgym quat_rotate 同 Hamilton 约定）。
        feet_quat = env.feet_quat  # (num_envs, num_feet, 4) wxyz
        fqw = feet_quat[..., 0:1]
        fqx = feet_quat[..., 1:2]
        fqy = feet_quat[..., 2:3]
        fqz = feet_quat[..., 3:4]
        foot_fwd_x = 2.0 * (fqx * fqz + fqw * fqy)
        foot_fwd_y = 2.0 * (fqy * fqz - fqw * fqx)
        foot_fwd_z = 1.0 - 2.0 * (fqx * fqx + fqy * fqy)
        foot_fwd = torch.cat([foot_fwd_x, foot_fwd_y, foot_fwd_z], dim=-1)  # 脚前向轴在世界系
        foot_yaw_world = torch.atan2(foot_fwd[..., 1], foot_fwd[..., 0])
        foot_yaw_rel = (foot_yaw_world - base_yaw + torch.pi) % (2.0 * torch.pi) - torch.pi
        diag["foot_yaw_l"].append(foot_yaw_rel[0, 0].item())
        diag["foot_yaw_r"].append(foot_yaw_rel[0, 1].item())
        diag["command_x"].append(env.commands[0, 0].item())
        diag["foot_z_l"].append(env.rigid_state[0, left_foot_idx, 2].item())
        diag["foot_z_r"].append(env.rigid_state[0, right_foot_idx, 2].item())
        diag["foot_force_l"].append(env.contact_forces[0, left_foot_idx, 2].item())
        diag["foot_force_r"].append(env.contact_forces[0, right_foot_idx, 2].item())
        diag["dof_pos"].append(env.dof_pos[0].cpu().numpy().tolist())
        diag["dof_vel"].append(env.dof_vel[0].cpu().numpy().tolist())
        diag["dof_torque"].append(env.torques[0].cpu().numpy().tolist())

        # Render and record video frame
        frame_count += 1
        if frame_count % 10 == 0:
            update_camera()  # 相机跟随机器人
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)

        if frame_count % 2 == 0:  # 50fps 视频（policy 100Hz，每 2 步 1 帧）
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            if img is not None and len(img) > 0:
                img = np.reshape(img, (CAM_H, CAM_W, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # HUD overlay：与 play.py viewer 录制一致的状态文字
                _fz_l = env.contact_forces[0, left_foot_idx, 2].item()
                _fz_r = env.contact_forces[0, right_foot_idx, 2].item()
                _draw_play_hud(img, env, 0, i,
                               env.commands[0, 0].item(), env.base_lin_vel[0, 0].item(), avg_vel,
                               _fz_l, _fz_r, _fz_l > 1.0, _fz_r > 1.0)
                video.write(img[..., :3])
                if frame_count % 500 == 0:
                    print(f"[play_gm] Frame {frame_count} mean brightness: {img.mean():.1f}")

        if i % 200 == 0:
            print(f"[play_gm] Step {i}/{total_steps} | vel_x={env.base_lin_vel[0, 0].item():.3f} | height={env.root_states[0, 2].item():.3f}")

    # Cleanup
    video.release()
    print(f"[play_gm] Video saved to {video_path} ({frame_count} frames)")

    # Package video as .pt for GM SDK auto-upload
    if os.path.exists(video_path):
        package_video_as_pt(video_path, train_cfg.runner.experiment_name)

    # Save diagnostic data
    save_diag_data(diag, train_cfg.runner.experiment_name)
    csv_path = save_diag_csv(diag, train_cfg.runner.experiment_name, env_cfg.env.num_actions, env.dt)
    csv_pt_path = package_csv_as_pt(csv_path, train_cfg.runner.experiment_name)

    # Print summary
    print("\n[play_gm] === Playback Summary ===")
    print(f"  Total steps: {total_steps}")
    print(f"  Frames recorded: {frame_count}")
    avg_vel = np.mean(diag["base_vel_x"])
    avg_height = np.mean(diag["base_height"])
    print(f"  Avg forward velocity: {avg_vel:.3f} m/s (target: {fix_vel})")
    print(f"  Avg base height: {avg_height:.3f} m")
    print(f"  Video: {video_path}")
    print(f"  Packaged for upload: logs/{train_cfg.runner.experiment_name}/model_isaac_video.pt")
    print(f"  Diagnostics: logs/{train_cfg.runner.experiment_name}/model_diag.pt")
    print(f"  CSV: {csv_path}")
    print(f"  CSV packaged for upload: {csv_pt_path}")

    # Wait for SDK to detect and upload model files
    import time
    print("[play_gm] Waiting 60s for SDK file upload...")
    time.sleep(60)
    print("[play_gm] Done.")


if __name__ == "__main__":
    args = get_args()
    play(args)
