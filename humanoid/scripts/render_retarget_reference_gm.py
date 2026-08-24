"""Render the raw X1 12-DOF retargeted reference in Isaac Gym.

This is intentionally not a policy rollout. The 12 joint positions come from
``walk_12dof.csv``. Reconstructed root pose and foot contacts are used only to
place the robot in the scene and annotate the video.
"""

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
from isaacgym import gymapi, gymtorch
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.retarget_reference import (
    X1_12DOF_JOINT_NAMES,
    interpolate_reference,
    load_retarget_reference,
)


EXPERIMENT_NAME = "x1_retarget_12dof_reference_render"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def quaternion_yaw_xyzw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def draw_hud(image, frame_index, frame_count, time_s, duration, contacts):
    lines = (
        "X1 12-DOF RETARGETED REFERENCE",
        "REFERENCE DATA / NO POLICY / NO RL ACTIONS",
        f"frame {frame_index + 1}/{frame_count}   t={time_s:05.2f}/{duration:05.2f} s",
        f"left contact: {'ON' if contacts[0] >= 0.5 else 'OFF'}    "
        f"right contact: {'ON' if contacts[1] >= 0.5 else 'OFF'}",
        "controlled joints: 12 (legs only)",
    )
    colors = ((255, 255, 255), (0, 255, 255), (220, 220, 220), (180, 255, 180), (220, 220, 220))
    for index, (line, color) in enumerate(zip(lines, colors)):
        position = (45, 65 + index * 48)
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("fps, width, and height must be positive")

    motion_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", "motions", "x1")
    joint_path = os.path.join(motion_dir, "walk_12dof.csv")
    root_path = os.path.join(motion_dir, "walk_12dof_contact_consistent.csv")
    reference = load_retarget_reference(joint_path, root_path)
    print(
        f"[reference_render] Loaded exact 12-DOF reference: "
        f"rows={len(reference.timestamps)}, duration={reference.duration:.6f}s"
    )
    print(f"[reference_render] Joint order: {list(reference.joint_names)}")
    print("[reference_render] Policy/checkpoint: none")

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / args.fps
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("failed to create Isaac Gym simulation")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane.static_friction = 0.6
    plane.dynamic_friction = 0.6
    gym.add_ground(sim, plane)

    asset_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "resources", "robots", "x1", "urdf", "X1_12DOF.urdf"
    )
    asset_root, asset_file = os.path.split(asset_path)
    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if robot_asset is None:
        raise RuntimeError(f"failed to load robot asset {asset_file}")

    dof_names = tuple(gym.get_asset_dof_names(robot_asset))
    if len(dof_names) != 12 or set(dof_names) != set(X1_12DOF_JOINT_NAMES):
        raise RuntimeError(f"asset DOFs do not match the 12-joint reference: {dof_names}")
    reference_indices = np.asarray(
        [reference.joint_names.index(name) for name in dof_names], dtype=np.int64
    )

    env = gym.create_env(
        sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1
    )
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(*reference.root_positions[0])
    initial_pose.r = gymapi.Quat(*reference.root_quaternions_xyzw[0])
    gym.create_actor(env, robot_asset, initial_pose, "x1_reference", 0, 0)
    gym.prepare_sim(sim)

    root_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_tensor = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1, 2)

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = args.width
    camera_properties.height = args.height
    camera_properties.horizontal_fov = 72.0
    camera = gym.create_camera_sensor(env, camera_properties)

    output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", EXPERIMENT_NAME)
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "play_output.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer at {video_path}")

    frame_count = int(round(reference.duration * args.fps))
    camera_position = None
    diagnostics = {"time_s": [], "root_pos": [], "joint_pos": [], "contacts": []}

    for frame_index in range(frame_count):
        time_s = frame_index / float(args.fps)
        joints, root_pos, root_quat, contacts = interpolate_reference(reference, time_s)
        ordered_joints = joints[reference_indices]

        root_tensor[0, :3] = torch.as_tensor(root_pos, device=root_tensor.device, dtype=root_tensor.dtype)
        root_tensor[0, 3:7] = torch.as_tensor(root_quat, device=root_tensor.device, dtype=root_tensor.dtype)
        root_tensor[0, 7:13] = 0.0
        dof_tensor[:, 0] = torch.as_tensor(ordered_joints, device=dof_tensor.device, dtype=dof_tensor.dtype)
        dof_tensor[:, 1] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_tensor))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        yaw = quaternion_yaw_xyzw(root_quat)
        forward = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
        right = np.asarray((-forward[1], forward[0]), dtype=np.float64)
        desired_xy = root_pos[:2] - 2.7 * forward + 1.7 * right
        desired_camera = np.asarray((desired_xy[0], desired_xy[1], root_pos[2] + 1.15))
        camera_position = desired_camera if camera_position is None else 0.12 * desired_camera + 0.88 * camera_position
        target_xy = root_pos[:2] + 0.35 * forward
        gym.set_camera_location(
            camera,
            env,
            gymapi.Vec3(*camera_position),
            gymapi.Vec3(target_xy[0], target_xy[1], root_pos[2] + 0.05),
        )
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        image = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
        if image is None or len(image) == 0:
            raise RuntimeError(f"empty camera image at frame {frame_index}")
        image = np.asarray(image).reshape(args.height, args.width, 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        draw_hud(image, frame_index, frame_count, time_s, reference.duration, contacts)
        writer.write(image)

        diagnostics["time_s"].append(time_s)
        diagnostics["root_pos"].append(root_pos.tolist())
        diagnostics["joint_pos"].append(ordered_joints.tolist())
        diagnostics["contacts"].append(contacts.tolist())
        if frame_index % 100 == 0:
            print(f"[reference_render] frame={frame_index}/{frame_count} t={time_s:.2f}s")

    writer.release()
    gym.destroy_sim(sim)

    diag_dir = os.path.join(output_dir, "exported_data", "gm_reference_render")
    os.makedirs(diag_dir, exist_ok=True)
    diag_path = os.path.join(diag_dir, "model_reference_diag.pt")
    torch.save(
        {
            "joint_names": dof_names,
            "source_joint_file": "walk_12dof.csv",
            "source_root_file": "walk_12dof_contact_consistent.csv",
            "policy": None,
            "fps": args.fps,
            **diagnostics,
        },
        diag_path,
    )
    manifest_path = os.path.join(output_dir, "reference_render_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": "12dof_retarget_reference_no_policy",
                "joint_count": 12,
                "source_frames": len(reference.timestamps),
                "video_frames": frame_count,
                "fps": args.fps,
                "duration_s": reference.duration,
                "resolution": [args.width, args.height],
                "video": os.path.basename(video_path),
            },
            handle,
            indent=2,
        )

    print(
        f"[reference_render] COMPLETE video={video_path} frames={frame_count} "
        f"fps={args.fps} resolution={args.width}x{args.height}"
    )
    print(f"[reference_render] Diagnostics: {diag_path}")
    print("[reference_render] Waiting 60s for SDK artifact upload...")
    time.sleep(60)


if __name__ == "__main__":
    main()
