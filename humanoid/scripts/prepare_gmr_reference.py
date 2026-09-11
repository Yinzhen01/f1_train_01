"""Audit GMR against the training URDF and export a non-cyclic 12-DOF clip.

Offline dependencies: numpy, scipy, mujoco. Training only needs the exported NPZ.
Optional bounded IK adapts the feet to the training geometry and joint limits.
No time scaling, cycle closure or dynamics repair is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


JOINT_NAMES = tuple(side + "_" + joint + "_joint" for side in ("left", "right")
                    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def fk(joints, positions):
    transforms = {"base_link": np.eye(4)}
    pending = list(joints)
    while pending:
        previous = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            transform = origin(joint.find("origin"))
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                angle = positions.get(joint.get("name"), 0.)
                transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
            transforms[joint.find("child").get("link")] = transforms[parent] @ transform
            pending.remove(joint)
        if len(pending) == previous:
            raise ValueError("URDF is not a connected tree rooted at base_link")
    return transforms


def stl_vertices(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError("Expected binary STL: " + str(path))
    triangles = np.frombuffer(raw, dtype=np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]), offset=84)
    return np.unique(triangles["v"].reshape(-1, 3), axis=0).astype(float)


def conform_limits(source_model, training_model, qpos, names, lower, upper):
    """Bounded pose IK: preserve source feet on the actual training geometry."""
    indices = [7 + names.index(name) for name in JOINT_NAMES]
    result = qpos.copy()
    source_data = mujoco.MjData(source_model)
    data = mujoco.MjData(training_model)
    source_feet = [source_model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    feet = [training_model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    addresses = [training_model.joint(name).qposadr[0] for name in JOINT_NAMES]
    changed, errors_p, errors_r, raw_p, raw_r = [], [], [], [], []
    for frame, source in enumerate(qpos):
        original_joints = source[indices]
        source_data.qpos[:] = source
        mujoco.mj_forward(source_model, source_data)
        target_p = source_data.xpos[source_feet].copy()
        target_R = source_data.xmat[source_feet].reshape(2, 3, 3).copy()
        source_R = Rotation.from_quat(source[[4, 5, 6, 3]])

        def evaluate(x):
            R = (Rotation.from_rotvec(x[3:6]) * source_R).as_matrix()
            data.qpos[addresses] = x[6:]
            mujoco.mj_forward(training_model, data)
            ep = data.xpos[feet] @ R.T + source[:3] + x[:3] - target_p
            er = Rotation.from_matrix(target_R.transpose(0, 2, 1) @ R @ data.xmat[feet].reshape(2, 3, 3)).as_rotvec()
            return np.r_[300 * ep.ravel(), 100 * er.ravel(), 20 * x[:3], 10 * x[3:6], .3 * (x[6:] - original_joints)]

        initial = np.r_[np.zeros(6), np.clip(original_joints, lower + 1e-9, upper - 1e-9)]
        before = evaluate(np.r_[np.zeros(6), original_joints])
        raw_p.append(float(np.linalg.norm(before[:6].reshape(2, 3) / 300, axis=1).max()))
        raw_r.append(float(np.linalg.norm(before[6:12].reshape(2, 3) / 100, axis=1).max()))
        fit = least_squares(evaluate, initial, bounds=(np.r_[np.full(3, -.05), np.full(3, -.2), lower],
                                                     np.r_[np.full(3, .05), np.full(3, .2), upper]),
                            max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        residual = evaluate(fit.x)
        if not fit.success:
            raise ValueError("Bounded IK did not converge at frame %d" % frame)
        result[frame, :3] = source[:3] + fit.x[:3]
        quat = (Rotation.from_rotvec(fit.x[3:6]) * source_R).as_quat()
        result[frame, 3:7] = quat[[3, 0, 1, 2]]
        result[frame, indices] = fit.x[6:]
        changed.append(frame)
        errors_p.append(float(np.linalg.norm(residual[:6].reshape(2, 3) / 300, axis=1).max()))
        errors_r.append(float(np.linalg.norm(residual[6:12].reshape(2, 3) / 100, axis=1).max()))
    return result, dict(changed_frames=changed, raw_ankle_fk_position_m=max(raw_p), raw_ankle_fk_rotation_rad=max(raw_r),
                        max_ankle_position_change_m=max(errors_p, default=0.),
                        max_ankle_rotation_change_rad=max(errors_r, default=0.),
                        method="bounded root-plus-12-leg IK, training URDF limits unchanged")


def prepare(source, urdf, gmr_xml, output, fit_limits=False, whole_body=False):
    with np.load(source, allow_pickle=False) as src:
        qpos = src["qpos"].copy()
        fps = float(src["fps"])
        names = src["joint_names"].astype(str).tolist()
    if len(set(names)) != len(names) or qpos.shape[1] != 7 + len(names):
        raise ValueError("Invalid GMR joint-name mapping")
    if not np.isfinite(qpos).all() or fps <= 0:
        raise ValueError("Nonfinite input or invalid FPS")
    joint_names = tuple(names) if whole_body else JOINT_NAMES
    if whole_body and (len(joint_names) != 29 or fit_limits):
        raise ValueError("Whole-body export requires exactly 29 joints and no implicit IK repair")
    positions = qpos[:, [7 + names.index(name) for name in joint_names]]
    robot = ET.parse(urdf).getroot()
    joints = robot.findall("joint")
    movable = {j.get("name"): j for j in joints if j.get("type") != "fixed"}
    if set(movable) != set(joint_names):
        raise ValueError("Training model does not match the named reference joints")
    lower = np.array([float(movable[name].find("limit").get("lower")) for name in joint_names])
    upper = np.array([float(movable[name].find("limit").get("upper")) for name in joint_names])
    model = mujoco.MjModel.from_xml_path(str(gmr_xml))
    training_model = mujoco.MjModel.from_xml_path(str(urdf))
    limit_report = None
    if fit_limits:
        qpos, limit_report = conform_limits(model, training_model, qpos, names, lower, upper)
        positions = qpos[:, [7 + names.index(name) for name in JOINT_NAMES]]
    if np.any(positions < lower - 1e-8) or np.any(positions > upper + 1e-8):
        raise ValueError("Reference exceeds training URDF limits; do not silently clip")
    root_rot = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    root_quat = root_rot.as_quat()
    root_pos = qpos[:, :3].copy()
    root_pos[:, :2] -= root_pos[0, :2]
    zero_fk = fk(joints, {})
    foot_names = [side + "_ankle_roll_link" for side in ("left", "right")]
    sole_local, normal_local, sole_vertices = [], [], []
    mesh_hashes = {}
    for name in foot_names:
        link = next(x for x in robot.findall("link") if x.get("name") == name)
        collision = link.find("collision")
        mesh = collision.find("geometry/mesh")
        path = (urdf.parent / mesh.get("filename")).resolve()
        vertices = stl_vertices(path) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        local = origin(collision.find("origin"))
        vertices = vertices @ local[:3, :3].T + local[:3, 3]
        rest = zero_fk[name]
        world = vertices @ rest[:3, :3].T + rest[:3, 3]
        bottom = world[:, 2] <= world[:, 2].min() + .0002
        center = world[bottom].mean(0)
        _, singular, vt = np.linalg.svd(world[bottom] - center, full_matrices=False)
        normal = vt[-1] * (1 if vt[-1, 2] >= 0 else -1)
        if bottom.sum() < 4 or singular[1] < .01:
            raise ValueError("Cannot calibrate training sole plane")
        sole_local.append(rest[:3, :3].T @ (center - rest[:3, 3]))
        normal_local.append(rest[:3, :3].T @ normal)
        sole_vertices.append(vertices)
        mesh_hashes[name] = sha(path)
    data = mujoco.MjData(training_model)
    source_data = mujoco.MjData(model)
    tracking_bodies = ["lumbar_pitch_link", "left_wrist_pitch_link", "right_wrist_pitch_link"] if whole_body else []
    body_pos_base, body_quat_base = [], []
    source_fk_position_error, source_fk_rotation_error = 0., 0.
    foot_pos_base, foot_quat_base, foot_min_z, foot_world, foot_normal_world = [], [], [], [], []
    max_position_error, max_rotation_error = 0., 0.
    for i, row in enumerate(positions):
        transforms = fk(joints, dict(zip(joint_names, row)))
        for j, name in enumerate(joint_names):
            data.qpos[training_model.joint(name).qposadr[0]] = row[j]
        mujoco.mj_forward(training_model, data)
        R = root_rot[i].as_matrix()
        if whole_body:
            source_data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, source_data)
        for name in joint_names:
            body = movable[name].find("child").get("link")
            bid = training_model.body(body).id
            p_gmr = data.xpos[bid]
            R_gmr = data.xmat[bid].reshape(3, 3)
            T = transforms[body]
            max_position_error = max(max_position_error, float(np.linalg.norm(p_gmr - T[:3, 3])))
            max_rotation_error = max(max_rotation_error, float(Rotation.from_matrix(R_gmr.T @ T[:3, :3]).magnitude()))
            if whole_body:
                src_body = model.body(body).id
                source_fk_position_error = max(source_fk_position_error, float(np.linalg.norm(
                    source_data.xpos[src_body] - (R @ T[:3, 3] + qpos[i, :3]))))
                source_fk_rotation_error = max(source_fk_rotation_error, float(Rotation.from_matrix(
                    source_data.xmat[src_body].reshape(3, 3).T @ R @ T[:3, :3]).magnitude()))
        if whole_body:
            body_pos_base.append([transforms[name][:3, 3] for name in tracking_bodies])
            body_quat_base.append([Rotation.from_matrix(transforms[name][:3, :3]).as_quat() for name in tracking_bodies])
        p, r, z, w, n = [], [], [], [], []
        for f, name in enumerate(foot_names):
            T = transforms[name]
            center = T[:3, 3] + T[:3, :3] @ sole_local[f]
            vertices = sole_vertices[f] @ (R @ T[:3, :3]).T + R @ T[:3, 3] + root_pos[i]
            p.append(center)
            r.append(Rotation.from_matrix(T[:3, :3]).as_quat())
            z.append(vertices[:, 2].min())
            w.append(R @ center + root_pos[i])
            n.append(R @ T[:3, :3] @ normal_local[f])
        foot_pos_base.append(p)
        foot_quat_base.append(r)
        foot_min_z.append(z)
        foot_world.append(w)
        foot_normal_world.append(n)
    if max_position_error > 1e-4 or max_rotation_error > 1e-4:
        raise ValueError("URDF parser/MuJoCo leg FK mismatch: position=%g m rotation=%g rad" % (max_position_error, max_rotation_error))
    if whole_body and (source_fk_position_error > .005 or source_fk_rotation_error > .01):
        raise ValueError("Whole-body source/training geometry differs: %g m, %g rad" % (source_fk_position_error, source_fk_rotation_error))
    foot_min_z = np.asarray(foot_min_z)
    extra_lift = np.maximum(0., .002 - foot_min_z.min(axis=1))
    if extra_lift.max() > .01:
        raise ValueError("Training foot mesh needs more than 1 cm clearance repair")
    root_pos[:, 2] += extra_lift
    foot_min_z += extra_lift[:, None]
    foot_world = np.asarray(foot_world)
    foot_world[:, :, 2] += extra_lift[:, None]
    foot_speed = np.linalg.norm(np.gradient(np.asarray(foot_world), 1 / fps, axis=0), axis=-1)
    # Kinematic labels, not measured contact forces. Preserve double support.
    contact = ((foot_min_z < .025) & (foot_speed < .35)).astype(float)
    velocity = np.gradient(root_pos, 1 / fps, axis=0)
    dq = np.gradient(positions, 1 / fps, axis=0)
    omega_steps = (root_rot[1:] * root_rot[:-1].inv()).as_rotvec() * fps
    omega = np.vstack([omega_steps[0], (omega_steps[:-1] + omega_steps[1:]) / 2, omega_steps[-1]])
    normals = np.asarray(foot_normal_world)
    metadata = dict(source_sha256=sha(source), training_urdf_sha256=sha(urdf), gmr_xml_sha256=sha(gmr_xml),
                    training_urdf_lf_sha256=hashlib.sha256(urdf.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                    source_name=source.name, frames=len(qpos), fps=fps, duration_s=(len(qpos)-1)/fps,
                    cyclic=False, joint_names=list(joint_names), foot_names=foot_names,
                    quaternion_order="xyzw", units="meters, radians, seconds",
                    max_leg_fk_position_error_m=max_position_error, max_leg_fk_rotation_error_rad=max_rotation_error,
                    limit_fit=limit_report, extra_clearance_lift_max_m=float(extra_lift.max()),
                    minimum_foot_mesh_z_m=float(foot_min_z.min()), max_joint_speed_rad_s=float(abs(dq).max()),
                    max_root_speed_m_s=float(np.linalg.norm(velocity, axis=1).max()),
                    contact_frames=contact.sum(0).tolist(), foot_mesh_sha256=mesh_hashes,
                    standing_max_sole_tilt_deg=float(np.rad2deg(np.arccos(np.clip(normals[np.r_[0:31, 126:139], :, 2], -1, 1))).max()),
                    boundary="Kinematic reference and FK audit only, not PD/contact/dynamics validation")
    extra = {}
    if whole_body:
        metadata.update(reference_variant="whole_body_29dof_v1", tracking_body_names=tracking_bodies,
                        initial_joint_angles=dict(zip(joint_names, positions[0].tolist())),
                        source_fk_position_error_m=source_fk_position_error,
                        source_fk_rotation_error_rad=source_fk_rotation_error)
        extra.update(body_pos_base=body_pos_base, body_quat_base=body_quat_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, fps=fps, joint_names=np.asarray(joint_names), dof_pos=positions,
                        dof_vel=dq, root_pos=root_pos, root_quat=root_quat, root_vel=velocity, root_ang_vel=omega,
                        foot_pos_base=foot_pos_base, foot_quat_base=foot_quat_base, foot_contact=contact,
                        sole_local=sole_local, sole_normal_local=normal_local,
                        lower=lower, upper=upper, metadata_json=json.dumps(metadata), **extra)
    metadata["output_sha256"] = sha(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--gmr-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-limits", action="store_true")
    parser.add_argument("--whole-body", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.urdf, args.gmr_xml, args.output, args.fit_limits, args.whole_body)
