"""Offline audit/render of the captured AMP policy, never a dynamics replay.

This file is intentionally outside the frozen cloud training implementation.
Reports retain every initial condition; rendering defaults to env 0, not best.
"""
import argparse
import codecs
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.features import angular_velocity_body, from_world_state
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.discriminator import AMPDiscriminator
from humanoid.amp.recovery import foot_collision_vertices
from humanoid.amp.learnability import assert_no_domain_randomization


def load_bundle(path):
    # torch's protocol-2 bytes representation uses the standard codecs encoder.
    # Allow only this primitive, never arbitrary pickle globals.
    torch.serialization.add_safe_globals([codecs.encode])
    packed = torch.load(path, map_location="cpu", weights_only=True)
    manifest = json.loads(packed["manifest_json"])
    with np.load(io.BytesIO(packed["npz_bytes"]), allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if manifest["fps"] != 100 or manifest["dr_unlocked"] is not False:
        raise ValueError("Unexpected evaluation protocol")
    assert_no_domain_randomization(manifest["environment"])
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite recorded states")
    return manifest, arrays


def episode(arrays, mode, env_index):
    valid = arrays[mode+"_valid"][:, env_index].astype(bool)
    if np.any(np.diff(valid.astype(int)) > 0):
        raise ValueError("A restarted episode was included in the validity mask")
    count = int(valid.sum())
    if not np.array_equal(valid, np.arange(len(valid)) < count):
        raise ValueError("Valid states must form one contiguous pre-reset prefix")
    time = arrays[mode+"_time"][:count]
    if len(time) and not np.allclose(time, (np.arange(count)+1)*.01, atol=1e-7, rtol=0):
        raise ValueError("Wrong or missing control ticks")
    names = ("root_state", "dof_pos", "dof_vel", "action", "torque", "base_lin_vel",
             "base_ang_vel", "foot_state", "foot_force", "key_positions_w", "failure")
    result = {key: arrays[mode+"_"+key][:count, env_index] for key in names}
    result["time"] = time
    result["initial"] = {key: arrays[mode+"_initial_"+key][env_index] for key in names}
    return result


def features_from_episode(data, experiment):
    def sequence(key):
        return torch.tensor(np.concatenate((data["initial"][key][None], data[key])), dtype=torch.float32)
    root, q, key = sequence("root_state"), sequence("dof_pos"), sequence("key_positions_w")
    return from_world_state(experiment.spec, root[1:, 3:7], root[1:, :3],
        angular_velocity_body(root[:-1, 3:7], root[1:, 3:7], .01), q[1:], (q[1:]-q[:-1])/.01,
        experiment.spec.joint_names, key[1:], experiment.spec.body_names)


def nearest_window_distance(windows, demonstration, mean, std):
    """Phase-free normalized RMS distance, independent of learned D weights."""
    if not len(windows):
        return None
    reference = ((demonstration-mean)/std).reshape(len(demonstration), -1)
    query = ((windows-mean)/std).reshape(len(windows), -1)
    distances, residuals = [], []
    for batch in query.split(128):
        nearest = torch.cdist(batch, reference).min(dim=1)
        distances.append(nearest.values/(query.shape[1]**.5))
        residuals.append((batch-reference[nearest.indices]).reshape(-1, 10, 39).square())
    values = torch.cat(distances).numpy()
    errors = torch.cat(residuals)
    groups = {"root_omega": (0, 3), "joint_position": (3, 15),
              "joint_velocity": (15, 27), "key_position": (27, 39)}
    return dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)), samples=len(values),
        nearest_window_group_mse={name: float(errors[..., start:end].mean())
                                  for name, (start, end) in groups.items()})


def heading_metrics(root_state):
    yaw = np.unwrap(Rotation.from_quat(root_state[:, 3:7]).as_euler("xyz")[:, 2])
    error = np.arctan2(np.sin(yaw), np.cos(yaw))
    return dict(heading_rms_to_world_x_deg=float(np.rad2deg(np.sqrt(np.mean(error**2)))),
        heading_change_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
        heading_abs_error_p95_deg=float(np.rad2deg(np.percentile(np.abs(error), 95))))


def velocity_spectrum(data, dt=.01, cutoff=10.):
    """Hann-windowed velocity power ratio; not a physical acceleration estimate."""
    x = np.asarray(data, dtype=np.float64)
    if len(x) < 20:
        return None
    transformed = np.fft.rfft((x-x.mean(axis=0))*np.hanning(len(x))[:, None], axis=0)
    power = np.abs(transformed)**2
    power[1:-1 if len(x) % 2 == 0 else None] *= 2
    frequency = np.fft.rfftfreq(len(x), dt)
    total = power.sum()
    return dict(cutoff_hz=cutoff, nyquist_hz=.5/dt,
                power_fraction_above_cutoff=0. if total < 1e-20 else float(power[frequency > cutoff].sum()/total))


def discriminator_metrics(discriminator, windows):
    with torch.no_grad():
        scores = discriminator(windows).flatten()
        reward = (1-.25*(scores-1).square()).clamp_min(discriminator.style_floor)
        return dict(score_mean=float(scores.mean()), normalized_reward_mean=float(reward.mean()),
                    zero_fraction=float((reward == 0).float().mean()),
                    negative_fraction=float((reward < 0).float().mean()), reward_floor=discriminator.style_floor)


def swing_runs(contact, minimum_frames=8):
    airborne = ~np.asarray(contact, dtype=bool)
    edges = np.diff(np.r_[False, airborne, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    # Exclude partial runs at either boundary and short force flicker.
    return int(np.sum((ends-starts >= minimum_frames) & (starts > 0) & (ends < len(contact))))


def foot_geometry(data, vertices, foot_names):
    height, speed = [], []
    for index, name in enumerate(foot_names):
        body = data["foot_state"][:, index]
        rotation = Rotation.from_quat(body[:, 3:7]).as_matrix()
        mesh = vertices[name]
        minimum = (rotation[:, 2, :] @ mesh.T).min(axis=1)+body[:, 2]
        bottom = mesh[mesh[:, 2] <= mesh[:, 2].min()+1e-5]
        sole_proxy = bottom.mean(axis=0)
        arm = np.einsum("nij,j->ni", rotation, sole_proxy)
        velocity = body[:, 7:10]+np.cross(body[:, 10:13], arm)
        height.append(minimum); speed.append(np.linalg.norm(velocity[:, :2], axis=1))
    return np.stack(height, axis=1), np.stack(speed, axis=1)


def rms(value):
    return None if not value.size else float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def analyze_episode(data, experiment, vertices, foot_names, duration, discriminator=None, auditor=None):
    count = len(data["time"])
    failed = bool(data["initial"]["failure"] or np.any(data["failure"]))
    report = dict(observed_s=count*.01, failure=failed,
                  survived=bool(count >= round(duration*100) and not failed))
    if not count:
        report["post_2s"] = None
        return report, None
    z, speed = foot_geometry(data, vertices, foot_names)
    geometry = dict(foot_mesh_min_z=z, sole_proxy_speed_xy=speed)
    active = data["time"] >= 2.
    if active.sum() < 20:
        report["post_2s"] = None
        return report, geometry
    q, velocity = data["dof_pos"][active], data["base_lin_vel"][active]
    force_contact = data["foot_force"][active, :, 2] > 5.
    # Net body force does not identify the other collider (ground vs self).
    # Height is a conservative ground-contact proxy, not a contact-pair query.
    contact = force_contact & (z[active] < .02)
    penetration = np.maximum(-z[active], 0.)
    sustained = dict(vx_mean=float(velocity[:, 0].mean()), vx_std=float(velocity[:, 0].std()),
        vy_abs_mean=float(np.abs(velocity[:, 1]).mean()),
        world_x_displacement_m=float(data["root_state"][active][-1, 0]-data["root_state"][active][0, 0]),
        joint_rom_rad=np.ptp(q, axis=0).tolist(), joint_std_rad=q.std(axis=0).tolist(),
        joint_accel_fd_rms_rad_s2=rms(np.diff(q, n=2, axis=0)/.01**2),
        joint_accel_native_rms_rad_s2=rms(np.diff(data["dof_vel"][active], axis=0)/.01),
        joint_jerk_fd_rms_rad_s3=rms(np.diff(q, n=3, axis=0)/.01**3),
        action_delta_rms=rms(np.diff(data["action"][active], axis=0)),
        torque_delta_rms_nm=rms(np.diff(data["torque"][active], axis=0)),
        contact_fraction=contact.mean(axis=0).tolist(),
        force_contact_fraction=force_contact.mean(axis=0).tolist(),
        force_contact_above_2cm_fraction=(force_contact & (z[active] >= .02)).mean(axis=0).tolist(),
        completed_swing_runs=[swing_runs(contact[:, i]) for i in range(2)],
        contact_sole_proxy_speed_mean_m_s=None if not contact.any() else float(speed[active][contact].mean()),
        geometric_penetration_max_mm=(penetration.max(axis=0)*1000).tolist(),
        geometric_penetration_p95_mm=(np.percentile(penetration, 95, axis=0)*1000).tolist())
    features = features_from_episode(data, experiment)
    windows = features.unfold(0, 10, 1).permute(0, 2, 1).contiguous()
    # Identical stride and post-transient cutoff for every policy/initialization.
    windows = windows[torch.from_numpy(data["time"][9:] >= 2.)][::10]
    sustained["demo_nearest_window_rms_zscore"] = nearest_window_distance(windows,
        experiment.windows, experiment.mean, experiment.std)
    sustained.update(heading_metrics(data["root_state"][active]))
    sustained["joint_velocity_spectrum"] = velocity_spectrum(data["dof_vel"][active])
    if discriminator is not None:
        sustained["learned_style"] = discriminator_metrics(discriminator, windows)
    if auditor is not None:
        sustained["frozen_auditor_style"] = discriminator_metrics(auditor, windows)
    report["post_2s"] = sustained
    return report, geometry


def plot_episode(data, geometry, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not len(data["time"]):
        return
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t = data["time"]
    axes[0].plot(t, data["base_lin_vel"][:, 0], label="policy vx")
    axes[0].axhline(.45, color="black", ls="--", label="command 0.45")
    axes[0].set_ylabel("m/s"); axes[0].legend()
    for index, label in ((3, "left knee"), (9, "right knee")):
        axes[1].plot(t, data["dof_pos"][:, index], label=label)
    axes[1].set_ylabel("rad"); axes[1].legend()
    for index, label in enumerate(("left mesh min", "right mesh min")):
        axes[2].plot(t, geometry["foot_mesh_min_z"][:, index]*1000, label=label)
    axes[2].axhline(0, color="black", ls="--"); axes[2].set_ylabel("mm"); axes[2].legend()
    for index, label in enumerate(("left Fz", "right Fz")):
        axes[3].plot(t, data["foot_force"][:, index, 2], label=label)
    axes[3].set_ylabel("N"); axes[3].set_xlabel("Physical time (s)"); axes[3].legend()
    for axis in axes:
        axis.grid(alpha=.25); axis.axvline(2., color="gray", ls=":")
    figure.suptitle("Actual policy rollout; first terminal frame included, no reset continuation")
    figure.tight_layout(); figure.savefig(output, dpi=150); plt.close(figure)


def render_episode(data, manifest, urdf, output):
    import cv2
    import imageio.v2 as imageio
    import mujoco
    import xml.etree.ElementTree as ET
    sys.path.insert(0, str(ROOT/"humanoid/scripts"))
    from render_gmr_policy import make_scene, label
    if not len(data["time"]):
        return None
    model, scene = make_scene(urdf, output)
    # The old 8m visual plane disappears on long walks although PhysX ground
    # remains infinite. Extend only this generated rendering scene, not assets.
    extent = np.maximum(8., np.max(np.abs(data["root_state"][:, :2]), axis=0)+4.)
    xml = ET.parse(scene)
    xml.find(".//geom[@name='render_ground']").set("size", "%g %g .1" % tuple(extent))
    xml.write(scene, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(scene))
    state = mujoco.MjData(model)
    address = [int(model.joint(name).qposadr[0]) for name in manifest["dof_names"]]
    root_address = int(model.joint("render_root").qposadr[0])
    renderer = mujoco.Renderer(model, height=600, width=640)
    option = mujoco.MjvOption()
    ground = model.geom("render_ground").id
    model.geom_group[ground] = 1
    option.geomgroup[:] = 0; option.geomgroup[1] = 1
    if not np.any(model.geom_group[np.arange(model.ngeom) != ground] == 1):
        option.geomgroup[:] = 1
    cameras = []
    for angle in (90, 20):
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = angle; cam.elevation = -17; cam.distance = 2.6
        cameras.append(cam)
    path = output/"policy_dual_view.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 50., (1280, 600))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    indices = np.arange(0, len(data["time"]), 2)
    if indices[-1] != len(data["time"])-1:
        indices = np.r_[indices, len(data["time"])-1]
    preview, max_error = [], 0.
    target = data["root_state"][0, :3].copy(); target[2] = .55
    try:
        for frame, i in enumerate(indices):
            root = data["root_state"][i]
            state.qpos[root_address:root_address+3] = root[:3]
            state.qpos[root_address+3:root_address+7] = root[[6, 3, 4, 5]]
            state.qpos[address] = data["dof_pos"][i]
            mujoco.mj_forward(model, state)
            actual = np.stack([state.xpos[model.body(name).id] for name in manifest["body_names"]])
            max_error = max(max_error, float(np.linalg.norm(actual-data["key_positions_w"][i], axis=1).max()))
            if max_error > .01:
                raise ValueError("Recorded/MuJoCo key-body FK mismatch exceeds 1 cm")
            target[:2] += (1-np.exp(-.02/.2))*(root[:2]-target[:2])
            panels = []
            for cam in cameras:
                cam.lookat[:] = target
                renderer.update_scene(state, camera=cam, scene_option=option)
                panels.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            view = np.concatenate(panels, axis=1)
            label(view, "AMP POLICY | t=%.2fs | vx=%.3f m/s" % (data["time"][i], data["base_lin_vel"][i, 0]), 16, 36)
            label(view, "Isaac Gym trajectory rendered by MuJoCo (not Sim2Sim)", 16, 575, scale=.55)
            writer.write(view)
            if frame == 0:
                cv2.imwrite(str(output/"first_frame.png"), view)
            if frame in (len(indices)//2, len(indices)-1):
                cv2.imwrite(str(output/("frame_%04d.png" % i)), view)
            if frame % 5 == 0 and data["time"][i] <= 8.:
                preview.append(cv2.cvtColor(cv2.resize(view, (640, 300)), cv2.COLOR_BGR2RGB))
    finally:
        writer.release(); renderer.close()
    imageio.mimsave(output/"preview_first8s.gif", preview, duration=100, loop=0)
    return dict(mp4=str(path), preview=str(output/"preview_first8s.gif"), frames=len(indices),
                fps=50, max_key_body_fk_error_m=max_error, scene=str(scene),
                visual_ground_half_extent_m=extent.tolist(),
                scope="rendered recorded PhysX states; no MuJoCo dynamics or Sim2Real claim")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--auditor-checkpoint", type=Path)
    parser.add_argument("--auditor-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--env-index", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest, arrays = load_bundle(args.bundle)
    experiment_name = manifest["identity"]["experiment"]
    files = {"f1_amp_walk02_s092": "lafan_walk02_s092.json", "f1_amp_walk02_recovery": "lafan_walk02_recovery.json",
             "f1_amp_walk02_recovery_static": "lafan_walk02_recovery_static.json"}
    files.update({"f1_amp_walk02_refine_"+group: "lafan_walk02_refine_"+group+".json"
                  for group in ("control", "smooth", "noamp")})
    files.update({"f1_amp_walk02_signal_"+group: "lafan_walk02_signal_"+group+".json"
                  for group in ("signed", "bridge")})
    files.update({'f1_amp_walk02_horizon_'+group: 'lafan_walk02_horizon_'+group+'.json'
                  for group in ('short', 'long')})
    files.update({'f1_amp_walk02_contact_'+group: 'lafan_walk02_contact_'+group+'.json'
                  for group in ('control', 'slip', 'tail')})
    files.update({'f1_amp_walk02_progress_'+group: 'lafan_walk02_progress_'+group+'.json'
                  for group in ('body', 'world')})
    files.update({'f1_amp_walk02_direction_'+group: 'lafan_walk02_direction_'+group+'.json'
                  for group in ('mix15', 'heading')})
    files.update({'f1_amp_walk02_sustain_'+group: 'lafan_walk02_sustain_'+group+'.json'
                  for group in ('control', 'progress3')})
    experiment = ScaledExperiment(ROOT, ROOT/"configs/amp"/files[experiment_name])
    if manifest["identity"] != experiment.identity() or manifest["dof_names"] != list(experiment.spec.joint_names):
        raise ValueError("Recorded/configuration identity mismatch")
    if not 0 <= args.env_index < manifest["num_envs"]:
        raise ValueError("Invalid environment selection")
    discriminator = None
    if args.checkpoint:
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
            raise ValueError("Checkpoint is not the recorded policy")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["amp_identity"] != experiment.identity():
            raise ValueError("Discriminator identity mismatch")
        discriminator = AMPDiscriminator(experiment.spec, experiment.mean, experiment.std,
                                          style_floor=experiment.cfg.get("signal", experiment.cfg.get('horizon', experiment.cfg.get('contact_refinement', experiment.cfg.get('progress', experiment.cfg.get('direction', experiment.cfg.get('sustain', {})))))).get("style_floor", 0.))
        discriminator.load_state_dict(checkpoint["amp_discriminator_state_dict"], strict=True)
        discriminator.eval()
    auditor, auditor_info = None, None
    if bool(args.auditor_checkpoint) != bool(args.auditor_sha256):
        raise ValueError("Frozen auditor requires a checkpoint and explicit SHA256")
    if args.auditor_checkpoint:
        if hashlib.sha256(args.auditor_checkpoint.read_bytes()).hexdigest() != args.auditor_sha256:
            raise ValueError("Frozen auditor SHA mismatch")
        state = torch.load(args.auditor_checkpoint, map_location="cpu", weights_only=True)
        for key in ("motion_sha256", "urdf_lf_sha256", "feature_fingerprint"):
            if state["amp_identity"][key] != experiment.identity()[key]:
                raise ValueError("Frozen auditor data/feature identity mismatch")
        auditor = AMPDiscriminator(experiment.spec, experiment.mean, experiment.std)
        auditor.load_state_dict(state["amp_discriminator_state_dict"], strict=True)
        auditor.eval()
        auditor_info = dict(sha256=args.auditor_sha256, identity=state["amp_identity"],
            completed_updates=state["completed_updates"],
            limitation="Same frozen scorer permits matched comparisons; not a calibrated quality score or probability. Source-policy bias remains.")
    vertices = foot_collision_vertices(experiment.kinematics.path)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(source_bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        identity=manifest["identity"], checkpoint_sha256=manifest["checkpoint_sha256"], modes={},
        frozen_auditor=auditor_info, duration_s=manifest["duration_s"],
        note="Post-2s metrics exclude injected initial velocity. Foot geometry is mesh height, not PhysX penetration depth. Ground-contact proxy uses >5 N Fz and mesh min height <2 cm; net force alone cannot distinguish self contacts. No automatic DR acceptance.")
    for mode in manifest["modes"]:
        rows = []
        for index in range(manifest["num_envs"]):
            data = episode(arrays, mode, index)
            result, geometry = analyze_episode(data, experiment, vertices, manifest["foot_names"], manifest["duration_s"], discriminator, auditor)
            result["env"] = index; rows.append(result)
            if index == args.env_index:
                folder = args.output/mode; folder.mkdir()
                if geometry is not None:
                    np.savez_compressed(folder/"geometry.npz", time=data["time"], **geometry)
                    plot_episode(data, geometry, folder/"curves.png")
                    if args.render:
                        result["render"] = render_episode(data, manifest, Path(experiment.kinematics.path), folder)
        report["modes"][mode] = dict(episodes=rows, survival_fraction=float(np.mean([row["survived"] for row in rows])))
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (args.output/"source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), modes={key: value["survival_fraction"] for key, value in report["modes"].items()})))


if __name__ == "__main__":
    main()
