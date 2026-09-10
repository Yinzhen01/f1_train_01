"""Render recorded Isaac Gym policy states using the exact training URDF in MuJoCo.

This does not run a MuJoCo dynamics rollout or alter recorded robot poses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from prepare_gmr_reference import stl_vertices, origin


def make_scene(urdf, output):
    # MuJoCo's URDF importer retains the original visual/collision transforms.
    model = mujoco.MjModel.from_xml_path(str(urdf))
    generated = output / 'training_urdf_import.xml'
    mujoco.mj_saveLastXML(str(generated), model)
    xml = ET.parse(generated)
    root = xml.getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((urdf.parent / '../meshes').resolve()))
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', name='ground_checker', type='2d', builtin='checker',
                  width='512', height='512', rgb1='.20 .24 .29', rgb2='.30 .35 .40')
    ET.SubElement(asset, 'material', name='ground_mat', texture='ground_checker',
                  texrepeat='16 16', texuniform='true', reflectance='.05')
    world = root.find('worldbody')
    body = world.find("body[@name='base_link']")
    if body is None:
        raise ValueError('URDF root body was not preserved')
    ET.SubElement(body, 'freejoint', name='render_root')
    ET.SubElement(world, 'geom', name='render_ground', type='plane', size='8 8 .1',
                  material='ground_mat', group='0')
    ET.SubElement(world, 'light', pos='1 -2 5', dir='-.2 .3 -1', diffuse='.85 .85 .85',
                  ambient='.3 .3 .3', castshadow='true')
    ET.SubElement(world, 'light', pos='-2 2 3', dir='.3 -.3 -1', diffuse='.4 .4 .4',
                  castshadow='false')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    global_element = visual.find('global')
    if global_element is None:
        global_element = ET.SubElement(visual, 'global')
    global_element.set('offwidth', '960')
    global_element.set('offheight', '720')
    scene = output / 'policy_render_scene.xml'
    xml.write(scene, encoding='utf-8', xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(scene)), scene


def foot_meshes(urdf, names):
    robot = ET.parse(urdf).getroot()
    result = []
    for name in names:
        shape = robot.find("link[@name='%s']/collision" % name)
        mesh = shape.find('geometry/mesh')
        vertices = stl_vertices((urdf.parent / mesh.get('filename')).resolve())
        vertices *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        transform = origin(shape.find('origin'))
        result.append(vertices @ transform[:3, :3].T + transform[:3, 3])
    return result


def label(image, text, x, y, color=(240, 240, 240), scale=.65):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(archive, manifest, urdf, output, episode=0, limit_frames=None):
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    digest = hashlib.sha256(urdf.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if digest != meta['urdf_lf_sha256']:
        raise ValueError('Rendering URDF differs from inference URDF')
    with np.load(archive, allow_pickle=False) as source:
        joint_names = source['joint_names'].astype(str).tolist()
        names = source['foot_names'].astype(str).tolist()
        prefix = 'episode_%d_' % episode
        data = {key[len(prefix):]: source[key].copy() for key in source.files if key.startswith(prefix)}
        normal_local = source['sole_normal_local'].copy()
        sole_local = source['sole_local'].copy()
        env_origin = source['env_origin'].copy()
    model, scene = make_scene(urdf, output)
    mjdata = mujoco.MjData(model)
    addresses = [int(model.joint(name).qposadr[0]) for name in joint_names]
    feet = [model.body(name).id for name in names]
    meshes = foot_meshes(urdf, names)
    root_adr = int(model.joint('render_root').qposadr[0])
    minimum_z, inclinations, fk_errors = [], [], []
    for i in range(len(data['time'])):
        root = data['root_state'][i]
        mjdata.qpos[root_adr:root_adr + 3] = root[:3]
        mjdata.qpos[root_adr + 3:root_adr + 7] = root[[6, 3, 4, 5]]
        mjdata.qpos[addresses] = data['dof_pos'][i]
        mujoco.mj_forward(model, mjdata)
        rot = mjdata.xmat[feet].reshape(2, 3, 3)
        pos = mjdata.xpos[feet].copy()
        minimum_z.append([float((mesh @ rot[f].T + pos[f])[:, 2].min()) for f, mesh in enumerate(meshes)])
        normals = np.einsum('fij,fj->fi', rot, normal_local)
        inclinations.append(np.rad2deg(np.arccos(np.clip(normals[:, 2], -1, 1))))
        if data['foot_diagnostics_valid'][i]:
            actual_sole = pos + np.einsum('fij,fj->fi', rot, sole_local)
            fk_errors.append(np.linalg.norm(actual_sole - data['sole_position'][i], axis=-1))
    minimum_z = np.asarray(minimum_z)
    inclinations = np.asarray(inclinations)
    fk_max = float(np.max(fk_errors)) if fk_errors else None
    # Small PhysX constraint drift can separate recorded body poses from ideal FK.
    if fk_max is not None and fk_max > .01:
        raise ValueError('MuJoCo/Isaac foot FK mismatch exceeds 1 cm: %g' % fk_max)
    report = {
        'role': 'MuJoCo rendering of Isaac Gym policy states; no MuJoCo dynamics',
        'episode': episode, 'scene': str(scene), 'training_urdf_lf_sha256': digest,
        'max_mujoco_vs_isaac_sole_position_error_m': fk_max,
        'minimum_foot_mesh_z_m': minimum_z.min(axis=0).tolist(),
        'maximum_geometric_penetration_m': np.maximum(0, -minimum_z.min(axis=0)).tolist(),
        'geometric_penetration_note': 'FK mesh minimum relative to z=0, not a PhysX contact-depth measurement.',
        'initial_sole_inclination_deg': inclinations[0].tolist(),
        'final_sole_inclination_deg': inclinations[-1].tolist(),
    }
    (output / 'render_geometry_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'render_geometry.npz', time=data['time'], minimum_z=minimum_z,
                        sole_inclination_deg=inclinations)
    option = mujoco.MjvOption()
    # URDF importer puts visual geometry in group 1 and collision geometry in 0.
    # Hide collision duplicates while keeping the ground visible.
    ground = model.geom('render_ground').id
    model.geom_group[ground] = 1
    option.geomgroup[:] = 0
    option.geomgroup[1] = 1
    if not np.any(model.geom_group[np.arange(model.ngeom) != ground] == 1):
        option.geomgroup[:] = 1
    cameras = []
    for azimuth in (90, 20):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = azimuth
        cam.elevation = -17
        cam.distance = 2.6
        cameras.append(cam)
    path = output / 'gmr_model5000_policy_dual_view.mp4'
    if path.exists():
        raise FileExistsError('Will not overwrite existing video: ' + str(path))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 50., (1600, 720))
    if not writer.isOpened():
        raise RuntimeError('VideoWriter could not open output')
    renderer = mujoco.Renderer(model, height=720, width=800)
    frame_indices = np.arange(0, len(data['time']), 2)
    if frame_indices[-1] != len(data['time']) - 1:
        # Always show a terminal state, even when it falls between 50 Hz frames.
        frame_indices = np.append(frame_indices, len(data['time']) - 1)
    if limit_frames:
        frame_indices = frame_indices[:limit_frames]
    target = data['root_state'][0, :3].copy()
    try:
        for n, i in enumerate(frame_indices):
            root = data['root_state'][i]
            mjdata.qpos[root_adr:root_adr + 3] = root[:3]
            mjdata.qpos[root_adr + 3:root_adr + 7] = root[[6, 3, 4, 5]]
            mjdata.qpos[addresses] = data['dof_pos'][i]
            mujoco.mj_forward(model, mjdata)
            target[:2] += (1 - np.exp(-.02 / .2)) * (root[:2] - target[:2])
            target[2] = .55
            panels = []
            for cam, title in zip(cameras, ('SIDE VIEW', 'FRONT OBLIQUE VIEW')):
                cam.lookat[:] = target
                renderer.update_scene(mjdata, camera=cam, scene_option=option)
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                label(frame, title, 22, 36)
                panels.append(frame)
            frame = np.concatenate(panels, axis=1)
            cv2.rectangle(frame, (0, 610), (1600, 720), (25, 30, 37), -1)
            label(frame, 'GMR KIT317 | model_5000 | DETERMINISTIC ISAAC GYM POLICY', 22, 644, scale=.72)
            label(frame, 't = %.2f / 4.60 s | 1x speed | MuJoCo visualization (no resimulation)' % data['time'][i], 22, 678)
            forces = data['foot_force'][i, :, 2]
            speed = np.linalg.norm(data['sole_velocity'][i, :, :2], axis=-1)
            if data['foot_diagnostics_valid'][i]:
                label(frame, 'CONTACT L/R: %s / %s' % tuple('ON' if f > 5 else 'OFF' for f in forces), 1050, 644)
                label(frame, 'Sole speed L/R: %.3f / %.3f m/s' % tuple(speed), 1050, 678)
            else:
                label(frame, 'INITIAL STATE', 1050, 644)
                label(frame, 'Contact data pending first physics step', 1050, 678)
            if data['physical_failure'][i]:
                label(frame, 'PHYSICAL TERMINATION', 460, 100, (70, 70, 255), 1.)
            writer.write(frame)
            if n in (0, len(frame_indices) // 2, len(frame_indices) - 1):
                cv2.imwrite(str(output / ('frame_%04d.jpg' % i)), frame)
    finally:
        renderer.close()
        writer.release()
    print(json.dumps(dict(video=str(path), frames=len(frame_indices), geometry=report), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--limit-frames', type=int)
    args = parser.parse_args()
    render(args.npz, args.manifest, args.urdf, args.output, args.episode, args.limit_frames)
