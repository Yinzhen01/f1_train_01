"""Read-only numerical/kinematic diagnostics; never a dynamic acceptance test."""
import numpy as np


def derivative_metrics(x, dt=.01):
    result = {}
    y = np.asarray(x, dtype=float)
    for name in ("velocity", "acceleration", "jerk"):
        y = np.diff(y, axis=0) / dt
        result[name] = {"abs_peak": float(np.max(np.abs(y))),
                        "rms": float(np.sqrt(np.mean(y * y))),
                        "abs_p99": float(np.percentile(np.abs(y), 99))}
    return result


def support_proxy(centers, support, dt=.01):
    """Full source_support > .5 runs, not force contacts or prior flat-only mask."""
    result = {}
    for side, label in enumerate(("left", "right")):
        mask = np.asarray(support[:, side]) > .5
        edges = np.diff(np.r_[False, mask, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        runs = []
        for start, end in zip(starts, ends):
            if end - start < 2:
                continue
            xy = centers[start:end, side, :2]
            lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            runs.append(dict(start_frame=int(start), end_frame_inclusive=int(end - 1),
                             duration_s=(int(end - start) - 1) * dt,
                             xy_path_mm=float(lengths.sum() * 1000),
                             xy_displacement_mm=float(np.linalg.norm(xy[-1] - xy[0]) * 1000),
                             max_speed_m_s=float(lengths.max() / dt)))
        result[label] = {"runs": runs, "max_path_mm": max([r["xy_path_mm"] for r in runs], default=0.)}
    return result


class SourceModelAudit:
    """Full mesh/FK recomputation on immutable GMR model (no simulation steps)."""
    def __init__(self, xml_path, spec):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.spec = spec
        if self.model.nq != 36:
            raise ValueError("Unexpected source model nq")
        self.body_ids = [self.model.body(n).id for n in spec.body_names]
        self.root_id = self.model.body(spec.root_body).id
        mujoco.mj_forward(self.model, self.data)
        self.feet = []
        for side in ("left", "right"):
            bid = self.model.body(side + "_ankle_roll_link").id
            geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid
                     and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
            if not geoms:
                raise ValueError("Missing source foot mesh")
            # This MJCF contains identical visual/collision copies. Verify that
            # they really coincide, then use one copy (as the source sole fit did).
            gid = geoms[0]
            for other in geoms[1:]:
                if (self.model.geom_dataid[other] != self.model.geom_dataid[gid]
                        or not np.allclose(self.model.geom_pos[other], self.model.geom_pos[gid])
                        or not np.allclose(self.model.geom_quat[other], self.model.geom_quat[gid])):
                    raise ValueError("Multiple distinct meshes require a combined sole audit")
            mid = self.model.geom_dataid[gid]
            a, n = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            vertices = self.model.mesh_vert[a:a + n].copy().astype(float)
            world = vertices @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid]
            bottom = world[world[:, 2] <= world[:, 2].min() + .0002]
            local_center = self.data.xmat[bid].reshape(3, 3).T @ (bottom.mean(0) - self.data.xpos[bid])
            self.feet.append((bid, gid, vertices, local_center))

    def evaluate(self, clip):
        z = clip.arrays
        # Map hinge addresses by name instead of assuming MuJoCo source ordering.
        addresses = [int(self.model.joint(n).qposadr[0]) for n in clip.all_joint_names]
        keys, heights, centers = [], [], []
        for row in z["qpos"]:
            self.data.qpos[:7] = row[:7]
            self.data.qpos[addresses] = row[7:]
            self.mj.mj_forward(self.model, self.data)
            root_r = self.data.xmat[self.root_id].reshape(3, 3)
            keys.append((self.data.xpos[self.body_ids] - self.data.xpos[self.root_id]) @ root_r)
            frame_h, frame_c = [], []
            for bid, gid, vertices, local in self.feet:
                r = self.data.geom_xmat[gid].reshape(3, 3)
                # Only z is needed for a full vertex mesh minimum.
                frame_h.append(float((vertices @ r[2] + self.data.geom_xpos[gid, 2]).min()))
                frame_c.append(self.data.xpos[bid] + self.data.xmat[bid].reshape(3, 3) @ local)
            heights.append(frame_h)
            centers.append(frame_c)
        heights, centers, keys = np.asarray(heights), np.asarray(centers), np.asarray(keys)
        error = np.linalg.norm(keys - clip.key_positions_b, axis=-1)
        return dict(source_mesh_min_height_mm=float(heights.min() * 1000),
                    saved_height_max_error_m=float(np.abs(heights - z["sole_min_height"]).max()),
                    saved_sole_center_max_error_m=float(np.abs(centers - z["sole_center"]).max()),
                    source_vs_training_keypoint_peak_mm={n: float(error[:, i].max() * 1000)
                                                         for i, n in enumerate(self.spec.body_names)},
                    support_proxy=support_proxy(centers, z["source_support"]))

    def model_differences(self, training_fk):
        result = []
        for joint in training_fk.joints:
            if joint["kind"] == "fixed":
                continue
            n = joint["name"]
            idx = self.spec.joint_names.index(n)
            mjj = self.model.joint(n)
            bid = int(self.model.jnt_bodyid[mjj.id])
            from scipy.spatial.transform import Rotation
            wxyz = self.model.body_quat[bid]
            rotation = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
            result.append(dict(joint=n,
                               origin_translation_difference_mm=float(np.linalg.norm(self.model.body_pos[bid] - joint["xyz"]) * 1000),
                               origin_rotation_difference_deg=float(np.degrees(Rotation.from_matrix(joint["rotation"].T @ rotation).magnitude())),
                               axis_difference=float(np.linalg.norm(self.model.jnt_axis[mjj.id] - joint["axis"])),
                               source_limits_rad=self.model.jnt_range[mjj.id].tolist(),
                               training_limits_rad=training_fk.limits[idx, :2].tolist()))
        return result


def clip_quality(clip, training_fk, source_audit):
    z, spec = clip.arrays, clip.spec
    lo, hi, vmax = training_fk.limits.T
    pos = clip.joint_pos
    tolerance = 1e-6
    violations = (pos < lo - tolerance) | (pos > hi + tolerance)
    v = np.diff(pos, axis=0) * spec.fps
    q = z["qpos"][:, 3:7]
    by_version = {}
    indices = [7 + clip.all_joint_names.index(n) for n in spec.joint_names]
    for key in ("qpos", "raw_qpos", "pre_lift_qpos"):
        by_version[key] = dict(joints=derivative_metrics(z[key][:, indices]),
                               root_z=derivative_metrics(z[key][:, 2]))
    return dict(clip_id=clip.id, frames=clip.frames, fps=100, duration_s=clip.duration,
                all_numeric_finite=all(np.isfinite(v).all() for v in z.values() if np.issubdtype(v.dtype, np.number)),
                quaternion_norm_max_error=float(np.abs(np.linalg.norm(q, axis=-1) - 1).max()),
                quaternion_sign_flips=int((np.sum(q[1:] * q[:-1], axis=-1) < 0).sum()),
                training_joint_limit_violations=int(violations.sum()),
                training_frames_with_limit_violations=int(violations.any(-1).sum()),
                training_limit_excess_peak_rad=float(np.maximum(np.maximum(lo - pos, pos - hi), 0).max()),
                violating_joints={n: int(violations[:, i].sum()) for i, n in enumerate(spec.joint_names) if violations[:, i].any()},
                training_velocity_limit_violations=int((np.abs(v) > vmax + tolerance).sum()),
                derivatives=by_version, max_root_z_lift_mm=float(z["root_z_lift"].max() * 1000),
                root_z_lift=derivative_metrics(z["root_z_lift"]),
                noncyclic_boundary=dict(root_xyz_end_minus_start_m=(z["qpos"][-1, :3] - z["qpos"][0, :3]).tolist(),
                                        joint_end_minus_start_rad=(pos[-1] - pos[0]).tolist(),
                                        legal_start_frames=[clip.first_start, clip.last_start],
                                        legal_windows=clip.last_start - clip.first_start + 1),
                source_geometry=source_audit.evaluate(clip),
                training_ready=False, dynamic_validation="not_performed")
