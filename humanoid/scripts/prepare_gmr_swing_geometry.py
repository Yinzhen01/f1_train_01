"""Emit derived reward-only foot hull JSON to stdout; do not modify the reference."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from prepare_gmr_reference import origin, sha, stl_vertices


def prepare(root):
    urdf = root / "resources/robots/x1/urdf/X1_12DOF.urdf"
    reference = root / "resources/motions/gmr_kit317_upright_12dof.npz"
    tree = ET.parse(urdf).getroot()
    with np.load(reference, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        sole = data["sole_local"].copy()
    hulls, meshes = [], []
    for foot, name in enumerate(metadata["foot_names"]):
        collisions = tree.findall("link[@name='%s']/collision" % name)
        if len(collisions) != 1:
            raise ValueError("Expected exactly one foot collision mesh")
        shape = collisions[0]
        mesh = shape.find("geometry/mesh")
        relative = mesh.get("filename")
        path = (urdf.parent / relative).resolve()
        vertices = stl_vertices(path) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        transform = origin(shape.find("origin"))
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3] - sole[foot]
        selected = vertices[ConvexHull(vertices).vertices]
        hulls.append(selected.tolist())
        meshes.append(dict(urdf_relative_path=relative, sha256=sha(path)))
    return dict(schema=1, foot_names=metadata["foot_names"], reference_sha256=sha(reference),
                urdf_lf_sha256=metadata["training_urdf_lf_sha256"], meshes=meshes,
                sole_local=sole.tolist(), hull_counts=[len(h) for h in hulls],
                hulls_sole_centered=hulls,
                boundary="Derived reward geometry only; URDF, reference and dynamics unchanged")


if __name__ == "__main__":
    print(json.dumps(prepare(Path(__file__).resolve().parents[2]), separators=(",", ":")))
