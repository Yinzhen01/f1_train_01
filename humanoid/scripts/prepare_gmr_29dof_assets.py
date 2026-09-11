"""Import the user's V1.4 29-DOF URDF without changing physical parameters.

Only repair its invalid XML declaration and make mesh paths portable. Original
files are never modified. Hashes and the precise adaptation are recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(model_dir, output):
    source = model_dir / "urdf/f1_mock_sensor_29DOF_nohand.urdf"
    raw = source.read_text(encoding="utf-8").replace("version='1.3'", "version='1.0'")
    robot = ET.fromstring(raw)
    joints = [j for j in robot.findall("joint") if j.get("type") != "fixed"]
    if len(joints) != 29 or len({j.get('name') for j in joints}) != 29:
        raise ValueError("Expected 29 unique actuated joints")
    assets = {}
    for mesh in robot.findall(".//mesh"):
        filename = Path(mesh.get("filename")).name
        src = model_dir / "meshes" / filename
        if not src.is_file():
            raise FileNotFoundError(src)
        mesh.set("filename", "../meshes/" + filename)
        assets[filename] = src
    urdf = output / "urdf/F1_V1_4_29DOF.urdf"
    if urdf.exists():
        raise FileExistsError("Refusing to overwrite an existing model: " + str(urdf))
    urdf.parent.mkdir(parents=True, exist_ok=True)
    (output / "meshes").mkdir(exist_ok=True)
    for filename, src in assets.items():
        target = output / "meshes" / filename
        if target.exists() and sha(target) != sha(src):
            raise FileExistsError(target)
        if not target.exists():
            shutil.copy2(src, target)
        assert sha(target) == sha(src)
    xml_bytes = ET.tostring(robot, encoding="utf-8", xml_declaration=True)
    xml_bytes = b"\n".join(line.rstrip() for line in xml_bytes.splitlines()) + b"\n"
    urdf.write_bytes(xml_bytes)
    manifest = dict(source_urdf=str(source), source_sha256=sha(source),
                    output_urdf=urdf.name, output_sha256=sha(urdf),
                    output_lf_sha256=hashlib.sha256(urdf.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                    adaptations=["XML declaration 1.3 to valid 1.0", "relative mesh paths ../meshes/", "normalize XML whitespace"],
                    physical_parameters_changed=False,
                    joints=[j.get("name") for j in joints],
                    mesh_sha256={name: sha(src) for name, src in assets.items()})
    (output / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "mesh_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model_dir, args.output)
