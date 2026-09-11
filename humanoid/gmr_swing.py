"""Pure Torch, world-frame swing diagnostics; no simulation or policy filtering."""
import hashlib
import json
from pathlib import Path

import torch


def load_geometry(path, reference_sha256, urdf_path, foot_names, device):
    geometry = json.loads(Path(path).read_text(encoding="utf-8"))
    if geometry["schema"] != 1 or geometry["foot_names"] != list(foot_names):
        raise ValueError("Swing geometry schema or left/right order mismatch")
    if geometry["reference_sha256"] != reference_sha256:
        raise ValueError("Swing geometry/reference mismatch")
    urdf = Path(urdf_path)
    digest = hashlib.sha256(urdf.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if geometry["urdf_lf_sha256"] != digest:
        raise ValueError("Swing geometry/URDF mismatch")
    for mesh in geometry["meshes"]:
        path = (urdf.parent / mesh["urdf_relative_path"]).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != mesh["sha256"]:
            raise ValueError("Swing geometry/mesh mismatch")
    # Repeated vertices pad the shorter hull without changing its minimum.
    hulls = [torch.tensor(v, dtype=torch.float32, device=device) for v in geometry["hulls_sole_centered"]]
    if len(hulls) != 2 or any(v.ndim != 2 or v.shape[1] != 3 or len(v) < 4
                              or not torch.isfinite(v).all() for v in hulls):
        raise ValueError("Invalid foot support hull")
    count = max(len(v) for v in hulls)
    points = torch.stack([torch.cat((v, v[:1].expand(count - len(v), -1)), dim=0) for v in hulls])
    return points, geometry


def minimum_height(sole_position, foot_quat_xyzw, centered_hulls):
    """Lowest COLLISION MESH vertex above z=0, not PhysX penetration depth.

    Convex-hull extrema equal full-mesh extrema under any rigid transform.
    The hull is in foot-link axes, translated to the calibrated sole center.
    """
    x, y, z, w = foot_quat_xyzw.unbind(-1)
    world_z_in_local = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                                   1 - 2 * (x.square() + y.square())), dim=-1)
    offsets = torch.einsum("efi,fvi->efv", world_z_in_local, centered_hulls).amin(dim=-1)
    return sole_position[..., 2] + offsets


def smoothstep(value):
    x = value.clamp(0., 1.)
    return x.square() * (3 - 2 * x)


def swing_envelope(contact, fps, boundary_s, ramp_s):
    """Reference-only gate; zero near BOTH boundaries of each off-contact run."""
    if fps <= 0 or boundary_s < 0 or ramp_s <= 0:
        raise ValueError("Invalid swing timing parameters")
    off = (contact.detach().cpu() < .1)
    gate = torch.zeros_like(contact, device="cpu")
    for foot in range(2):
        start = None
        for frame in range(len(off) + 1):
            active = frame < len(off) and bool(off[frame, foot])
            if active and start is None:
                start = frame
            if not active and start is not None:
                end = frame - 1
                indices = torch.arange(start, end + 1)
                distance = torch.minimum(indices - start, end - indices).float() / fps
                gate[start:end + 1, foot] = smoothstep((distance - boundary_s) / ramp_s)
                start = None
    return gate.to(contact.device)


def sample_envelope(envelope, times, fps):
    index = (times * fps).clamp(0., len(envelope) - 1.)
    lo = index.long()
    hi = (lo + 1).clamp(max=len(envelope) - 1)
    alpha = (index - lo).to(envelope.dtype)[:, None]
    return torch.lerp(envelope[lo], envelope[hi], alpha)


def swing_terms(actual_height, target_height, phase_gate, vertical_force,
                valid, settings):
    # The reference must independently require genuine clearance. A moving
    # heel/toe near the floor is not necessarily an unintended swing contact.
    height_gate = smoothstep((target_height - settings.gate_height_low_m) /
                             (settings.gate_height_full_m - settings.gate_height_low_m))
    gate = phase_gate * height_gate * valid[:, None]
    deficit = (target_height - actual_height - settings.height_tolerance_m).clamp_min(0.)
    normalized = (deficit / settings.height_sigma_m).square().clamp(max=settings.max_height_cost)
    contact = (vertical_force > settings.contact_threshold_n).to(actual_height.dtype)
    return {"gate": gate, "height_deficit": deficit,
            "unexpected_contact": contact * gate,
            "clearance_cost": (normalized * gate).sum(dim=1),
            "contact_cost": (contact * gate).sum(dim=1)}
