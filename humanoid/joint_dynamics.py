"""Shared robot joint-dynamics configuration loader.

The external JSON file is the source of truth used by Isaac Gym training,
Isaac Gym playback, and MuJoCo inference.  Values are mapped by joint name so
that a changed DOF order cannot silently attach an armature to the wrong joint.
"""

import json
import math
import os
from functools import lru_cache

from humanoid import LEGGED_GYM_ROOT_DIR


ARMATURE_INFERENCE_MODES = ("training", "nominal", "zero")


def resolve_robot_config_path(path):
    """Resolve a config path that may contain the project-root placeholder."""
    if not path:
        raise ValueError("joint armature config path is empty")
    resolved = path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    return os.path.abspath(resolved)


def _validate_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
    return value


@lru_cache(maxsize=None)
def load_joint_armature_config(path):
    """Load and validate the shared per-joint armature configuration."""
    resolved_path = resolve_robot_config_path(path)
    with open(resolved_path, "r", encoding="utf-8") as config_file:
        raw = json.load(config_file)

    if raw.get("schema_version") != 1:
        raise ValueError(
            f"unsupported joint dynamics schema in {resolved_path}: "
            f"{raw.get('schema_version')!r}"
        )

    armature = raw.get("armature")
    if not isinstance(armature, dict):
        raise ValueError(f"missing armature object in {resolved_path}")

    joint_order = armature.get("joint_order")
    joints = armature.get("joints")
    if not isinstance(joint_order, list) or not joint_order:
        raise ValueError(f"armature.joint_order must be a non-empty list in {resolved_path}")
    if len(joint_order) != len(set(joint_order)):
        raise ValueError(f"armature.joint_order contains duplicate names in {resolved_path}")
    if not isinstance(joints, dict) or set(joints) != set(joint_order):
        raise ValueError(
            f"armature.joints keys must exactly match armature.joint_order in {resolved_path}"
        )

    normalized_joints = {}
    for joint_name in joint_order:
        entry = joints[joint_name]
        if not isinstance(entry, dict):
            raise ValueError(f"armature entry for {joint_name} must be an object")
        nominal = _validate_number(entry.get("nominal"), f"{joint_name}.nominal")
        train_range = entry.get("train_range")
        if not isinstance(train_range, list) or len(train_range) != 2:
            raise ValueError(f"{joint_name}.train_range must contain [low, high]")
        low = _validate_number(train_range[0], f"{joint_name}.train_range[0]")
        high = _validate_number(train_range[1], f"{joint_name}.train_range[1]")
        if low > high:
            raise ValueError(f"{joint_name}.train_range low exceeds high")
        if nominal < low or nominal > high:
            raise ValueError(f"{joint_name}.nominal is outside its train_range")
        normalized_joints[joint_name] = {
            "nominal": nominal,
            "train_range": (low, high),
        }

    return {
        "path": resolved_path,
        "robot": raw.get("robot", ""),
        "joint_order": tuple(joint_order),
        "joints": normalized_joints,
    }


def armature_values_for_dof_order(config, dof_names):
    """Return nominal values and train ranges in the simulator's DOF order."""
    dof_names = tuple(dof_names)
    configured_names = set(config["joint_order"])
    actual_names = set(dof_names)
    if configured_names != actual_names:
        missing = sorted(actual_names - configured_names)
        extra = sorted(configured_names - actual_names)
        raise ValueError(
            "joint armature config does not match simulator DOFs: "
            f"missing={missing}, extra={extra}"
        )

    nominal = [config["joints"][name]["nominal"] for name in dof_names]
    train_ranges = [config["joints"][name]["train_range"] for name in dof_names]
    return nominal, train_ranges


def configure_inference_armature(env_cfg, mode):
    """Select the Isaac Gym armature behavior used during inference.

    ``training`` preserves the task configuration. It cannot reconstruct a
    historical setting that was never stored in a checkpoint; use ``zero``
    explicitly for checkpoints known to have been trained with zero armature.
    """
    mode = str(mode).lower()
    if mode not in ARMATURE_INFERENCE_MODES:
        raise ValueError(
            f"unsupported armature inference mode {mode!r}; "
            f"expected one of {ARMATURE_INFERENCE_MODES}"
        )

    domain_rand = env_cfg.domain_rand
    if mode == "nominal":
        config_path = getattr(domain_rand, "joint_armature_config_file", "")
        if not config_path:
            raise ValueError(
                "armature_mode=nominal requires joint_armature_config_file"
            )
        domain_rand.use_nominal_joint_armature = True
        domain_rand.randomize_joint_armature = False
    elif mode == "zero":
        domain_rand.use_nominal_joint_armature = False
        domain_rand.randomize_joint_armature = False
        env_cfg.asset.armature = 0.0

    return mode
