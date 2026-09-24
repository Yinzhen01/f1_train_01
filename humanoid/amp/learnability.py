"""Fail-closed no-DR gate and staged budget, independent of Isaac Gym.

This verifies configuration, not physical runtime parameter readback or policy
effectiveness. Smoke success must never unlock domain randomization.
"""
from collections.abc import Mapping

NO_DR_SWITCHES = (
    "randomize_friction", "push_robots", "add_ext_force", "continuous_push",
    "randomize_base_mass", "randomize_com", "randomize_link_com",
    "randomize_base_inertia", "randomize_link_inertia", "randomize_gains",
    "randomize_torque", "randomize_link_mass", "randomize_motor_offset",
    "randomize_joint_friction", "randomize_joint_friction_each_joint",
    "randomize_joint_damping", "randomize_joint_damping_each_joint",
    "randomize_joint_armature", "randomize_joint_armature_each_joint",
    "randomize_coulomb_friction", "add_lag", "randomize_lag_timesteps",
    "randomize_lag_timesteps_perstep", "add_dof_lag", "randomize_dof_lag_timesteps",
    "randomize_dof_lag_timesteps_perstep", "add_dof_pos_vel_lag",
    "randomize_dof_pos_lag_timesteps", "randomize_dof_pos_lag_timesteps_perstep",
    "randomize_dof_vel_lag_timesteps", "randomize_dof_vel_lag_timesteps_perstep",
    "add_imu_lag", "randomize_imu_lag_timesteps", "randomize_imu_lag_timesteps_perstep",
    "enable_delivery",
)


def _items(value):
    if isinstance(value, Mapping):
        return dict(value)
    return {name: getattr(value, name) for name in dir(value) if not name.startswith("_")}


def assert_no_domain_randomization(cfg):
    """Accept nested config objects or their serialized manifest dictionaries."""
    cfg = _items(cfg)
    dr = _items(cfg["domain_rand"])
    nominal = "use_nominal_joint_armature"
    if dr.get(nominal) is not True:
        raise ValueError("No-DR learnability requires nominal armature, not zero armature")
    missing = set(NO_DR_SWITCHES) - set(dr)
    if missing:
        raise ValueError("Incomplete domain randomization configuration: " + ", ".join(sorted(missing)))
    # All current on/off options are bool. Reject truthy future boolean options
    # and non-boolean overrides of known switch prefixes (e.g. CLI integer 1).
    flags = {key: value for key, value in dr.items()
             if isinstance(value, bool) or key.startswith("randomize_") or key in NO_DR_SWITCHES}
    enabled = [key for key, value in flags.items() if key != nominal and value is not False]
    if enabled:
        raise ValueError("Domain randomization/delay is forbidden before effectiveness review: " + ", ".join(sorted(enabled)))
    if _items(cfg["noise"]).get("add_noise") is not False:
        raise ValueError("Observation noise must remain disabled during learnability verification")
    terrain = _items(cfg["terrain"])
    if terrain.get("mesh_type") != "plane" or terrain.get("curriculum") is not False:
        raise ValueError("Learnability verification requires the fixed plane without terrain curriculum")
    return dict(configuration_verified=True, stage="learnability_no_dr",
                domain_randomization=False, observation_noise=False,
                nominal_armature=True, flags=flags, dr_unlocked=False)


def validate_learnability_budget(cfg):
    if cfg.get("learnability_only") is not True:
        raise ValueError("Recovery configuration must remain learnability-only")
    if cfg.get("smoke") != {"num_envs": 32, "updates": 10}:
        raise ValueError("Unexpected recovery smoke budget")
    if cfg.get("formal") != {"num_envs": 4096, "updates": 1000}:
        raise ValueError("Initial no-DR training is capped at 1000 updates; extension requires review")
    if cfg.get("evaluation_updates") != [500, 1000]:
        raise ValueError("Independent evaluation checkpoints at 500/1000 updates are required")
    if cfg.get("max_experiment_rounds") != 20:
        raise ValueError("Experiment rounds and PPO updates are separate budgets")
    return True


def evaluation_checkpoint_name(completed_updates, evaluation_updates):
    """Use completed updates, not the legacy runner's zero-based file suffix."""
    if completed_updates in evaluation_updates:
        return "model_eval_%04d.pt" % completed_updates
    return None
