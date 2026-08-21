# X1 armature configuration

`resources/robots/x1/config/joint_dynamics.json` is the single source of truth
for X1 armature values. Each entry contains:

- `nominal`: deterministic value for no-DR training and inference;
- `train_range`: per-environment sampling range when armature DR is enabled.

The Isaac Gym environment always initializes every DOF from `nominal`. If
`randomize_joint_armature` is true, samples from `train_range` replace the
nominal values at environment creation and reset. Both `play.py` and
`play_gm.py` disable armature DR and therefore use the same nominal values.

`sim2sim.py` loads the same file and writes nominal values to MuJoCo's
`model.dof_armature` by joint name. A missing or mismatched joint is a hard
error rather than an index-based silent mismatch.

The hip and knee nominal value `0.02505` is currently the midpoint of the
historical training range, not a system-identification result. Replace it when
identified actuator values become available. Old checkpoints trained with
armature disabled used zero armature; set `use_nominal_joint_armature = False`
when reproducing those historical dynamics exactly.
