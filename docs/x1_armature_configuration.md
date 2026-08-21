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

Isaac Gym standard playback accepts only `--armature_mode=nominal`: it disables
armature DR and reads the shared nominal values. Randomized robustness
evaluation is a separate workflow rather than a playback mode, so a normal
render cannot silently change the evaluation dynamics. Zero-armature playback
is intentionally unsupported because those checkpoints are excluded from the
maintained experiment set.

Standard playback also forces the complete deterministic no-DR environment:
plane friction `0.6`, restitution `0`, observation noise off, pushes off,
rigid-body/actuator/joint randomization off, action and sensor lag off, and
ankle delivery randomization off. Both local `play.py` and cloud `play_gm.py`
use the same helper so a full-DR task cannot leak random settings into a normal
render. Randomized robustness evaluation must use a separately named task and
must report its sampled ranges and seed.

The hip and knee nominal value `0.02505` is currently the midpoint of the
historical training range, not a system-identification result. Replace it when
identified actuator values become available. No-DR training must still use the
nominal value; disabling armature randomization must never imply zero armature.
