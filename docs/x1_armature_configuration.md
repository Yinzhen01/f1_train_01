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

The nominal values and DR ranges follow upstream branch
`x1-training-all-parameter`, with symmetric left/right values:

- hip pitch: nominal `0.208`, range `[0.1664, 0.2496]` kg m^2;
- hip roll: nominal `0.025`, range `[0.0001, 0.05]` kg m^2;
- hip yaw: nominal `0.0148`, range `[0.01184, 0.01776]` kg m^2;
- knee pitch: nominal `0.2728`, range `[0.21824, 0.32736]` kg m^2;
- ankle pitch: nominal `0.15`, range `[0.12, 0.18]` kg m^2;
- ankle roll: nominal `0.035`, range `[0.028, 0.042]` kg m^2.

The upstream comments mark hip roll as unidentified; its nominal is the center
used by upstream playback rather than an identified physical value. No-DR
training and standard inference both use these nominal values; disabling
armature randomization must never imply zero armature.
