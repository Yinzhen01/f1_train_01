# Finite GMR policy inference

This is a deterministic evaluation of `TASK_20260910_187/model_5000.pt`, not
another training run or kinematic playback of the target reference. The source
training commit is `181f8034e62105f4138b90ebb560f63c0d1640a4`. Checkpoint SHA256:
`cf5b3c4267dd874eada01ba5b1dd7076cc6f0b852d2222101c21c6c548abc476`.
The file contains `iter=4999`: the runner stores a zero-based iteration index,
while `model_5000.pt` is named by the total number of updates.

## Protocol

- Run `humanoid/scripts/record_gmr_policy.py` through a registered Gradmotion
  task using the same Isaac Gym image and nominal dynamics as training.
- Mount the exact checkpoint through the platform; verify SHA256 and stored
  iteration before inference. Do not fall back to a different checkpoint.
- Use `x1_gmr_clip`, one environment, seed 5, three repeated episodes, fixed
  reference start at zero and deterministic `act_inference`. No PPO updates.
- Preserve the 4.6-second finite clip, the reference-derived commands, the
  phase encoding, PD gains, torque limits, action scaling and observation history.
- Start from the normal reset history at t=0. Log initial state separately;
  exclude its not-yet-refreshed rigid-body/contact measurements from foot metrics.
- Attach a read-only observer to termination checking. Capture post-physics
  states BEFORE automatic reset, including the physical failure/clip-end sample.
- Save root position/quaternion, named joint positions/velocities, torques,
  reference states, actual contact forces and sole-point velocities. Stop each
  episode at the first true terminal; do not loop or hide failures.
- Upload `model_gmr_rollout.pt`, containing an NPZ byte payload and JSON manifest.
  The manifest records source checkpoint, code commit, effective properties and
  per-episode metrics. Repetitions are not randomized robustness testing.

## Rendering and evidence boundary

`humanoid/scripts/render_gmr_policy.py` imports the exact training URDF into
MuJoCo, adds a free root and a display ground plane, and renders the recorded
states with a side and front-oblique camera. It calls `mj_forward`, not `mj_step`.
Therefore the dynamics evidence is Isaac Gym policy inference, while MuJoCo is
the visualization backend. This is NOT a MuJoCo Sim2Sim validation.

Rendering checks the training URDF hash, named joint mapping and foot FK against
the recorded Isaac Gym sole poses. Mesh minimum height is geometric penetration
relative to z=0, not the physics engine's reported contact penetration. Sole-point
horizontal speed includes foot rolling; it is not a pure slip distance.

Local tests and a render-model smoke are prerequisites, not a substitute for the
actual cloud inference. Do not report successful inference until the trajectory
artifact is uploaded and its terminal samples have been inspected.
