# Finite GMR policy inference

## 2026-09-12 swing checkpoint inference preparation

Training `TASK_20260911_171` completed normally at 2026-09-11 17:13:28
Asia/Shanghai. Final `model_8000.pt` (model record 3922306) is 10,323,330 bytes,
SHA256 `4c865c178a12a4796013b6b138972becb875301d7198543a6909c9a76ef7fe50`.
The downloaded checkpoint contains iteration 7999 and 33 finite network tensors.
Its training commit is `a92dfe6346bcd1e853a1c923fec33667e9b601bb`.

The `x1_gmr_swing` identity entry binds this final checkpoint to the unchanged
12-DOF URDF and upright reference. Inference uses the existing one-environment,
seed-5, three-episode, deterministic t=0 protocol, with no learning updates.
Only identity validation and the video label are adapted; training rewards,
observations, policy weights and dynamics are not modified.

Planned output: `outputs/policy_videos/gmr_swing_task171_model8000/`.
The comparison baseline is the verified task116/model7000 inference below.
At this preparation stage no cloud inference or improvement is claimed.
Local checks passed: 89 unit tests, strict loading of all 33 model tensors,
finite actor/critic forward outputs, and exact-URDF MuJoCo rendering backend.
User approved publication of this inference-only adaptation on 2026-09-12.

## 2026-09-11 final checkpoint evaluation

The new evaluation on this branch is bound to `TASK_20260911_116/model_7000.pt`
(stored iteration 6999), SHA256 `ccaf51ecb069e6ecb80ee9fe07eda4a63dab627e4592d15b4240b5d024701857`.
Training finished normally at 2026-09-11 15:15:58 Asia/Shanghai and the final
artifact was downloaded and checked (33 finite network tensors).
The identity allowlist binds source task, checkpoint number/hash, action dimension,
reference hash and training URDF hash. Evaluation preserves the training environment,
uses one environment, seed 5, three deterministic episodes from time zero, and
performs no learning updates. Physical failures remain visible in terminal frames.

The recorder now logs the chest quaternion separately when the waist is movable;
pelvis rotation is not a chest tilt metric for the 29-DOF model. Action and torque
differences exclude the reset sample. MuJoCo labels and filenames use the actual
checkpoint number, and named joints must cover the rendering model exactly.
Local validation: 71 tests passed; exact training URDF loaded and rendered
in MuJoCo. These checks are not a completed cloud inference result.

### Verified cloud result

Registered inference `TASK_20260911_129` completed normally at 16:16:40
Asia/Shanghai with code `cafa4ea72c662c987f99446491882ab904fa2939`. The three recordings
have identical root/joint trajectories, each 461 samples at 100 Hz, without
physical termination. Repeated identical deterministic episodes are not independent
robustness trials. Artifact identity, iteration, timeline and URDF were checked.
PD, effective torque limits, armature, passive damping/friction, action scale and
control period match the corresponding training startup runtime log.

三次完整 4.6 秒 deterministic rollout 均未触发物理失败。实际水平位移约 1.66 m，目标约 1.74 m。与原 008/model_5000 相同动力学的推理比较，动作差分 RMS 从 0.24071 降至 0.20548（-14.6%），力矩差分 RMS 从 4.4586 降至 3.8934 N·m（-12.7%）。但接触足底平均水平速度从 0.02997 增至 0.03992 m/s，不能声称地滑也改善。根部位置 RMSE 0.1093 m，胸部倾角误差均值 1.42°。100 Hz 记录中无力矩饱和，最大上限利用率 71.0%；不覆盖 1 kHz PD 内所有瞬态。

Output directory: `outputs/policy_videos/gmr_smooth_task116_model7000/`.
Delivery MP4: `smooth_12dof_model7000.mp4`; preview: `smooth_12dof_model7000_preview.gif`.
Both views are rendered at 1600x720, 50 fps, 231 frames (4.62 seconds including
the final t=4.60 sample). First, middle and terminal frames were visually checked.
MuJoCo/Isaac sole FK discrepancy is below 1 micrometer.
The NPZ and manifest are actual policy dynamics from Isaac Gym, not target playback.
MuJoCo visualizes those states and does not rerun their dynamics.

The sections below describe the original protocol and historical checkpoint.

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
