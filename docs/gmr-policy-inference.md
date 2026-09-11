# Finite GMR policy inference

## 2026-09-12 swing checkpoint inference and rendering

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

Output: `outputs/policy_videos/gmr_swing_task171_model8000/`.
The comparison baseline is the verified task116/model7000 inference below.
Local checks passed: 89 unit tests, strict loading of all 33 model tensors,
finite actor/critic forward outputs, and exact-URDF MuJoCo rendering backend.
User approved publication of this inference-only adaptation on 2026-09-12.

### Verified swing result

Registered inference `TASK_20260912_011` ran from 06:31:30 to 06:35:38 on
2026-09-12 Asia/Shanghai, with code `1cfdf8a4dd1311aec69a93044abd6a788f11eb0d`.
It completed normally. Artifact record 3924331 (`model_gmr_rollout.pt`) was
downloaded and validated, SHA256
`39333e1047a94358b640ad3d22c1774f3952b990c76160df1c7ae365865c3f42`.
Identity, monotonic timeline, terminal capture and recomputed manifest metrics
passed. PD, effective torque limits, armature, passive damping/friction, action
scale and control period match both source-training startup and model7000.

三次确定性回放均完整到达 4.6 秒，未触发物理失败；根部/关节轨迹完全一致，
不等于三个独立鲁棒性试验。实际水平位移 1.668 m，目标 1.741 m。
与 model7000 同口径比较：

| 指标 | model7000 | model8000 |
|---|---:|---:|
| 接触足底平均水平速度 | 0.03992 m/s | 0.02340 m/s（-41.4%） |
| 接触足底速度 P95 | 0.12289 m/s | 0.12338 m/s |
| 门控摆动期意外触地比例 | 16.79% | 14.65% |
| 门控摆动期平均高度不足 | 18.97 mm | 13.50 mm（-28.8%） |
| 根部位置 RMSE | 0.1093 m | 0.0797 m |
| 全关节角 RMSE | 0.08895 rad | 0.09723 rad |
| 躯干倾角误差均值 | 1.42° | 1.79° |
| 动作差分 RMS | 0.20548 | 0.20593 |
| 力矩差分 RMS | 3.893 N·m | 3.915 N·m |

平均足部指标有改善，但高速度瞬态、抖动指标没有明显改善，且姿态跟踪存在退步。
足底中心速度包含滚动，不是纯滑移距离；摆动期门控排除切换边界和参考近地样本，
高度不足包含 5 mm 容差。左/右足最大几何穿透约 0.719/0.469 mm，
并非 PhysX 接触深度。100 Hz 记录点没有力矩饱和，不涵盖 1 kHz PD 全部瞬态。

MuJoCo renders the actual Isaac Gym states, not the reference motion, and does
not reintegrate dynamics. Maximum cross-engine sole FK discrepancy is about
1.04 micrometers. First, middle and terminal frames were visually checked.
Delivery: `swing_12dof_model8000.mp4` (side/front-oblique, 1600x720, 50 fps,
231 frames, 4.62 s) and `swing_12dof_model8000_preview.gif`.
Detailed local evidence: `comparison_metrics.json`, `manifest.json`,
`rollout.npz`, `rendered/render_geometry_metrics.json`, and `推理渲染报告.md`.
This nominal single-motion evaluation is not a robustness or Sim2Real result.

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
