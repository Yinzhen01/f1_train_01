# V1.4 GMR KIT317 whole-body training

## Status and lineage

Branch `experiment/gmr-kit317-29dof` was created at user request on 2026-09-11,
from `dff2a0fc8ac65b9da0076c80cdc59f5a059f9760`. Worktree:
`F:\robot_f1\x1-training-gmr-29dof`. The old 12-DOF worktree, its uncommitted
skill/document edits, completed tasks and checkpoints remain untouched.

This is a NEW 29-action policy trained from random initialization, not a resume
of `TASK_20260911_008/model_5000.pt`. The old model's actor input/output, critic
and state-estimator dimensions are incompatible. No partial weight transfer is
implemented or claimed. The previously proposed 12-DOF smoothness continuation
has not been launched.

User approved submission on 2026-09-11. Implementation commit
`9688e5ae4d0548ec5b4d8ba49cb57b9983168671` is pushed to
`Yinzhen01/f1_train_01`, branch `experiment/gmr-kit317-29dof`.
Registered Isaac Gym smoke `TASK_20260911_111` completed successfully;
formal task `TASK_20260911_112` was accepted for startup at 14:04:44 Asia/Shanghai.
At 14:09:13 it is actually running PPO: log iteration 23/5000, 1,179,648
timesteps. Task `commitId`, model/reference hashes, 2048 environments and all
runtime dynamics match the smoke-validated code. Query Gradmotion for newer state.
All 69 local tests pass, including the real 29-action network forward/backward,
actual observation builder and critic velocity slice, name-mapped PD/armature,
body-target FK, dense 100 Hz foot-mesh clearance and reset-safe reward formulas.
Changed Python files pass compilation and `git diff --check`.

## Model and reference

Task: `x1_gmr_29dof`. Robot: the user's F1 V1.4 model associated with this GMR
retargeting, NOT Unitree G1, not the fixed-waist X1_12DOF asset, and not an
unrelated legacy file with `29DOF` in its name.

- Robot: `resources/robots/f1_v1_4/urdf/F1_V1_4_29DOF.urdf`.
- Source: `F:\f1_sim\07_PHUMA\asset\humanoid_model\f1\v1.4\urdf\f1_mock_sensor_29DOF_nohand.urdf`.
- Import repairs invalid XML declaration `1.3` to `1.0` and changes mesh paths
  to portable relative paths. Origins, axes, limits, inertias and passive
  damping/friction remain the source URDF values. Mesh copies are SHA256 checked.
- Provenance: `resources/robots/f1_v1_4/asset-manifest.json` (57 referenced meshes).
- Reference: `resources/motions/gmr_kit317_foot_flat_29dof.npz` and its JSON report.
- Source: GMR `KIT_317_walking_medium08_f1_v1_4_foot_flat.npz`, SHA256
  `8458117d35eaf773c2986c30d104446e56197cb39dd4980a0809b818bbec9ddd`.

The 139 frames at 30 Hz span 4.6 seconds. All 29 named joint trajectories are
retained: legs 12, arms 12 (six per arm, no wrist roll), waist 3 and neck 2.
The source neck targets are zero because GMR had no head task; those two joints
are still actively controlled, not mechanically fixed or human-head imitation.
The reference includes chest/wrist body poses derived from the actual 29-DOF
URDF, foot poses/contact labels and root/joint velocities. It does not reuse the
fixed-waist 12-DOF `upright_chest_v1` transformation.

Independent FK agrees with source GMR joint-body positions within
`3.48e-7 m` and rotations within `8.92e-7 rad` over all source frames. Joint
positions, exported velocities and interpolation slopes pass URDF limits.
Standing sole tilt at source frames is below `0.00336 degrees`; sampled foot
mesh clearance is at least 2 mm. These are kinematic tests, not balance/contact
or torque-feasibility guarantees. Linear interpolation is not temporal smoothing.

## Observation and control

Actor: 98 features/frame × 66 history frames = 6468. Short history: 490.
Critic: 141 features/frame × 3 = 423. Its latest linear-velocity target starts
at index 403 (not the old 12-DOF index 199). Policy output: 29 joint-position
offsets from the reference first-frame posture, action scale 0.5 rad/unit.
This is not residual control around the time-varying reference.

Control period remains 0.01 s (sim dt 0.001 s × decimation 10). Nominal no-DR,
no observation noise, no action/IMU lag, no added delivery filter. Hard effort
limits and passive joint dynamics come from the V1.4 URDF, so dynamics are NOT
identical to the prior 12-DOF model. Shared named armature configuration:
`resources/robots/f1_v1_4/config/joint_dynamics.json`. Legs retain the previous
nominal convention; upper-body armature 0.01 follows the V1.4 simulation MJCF
default. Neither is a new system-identification result.

PD Kp/Kd (Nm/rad, Nm s/rad): inherited legs 30/3, 40/3, 35/4, 100/8,
35/1.5, 35/1.5; waist 100/4; shoulders/elbows 40/2; wrists/neck 5/0.5.
Upper-body PD values are initial simulation tuning choices, not verified
hardware settings. The smoke read back positive PD/armature on all 29 joints;
its last-substep ankle saturation metric was zero across all 20 updates.
Continue inspecting saturation and early falls in the longer run. MuJoCo is used only for offline model/FK checks;
an aligned 29-DOF sim2sim deployment asset/pipeline is not delivered here.

## Rewards

Inherited foot/root/contact/velocity/slip and joint-limit rewards remain active.
Joint-position reward is split by group so quiet upper-body joints cannot
artificially improve the reported leg tracking: legs weight 6, arms 2, waist 1,
neck 0.25, each `exp(-mean(group_error_rad^2)/0.04)`. Chest world orientation
and hand base-relative position have weight 1 each; chest orientation is not
approximated by pelvis orientation.

- Action smoothness: weight -1.2, `mean((a_t-a_prev)^2)` plus
  `mean((a_t-2*a_prev+a_prevprev)^2)`. No L1 action-amplitude/default-pose bias.
- Torque rate: weight -5e-5,
  `mean(((tau_t-tau_prev)/(torque_limit*dt))^2)`, units 1/s² before weighting.
- Both are multiplied by dt by the standard reward accumulator. Histories are
  gated for the first one/two steps following reset; no cross-episode penalty.
- Torque samples are the last physics substep of each 100 Hz control interval,
  NOT all 1000 Hz torque samples. This penalty alone cannot rule out substep chatter.

The new temporal penalty weights are starting choices for this new 29-DOF
experiment. Cloud smoke confirmed both reward functions execute with nonzero
penalties, but has not demonstrated improved deterministic video smoothness.
Debug scalars capture action/torque differences BEFORE history overwrite and
reset, plus ankle saturation and contact-force peaks. Compare group RMSE,
completion/fall rate, slip and these physical metrics, not total reward against
the differently weighted 12-DOF run.

## Execution gates

```bash
# Run only as a registered Gradmotion task when using the cloud.
python humanoid/scripts/train.py --task=x1_gmr_29dof --headless --num_envs=128 --max_iterations=20 --seed=5 --run_name=kit317_29dof_smoke
python humanoid/scripts/train.py --task=x1_gmr_29dof --headless --num_envs=2048 --max_iterations=5000 --seed=5 --run_name=kit317_29dof_baseline
```

5000 updates / 2048 environments are the initial baseline, not a convergence
guarantee. Cloud submission requires explicit commit/push approval, exact
remote/branch verification, resource readback for 4090D 24 GB ESKU000001 and a
compatible image, then registered smoke. Check actual commit,
29-joint mapping, reference/asset hash, nonzero PD/armature for all joints,
active rewards, finite increasing PPO logs, checkpoint upload and final status.
Start formal training only after smoke passes. No SSH/background training bypass.

The URDF importer reports duplicate unnamed visual/collision geometry warnings
in MuJoCo; model loading and independent FK still pass. Retain source collision
coverage instead of inventing new collision bodies. This is not a whole-body
self-collision certification; verify PhysX import/contact behavior during smoke.

## Registered cloud validation (2026-09-11, Asia/Shanghai)

- Owner user ID `4190`, project `PRO_20260820_014`, resource `ESKU000001`
  confirmed as 1 x 4090D 24GB; image `BJX00000001/V000124`.
- Smoke `TASK_20260911_111`: scratch, 128 environments, 20 PPO updates,
  seed 5. Task status 5, end time 14:01:08. Platform `commitId` matches
  `9688e5ae4d0548ec5b4d8ba49cb57b9983168671`; no resume fields are populated.
- Logs contain iterations 0 through 19, 61,440 timesteps, and training-completed
  marker without NaN/Inf, Traceback, RuntimeError or CUDA OOM. Runtime logs
  verify 29 ordered joints, 98/141 frame dimensions, all PD and actual PhysX
  armature values, effective torque limits, and all active reward weights.
- Actual URDF SHA256:
  `79a51ab2e8e111c0cd0e1167b203ef7df02cdcc504a52ae7a117e07f06d03be0`.
  Reference SHA256:
  `08730ae2264bc4598332a1f0e5adb572442662192e3bad564870350f0969c3b6`.
- `model_20.pt` uploaded at 14:01:19, model record `3919718`, SHA256
  `b804e92d5f43197a10a8c63dadd54f45cce2cd0d37e6fd2d15febb120fda2321`.
  Downloaded checkpoint has 33 finite model tensors, actor output 29 and critic
  input 423. Its saved internal iteration is 19 (runner convention), not a
  missing twentieth update.
- Last smoke log mean reward 23.31, mean episode length 287.16 control steps
  (about 2.87 s). These are stochastic training aggregates, not deterministic
  full-clip success; random phase resets and early termination affect them.
- Twenty-update diagnostic means: action-difference RMS 0.70675 action units,
  torque-difference RMS 17.99 Nm. Exploration std is still approximately 0.5.
  Last-substep ankle saturation fraction is zero in all twenty logged points.
  Maximum logged contact-force-peak metric is 1953.12 N; this metric averages
  per-step maxima within each update and is NOT the absolute physics peak.
- Formal `TASK_20260911_112`: scratch, 2048 environments, 5000 updates, seed 5,
  same branch/model/reference/rewards and 4090D image; no 12-DOF weight reuse.
  At 14:09:13, status 3, actual `commitId` is `9688e5a`, log iteration 23/5000
  with 1,179,648 timesteps and finite losses. Runtime PD, armature, passive
  dynamics, torque limits and action scale exactly match the smoke runtime.
  Worker: `rl-prod-4190-task-20260911-112-r7j69-run-rl-prod-3562545292`.
  Initial log ETA approximately 2.1 hours is provisional, not a completion time.
- Ignored local artifacts: `outputs/gradmotion-29dof/` contains smoke/startup logs,
  smoke checkpoint and `run-provenance.json`. Temporary task-creation JSON files were
  removed immediately after successful creation; credentials are not copied.

This closes the runtime smoke gate only. It does not establish convergence,
reduced jitter, whole-body self-collision safety, sim2sim equivalence or hardware
readiness. The 12-DOF smoothness continuation remains unstarted and untouched.
