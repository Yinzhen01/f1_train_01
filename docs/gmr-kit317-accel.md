# KIT317 12-DOF acceleration-weight controlled experiment

## Approved scope

User approved three training groups and publishing a new independent branch on
2026-09-12. Branch `experiment/gmr-kit317-accel-12dof`, worktree
`F:\robot_f1\x1-training-gmr-accel`, base `fa391e6` from swing.
Existing main, swing, smooth and 29-DOF branches are untouched.

| Registered task | Acceleration scale | Purpose |
|---|---:|---|
| x1_gmr_accel_1x | -1e-7 | Extra-training control, unchanged swing objective |
| x1_gmr_accel_3x | -3e-7 | Moderate acceleration cost |
| x1_gmr_accel_10x | -1e-6 | Strong acceleration cost |

Each group starts from ORIGINAL `TASK_20260911_171/model_8000.pt`, not another
group or smoke output. SHA256
`4c865c178a12a4796013b6b138972becb875301d7198543a6909c9a76ef7fe50`,
stored iteration 7999, source training code `a92dfe6346bcd1e853a1c923fec33667e9b601bb`.
All 33 network tensors and exploration standard deviation are strictly restored.
Both Adam optimizers are freshly initialized in ALL groups, including control.
Fixed learning rate 1e-5, seed 5, no DR. This is weight-based fine-tuning, not
exact optimizer or RNG continuation. One seed per group is screening, not a
statistically conclusive estimate of training variability.

## Single-variable contract

Only the existing `dof_acc` coefficient differs across groups. It penalizes
`sum_j ((dq_j(t)-dq_j(t-1))/control_dt)^2`; standard reward accumulation multiplies
the coefficient by control dt. Reward implementation, reset handling, 12-joint
sum, PD, armature, passive dynamics, torque limits, reference, commands,
observations/actions, policy architecture and every other reward remain unchanged.
No jerk, contact-impact, new per-joint scaling or extra motion filter is added.
Do not compare total returns directly across different reward scales.

The new environment adds read-only diagnostics before reset/history overwrite:
joint/ankle acceleration RMS, sampled peak, raw cost, weighted cost. Diagnostics
exclude first post-reset step; the inherited reward is deliberately unchanged.
These are 100 Hz velocity differences, NOT complete 1 kHz acceleration peaks.
The existing contact-force and action/torque diagnostics remain available.

## Execution and acceptance

- Dedicated entry: `humanoid/scripts/train_gmr_accel.py`.
- Hparams: `humanoid/envs/x1/x1_gmr_accel_config.py`.
- Each group: registered 512 environments / 20-update smoke, then a separate
  4096 environments / 1000-update training from original model8000, final model9000.
- Source task/hash, current exact commit, reference and URDF hashes are checked
  before learning; no missing-checkpoint fallback to scratch. Training is bounded
  to these two environment/budget combinations.
- Use only verified 4090D 24GB `ESKU000001`, image `BJX00000001/V000124`; re-read
  resource and task configuration before run. Same account unless an actual
  platform limit requires an authorized eligible existing account. No new account
  registration, recharge, unrelated task stop or resource-model fallback.
- Smoke gates: all groups actually run, finite/growing PPO logs, exact source and
  effective dynamics, only intended reward coefficient difference, normal finish,
  uploaded model8020 and local strict finite-state check. Formal training does not
  use the smoke outputs.
- After completion, evaluate every final model with the same deterministic t=0
  clip protocol and compare against BOTH frozen model8000 and 1x continuation.
- Compare acceleration RMS/quantiles/peaks, action/torque differences, contact
  force transients, foot-contact velocity, swing clearance, physical failures,
  walking displacement, root/joint/trunk tracking. Less motion is not a win.
- Follow-up high-rate inference logging is diagnostic only and must not change
  policy action rate or dynamics. Nominal single-clip results are not robustness
  or real-hardware safety evidence.

## Status

98 local tests passed, including recursive equality of every environment field
except the selected acceleration weight and all PPO fields except the log name.
The real model8000 restored all 33 tensors; next update is 8000, both optimizers
are empty, and actor/critic forward outputs are finite. Syntax checks passed.
Training code was published as `eccd14eed21ffe980c90abd7dc740d043e01fcaf`.
All three registered smoke jobs completed normally and uploaded model8020 and
the full configuration manifest. Local validation confirmed stored iter8019,
33 finite strictly loadable tensors, all 33 updated from source, finite actor /
critic outputs, identical effective source dynamics and all other reward scales.
Downloaded manifests differ only in the intended weight and run-name metadata.
This verifies execution and comparison setup, not better motion quality.

| Group | Passed smoke | Formal training | Snapshot at 2026-09-12 08:42 CST |
|---|---|---|---|
| 1x | TASK_20260912_030 | TASK_20260912_033 | Running; 80 / 1000 new updates |
| 3x | TASK_20260912_031 | TASK_20260912_034 | Running; 60 / 1000 new updates |
| 10x | TASK_20260912_032 | TASK_20260912_035 | Running; 60 / 1000 new updates |

All formal jobs actually entered PPO with the exact training commit, original
model8000 hash, 4096 environments and intended weights. No numerical error was
found in this startup snapshot. Re-query for current status. All are owned by
user4190 under PRO_20260820_014, with one 4090D each. Personal-storage mounting
must be empty on this resource; source checkpoint mounting is separate and
verified. The first smoke start was rejected before execution for the optional
personal mount; only that field was repaired, then the same task ran normally.

The existing thread heartbeat `gmr` now checks every 20 minutes, quietly while
normal, then validates final models and performs the approved identical inference
comparison. No final result or motion improvement is claimed yet. Keep the host
and app running for local follow-ups. Inference must record actual Isaac Gym
states; MuJoCo visualization of those states is not a Sim2Sim test.

Run records, sanitized logs, manifests, checksums and validation reports belong
in ignored `outputs/gradmotion-accel/`. `smoke-verification.json` and
`train-start-verification.json` preserve the completed gates. Source code is
frozen at the training commit; later documentation commits are not training code.
