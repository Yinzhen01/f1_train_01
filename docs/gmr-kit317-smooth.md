# KIT317 12-DOF smoothness continuation

## Scope and lineage

User authorized the previously proposed 12-DOF continuation on 2026-09-11.
Independent branch `experiment/gmr-kit317-smooth-12dof`, worktree
`F:\robot_f1\x1-training-gmr-smooth`, base `dff2a0f`. The original upright
worktree's uncommitted skill/document edits are untouched. The separate
29-DOF training `TASK_20260911_112` is not edited or stopped.

Source: completed `TASK_20260911_008` (`9829d1d`), `model_5000.pt`, model
record `3919383`, SHA256
`5e89389e9eb3392759a78ff6295c2a395d3af0c8855c49499bf57fa8e7a8c5f9`.
Source saved internal iteration is 4999, representing 5000 completed updates.
The dedicated entry point starts at update index 5000, avoiding a duplicate
4999 update. Both smoke and formal continuation start from this original
checkpoint; the smoke is not a source for the formal run.

All 33 network tensors, including actor, critic, history encoder, velocity
estimator and learned exploration std, are restored strictly. Local real-source
loading confirmed bit-exact tensor equality; source mean std is 0.06639286,
not a reset to 0.5. Adam states are intentionally fresh because rewards change.
Learning rate is fixed at 1e-5 for both optimizers; the old final checkpoint's
PPO optimizer rate was 2.25e-5. This is weight-based fine-tuning, not a bit-exact
continuation of the old optimizer/random-number state.

## Changes and unchanged contracts

Task `x1_gmr_smooth` inherits `x1_gmr_upright`. The only environment-config
changes are two temporal reward coefficients; tests recursively compare all
other configuration fields. URDF, meshes, joint limits, nominal armature,
passive dynamics, PD, action scale, 100 Hz control, no-DR settings, observation
shape/semantics, default posture, reference/contact targets, trunk reward and
random phase reset distribution remain unchanged. No delivery filter is added.

Reference SHA256 remains
`d625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb`.
Actor observations: 47 x 66 = 3102; short history 235; critic 219; output 12.
The critic linear-velocity target index remains 199. This is not the original
pelvis-tilt 12-DOF reference and not the 29-DOF policy.

- `action_smoothness=-1.2` applied to the MEAN of first- and second-difference
  squared actions over 12 joints. This equals -0.1 times their sums, 10x the
  original -0.01 temporal coefficient. The old 0.05 L1 action-amplitude term
  is removed to avoid adding a default-pose preference.
- `gmr_torque_rate=-5e-5` applied to
  `mean(((tau_t-tau_prev)/(effective_torque_limit*dt))**2)`, before dt scaling.
- First differences and torque histories are valid only after the first
  post-reset step; second differences only after two previous actions exist.
- The standard accumulator multiplies all reward coefficients by dt=0.01 s.
- Debug snapshots precede environment reset and history overwrite: action/torque
  difference RMS, ankle saturation fraction, and contact-force peak. Samples
  are the last physics substep in each 100 Hz control interval, not all 1 kHz
  PD substeps. Logged peak-force scalars are averaged over an update.

## Execution and verification

Entry: `humanoid/scripts/train_gmr_smooth.py`; hparams:
`humanoid/envs/x1/x1_gmr_smooth_config.py`. It requires explicit task, resume,
source ID, checkpoint number and SHA256. A missing/mismatched mounted checkpoint
fails closed; no random initialization, download fallback or arbitrary latest
model selection. Model hash is checked before torch loading. The script bypasses
the generic directory loader only to enforce this strict source validation.

Registered Gradmotion only, owner 4190 / project `PRO_20260820_014`,
4090D 24GB `ESKU000001`, image `BJX00000001/V000124` after fresh resource readback.
Source checkpoint mount convention is the one verified by inference task 105.
No credentials or signed URLs are committed, no new account is registered.

- Local: 67 unit tests passed, changed Python files compile, diff whitespace
  check passed. Real `model_5000.pt` strict-loading test restored all 33 tensors.
- Planned registered smoke: 512 environments, 20 additional updates, seed 5;
  indices 5000..5019, final `model_5020.pt` (internal iter 5019).
- Formal after successful smoke: 4096 environments, 2000 additional updates,
  seed 5; indices 5000..6999, final `model_7000.pt` (internal iter 6999).
- Verify actual commit, source/hash/std, all loaded tensors, runtime parameters,
  fixed learning rate, active nonzero temporal rewards, finite increasing PPO
  logs, final smoke status and model upload before starting formal training.

Cloud tasks have not yet been submitted at this preparation snapshot. A local
unit test or smoke success cannot demonstrate reduced jitter or convergence.
Evaluate deterministic full-clip completion, trunk/foot tracking, slip and
action/torque variation after fine-tuning; changed total rewards cannot be
compared directly to the old objective. The old recorder has an explicit
task/checkpoint identity allowlist and must be extended before using model_7000.
