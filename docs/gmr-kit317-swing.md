# KIT317 12-DOF swing execution experiment

## Scope and source

User approved stage-one swing execution changes and training on 2026-09-11,
and separately approved publishing this independent experiment branch.
Branch: `experiment/gmr-kit317-swing-12dof`; base `af2ec05`.
Worktree: `F:\robot_f1\x1-training-gmr-swing`.
The smooth and 29-DOF worktrees/branches remain unchanged.

Restore completed `TASK_20260911_116/model_7000.pt`, record 3919932,
SHA256 `ccaf51ecb069e6ecb80ee9fe07eda4a63dab627e4592d15b4240b5d024701857`.
Internal iteration 6999 represents 7000 completed PPO updates. Strictly restore
all 33 network tensors, including critic, history encoder, velocity estimator
and exploration std. Both Adam optimizers start fresh because rewards changed.
Fixed learning rate 1e-5. This is weight-based fine-tuning, not exact optimizer
or random-state continuation. A missing/wrong source cannot fall back to scratch.

## Exactly two new reward terms

Task `x1_gmr_swing` inherits `x1_gmr_smooth`. Preserve all previous reward
coefficients, observation/action semantics, random phase starts, reference,
URDF, PD, armature, passive dynamics, friction, no-DR and control timing.
Tests recursively compare all inherited environment/PPO configuration fields.
Do NOT add stance anchors, landing impact penalties or delivery filters here.

Reference NPZ SHA256 remains
`d625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb`.
The derived JSON is reward geometry, not a new motion or physical asset.
It contains full collision-mesh convex hull vertices in foot-link axes relative
to the existing sole center (1067 left / 975 right). A minimum over a convex
hull equals the minimum over its source mesh under a rigid transform. Validate
URDF, mesh, reference hashes and sole calibration before training. Target foot
poses use the same interpolated reference as existing foot rewards; they are
not reconstructed from the actual root or actual joint state.

All lengths below are world-z meters above the existing plane (not measured
PhysX contact penetration). Quaternions are xyzw. For each foot:

1. Find reference off-contact runs (`foot_contact < 0.1`). Gate is zero within
   40 ms of the first/last off-contact sample, then smoothstep-ramped over 60 ms.
   Interpolate the resulting 30 Hz envelope at the current 100 Hz motion time.
2. Multiply by a smooth reference-height gate: zero at/below 5 mm, one at/above
   15 mm. Thus near-floor rolling/toe-off targets are not forced into clearance.
3. Disable new terms during the first post-reset step (`episode_length <= 1`).
4. Clearance deficit `d = max(h_ref - h_actual - 0.005, 0)`; per-foot cost
   `gate * min((d / 0.02)^2, 9)`. Sum feet; scale `gmr_swing_clearance=-1`.
   A 2 cm deficit beyond tolerance costs 1 at full gate. No extra reward for
   over-lifting; existing joint/foot tracking still constrains overshoot.
5. Unintended contact cost `gate * (Fz > 5 N)`, summed over feet; scale
   `gmr_swing_contact=-1`. Keep the old instantaneous force threshold, not a
   new threshold/history heuristic. Existing contact/slip rewards stay active.
6. The standard accumulator multiplies reward scales by control dt=0.01 s.

Positive tracking rewards and old negative costs are unchanged. Total returns
under the new objective must not be compared directly to old total returns.

## Diagnostics and validation boundaries

Capture before reset: gated clearance deficit, unintended contact fraction,
contact sole-center horizontal speed, and both raw new costs. Keep the existing
action/torque variation and trunk tracking diagnostics. These sample the last
PD substep at 100 Hz, NOT every 1 kHz physics step. Sole-center velocity includes
foot rolling and is not by itself literal contact-point sliding distance.

The motivating deterministic model7000 rollout has 97.7% of its existing squared
contact-foot-speed cost during reference swing, but this is one motion/seed,
not a population causal result. At t=2.04 s the left reference foot mesh is
about 5 cm above ground while the actual foot carries about 515 N vertically.
GMR contact labels remain kinematic estimates; boundary/clearance gates limit
overconstraint but do not establish that every label is dynamically correct.

## Execution plan

- Entry: `humanoid/scripts/train_gmr_swing.py`.
- Hparams: `humanoid/envs/x1/x1_gmr_swing_config.py`.
- Registered Gradmotion smoke: 512 environments, 20 additional updates,
  seed 5, source original model7000, final model7020.
- After successful smoke: 4096 environments, 1000 additional updates,
  seed 5, source original model7000 (NOT smoke output), final model8000.
- Only 4090D 24GB `ESKU000001`; freshly confirmed `BJX00000001/V000124`
  lineage. Read identity/retirement registry before account use. No credential
  or signed URL is committed. Re-read resource/task configuration before run.
- Smoke gate: exact commit/source/reference, 33 loaded tensors, fresh optimizers,
  unchanged runtime dynamics, both new rewards bound/nonzero, finite growing PPO
  logs, normal completion and model upload. Local tests are not Gym smoke.

Final acceptance must jointly check full-clip walking distance/root tracking,
unintended swing contact/clearance/landing speed, stable-stance drift and
action/torque variation. Lower foot speed achieved by standing still is failure.
No improvement claim is made before deterministic inference of the new model.

## Local verification, 2026-09-11

- 86 tests passed, including all previous tests. Random-orientation full-mesh
  minima agree with hull minima within 0.2 micrometers.
- Real source model7000 loaded all 33 tensors; internal iter 6999, next index
  7000, mean exploration std 0.0549730547, both optimizer states empty.
  Real actor/critic forward pass is finite with unchanged input/output shapes.
- Re-evaluating the recorded model7000 trajectory: the new minimum-height
  calculation differs from independent MuJoCo geometry by at most 0.754 um.
  Gated unintended contact fraction is 0.167931; new full-clip negative
  contributions would be -2.058927 (height) and -0.278785 (contact), including dt.
  These are OFFLINE costs of the old policy, not new training results.
- The 1.45 s right-foot and 2.04/2.19 s left-foot events receive full gate.
  The 3.50 s event is intentionally almost excluded because it is near an
  off-contact boundary. This experiment does not supervise every contact
  transient; touchdown/late-toe-off handling remains a separate future stage.

## Registered smoke verification

- Code `a92dfe6346bcd1e853a1c923fec33667e9b601bb` published to the isolated branch.
- `TASK_20260911_168`, owner 4190, status 5, Beijing 16:46:32--16:47:43.
  Exactly indices 7000..7019 ran. Source task/hash, reference hash, actual
  code commit, all 33 tensors and fixed 1e-5 learning rate match the plan.
- Runtime PD, armature, effective torque limits, passive damping/friction,
  action scale and control dt exactly match task116's saved runtime manifest.
  All old reward scales exactly match; only the two new -1 terms are added.
- Geometry SHA256 `db1ad11fb11cea2704494ce14a7e1c94899fe83822cbc9195b6b83237ab9dd47`.
  Active reference samples before height gating: 35 left / 38 right.
- Both new episode reward terms are nonzero. No NaN, Inf, Traceback,
  RuntimeError or CUDA OOM found. Final smoke reward 45.65 / episode length
  412.39 are stochastic training aggregates, not a deterministic improvement.
- `model_7020.pt` upload is confirmed by SDK and model list. Local download
  failed due DNS resolution / HTTPS routing; its contents are NOT locally
  verified. Cloud smoke completion and upload gates are satisfied. Formal
  training still uses locally hash-verified ORIGINAL model7000, not model7020.
- Formal `TASK_20260911_171` run accepted at Beijing 16:51:50, instance
  start 16:52:06; 4096 environments, 1000 additional updates, same code/inputs.
  Actual PPO startup was verified through index 7055/8000 (56 new updates).
  Code/source/reference, 33 loaded tensors, fresh optimizers, 4096 environments,
  fixed 1e-5 learning rate and both new nonzero terms match the approved plan.
  Runtime dynamics and reward weights exactly equal the smoke configuration.
  No numeric/error markers found in the startup log. Last ten iteration times
  average 1.08 s; approximately 17 minutes of PPO remained at that snapshot,
  excluding completion/upload overhead. This is not a convergence estimate.
- Logs and source/run provenance are in ignored `outputs/gradmotion-swing/`.
  Creation payloads were removed after successful task creation.
