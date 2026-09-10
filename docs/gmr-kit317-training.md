# GMR KIT317 foot-flat, 12-DOF training

## Scope and provenance

Independent copy of `F:\robot_f1\x1-training`, starting at
`38e4efb62da14fa7fc0612939e5935bf09c16432`. Original checkout and existing worktrees
are not modified. Current working-copy files were copied and hash-checked;
unrelated README/skill edits remain uncommitted. The current URDF knee-inertia
edits are intentionally included because they define the requested training
baseline. No old checkpoint is resumed.

Source: `F:\robot_f1\GMR\outputs\kit317_walking_f1_v1_4\foot_flat\KIT_317_walking_medium08_f1_v1_4_foot_flat.npz`.
Source SHA256: `8458117d35eaf773c2986c30d104446e56197cb39dd4980a0809b818bbec9ddd`.
The source remains unchanged. 139 samples at 30 Hz span timestamps 0..4.6 s.
Only the named left/right hip pitch/roll/yaw, knee pitch, ankle pitch/roll joints
are controlled. The current 12-DOF URDF retains its fixed upper-body model;
this is not a dynamically equivalent 29-DOF whole-body policy.

## Geometry and limits

The GMR V1.4 right leg has slightly different calibrated geometry from the
training URDF. Direct angle copying causes up to 1.98 mm ankle-position error
and 0.00448 rad ankle-orientation error. Also, source left ankle pitch reaches
-0.4837 rad versus the training lower bound -0.4100; left hip roll reaches
0.2214 versus the upper bound 0.2000.

`humanoid/scripts/prepare_gmr_reference.py --fit-limits` performs bounded
root-plus-leg IK using the actual training URDF to follow source ankle poses.
It does not widen limits, clip the final motion, change playback speed or
wrap the clip. Both source models and derived data have SHA256 provenance in
`resources/motions/gmr_kit317_foot_flat_12dof.json`.

Validated sample-frame results: ankle position deviation <=0.332 mm,
orientation deviation <=0.001322 rad; standing sole tilt <=0.00334 degrees;
foot collision-mesh minimum z >=2 mm; all 12 joint positions/finite-difference
velocities within training hard limits. Contact labels use foot height <25 mm
and speed <0.35 m/s: these are kinematic labels, not measured contact forces.
URDF forward kinematics independently agrees with MuJoCo's URDF import.

## Training definition

Task: `x1_gmr_clip`, config: `humanoid/envs/x1/x1_gmr_clip_config.py`.
No dynamics randomization, no observation noise, plane friction 0.6, shared
nominal armature. Existing PD, timestep and action scaling are unchanged.
Policy outputs 12 actions, interpreted as joint-target offsets from the existing
default pose (not residuals added to the motion reference).

The complete standing/walking/stopping clip is finite, not periodic. Reference
sampling clamps at the end and the environment terminates at 4.6 s; completion
does not bootstrap into a new clip. Reference-state initialization chooses the
clip start with probability 0.25 and otherwise a random time leaving >=0.3 s.
Root and joint positions AND velocities are initialized from the same time.
Commands are the reference root velocities, not independently sampled speeds.

Actor width remains 47 (66 frames); critic remains 73 (3 frames). Phase channels
change to `sin(pi*t/4.6), cos(pi*t/4.6)`, so start/end are distinguishable.
Old checkpoint semantics are incompatible despite identical tensor widths.
Rewards track joints, velocities, feet, contacts, root position/orientation;
they include actual-contact slip and the existing smoothness/limit penalties.
Procedural gait, default-pose and unconditional flat-foot rewards are excluded.

```bash
python humanoid/scripts/train.py --task=x1_gmr_clip --headless --num_envs=16 --max_iterations=5 --seed=5 --run_name=gmr_kit317_smoke
python humanoid/scripts/train.py --task=x1_gmr_clip --headless --num_envs=4096 --max_iterations=5000 --seed=5 --run_name=gmr_kit317_12dof
```

Formal execution must use a registered Gradmotion task, 1x4090D 24 GB
(`ESKU000001`), after smoke training succeeds. Existing tasks are not stopped.

Cloud smoke `TASK_20260910_176` completed successfully at commit `a658295`:
16 environments, 5 updates, GPU PhysX, the expected 139-frame reference and
12-DOF URDF hash; `model_0.pt` and `model_5.pt` uploaded. The follow-up diagnostic
fix overrides the inherited slip log (which incorrectly reads angular velocity)
with actual-contact horizontal sole velocity. Reward math is unchanged; CPU
tensor tests cover zero slip, translation, swing masking and perfect tracking.
All 30 local tests pass. Dense 100 Hz FK checks 461 samples and finds minimum
collision-mesh height 0.509 mm; peak interpolated joint speed is 7.891 rad/s.
Piecewise-linear samples retain peak finite-difference acceleration around
498 rad/s^2, so dynamic feasibility and smooth policy behavior remain open.

## Validation boundary

CPU tests and reference FK do not prove dynamic stability, friction feasibility,
Isaac Gym contact equivalence, policy convergence, or hardware safety. Sampled
foot-mesh clearance is not a continuous-time collision guarantee. Upper-body
motion is not trained. MuJoCo policy deployment requires a task-specific finite
clip command/phase scheduler; the existing periodic `sim2sim.py` must not be used
unchanged, and its legacy MJCF has not been synchronized to the copied knee
inertia edits. Training status is recorded separately in Gradmotion task records.
