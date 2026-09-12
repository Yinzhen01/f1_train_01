# KIT317 acceleration-weight sweep results

## Verified training and inference

All three 12-DOF groups completed 1000 additional updates from the original
model8000, then three deterministic 4.6-second inference repeats each. All clips
completed without physical termination. Repeats within a group were identical;
this is one training seed per group, not a robustness or statistical-significance
test. Source configuration and asset checks passed.

Training code: `eccd14eed21ffe980c90abd7dc740d043e01fcaf`.
Inference code: `5cf4359a44eae8389ba1c62f9613f72213a79e62`.
Inference-only adaptation added exact model identities, safe model loading and
video labels. No training reward or dynamics change. 100 local tests passed.

| Group | Training task | Inference task | Final model9000 SHA256 |
|---|---|---|---|
| 1x | TASK_20260912_033 | TASK_20260912_054 | d258477d65787c7b89740eb4df4261b04e2fa1b204fdb9e64400a71ab595bdae |
| 3x | TASK_20260912_034 | TASK_20260912_055 | 576d5cc511def5318a6c41d4b8e1aadd6dda56b2986f2f2f58ad806a32a9cbb7 |
| 10x | TASK_20260912_035 | TASK_20260912_056 | 22fb7520a219a265c39b627e8c453136adc57bed019ab4dccff4614f0a58bb37 |

All final checkpoints have stored iter8999, 33 finite strictly loadable tensors,
and finite actor/critic forward passes. Actual inference parameters, DOF order,
PD, torque limits, armature/passive properties, action scale, control timestep,
reference and URDF hashes match the frozen model8000 inference.

## Results, not a winner declaration

| Metric | Frozen model8000 | 1x continuation | 3x | 10x |
|---|---:|---:|---:|---:|
| Joint acceleration RMS, rad/s² | 66.181 | 54.573 | 61.485 | 60.156 |
| Ankle acceleration RMS, rad/s² | 93.452 | 68.810 | 81.646 | 83.188 |
| Absolute joint acceleration peak, rad/s² | 984.213 | 951.708 | 1042.734 | 1020.666 |
| Action-difference RMS | 0.2059 | 0.1833 | 0.1953 | 0.1916 |
| Torque-difference RMS, N·m | 3.915 | 3.511 | 3.861 | 3.679 |
| Mean contact sole speed, mm/s | 23.396 | 25.665 | 25.072 | 33.575 |
| Sampled vertical contact-force peak, N | 9431.680 | 6528.997 | 8403.838 | 4702.321 |
| Horizontal displacement, m | 1.668 | 1.579 | 1.645 | 1.639 |
| Joint tracking RMSE, rad | 0.0972 | 0.0935 | 0.0973 | 0.0919 |

The target displacement is approximately 1.741 m. Acceleration is calculated
from 100 Hz velocity samples after the initial 0.1 seconds, without reset pairs.
The mean training acceleration diagnostic over updates8900–8999 was
70.26 / 73.43 / 70.65 rad/s² for 1x / 3x / 10x, respectively. This is a different
stochastic multi-environment statistic from deterministic rollout RMS.

- There is no monotonic improvement from increasing the weight. Relative to 1x,
  3x and 10x joint acceleration RMS are 12.7% and 10.2% higher.
- 1x reduces acceleration RMS by 17.5% versus frozen model8000 but also reduces
  walking displacement by 5.4%. It is not established as the best overall policy.
- 10x lowers the sampled contact-force peak by 28.0% versus 1x, but its mean
  contact sole speed is 30.8% higher (43.5% above model8000).
- Do not replace a deployment policy or launch more training automatically.
  A possible next study is phase-aware, thresholded ankle acceleration costs,
  with explicit walking/foot-slip/tracking non-regression checks and independent
  training seeds. This was not implemented in this experiment.

## Artifacts and boundary

Media delivery validated on 2026-09-12 at 10:12 CST: all three dual-view videos
are H264, 1600x720, 50fps, 231 frames, 4.62 seconds. The synchronized four-panel
comparison adds frozen model8000 and is H264, 1280x1020, with the same frame count
and timing. Full decode and first/middle/last comparison-frame inspection passed.
GIF previews are reduced to 12fps and are not suitable for judging high-frequency
jitter; use the full-rate MP4s. No trajectory smoothing or dynamics resimulation
was applied. Monitoring is paused after this delivery; no new training is started.

Local artifacts live under `outputs/policy_videos/gmr_accel_comparison/` and the
three `gmr_accel_{1x,3x,10x}_model9000/` folders. Full data, model hashes,
configuration manifests, task records, sanitized logs, training curves and
validation scripts are retained in ignored `outputs/gradmotion-accel/`.
The Chinese report is `outputs/policy_videos/gmr_accel_comparison/实验结果.md`.

MuJoCo only renders recorded Isaac Gym states. Contact sole-center speed includes
rolling and is not pure slip distance. Geometric mesh penetration is not solver
contact depth. Sampled contact forces are not real hardware forces; 100 Hz does
not resolve all 1 kHz impacts. No Sim2Sim, robustness or hardware-safety claim.
