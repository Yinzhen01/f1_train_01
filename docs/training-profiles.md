# X1 训练运行配置

本项目用一套代码覆盖 nominal 无 DR、重定向步态模仿、Stage-1 DR、课程式完整
DR 和直接完整 DR 基线。动力学或训练目标差异由已注册 task 表达；从零训练或
恢复 checkpoint 由运行配置表达，不再为相同动力学复制实验分支。

配置源为 `configs/training/x1_profiles.json`。可以独立检查当前预设：

```powershell
python -m humanoid.training_profiles
```

## 预设矩阵

| profile | task | 初始化 | 默认更新数 |
| --- | --- | --- | ---: |
| `no_dr_scratch` | `x1_dh_stand_no_dr_mixed` | 随机初始化 | 2000 |
| `stage1_from_no_dr` | `x1_dh_stand_dr_stage1` | 显式 no-DR checkpoint | 2000 |
| `full_dr_from_stage1` | `x1_dh_stand_dr_full` | 显式 Stage-1 checkpoint | 3000 |
| `full_dr_scratch` | `x1_dh_stand_dr_full` | 随机初始化 | 6000 |
| `retarget_walk_no_dr` | `x1_dh_stand_retarget_walk` | 12关节重定向参考、随机初始化 | 3000 |
| `retarget_walk_resume` | `x1_dh_stand_retarget_walk` | 显式重定向训练 checkpoint | 2500 |
| `retarget_walk_vx045_no_dr` | `x1_dh_stand_retarget_walk_vx045` | 0.45m/s 时间缩放参考、随机初始化 | 3000 |
| `retarget_walk_vx045_resume` | `x1_dh_stand_retarget_walk_vx045` | 显式重定向 checkpoint | 1500 |

默认更新数用于初始评估，可通过 `--max_iterations` 覆盖；不要求任何阶段机械跑满
固定次数。`seed` 和 `num_envs` 的默认值分别为 5 和 4096，也允许显式覆盖。

## 启动示例

直接完整 DR 基线与课程式完整 DR 共用同一个环境配置：

```powershell
python humanoid/scripts/train.py `
  --training_profile=full_dr_scratch `
  --run_name=direct_full_dr `
  --headless
```

从 Stage-1 checkpoint 进入完整 DR 时必须显式给出来源，禁止隐式加载“最后一个”
checkpoint：

```powershell
python humanoid/scripts/train.py `
  --training_profile=full_dr_from_stage1 `
  --load_run=<stage1-run> `
  --checkpoint=6700 `
  --run_name=stage1_to_full_dr `
  --headless
```

无 DR 与 Stage-1 同理：

```powershell
python humanoid/scripts/train.py --training_profile=no_dr_scratch --headless

python humanoid/scripts/train.py `
  --training_profile=retarget_walk_no_dr `
  --run_name=walk_csv_imitation `
  --headless

python humanoid/scripts/train.py `
  --training_profile=retarget_walk_resume `
  --load_run=<retarget-run> `
  --checkpoint=500 `
  --run_name=walk_csv_imitation_continue `
  --headless

python humanoid/scripts/train.py `
  --training_profile=retarget_walk_vx045_resume `
  --load_run=<retarget-run> `
  --checkpoint=2000 `
  --run_name=walk_csv_vx045_adaptation `
  --headless

python humanoid/scripts/train.py `
  --training_profile=stage1_from_no_dr `
  --load_run=<no-dr-run> `
  --checkpoint=4700 `
  --headless
```

## 配置边界

- `no_dr_scratch` 固定使用共享 nominal armature，关闭全部 DR 和观测噪声。
- `retarget_walk_no_dr` 同样固定 nominal armature 并关闭 DR/噪声，但把
  `resources/motions/x1/walk_12dof.csv` 的 12 个受控腿部关节作为相位条件参考
  轨迹；该仓库资产已从原始 `walk.csv` 按名称提取，不包含浮动基座、腰、
  手臂、颈部或头部列。
- 重定向参考按关节名称映射，不依赖 CSV 列顺序；选用 0.600–5.533 秒近闭合
  周期并线性插值，超出 X1 URDF 的参考角会裁剪到实际关节限位。
- `retarget_walk_no_dr` 的首阶段采用 reference-first 奖励：关节参考权重为 6，
  速度跟踪权重为 4，`tracking_sigma=30`，低速项权重为 2；默认站姿奖励及
  人工生成的接触数、腾空时间、抬脚高度和 `track_vel_hard` 暂时关闭。防滑、
  碰撞、关节限位、动作平滑和基础姿态稳定约束仍保留。接触类奖励只能在从
  重定向动作得到足端接触/离地标签后重新加入，避免固定 50/50 相位与数据冲突。
- `retarget_walk_vx045_*` 保留同一 12 关节参考，将 1.259765m 路径的播放时间
  从 4.933333s 压缩到 2.799477s（1.762234 倍速），并同步把前进指令设为
  0.45m/s；时间轴与速度指令必须配套修改。该快速配置从同一参考动作和 X1
  URDF 前向运动学得到足底离地曲线：原始左右峰值约 15.5/11.8cm，训练目标
  再提高 5%+5mm（约 16.8/12.9cm）。权重为 3 的 `ref_feet_clearance`
  同时跟踪该曲线并惩罚摆动脚低于目标；接触数和腾空时间项仍关闭。
- `stage1_from_no_dr` 固定 nominal armature，只启用窄范围 Stage-1 DR。
- 两种完整 DR 运行都使用 `x1_dh_stand_dr_full`；区别仅为初始化来源和日志实验名。
- 普通 Isaac Gym 推理仍使用 `--armature_mode=nominal`，并由推理脚本关闭 DR
  和观测噪声；训练 profile 不替代推理安全约束。
- 恢复 profile 必须显式指定 `--load_run` 与非负 `--checkpoint`，避免错误谱系。
