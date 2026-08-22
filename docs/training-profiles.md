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
| `retarget_walk_vx045_conservative_no_dr` | `x1_dh_stand_retarget_walk_vx045` | 原始抬脚高度与接触联合掩码、随机初始化 | 3000 |
| `retarget_walk_vx045_geometry_no_dr` | `x1_dh_stand_retarget_walk_vx045_geometry` | 保守抬脚加站姿几何约束、随机初始化 | 3000 |
| `retarget_walk_native_geometry_no_dr` | `x1_dh_stand_retarget_walk_native_geometry` | 接触一致原始周期/0.124m/s、保守抬脚加站姿几何约束 | 3000 |
| `retarget_walk_native_geometry_resume` | `x1_dh_stand_retarget_walk_native_geometry` | 显式原生速度站姿几何 checkpoint | 2000 |
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
  --training_profile=retarget_walk_native_geometry_resume `
  --load_run=<native-geometry-run> `
  --checkpoint=1000 `
  --max_iterations=2000 `
  --run_name=walk_csv_native_geometry_continue `
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
  `resources/motions/x1/walk_12dof_contact_consistent.csv` 的 12 个受控腿部关节
  作为相位条件参考轨迹；该资产只保留重建后的浮动基座、12 个受控关节和
  左右接触置信度，不包含腰、手臂、颈部或头部列。
- 重定向参考按关节名称映射，不依赖 CSV 列顺序；全身关节自相关得到 146 帧
  整周期，选用 0.600–5.467 秒周期并线性插值，超出 X1 URDF 的参考角会裁剪到
  实际关节限位。未加速指令使用接触一致速度 0.124m/s，不再采用原始根轨迹的
  0.255m/s。
- 参考接触表明每脚硬接触占空比约 54.1%，完整周期含 12/146 帧双支撑。相位
  采样偏移使用 `26/146`，使环境相位掩码与参考接触达到约 99.3% 一致；旧的
  `0.5` 偏移只有约 36.3% 一致，会让相位观测、基础高度约束与参考动作错位。
- `retarget_walk_no_dr` 采用 reference-first 奖励：关节参考权重为 6，数据驱动
  的软接触时序权重为 2，速度跟踪权重为 4，`tracking_sigma=30`，低速项权重
  为 2。旧 `feet_contact_number` 继续关闭，由 `ref_feet_contact` 替代；旧
  `feet_air_time` 会在计划支撑开始时即触发而不要求真实落脚，旧
  `feet_clearance` 只要求 3–6cm 且与 13.6–16.4cm 的参考峰值冲突，因此二者
  继续关闭。防滑、碰撞、关节限位、动作平滑和基础姿态稳定约束仍保留。
- `retarget_walk_vx045_*` 保留同一 12 关节参考；每周期两步、平均单步
  0.3013507m，因此把周期从 4.866667s 压缩到 1.339336s（约 3.63 倍速），
  并同步把前进指令设为 0.45m/s，对应约 89.6 步/min。该快速配置从正确的
  mesh 脚底点和重建根轨迹得到离地曲线：原始左右峰值约 16.4/13.6cm，训练
  的保守版本不再放大高度，`ref_feet_clearance` 权重为 2，并要求参考接触和
  参考高度同时判定为摆动才施加抬脚目标；旧接触数和腾空时间项仍关闭。
- `retarget_walk_vx045_geometry_no_dr` 在保守抬脚版本上增加逐相位站姿几何：
  足和膝横向距离来自同一 URDF/CSV 的前向运动学，足距参考下限 12cm、膝距
  参考下限 14cm；对应奖励权重为 1.0/0.5。足部朝向使用 ankle-roll 局部
  `+Z` 投影到机身平面，去除左右脚公共偏航后把摆动参考裁剪到 ±10°，接触脚
  目标为 0°，奖励权重 0.5；另加权重 0.5 的 hip-yaw 专项参考。旧的 XY 总
  `feet_distance`/`knee_distance` 在该 task 中关闭，避免用前后错位掩盖横向
  交叉。左右 hip-roll 参考均距物理限位保留 0.02rad 余量。
- `retarget_walk_native_geometry_no_dr` 保留同一套抬脚、接触、足/膝横向
  间距和足朝向奖励，但将周期恢复为接触一致参考的 `4.8667s`，并把
  前进指令固定为 `0.124m/s`。该速度来自周期重建位移 `0.6027m`，而非
  会造成支撑脚滑动的原始世界根轨迹速度。
- 接触修正后的两条任务分别写入新的 experiment
  `x1_dh_stand_retarget_walk_periodic_contact` 与
  `x1_dh_stand_retarget_walk_periodic_contact_vx045`。旧任务无论使用原始根速度、
  旧快速周期或 `phase_offset=0.5`，其 checkpoint 都不作为新配置的等价比较
  基线；若显式恢复，只能视为重新适应初始化，并必须另做 smoke 验证。
- `stage1_from_no_dr` 固定 nominal armature，只启用窄范围 Stage-1 DR。
- 两种完整 DR 运行都使用 `x1_dh_stand_dr_full`；区别仅为初始化来源和日志实验名。
- 普通 Isaac Gym 推理仍使用 `--armature_mode=nominal`，并由推理脚本关闭 DR
  和观测噪声；训练 profile 不替代推理安全约束。
- 恢复 profile 必须显式指定 `--load_run` 与非负 `--checkpoint`，避免错误谱系。
