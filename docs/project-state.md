# 项目状态

本文档用于长期项目的低成本上下文恢复，只记录会影响后续工作的阶段、决策、阻塞和风险。Gradmotion 任务实时状态应使用 gm-cli 查询，不在这里伪装成实时信息。

## 当前阶段

X1 nominal-armature 可训练性正在验证，旧零 armature 实验已退出管理矩阵；后续从干净 nominal 基线重新建立 Stage-1→完整 DR 路线。

## 当前目标

- 保持混合速度指令与新版奖励下的 X1 策略可训练。
- 比较 nominal 无 DR、直接完整 DR、nominal→Stage-1→完整 DR 三条有效路线的收敛速度和行为质量。
- 以 Isaac Gym 训练时的有效参数为标准，统一训练、Isaac Gym 推理和 MuJoCo 推理配置。
- 让每次训练可追溯到 Git 分支、commit、Gradmotion task、起始 checkpoint 和 DR 范围。

## 已完成

- 直接完整 DR：`TASK_20260820_201`，从随机初始化训练，并启用包含 armature 在内的完整随机化；该路线保留。
- 无 DR nominal-armature 基线：`TASK_20260821_006`，分支
  `experiment/no-dr-nominal-armature` / `0c7bfc0`，随机初始化、seed 5、
  4096 environments。用户确认该阶段无需机械跑满 5000 iterations，任务于
  `2026-08-21 10:09:44` 计划内早停；状态 6 不代表训练失败，最高完整
  checkpoint 为 `model_4700`，作为后续有效恢复源。
- 直接完整 DR 的 deterministic nominal Isaac Gym 推理：`TASK_20260821_014`，
  使用 `TASK_20260820_201/model_6000`、分支
  `experiment/nominal-armature-pipeline` / `dee653b`。日志确认 nominal 动力学、
  plane friction `0.6`、DR/观测噪声关闭、12 关节 nominal armature、2000 steps/
  1000 帧；平均前进速度 `0.349 m/s`（目标 `0.4 m/s`），平均高度 `0.607 m`。
  视频已下载并验证为 1920×1080、50 fps、20 秒，路径为
  `outputs/isaacgym/TASK_20260821_014/play_output.mp4`。
- 无 DR 最终 checkpoint 的 deterministic nominal Isaac Gym 推理：
  `TASK_20260821_018`，源为 `TASK_20260821_006/model_4700`，推理 commit
  `2d60bc3`。日志确认 nominal 动力学、plane friction `0.6`、DR/观测噪声
  关闭、12 关节 nominal armature、2000 steps/1000 帧；平均前进速度
  `0.367 m/s`（目标 `0.4 m/s`），平均高度 `0.608 m`。视频已下载并验证为
  1920×1080、50 fps、20 秒，路径为
  `outputs/isaacgym/TASK_20260821_018/play_output.mp4`。
- 在 `refactor/shared-armature-config` / `90d3b5f` 中建立 X1 armature 外部共享配置，并接入 Isaac Gym 训练/推理与 MuJoCo 推理。
- 为共享 armature 增加 URDF/MJCF 关节一致性测试和 MuJoCo 实际模型加载检查。
- 无 DR nominal-armature 中间 checkpoint 回放：`TASK_20260821_009`，使用
  `TASK_20260821_006` 的 `model_1600`、`armature_mode=nominal`。Isaac Gym
  完成 2000 steps、1000 帧、20 秒 1920×1080 视频，平均前进速度
  `0.342 m/s`（目标 `0.4 m/s`），平均高度 `0.609 m`；抽帧检查显示机器人
  全程保持直立并交替单脚支撑。视频已下载到
  `outputs/isaacgym/TASK_20260821_009/no_dr_nominal_model1600_isaacgym.mp4`。

## 正在进行

- 新 Stage-1 DR：`TASK_20260821_028`，源为
  `TASK_20260821_006/model_4700`，分支 `experiment/nominal-armature-pipeline`，
  seed 5、4096 environments，初始追加 2000 PPO updates。保留 nominal
  armature，仅随机 friction `[0.45, 0.80]`、base mass `[-1, 1] kg`、COM
  各轴 `±0.015 m`、PD gains/torque `[0.95, 1.05]`。
- 标准 Isaac Gym 推理收敛为 nominal-only；随机 DR 鲁棒性测试使用独立评估流程，不与普通渲染混用。

## 已作废并退出实验矩阵

- `TASK_20260820_191`：无 DR 训练使用零 armature。
- `TASK_20260820_199`：Stage-1 DR 继承零 armature 动力学。
- `TASK_20260820_208`：课程路线从零 armature Stage-1 checkpoint 启动，后续不作为主要课程基线。
- 基于上述 checkpoint 的推理任务和视频不再引用、渲染或参与指标比较。
- 上述 ID 仅作为作废原因索引，不保留性能结论，也不作为任何新任务的恢复来源。

## 下一步

1. 监控 `TASK_20260821_028`；达到行为门槛即停止扩展并做 nominal 推理，
   否则每次仅追加 1000 PPO updates 后复评。
2. 从通过门槛的新 Stage-1 checkpoint 继续完整 DR；初始追加 3000 PPO
   updates，按评估结果自适应延长并做 nominal 推理。
3. 使用 `TASK_20260821_014` 作为直接完整 DR 路线的公平 nominal A/B 对照。

## 关键决策

- `2026-08-20`：先用无 DR 验证可训练性，再逐步增加域随机化，并保留直接完整 DR 作为 A/B 基线。
- `2026-08-20`：训练指令保持旧版的前后、横移、转向和站立随机指令，其他调试与奖励调整采用新版方案。
- `2026-08-21`：Isaac Gym 训练时的有效动力学参数作为跨模拟器整理基准。
- `2026-08-21`：armature 由外部 JSON 按关节名称统一读取；无 DR 使用 nominal，DR 使用 train range。
- `2026-08-21`：常规 Isaac Gym 推理只允许 `nominal`；零 armature 结果退出实验矩阵，随机鲁棒性由独立评估流程承担。
- `2026-08-21`：nominal 无 DR 以可训练性和稳定行为为阶段目标，不要求固定跑满 5000；`model_4700` 作为本轮有效终点。
- `2026-08-21`：采用 `AGENTS.md`、专项文档、自动化测试和项目 Skill 分层管理工程规则与状态。

## 风险与注意事项

- 旧零 armature 任务不得再作为恢复训练来源或比较基线，避免污染后续实验谱系。
- 当前 hip/knee nominal `0.02505` 是训练范围中点而非系统辨识真值；新路线物理上更合理，但仍需后续辨识校准。
- Windows 本地当前可做 MuJoCo 检查，但没有 Isaac Gym；Isaac Gym 运行正确性必须由云端 smoke 验证。
- 部分动力学参数位于 URDF/MJCF，部分位于配置或代码；只同步模型文件不能保证两个模拟器一致。
- 训练 reward、episode length 或视频改善不等于 Sim2Real/真机正确；需要分别保留参数一致性、跨仿真和真机证据。
- Gradmotion task 的实时状态、资源余额和 ETA 会变化，使用前必须重新查询。

## 更新规则

- 阶段变化、关键实验终态、重要决策、长期阻塞或风险变化时更新。
- 日常每十分钟监控数据和临时调试输出不写入本文档。
- 记录任务时至少包含 task ID、Git commit、起始 checkpoint、目标迭代和 DR 方案。
