# 项目状态

本文档用于长期项目的低成本上下文恢复，只记录会影响后续工作的阶段、决策、阻塞和风险。Gradmotion 任务实时状态应使用 gm-cli 查询，不在这里伪装成实时信息。

## 当前阶段

X1 可训练性已验证，正在推进分阶段域随机化、完整域随机化对照，以及 Isaac Gym/MuJoCo 动力学一致性治理。

## 当前目标

- 保持混合速度指令与新版奖励下的 X1 策略可训练。
- 比较无 DR、Stage-1 DR、直接完整 DR、Stage-1→完整 DR 四条路线的收敛速度和行为质量。
- 以 Isaac Gym 训练时的有效参数为标准，统一训练、Isaac Gym 推理和 MuJoCo 推理配置。
- 让每次训练可追溯到 Git 分支、commit、Gradmotion task、起始 checkpoint 和 DR 范围。

## 已完成

- 无 DR 混合指令基线：`TASK_20260820_191`，历史记录的最终 checkpoint 为 `model_4999`。
- Stage-1 DR：`TASK_20260820_199`，历史记录的最终 checkpoint 为 `model_6000`。
- 建立直接完整 DR 分支 `experiment/full-domain-rand-baseline`，对应任务 `TASK_20260820_201`。
- 建立 Stage-1→完整 DR 分支 `experiment/stage1-to-full-domain-rand`，对应任务 `TASK_20260820_208`。
- 在 `refactor/shared-armature-config` / `90d3b5f` 中建立 X1 armature 外部共享配置，并接入 Isaac Gym 训练/推理与 MuJoCo 推理。
- 为共享 armature 增加 URDF/MJCF 关节一致性测试和 MuJoCo 实际模型加载检查。
- Isaac Gym nominal armature 回放：`TASK_20260821_005`，使用分支
  `refactor/shared-armature-config` / `eb8ea24` 和 Stage-1 `model_6000`。
  运行时回读的 12 个 armature 与共享配置完全一致；完成 2000 steps、
  1000 帧、20 秒 1920×1080 视频，平均前进速度 `0.196 m/s`（目标
  `0.4 m/s`），平均高度 `0.610 m`。机器人未摔倒但速度不足且存在交叉步。

## 正在进行

- 无 DR nominal-armature 基线：`TASK_20260821_006`，分支
  `experiment/no-dr-nominal-armature` / `0c7bfc0`，随机初始化、seed 5、
  4096 environments、目标 5000 iterations。云端启动日志已确认
  `[armature] mode=nominal` 并进入 PPO 训练。
- 在 Isaac Gym 可用环境中验证完整 DR 的 armature 范围采样；nominal 模式的运行时回读已经通过。
- 获取 `TASK_20260820_201` 与 `TASK_20260820_208` 的最新终态和指标，完成直接训练与课程训练对比。
- 评估共享 armature 变更应合入哪个训练分支，并区分旧 checkpoint 与新动力学下的推理结果。

## 下一步

1. 在 Gradmotion 基于 `refactor/shared-armature-config` 做最小 smoke：分别检查 no-DR nominal 和 full-DR sampling 日志/运行时回读。
2. 把 smoke 的 task ID、commit、配置与 checkpoint 写入本文件或专项实验记录。
3. 对同一 checkpoint 分别做 Isaac Gym 与 MuJoCo 推理，核对关节顺序、PD、动作缩放、控制周期、armature、friction、damping、延迟和观测构造。
4. 确认行为和指标后，再决定合并与推送；本次管理框架同步本身不改变正在运行的云端任务。

## 关键决策

- `2026-08-20`：先用无 DR 验证可训练性，再逐步增加域随机化，并保留直接完整 DR 作为 A/B 基线。
- `2026-08-20`：训练指令保持旧版的前后、横移、转向和站立随机指令，其他调试与奖励调整采用新版方案。
- `2026-08-21`：Isaac Gym 训练时的有效动力学参数作为跨模拟器整理基准。
- `2026-08-21`：armature 由外部 JSON 按关节名称统一读取；无 DR 使用 nominal，DR 使用 train range。
- `2026-08-21`：Isaac Gym 推理通过 `training`、`nominal`、`zero` 三种显式模式区分任务配置、新共享默认值和历史零 armature checkpoint。
- `2026-08-21`：采用 `AGENTS.md`、专项文档、自动化测试和项目 Skill 分层管理工程规则与状态。

## 风险与注意事项

- 旧的 no-DR/Stage-1 checkpoint 在 armature 关闭时使用的是零 armature；用新 nominal 参数推理会改变动力学，不能直接视为训练环境复现。
- Stage-1 `model_6000` 使用新 nominal armature 后平均速度由历史零 armature
  回放的约 `0.299 m/s` 降至 `0.196 m/s`；这是单次同策略对照，说明参数敏感性，
  尚不能单独量化每个关节 armature 的贡献。
- Windows 本地当前可做 MuJoCo 检查，但没有 Isaac Gym；Isaac Gym 运行正确性必须由云端 smoke 验证。
- 部分动力学参数位于 URDF/MJCF，部分位于配置或代码；只同步模型文件不能保证两个模拟器一致。
- 训练 reward、episode length 或视频改善不等于 Sim2Real/真机正确；需要分别保留参数一致性、跨仿真和真机证据。
- Gradmotion task 的实时状态、资源余额和 ETA 会变化，使用前必须重新查询。

## 更新规则

- 阶段变化、关键实验终态、重要决策、长期阻塞或风险变化时更新。
- 日常每十分钟监控数据和临时调试输出不写入本文档。
- 记录任务时至少包含 task ID、Git commit、起始 checkpoint、目标迭代和 DR 方案。
