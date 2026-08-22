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

- 重定向步态周期与奖励重新对齐：`walk.csv` 的 12 关节自相关在 146 帧
  （4.866667s）处达到完整周期，73 帧为左右半周期；每脚硬接触占空比约
  `54.1%`，每周期 `12/146` 帧为双支撑。原 `phase_offset=0.5` 与参考接触仅
  `36.3%` 一致，改为 `26/146=0.178082` 后达到约 `99.3%`，使相位观测、
  基础高度约束和参考动作统一。新增 `ref_feet_contact`（权重 2）直接跟踪 CSV
  的软接触置信度，并记录接触一致率、提前落脚和缺失支撑；旧
  `feet_contact_number`、`feet_air_time` 和 3–6cm `feet_clearance` 继续关闭，
  参考足底高度与防滑奖励保留。实现提交为 `b8fb0ea`，分支
  `experiment/retarget-walk-imitation`。

- 上述周期奖励的 Isaac Gym smoke：`TASK_20260822_082`，4090D、随机初始化、
  seed 5、4096 environments、50 PPO updates，实际 commit
  `b8fb0ea8aed7874490be7cd3f7766322d0ffe0c0`。运行时确认 nominal armature、
  DR/噪声关闭、147 帧参考、4.866667s 源周期及 `phase_offset=0.178`。任务自然
  完成 `4,915,200` timesteps，最高 checkpoint `model_50`；最终 reward `2.01`、
  episode length `124.42`、value/surrogate/state-estimator loss
  `0.0066/0.0034/0.0328`。接触一致率由 iteration 0 的 `0.4787` 增至最终
  `0.6018`，提前落脚由 `0.2563` 降至 `0.1887`，缺失支撑由 `0.7367` 降至
  `0.5806`；未发现 NaN/Inf、Traceback、CUDA OOM 或 RuntimeError。该结果证明
  新奖励和诊断链路可运行且早期方向正确，不代表 50 次更新已经得到稳定步态。

- 重定向根轨迹接触一致性复核：旧分析错误地把 ankle-roll 局部 `z` 当作竖直
  方向；实际 mesh/URDF 显示左右脚底分别位于局部 `-y/+y`，正确脚底中心约为
  `[0,-0.0408,0.005]` 与 `[0,+0.0408,0.005]`。新预处理按 146 帧全身关节周期、
  半周期左右换脚和 5 帧软双支撑窗口，全局最小二乘求解基座平移。415 帧数据
  的平均前向步长为 `0.30135m`，接触一致速度为 `0.12247m/s`；原始根轨迹为
  `0.25470m/s`。支撑脚水平滑移 P95 从 `0.31233m/s` 降到 `0.00547m/s`，RMS
  从 `0.13921m/s` 降到 `0.04940m/s`。派生资产和完整指标位于
  `resources/motions/x1/walk_12dof_contact_consistent.csv` 与
  `walk_contact_diagnostics.json`；本地 33 项单元测试通过。该结果是运动学接触
  一致性证据，不替代 Isaac Gym 动力学训练或真机验证。后续训练使用新的
  `x1_dh_stand_retarget_walk_contact*` experiment；此前按 0.255m/s 原始根速度或
  2.799477s 快速周期训练的 checkpoint 不再作为等价基线。

- 接触一致参考的云端 Isaac Gym smoke：`TASK_20260822_079`，分支
  `experiment/retarget-walk-imitation` / `4199f1b`，从随机初始化运行
  `retarget_walk_vx045_no_dr`，seed 5、4096 environments、50 PPO updates；
  nominal armature 固定，DR 与观测噪声关闭。启动日志确认实际加载
  `walk_12dof_contact_consistent.csv` 的 12 个关节、147 帧和 `4.866667s`
  周期。任务于 `2026-08-22 14:38:46` 完成，共 `4,915,200` timesteps，
  最高 checkpoint 为 `model_50`；最终 reward `2.39`、episode length
  `124.76`，value/surrogate/state-estimator loss 为
  `0.0047/-0.0009/0.0231`，约 `102,564 steps/s`。未发现 NaN/Inf、
  Traceback、CUDA OOM 或 RuntimeError。该 smoke 证明新资产、配置、Isaac
  Gym 环境与 PPO 链路可以运行；短训练仍有明显速度过冲和足滑，不代表已学会
  稳定步态，后续应进行长训练和 deterministic nominal 推理后再评价动作质量。

- 重定向步态 reference-first smoke：`TASK_20260821_163`，分支
  `experiment/retarget-walk-imitation` / `d0e17ad`，从随机初始化训练，seed 5、
  4096 environments、500 PPO updates；使用固定 nominal armature，全部 DR 与
  观测噪声关闭。运行时确认加载 `walk_12dof.csv` 的 12 个命名关节、149 帧、
  4.933333 秒闭合片段。任务于 `2026-08-21 15:44:42` 完成，最终 iteration
  `499/500`、`49,152,000` timesteps，最高 checkpoint 为 `model_500`；reward
  `158.78`、episode length `2315.50`，value/surrogate/state-estimator loss 为
  `0.0557/0.0042/0.0033`。参考关节 MAE 从 `0.3111` 降至 `0.1521 rad`，RMSE
  从 `0.3725` 降至 `0.2081 rad`；vx error `0.0352 m/s`、response `0.8820`、
  too-slow `0.0034`、lateral drift `0.0221 m/s`、foot slip `0.9913 m/s`。
  未发现 NaN/Inf、Traceback、CUDA OOM 或 RuntimeError；这些训练指标证明参考
  轨迹学习已启动，但在 deterministic nominal Isaac Gym 视频回放前不把动作
  外观判定为通过。
- 直接完整 DR：`TASK_20260820_201`，从随机初始化训练，并启用包含 armature 在内的完整随机化；该路线保留。
- 无 DR nominal-armature 基线：`TASK_20260821_006`，分支
  `experiment/no-dr-nominal-armature` / `0c7bfc0`，随机初始化、seed 5、
  4096 environments。用户确认该阶段无需机械跑满 5000 iterations，任务于
  `2026-08-21 10:09:44` 计划内早停；状态 6 不代表训练失败，最高完整
  checkpoint 为 `model_4700`，作为后续有效恢复源。
- nominal-armature 课程完整 DR：`TASK_20260821_073`，从有效
  `TASK_20260821_028/model_6700` 恢复，分支
  `experiment/nominal-armature-pipeline` / `fcfa166`，4096 environments、
  seed 5，追加 3000 PPO updates。任务于 `2026-08-21 12:27:29` 正常完成，
  最终日志为 iteration `9698/9699`，本阶段新增 `294,912,000` timesteps，
  最高完整 checkpoint 为 `model_9600`。最终单次 reward `110.31`、episode
  length `2036.58`、vx response `0.8117`、active vx error `0.1627`、foot
  slip `1.0285`；未发现 NaN/Inf、Traceback、CUDA OOM 或 RuntimeError。
- 上述完整 DR 最终模型的 deterministic nominal Isaac Gym 推理：
  `TASK_20260821_121`，源为 `TASK_20260821_073/model_9600`，推理 commit
  `d8681f3`，使用 L20、`x1_dh_stand_dr_full` 和
  `--armature_mode=nominal`。日志确认 nominal dynamics、plane friction
  `0.6`、推理期 DR/观测噪声关闭、12 关节 nominal armature、50 Hz 平滑
  相机、2000 steps/1000 帧，平均前进速度 `0.347 m/s`（目标 `0.4 m/s`）、
  平均高度 `0.602 m`，无 Traceback/CUDA OOM/RuntimeError。视频已下载并
  验证为 1920×1080、50 fps、20 秒、1000 帧，路径为
  `outputs/isaacgym/TASK_20260821_121/play_output.mp4`。同目录保存 2000 点
  `isaac_diag.csv`、`torque_summary.csv`、全关节和 hip 扭矩时间曲线；本次
  最大绝对扭矩为右 hip roll `60.62 N·m`，约为有效上限的 `47.5%`。
- Stage-1 `model_6700` 的最终 nominal 扭矩诊断：资源池任务
  `TASK_20260821_110` 于 `2026-08-21 12:33:12` 最先完成，原队列任务
  `TASK_20260821_074` 随后于 `12:38:39` 完成等价重放；两者推理 commit
  均为 `d8681f3`。日志确认 nominal dynamics、plane friction `0.6`、推理期
  DR/观测噪声关闭、12 关节 nominal armature、50 Hz 平滑相机、2000 steps/
  1000 帧，平均前进速度 `0.340 m/s`、平均高度 `0.610 m`，均无
  Traceback/CUDA OOM/RuntimeError。以最先完成的 `TASK_110` 作为交付源；
  视频已验证为 1920×1080、50 fps、20 秒、1000 帧，输出位于
  `outputs/isaacgym/TASK_20260821_110/`。2000 点扭矩数据中最大绝对值为
  左膝 `111.30 N·m`，最高限幅利用率为右踝 pitch `79.1%`，未触及硬限幅。
- 直接完整 DR 的 deterministic nominal Isaac Gym 推理：`TASK_20260821_014`，
  使用 `TASK_20260820_201/model_6000`、分支
  `experiment/nominal-armature-pipeline` / `dee653b`。日志确认 nominal 动力学、
  plane friction `0.6`、DR/观测噪声关闭、12 关节 nominal armature、2000 steps/
  1000 帧；平均前进速度 `0.349 m/s`（目标 `0.4 m/s`），平均高度 `0.607 m`。
  视频已下载并验证为 1920×1080、50 fps、20 秒，路径为
  `outputs/isaacgym/TASK_20260821_014/play_output.mp4`。
- 直接完整 DR 的相机修正版推理：`TASK_20260821_037`，仍使用
  `TASK_20260820_201/model_6000`，推理 commit `eb25b2d`；策略、nominal
  动力学、12 关节 armature 和推理指令与 `TASK_20260821_014` 相同，仅将
  跟随相机从每 5 个视频帧更新一次改为与 50 fps 录像逐帧同步，并保留
  `0.5 s` EMA 时间常数。日志确认 2000 steps/1000 帧，平均前进速度
  `0.349 m/s`、平均高度 `0.607 m`，无 Traceback/CUDA OOM。视频已下载并
  验证为 1920×1080、50 fps、20 秒，路径为
  `outputs/isaacgym/TASK_20260821_037/play_output.mp4`。对前 500 帧侧方背景
  的光流分析显示：旧视频 `80.6%` 相邻帧近似静止且每第 5 帧跳变；新版
  每帧连续移动，背景水平运动跳变量的 95 分位由约 `1.050 px` 降至
  `0.038 px`（480×270 分析尺度）。后续以修正版视频替代旧视频做视觉比较。
- `TASK_20260821_037` 的可复现扭矩诊断重放：`TASK_20260821_055`，仍使用
  `TASK_20260820_201/model_6000`、deterministic nominal Isaac Gym 和 50 Hz
  平滑相机，推理 commit `0e904f8`。新旧 MP4 的 SHA-256 完全相同，确认策略
  轨迹与 `TASK_20260821_037` 一致。以 100 Hz 记录 20 秒、2000 个控制步的
  12 关节实际施加扭矩；整机最大绝对扭矩为左膝 `105.92 N·m`，左右 hip
  pitch 峰值分别为 `91.05/93.44 N·m`，左右 hip roll 为
  `37.34/45.73 N·m`，左右 hip yaw 为 `37.65/55.48 N·m`，均未触及硬限幅。
  原始数据、统计表和曲线分别保存在
  `outputs/isaacgym/TASK_20260821_055/isaac_diag.csv`、
  `torque_summary.csv`、`torque_time_series.png` 和
  `hip_torque_time_series.png`；对应视频亦已下载并验证为 1920×1080、
  50 fps、20 秒、1000 帧。
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

- 重定向步态站姿几何修正正式训练：`TASK_20260822_095`，分支
  `experiment/retarget-walk-stance-geometry`，实际训练 commit
  `0797292b4b4465bf90a45927400c785f847d7b5b`，从随机初始化，seed 5、
  4096 environments、3000 PPO updates、1×4090D。该任务基于保守抬脚版本，
  新增机身坐标系下的足横向间距、膝横向间距、足朝向和 hip yaw 参考奖励，
  权重分别为 `1.0/0.5/0.5/0.5`；关闭旧的足/膝 XY 总距离奖励。足横向参考
  下限/安全下限为 `0.12/0.10 m`，膝横向为 `0.14/0.12 m`；摆动足参考朝向
  限制在 `±10°`，支撑足目标为 `0°`。nominal armature 固定，DR 和观测噪声
  关闭。云端 smoke `TASK_20260822_092` 已完成 50 updates，未发现 NaN/Inf、
  Traceback、CUDA OOM 或 RuntimeError；正式任务启动日志已确认加载 147 帧周期
  参考和新几何目标。初期 `77/3000` 时足横向 `0.1883 m`（目标 `0.1855 m`）、
  膝横向 `0.1695 m`（目标 `0.1914 m`）、支撑足绝对 yaw `0.1294 rad`；这些
  仍是 scratch 早期诊断值，不作为最终步态结论。
- 重定向步态保守抬脚奖励正式训练：`TASK_20260822_089`，分支
  `experiment/retarget-walk-clearance-conservative`，训练 commit
  `ecd12ae435690352766e77123e2f108d81b69ccf`，从随机初始化，seed 5、
  4096 environments、3000 PPO updates、1×4090D。使用重定向数据的原始足部
  高度、`ref_feet_clearance=2.0`，且只有“参考非接触且高度超过阈值”时才启用
  摆动高度奖励；nominal armature 固定，DR 和观测噪声关闭。此前独立 smoke
  `TASK_20260822_087` 已正常完成 50 updates，最高 checkpoint 为 `model_50`。
- 当前原版周期接触训练的阶段 checkpoint `TASK_20260822_084/model_2500`
  已由 `TASK_20260822_086` 完成 deterministic nominal Isaac Gym 推理。
  日志确认 commit `2516501d7be2edcfeaa102473d38058c8a66f9bc`、nominal
  armature、plane friction 0.6、DR/噪声关闭；视频已验证为 1920×1080、
  50 fps、20 秒、1000 帧，位于
  `F:\robot_f1\worktrees\retarget-walk-imitation\outputs\isaacgym\TASK_20260822_086\play_output.mp4`。
- 旧 profile 下的 `TASK_20260821_061` 因余额不足未运行，不作为有效训练
  任务，也不得作为恢复来源。
- 独立资源池 B：profile `x1-pool-b-20260821`、project
  `PRO_20260821_019`。为规避 4090D SKU 的全局排队，已在 1×4090 SKU 上
  创建并启动第二个等价诊断 `TASK_20260821_111`，与资源池 A 的
  `TASK_20260821_110` 竞速；两者源 checkpoint、nominal 推理参数和诊断代码
  一致，先完整产出者作为交付源。账号密码和 API key 仅保存在本机凭据
  管理器与 gm profile，不写入仓库。
- 新 Stage-1 DR：`TASK_20260821_028`，源为
  `TASK_20260821_006/model_4700`，分支 `experiment/nominal-armature-pipeline`，
  seed 5、4096 environments，初始追加 2000 PPO updates。保留 nominal
  armature，仅随机 friction `[0.45, 0.80]`、base mass `[-1, 1] kg`、COM
  各轴 `±0.015 m`、PD gains/torque `[0.95, 1.05]`。
- Stage-1 中间检查点 `model_6200` 已通过 `TASK_20260821_040` 做一次
  deterministic nominal Isaac Gym 渲染，推理 commit `0b18660`。日志确认
  nominal 动力学、DR/噪声关闭、12 关节 nominal armature、50 Hz 平滑相机，
  2000 steps/1000 帧；平均前进速度 `0.400 m/s`（目标 `0.4 m/s`），平均高度
  `0.614 m`，无异常。视频已验证为 1920×1080、50 fps、20 秒，路径为
  `outputs/isaacgym/TASK_20260821_040/play_output.mp4`。这是中间视觉检查，
  不替代 Stage-1 最终 checkpoint 的门槛评估。
- 标准 Isaac Gym 推理收敛为 nominal-only；随机 DR 鲁棒性测试使用独立评估流程，不与普通渲染混用。

## 已作废并退出实验矩阵

- `TASK_20260820_191`：无 DR 训练使用零 armature。
- `TASK_20260820_199`：Stage-1 DR 继承零 armature 动力学。
- `TASK_20260820_208`：课程路线从零 armature Stage-1 checkpoint 启动，后续不作为主要课程基线。
- 基于上述 checkpoint 的推理任务和视频不再引用、渲染或参与指标比较。
- 上述 ID 仅作为作废原因索引，不保留性能结论，也不作为任何新任务的恢复来源。

## 下一步

1. 以 `TASK_20260821_073/model_9600` 和 `TASK_20260821_121` 作为本轮完整
   DR 课程的训练与 nominal 推理结果；结合最近训练窗口与视频复核，再决定
   是否需要从 `model_9600` 追加 1000 updates，不机械续训。
2. 以 `TASK_20260821_110` 的视频、100 Hz 扭矩数据和曲线作为 Stage-1 最终
   nominal 诊断交付；`TASK_074` 仅保留为结果一致的独立重复。
3. 继续只读监控资源池 B 的 `TASK_20260821_111`；若获得 GPU 则允许自然
   完成，不主动停止；诊断任务结束后，资源池账号
   继续保留给后续独立推理或追加训练，避免必须等待现有账号 GPU 全部释放。

## 关键决策

- `2026-08-21`：代码主线收敛到统一的 X1 训练预设；no-DR、Stage-1 和
  full-DR 由 task 配置区分，直接 full-DR 与 Stage-1→full-DR 共用
  `x1_dh_stand_dr_full`，通过 scratch/resume 运行配置区分，不再为相同动力学
  长期维护独立实验分支。
- `2026-08-20`：先用无 DR 验证可训练性，再逐步增加域随机化，并保留直接完整 DR 作为 A/B 基线。
- `2026-08-20`：训练指令保持旧版的前后、横移、转向和站立随机指令，其他调试与奖励调整采用新版方案。
- `2026-08-21`：Isaac Gym 训练时的有效动力学参数作为跨模拟器整理基准。
- `2026-08-21`：armature 由外部 JSON 按关节名称统一读取；无 DR 使用 nominal，DR 使用 train range。
- `2026-08-21`：常规 Isaac Gym 推理只允许 `nominal`；零 armature 结果退出实验矩阵，随机鲁棒性由独立评估流程承担。
- `2026-08-21`：nominal 无 DR 以可训练性和稳定行为为阶段目标，不要求固定跑满 5000；`model_4700` 作为本轮有效终点。
- `2026-08-21`：Isaac Gym 标准渲染的跟随相机与 50 fps 录制逐帧同步，
  使用约 `0.5 s` 的时间平滑；该调整只影响画面，不改变策略、观测、动作或物理仿真。
- `2026-08-21`：采用 `AGENTS.md`、专项文档、自动化测试和项目 Skill 分层管理工程规则与状态。
- `2026-08-21`：用户授权按并行需求提前申请 Gradmotion 账号和 GPU 资源池，
  不要求等待现有机器耗尽；不得中断正在运行的有效任务，凭据不得进入仓库。

## 风险与注意事项

- 旧零 armature 任务不得再作为恢复训练来源或比较基线，避免污染后续实验谱系。
- 当前共享 nominal 已逐项对齐上游 `x1-training-all-parameter` 分支：hip
  pitch/roll/yaw 为 `0.208/0.025/0.0148`，knee pitch 为 `0.2728`，ankle
  pitch/roll 为 `0.15/0.035` kg m^2；左右对称。上游仍将 hip roll 标记为
  未辨识，其 `0.025` 只是推理中心值，后续仍需系统辨识校准。
- Windows 本地当前可做 MuJoCo 检查，但没有 Isaac Gym；Isaac Gym 运行正确性必须由云端 smoke 验证。
- 部分动力学参数位于 URDF/MJCF，部分位于配置或代码；只同步模型文件不能保证两个模拟器一致。
- 训练 reward、episode length 或视频改善不等于 Sim2Real/真机正确；需要分别保留参数一致性、跨仿真和真机证据。
- Gradmotion task 的实时状态、资源余额和 ETA 会变化，使用前必须重新查询。

## 更新规则

- 阶段变化、关键实验终态、重要决策、长期阻塞或风险变化时更新。
- 日常每十分钟监控数据和临时调试输出不写入本文档。
- 记录任务时至少包含 task ID、Git commit、起始 checkpoint、目标迭代和 DR 方案。
