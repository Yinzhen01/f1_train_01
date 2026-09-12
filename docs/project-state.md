# 项目状态

本文档用于长期项目的低成本上下文恢复，只记录会影响后续工作的阶段、决策、阻塞和风险。Gradmotion 任务实时状态应使用 gm-cli 查询，不在这里伪装成实时信息。

## 当前阶段

2026-09-12：用户已批准相位加速度新分支提交推送及云端短测，明确选择三组
实验前的 `TASK_20260911_171/model8000`，不是上一轮 1x/model9000。
已添加严格恢复入口 `train_gmr_phase_smoke.py`，只允许 512 环境、20 次新增更新、
seed5、固定 lr1e-5、全部网络恢复及优化器重置，目标 model8020。118 项本地
测试通过。仅授权短测，不启动长训练，不恢复刚删除的旧定时任务。
以下为实现阶段历史；实时发布和云端结果见 `docs/gmr-kit317-phase-accel.md`。

2026-09-12：用户要求处理摆动期门控空档及相位化加速度惩罚。本 worktree 新建
`experiment/gmr-kit317-phase-accel-12dof`，从 `bbefd04` 派生；新环境
`x1_gmr_phase_accel` 将禁止触地门控与参考高度脱钩，按左右腿参考摆动相位
累计 1 kHz 加速度平方，保留原全相位弱加速度惩罚及全部动力学/网络/参考。
原三组实验保持不变。115 项本地测试、语法检查、原轨迹门控回归通过；原
1x 轨迹中 24 个先前零门控的触地脚样本得到覆盖，高度代价未变。这不是训练
改善证据。新系数 -0.02 / 归一化 100 rad/s² 仍待云端短测，本机没有 Isaac Gym。
本轮未提交、未推送、未启动训练，尚未指定源 checkpoint 或建立新严格恢复入口。
接触力仍是 100 Hz 端点采样，不能声称捕获全部短脉冲。实现和后续门禁见
`docs/gmr-kit317-phase-accel.md`。下方保留旧实验交付记录，不表示本轮正在训练。

2026-09-12 10:12 北京时间：本轮三组加速度实验的训练、推理、指标比较和视频
均已完成并核验。三组双视角视频及包含原model8000的四组同视角对比均为
50fps、231帧、4.62秒H264；完整解码与对比首/中/末帧检查通过。
最终媒体与中文报告位于 `outputs/policy_videos/gmr_accel_comparison/`，
各组完整双视角视频保留在对应 `gmr_accel_{1x,3x,10x}_model9000/` 目录。
本轮未证明提高全局加速度权重解决抖动；未替换部署策略，未追加训练。
相位门控/1kHz诊断属于后续讨论，未混入本轮实验。本轮监控在交付后暂停。
下方为完成前的历史快照，不表示仍有渲染进程运行。

2026-09-12 09:24 北京时间：加速度三组正式训练033/034/035及推理054/055/056
均正常完成，model9000、配置清单和轨迹包已下载严格核验。
三组均完整走完4.6秒且无物理终止。加速度RMS为54.57/61.49/60.16 rad/s²，
提高权重未优于1x同预算对照。1x位移减少；10x接触力峰值降低但足底速度增大。
推理提交5cf4359，仅身份、安全加载与视频标签适配；100项本地测试通过。
当前正在生成完整MuJoCo可视化视频，尚未替换部署模型或追加训练。
详见 `docs/gmr-kit317-accel-results.md`，完整证据在ignored输出目录。
下方保留实验启动快照，不代表实时状态。

2026-09-12：用户批准新建加速度权重三组对照及新分支提交推送。
本 worktree 为 `experiment/gmr-kit317-accel-12dof`，从 swing `fa391e6` 派生。
三组 `dof_acc=-1e-7/-3e-7/-1e-6`，均从原 TASK_20260911_171/model8000
恢复，seed 5、固定学习率 1e-5、相同新优化器、相同其余奖励与动力学。
每组先 512 环境/20 次更新 smoke，再独立从原模型做 4096 环境/1000 次更新。
98 项本地测试、真实源模型 33 张量严格恢复及有限前向检查通过。
训练代码 `eccd14eed21ffe980c90abd7dc740d043e01fcaf` 已推送；三组 smoke
TASK_20260912_030/031/032 全部正常完成，model8020 与配置清单下载核验通过。
正式任务 TASK_20260912_033/034/035 分别对应 1x/3x/10x，均已实际进入 PPO，
截至 2026-09-12 08:42 北京时间新增完成 80/60/60 次更新（目标各1000）。
原 model8000 恢复、有效权重、动力学和4096环境已从启动日志核验，尚无最终效果结论。
本任务原心跳 `gmr` 已更新为每20分钟监控三组，完成后进行同协议推理与视频比较。
核验报告和任务记录在 ignored `outputs/gradmotion-accel/`；实时状态须重新查询。
详见 `docs/gmr-kit317-accel.md`。下方为来源历史。

2026-09-12 复核：`TASK_20260911_171` 已于 2026-09-11 17:13:28 正常完成，
最终 model8000 已上传并下载，内部迭代 7999、33 个网络张量均为有限值。
用户批准的推理适配 `1cfdf8a` 已提交推送，没有改变训练奖励或动力学。
注册推理 `TASK_20260912_011` 已于 2026-09-12 06:35:38 正常完成，
轨迹包下载/身份/时间线/有效动力学均已校验，MuJoCo 双视角视频已生成。
三次相同确定性回放均完整走到 4.6 秒，无物理失败；实际位移 1.668 m。
相对 model7000：平均接触足底速度 -41.4%，摆动期高度不足 -28.8%，
意外触地比例 16.79%→14.65%；但速度 P95 和抖动未明显改善，
关节 RMSE 与躯干倾角误差增加，不能声称地滑或抖动已解决。
产物目录：`outputs/policy_videos/gmr_swing_task171_model8000/`。
重复确定性试验不是独立鲁棒性试验，MuJoCo 仅展示 Isaac Gym 实际策略轨迹。
详见 `docs/gmr-policy-inference.md`。下方保留训练启动时的快照。

2026-09-11：本 worktree 新增独立 `experiment/gmr-kit317-swing-12dof`，
从 smooth `af2ec05` 派生，只加摆脚最低点高度不足和摆动期意外接触两项惩罚。
恢复源为已完成的 `TASK_20260911_116/model_7000.pt`；计划先 512 环境/20 次
更新 smoke，再从原 model7000 做 4096 环境/1000 次更新。旧奖励、参考、
观测/动作和全部动力学不变。代码 `a92dfe6` 已按用户批准发布。
86 项本地测试和真实 model7000 的 33 张量恢复通过。短测试
`TASK_20260911_168` 正常完成并上传 model7020；恢复身份、旧奖励/动力学
不变及两项新奖励执行均已核实。短测试模型本地下载受 DNS/HTTPS 路由影响，
其内容未本地复核；正式训练不使用该模型。
正式训练 `TASK_20260911_171` 已从原 model7000 实际推进至 7055/8000，
4096 环境、固定学习率 1e-5、seed 5，运行 commit 与 smoke 一致。
尚未完成，不声称摆脚/地滑已改善；后续需确定性推理对比。
实验定义及验收边界见 `docs/gmr-kit317-swing.md`。以下为来源历史。

2026-09-11 终态核实：`TASK_20260911_116` 已于北京时间 15:15:58 正常完成，
最终 `model_7000.pt` 已上传、下载并核对内部迭代 6999 和 33 个有限网络张量。
当前仅适配推理身份检查、轨迹诊断和视频标签，不改变训练配置、参考或动力学。
71 项本地测试及真实训练 URDF 的 MuJoCo 渲染后端检查通过。
推理任务 `TASK_20260911_129` 已于 16:16:40 正常完成，产物已下载、解包并核实；
推理 commit 为 `cafa4ea72c662c987f99446491882ab904fa2939`。三次均跑到 4.6 秒但不等于跟踪成功；
12 DOF 平滑指标有所改善，接触足底速度略升。
详见 `docs/gmr-policy-inference.md`。

本 worktree 为 2026-09-11 新建的 `experiment/gmr-kit317-smooth-12dof`。
用户要求同时启动 12 自由度平滑续训；恢复源已核实为已完成的
`TASK_20260911_008/model_5000.pt`，不是随机初始化，也不是 29 关节模型。
新任务 `x1_gmr_smooth` 保持原参考和全部动力学，只加强动作差分惩罚并
加入归一化力矩变化率惩罚。固定学习率 1e-5，网络权重保留，优化器重置。
67 项本地测试和真实 checkpoint 的 33 个张量严格恢复校验通过。
代码 `21788f6` 已提交并推送。512 环境/20 次更新恢复 smoke
`TASK_20260911_114` 已正常完成（5000..5019），上传 `model_5020.pt`；
33 个张量均有更新且为有限值，恢复身份、运行时参数及新奖励执行已核实。
正式续训 `TASK_20260911_116` 从原 008/model_5000 追加 2000 次更新，
4096 环境、seed 5、4090D，于 14:30:35 被受理，14:35:07 已实际推进到
`5035/7000`（本轮新增 36 次更新）。代码 `21788f6`、源模型/参考身份、
固定学习率及全部运行时动力学与 smoke 一致，启动日志未见数值异常。
接触力瞬态峰值仍较大，不能据 smoke 声称抖动解决；旧 runner 的续训 ETA
显示错误，应按新增更新数/耗时估算。详见 `docs/gmr-kit317-smooth.md`；以下为来源历史。

本 worktree 于 2026-09-11 增加独立 `x1_gmr_upright` 实验：纠正固定腰部
参考的躯干语义，静止直立、行走跟踪缩幅胸部倾角，重建足部一致目标并增加倾角奖励。
旧 `x1_gmr_clip` 保留。48 项本地测试通过，代码 `770a0fc` 已发布；512 环境、
20 次更新的 Isaac Gym smoke `TASK_20260911_007` 正常完成，模型已上传。
新参考与奖励运行可用，但不是后倾改善/策略收敛证明；长训练未启动。根部/关节加速度高于旧参考，须动态验证。详见
`docs/gmr-kit317-upright.md`。以下为旧 GMR 基线和源仓库历史。

此独立副本于 2026-09-10 接入 GMR KIT317 足底修正后的 12 关节非循环动作跟踪。
实验定义、源数据、模型差异、验证边界见 `docs/gmr-kit317-training.md`。
从随机策略开始 nominal 无 DR 训练；不继承下方历史 checkpoint。下方内容保留为
源仓库历史，不代表本实验实时状态。

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
- 当前 hip/knee nominal `0.02505` 是训练范围中点而非系统辨识真值；新路线物理上更合理，但仍需后续辨识校准。
- Windows 本地当前可做 MuJoCo 检查，但没有 Isaac Gym；Isaac Gym 运行正确性必须由云端 smoke 验证。
- 部分动力学参数位于 URDF/MJCF，部分位于配置或代码；只同步模型文件不能保证两个模拟器一致。
- 训练 reward、episode length 或视频改善不等于 Sim2Real/真机正确；需要分别保留参数一致性、跨仿真和真机证据。
- Gradmotion task 的实时状态、资源余额和 ETA 会变化，使用前必须重新查询。

## 更新规则

- 阶段变化、关键实验终态、重要决策、长期阻塞或风险变化时更新。
- 日常每十分钟监控数据和临时调试输出不写入本文档。
- 记录任务时至少包含 task ID、Git commit、起始 checkpoint、目标迭代和 DR 方案。
