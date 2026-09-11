# KIT317：固定腰部躯干参考与奖励修正

## 状态与实验边界

2026-09-11，本地完成新参考、奖励、独立任务及测试。工作区：
`F:\robot_f1\x1-training-gmr-upright`，分支：
`experiment/gmr-kit317-upright-12dof`，基于 `133a85b`。
本地修改完成后，用户于 2026-09-11 授权提交/推送本实验分支，并先进行
512 环境、20 次更新的 Gradmotion smoke。训练代码已发布为 `770a0fc`；
云端 `TASK_20260911_007` 正常完成，实际 commit 与参考哈希均已核对，详见下方云端记录。
旧工作区、旧 `x1_gmr_clip` 任务、参考数据和 checkpoint 均保留。

本实验修改参考和奖励定义，不修改 URDF、腰部自由度、关节限制、PD、armature、
控制周期、观测维数、动作空间或域随机化。仍为 nominal、无 DR、12 个腿关节、
4.6 秒非循环动作。旧 checkpoint 即使张量形状兼容，也不是此新目标的已验证策略。

## 为什么不能只增大旧姿态奖励

原始 29 DOF GMR 的骨盆朝后倾，活动腰部补偿后胸部朝前；12 DOF 训练模型的腰部固定。
若仅保留骨盆四元数、丢弃腰部动作，原 `gmr_root_rotation` 就会奖励整段躯干后倾。
提高该旧目标的权重会加强错误监督。因此先重建一致的固定腰部参考，再强化倾角跟踪。

## 新参考定义

输入为旧 12 DOF 参考、原 29 DOF 足底修正数据、原 GMR XML 和实际训练 URDF。
脚本核验输入哈希、模型关节顺序、固定腰部、时间轴与足部 mesh。

1. 用原 GMR 模型正运动学取胸部 `lumbar_pitch_link` 的世界旋转。
   乘以训练模型固定胸部坐标旋转的逆，得到身体轴对齐的根部姿态候选。
2. 保留原骨盆的 yaw，取对齐后胸部 roll/pitch 趋势，先用 9 帧三阶
   Savitzky–Golay 滤波，再乘以 **0.6** 的幅度系数和站立/行走混合权重。
   0.6 是本算例经过运动学检查选取的工程参数，不是算法自动识别的人体比例。
3. `0–1.0 s`、`4.2–4.6 s` 设为直立；`1.0–1.5 s` 和 `3.7–4.2 s`
   用五次 smoothstep 平滑过渡。时间段仅适用于该 KIT317，不应直接套用于其他动作。
4. 将旧参考的双脚世界位姿作为目标，重新求根部平移和 12 关节角。
   允许行走 roll/pitch 各偏离上述目标至多 1°；静止仅有 1e-6 rad 数值容差。
   因为固定腰部改变了腿部几何关系，不能只改根部四元数而保留旧关节/足部标签。
5. 求解保持原关节上下限，并约束相邻帧斜率至 URDF 速度上限的 98%。
   对根部修正量进行 21 帧三阶滤波，并在原约束下重新求解，共三遍。
   不直接滤波最终关节角后跳过足部复查。
6. 用新根部/关节轨迹重新计算根部线速度、世界角速度、关节速度、
   足底在基座坐标中的位置和旋转。保留时间轴、接触标签、航向和足底标定。
   100 Hz 插值检查后整体抬高 0.38343 mm，使最小足部 mesh 高度为 0.5 mm。

求解残差、权重和候选验收阈值均在 `humanoid/scripts/prepare_gmr_upright.py` 中。
该脚本拒绝覆盖已有输出；失败候选不进入正式参考文件。

## 奖励变化

新任务：`x1_gmr_upright`。配置与环境分别为：

- `humanoid/envs/x1/x1_gmr_upright_config.py`
- `humanoid/envs/x1/x1_gmr_upright_env.py`

新增 `gmr_trunk_tilt`，权重 **1.0**，`trunk_tilt_sigma=0.1`。
其余奖励系数继承旧 GMR 配置；原 `gmr_root_rotation=1.0` 保留，但同样跟踪新参考。
不启用“所有时刻都强迫直立”的旧 `orientation` 奖励，以免与行走前倾目标冲突。

令 `g`、`g*` 分别为实际姿态和参考姿态下、表达在各自身体坐标系的单位重力向量：

```text
e² = ||g - g*||²
r_trunk = exp(-e² / 0.1²)
```

世界 yaw 改变不会改变该倾角误差。对于单轴倾角差 θ，
`e² = 4 sin²(θ/2)`；误差约 5.73° 时单项奖励降到 `exp(-1)`。
静止目标是 0°，行走目标是新参考的适度前倾，不是同一时刻同时监督两个不同目标。
奖励权重沿用训练器乘控制步长的机制：0.01 秒控制周期下，新增单步奖励上限为 0.01。

新增日志项：`gmr_trunk_tilt_error_deg`、`gmr_actual_pitch_deg`、
`gmr_target_pitch_deg`。保留足滑、关节跟踪和接触指标。
由于新增奖励且参考改变，不能仅按新旧总 reward 数值判断性能优劣。

## 本地验证结果

参考文件：`resources/motions/gmr_kit317_upright_12dof.npz`。
SHA-256：`d625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb`。
源文件与 mesh 哈希、详细指标见同名 JSON，配置中固定该 NPZ 哈希。

| 指标 | 新参考结果 |
| --- | ---: |
| 静止根部 pitch 绝对值最大值 | 0.0000573°，数值上直立 |
| 全 139 帧根部 pitch 平均值 | +4.506°，正值为前倾 |
| 行走前倾峰值 | +12.162° |
| 相对旧参考根部平移最大变化 | 44.314 mm |
| 原始采样帧踝位姿位置误差上界 | 0.826 mm，含整体抬高的保守界 |
| 原始采样帧踝朝向最大变化 | 0.001614 rad，约 0.0925° |
| 100 Hz 足底位置相对旧参考最大变化 | 1.242 mm |
| 100 Hz 足部 mesh 最低高度 | +0.500 mm，无采样点几何穿透 |
| 静止足底法向倾角最大值 | 0.004743° |
| 相邻帧关节角插值斜率最大值 | 8.722 rad/s，各关节分别满足 URDF 速度上限 |
| 根部加速度峰值 | 11.709 m/s² |
| 关节加速度峰值 | 119.340 rad/s² |

旧参考同样按 139 帧统计的 pitch 均值为 -8.505°；与之前 100 Hz 推理时间轴
统计得到的数值略有差异，不应把不同采样均值混为同一指标。

注意：根部加速度峰值比旧参考的 7.833 m/s² 高，关节加速度也比旧参考的
95.954 rad/s² 高。本轮只是剔除了更大的 IK 尖峰，**不是所有动态指标均优于旧参考**。
12 m/s² 是此候选筛选用的工程阈值，不是机器人动力学可行性标准。
100 Hz 几何检查不是连续时间穿透证明，足底轨迹接近也不等于策略实际足滑改善。

本地 `unittest` 共 48 项通过，包括实际奖励张量数学、yaw 不变性、错误参考拒绝、
旧系数继承、参考哈希、速度上下限，以及独立 URDF FK 的局部足底标签和 100 Hz mesh 检查。
Python 编译检查通过。随后通过下述 Isaac Gym 短训练；仍未验证策略收敛、力矩可行性、
摩擦锥、动态平衡或真机安全。

对照图位于 `outputs/upright_comparison/`：`reference_poses.png` 与
`reference_curves.png`。它们仅展示新旧参考，不是新训练策略视频。

## 云端短训练记录（2026-09-11）

- 代码：`770a0fc3d42660293cd4abfaff04cae8aa93e8c7`，已推送 `publish` 新实验分支。
- 有效任务：`TASK_20260911_007`，项目 `PRO_20260820_014`，状态 `5`（正常完成）。
- 平台最终时间：北京时间 `09:10:55–09:13:13`；PPO 循环日志耗时 `15.16 s`。
- 1×4090D 24GB `ESKU000001`，镜像 `BJX00000001/V000124`。
- 随机初始化，512 environments，seed 5；完成 20 次更新、245,760 timesteps。
- 运行日志确认 139 帧/4.6 秒、12 关节、nominal 无 DR、正确 URDF/参考哈希，
  `upright_chest_v1`、倾角奖励权重 1.0 / sigma 0.1。
- 检查 53,486 字符日志，未发现 Traceback、RuntimeError、CUDA OOM、NaN 或
  loss/reward/std 非有限值；平台已上传 `model_0.pt` 和 `model_20.pt`。
- 新增 3 个躯干姿态 debug 指标和 `Episode/rew_gmr_trunk_tilt` 均出现在图表 API，
  所取样本为有限值。

原项目账号的 `TASK_20260911_006` 因余额不足未能启动，草稿保留。
按照既有账号池规则使用另一现有账号完成短训练，没有新注册、购买或启动长训练。
临时环境变量仅用于该命令，不修改全局默认 profile。账号来源与失败尝试记录在
本地忽略目录 `outputs/gradmotion-upright-smoke/run-record.json`，不含凭据。

末次训练批次：mean reward 5.05、mean episode length 105.50 steps、
接触足底水平速度 0.1019 m/s、躯干倾角误差约 11.97°。
这些是随机参考起点、带探索动作的短训练统计，不是完整 4.6 秒确定性推理；
**只通过运行链路验证，尚不能判定后倾改善或足滑优于旧策略**。
本地另保存 `metrics.json` 与 `training-excerpt.log`。

## 复现与下一步

从本工作区生成新文件时，指定一个尚不存在的输出路径：

```powershell
python humanoid/scripts/prepare_gmr_upright.py --baseline resources/motions/gmr_kit317_foot_flat_12dof.npz --source ../GMR/outputs/kit317_walking_f1_v1_4/foot_flat/KIT_317_walking_medium08_f1_v1_4_foot_flat.npz --source-xml ../GMR/assets/f1_v1_4/custom.xml --urdf resources/robots/x1/urdf/X1_12DOF.urdf --walking-tilt-scale 0.6 --output outputs/upright_reproduction/reference.npz
python -m unittest discover -s tests -v
python humanoid/scripts/inspect_gmr_upright.py --baseline resources/motions/gmr_kit317_foot_flat_12dof.npz --reference resources/motions/gmr_kit317_upright_12dof.npz --urdf resources/robots/x1/urdf/X1_12DOF.urdf --output outputs/upright_comparison_new
```

提交/推送与短训练已得到用户确认。通过 Gradmotion 正式注册短训练，提交前回读资源列表确认
4090D `ESKU000001` 可用，不自动更换机型。建议 smoke 入口：

```bash
python humanoid/scripts/train.py --task=x1_gmr_upright --num_envs=512 --max_iterations=20 --seed=5 --run_name=gmr_upright_smoke --headless
```

上述环境、哈希、奖励、有限值与上传检查已经通过。
下一步需用户授权后再从随机初始化开展正式训练；本配置继承 5000 次迭代，但本轮没有启动该训练。
验收必须比较固定从 0 秒开始的完整推理：完成率、躯干倾角误差、实际足滑、穿透、
关节跟踪和扭矩限幅，而不是只比较总奖励。

旧 `record_gmr_policy.py` 刻意限定旧 `x1_gmr_clip/model_5000`。
新实验推理前需显式扩展其任务/参考身份检查和 manifest，不能借用旧任务名冒充新策略。
