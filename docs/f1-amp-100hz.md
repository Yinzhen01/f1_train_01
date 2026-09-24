# F1 V1.4 / 100 Hz AMP 接入说明与交付入口

## 1. 当前结论与代码基线

2026-09-24：完成多片段数据适配、10 帧有效历史、连续判别器、PPO 数据接口和 CPU 验证。
**没有完成 Isaac Gym 环境运行或正式 AMP 训练，没有动力学/真机放行。**

- 工作树：`F:\robot_f1\worktrees\f1-amp-100hz-12dof`
- 分支：`experiment/f1-amp-100hz-12dof`
- 基础提交：`3c4b611839f68e1e0ccbc4cea607b0bf3d5653fd`
- 原分支：`experiment/gmr-kit317-cycle-contact-12dof`；用户确认后创建独立工作树。
- 理由：原分支干净，已有 12 DOF GMR/100 Hz 控制链；不混入 main 的未提交 URDF 修改或 scuff 的未提交擦地奖励。
- 现有环境、Actor、奖励、动力学和训练入口全部保持原样。AMP 是独立准备模块，未注册新训练任务。
- 未提交、未推送。运行产物在仓库外独立目录，不提交数据或旧训练结果。

统一交付入口：本文件。数值报告见
[质量报告](F:/robot_f1/outputs/f1_amp_100hz_20260924_verified/质量报告.md)、
[AMP 清单](F:/robot_f1/outputs/f1_amp_100hz_20260924_verified/amp_manifest.json)、
[CPU 结果](F:/robot_f1/outputs/f1_amp_100hz_20260924_verified/cpu_dry_run.json)、
[最终核验与完整测试日志](F:/robot_f1/outputs/f1_amp_100hz_20260924_verified/verification.json)。

## 2. 证据来源与边界

|来源|负责的事实|使用限制|
|---|---|---|
|[动作数据筛选总览 8.5](F:/robot_f1/outputs/motion_dataset_visual_review_20260923/动作数据筛选总览.md)|人体先插值 100 Hz 再 IK；源分组、几何后处理|原文不是动力学验收|
|[原核验清单](F:/robot_f1/outputs/motion_dataset_visual_review_20260923/f1_retarget_100hz_20260923/retarget_manifest.json)|10 个 NPZ 哈希、WALK_10 hold|只读，未升级 training_ready|
|[训练 URDF](F:/robot_f1/worktrees/f1-amp-100hz-12dof/resources/robots/x1/urdf/X1_12DOF.urdf)|12 关节真实名称、限位、link 坐标与 FK|与 GMR 的 F1 MJCF 不完全相同|
|[旧单轨迹加载器](F:/robot_f1/x1-training-gmr-cycle/humanoid/gmr_motion.py)|旧数据接口与按名映射参考|不复用末端 clamp 生成 AMP 历史|
|[robotparty 判别器](F:/robotparty/roboto_origin/modules/roboparty_train/rsl_rl/rsl_rl/modules/amp.py)、[PPOAMP](F:/robotparty/roboto_origin/modules/roboparty_train/rsl_rl/rsl_rl/algorithms/ppo_amp.py)|连续 LSGAN、独立优化器、无梯度风格奖励参考|未修改；不复制 23 关节/67 维或其任务混合系数|

本次按来源核对原则将“输入读取通过”“CPU 计算通过”“模型不一致”“未做动力学”分开记录。没有搜索新数据、重新运行 GMR、修改或覆盖原数据/视频/manifest/机器人资产。

## 3. 输入、派生输出和分组

输入根目录：`F:\robot_f1\outputs\motion_dataset_visual_review_20260923\f1_retarget_100hz_20260923`

输出根目录：`F:\robot_f1\outputs\f1_amp_100hz_20260924_verified`

请以此 `_verified` 目录为交付版本。早期 `f1_amp_100hz_20260924` 快照未删除/覆盖，仅保留复核过程；最终代码格式清理后在新目录完整重生成并校验，避免原快照代码哈希失配。

全部 10 段输入读取成功：8 段 600 帧行走、2 段 401 帧站立，共 5,602 帧；严格 100 Hz，原 qpos 为 N×36、根四元数 wxyz。

|分组|片段|采样/统计权限|
|---|---|---|
|首批行走候选|01/02/03/04|CPU 首批采样及归一化拟合；尚不允许正式训练|
|储备|08/09/10|不进入首批采样/统计；10 保留人工 hold|
|留出|15|只用于最终独立评估；本轮仅完整性/质量诊断，不参与拟合、归一化和调参|
|站立单组|R2_009/R2_010|不混入首批行走；同演员同源，不作为独立泛化对照|

多片段采样：显式片段权重或时长权重 → 选择允许片段 → 选择合法起点 → 连续 10 帧。
默认四条等片段权重各 0.25；非按原文件帧数隐式加权。

派生 `amp_features.npz` 每段包含：

- `features`: float32 `[N,39]`，未归一化 SI 特征。
- `time_s`、`fps=100`、`valid_frame`：第 0 帧 false，其余 true。第 0 帧导数占位不能送入判别器或拟合统计。
- `feature_spec_json`：顺序、单位、坐标系、采样率和窗口协议。
- `metadata_json`：输入 motion SHA256、训练 URDF LF SHA256、分组、`training_ready=false`、`cyclic=false`。

同目录 `quality.json` 保存最终/raw/pre_lift 三版本诊断；没有替换原 qpos 或 raw_qpos。
根 `normalization.npz` 只拟合 01–04 的 2,396 个有效帧，之后冻结，两侧共用同一均值/标准差；不在线吸收 holdout。
根 `amp_manifest.json` 记录输入/输出 SHA、分组、fps、帧数、时长、hold 和权限。
`split_eligible_for_training=true` 仅代表候选分组；所有 `allowed_for_training=false`，不能解释为已放行。

## 4. 39 维特征协议

|切片（Python）|特征|单位/坐标系|
|---|---|---|
|`0:3`|基座角速度|rad/s，当前根本体系|
|`3:15`|12 腿关节位置|rad，原 qpos 命名映射，不裁剪|
|`15:27`|12 腿关节速度|rad/s，100 Hz 后向差分|
|`27:39`|左膝、右膝、左踝、右踝 link 原点相对根位置|m，根本体系，各 xyz 连续展开|

关节顺序：左腿 hip_pitch / hip_roll / hip_yaw / knee_pitch / ankle_pitch / ankle_roll，然后右腿同序。
具体名称写入配置和 FeatureSpec，不依赖 qpos 的固定关节列号。body 为左右 `knee_pitch_link`、`ankle_roll_link`，根为 `base_link`。

源 wxyz 显式转换为内部 xyzw、归一化；角速度用相邻 SO(3) 最短相对旋转，不对四元数分量做差。
两侧相同的因果差分：`qdot[k]=(q[k]-q[k-1])/0.01`；策略端使用每个真实物理状态的关节角，不读取加噪/延迟 Actor 观测来算 AMP。
示范关键点通过训练 URDF FK 重新构造，策略通过相同命名的 link 世界位置减根位置并逆旋转，再进入同一个 `build_features`。

不含：绝对根位置、高度、根竖直速度、全局航向、source_support 接触标签、17 个上身零关节。
根姿态只用于相对旋转和坐标变换；全局平移/航向不变性已有测试。

重要：训练 URDF 与源 MJCF 不同，因此未直接复制源模型身体坐标。这个 FK 转换解决特征几何来源一致性，**不解决超限或物理可执行性**。

## 5. 窗口、reset 与 PPO 接口

```text
示范：单片段 qpos → 按名映射/因果差分/训练模型 FK → [N,39]
    → 合法起点（600帧片段为1..590）→ [B,10,39]

策略：reset 后 prime 初始真实状态（仅导数种子，不计入历史）
    → 每 0.01 s 物理步后、自动 reset 前采集状态
    → 同一特征函数 → episode-local 历史 → 只取 valid_mask=true

两侧：冻结共用归一化 → [B,390] → 连续标量 D → [B,1]
```

10 帧是 9 个间隔，跨度 **0.09 s**。不跨文件、不循环、不跨 reset、不夹带其他片段。
策略前 9 步不完整历史不会送入判别器；内部未填满位置用 NaN 哨兵，返回给判别器的是有效窗口组成的紧凑 batch。
terminal 的 reset 前状态属于旧 episode，可以形成有效最后窗口；下一 episode 必须重新 prime。

`AMPBridge` 对现有 DHPPO 接口的接法如下（未来任务接入说明，不是本次已运行的 Isaac Gym 环境）：

1. reset 后取得刷新过的 root / joint / rigid body 状态，调用 `bridge.prime(...)`。
2. 100 Hz post-physics、自动 reset **之前**调用 `bridge.capture_pre_reset(...)`。
3. 原环境返回 reward/done/infos 后，由 runner 调 `bridge.process_env_step(ppo, ...)`，替代该处直接调用 `ppo.process_env_step`。
4. 消费完 terminal 转移后，重新获取 reset 环境的状态，仅对这些环境 prime；不能提前 prime 覆盖未消费的转移。
5. 原 `ppo.compute_returns` / `ppo.update` 保持独立；调用 `bridge.update_discriminator(...)` 消费本 rollout 中的有效策略窗口。

外部非 terminal reset 同样必须 prime。缺帧、重复 tick 或 episode 标识不连续会报错，不偷偷拼窗。
Actor 原 66 帧、短历史 5 帧、Critic 3 帧不变；不得把它们跟 AMP 的 10 帧绑在一起。
尚需在真实 Isaac Gym 回读控制周期、关节/body 名称、URDF SHA 和刚体位置语义（link 原点而非 COM），验证上述时序；当前未自动挂接已有 KIT317 训练任务。

## 6. 判别器、奖励与梯度边界

新网络 `390 → 256 → 128 → 1`，ELU，无 Sigmoid；宽度可配置。
LSGAN 为 `0.5 * [mean((D(demo)-1)^2) + mean((D(policy)+1)^2)]`，加示范输入的零中心梯度惩罚。
风格奖励：`0.01 * style_scale * max(0, 1 - 0.25*(D-1)^2)`，在 no_grad 下计算。
混合：`task_weight * task_reward + style_weight * style_reward`。
准备配置 `task_weight=1, style_weight=0`；CPU 测试临时用非零 style_weight 验证接线，不是调出的 F1 参数。
warm-up 无效窗口的 style 为 0，任务项仍按配置计算。

- PPO optimizer 只更新 Actor/Critic。
- 独立 discriminator Adam 只更新 D；policy/demo 输入先 detach。
- 归一化 buffer 冻结、两侧共用；不因 D 更新改变。
- CPU 单步已经验证 PPO 更新不改变 D、D 更新不改变 Actor，且两者各自确有参数变化。

## 7. 文件清单

|文件（相对本工作树）|用途|
|---|---|
|`humanoid/amp/__init__.py`|独立包，不引入 Isaac Gym|
|`humanoid/amp/features.py`|共享39维协议、四元数、名称映射、只读训练 URDF FK|
|`humanoid/amp/dataset.py`|10段严格加载、split/hold 防护、片段采样、训练组归一化|
|`humanoid/amp/history.py`|因果策略状态流、10帧历史、valid mask|
|`humanoid/amp/discriminator.py`|连续 LSGAN、梯度惩罚、独立优化器|
|`humanoid/amp/integration.py`|pre-reset 捕获与 DHPPO reward/update 接口|
|`humanoid/amp/quality.py`|微分、限位、源网格与支撑代理复算、模型差异|
|`humanoid/amp/dry_run.py`|真实 DHPPO 类的 CPU 合成数据单步验证|
|`configs/amp/f1_100hz.json`|固定分组/特征/窗口和可配置网络、采样、奖励|
|`humanoid/scripts/prepare_f1_amp.py`|生成新独立数据/报告目录，拒绝覆盖|
|`humanoid/scripts/verify_f1_amp_bundle.py`|复核输入/派生输出/归一化/资产哈希，运行全套测试并保存新核验证据|
|`tests/test_f1_amp.py`|23项新增 CPU/真实数据测试|
|`docs/f1-amp-100hz.md`、`docs/project-state.md`|统一说明与当前状态|

## 8. 命令与实际验证

本机 Python：`C:\Users\HP\AppData\Local\Programs\Python\Python38\python.exe`，PyTorch 2.4.1+cu121，但本次 device 严格为 CPU；NumPy 1.24.4、SciPy 1.10.1、MuJoCo 3.2.3。
仓库预期训练环境是 PyTorch 1.13 / Isaac Gym；本机成功不能替代那个环境的 smoke。

在本工作树执行（PowerShell）：

```powershell
Set-Location 'F:\robot_f1\worktrees\f1-amp-100hz-12dof'
& 'C:\Users\HP\AppData\Local\Programs\Python\Python38\python.exe' -m unittest discover -s tests -p test_f1_amp.py -v
& 'C:\Users\HP\AppData\Local\Programs\Python\Python38\python.exe' -m unittest discover -s tests -v
```

实际结果：新增 AMP **23 项通过**；全仓库 **165 项通过，无失败/错误/跳过**。
测试涵盖实际10个NPZ、名称乱序、四元数、窗口跨度/边界、split/hold/归一化泄漏、reset mask、特征布局、D前向/损失/奖励、种子复现、独立MuJoCo对训练URDF FK复核和独立优化器。
MuJoCo 导入原 URDF 会警告部分 geom 名为空/重复；未改资产压掉警告，刚体名称与 FK 校验仍通过。
普通导入旧 runner 依赖本机未安装的 wandb；CPU QA 通过隔离包命名空间加载**原 DHPPO/ActorCriticDH/RolloutStorage 源文件**，没有伪造 PPO、安装云训练依赖或修改原 runner。

已执行的生成命令（目录已存在，再执行会明确拒绝覆盖；复现时改为新的输出目录）：

```powershell
& 'C:\Users\HP\AppData\Local\Programs\Python\Python38\python.exe' humanoid/scripts/prepare_f1_amp.py --data-root 'F:\robot_f1\outputs\motion_dataset_visual_review_20260923\f1_retarget_100hz_20260923' --source-model 'F:\robot_f1\GMR\assets\f1_v1_4\scene.xml' --output 'F:\robot_f1\outputs\f1_amp_100hz_20260924_verified'
& 'C:\Users\HP\AppData\Local\Programs\Python\Python38\python.exe' humanoid/scripts/verify_f1_amp_bundle.py --bundle 'F:\robot_f1\outputs\f1_amp_100hz_20260924_verified' --result 'F:\robot_f1\outputs\f1_amp_100hz_20260924_verified\verification.json'
```

CPU dry-run：`[16,10,39] → [16,1]`；真实类的合成 PPO/D 单步损失有限；8 个合成环境前 9 步有效数全为0，第10步起为8。
同一轨迹经 demo 与策略接口的最大特征差 `4.77e-7`；只能证明合成状态下的特征一致性。
282 个原输入/模型文件运行前后哈希一致，10 条原 BVH SHA 和派生特征逐值重算已另行核验。

## 9. 明确未解决的问题

最重要的新发现：不能把“GMR 模型下未超限”当作“训练模型下未超限”。

|首批候选|训练关节限位超限次数（帧×关节）|最大超限 rad|训练速度限值超限（间隔×关节）|
|---|---:|---:|---:|
|01|56|0.01745|0|
|02|47|0.01566|0|
|03|198|0.09000|2|
|04|181|0.09000|36|

01/02 主要是左髋 roll 超过训练上限0.2 rad；03/04主要是踝 pitch 低于训练下限−0.41 rad，源最低达到−0.5 rad。03还含少量右髋roll超限。
右髋 yaw joint 原点差约6.05 mm/0.493°，右膝约5.86 mm/0.493°；组合后的末端差应看逐帧 FK，不可直接相加。以01为例右踝link位置峰值差约1.73 mm。

01–04加速度峰值依次311.3/376.4/438.7/623.0 rad/s²，jerk约3.05万–6.21万 rad/s³；没有用“可读/未NaN”放行这些尖峰。
10保留后处理尖峰hold，加速度约2705.3、jerk约46.2万，未进入拟合。
源模型足底最低约2 mm是逐帧抬根的几何结果，不是接触力/动力学验收。01–04根Z加速度峰值约28.8–73.5 m/s²。
完整 `source_support>.5` 连续段的足底XY路径仍为厘米量级；例如01左/右最大路径31.2/29.3 mm。本口径含支撑转接，不同于此前更短的平足窗口，不作跨口径改善率。
这些支撑标签只是人体启发式代理，未检查力矩、摩擦约束、自碰撞、平衡或真机。

## 10. 下一步最小建议（本次未执行）

1. 先确认最终训练物理模型，处理源/训练模型的限位和几何差异；不能简单放宽URDF限位或硬裁剪角度。
2. 必要时另开数据版本，做接触约束下的轨迹修正/平滑，并重新核验速度、加速度、jerk与足滑。原数据继续只读。
3. 模型/数据门槛通过后，再接入独立Isaac Gym AMP任务，先8–32环境、少量控制步检查pre-reset/prime时序、特征和梯度，再考虑小规模训练smoke。
4. 此前所有“正式训练、动力学可执行、真机可部署”均保持未验证。`assert_formal_training_allowed` 与采样用途检查故意拒绝正式训练；不能只把JSON的training_ready改成true绕过物理验收。
