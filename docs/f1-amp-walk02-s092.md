# 单段 WALK_02 / 左髋 roll 0.92 AMP 仿真实验

## 范围与授权

2026-09-24 用户要求缩放后进行 AMP 训练。本实验仅使用缩放后的 WALK_02，
不使用 01/03/04、储备、站立或留出动作；原数据、模型和原十段 AMP 准备门禁不变。
从零训练，不恢复旧 GMR tracking checkpoint，不替换已有部署策略。

数据来自 `F:/robot_f1/outputs/f1_lafan_walk02_scaled_20260924/motion.npz`，
SHA256 `3c24a7b2abc7fd4bada5a1840e466675bc43beda1fba49e0b98ee858037c173a`。
仓库内将这个 717 KB 文件明确保存为云端实验输入：
`resources/motions/amp_lafan_walk02_s092/motion.npz`，不是覆盖原数据或提交旧训练产物。

## 实验定义

- F1/X1 12 DOF，原训练 URDF、PD、动作缩放、nominal armature、1 kHz 物理、100 Hz 控制不变。
- no-DR 平地、摩擦 0.6，沿用 no-DR 配置；6 秒 episode。
- Actor 为 66×47，短历史 5×47，Critic 为 3×73，网络形状不变。
- 不沿用手工正弦步态：phase 槽位固定为 sin=0、cos=1，Critic 的关节差参考为默认位姿；
  因此观测语义、奖励与旧 checkpoint 不可直接比较，明确 scratch-only。
- 只拟合该缩放片段的 599 个有效特征帧；连续10帧共有590个合法窗口，不循环、不跨reset。
- 39维特征与旧协议一致，不含绝对根高度/位置、根竖直速度或 source_support 标签。
- 冻结归一化，每维 std 下限0.01；示范与策略共用；无留出泄漏。
- 连续 LSGAN 390→256→128→1，无 Sigmoid；独立 Adam lr=1e-4，gradient penalty=10。
- PPO 原实现，scratch lr=3e-4；其余继承；24控制步/rollout。

## 奖励（实验起点，不是已调优参数）

任务项保留速度跟踪、躯干朝向、速度稳定性、平滑/能耗/碰撞/关节限位项，
去掉旧正弦关节参考、相位接触数和脚高等奖励，不混入 KIT317 的逐帧 tracking。
前进指令固定0.45 m/s（示范本体系前进速度中位数约0.4507），横移/转向指令为0。

`r = r_task + r_style`，其中 `r_style = 0.01 × 2 × max(0, 1-(D-1)^2/4)`。
两个权重为1，style尺度为2；这不是“风格奖励占比固定50%”，实际贡献记录到日志。
前9个不完整策略历史步风格奖励为0；任务项继续正常计算。
完整任务权重见 `humanoid/envs/x1/x1_amp_config.py`；PPO loss 不反传到 D，D loss 不反传到 Actor。

## 门槛与证据边界

原始数据仍 `training_ready=false`：不声称动力学示范、接触、摩擦、力矩或真机验收通过。
此次用户授权的是独立仿真实验。新入口区分“实验准入”与“物理执行/部署放行”，
不能只改源 JSON 布尔字段开始长训练。

静态准入：哈希/血缘、只改变左髋roll、100Hz时间轴、四元数、12关节硬限位与速度、
源足底非穿地、训练模型FK重算、有限特征、不跨文件/片段/episode、无分组泄漏。
加速度/jerk/根Z和支撑代理数据记录为风险，不伪造物理通过证据。
左脚最低间隙0.412mm、支撑代理足滑增大、关节加速度376.44rad/s²、
jerk37482.41rad/s³仍存在；原源模型与训练模型几何不同。

短测：seed5、32环境、10次更新、1×4090D。额外强制一次env0终止用于检验reset边界。
必须实际验证100Hz控制、Gym关键刚体与训练URDF FK误差≤0.5mm、reset历史不串、
非零风格奖励、有效窗口、Actor和D均更新且有限、完整正常结束。

正式入口：seed5、4096环境、3000次更新，从零开始。
必须提供与当前输入、配置、特征、训练模型及实现指纹一致的云端短测证据。
实现指纹覆盖所有humanoid Python、configs JSON和X1模型资产；CRLF标准化。
小型短测证书可独立归档供云端正式入口读取，但改变训练代码/配置/模型会使其失效。
云端另检查 exact Git commit；不靠本地测试冒充短测。

## 接线

环境 `compute_reward` 中、自动reset之前捕获真实状态；runner适配层消费terminal
transition、清理历史后，仅对reset环境prime根四元数和关节位置；不用加噪Actor观测。
pre-reset前12次用训练URDF FK检查刚体是link坐标而非COM。
checkpoint保存Actor/Critic、PPO优化器、D与其优化器、冻结统计和数据身份。
`model_amp_manifest.pt` 保存有效运行配置、证书和来源，用于Gradmotion产物下载复核。

当前云端状态和task ID以任务记录为准。数据完整性、CPU测试、短测、正式训练完成、
步态质量、真机安全是不同结论。

首轮短测 `TASK_20260924_019` 在训练前的时间步校验失败。Isaac Gym原生float32的
0.001秒回读为0.0010000000474974513，乘10后的误差约4.75e-10秒，超过旧1e-10容差。
修正为1e-6相对容差并严格要求decimation=10，新增错误频率/NaN/Inf拒绝测试；
未调整任何控制、仿真或动力学参数。该失败任务保留，不作为通过证据。

重测 `TASK_20260924_021` 在2026-09-24 11:01:34正常结束，32环境×10更新。
7060个合法历史窗口，37次reset检查，620个不完整历史被排除；Actor与D均更新且有限，
真实刚体FK最大误差1.58253e-6米。已下载model_10与model_amp_manifest并检查身份、
冻结归一化、checkpoint更新次数和优化器；旧DHPPO将状态估计器MSE与PPO合并在主
优化器更新，所以legacy独立es_optimizer.state为空是源码既有行为，不是AMP断梯度。
证书 `resources/motions/amp_lafan_walk02_s092/cloud_smoke_certificate.json`
仅证明接入短测通过；不代表随机初始化的10次更新已得到合格步态。
