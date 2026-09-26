# AMP 第15/16轮：小幅方向监督对照

## 状态与动机

2026-09-26：独立分支`experiment/f1-amp-direction-refine`从已发布的`e8af600`建立。
两组已实现，276项CPU回归通过（21.462s），含真实2000源状态逐张量恢复；
真实短测086(mix15)/087(heading)均已正常完成；完整日志10条更新、产物、源配置等价、
实际奖励累计量、物理失败和超时重置均独立核验通过，证书已归档。短测代码f556b0f。
正式088(mix15)/089(heading)分别于14:20:35/14:21:09正常完成，训练代码26b4e40，
均从原2000恢复，仅追加250至2250，不使用2010。完整产物/250条更新/24项同初态已核验。
已完成正式实验16/20；mix15仅11/32存活，heading31/32但地滑仍退化，均未通过整体有效性验收。
短测不额外计数，最多再剩4轮；DR/noise继续关闭，不宣称完成总目标。
离线分析已适配真实15%奖励混合及heading系数，279项完整CPU回归通过（18.896s）。

上一轮完整world切换在源参考轨迹上导致进展收益每步下降0.022911；实际训练后31/32初态
失败，全部终止帧出现后退/负pitch。固定2–4秒已减速，估计器早期误差约0.05m/s，
临近失败才明显增大；不是已证明的唯一因果链。证据与视频见前一工作区
`F:/robot_f1/worktrees/f1-amp-progress-frame/docs/amp-progress-frame.md`。

本轮不续训world2250，不添加人工步态、参考关节跟踪、足部相位标签或额外速度观测。

## 唯一干预

| 轮次 | 组 | 进展奖励输入 | heading系数 |
|---|---|---|---:|
| 15 | mix15 | `0.85*r_body + 0.15*r_world` | -0.5，不变 |
| 16 | heading | 原机身坐标`r_body` | -1.5，原值-0.5 |

两组进展scale仍为2，再乘dt=0.01。heading保持原`1-cos(yaw)`公式。
源参考轨迹上两种干预的离线平均每步收益变化分别约-0.003437/-0.003512，
只用于控制调整幅度，不是未来收益或效果保证。body2250的同源250次续训保留为已有对照。

均保留AMP权重1、scale5、39维×10帧特征、示范数据、D结构及训练流程，
style_floor=0、bridge梯度正则=1。足滑-2、接触尾段-0.0025及其余奖励不变。
100Hz控制、1kHz物理、12关节、PD、动作/观测及全部机器人资产不变；DR/noise保持关闭。
配置门禁允许且仅允许混合系数字段及指定heading系数变化。

## 恢复与预算

- 源：`TASK_20260926_071`的`model_8802000.pt`，真实完成2000次更新。
- SHA256：`4804076ff5f88be3dbfffc86f7a9e3e342b255107a2f780bd3c700d0f8cb512e`。
- 策略、状态估计器、D、三套优化器、50000窗口经验池完整恢复，学习率5e-5。
- 每组短测32环境×10更新，核验物理失败与超时两类重置、实际奖励调用/累计量。
- 每组正式4096环境×250更新，2000→2250；不得从2010短测产物恢复。
- 相同独立种子5/105、静止16+参考16初态、60秒确定性评估；保留所有失败前缀。
- 仅用已有账号4409、项目PRO_20260924_005、4090D24G/ESKU000001、BJX00000001/V000124。
- 不新增账号/充值/购买，不覆盖旧分支/模型/数据；training_ready=false，非真机验收。

## 实现及验收证据

- 配置：`configs/amp/lafan_walk02_direction_{mix15,heading}.json`。
- 实现：`humanoid/amp/direction.py`、`humanoid/envs/x1/x1_amp_direction_env.py`。
- 真实调用诊断：body/world/selected累计量与heading原始cost、调用次数、实际dt权重。
- 门禁：`tools/amp/verify_direction_smoke.py`、`verify_direction_formal.py`。
- 测试：`tests/test_f1_amp_direction.py`，包括真实2000模型逐张量恢复、不允许额外改动。
- 日志/模型/渲染在本工作区`outputs/amp-direction/`，不提交私有产物。

先严格核对运行与恢复身份，再比较存活、前进、偏航、足滑、加速度、动作/力矩差分、
冻结D和最近示范距离及视频。AMP奖励非零或训练奖励上升不能证明效果通过。
有效性未通过前不得开启DR，也不得宣称完成总目标。

## 第15/16轮结果：短周期已足以排除退化方案

两组4096环境×250次PPO更新，分别记录24576000次真实控制转移。
策略、状态估计器、D、三套优化器及经验池的最终模型与评估副本逐张量相等；
相对于071源模型，24项已记录初始状态逐位相同，运行时PD/动力学相同。
每组完整日志都含2个已核验的SDK pika连接重置栈；属于上传通信路径，不是训练栈，
但不能称日志完全无错误。最终所需文件齐全、哈希与运行身份一致，无NaN/Inf或OOM。

| 指标 | 源tail2000 | 同源body2250对照 | mix15 / 088 | heading / 089 |
| --- | ---: | ---: | ---: | ---: |
| 静止初态存活60s | 14/16 | 15/16 | 6/16 | 16/16 |
| 参考初态存活60s | 16/16 | 16/16 | 5/16 | 15/16 |
| 参考机身vx m/s | 0.4000 | 0.3910 | 0.2301 | 0.3722 |
| 参考世界vx m/s | 0.2651 | 0.2008 | 0.2315 | 0.3608 |
| 参考航向RMS deg | 49.65 | 61.01 | 13.53 | 13.68 |
| 参考接触足底代理滑速 m/s | 0.12551 | 0.14671 | 0.10979 | 0.13983 |
| 参考关节位置二阶差分RMS rad/s² | 57.50 | 55.78 | 47.83 | 54.16 |
| 参考动作一阶差分RMS | 0.25539 | 0.28300 | 0.21784 | 0.25545 |
| 参考力矩一阶差分RMS Nm | 5.2505 | 5.5687 | 4.5578 | 5.2984 |
| 参考最近示范窗口距离 | 0.64749 | 0.65869 | 0.69128 | 0.64158 |
| 参考同一冻结D风格分数 | 0.65835 | 0.68517 | 0.60708 | 0.65932 |

除存活数外，表中为第2秒后每个初态有效前缀的指标，再对16个初态等权平均；
包含失败，不是同长度轨迹，尤其mix15的较低加速度/滑速受减速与早跌影响，不能当作改善。
接触代理为Fz>5N且脚mesh最低点<2cm；这是离线诊断，不改连续力加权的训练奖励。
冻结D不是概率，也不能单独作为动作通过标准；两组AMP风格奖励均真实参与更新。

heading相对源模型的参考航向误差下降72.4%，世界前进改善，示范距离大致持平；
但机身速度下降7.0%、滑速增加11.4%，参考env9在28.95s失败，尚不能整体验收。
相对相同250预算body对照，heading多项指标更好，说明航向监督是有价值的干预候选，
不是“只要再训练更久即可”的证据。

参考env9失败前4–2秒机身vx约0.220m/s，最后约2秒为-0.435m/s；
加速度RMS从41.89降至18.55，滑速从0.0988降至0.0219。曲线与终止帧可见停步后后倒，
低加速度/滑速与失败并存。最后窗口201帧（含终止帧）、前窗200帧，属描述性诊断，
不是唯一因果证明；未记录非脚部接触力，不断言只有某个终止原因。

### 统一证据入口

- [全部初态比较和失败编号](../outputs/amp-direction/comparison-long60/comparison.json)
- [参考初态比较图](../outputs/amp-direction/comparison-long60/reference_comparison.png)
- [静止初态比较图](../outputs/amp-direction/comparison-long60/standing_comparison.png)
- [完整250次更新曲线](../outputs/amp-direction/training-curves/training_curves.png)
- [真实奖励坐标/混合重建](../outputs/amp-direction/progress-frame/progress_frame_report.json)
- [mix15产物审计](../outputs/amp-direction/TASK_20260926_088/formal_artifact_audit.json)
- [heading产物审计](../outputs/amp-direction/TASK_20260926_089/formal_artifact_audit.json)
- [heading失败前曲线](../outputs/amp-direction/drift-089-long60/reference_failure_env9.png)

评估checkpoint SHA256：

- 088：`ff319d187e905af4a147e446c541eb139271bc3ad330ac39a2883d74de9ac15c`。
- 089：`268a0e93db4fc80592713bdba16fc05174225182f193f666e84aaed5247edb62`。

### 视频（MuJoCo显示已记录Isaac Gym状态，不是Sim2Sim）

| 对象 | 物理前缀 | 视频 |
| --- | --- | --- |
| mix15 静止env0 | 8.31s，失败 | [MP4](../outputs/amp-direction/render-088-long60/standing/policy_dual_view.mp4) |
| mix15 参考env0 | 60s，存活 | [MP4](../outputs/amp-direction/render-088-long60/reference/policy_dual_view.mp4) |
| heading 静止env0 | 60s，存活 | [MP4](../outputs/amp-direction/render-089-long60/standing/policy_dual_view.mp4) |
| heading 参考env0 | 60s，存活 | [MP4](../outputs/amp-direction/render-089-long60/reference/policy_dual_view.mp4) |
| heading 参考env9 | 28.95s，失败 | [MP4](../outputs/amp-direction/failure-089-reference-env9/reference/policy_dual_view.mp4) |

五段1280×600、50fps，ffmpeg全帧解码均通过，帧数416/3001/3001/3001/1448。
每目录另有preview_first8s.gif和首/中/末PNG，代表性画面已检查。包含最后状态，因此
编码时长比物理前缀多0.01–0.02s。最大关键刚体FK误差5.93微米；未修改机器人资产。
已有未命名geom重复告警未影响最终加载/显示与FK核验。

本轮机器起止历时分别546/575秒，包含启动、训练、评估等，不是PPO纯循环时间。
已同步账单：086/087/088/089实际deductAmount为0.09/0.09/0.81/0.90，账号可用赠金37.04；
币种及计费runtime单位未返回，不将起止秒数当作计费时长，记录在本地machineSessions。

## 后续边界

不继续mix15分支的学习状态；保留heading2250为候选，与源及body对照一起保存。
下一步优先检查停步失稳与接触滑动的实际奖励贡献，再决定剩余最多4轮的单变量方案；
仍采用先10次短测、再250次续训与60秒独立评估。此处未创建第17轮，不自动开启DR。
