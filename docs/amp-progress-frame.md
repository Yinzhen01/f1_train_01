# AMP 第13/14轮：速度监督坐标系对照

## 当前状态

2026-09-26：隔离分支 `experiment/f1-amp-progress-frame` 已实现；262项CPU测试
通过，包括真实源checkpoint逐张量恢复。尚未通过本轮云端短测或正式效果验收。
域随机化及观测噪声继续关闭。第1–12轮已完成，本轮两个正式实验计作13/14，
总上限20；不把短测/回放计作额外的正式训练实验。

## 原因与唯一变量

上轮tail2000参考初始化16/16存活，但航向RMS49.65度。机身前向0.400m/s，
世界X前进仅0.265m/s、世界横向绝对速度0.281m/s；仍获机身前进奖励
+0.01427/步，而航向惩罚仅−0.001756/步。证据在上一分支
`F:/robot_f1/worktrees/f1-amp-contact-refine/docs/amp-contact-refinement.md`。

这不是声称机身坐标速度奖励普遍错误；它与本次“固定世界+X直走”的任务语义
存在缺口。AMP继续通过根坐标下39维×10帧运动窗口学习风格，不添加绝对航向、
人工步态相位或参考关节跟踪，不把任务奖励改动包装成AMP算法改进。

| 轮次 | 组 | recovery_progress速度输入 | 指令数值 |
|---|---|---|---|
| 13 | body | `base_lin_vel[:, :2]`，原机身坐标 | `[0.45, 0]` |
| 14 | world | `root_states[:, 7:9]`，世界坐标 | `[0.45, 0]` |

沿用完全相同的`centered_velocity_reward`公式与scale=2（每步再乘dt=0.01）。
只改变任务速度的坐标语义；观测维度和其中数值不变，既有Euler角包含yaw。
保留足滑−2、接触尾段−0.0025及其余全部奖励、AMP权重/结构/特征/示范数据。
两组共享实现，实际奖励函数记录body/world/selected累计值，验收实际选择与dt权重。
这些记录只用于核验，不进入观测或学习损失。

## 恢复、预算与保护

- 同源任务：`TASK_20260926_071`，`model_8802000.pt`。
- SHA256：`4804076ff5f88be3dbfffc86f7a9e3e342b255107a2f780bd3c700d0f8cb512e`。
- actor、状态估计器、D、三套优化器和50000条经验完整恢复；学习率5e-5。
- style_floor=0、bridge梯度正则=1；100Hz控制/1kHz物理/decimation10不变。
- 两组短测各32环境×10更新，2000→2010，强制核验物理失败/超时两种重置。
- 短测证书绑定本轮实现指纹及组别，不能复用上轮证书。
- 正式各4096环境×250更新，2000→2250；不从2010短测模型恢复。
- 60秒episode、32初态独立60秒评估，固定种子5/105，与071全部记录初态逐项核对。
- 只用既有合格账号4409、4090D24G（ESKU000001）、V000124，不新增账号/充值。
- 不覆盖源数据、模型或旧分支。training_ready=false，不宣称真机或sim2sim通过。

## 验收

先核对准确代码/源SHA/250条有限更新、配置和运行时动力学等价、完整模型产物，
再比较全部初态的存活率、世界前进/横向速度和航向、足滑、加速度、动作/力矩差分、
高频能量与冻结D/示范距离。不丢弃失败前缀，也不以当前D分数或训练总奖励验收。
一组只跑通或某项变好仍不解锁DR；策略必须保持有效行走和AMP风格。

## 代码与证据入口

- 配置：`configs/amp/lafan_walk02_progress_{body,world}.json`。
- 训练：`humanoid/scripts/train_f1_amp.py --experiment=progress_<group>`。
- 实现：`humanoid/amp/progress.py`、`humanoid/envs/x1/x1_amp_progress_env.py`。
- 门禁：`tools/amp/verify_progress_smoke.py`、`verify_progress_formal.py`。
- 本地私有产物：`outputs/amp-progress/`；不提交日志、checkpoint或签名下载URL。
