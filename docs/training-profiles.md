# X1 训练运行配置

本项目用一套代码覆盖 nominal 无 DR、Stage-1 DR、课程式完整 DR 和直接完整
DR 基线。动力学差异由已注册 task 表达；从零训练或恢复 checkpoint 由运行配置
表达，不再为相同动力学复制实验分支。

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
  --training_profile=stage1_from_no_dr `
  --load_run=<no-dr-run> `
  --checkpoint=4700 `
  --headless
```

## 配置边界

- `no_dr_scratch` 固定使用共享 nominal armature，关闭全部 DR 和观测噪声。
- `stage1_from_no_dr` 固定 nominal armature，只启用窄范围 Stage-1 DR。
- 两种完整 DR 运行都使用 `x1_dh_stand_dr_full`；区别仅为初始化来源和日志实验名。
- 普通 Isaac Gym 推理仍使用 `--armature_mode=nominal`，并由推理脚本关闭 DR
  和观测噪声；训练 profile 不替代推理安全约束。
- 恢复 profile 必须显式指定 `--load_run` 与非负 `--checkpoint`，避免错误谱系。
