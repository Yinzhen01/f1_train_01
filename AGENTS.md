# AGENTS.md

## 目的

本文件是 X1 强化学习训练仓库的代理工作入口，负责说明稳定项目事实、规则路由、验证边界和安全约束。详细状态、Git 流程和专项技术说明放在 `docs/`，不要把一次性任务流水写入本文件。

## 项目介绍

- 项目名称：`x1-training`（发布仓库为 `Yinzhen01/f1_train_01`）。
- 项目类型：Isaac Gym 人形机器人强化学习训练与 MuJoCo 推理验证。
- 当前机器人：智元 X1，主要任务为 12 DOF 双足行走。
- 核心目标：验证策略可训练性，逐步加入域随机化，并保持 Isaac Gym 训练、Isaac Gym 推理和 MuJoCo 推理中的模型与动力学参数可追溯、一致。
- 主要技术栈：Python 3.8、PyTorch 1.13、Isaac Gym Preview 4、MuJoCo、PPO。
- 主要运行环境：Windows 用于代码、配置和 MuJoCo 检查；Gradmotion GPU 环境用于 Isaac Gym 训练与云端推理。
- 证据边界：静态检查、单元测试、仿真回放、云端训练和真机结果是不同证据层级，不得相互替代。

## 仓库工作规则

- 以 Isaac Gym 训练模型及其运行时参数为本项目仿真基准；MuJoCo 对齐时必须追踪 URDF/MJCF、配置文件和训练代码中的共同参数。
- X1 模型文件位于 `resources/robots/x1/`。URDF、MJCF、mesh 和关节动力学配置应配套修改并验证名称、顺序与路径。
- X1 armature 的共享源是 `resources/robots/x1/config/joint_dynamics.json`；训练与推理按关节名称读取，不要重新引入按数组下标维护的重复参数。
- 域随机化实验必须明确区分：关闭 DR、阶段 DR、完整 DR；记录恢复 checkpoint、启用范围、训练迭代和比较基线。
- 修改奖励、指令分布、观测、动作缩放、PD 参数、延迟或动力学时，说明是否会破坏旧 checkpoint 的可比性。
- `logs/`、`outputs/`、视频和 checkpoint 是运行产物，不应随普通代码提交；需要保存时明确目标位置和用途。
- Gradmotion 凭证只能保存在被忽略的本地配置中，不得写入代码、日志、文档或提交。

## 仓库地图

- `humanoid/algo/`：PPO、策略网络和训练器。
- `humanoid/envs/base/`：通用机器人环境、域随机化与基础配置。
- `humanoid/envs/x1/`：X1 环境、奖励及 no-DR/stage-DR/full-DR 任务配置。
- `humanoid/scripts/`：训练、Isaac Gym 回放、Gradmotion 回放、导出和 MuJoCo sim2sim 脚本。
- `humanoid/joint_dynamics.py`：跨模拟器关节动力学配置加载与名称映射。
- `resources/robots/x1/`：X1 URDF、MJCF、mesh 和共享物理配置。
- `tests/`：不依赖 Isaac Gym 的本地自动化检查。
- `docs/`：项目状态、Git 流程和专项技术说明。
- `.agents/skills/`：项目内 Gradmotion CLI 与账号流程技能；调用前读取对应 `SKILL.md`。
- `logs/`、`outputs/`：本地生成产物，默认忽略。

## 常用命令

- 安装训练依赖：`pip install -e .`
- 安装 MuJoCo 推理依赖：`pip install -e ".[deploy]"`
- 训练：`python humanoid/scripts/train.py --task=x1_dh_stand --run_name=<name> --headless`
- 无 DR 混合指令训练：`python humanoid/scripts/train.py --task=x1_dh_stand_no_dr_mixed --run_name=<name> --headless`
- Stage-1 DR 训练：`python humanoid/scripts/train.py --task=x1_dh_stand_dr_stage1 --run_name=<name> --headless`
- 完整 DR 训练：`python humanoid/scripts/train.py --task=x1_dh_stand_dr_full --run_name=<name> --headless`
- 配置化训练：`python humanoid/scripts/train.py --training_profile=<profile> --run_name=<name> --headless`
- 查看训练预设：`python -m humanoid.training_profiles`
- Isaac Gym 推理：`python humanoid/scripts/play.py --task=<task> --load_run=<run> --armature_mode=nominal`
- MuJoCo 推理：`python humanoid/scripts/sim2sim.py --task=<task> --load_model=<exported-policy>`
- 本地单元测试：`python -m unittest discover -s tests -v`
- Python 语法检查：`python -m py_compile <changed-python-files>`
- 差异格式检查：`git diff --check`

## 验证策略

- 文档或纯配置修改：运行 `git diff --check`，并检查文档中的路径和命令真实存在。
- 共享动力学、URDF 或 MJCF 修改：运行 `python -m unittest discover -s tests -v`，再用 MuJoCo 实际加载模型；Isaac Gym 侧必须在可用环境中做 smoke 运行和运行时参数回读。
- 环境、奖励、观测、动作或 DR 修改：至少执行语法检查、相关单元测试和短 smoke 训练；长训练前确认日志持续增长且无 NaN/Inf、Traceback、CUDA OOM。
- checkpoint 推理：记录 checkpoint、代码 commit、任务配置和有效运行时参数。视频只能证明该回放场景中的行为，不能单独证明鲁棒性或 Sim2Real 成功。
- MuJoCo 与 Isaac Gym 对齐：比较的不只是策略权重，还包括关节顺序、默认位姿、PD、动作缩放、控制周期、armature、friction、damping、延迟和观测构造。
- 本地没有 Isaac Gym 时，明确报告“未做 Isaac Gym 运行验证”，不得以 `py_compile` 或 MuJoCo 加载通过代替。

## Git 协作职责

详细流程见 `docs/git-workflow.md`。

- 一个实验或管理变更使用独立分支/worktree，避免污染正在训练的基线分支。
- 暂存时按路径选择当前任务文件，不使用 `git add .`。
- 提交沿用仓库现有 Conventional Commits 风格，使用简洁英文摘要。
- `origin` 是上游仓库；`publish` 是用户发布仓库。push 前必须确认目标 remote 和分支。
- 不自动 pull、merge、rebase、stash、删除分支、重写历史或 force-push。
- 发现工作区脏、旧实验仍在运行或 checkpoint 与代码 commit 不匹配时，先报告再继续。

## 任务路由

- Gradmotion 账号、profile、任务、checkpoint、日志和数据：读取 `.agents/skills/gm-cli/SKILL.md`；新账号流程再读取 `.agents/skills/register-limxdynamics-account/SKILL.md`。
- 修改机器人资产或动力学：读取 `docs/x1_armature_configuration.md`，并检查 URDF/MJCF/配置/代码四个层级。
- 恢复项目上下文、判断当前阶段或规划下一步：读取 `docs/project-state.md`。
- branch、commit、push、merge、stash、冲突或 PR：读取 `docs/git-workflow.md`。
- 推理与训练异常：先复现或读取日志，再定位配置、观测、动作、奖励和物理参数，不直接根据视频猜结论。

## 知识索引

| 主题 | 读取时机 | 来源 |
| --- | --- | --- |
| 项目管理方式 | 调整代理规则、文档结构或工程记忆时 | `docs/project-management.md` |
| 项目状态 | 恢复上下文、规划下一步、判断实验阶段时 | `docs/project-state.md` |
| Git 工作流 | branch、commit、push、merge、rebase、stash、冲突或 PR 时 | `docs/git-workflow.md` |
| X1 armature | 修改或核对训练/推理 armature 时 | `docs/x1_armature_configuration.md` |
| X1 训练预设 | 从零/恢复训练、选择 no-DR/Stage-1/full-DR 时 | `docs/training-profiles.md` |
| 用户使用说明 | 安装、训练、回放、导出或添加环境时 | `README.zh_CN.md` |

不存在的需求、架构、测试、部署或安全文档不要臆造。确有长期需要时再创建，并在本表登记读取条件。

## 层级化与受控演进

1. 能用代码、测试或运行时断言表达的约束，优先自动化。
2. 只适用于某个目录的规则，放在该目录的局部 `AGENTS.md`。
3. 详细流程、状态和设计放在 `docs/`。
4. 可重复的复杂 Gradmotion 流程放在 `.agents/skills/` 或脚本中。
5. 只有全仓库通用、长期有效、每次任务都应知道的规则才进入根 `AGENTS.md`。

可以直接维护已确认的路径、命令、文档索引和项目事实。改变分支策略、发布流程、质量门禁、团队职责或删除既有规则前，应先获得用户确认。

## 演进记录

- `2026-08-21`：从 `F:\project_reviewed_agent` 同步并适配项目管理框架，建立 X1 项目规则路由、验证边界、Git 安全约束和工程状态入口。
