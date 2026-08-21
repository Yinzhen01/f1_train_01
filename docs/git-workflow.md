# Git 工作流

本文档定义 X1 训练仓库的 Git 协作细则。执行 branch、worktree、commit、push、merge、rebase、stash、冲突处理或 PR 准备时读取。

## Remote 职责

- `origin`：上游 `yanni-00/x1-training`，用于查看上游变化；不要默认向其推送。
- `publish`：用户仓库 `Yinzhen01/f1_train_01`，实验分支和项目成果的主要发布目标。
- push 前再次执行 `git remote -v`，确认目标 remote、账号和分支，不在文档或输出中展示凭证。

## 默认检查

- 编辑、创建/切换分支、提交或 push 前运行 `git status --short --branch`。
- 区分当前任务相关改动、运行产物和用户已有改动。
- 只按路径暂存当前任务文件，不使用 `git add .`。
- 提交前检查 `git diff --cached --check`、`git diff --cached --stat` 和必要测试。
- checkpoint、云端任务和代码必须记录对应 commit；不得假定工作区当前代码就是训练时版本。

## 分支与 worktree

- 一个实验或一类不可拆分的工程变更对应一个分支。
- 使用小写 kebab-case 和现有前缀：`experiment/`、`feat/`、`fix/`、`docs/`、`refactor/`、`chore/`。
- 训练任务已经从某个 commit 启动后，不在原分支上混入会改变实验定义的改动；新方案另开分支。
- 主工作区存在用户改动或运行产物时，优先使用独立 worktree 隔离，不移动或删除用户文件。
- 创建、切换、删除分支/worktree 前确认用户意图；删除前必须确认分支已保存且目录目标精确。

## 提交组织

- 沿用仓库的 Conventional Commits 风格，使用简洁英文摘要，例如：
  - `feat: add stage-one dynamics randomization`
  - `fix: align MuJoCo joint armature`
  - `docs: add experiment tracking workflow`
  - `refactor: share X1 dynamics config`
- 一个提交表达一个清晰意图。行为、机械格式化和无关文档不要混在一起。
- 不提交 API key、签名 URL、密码、私钥、下载临时链接、大型 checkpoint、日志或渲染产物。
- 用户未要求 commit 时，完成修改和验证后报告未提交状态，由用户决定是否提交。

## 同步与历史操作

- `fetch` 是只读刷新，可用于比较；不要把它描述为已同步工作区。
- 不自动 pull、merge、rebase、squash、stash 或恢复 stash。
- 默认禁止 `git reset --hard`、`git clean -fd`、破坏性 checkout、删除分支和 force-push。
- 如确需 force-push，必须由用户明确批准，并优先使用 `--force-with-lease`。

## 冲突处理

- 不机械选择 ours/theirs；先说明冲突文件、实验含义和 checkpoint/配置兼容性。
- 奖励、指令、观测、动作、资产和 DR 配置冲突属于实验定义冲突，需要保留明确选择依据。
- 解决后运行受影响模块的验证，并更新 `docs/project-state.md` 中长期有效的决策或风险。

## Push 与交付

- push 前确认当前分支、提交、工作区状态、remote 和待推送 commit 范围。
- 新分支发布到用户仓库时使用：`git push -u publish <branch-name>`。
- 未经用户明确要求不自动 push。
- 交付时报告：分支、commit、验证、未推送状态、已知限制，以及对应 Gradmotion task/checkpoint（如果有）。
