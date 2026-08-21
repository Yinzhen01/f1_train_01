# 项目管理方式

本项目采用“根 `AGENTS.md` 路由 + 专项文档 + 自动化验证 + 项目级 Skill”的管理方式。框架来自 `F:\project_reviewed_agent`，内容已按 X1 强化学习仓库的真实结构和工作流适配。

## 文件职责

- `AGENTS.md`：代理每次进入仓库时需要知道的稳定事实、安全边界和文档读取条件。
- `docs/project-state.md`：长期项目的当前阶段、目标、已完成事项、风险和下一步。
- `docs/git-workflow.md`：分支、worktree、提交、push 和冲突处理细则。
- `docs/x1_armature_configuration.md`：X1 armature 跨 Isaac Gym/MuJoCo 的专项说明。
- `.agents/skills/`：Gradmotion CLI 和账号申请等可重复流程。
- `tests/`：能自动验证的模型、配置和映射约束。

## 使用原则

1. 新任务开始时先查看工作区、当前分支和 `docs/project-state.md` 中与任务相关的状态。
2. 根规则只保留稳定内容；单次训练编号、短期日志和临时结论不写入 `AGENTS.md`。
3. 阶段目标、重要实验结果、关键决策或长期风险变化时更新 `docs/project-state.md`。
4. 能由测试、断言或日志验证的规则优先自动化，不只写成文字要求。
5. Gradmotion 等外部系统操作使用项目 Skill，并将任务 ID、代码 commit、配置和 checkpoint 建立对应关系。
6. 静态检查、MuJoCo、Isaac Gym、云端训练和真机验证分别报告，避免证据越级。
7. 相同代码和动力学配置下的 scratch/resume、seed、训练步数与资源差异使用训练
   profile 和 task 元数据表达；只有代码、资产或配置定义变化才新增实验分支。

## 维护边界

- 可直接修正已确认的路径、命令、索引和过时项目事实。
- 新增强制质量门禁、改变 Git/发布策略、删除规则或扩大外部系统权限前先征求用户确认。
- 原模板的占位目录、占位命令和模板仓库介绍不复制到本项目。
- 本项目原有 README 保持为训练代码的用户说明，不用模板 README 覆盖。
