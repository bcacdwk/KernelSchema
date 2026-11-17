# R2 B：提示、材料、自检与入口

本轮已完成并停止写入。没有模型调用、GPU、正式提取、旧 run 转换或个人配置修改，没有 Git commit。正常六阶段的实际材料与提示已由 R2 A 接通；恢复、维护、发布的收口留 R3，完整手册与有限真实 CLI 验证留 R4。

## 已实现

- 唯一正式入口为 `schema_extract/extract_yaml.py`，删除旧文件；冻结启动、后台启动、clock 登记与 standalone 的实际入口引用同步。`snapshots` 删除旧包布局探测，冻结完整 `prompt/**/*.md` 和当前 Python 执行器。
- 固定指令集中在 `prompt/common.md`、`roles/{worker,reviewer}.md`、六个 `phases/*.md` 与 `operations/*.md`。operations 覆盖普通两角色、提案/证据、自检、返回、裁决、独立签收、当前 run 维护、恢复、回执修复、重试、review_eval、monitor 工程交接、wakeup 与 probe。源码中的机械 schema、枚举和普通诊断保留。
- 阶段页使用当前字段与整区 `from`，明确 P2 `{id, instance}` 与 P4 `{id, wait_for}`。取消 P1→P2 共会话说明；每阶段、角色及 P5 节点按本次材料建立上下文。P6 先读最终成果，再读过程记录核对知识去向。
- common 简短说明设计重点、必要补充和清楚命名；具体字数、大小写、标点与排版交 Template。worker 自查遗漏/冗余，reviewer 独立判断。框架问题仍复用 `framework_candidates`，保留原候选及证据；CLI 汇总以 reviewer 提交同 ID 的候选作为 `confirmed`，其他保留为 `candidate`，没有新增台账。
- 默认新阶段调用由 A 的配置提供 Claude / `qwen3.8-max[1m]` / max。CLI 传 `backend / agent_binary / model / worker_effort / reviewer_effort`，不传未指定的可选值。Codex 必须显式给模型与两角色 effort；没有后端 fallback。Supervisor 的 `--codex-bin` 仅属于 Codex queue 提醒。
- review_eval 改用 `AgentCLI` 和当前字段，补齐 nonce、独立 scope/session、只读挂载、请求/报告模型与 effort、native status、usage 和日志。入口为 `extract_yaml.py review-eval --suite ... --output ...`；suite 必须明确指定，没有旧 archive fixture 默认值或字段转换。
- CLI 删除 `import-legacy`、`continue`、`revise`、`compatibility-report`、`upgrade-runtime` 的注册、分支与对应 import。普通阶段的 `decision=continue` 保留。跨版本模块的物理删除由 R3 完成。

## 对外接口

```python
prepare_attempt_materials(state, lock, attempt, *, rules_lock=None, executor_lock=None) -> dict
build_prompt(state, lock, attempt, *, rules_lock=None, executor_lock=None) -> str
render_operation(name, context=None, *, rules_lock=None) -> str
```

`Store.prepare_attempt` 在写 attempt.json 前合入第一项的返回值，包含 `materials` 与 `isolation`。渲染只读取冻结 Markdown，并在操作文字后附 JSON；没有占位符模板引擎。`build_prompt` 生成本次完整 stdin，不依赖首轮系统提示或其他阶段对话。`provided_documents` 只列本次实际嵌入全文，`read_specs` 去重；`stages.read_specs` 已直接使用 `prompt/roles/reviewer.md`。

| 隔离位置 | 内容与来源 |
| --- | --- |
| `/workspace` | 本次完整受控材料；worker 仅 Runner 预建的授权直属文件可写，reviewer 全只读 |
| `/source` | 原冻结 source，路径、文件 SHA 与 repository/ref 身份记录在材料清单 |
| `/rules` | 本次提示及其规范引用、Template 检查需要的文件，逐文件只读映射 |
| `/executor` | reading/artifacts/common/inputs 及包入口，供固定检查 helper 导入 |
| `/materials` | 本次复制的 baseline、必要回执、handoff、裁决候选和阶段检查参数；只读且记录原宿主来源及 SHA |

不会挂整个 run、state-root 或 session 状态。`materials/manifest.json` 与 attempt 中的材料描述保留来源映射。回执按 `state.attempts` 的真实顺序选取，避免把 JSON 字典的 UUID 排序当作时间顺序。P6 带当前所选节点及 generation 的 P2/P4/P5 accepted receipts，包含 worker 的 complete/continue 记录；不是只有 reviewer 短交接。A 已将 P5 handoff 绑定本次实际 checkpoint。

模型用原生 Read/Glob/Grep 读材料。源码引用仍是可发布的 `repository_id@ref:relative_path::symbol`；`/source` 只是本次读路径。提案中的相对路径与行范围仍由 Runner `read_evidence` 在宿主冻结源码核实，不把逻辑路径或预览位置写成最终证据。

AgentCLI 已将实际 stdin 保存到 `attempt/prompt.txt`，登记摘要并在隔离内只读映射；规则快照另外完整保存 prompt 树。

## 只读 helper

`isolation.check_command` 指向冻结 `reading.py --check-materials /materials/check.json --read-only --skip-hardware`；模型只调用 `/run/extract/check`，包装器拒绝额外参数。

- reviewer 直接检查当前 raw；worker 调用 R1 的 `normalize_candidate` 在专用临时目录预览，再运行阶段检查。当前 workspace 的字节不改；权限和关联修正越界时整组拒绝。
- 输出区分 `raw_passed / preview_passed / pending_corrections / corrections`；有待落地修正返回 1，不宣称当前 raw 已通过。
- P1 不要求未来 Design；P4 只要求完整 root 与合法目录/计划，不要求未来 YAML 存在。P2 worker 的选择尚未提交，复用 `_early_check(P2)` 核已知字段并标 `selection=awaiting_proposal`，最终精确集合由 Runner 用 proposal.plan 检查。P2 reviewer 使用本次 `plan_draft`。
- helper 先核 `/source` 的完整冻结 manifest，再复用 `validate_artifact_sources` 核同一 repository/ref 与具体文件/符号。Template 的本地仓库检查仍按其 API 执行；不会用隔离内无宿主 Git 的结果代替 Extract 的冻结来源核验。

`reading.stage_check(record, *, workspace=None, rules=None, baseline=None, source=None)` 的可选路径用于宿主离线验证；正式固定 helper 只用已登记的隔离路径。

## 验证

- `test_prompt_materials`：4 项通过。覆盖真实 namespace 内运行阶段 helper、冻结源码身份、宿主 state 不可见、额外参数拒绝、reviewer raw 只读、P1 规范化预览、P2/P4 未来文件与字段权限、P6 worker 过程记录和实际嵌入材料。回执时间顺序及 generation 过滤修正后，P6 用例另行复验通过。
- `test_review_eval`：3 项通过。使用当前 Template 小样本，检查两后端的注册身份、独立只读 scope、nonce/四项判断/字段定位和 Codex 必须显式配置；模型响应使用离线替身。
- 临时新源通过新脚本 `extract --queue`：实际冻结并核对 Claude/Qwen/max 默认配置，再运行其冻结 `extract_yaml.py --help`；没有启动模型。
- 全部固定操作渲染、prompt 内直接本地链接、全 prompt 冻结清单、新入口 help、相关文件编译与 `git diff --check` 通过。
- R2 A 已用真实 `prepare_attempt_materials/build_prompt`、冻结来源、Template 和 receipt 跑通单 root 两后端及多来源正常链，见 [R2_A](R2_A.md)。本轮没有把替身通过表述为真实模型语义审阅。

## 交 R3 / R4

1. R3 清理 runtime/recovery/maintenance/publication/standalone 等尚存的跨版本、旧来源和恢复调用方，再物理删除失用模块。CLI 已不暴露这些迁移命令；本轮没有为旧测试保留兼容入口。
2. monitor/wakeup 本轮只改固定指令装配与入口路径；未改变队列、调度、恢复、预算或 fence 语义，也未实际发送 queue 消息。recovery 的模型交接指令由 A 的 Runner 接到 `render_operation('recovery', ...)`，其原后端恢复协议仍需 R3 收口。
3. 活动 Extract README/REVIEW/tools README/issues README 和 EXTRACT_SCHEMA 仅同步入口/阶段路径与会话特例。长手册中的迁移、旧字段与旧能力描述由 R4 统一；`schema_template/README.md`、`schema_docs/2.contract.hardware_specification.md` 各仍有一处旧 phase 链接，本轮未改 Template/schema_docs。archive、wiki 不批量替换。
4. 旧 `test_review_protocol / test_design_derivation / test_semantic_revision / test_cli_recovery` 等仍含旧状态、旧入口或旧字段假设，需后轮按最终当前协议删改，不能以兼容壳通过。
5. R4 用完整动态提案 schema 验真实 Claude 工具读写、精确 resume、独立 reviewer，以及 Codex 在无新 procfs 的实际工具能力；本轮的真实隔离 helper 证据只证明固定 helper 和相应材料映射，不证明两家 CLI 的完整工具调用。
