# B31：复用 AKA 原生实现与标准评测

2026-11-21。AKA 基础提交 `b89d061e606f8eae5415e0bdfcbd88261a5d58dd`；参照原上游 `bf5557ac04f6ff9988901d511cf25e1cbe61a1bc` 的 Light plugin，而非先前 strict Schema 控制器。未提交、推送、修改 gitlink/release，未修改 KS 活跃开发树，未访问 AKA_experiment。

## 已完成

- `--schema --schema-path FILE_OR_DIRECTORY` 走新增 standard 路径；不带开关仍走原入口。`--schema-config` 保留为显式历史 strict 路径，不作为 raw Schema 的默认入口。
- `standard_plugin.py` 调用原 `aka.workflow.light.plugin.apply`。只在 workflow 内借用 Agent 加入 Schema 上下文，并记录原生结束原因。原 Agent owner、工具、ProblemProvider、Source/Candidate、Executor、EvaluatorFactory、资源配置和 Light 多 episode 主循环保留。
- 目录读取 YAML 节点与 dispatch.md，文档通过 `--schema-doc`/`docs` 明确选择；不自动加入全语言文档。正文固定在 launch config 中，每个新会话获得同一设计，同会话 resume 不重复前缀。源码延续仍遵守原 Light incumbent/候选管理，不把失败 handoff 的候选强行提升。
- 不再要求先做 writing review/semantic/fidelity 接受、特殊 native transport 资格、只读证明、bwrap/proc 探针、K1 72+2、80 原 batch 或负例 suite。用户显式选择 production/light-sandbox 时仍使用原 AKA 自有策略。
- KS 同步入口 `aka.workflow.schema.standard.implement(...)`，B30 已对接 API/CLI/driver。参数为 schema/problem/repo_root/workspace/platform，另支持 framework、agent_cli、max_iters、docs、profile、patches、settings、runtime、prepare_only 及可选 owner_account/owner_limits/owner_run_id。
- 结果 JSON 包含 status、exit_code、stop_reason、workspace、schema_identity、evaluation_scope、完整原版本记录、native_state、artifacts、feedback_files。执行完成不推导正确性或提升成功；失败/未知值原样保留。普通异常返回失败结果和原生记录，signal/KeyboardInterrupt 仍交原恢复机制。同 workspace 相同已记录请求复用结果，改变请求拒绝覆盖。
- B30 的 driver 使用实际结果/feedback_files 原文继续写作，并用新 workspace 开后续实现。写作输出不需要严格验收门才能交 coder。

## 原生接点与评测范围

已阅读原 Light workflow/Agent/ProblemProvider/Executor/Evaluator 和 Gateway 源码、quickstart，以及 `src/aka/execution/skills/atrex-gpu-gateway/SKILL.md` 与 reference-dir/evaluation/aka-workspace/eval-manifest 参考。

标准支持范围为原 native Atrex-Bench Task：reference.py、input.py、shapes.json，位于真实 bench runtime 目录内；正常源树或显式 CUDA manifest。原 Light 正确性、性能、同 allocation ABBA 提升由原组件解释。无新 correctness 判据或新 Agate evaluator。本轮新标准结果不能倒推旧 strict Task/qualification 通过。

原 AKA `gateway_request.prepare_request(full)` 还要求每 shape 的真实性能 baseline metadata。自带 vector-add 三文件示例没有此 metadata，因此初始 Task load/freeze 与完整性能请求准备不是同一结论。没有编造 latency；已准备一次普通标准 Agate 参考实现基线测量。待真实回执后导入 metadata 再使用原 AKA 流程。

## 可选同父账户

默认未传 owner_account 时完全使用 AKA 原生预算、usage 和恢复，不导入 KS budget。

显式传入现有账户时，独立 `standard_budget.py` 复用 `OwnerAccount.reserve_group`，按 owner_limits 明确预留有限 native CLI invocation 与 GPU submission 槽。这样原模型会话中可使用正常 GPU tools。每个原生 invocation 记录原 usage；每个 distinct confirmed Job ID 计入原账户；重复查询不重复计费。只释放未使用槽，unknown 保留 fence。共享账户自动 replay 关闭，原 job query/cancel 仍可结算。parent active deadline 触发原会话/执行取消，不能把 local timeout 当远端终止。

此可选适配统计 native CLI invocation，并非每个 provider HTTP turn。因原生 token 在 invocation 后汇总，要求共享账户 token 字段不设硬 cap；原 AKA token admission 仍可使用。未新建余额，未读写本轮真实 parent 做测试。

## 验证

命令：

```text
PYTHONPATH=src:tests /tmp/schema-b18-model-budget-usage-20261121/venv/bin/python -m pytest -q tests/test_schema_standard.py tests/test_schema_off.py
```

结果：12 passed。覆盖：只改变 workflow 组成；raw Schema 准备；原 Agent 工具/请求保留；同设计两个真实 Light CPU episode；native failure/partial performance 反馈；已完成请求复用；CLI on/off；真实 composition boot；临时父账户模型内 GPU tool 调用计费和 unknown fence。CPU fixture 的模型/时间均明确合成，不是 GPU 验收。

Framework、Legacy、Light、Schema 四 distributions 已从源 sdist 构建并装入独立 `/tmp/schema-b31-standard-20261121/install-venv`。fresh `python -I` 在 `/tmp` 对官方 native Task 执行 prepare_only 并实际 boot 验证：TrackedWorkflow，原 Agent，原 LegacyExecutor，原 LegacyEvaluatorFactory，`kernelschema_imported=false`，0 模型/0 GPU。

没有重跑旧 155 项 strict suite。旧 optional test 只更新了 missing-config 提示断言。仓库 `.gitignore` 原有 `tests/` 规则，新增 `tests/test_schema_standard.py` 后续需 `git add -f`。

源文件 SHA 清单：`/tmp/schema-b31-standard-20261121/source-freeze.json`。新说明：AKA `docs/schema-standard-workflow.md`；README 与 optional-module 文档已引向标准路径。

## 给根的标准 baseline slot

官方 runtime 来源 `https://github.com/alibaba/atrex-bench`，Apache-2.0，固定 commit `1fa1f0d8a663093a22946270e7d6accb00c33698`。Git direct 超时，GitHub API source archive 成功。真实 src/atrex_bench、scripts、LICENSE 和包元数据已物化，archive SHA/origin/commit 留在 `/tmp/schema-b31-standard-20261121/provenance.json`。

Task：`/tmp/schema-b31-standard-20261121/atrex-bench/data/native-atrex-bench-vector-add`，来自 AKA 原示例，保留 small/medium/large 3 shapes。Schema：同临时根 `vector-add-schema.yaml`。原始输入 hash 位于 `baseline-plan.json`。

单 baseline 请求 `prepared-reference-baseline-request.json`：

- L20D，reference.py 本身作 candidate；普通 Agate SDK build_eval_request，serial shape_shards=1，30 bench_iters，job timeout 600s。
- request SHA256 `51724eeacc11f61faa3c33c7d8b24fb90d5b8b64030ec02af3c77e58b6283dd8`。
- stable key `schema-b31-reference-da572e069737cce07d3d700751e952b9`。
- 限定至多 1 job、0 model，未提交。
- `baseline_job.py` SHA256 `70587394e3e559b7b8ef4368b443cee89514db784039af2efb4d318fa9999ca4`。inspect 不读 auth/parent；submit 先原 parent reserve/authorize 并持久化 intent，仅一次 POST、不重试。status 只读原 job_id，每次 wait 30s；unknown 不重发。import 复用原 AKA result interpreter 检查真实 correctness/performance 后导入 metadata，明确标注 measured PyTorch reference 来源。
- 单独临时合成账户测试验证 submit→query→拒绝再次 submit，累计仅 1 evaluator job；没有接真实账户。

根收到的可审执行 argv 为 SDK Python + baseline_job.py 的 submit/status 动作，原私有认证路径仅给 SDK 进程，原 parent authority 沿用既有路径，未把内容写入源码/日志/模型。本报告与源码验证时实际费用仍为 0 模型、0 GPU；真实 slot 后续由根统一安排。
