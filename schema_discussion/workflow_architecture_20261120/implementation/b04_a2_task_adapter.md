# B04 / A2 Task、source、Evaluation 适配

状态：**A2.interface/cpu 已实现并验证；A2.real 未验证。**
AKA 自己的 Task/Source/Evaluator/Executor/CommandEvaluator/Experiment 接口实际运行了固定 KS 完整 CPU evaluator。
没有使用另一后端的 pass 代签，没有 GPU/Agate 任务、native ABBA 或实际晋升记录。
本报告对应当前未提交 AKA 工作树；不改 KS 源码、计划、AGENTS 或 EXP。

## 固定来源与环境

- AKA：`/home/shaohanyong.shy/atrex-kernel-agent`，`feat/schema-workflows-v1`；A1 基线
  `a668f026ee30eab9cfc9832ecd1a0d110244adc6`。
- KS：仅从独立工作树的固定 commit `1a6c20171e2e02caf1c793738aeb9ab7ed2b6df6` 做 `git archive`，
  展开至 `/tmp/schema-b04-a2-ks-1a6c2017`，不消费正在修改的 runtime editable 或活动 main 脏树。
- 选取 pyproject.toml、README.md、kernelschema/、schema_template/、schema_extract/、schema_docs/、
  examples/workflows/row_softmax/、tests/workflows/fixtures/v1/row_softmax/ 和两个 evaluator 测试文件；
  共 265 个原文件。archive SHA-256：
  `48c504ce3083c37e0aa97a0b422108f7eb77b16ad3290b077d15400bb7031433`。
  逐文件摘要保留 `/tmp/schema-b04-a2-ks-provenance.json`，完成前已复核 265 个原文件均未改变。
- 隔离 `/tmp/schema-b04-a2-venv`，Python **3.13.12**；editable 指向上述 archive。
  通过环境内 `.pth` 只读复用已有 CPU 依赖目录，torch **2.10.0**、pytest **9.2.0**；未安装/下载解释器，未改系统。
  torch 的 CUDA build 字符串 13.1 不是当前设备资格；实际执行 `device=cpu, mode=cpu_logic, qualification=none`。
- 阅读并应用 AKA 自身
  [atrex-gpu-gateway SKILL 0.15](../../../../atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md)
  及其实际 reference-directory/AKA-workspace 接口；本批没有重试网络认证、寻找凭据或提交 GPU。

## 实际代码与最小原接点

AKA 新增 `src/aka/workflow/schema/{task,source,evaluation,command}.py`；
接口说明在 `docs/schema-task-adapter.md`，专属测试 `tests/test_schema_task_adapter.py`。
KS parser/evaluator/comparator 没有复制进 AKA。

仅三处旧模块的显式 Schema adapter 条件：

1. `aka.evaluation.tool_requests.request_from_argv` 惰性分派，避免在 CommandEvaluator 之前强制构造
   Atrex 的 reference.py/input.py/shapes.json 请求。
2. `LightWorkflow._campaign` 允许此显式 Task adapter 使用已注入的受信 Evaluation，而不伪造 runtime_root 目录。
3. `ManagedCampaign.setup_baseline` 仅对该 adapter 省略不存在的公开 test_kernel.py harness，保留原
   InitialSource、原 baseline source 检查、V0、never 及后续准备顺序。

没有修改 A1 的 LongHorizon loop/Store/bridge，没有做 A3 plugin/switch/子模块安装、A4 seed 事务或 A5 父账本/三 audit。
普通 Task 路径不导入 KS/Schema；非 Schema 的原 harness 要求不变。

## 稳定 Task/source API

`SchemaProblemProvider(task_ref, scope_ref, materials_ref=None, role="coder")`
实现现有 `ProblemProvider`，`load(bundle_root, snapshot_directory=private_root)` 返回现有 `LoadedProblem`：

| 原生角色 | 映射 |
| --- | --- |
| AgentProblem | 原字节 Contract/acceptance，加 Materials 中该角色允许的文件；私有 evaluator/input 同路径或同字节别名拒绝 |
| reference/public_contract | 公共 contract.yaml；不是可执行 Python reference wrapper |
| InitialSource(kind=kernel) | Task.baseline_source 的 FP32 softmax+copy；在公开/初始化前由 KS stage 校验完整 evaluator/baseline 固定字节 |
| EvaluationInput | adapter=`kernelschema-row-softmax-v1`，私有 Task/scope/完整闭包，原 snapshot codec 保留 bytes/modes/roles |
| runtime_files | 空；不经 bootstrap 给模型复制私有 evaluator |

原 `private_reference_dir` 是 legacy 私有 shapes 标记，保持 None。宿主私有 EvaluationInput 才是边界，
`content.assert_private` 检查公开材料与明显误复制；A3 仍须落实模型角色文件系统 allowlist。

`selected_seed(bundle, task, seed)`：null 只选择 Task.starter_source；显式 seed 坏 hash/缺文件立即拒绝，
不退 starter/G。该函数只读，不安装。C0 starter 或 partial 的事务继续交 A4；never 不跳 V0。

`freeze_native_source(publication, workspace, candidate_workspace, revision=None, expected=None)`：
读取真实 CandidateWorkspace 单文件源码，拒绝非 0644、BOM/非法 UTF-8、缺文件、辅助源声明/本地辅助导入、
非 single_file。`SourceBinding` 包含规范 Source Ref、原 native source_identity、kernel 原字节 SHA-256、size、mode。
solution.json 只作为独立元数据检查，不进入 C；不消费 public-candidate 整包当源码。
原 single_file native identity 是 kernel 字节 SHA-256，公共 C 是规范 Source manifest SHA-256，两者明确分列。
CRLF、无尾换行、非 ASCII 注释经过 native codec 和实际 worker compile/load 完整保留。

## 真实 Evaluation/Execution 接口

`SchemaExecution(private_ops_root, channel)` 实现原 Executor，接收原 CommandSpec，返回原 JobResult。
channel 可为：

- `LocalChannel(KS.LocalExecutor(private_jobs, python=...))`：完整 CPU 逻辑与计时；直接使用 KS 现有 prepare/submit/query/settle。
- `AgateChannel(KS.AgateDevExecutor(private_jobs, transport), profile=..., environment=..., frozen_inputs=bytes)`：
  直接适配 KS 已实现 dev API；消费外部已冻结真实 probe 捕获，不发现认证、不探测/安装远端环境、不换通道。
  它的接口代码可调用，但本批没有实际服务/设备资格。

`SchemaEvaluatorFactory(execution=..., operation=EvaluationOperation(...))` 实现原 create/create_command。
`operation` 也可为 A5 提供的可重放 `EvaluationRequest -> EvaluationOperation` 回调，支持原 Runtime 长驻；
重复 prepare/recovery 必须解析原 operation，不能分配新 key/预算。

`EvaluationOperation(operation_id, submission_key, owner, reservation, expected_source=None, mode="schema_task")`：
显式一个逻辑 evaluator_job 与正 active_ms。`execution.locator(op)` 在提交前可取得 adapter/native 根；
A5 必须在原账户保存定位和预约后使用。适配器不拥有余额。worker 预约取原/native timeout 更紧值，
准备/查询时间和累计 active clock 继续属于 A5。原未知作业不以重试返回值释放预约。

Evaluator `prepare` 支持 v0/framework/schema_task，先拒 subset，冻结真实 C 和私有 intent。
`PreparedEvaluation.inputs=()`，不把 evaluator 写入 Agent 目录。KS stage/worker 负责已固定完整组件，
candidate/baseline 从验证过的 bytes buffer 直接 compile，原 `loaded.kernel.py` 再与 C→native 映射核对。
AKA 无复制结果、写 out、allclose 或另一个计时实现。

CommandEvaluator 通过原 ToolRequest/CommandSpec/CommandEvaluation 实际支撑 native V0/full Evaluate。
Task 拥有全部用例/计时，subset、correctness_only、手改 multi-seed/timed-runs、ad-hoc command、baseline ABBA 明确拒绝。
无 `--version` 的原 native V0 用 owner 的 `operation.mode="v0"` 绑定；显式 `--version v0` 同样检查 baseline C。
返回 RESULT_JSON 仅公开状态和 coverage-shape 聚合性能，不公开私有 case/seed/guard/worker 内容。

原始 KS evaluator 执行全部 36 cases/72 basic calls、两次复用调用、FP64 golden、原 out backing/storage/view、
全写、原 x 与 guard，及全部 80 个顺序完整 forward 计时 batch；构造/首调/20 warmup/100 调 batch/原 ns 样本保留。
坏候选完整正确性失败后不伪造计时；baseline 正例获得全部计时。

CPU 正例 VerificationResult=UNAVAILABLE、all_pass=false，原 report.correctness/performance=unknown；
四关键坏候选为 FAIL。**真实受信 CUDA report 满足完整 build/correctness/performance、原主体/环境/输入/计时一致时，
V0/framework/schema_task 有可达 PASS 分支**，不硬编码永久 CPU-only，也不要求 baseline 满足 R0 fidelity。
该分支只有 Task 通过与候选 latency，incumbent/improvement 均空，不能充当 native ABBA 或晋升。
纯 projection 单元测试使用明确的假设 CUDA facts 验证该控制分支；这不是 GPU execution 证据。

## 原报告与复用

`execution.consume(prepared.context, original_JobResult)` / `read_operation(op)` 只取回原终态，不依赖当前可变 workspace。
它重新调用 KS query 验原 loaded/source/evaluator/environment/frozen inputs，核原 absorption/report Ref、真实闭包、
report subject、case IDs 与样本；原 worker 不支持的 CUDA pass 明确拒绝。
返回原 bundle、report Ref/document、私有 worker、source_binding、channel、receipt ID。

`require_applicable(evidence, task_bundle=..., task=..., source_binding=..., channel=..., environment=...,
frozen_inputs_sha256=...)` 要求当前准确 Task/C/evaluator/输入+种子/计时/实际环境/channel 配对，返回未改写的旧 report Ref。
只接收 trusted consume 的 evidence，不是任意外部 JSON 的信任入口；不重贴新 operation/hash，也不判当前 R fidelity。

## A3/A4/A5 的实际调用点

- A3：使用上述真实 ProblemProvider/两种 Evaluator 注入原 `light/runtime.py:ManagedRuntimeFactory.open`；
  还要完成资源/prompt/角色 allowlist、执行器选择、固定发行和安装。这里未冒称整个 plugin 已接完。
- A4：用 `selected_seed`、SourceBinding/原 native source、A1 prepare_episode seam 完成 baseline/C0/后续准确 C/HEAD/G；
  恢复不能用当前工作区代原已测 C。这里没有新增 seed installer/父事务。
- A5：`_run_v0_evaluator`→`measure_for_acceptance`→`SupervisorRuntime.execute_trusted`→tool request→CommandEvaluator，
  V0 与预取 reviewer 真调用均需同 owner。模式/原 source/operation/locator 应先固定。
- 实际 `RecordedABBAValidator.verify` 走独立 `execute` callback 和 same_allocation_abba record，
  `GatewayABBAValidator` 也有自己的 comparison 入口；不会因新 Evaluator 就自动获得资格。
  A5 必须在这些实际 callback/三 audit 接入完整 Task E + 当前 fidelity + 源码/版本/owner 准入，再安排原比较。
  当前 Schema command 对 ABBA 明确 unsupported；这不是永久禁止将来合格 Schema 候选晋升。

已核原 `evaluation/native_abba.py:prepare_comparison` 只构造 Atrex reference/shapes 及原 comparison result；
`drivers/native_harness.py` 是 canonical Atrex-Bench run_eval 包装器，不实现完整 storage/guard/golden64。
当前 checkout 没有可资格的本地 Atrex-Bench 源实现；`valid.py` 名称不证明 hook，legacy collector 还明确排除它。
因此本题默认 native comparator 路径未支持完整 Task；采用框架明确支持的受信 Evaluation/Executor 注入。
实际 Agate/native V0/ABBA/完整 profile 资格仍需认证、dev ACL、pinned bwrap/namespaces/CUDA、完整 frame 和 cleanup 证据。

## 验证与原件

初轮完整新 Task 测试 **13 passed / 134.73s**；补齐字节/闭包检查后 **14 passed / 136.33s**。
主回归（Task + 原 off + A1 episode）：**32 passed / 178.53s**。
随后当前 operation resolver/readback/原命令接点及完整 Task 执行的针对性复验：
**7 passed / 143.25s**（`/tmp/schema-b04-a2-verify-current/`）；当前纯 Task/preflight 检查
**10 passed / 3.46s**（`/tmp/schema-b04-a2-preflight-current/`）；最新默认 off 回归
**1 passed / 4.14s**（`/tmp/schema-b04-a2-off-current/`）。当前共有 16 个 A2 测试，另有原 A1/off 18 个。
`git diff --check` 和可选模块 compileall 通过。未扩大到无关 GPU/Wiki 测试。

主回归实际原件 `/tmp/schema-b04-a2-final/`，六个独立原 controller 操作，
去重索引 `/tmp/schema-b04-a2-evidence-index.json`；published 与 staging 复制件不是额外费用。
其中：

| 候选 | original native-terminal SHA-256 | CPU 检查 |
| --- | --- | --- |
| 共享同字节 baseline | d11335032c62853ad3c6a9778636072d5bac22356387b1d9affa90cf160d5d1a | 72 + reuse 通过，80 timing batches，CUDA unknown |
| 漏 out 末列 | b41f0e600eb5dc5cc243293b71780567c98bdb0f284b6a5c115872cd7324ab8d | out_fully_written 失败 |
| 改 x | b801f0370105ff090d2cb4659de8e14d02f1f98e0cce404f285b23cba04e85ac | x_bytes 失败 |
| 超差 | fdcd5b1b11beb686f2ce4ed24021b3216a7aabe5a084ea941db9376edac15d83 | elementwise_tolerance 失败 |
| 同 out 换 storage | 82afc8b3bc83b7d20294dfdcba8123e3cb5aaac5b2280d1e6de14699e6cac1ad | return_identity=true，out_storage=false |
| CRLF/非 ASCII/无尾换行 | e001bf3d4da13d544c4d8bf53b351b780b1cc942a2d485879e340122b6e65db1 | 实际 loaded hash 一致，72 + reuse，80 batches |

`tests/` 被上游忽略，只有精确本测试执行 `git add -N -f` 使 diff 可见，没有暂存内容或提交。
未配置/提交/推送 AKA 或 KS 工作树，没有远端操作；原 native 回归在临时候选仓库内生成的版本记录只属 CPU fixture。
正式 Python 3.11 和全新安装/模型/GPU/Agate/完整 Schema fixed 集成仍由后续单元验证。
