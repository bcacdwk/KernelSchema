# B10 / K5 有界真实 unbound design 试验

2026-11-20。结论：**原公开 unbound row-softmax 目标合法且足以启动有意义的通用设计写作；本次真实 planner 在 120 秒截止处被终止，未产生完整草稿，未启动 reviewer，未发布 Schema。** 本批停止，不重试。它既不是完整 K5.real 通过，也不是模型或服务不可用的证据。

源码仅只读使用 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，固定并在结束时复核 HEAD `76ef9b1427e4d24174557054d049e7b134c9958d`，工作树干净。仅写本报告与 `/tmp/ks-b10-k5-design-trial-20261120-UMcDnS` 私有产物；没有改生产代码、测试、公开说明、AKA、原 K3 capture、既有失败账户或任何已准入请求，没有派生代理、提交或推送。实际 GPU/Agate jobs=0。

## 合法目标与零调用预检

[WRITING.md:19](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/WRITING.md:19) 明确允许原 unbound environment；[target-environment.md:41](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/docs/workflows/target-environment.md:41) 明确区分 explicit unbound writing 与 CUDA channel qualification。`_writing_policy.template_contract` 对原 environment 的**准确字节**生成保守的 unselected Contract，因此 B06 缺少真实设备材料不是这项限定写作的收费前阻塞条件。

这项设计仍应固定数学路线、工作归属、流水与全 Task 约束，具备实际写作意义。它不能确认设备容量、编译栈可用性、数值实测或性能；原公开环境中的 H20 只是候选，未被选定为本次目标。没有构造 H20/SM103、版本或设备资料来源，也没有将 pending Wiki locator 说成真实证据。

| 预检项 | 原事实与处理 |
| --- | --- |
| 本地安装 | Python 3.13.12；现有 `k2-cpu/bin/python` 实际导入固定 worktree。没有安装/下载。 |
| 实际 executable/profile | `ClaudeProfile.inspect` 只读 `--version/--help`，准确匹配公开 profile：CLI 2.1.316，executable SHA-256 `fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824`；profile SHA-256 `76fd98e0094f4a3e07b849c8f7a9207752689dd236105bd82919d904b405603e`。 |
| 模型与认证 | requested `qwen3.8-max[1m]`，effort `xhigh`，保留公开 accepted-reported 名称集合。只用官方 `auth status`、`existing_cli_auth`、safe-mode/user-only 设置源确认 loggedIn；未提取或复制凭证、未改全局配置。认证状态原件留私有 `/tmp`，不进入选定闭包。 |
| 已有 K3 证据 | 读取 B03、公开 profile/qualification summary；旧四次 transport 调用及旧费用保持历史身份，不复用为本次写作通过。不打开或修改历史 452 文件私有 capture。当前角色投递及 body/alias 投影需要本次自身验证。 |
| Task/规则闭包 | 原 `freeze_task` 冻结 Task/必要 evaluator/Source 依赖；原 `writing.freeze_rules` 冻结完整 DOC0–6、方法、Template 传递源码与现行写作策略。请求闭包 **93 文件**；私有 evaluator 仅保留为宿主依赖，不进入角色。 |
| 公开参考 | 原 material manifest 的准确字节/摘要/角色许可，官方 Triton v3.5.1 commit `0add68262ab0a2e33b84524346cb27cbb2787356` 的 tutorial 与 MIT license；额外显式授予原公开 `schema_template/starter.yaml` 空骨架，不授予手写 R0。KS 自有许可仍如实标为 unresolved，不据公开可读性声称再分发许可。 |
| 当前目标 | 唯一 `workflow_objective` 为本次授权下 Codex 撰写的明确 unbound 写作目标；不改 Task、四个 coverage ID、ABI、判据、workload 或原性能目标。要求完整初始设计、`next_action=none`，不产代码或伪造 REALIZATION.Feedback。 |
| 实际角色预览 | 原 `prepare_role_input` 生成 planner 354,691 bytes、reviewer 公共底稿 347,433 bytes，均低于 2 MB profile 上限；69 entries，planner 24 个正文/16 aliases/29 host implementation identities，所有正文/alias 摘要核对。没有截断语义文件或删必要依赖。reviewer 底稿没有伪造待审 Schema；真实 subject 只能由实际 planner 候选产生。 |
| 资料权限 | 无 R0/raw Schema、current Source、baseline/evaluator/inputs 私有正文进入角色；源码 locator 不授予读取权限。planner 仅能提出 Schema，reviewer 仅审阅；均无模型 tools/MCP/skills、文件、shell、网络或派生代理权限。 |
| 输出合同 | 使用原 `ks.claude.output.v1`；planner 内层必须完整 `ks.writing.draft.v1`，保留 locked Contract、准确 bindings；合法发布后才生成独立 `ks.writing.review.v1` 的全部 obligation/准确 subject/location/quote。没有换成自由文本或放松格式。 |

冻结请求 SHA-256 为 `b1a2d76676d322c5be75428f6f8ca9a716a687f46adcc6a2d3c428e2372f5050`。预检通过后已向 supervisor 报告，再调用一次公共 `kernelschema.api.design(request=..., out=...)`；没有绕过原 writing backend。

## 原单账户、实际调用与失败

本 scope 是独立且非重试的 design 资格试验，纳入 supervisor 的 K6 最多 10 次模型调用安排，本 scope 原先至多 2 次。唯一账户为：

- authority：`/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/owner.json`
- account ID：`db0c4216abfe47ca914b4e39f507e1f6`
- owner：`b10-k5-unbound-design-f179d35885eb4af2b06908222ee96a4f`
- parent/request/native workflow limits 均为 planner≤1、reviewer≤1、coder=extract=0、model_calls≤2、evaluator_jobs=0、累计 active≤600,000 ms、tokens=null。profile 保留 native timeout≤120,000 ms 和原 `--max-budget-usd 0.50`；它不是宿主美元硬上限。planner 准入预留 reviewer 的 120,000 ms 最低收尾容量。

实际 API 起点为 `2026-11-20T13:48:45.757312+00:00`，run ID `50c816956e0f453fbc01334fbf048175`，原 operation `50c816956e0f453fbc01334fbf048175_planner`。实际投递输入 **354,681 bytes**，正文/别名及 fixed_assignment 与预检相同；10-byte 差别仅为预览与正式 writing-input 的 Ref 路径名称。

| 项目 | 实际结论 |
| --- | --- |
| 新 model calls | **1**：planner=1；reviewer=0、coder=0、extract=0。未使用的 reviewer 额度不构成失败后重试授权。 |
| 原生事件 | 1 init、1 ui_invalidate、1,329 thinking_tokens 事件；没有 assistant 完整响应或 result terminal。事件数量不是账单 token 数。原流 260,405 bytes 完整保留。 |
| 实际权限/模型声明 | init 为 `qwen3.8-max[1m]`、`dontAsk`、tools=[]、mcp_servers=[]、skills=[]；仅两个预先声明的 builtin plugins，cwd=`/tmp/ks-claude-scope-70exarwm`，结束后仍空。 |
| 进程终态 | `returncode=-9`、`timed_out=true`、`spawn_error=null`、`not_executed=false`。原 supervisor/child 的 PID+start_ticks+boot_id 均核验 `alive=false`；没有原操作在途。 |
| 耗时 | native active **119,807 ms**；唯一 owner/request 累计 **123,022 ms**，未重复将 native 时间加到父 active clock。launch 原截止为 120,000 ms，启动/准备消耗同一窗口。 |
| 用量/费用 | **tokens=null；CLI reported USD=null；accounting_complete=false**。缺失终态用量不得记零；native_rounds/physical_retries 也为 null。无法从 thinking 事件推算账单、费用下界或与历史费用相加得出已知总额。 |
| 原错误 | `missing_terminal`、`nonzero_exit`、`timeout_terminated`。stderr 另保留 `[claude-code:unrecognized_model] {"model":"qwen3.8-max[1m]","query_source":"sdk"}`；不把该警告扩张成模型不可用结论。 |
| API 结果 | `status=failed`，`recovery=new_request`，`stop_reason=planner returned no complete valid writing proposal`；Schema/scope=null，structure/permissions/semantics/publication 四门均 unknown。`new_request` 是原接口投影，不授权本批重试。 |
| 账户/fence | 原 operation 已终止并吸收一次；`usage.unsettled=[]`、`owner.fences()=[]`。tokens 无有限上限，owner 没有美元硬限维度，因此未知统计不会形成受限余额 fence；**空 fence 不表示费用已知、成功或可继续花费**。本 scope 明确关闭，后续新增调用=0。 |
| GPU | **0 jobs**，没有 Agate 认证、probe、qualification、kernel、timing 或 optimize。 |

历史 K3 的 4 calls / 15,291 tokens / reported USD `0.232672750000000010` 仍只是原记录，未改写或计入本批已知增量。本次 USD 未知；本报告不对旧 AKA 账户或其费用作重新结算。

## 失败闭包与原操作恢复

因为没有完整模型 proposal，原 Template normalization、候选 precheck、Schema/change_report 发布、独立 reviewer 与 writing_review 均未到达；没有伪造可审查候选或任何通过结论。retained draft 的 content=null、ok=false，绑定真实失败收据。

在确认 writing phase=done、唯一原调用 terminated 后，用本地审计钩子禁止任何 subprocess/exec/网络连接，再执行一次 `design(continue_run=原run)`。结果与首份 API 结果完全相等，新增进程=0、新增模型 calls=0；原 owner descriptor/operations、原 native 所有文件字节不变。只进行正常原终态重放，没有删除结果投影、模拟新失败或启动 reviewer。

通过原 `Bundle.closure/import_bundle` 选择 request/result/usage、retained draft、writing-input、native receipt 及明确 raw evidence，共 **108 文件**导入新私有 `selected-bundle`。逐文件核对原字节，并从 receipt envelope 还原核对 **10 项**内嵌原始数据的 base64/SHA-256。没有递归复制 owner、认证状态或 `.runtime/claude`；其中因 Task/request 必须保留的私有 evaluator 是宿主依赖，仍不是模型可见资料。此包可读/导入，不承诺搬迁原私有运行后的继续执行。

核心审计路径：

- [零调用预检](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/preflight.json)
- [原固定请求](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/input/request.json)
- [实际 API 结果](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/run/result.json)，结果 SHA-256 `6976c24012c70db0b5ef8b5000182e34092854070f85936d3d71ba108f0542b8`
- [原失败 receipt](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/run/artifacts/dc781708b737403fbfa08adbb031e12b/claude-receipt.json)，SHA-256 `2808b8778e2b0c737b56db4aab8d592133a48333d9ecaf241ed71142441ae608`
- [原 usage index](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/run/artifacts/0a099db864824a93a1850414a8973dff/usage.json)
- [原 retained draft](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/run/artifacts/0d5547c544f748acb2d6f6a633be2ab8/draft.json)
- [原完整事件](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/run/.runtime/claude/50c816956e0f453fbc01334fbf048175_planner/events.jsonl)及[原 exit](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/run/.runtime/claude/50c816956e0f453fbc01334fbf048175_planner/exit.json)
- [终态、权限、恢复、闭包审计](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/terminal-audit.json)与[选定根清单](/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/selected-roots.json)，导入目录 `/tmp/ks-b10-k5-design-trial-20261120-UMcDnS/selected-bundle`

## 120 秒硬上限：实际产品限制与最小后续建议

120 秒**同时**是本次资格 scope 的明确限制、所选公开 profile 的值，以及当前产品代码的强制上限：

1. [ClaudeProfile 默认值](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/claude.py:96) 为 `timeout_ms=120000`；[__post_init__](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/claude.py:113) 拒绝超出 `1..120000` 的任何配置。不是单纯示例命令的默认值。
2. [runtime INTERFACE:64](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/INTERFACE.md:64) 明文写 “Timeout is at most 120 seconds”。B03 资格记录也明确该上限。公开 profile 未声称更长调用已经资格通过。
3. [invoke:732](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/claude.py:732) 再取 profile timeout、固定 request active 上限、原 owner 剩余预约的最小值；`freeze_launch_deadlines` 固定原 UTC/monotonic 截止，supervisor/child 在实际启动前再次收紧，等待/杀死也沿用原截止。父账户有 600 秒不会把单次改为 600 秒。

因此，**现行 API 只能显式调短，不能仅通过配置给真实业务 design 更长单次时间**。本次输入包含约 355 KB 完整规则/资料，xhigh 模型在原窗口内持续有 thinking 事件而未给完整输出；这暴露真实业务可用性限制，不能只登记为一般发布材料缺项。现有事实不足以断言改长一定成功，也没有得到模型草稿来诊断 YAML、数学或提示格式缺陷。权限、冻结输入和有界终止机制本次按合同生效；失败原因是原有限调用没有在截止前完成，不是 GPU/设备材料不足。

最小建议是将 **qualification scope 的 120 秒上限**与**实际业务 profile 可显式配置的有限 timeout**分开：保留默认 120 秒及本次/旧资格原值；若后续产品支持更长，需明确修订运行时/profile 合同与验证边界，仍由固定 request、同一 parent 的剩余 active/原绝对截止取紧。必须在最终 launch、supervisor、child、恢复和 minimum-review reserve 路径验证不延长原窗口；更长 profile 不自动继承旧 120 秒资格。此项是后续有界 runtime 工作，本代理没有改代码、原 profile、request 或账户限额。

后续实际调用只能在 supervisor 明确选择并冻结新业务条件、完整预算与资格后另行安排；**本 scope 的失败不会重跑，也不能借新 owner 消除这次失败或未知费用**。当前仍需保留原运行、费用未知和最后结果，继续原调用只能重读/结算原事实，不能产生新调用。完整 K5.real 的实际设计成功、独立语义审阅、反馈驱动 optimize，以及 GPU 完整任务资格均未获得。

## 本批操作审计

首次找 AGENTS 的 `rg --files` 范围过宽，曾枚举出 AKA_experiment 文件名；没有打开其文件内容、写入或运行。已立即向 supervisor 报告并将后续检索限定于固定 KS、明确报告与本次 `/tmp`。不能声称全程零目录枚举。

本次 `/tmp` 准备脚本第一次因追加 Template skeleton 后未按 ID 排序而在 Materials 校验处本地退出；尚无请求发布/准入、native intent 或模型调用。随后只修正未准入数组排序，复用同一个已创建但零操作的 owner 与同一个未发布 Publication，原材料/目标字节和认证记录不变。该零调用错误保存在 `zero-call-preparation-error.txt`。实际收费失败之后没有改请求、增限、换 owner 或再次模型调用。

没有开展新全套测试。本次检查是实际 API、原无工具模型进程、原终态核验、必要闭包逐字节核验和零调用恢复；它们不代签 GPU、optimize 或完整 K5.real。
