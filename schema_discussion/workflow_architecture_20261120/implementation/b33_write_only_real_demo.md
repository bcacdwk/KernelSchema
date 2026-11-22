# B33：真实 write-only design / optimize 与标准 AKA handoff

2026-11-21 UTC。已运行并结算一次真实 design：原生 CLI 成功返回 Schema 文本，但 KS 文件 API 因外层 `reason` 字段缺失而报告 failed。已从原生输出逐字保留可读 `root.yaml` 供根决定标准 AKA handoff；原 API 失败结果不改，第二个 Markdown raw 节点保留解析诊断。当前累计 planner1/reviewer0/GPU0，USD0.53898625，optimize 尚未调用。没有做语义或效果验收。

## 固定环境与原件

- KS 固定提交 `ad90f7b9adb2f83d01302ea9fa4462390f464745`。以 `git archive` 从该提交提取必需公开源码/资源到 [/tmp/schema-b33-write-only-real-20261121/source](/tmp/schema-b33-write-only-real-20261121/source)，不安装活跃树。139 文件的逐文件 SHA 在 [source-manifest.json](/tmp/schema-b33-write-only-real-20261121/source-manifest.json)；archive SHA256 `080a029968efaa67801ea5708fd5345045ca438c6b1a12ddd5382ac47ce41eb8`。
- 复用现有 `/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python`（3.13.12）和已安装 PyYAML/jsonschema，无安装或下载。API import 与所有必需 source resources 均解析到上述快照；CLI `design --help` 可用。
- 原 Claude CLI `2.1.316` 可执行文件 SHA256 `fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824`；只读 version/help 与 `auth status --json` 返回 authenticated。未读取/打印凭证。完整的非秘密预检在 [preflight-environment.json](/tmp/schema-b33-write-only-real-20261121/preflight-environment.json)。
- 原 B21 profile `.parameters.claude` 原样复制到 [claude-profile.json](/tmp/schema-b33-write-only-real-20261121/materials/claude-profile.json)，canonical SHA256 `3e1766d7fad58da4e76b77c28a296b0adcb225c04e7e56704dac45a848efc530`。`qwen3.8-max[1m]` / xhigh / existing CLI auth / 600000ms / USD2.00 native stopping threshold。未改变 effort/timeout，没有费用 probe；美元阈值不是硬账单上限。

## 新任务与有限输入

[task.md](/tmp/schema-b33-write-only-real-20261121/materials/task.md) 为新的 1,098-byte 英文任务，目标 L20D/Triton。数学接口按原 `Model(alpha: float = 1.0).forward(x, residual)`，返回 `x + alpha * residual`；要求保持原参考的数值行为、输出 dtype 与 shape。输入为两个同长度连续的一维 CUDA bfloat16 张量。small=`N1024,alpha1.0`、medium=`N65536,alpha0.5`、large=`N1048576,alpha2.0`。没有把任务名称当成固定 x+y。

显式 references 仅为原公开 `reference.py`（371 bytes）、`input.py`（238 bytes）、`shapes.json`（347 bytes）的逐字副本。来源为 `/tmp/schema-b31-standard-20261121/atrex-bench/data/native-atrex-bench-vector-add`，原 AKA 示例数学接口；实际 bench runtime 的官方来源/commit 为 `https://github.com/alibaba/atrex-bench` / `1fa1f0d8a663093a22946270e7d6accb00c33698`，原 provenance 保留在 B31 根。写作没有使用隐藏优化实现、旧 row-softmax Task、旧私有 evaluator 或旧失败 Scope。必要 common/operation guide/starter 由新入口选择。

实际直接文件 API 的 CPU 输入投影为 **50,793 bytes**，15 entries（9 正文、6 aliases、0 host bodies）；[input-size.json](/tmp/schema-b33-write-only-real-20261121/input-size.json)、[完整 payload](/tmp/schema-b33-write-only-real-20261121/planner-preview.json)。此预览在 `api._invoke` 边界换为 prepare-only 投影，显式禁止 `ClaudeRuntime.admit/invoke`；没有创建 owner、operation 或收费调用。Task evaluator/baseline 均为 null，仅用于 write_only。结果不是一次模型运行。

## 调用与预算

真实命令和等价 CLI 在 [commands.md](/tmp/schema-b33-write-only-real-20261121/commands.md)，[run_writing.py](/tmp/schema-b33-write-only-real-20261121/run_writing.py) 只调用公开 `api.design/optimize`，无自动 retry 或 implementation。

计划仅两次写作：一次 design；拿到实际 AKA implementation 后一次 optimize。每次直接文件 API 自建独立有限单写作 owner：planner1/reviewer0/coder0/extract0/model1/GPU0/active610000ms/tokens=null，加上原 profile 600s/USD2.00 原生阈值。根已确认按新标准任务分别记账；旧 B14 费用/剩余额度保留为历史，不转移、不重置，不为复用 parent 退回手工 Ref 包装。

输出根分别为 `/tmp/schema-b33-write-only-real-20261121/design` 与 `/tmp/schema-b33-write-only-real-20261121/optimize`。实际可消费 Schema 目录取原结果 `extensions.information.schema_directory`；会连同 exact raw、rationale、candidate draft、diagnostics、native receipt 原样保留。design 后将这个目录交 B32 的固定 AKA `9dfe8c66b7ccca8081e4a781896fe86c5d8af9ac` 标准 L20D 实现。optimize 只使用届时实际返回的 result/declared feedback files，不预造反馈。

本报告会在每次真实写作完成后追加实际产物和用量。默认不修改源码，未提交/推送，未派生代理，未访问禁止实验仓。

## 原预检后的标准 Task 标签更新与 design 准入

B32 的真实标准 Agate runtime 要求数字 shape ID；根确认将尚未准入的写作输入更新为其实际标准 Task：`/tmp/schema-b32-standard-real-20261121/numeric-shapes/atrex-bench/data/native-atrex-bench-vector-add`。仅 case labels 改为 `0/1/2`，`reference.py`/`input.py` SHA 与原件完全相同，numel/alpha/dtype/数学行为不变。原始命名 case 的材料保留在 [original-named-shapes-input](/tmp/schema-b33-write-only-real-20261121/original-named-shapes-input)，原50,793-byte预览保留。

实际调用前的新 Task 为1,176 bytes，shape JSON334 bytes（SHA256 `314314866671d5b1ac7655178a23ec23a0d6d533afb0c8364b2f134529145d34`）；[numeric-input-manifest.json](/tmp/schema-b33-write-only-real-20261121/numeric-input-manifest.json)。更新后的 CPU payload **50,936 bytes**（同9正文/6aliases），见 [input-size-numeric.json](/tmp/schema-b33-write-only-real-20261121/input-size-numeric.json)。B32 reference baseline 已由原 job `ev_1861beac3248` 实测；其时延没有作为本次 design 效果要求或反馈注入。

根移交唯一模型 slot 后启动公开 API `api.design` 一次，run ID `963b199b6b5a46b5a940a7ea7fda32d0`，输出根 `/tmp/schema-b33-write-only-real-20261121/design`；全程使用原600s/USD2.00/xhigh profile和有限planner1/reviewer0/GPU0新owner。optimize尚未调用。


## 真实 design 结果与保留的 Schema

原 API [result.json](/tmp/schema-b33-write-only-real-20261121/design/result.json) 为 `failed`，`schema=null`，原因 `planner returned no complete valid writing proposal`。对应原生 CLI terminal 为 `success` / `is_error=false` / `end_turn`，并返回完整、可解析的 JSON。原生回执唯一错误为 `invalid_role_output`。该差异保留，不因 native success 或进程 exit0 宣称文件 API 完成。

小范围原因明确：外层 JSON 有 protocol、role、assignment_sha256、status、kind、content，缺独立 `reason`；实际解释已位于 `content.reason`。`RoleOutput.parse` 在 `runtime/claude.py:426` 严格要求七字段，故拒绝；`WritingBackend._draft` 只在 receipt.ok 时读取 content，导致 candidate_draft 的 content=null、retained_files=[]。此处尚未到 write_only 文件解析，与 Template、语义或性能门无关。

按保留已有产物的任务要求，在原 run 外 [retained-native-draft](/tmp/schema-b33-write-only-real-20261121/retained-native-draft) 保存原 terminal result 文本、解析后的 envelope、逐字 raw 和原 content.reason。没有改动任何原 result、receipt、raw 文本或源码，也没有补 reason 后重放、补跑模型：

| 原节点 | 原样保存路径 | bytes / SHA256 | 基本可读性 |
| --- | --- | --- | --- |
| root | [root.yaml](/tmp/schema-b33-write-only-real-20261121/retained-native-draft/raw/root.yaml) | 8076 / `22022d1880e1336e9efb059a1bb285e619646bba84ca7da07e71fda21321c88c` | 非空 YAML mapping |
| dispatch | [dispatch.yaml](/tmp/schema-b33-write-only-real-20261121/retained-native-draft/raw/dispatch.yaml) | 748 / `9452d314ed8554bc8dc7c085f166096143cea3ad5d5c52bd49f750bae8f5e000` | 模型将 Markdown 放进 raw，YAML 解析失败；文本保留 |

[原 rationale](/tmp/schema-b33-write-only-real-20261121/retained-native-draft/rationale.md) 单独保存。可给标准 AKA 的输入是上表 **root.yaml 单文件**；不要把包含误标 Markdown 节点的整个 raw 目录冒称全部可读。该手工导出仅提取已付费调用真实返回的文件，不是 API 发布成功或新 SchemaRef。root 内容尚未经过语义、实现或效果验收；这些不是本次写作完成前的附加门。根与 B32 已收到这一文件路径和边界，后续实现 slot 仍由根分配。

建议的窄修复交给根：仅在显式 write_only planner 的返回/保留边界，保留匹配 role/assignment 的可解析 content 与原生 envelope；缺非关键外层说明字段可以作为诊断，或用已返回 content.reason 做窄兼容。不得全局放宽 coder/reviewer/strict 身份约束。无论返回是否合格，已有 raw 文本应在失败/partial 记录中可见。当前第二个节点的坏 YAML 本应按 write_only 约定保留 partial，而非因为外层字段问题连可读 root 也只留在 opaque native event 中。B33 未实施源码变更或测试替身重放。

## 本轮实际新账户用量

[design-summary.json](/tmp/schema-b33-write-only-real-20261121/design-summary.json) 汇总实际文件与回执路径；[原 native receipt](/tmp/schema-b33-write-only-real-20261121/design/artifacts/aa93bf4a395544d09d1adf62835e3f31/claude-receipt.json) 和原 events 完整保留。

- host model invocation=1，native rounds=1，planner=1，reviewer=0，GPU jobs=0。
- tokens=32,392：input6 + cache_creation14,437 + cache_read0 + output17,949，原生费用 **USD0.53898625**。
- native duration408,113ms；runtime计费active410,595ms；KS request累计active413,181ms。三者各自口径保留，不混算。
- 原 `qwen3.8-max[1m]` / xhigh / 600s / USD2.00 profile 完全不变。原生 `physical_retries=null`，不推断 provider 内部重试数量；没有第二次 host 调用。
- accounting_complete=true，terminated=true，unsettled=[]。唯一模型 slot 已结算并移交根。旧 B14 账户未读取/修改，旧费用保持历史独立。

optimize仍为0调用，等待标准实现的实际结果/反馈与根后续移交。
