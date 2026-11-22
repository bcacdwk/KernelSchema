# B37：简化路径的有限交付核对

2026-11-21 UTC。依据 [用户最新简化要求](../simplified_execution_20261121.md)，核对默认 design/optimize 写文件、AKA 标准 implement、安装与公开命令。没有把旧 strict 的独立 review、六 job、fidelity、特殊 transport 或陌生 GPU 验收重新作为默认门槛。

本轮固定代码为 KS `df0dfa6754c3bdb5f780d1d44e43086d5bfd4463`、AKA `621481a9b8685f4880e32a9ddce56f4e3d94cd07`。只写本报告与 `/tmp/schema-b37-simplified-delivery-check-20261121`；未改生产源码、测试、产品文档、原运行产物或账本，未安装、提交、推送，未调用模型/GPU、启动 native 服务、派生代理或查询 B36 进程。未访问禁止实验仓。

**当前结论：简化路径的工程行为与用户要求一致；现有安装配对与少量旧说明仍需最小收口。实际新 kernel 成功尚未由本报告建立。** 下述结论分开记录，不把 CPU 工程检查、原生运行完成、baseline PASS 或写作文件产生合并为同一成功状态。B36 的最终结果与根最终 source pair 尚待根提供，本报告不会提前推定。

## 1. 默认写作：可用，完成含义清楚

- `kernelschema/api.py:291`、`:301` 接受直接 `task`/`out`，可选 `references`、runtime profile；optimize 另给 `schema` 和可选 `feedback`。`cli.py:25` 暴露对应路径参数。调用者不需要手写 Ref/request、evaluator 或 performance baseline。
- `writing.py:184` 的 `write_task` 将任务与明确输入完整复制；Task 的 evaluator/baseline/starter 为 null。`:245` 固定 planner 1、reviewer 0、model 1、evaluator jobs 0。显式 profile 保持调用者选定值；省略时采用已安装 CLI/auth 与文档列出的有限调用设置。这不代表真实调用必定成功，也不是硬美元账单上限。
- `writing.py:416` 的 write-only 分支绕过旧 Task adapter/locked Contract/selection/action catalog；`:635` 发布后直接终止，没有进入 reviewer。基本可读性只要求安全 node ID、唯一 root、UTF-8 与非空 YAML mapping；Template 长度和语义/性能不挡发布。语义及权限 review 标为 `not_run`，不会冒称设计有效。
- `runtime/claude.py:425` 仅对 write-only planner 容忍缺失或不可用的外层 reason，从实际 `content.reason` 取解释；额外外层字段不充当运行事实。protocol/role/assignment/status/kind 及 content 类型仍检查。strict/coder/reviewer 保持原规则。
- `writing.py:494` 在接受判断前保存 `draft.json`、`native_responses`、可提取 raw/dispatch 与原调用证据；坏 YAML 保留 partial+诊断，身份/native 失败保持失败。`:739` 结果指向实际 `candidate_draft`、`retained_files`、`call_artifacts`、rationale 与 schema_directory。不会为了外层 reason 再把已返回的稿件变成空值。
- `raw` 节点固定保存为 YAML；独立 `dispatch` 保存为 `dispatch.md`。B33 的 Markdown dispatch 原先被放进 raw，因此 B34 的离线重解释仍是 partial；不能宿主改名后冒称原 API 成功。

`kernelschema/WRITING.md`、design/optimize README 和 standalone 主入口已清楚说明完成只代表写出文件、未做效果验收；request/continue 与显式 strict 仍可保留。少数旧概述存在冲突，见第 3 节。

## 2. 标准 implement：复用原流程，保留真实反馈

`api.implement(schema=..., problem=..., repo_root=..., workspace/out=...)` 在 `kernelschema/api.py:311` 直接调用 `aka.workflow.schema.standard.implement`。legacy request/continue 是互斥入口，不把新路径翻译成旧五项通过。

- AKA `standard.py:111` 解析原 Light composition，仅把 workflow row 换成 standard decorator。`standard_plugin.py:31` 调用原 `light.apply`，借用原 Agent 加 Schema 上下文，保留原 ProblemProvider、Source/Candidate、Executor、EvaluatorFactory、工具与 Light 主循环。默认没有 KS OwnerAccount 导入；只有显式 owner 参数启用原账薄适配。
- `standard.py:20` 读取显式 raw 文件，目录贡献 YAML 节点和 dispatch.md，额外 docs 需明确选择。`:44` 将其标为设计输入而非评测证据。B37 使用固定提交的此函数读取 B33 root，确认完整 8076 bytes/SHA `22022d1880e1336e9efb059a1bb285e619646bba84ca7da07e71fda21321c88c` 原样可消费，没有要求严格 SchemaRef。
- `bootstrap/workflow_cli.py:80` 仅在 `--schema` 为真时导入标准/legacy Schema 模块；省略或 `--no-schema` 返回原 `run_profile`，不会读取传入的失效 Schema 路径。新的 `WorkflowLaunch.exact_workspace` 默认 false；标准 API 在 `standard.py:207` 设 true，原 preparation 保留该工作目录，但仍执行原 runtime/env/lease 流程，普通 CLI 继续生成原 `kernel_opt_*` 子目录。
- `standard.py:144` collector 返回完整原版本记录、native state、candidate/kernel 路径及 episode feedback，JSON 浮点保留。顶层 `completed` 只由 native exit 0 得到，不能由它推断候选正确或被提升。现有测试明确覆盖 exit 0 + FAIL/partial 原记录，源码没有把 FAIL 改为 PASS。
- KS `examples/workflows/row_softmax/driver.py:346` 保留完整标准结果 JSON 与声明的原反馈文件，下一次 optimize 直接消费这些原文。显式启用后是有限 design→standard implement→一次 optimize→必要时一次 standard implement；没有新 comparator。未保存结果的中断 native campaign 不自动重投。此 driver 仍是固定 row-softmax 示例；任意任务使用文件 API/CLI。

B37 将固定 collector **只读**作用于 B35 已终态的真实 workspace，得到 2 个原版本与 5 个原反馈文件：v1 仍为 `long_horizon.status=invalid_handoff`、`candidate_commit=null`；其 performance 明确 `measurement_source=canonical_incumbent_carry_forward`、`measurement_subject=incumbent`、`carried_from_version=v0`。`correctness=PASS` 因此不能当作新候选通过。这次投影不是重跑、不是原 API 结果修订，更不是实际新 kernel 成功。

## 3. 最小交付缺口

| 位置与事实 | 影响 | 最小建议 |
| --- | --- | --- |
| AKA `621481a9` 的 `optional/KernelSchema` gitlink 与 `src/aka/workflow/schema/release.json` 仍是 `77ad741ca5b088d24f4cd23018eb0efcd491f15f`。`docs/schema-optional-module.md:32` 的 named submodule + editable 命令因此会装旧 KS。 | 不阻塞无需导入 KS 的 raw standard，但按该安装链不能得到 df0 的新写作/retention，尚非最终两仓配对。 | 根最终固定 KS 后更新 named gitlink 与相应 provenance；或者在 standard 安装段明确安装新的独立 KS checkout。legacy verify_release 不应被重新作为 standard 的门槛。 |
| `tools/build_distributions.py:100` 默认构建全部 OWNERS，已包括 Schema；`packages/atrex-aka-schema/pyproject.toml` 强依赖 `kernelschema==0.1.0`。通用 `docs/application-plugin-migration.md:46` 仍直接 build 全部再 `pip install *.whl`。 | 原普通/off 安装命令也进入新增 KS 依赖；该段没有先安装 KS source/editable，KS 又不承诺完整 wheel 资源。是否从包索引找到同名包不等于正确 checkout。 | 普通安装命令显式选择原 distributions；Schema opt-in 安装先装匹配 KS source/editable，再装同 revision 的 Framework/Legacy/Light/Schema。复用现有 builder 的 distribution 参数即可。 |
| KS `schema_implement/README.md:3` 将整个 implement API/CLI 仍描述为旧 reference backend，`:18` 指向已无对应标题的 `standalone.md#agate`；`docs/workflows/protocol.md:18` 仍称所有 writing backends row-softmax only。 | 与新 API/CLI 冲突，用户容易走回旧 qualifier 或误以为新 task 文件接口不可用。 | README 首段给 standard raw 路径，原 request/reference 内容标为 legacy；修复失效链接。protocol 概述区分任意文件写作与固定 row-softmax driver/strict adapter。 |
| AKA vector-add 示例 README `:12` 称 metadata optional，`:36` 随即给普通 optimize 命令。 | 三文件仅足以 Task load/freeze；原 `gateway_request.py:235` 的 full run 会调用 `validate_performance_baseline`，无测量 metadata 仍会失败。 | 明确 metadata 对加载可省，对普通 full evaluation 需实测；链接标准 baseline 步骤后再给 optimize 命令。不要放占位 latency 或把 reference 自测说成优化效果。 |
| `docs/schema-standard-workflow.md:60` 已说明真实 runtime 与 measured metadata，给了普通 `agate eval` 前置，但没有写数字 shape ID，也没有说明回执如何放入原 metadata 格式；KS standalone 只称“normal native problem”。 | 首次按单页快速入口操作仍可能重复 B32 的 named-ID 拒绝或完成 baseline 后仍缺 full 所需格式。 | 在同一短段明确 real `atrex-bench` 源码（不是空 marker）、整数可解析的 shape keys，以及将实际回执逐 shape 微秒保存到 `metadata.shapes[ID].production_performance.performance_us`，附硬件与 job 来源；KS 快速入口链接该段。不需要增加新工具/评测器。 |

可顺手在标准结果说明补一个短例：`completed`、baseline 的 PASS 或 `canonical_incumbent_carry_forward` 的 PASS 不是新候选成功；查看原 `candidate_commit`、measurement subject/source、实际候选 evaluation 与 native promotion。这里是解释已有记录，不要求额外质量门或自定义验证流程。

## 4. 实际 demo 事实与未完成范围

| 阶段 | 本报告可确认的事实 | 不能据此声称 |
| --- | --- | --- |
| B32 | 原标准 L20D reference baseline `ev_1861beac3248` 三 shape all_pass；真实 metadata SHA `17d5b6e61678e635bb168f3afa7091237709f5fbf22f7d30c7c13f4eefa9f740`。任务为 `Model(alpha).forward(x,residual)=x+alpha*residual`，bf16 连续 1D，数字 keys 0/1/2。 | 不是新 Triton kernel 或优化收益。 |
| B33 | 1 次真实 planner、0 reviewer/GPU，32,392 tokens、USD0.53898625；原 API 在旧 ad90 上因外层 reason 失败，原 root 已逐字保留。 | 原 API 失败没有变成成功，optimize 当时尚未调用。 |
| B34 | df0 窄修复与 60 个 CPU 用例；原 B33 stream 离线新投影得到 partial，90 个旧文件不变。 | 不是第二次真实写作，也不是旧调用恢复成功。 |
| B35 | 1 次真实标准 CLI 与唯一 V0 `ev_61c7e12b7ce7`；600 秒 shell snapshot helper 故障后 invalid_handoff，kernel 仍是 reference，候选 full/ABBA 均未发生。621 的三项修复由 16 个 CPU 用例覆盖。 | 没有 Triton 产物或新候选实测成功；不能以 completed+空反馈或 carry-forward PASS 代替。 |
| B36 / 最终配对 | 根监督的新标准 campaign 正在独立持有 slot；本 agent 未干预或查询其进程。 | 未收到根最终结果前不记为成功，不迁移 B35 失败记录。 |

B32 的本地官方 `atrex-bench@1fa1f0d8a663093a22946270e7d6accb00c33698` 是 Task/runtime 布局来源。其真实远端回执是 `source=job_install`、`atrex-bench==0.1.0.post20261111022558`、evaluator SHA `183b6288b85f0cd7ee3828c1a2a9db922f8971774e0122f5fceccfb58e09456e`，不把两者当成同一 Git revision。B35 回执缺 request_digest，不能移用 B32 的远端 hash 充当 B35 环境事实。

## 5. 核对证据

已阅读简化要求、B30–B35 报告、新 API/CLI/写作与标准插件，以及少量实际原 Light/Task/metadata/workspace 接点。没有重跑全库或旧 strict suite。B34 的 32+27+1 用例日志和 B35 的 16 用例记录按其原范围复用，未称作本 agent 新跑的测试。

本次额外只做静态与终态资料检查，脚本为 [check_fixed.py](/tmp/schema-b37-simplified-delivery-check-20261121/check_fixed.py)，结果为 [verification.json](/tmp/schema-b37-simplified-delivery-check-20261121/verification.json)：

- 使用 `git show <固定 SHA>:<path>` 核对 B34 manifest 的 8 文件和 B35 public-fixes manifest 的 8 文件，全部逐字匹配固定提交。
- 13 个选定 Python 源文件通过 AST 解析；未导入或启动实际 AKA/KS 运行时。
- 执行固定 standard 模块的纯文件 reader/collector，确认 B33 raw 原样消费、B35 原反馈存在且保留 carry-forward/失败语义。
- B32 实测 metadata SHA 与数字 shape 集合匹配；最终 gitlink/release 差异明确保存。

本报告的“工程可用”限定为这些默认接口和已记录验证支持的行为。根完成第 3 节最小收口并提供新真实结果后，可在本报告追加最终配对与 demo 状态；不要求重新建立旧严格验收平台。
