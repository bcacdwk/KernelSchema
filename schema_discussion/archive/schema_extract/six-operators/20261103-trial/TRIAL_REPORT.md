# 六类算子提取试用记录

> 归档说明：2026-11-03 至 11-04 的试用与质量初验记录，正文计数和判断保留当时含义。 [归档索引](../../README.md)。

2026-11-03，六项固定实现均通过 P1–P6 独立审阅并发布，共 **77 个节点 YAML、6 份 dispatch 索引和 6 份发布回执**。原始交付使用冻结的 Schema Extract **2.2.2**；全部运行已结算，交付文件与回执摘要一致，六份完整证据审计均为 `verified`，没有核验错误或缺失证据。

分组 GEMM 现有[2.2.4 校正版](../../../../schema_wiki/archive/six-operators/revisions/2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md)，已单独终审、审计和导出；GDN 后续的 [2.2.7 六节点修订版](../../../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md)也已终审、正式 export 并验收，当前推荐版本合计 **81 个节点**。日常阅读使用修订版，下表保留初次交付，便于追溯原始审阅与修订。

2026-11-04 的[算子结果初验](OPERATOR_REVIEW.md)确认主要分层基本成立，同时登记条件流程与角色成员范围的规范遵循问题；四类算子保留初验意见，先核对现有合法写法再判断规范或工具是否需要修订。该质量复核与本页的原始运行完成、证据审计结论分别记录。

后续 2.2.5 已修复 Reviewer 文档验收机制，并增加正式的单文件阅读视图；42 项提取测试、148 项模板测试及 10 个真实语义样例通过。生产提示下对原 GDN 的隔离终审要求重开 P3，详情见[验证记录](../20261104-review-cases/VALIDATION.md)。本次没有重新提取或覆写原始六项交付。

| 原始交付算子 | 节点数（含 root） | 记账调用数 | 运行与发布回执 |
| --- | ---: | ---: | --- |
| [fp8_scaled_grouped_gemm](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | 3 | 23 | [4aac1ea4-94fa-4b4c-93ca-d48a8eb08996](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/.extraction/receipt.json) |
| [nvfp4_fused_moe](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md) | 60 | 188 | [3563c3b7-2b7c-4967-9d41-9c2042e53d2c](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/.extraction/receipt.json) |
| [fmha_mxfp8_qkv](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md) | 8 | 32 | [acb68909-aaaa-4bc2-8b16-94904752af8e](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/.extraction/receipt.json) |
| [mla_expanded_fp8](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | 1 | 14 | [e3f7704f-061f-48ff-8131-cc317afe6003](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/receipt.json) |
| [dual_cache_latent_bf16](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) | 3 | 22 | [d5a73c01-7c3a-45b2-a3ce-0d5eec634808](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/.extraction/receipt.json) |
| [gdn_prefill_bf16_state_bf16](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | 2 | 22 | [97902388-eda0-4cbd-b8a6-ba71440e86ab](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/receipt.json) |

调用数合计 301，按六条互不重叠的 lineage 计数，包含接续前的尝试；不再次叠加前任总数。4 次调用的提供者 usage 未知，其余保留原值；缓存输入已包含于 input，且续会话计数范围仍未确认，因此不汇总 token 或据此估算费用。

## 来源与验证范围

来源为 `kernel_wiki/aka-cuda-operators-20261030` 中选定的固定实现，输入锁和发布回执保留源码版本、逐文件 SHA-256、规则身份及 P6 批准身份。检查显式使用 `--skip-hardware`，硬件保持模板占位；交付忠实保留源码限制与外部比较器缺口，不宣称重新完成 GPU 验收或复现全部外部比较器。

完整本地证据位于 `.schema_extract/v2/`，六份审计报告位于 `.schema_extract/trials/six-families/audit/*.completed.json`，批次核验表为 `.schema_extract/trials/six-families/batch-acceptance.json`。Supervisor 接续和决策记录位于同目录的 `SUPERVISOR.md`、`checks/`、`recovery/` 与 `post-batch/`。

## 已解决的问题

- [外部比较器缺项阻塞 P1](../../../issues/archive/extract_workflow/p1-external-comparator-gate.json)：明确来源缺口与核心计算事实的边界，保留已知判据后继续忠实提取。
- [P4 索引布局硬停止](../../../issues/archive/extract_tools/p4-index-layout-hard-stop.json)：2.2.3 曾用有限格式修复处理；2.2.6 将计划内容与段落排版解耦，删除特殊错误及修复分支。
- [并发分支的错误调用定位](../../../issues/archive/extract_tools/p5-handoff-attempt-locator.json)：2.2.4 绑定真实失败 attempt，旧记录不猜测兄弟调用。
- [语义 summary 与文件哈希混称](../../../issues/archive/extract_workflow/proposal-summary-ownership.json)：2.2.4 明确模型填写非空语义总结，工具生成文件 SHA-256。
- [P6 树结构的产物证据](../../../issues/archive/extract_workflow/p6-tree-artifact-evidence.json)：2.2.4 允许在完整引用和实际树检查支持下提交产物树结论，源码语义结论仍需来源依据。

## 2.2.4 的真实验证

真实验证发现并修正了一处转录范围：原始分组 GEMM 的 `root.yaml` 与 `umma_wide.yaml` 将 graph API 失败概括为关闭 graph 的原因；源码只检查构建及 launch 返回值，销毁调用的返回值被忽略。校正版的 root 和 wide 已明确区分这两类调用，全部三个节点通过独立审阅、P6 终审和完整证据审计，并通过正式 `export` 发布到独立版本目录。原始交付及其回执保留，依据和修订记录归档于 `.schema_extract/trials/six-families/post-batch/archive/grouped-gemm-graph-error-scope.json`。

2.2.4 的 29 项提取回归已通过。MoE 第一次因 `tree.evidence=[]` 被拒的原始 P6 提案，在新门禁下完成只读回放并通过实际 60 节点树检查；原运行状态、原始回执和既有发布未被改写。

新版本的真实多节点验证已完成，使用相同源码字节和绑定的 `fp8_scaled_grouped_gemm`，run 为 `b61c62b1-b0dd-4ed8-a21f-e9b7227a0d3b`。共 15 次调用、0 次格式修复、0 次 CLI 失败、1 次正常语义退修，约 86 分钟完成；全部 15 次调用的证据审计通过。P6 的 `tree.evidence=[]` 实际使用完整产物证据和树检查通过放行。

原 2.2.2 接续运行有 21 次调用，其中 7 次空 `summary` 修复；本次为 0 次。这是一对独立模型运行的样本观察，不用于推算整体成功率、性能或费用。完整结果位于 `.schema_extract/trials/six-families/post-batch/validation-2.2.4.json` 及同目录审计文件；首次试用发布与随后版本导出的时间分别记录。

## 2.2.6 的 issue 复核

按长期可用性、维护成本和必要复杂度复核全部 16 项，逐项结论见[分类与处理方案](../../../issues/archive/TRIAGE.md)。Supervisor 确定方案与验收，subagent 完成实现和必要测试。

P4 直接接受正确表格后的说明段与覆盖标题空行，仍检查唯一真实覆盖段、计划值、父关系和链接；47 项提取回归及两份原始失败稿只读重放通过，原稿哈希未变。成员端点沿用现有整数表达式，字段、关联检查和有限图共用解析；152 项模板回归通过。[成员问题](../../../issues/archive/schema_template/symbolic-member-bounds.json)按部分采纳归档，旧 MoE 的最大范围与正文取消规则仍待独立修订。

另复核发现原 MLA 的冻结输入实际没有 catalog 已登记的校注，原归档关于已收集的判断不成立。[校注来源问题](../../../issues/archive/kernel_wiki/mla-fp8-source-comment-drift.json)现由既有输入入口收集可选文件，保持原角色绑定；当前真实输入只读收集共 31 文件并包含校注，现有 `annotation_program` 另核验当前原件/校注一致。该独立核验不表示 extract 增加自动等价证明，旧快照不变。

条件流程按既有 dispatch 区分实际提交和交接关系；GDN 与 sparse probe 的具体划分已写入方案。此轮不重新运行六项提取，也不把规则与工具修复视为旧算子获得新的批准。历史索引阶段文字及上游 kernel 覆盖缺口继续保留真实状态。

## 2.2.7 的 GDN 交付

2026-11-04，GDN 在冻结 2.2.7 下完成 6/6 节点和 P6；[六节点索引](../../../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md)与[交付回执](../../../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/receipt.json)已通过正式 export 发布，Supervisor 已验收。源码与输入角色绑定不变，实际任务固定 `norm=false` 并提供 packed 元数据；缓存/直接 QK、分段及 Perm 存在性分别由完整节点表达。修订细节见[算子复核记录](OPERATOR_REVIEW.md)。

| 身份 | 记录 |
| --- | --- |
| Run | `45f9bb51-77f3-4285-b7a7-251cf2744fa1` |
| 最终 checkpoint | `349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d` |
| 最终 P6 reviewer | `1283b632-a802-4625-b666-a0eb3dce449d` |
| P6 receipt SHA-256 | `ed4f3e2dc665dd7bf3ee1dd23e16e6d85170ad40ea52a259d87f2ecdb4280cf0` |
| Export ID | `4cae56f4243c563903c7102fbb2361c3d7abcd1014b3707be9c486e3aa251a48` |

正式 `audit --attempt 1283b632-a802-4625-b666-a0eb3dce449d --verify` 的现存[报告](../../../../.schema_extract/trials/six-families/post-batch/artifact-repairs-2.2.7/gdn-quota-restored-final-audit-20261104T125856480549Z.json)仅覆盖最终这一次 P6 调用：`verified`，`errors=[]`、`unavailable=[]`。[恢复与交付记录](../../../../.schema_extract/trials/six-families/post-batch/artifact-repairs-2.2.7/quota-restored-recovery.jsonl)在 `2026-11-04T12:59:43.792755+00:00` 的 `gdn_export_delivery_verified` 事件确认 6 个 YAML、dispatch 及交付回执共 8 文件与目标一致，原目录 4 文件 SHA 不变。

Run 为 `ready/P6`；原目标已有历史产物，故顶层 `published=false`、`publication.status=conflict` 保留，独立修订目录对应 `exports[].status=published` 已构成交付，不再恢复该 run。旧两节点并未原地修复或重新批准。核验沿用 `skip-hardware`；core copyback、mini-chunk 读取退休、跨 stream/重入和外部比较器缺口保留，没有 kernel 修改或 GPU 验证。FMHA、MoE 仍在修订，Sparse 保留来源阻塞，整体条件流程 issue 继续待决。

以上保留该交付时点；后续 2.2.9 已澄清源缺陷抽象边界，Sparse 的低层槽位问题现已[记录归档](../../../issues/archive/kernel_wiki/sparse-tc-control-layout-overlap.json)，不再作为提取阻塞，实际 Design/P6 仍须独立完成和审阅。
