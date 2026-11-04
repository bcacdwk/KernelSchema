# 六类算子提取试用记录

2026-11-03，六项固定实现均通过 P1–P6 独立审阅并发布，共 **77 个节点 YAML、6 份 dispatch 索引和 6 份发布回执**。原始交付使用冻结的 Schema Extract **2.2.2**；全部运行已结算，交付文件与回执摘要一致，六份完整证据审计均为 `verified`，没有核验错误或缺失证据。

分组 GEMM 现有[2.2.4 校正版](../schema_wiki/revisions/2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md)，已单独终审、审计和导出；日常阅读优先使用校正版。下表保留初次交付，便于追溯原始审阅与修订。

| 原始交付算子 | 节点数（含 root） | 记账调用数 | 运行与发布回执 |
| --- | ---: | ---: | --- |
| [fp8_scaled_grouped_gemm](../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | 3 | 23 | [4aac1ea4-94fa-4b4c-93ca-d48a8eb08996](../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/.extraction/receipt.json) |
| [nvfp4_fused_moe](../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md) | 60 | 188 | [3563c3b7-2b7c-4967-9d41-9c2042e53d2c](../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/.extraction/receipt.json) |
| [fmha_mxfp8_qkv](../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md) | 8 | 32 | [acb68909-aaaa-4bc2-8b16-94904752af8e](../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/.extraction/receipt.json) |
| [mla_expanded_fp8](../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | 1 | 14 | [e3f7704f-061f-48ff-8131-cc317afe6003](../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/receipt.json) |
| [dual_cache_latent_bf16](../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) | 3 | 22 | [d5a73c01-7c3a-45b2-a3ce-0d5eec634808](../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/.extraction/receipt.json) |
| [gdn_prefill_bf16_state_bf16](../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | 2 | 22 | [97902388-eda0-4cbd-b8a6-ba71440e86ab](../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/receipt.json) |

调用数合计 301，按六条互不重叠的 lineage 计数，包含接续前的尝试；不再次叠加前任总数。4 次调用的提供者 usage 未知，其余保留原值；缓存输入已包含于 input，且续会话计数范围仍未确认，因此不汇总 token 或据此估算费用。

## 来源与验证范围

来源为 `kernel_wiki/aka-cuda-operators-20261030` 中选定的固定实现，输入锁和发布回执保留源码版本、逐文件 SHA-256、规则身份及 P6 批准身份。检查显式使用 `--skip-hardware`，硬件保持模板占位；交付忠实保留源码限制与外部比较器缺口，不宣称重新完成 GPU 验收或复现全部外部比较器。

完整本地证据位于 `.schema_extract/v2/`，六份审计报告位于 `.schema_extract/trials/six-families/audit/*.completed.json`，批次核验表为 `.schema_extract/trials/six-families/batch-acceptance.json`。Supervisor 接续和决策记录位于同目录的 `SUPERVISOR.md`、`checks/`、`recovery/` 与 `post-batch/`。

## 已解决的问题

- [外部比较器缺项阻塞 P1](issues/archive/extract_workflow/p1-external-comparator-gate.json)：明确来源缺口与核心计算事实的边界，保留已知判据后继续忠实提取。
- [P4 索引布局硬停止](issues/archive/extract_tools/p4-index-layout-hard-stop.json)：2.2.3 明确标题位置和首差异诊断，纯布局错误进入有限修复。
- [并发分支的错误调用定位](issues/archive/extract_tools/p5-handoff-attempt-locator.json)：2.2.4 绑定真实失败 attempt，旧记录不猜测兄弟调用。
- [语义 summary 与文件哈希混称](issues/archive/extract_workflow/proposal-summary-ownership.json)：2.2.4 明确模型填写非空语义总结，工具生成文件 SHA-256。
- [P6 树结构的产物证据](issues/archive/extract_workflow/p6-tree-artifact-evidence.json)：2.2.4 允许在完整引用和实际树检查支持下提交产物树结论，源码语义结论仍需来源依据。

## 当前版本验证

真实验证发现并修正了一处转录范围：原始分组 GEMM 的 `root.yaml` 与 `umma_wide.yaml` 将 graph API 失败概括为关闭 graph 的原因；源码只检查构建及 launch 返回值，销毁调用的返回值被忽略。校正版的 root 和 wide 已明确区分这两类调用，全部三个节点通过独立审阅、P6 终审和完整证据审计，并通过正式 `export` 发布到独立版本目录。原始交付及其回执保留，依据和修订记录归档于 `.schema_extract/trials/six-families/post-batch/archive/grouped-gemm-graph-error-scope.json`。

2.2.4 的 29 项提取回归已通过。MoE 第一次因 `tree.evidence=[]` 被拒的原始 P6 提案，在新门禁下完成只读回放并通过实际 60 节点树检查；原运行状态、原始回执和既有发布未被改写。

新版本的真实多节点验证已完成，使用相同源码字节和绑定的 `fp8_scaled_grouped_gemm`，run 为 `b61c62b1-b0dd-4ed8-a21f-e9b7227a0d3b`。共 15 次调用、0 次格式修复、0 次 CLI 失败、1 次正常语义退修，约 86 分钟完成；全部 15 次调用的证据审计通过。P6 的 `tree.evidence=[]` 实际使用完整产物证据和树检查通过放行。

原 2.2.2 接续运行有 21 次调用，其中 7 次空 `summary` 修复；本次为 0 次。这是一对独立模型运行的样本观察，不用于推算整体成功率、性能或费用。完整结果位于 `.schema_extract/trials/six-families/post-batch/validation-2.2.4.json` 及同目录审计文件；首次试用发布与随后版本导出的时间分别记录。
