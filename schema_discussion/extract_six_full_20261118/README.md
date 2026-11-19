# 六算子完整提取交付

北京时间 2026-11-19 16:30 完成交付核对：六项逻辑任务均已有正式产物，共 15 份完整设计和六份中文 `dispatch.md`。其中五项通过正常 P6 并由 Runner 发布；FMHA 按用户最后指令由 xhigh 定点小修后交付，保留原自动流程未通过的事实。

阶段模型均按原登记使用 Claude Code `qwen3.8-max[1m]` / `max`；FMHA 最后四处轴词与必要版本更新由用户指定的内部 `gpt-6-astra / xhigh` 子代理完成，没有再发起 P3/P5/P6 模型循环。

| 算子 | 设计数 | 正式阅读入口 | 交付核对 |
| --- | ---: | --- | --- |
| FP8 Grouped GEMM | 2 | [dispatch](../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | [报告](DELIVERY_GEMM.md) |
| NVFP4 MoE | 3 | [dispatch](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md) | [报告](DELIVERY_MOE.md) |
| FMHA MXFP8 | 3 | [dispatch](../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md) | [用户授权小修交付报告](DELIVERY_FMHA.md) |
| MLA FP8 | 1 | [dispatch](../../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | [报告](DELIVERY_MLA.md) |
| Sparse Attention | 4 | [dispatch](../../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) | [报告](DELIVERY_SPARSE.md) |
| GDN Prefill | 2 | [dispatch](../../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | [报告](DELIVERY_GDN.md) |

六份交付的实际文件集合、摘要、回执、交付依据和调用结清汇总见 [DELIVERY_INDEX.json](DELIVERY_INDEX.json)。六个原 run 共201次已登记阶段调用，均已结清；另起后取消的 Sparse successor 保留其1次调用记录，不混入这201次。Supervisor及内部子代理用量不包含在该阶段调用数中。

## 本轮续接与小修

Sparse 和 FMHA 在 P6 遇到同一旧工具文件清单错误，合法候选被误拒。按[用户决定](recovery_reassessment/USER_DECISION.md)，修复执行器以显式绑定接上原任务，保留源码、语义规则、候选、历史批准与调用账，并留下待审计标记。原冻结文件未热改。

Sparse 只新增一次正常 P6 审阅后发布，五份正文与原 `66ad…` 候选字节完全相同。

FMHA 新 P6 返回了归约轴词问题，但回执又出现虚拟 EOF 多一行。用户明确要求[由 xhigh 小修后交付](recovery_reassessment/FMHA_USER_MINOR_FIX_DECISION.md)，不打回继续流程。小修仅 root/split_kv 中四处 QK 轴词与两份 Version；mma_sync、dispatch 和其他正文保持。交付凭据明确 `completion_basis=user_authorized_minor_fix`，没有伪造原 P6 的 `approve`，原自动 run 保持其 `invalid_receipt/reopen` 历史。

因此，FMHA 的原自动状态不代表当前正式目录尚未交付，也不能据此自动恢复旧 run 覆盖本次结果；当前交付以这次用户授权的小修凭据为准。

## 留给用户决定的检查

FMHA 与 Sparse 的 `.extraction/receipt.json` 均有 `continuation_audit`，状态保持 `pending_user_decision`。其中包括工具故障处的历史审批适用性、Sparse 的寄存器/无 spilling 表述疑点、FMHA 的来源/措辞限制及本次人工小修记录。没有把这些标记冒称已经完成独立审计。

本轮只做正常交付所需的文件、来源、回执与机械检查；未追加广泛语义审计、GPU正确性或性能实验。之前各项历史证据审计的 `partial` 范围仍按对应报告披露；Sparse 最后一次 P6 的证据核验 `verified` 不代表其全部历史已重新全面审计。

其他 issue 的后续修改按用户之后的指挥处理。记录入口：[问题索引](ISSUES.md)、[14项决策审阅](ISSUE_DECISION_REVIEW.md)；北京时间展示 issue 是该审阅之后新增的待办。

## 监督与记录

按用户要求，原定时提醒已停止，随后在对话内每五分钟直接检查。现在六项逻辑交付均已完成，直接轮询也已结束；无在途模型调用。运行身份和原自动状态见 [RUNS.json](RUNS.json)，历史位置见 [归档与输入](ARCHIVE_INPUTS.md)。

本轮未提交或推送 Git；旧产物、失败记录、被取消 successor 和已用调用均保留。其他来源、语义规则或任务变化后的承接不属于此次工具修复续接的实现范围。
