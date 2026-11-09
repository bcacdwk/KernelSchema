# Schema Extract 暂停交接

> 归档说明：2026-11-09 暂停时的结果与未完成项快照。后续处理需依据新的用户指示。 [归档索引](../../README.md)。
> 结果文件随后已整体归入六算子专题；正文中的原发布位置指暂停快照时点，现位置见[归档映射](../../../../schema_wiki/archive/six-operators/LOCATIONS.json)。本机运行目录及当时的审阅包未作迁移。

已按用户要求停止新提取、子代理派发和发布；状态采集于 2026-11-09T04:14:57.243253+00:00。

全局暂停门禁 `f15a81ef-b36f-47ac-8bfd-915b70bb5916` 保持有效，定时巡检已关闭。两条未完成 run 已 paused，所有在途与终止未确认记录均清空。

| 算子 | 当前成果 | 已批准 / 计划 | 待审候选 | 需继续写作或返工 | 失败尝试、暂无创作结果 | 未开始 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [GDN](<gdn-nodes.md>) | [已交付](<../../../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16>) | 6 / 6 | 0 | 0 | 0 | 0 |
| [MLA](<mla-nodes.md>) | [已交付](<../../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8>) | 2 / 2 | 0 | 0 | 0 | 0 |
| [GEMM](<gemm-nodes.md>) | [已交付](<../../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm>) | 7 / 7 | 0 | 0 | 0 | 0 |
| [SPARSE](<sparse-nodes.md>) | [已交付](<../../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16>) | 8 / 8 | 0 | 0 | 0 | 0 |
| [FMHA](<fmha-nodes.md>) | 已暂停；未完成本轮 P6 / 发布 | 26 / 43 | 4 | 13 | 0 | 0 |
| [MOE](<moe-nodes.md>) | 已暂停；未完成本轮 P6 / 发布 | 96 / 154 | 9 | 1 | 5 | 43 |

## 已保留的内容

- 四个已交付算子的 YAML、dispatch 和交付回执已复制到审阅包；原发布目录保持不变。
- FMHA 与 MoE 保存了当前工作树、各未批准节点的已记录草稿，以及当前 state/input 身份副本。
- 已逐项比对未批准节点最后一次尝试的工作区；本轮未发现相对所选安全稿额外的未收录差异片段，所有原始尝试目录仍保留。
- 每个节点均列出状态、下一角色、调用数、版本、checkpoint、原文件路径和最近尝试。未开始项也单独入表；退役历史节点另标，不计入当前计划。
- 原 `.schema_extract/v2` 的所有 checkpoint、尝试、回执与历史 run 均保留；审阅包不替代原运行状态，也不作为新的发布结果。

## 本轮收尾要点

- FMHA 首个版本纠错已通过独立复核：`tcg_d128_flat_tma_xpose_narrow` 的版本 4 稿已进入已批准工作树。当前工作 checkpoint 与旧 normal cursor 不同，复查应以清单所列 working checkpoint 为准。
- FMHA 剩余节点均有既往工作记录；其中原先的 pending 标签不应一概解释为完全未开始。后续需要按各节点维护任务和原草稿决定继续修订或重写。
- MoE 的已批准成果为 96/154；失败与草稿状态分别保留，43 个未开始节点列在逐节点清单中。遗留的失败调用与准备未派发记录已在暂停门禁下结算，没有重新发起模型调用。
- 调用次数、旧预算追加、一次性决定、源/config、语义与执行绑定保留；本轮暂停没有追加预算或重新提取。

## 回看入口

- [逐节点 CSV](<nodes.csv>)；[完整 JSON 清单](<../../../../.schema_extract/supervision/pause-closeout-20261109/inventory.json>)。
- [审阅副本目录](<../../../../.schema_extract/supervision/pause-closeout-20261109/review-bundle>)；[压缩审阅包](<../../../../.schema_extract/supervision/pause-closeout-20261109/review-bundle.zip>)。
- [停稳与无新增调用核对](<../../../../.schema_extract/supervision/pause-closeout-20261109/settlement-verification.json>)；[凭据状态核对（不含密钥）](<../../../../.schema_extract/supervision/pause-closeout-20261109/credential-status.json>)。

## 后续决定

目前不继续任何子代理或提取。后续先由用户决定保留、补修或重提取范围；再按原 run 的真实状态、当前规则/执行器与届时预算重新评估。
旧的自动推进与分批暂停/补时安排已被这次明确暂停取代；不要直接释放 hold 或重放历史 resume。恢复前需确认新执行会话加载当前凭据。

密钥及其摘要不写入本交接或审阅包。凭据核对只检查现有本机配置，没有模型试跑或执行守护进程重启。
