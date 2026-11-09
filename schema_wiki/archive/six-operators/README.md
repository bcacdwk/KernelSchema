# 六算子提取尝试：结果归档

六算子提取尝试产生的初始结果、后续修订和阅读勘误全部集中在本目录。版本对应关系见下表及[修订记录](revisions/README.md)；试用、Reviewer 验证、恢复和暂停过程见 [Schema Extract 六算子档案](../../../schema_extract/archive/six-operators/README.md)。

```text
six-operators/
├── initial/       六项初始交付，保留批次与算子子目录
├── revisions/     后续修订，按当时提取工具版本分组
├── errata/        原始结果的索引勘误
└── LOCATIONS.json 旧路径、新路径和文件 SHA256 对照
```

YAML、dispatch 和 `.extraction/receipt.json` 已整体迁入上述目录，内容保持原字节。归档搬迁不新增批准或发布身份；回执内的原输出地址仍是历史事实，当前位置以 [LOCATIONS.json](LOCATIONS.json) 为准。

本次没有修改 Git 忽略的运行状态。未来若恢复旧 run，需要先核对其记录的 output/export 地址与本归档的对应关系，不能假定原路径仍存放这些结果。

## 六算子的历史与修订

| 算子 | 初始交付 | 后续正式版本 / 本批暂停记录 |
| --- | --- | --- |
| 分组 GEMM | [2.2.2 三节点原稿](initial/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | [2.2.4 三节点校正版](revisions/2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) → [2.5.0 七节点版](revisions/2.5.0/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) |
| MoE | [2.2.2 六十节点原稿](initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md)，配合[阅读勘误](errata/nvfp4_fused_moe.md) | [2026-11-09：96/154 的暂停清单](../../../schema_extract/archive/six-operators/20261109-pause/moe-nodes.md)；尚非新交付 |
| FMHA | [2.2.2 八节点原稿](initial/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md) | [2026-11-09：26/43 的暂停清单](../../../schema_extract/archive/six-operators/20261109-pause/fmha-nodes.md)；尚非新交付 |
| MLA | [2.2.2 单节点原稿](initial/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | [2.5.0 两节点版](revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) |
| 稀疏注意力 | [2.2.2 三节点原稿](initial/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md)，配合[旧索引勘误](errata/dual_cache_latent_bf16.md) | [2.5.0 八节点版](revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) |
| GDN | [2.2.2 两节点原稿](initial/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | [2.2.7 六节点版](revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) |

初始交付与早期校正版的验证范围见[试用记录](../../../schema_extract/archive/six-operators/20261103-trial/TRIAL_REPORT.md)。条件流程、成员范围等表达问题见[质量初验](../../../schema_extract/archive/six-operators/20261103-trial/OPERATOR_REVIEW.md)及 [issue 历史总表](../../../schema_extract/issues/archive/TRIAGE.md)。

## 阅读勘误的边界

`errata/` 两页只修正原发布索引中过期的“待生成”阶段说明并提供阅读入口，原表格、YAML 和发布回执保留。勘误不是新的语义审阅或批准；MoE 的表达保留意见、Sparse 原三节点的历史问题仍按相应版本理解。

旧路径 `schema_wiki/errata/` 的文档现归入本专题的 `errata/`。历史 issue 的原路径文字保留，当前链接已更新。

## 部分结果与过程记录

FMHA、MoE 的新一轮进度以[暂停交接](../../../schema_extract/archive/six-operators/20261109-pause/README.md)及[逐节点 CSV](../../../schema_extract/archive/six-operators/20261109-pause/nodes.csv)登记。清单包含已批准节点、待审稿、失败或未完成项及未开始项，不能用原有八节点、六十节点交付代替新一轮 P6。

相关独立评估、推进方案和多次巡检记录统一从[六算子研究档案](../../../schema_extract/archive/six-operators/README.md)查阅。本页不搬入 `.schema_extract/` 的运行状态或私有 checkpoint，也不把部分稿发布为完整结果。
