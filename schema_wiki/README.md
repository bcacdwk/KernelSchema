# Schema Wiki

当前收录六个算子的 15 份完整设计。先读各算子的 `dispatch.md`，再按选择条件阅读对应 YAML；共享 Contract 位于各目录的 `root.yaml`。

| 算子 | 设计数 | 阅读入口 | 当前版本与检查 |
| --- | ---: | --- | --- |
| FMHA MXFP8 | 3 | [dispatch](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md) | [文件与验收元数据](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/.extraction/current.json) |
| FP8 Grouped GEMM | 2 | [dispatch](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | [文件与验收元数据](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/.extraction/current.json) |
| GDN Prefill | 2 | [dispatch](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | [文件与验收元数据](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/current.json) |
| MLA FP8 | 1 | [dispatch](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | [文件与验收元数据](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/current.json) |
| NVFP4 MoE | 3 | [dispatch](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md) | [文件与验收元数据](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/.extraction/current.json) |
| Sparse Attention | 4 | [dispatch](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) | [文件与验收元数据](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/.extraction/current.json) |

[当前总索引](../schema_discussion/schema_audit_and_cleanup_20261119/CURRENT_SCHEMA_INDEX.json)汇总文件摘要、版本、固定来源与语义审阅证据。各 `.extraction/current.json` 是当前成品清单和检查记录，不是 Runner 的阶段回执。验收覆盖来源语义审阅和只读结构检查；硬件字段仍未完整绑定，GPU 与完整动态域数值正确性没有由这些检查证明。外部比较器缺口、源码限制和调用前提以各 YAML、dispatch 为准。

## Catalog 输出目录

输出目录按 [kernel_wiki/catalog.json](../kernel_wiki/catalog.json) 的 44 条 `paths.readme` 父路径镜像；六项已有成品，其余保留输出占位。

| 分类 | 输出目录 | 算子数 |
| --- | --- | ---: |
| GEMM | [gemm](aka-cuda-operators-20261030/operators/gemm/)（dense / grouped / grouped_masked） | 15 |
| MoE | [moe](aka-cuda-operators-20261030/operators/moe/) | 5 |
| FMHA | [fmha](aka-cuda-operators-20261030/operators/fmha/) | 8 |
| MLA | [mla](aka-cuda-operators-20261030/operators/mla/) | 4 |
| 稀疏注意力 | [sparse_attention](aka-cuda-operators-20261030/operators/sparse_attention/) | 4 |
| 线性注意力 | [linear_attention](aka-cuda-operators-20261030/operators/linear_attention/) | 8 |

算子说明、源码与证据从 [Kernel Wiki](../kernel_wiki/CATALOG.md)进入。提取操作见 [Supervisor 手册](../schema_extract/README.md#输入与输出)和[流程总览](../EXTRACT_SCHEMA.md)。旧结果、交付回执与调查材料见[历史档案](../schema_discussion/archive/README.md)。
