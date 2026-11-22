# Schema Wiki

当前收录 18 个算子的 48 份完整设计：原六项 15 份，本批十二项 33 份。先读各算子的 `dispatch.md`，再按选择条件阅读对应 YAML；共享 Contract 位于各目录的 `root.yaml`。

| 算子 | 设计数 | 阅读入口 | 成品清单 |
| --- | ---: | --- | --- |
| FMHA MXFP8 | 3 | [dispatch](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md) | [current.json](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/.extraction/current.json) |
| FP8 Grouped GEMM | 2 | [dispatch](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | [current.json](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/.extraction/current.json) |
| GDN Prefill | 2 | [dispatch](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | [current.json](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/current.json) |
| MLA FP8 | 1 | [dispatch](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | [current.json](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/current.json) |
| NVFP4 MoE | 3 | [dispatch](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md) | [current.json](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/.extraction/current.json) |
| Sparse Attention | 4 | [dispatch](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) | [current.json](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/.extraction/current.json) |

原六项的 `current.json` 与[原六项总索引](../schema_discussion/schema_audit_and_cleanup_20261119/CURRENT_SCHEMA_INDEX.json)保持原有证据边界。

本批十二项如下。九个修订目录以 current 记录完整的 19 YAML/9 dispatch 和有限源复核；它们不是新的 Runner P6 回执。另外三目录的 14 YAML/3 dispatch 保持原发布字节，继续使用原 receipt。十二份原 receipt 均保留历史原义。

| 算子 | 设计数 | 阅读入口 | 当前字节依据 |
| --- | ---: | --- | --- |
| bf16_nvfp4_gemm | 7 | [dispatch](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/dispatch.md) | [原 receipt](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/.extraction/receipt.json) |
| fp8_mxfp4_grouped_gemm_masked | 1 | [dispatch](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/dispatch.md) | [current](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/.extraction/current.json) |
| bf16_fused_moe | 1 | [dispatch](aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/dispatch.md) | [current](aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/.extraction/current.json) |
| fp8_scaled_fused_moe | 3 | [dispatch](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/dispatch.md) | [原 receipt](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/.extraction/receipt.json) |
| fmha_bf16 | 4 | [dispatch](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/dispatch.md) | [原 receipt](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/.extraction/receipt.json) |
| fmha_fp8_per_token_head | 3 | [dispatch](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/dispatch.md) | [current](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/.extraction/current.json) |
| mla_absorbed_fp8 | 3 | [dispatch](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/dispatch.md) | [current](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/.extraction/current.json) |
| mla_nvfp4 | 1 | [dispatch](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/dispatch.md) | [current](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/.extraction/current.json) |
| paged_indexed_latent_fp8 | 3 | [dispatch](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/dispatch.md) | [current](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/.extraction/current.json) |
| paged_latent_mxfp8_nvfp4 | 2 | [dispatch](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/dispatch.md) | [current](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/.extraction/current.json) |
| gdn_conv_mtp_bf16_state_bf16 | 3 | [dispatch](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/dispatch.md) | [current](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/.extraction/current.json) |
| kda_raw_prefill_state_bf16 | 2 | [dispatch](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/dispatch.md) | [current](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/.extraction/current.json) |

[本批九项当前索引](../schema_discussion/extract_schema_review_20261121/CURRENT_SCHEMA_INDEX.json)与[实施报告](../schema_discussion/main_integration_20261122/FINAL_REPORT.md)记录修订范围和验证结果；[原交付索引](../schema_discussion/extract_twelve_20261119/DELIVERY_INDEX.json)保留修订前的 33 YAML/12 dispatch 摘要。硬件资料占位、外部比较器缺口、来源同步风险与性能例外仍以各 dispatch/YAML 为准，结构检查和有限源复核不证明 GPU 运行或全动态域数值正确性。

## Catalog 输出目录

输出目录按 [kernel_wiki/catalog.json](../kernel_wiki/catalog.json) 的 44 条 `paths.readme` 父路径镜像；18 项已有成品，其余保留输出占位。

| 分类 | 输出目录 | 算子数 |
| --- | --- | ---: |
| GEMM | [gemm](aka-cuda-operators-20261030/operators/gemm/)（dense / grouped / grouped_masked） | 15 |
| MoE | [moe](aka-cuda-operators-20261030/operators/moe/) | 5 |
| FMHA | [fmha](aka-cuda-operators-20261030/operators/fmha/) | 8 |
| MLA | [mla](aka-cuda-operators-20261030/operators/mla/) | 4 |
| 稀疏注意力 | [sparse_attention](aka-cuda-operators-20261030/operators/sparse_attention/) | 4 |
| 线性注意力 | [linear_attention](aka-cuda-operators-20261030/operators/linear_attention/) | 8 |

算子说明、源码与证据从 [Kernel Wiki](../kernel_wiki/CATALOG.md)进入。提取操作见 [Supervisor 手册](../schema_extract/README.md#输入与输出)和[流程总览](../EXTRACT_SCHEMA.md)。本批基线、workflow 自然提交与公开证据范围见[公开历史说明](../schema_discussion/main_integration_20261122/PUBLIC_HISTORY.md)。
