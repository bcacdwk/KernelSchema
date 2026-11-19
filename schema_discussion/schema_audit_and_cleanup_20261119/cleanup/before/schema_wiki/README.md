# Schema Wiki

当前六个算子共 15 份设计。2026-11-19 的 issue 优化与人工审修已完成，详见[本轮汇总](../schema_discussion/issue_optimization_20261119/README.md)及[当前文件与审修索引](../schema_discussion/issue_optimization_20261119/SCHEMA_REVIEW_INDEX.json)。

## 当前交付

| 算子 | 设计数 | 阅读入口 | 当前修订与检查 |
| --- | ---: | --- | --- |
| FP8 Grouped GEMM | 2 | [目录](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | [存储名称归属修订](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/.extraction/issue-review-20261119.json) |
| NVFP4 MoE | 3 | [目录](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md) | 原正常 P6 交付 |
| FMHA MXFP8 | 3 | [目录](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md) | [遗留问题专项审修完成](../schema_discussion/issue_optimization_20261119/SCHEMA_AUDIT_FMHA.md) |
| MLA FP8 | 1 | [目录](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | [非 Tensor 参数表达修订](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/issue-review-20261119.json) |
| Sparse Attention | 4 | [目录](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) | [遗留问题专项审修完成](../schema_discussion/issue_optimization_20261119/SCHEMA_AUDIT_SPARSE.md) |
| GDN Prefill | 2 | [目录](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | 原正常 P6 交付 |

原提取凭据保持原字节；后续人工修订分别保存来源核对、当前 manifest 和完整目录检查结果。FMHA/Sparse 本次由各自一位 xhigh 承担 worker 与 reviewer，未重新运行 extract 或 P6。原回执中的 `continuation_audit` 保留交付当时的记录，当前专项结论由上表链接及各算子的 `.extraction/schema-audit-20261119.json` 给出。

MLA/GEMM 的限域修订已通过[独立复核](../schema_discussion/issue_optimization_20261119/ROUND4_REVIEW_AND_ARCHIVE.md)。FMHA 正式外部比较器等缺失证据，以及 Sparse 已知源控制区布局缺陷，均在专项报告中保持明确边界；本轮没有修 kernel 或运行 GPU 验证。

## 既有材料入口

| 材料 | 位置 |
| --- | --- |
| 六算子的初始结果、各版修订、勘误及位置映射 | [六算子结果归档](archive/six-operators/README.md) |
| 试用、Reviewer 验证材料、恢复评估与执行记录 | [六算子过程档案](../schema_extract/archive/six-operators/README.md) |
| 历史尝试的部分成果与未完成项 | [2026-11-09 暂停清单](../schema_extract/archive/six-operators/20261109-pause/README.md) |
| 全部结果专题归档 | [archive](archive/README.md) |

```text
schema_wiki/
├── archive/six-operators/
│   ├── initial/       六项初始结果
│   ├── revisions/     同一尝试后续形成的版本
│   ├── errata/        阅读勘误
│   └── LOCATIONS.json 归档前后位置与原文件摘要
└── aka-cuda-operators-20261030/operators/
                        当前六项交付与其他 catalog 输出占位
```

归档中的 YAML、dispatch 和原回执保持原字节。回执中的原发布位置保留其历史含义；当前文件位置见 [LOCATIONS.json](archive/six-operators/LOCATIONS.json)。本次目录归置没有修改运行数据或恢复提取。

## Catalog 输出目录

输出目录按 [kernel_wiki/catalog.json](../kernel_wiki/catalog.json) 的 44 条 `paths.readme` 父路径镜像。上表六项已写入本轮交付，其余 38 项保持输出占位；旧产物仍在历史归档中。

| 分类 | 初始输出目录 | 算子数 |
| --- | --- | ---: |
| GEMM | [gemm](aka-cuda-operators-20261030/operators/gemm/)（dense / grouped / grouped_masked） | 15 |
| MoE | [moe](aka-cuda-operators-20261030/operators/moe/) | 5 |
| FMHA | [fmha](aka-cuda-operators-20261030/operators/fmha/) | 8 |
| MLA | [mla](aka-cuda-operators-20261030/operators/mla/) | 4 |
| 稀疏注意力 | [sparse_attention](aka-cuda-operators-20261030/operators/sparse_attention/) | 4 |
| 线性注意力 | [linear_attention](aka-cuda-operators-20261030/operators/linear_attention/) | 8 |

后续新任务应明确选择输出目录，操作见 [Supervisor 手册](../schema_extract/README.md#输入与输出)和[流程总览](../EXTRACT_SCHEMA.md)。恢复已有 run 前须先核对其记录的 output/export 位置与归档映射；历史产物归档不等于重新发布，也不替代新一轮 P6。

算子说明、源码与证据从 [Kernel Wiki 总索引](../kernel_wiki/CATALOG.md)进入。
