# 六算子全量重提取：输入与历史归档

2026-11-18，启动基线 `24d2b3d`。本轮使用当前流程，为以下六个 catalog ID 新建 P1–P6 完整提取。输入来自 [44 题 catalog](../../kernel_wiki/catalog.json)，历史稿仅作归档证据。

## 六个输入与正式输出

下表路径相对于 `aka-cuda-operators-20261030/operators/`：源码根为 `kernel_wiki/`，正式输出根为 `schema_wiki/`。每题入口为源码目录下 `candidate/solution.py`；合同、输入、reference、形状域和验收资料按 catalog 的 `paths` 绑定。

| Catalog ID | 源码 / 输出共用相对路径 | 历史最后成果 |
| --- | --- | --- |
| `fp8_scaled_grouped_gemm` | `gemm/grouped/fp8_scaled_grouped_gemm` | 2.5.0，7 节点交付 |
| `nvfp4_fused_moe` | `moe/nvfp4_fused_moe` | 原 60 节点交付；后续 96/154 批准后暂停 |
| `fmha_mxfp8_qkv` | `fmha/fmha_mxfp8_qkv` | 原 8 节点交付；后续 26/43 批准后暂停 |
| `mla_expanded_fp8` | `mla/mla_expanded_fp8` | 2.5.0，2 节点交付 |
| `dual_cache_latent_bf16` | `sparse_attention/dual_cache_latent_bf16` | 2.5.0，8 节点交付 |
| `gdn_prefill_bf16_state_bf16` | `linear_attention/gdn_prefill_bf16_state_bf16` | 2.2.7，6 节点交付 |

MLA 的当前 catalog 另声明 [校注入口](../../kernel_wiki/annotations/mla_expanded_fp8/solution.py)，新输入应随原入口一并冻结；它不替代原实现绑定。

## 已有结果和过程记录

- [结果专题](../../schema_wiki/archive/six-operators/README.md)：`initial/` 保存六项初始交付，`revisions/` 保存后续正式版本，`errata/` 保存勘误。现有 [LOCATIONS.json](../../schema_wiki/archive/six-operators/LOCATIONS.json) 的 125 个 payload 已逐项验证存在且 SHA256 一致，本轮未重复搬运。
- [过程专题](../../schema_extract/archive/six-operators/README.md)：试用、审阅、恢复评估、执行记录与 [11-09 暂停清单](../../schema_extract/archive/six-operators/20261109-pause/README.md) 已归档。FMHA / MoE 的部分成果不等于本轮 P6 完成。
- [11-17 两题验收讨论](../extract_live_acceptance_20261117/README.md)保留原位；它属于 FA4/QSA。其本机运行目录随下表整体归档，旧报告中的绝对路径仍表示记录当时位置。其他方法论讨论和 Template 案例保持原位。

## 本次完成的运行归档

主代理确认旧提取 Runner、模型与 monitor 均不在运行后，以下目录通过同盘 `rename` 整体移到 `.schema_extract/archive/20261118-before-six-rerun/` 下同名目录；目录 inode、设备和权限保持一致，内部文件、冻结记录、checkpoint 和草稿未改写。

| 原位置 | 归档位置（相对于归档根） |
| --- | --- |
| `.schema_extract/v2/` | `v2/`；25 个历史 run，均属于上述六算子 |
| `.schema_extract/live_acceptance_20261117/` | `live_acceptance_20261117/`；FA4/QSA 验收运行 |
| `.schema_extract/preparations/` | `preparations/` |
| `.schema_extract/review-eval/` | `review-eval/` |
| `.schema_extract/supervision/` | `supervision/`；包括恢复、暂停清单与完整审阅包 |
| `.schema_extract/trials/` | `trials/`；包括 `six-families/` |

本机 [README](../../.schema_extract/archive/20261118-before-six-rerun/README.md) 与 [LOCATIONS.json](../../.schema_extract/archive/20261118-before-six-rerun/LOCATIONS.json)给出绝对路径映射。旧 state、inputs、回执和讨论报告中的绝对路径保留历史含义；搬迁不重新绑定旧 run，不能直接重放原 resume / monitor 命令。新任务可重新创建默认 `.schema_extract/v2`。

11-09 暂停清单中的最后 run 如下，均位于归档后的 `v2/runs/<run_id>/`；其他早期尝试也完整保留。

| 简称 | 历史 run ID |
| --- | --- |
| GEMM | `d4e41124-e29c-458c-a1ce-2a9bf87ee0a8` |
| MoE | `feb0c655-4e13-43ac-817d-1f457f31c029` |
| FMHA | `45ea6608-66b2-4915-b47d-59d2c115602e` |
| MLA | `531567f1-bcae-45ad-acf6-fd674c055fdc` |
| Sparse | `8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7` |
| GDN | `45f9bb51-77f3-4285-b7a7-251cf2744fa1` |

## 输出准备与范围

六个正式输出叶目录检查时均只有 `.gitkeep`；本次已删除这六个占位文件并移除空叶，父目录保留，使 Runner 可在不存在的输出路径发布。没有发现其他需要归档的活跃六算子交付或报告。已补上此前缺失的 [Schema Wiki archive 总索引](../../schema_wiki/archive/README.md)。本次归档未改源码、schema 字段、Template 案例或 kernel 输入，也未启动提取或提交 Git。
