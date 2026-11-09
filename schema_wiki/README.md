# Schema Wiki

本目录是 schema 的输出区。六个典型算子提取尝试中已经形成的原稿、修订版和勘误，已整体收拢到 [六算子结果归档](archive/six-operators/README.md)。当前顶层的 catalog 镜像保留输出占位，历史产物按专题保存。

## 既有材料入口

| 材料 | 位置 |
| --- | --- |
| 六算子的初始结果、各版修订、勘误及位置映射 | [六算子结果归档](archive/six-operators/README.md) |
| 试用、Reviewer 验证材料、恢复评估与执行记录 | [六算子过程档案](../schema_extract/archive/six-operators/README.md) |
| 最近一次部分成果与未完成项 | [2026-11-09 暂停清单](../schema_extract/archive/six-operators/20261109-pause/README.md) |
| 全部结果专题归档 | [archive](archive/README.md) |

```text
schema_wiki/
├── archive/six-operators/
│   ├── initial/       六项初始结果
│   ├── revisions/     同一尝试后续形成的版本
│   ├── errata/        阅读勘误
│   └── LOCATIONS.json 归档前后位置与原文件摘要
└── aka-cuda-operators-20261030/operators/
                        catalog 镜像的输出占位
```

归档中的 YAML、dispatch 和原回执保持原字节。回执中的原发布位置保留其历史含义；当前文件位置见 [LOCATIONS.json](archive/six-operators/LOCATIONS.json)。本次目录归置没有修改运行数据或恢复提取。

## Catalog 输出目录

输出占位按 [kernel_wiki/catalog.json](../kernel_wiki/catalog.json) 的 44 条 `paths.readme` 父路径镜像。六项旧产物归档后，其原叶目录恢复为空 `.gitkeep` 占位，其他未提取目录原样保留。

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
