# Kernel Wiki

这里收录 L20D（B300）算子实现、题目合同、用例和验证证据，供查找实现、比较计算边界与提取 schema 使用。当前批次共 **44 个算子：20 个 AKA 解、24 个参考实现**；原始材料中使用 SM103 等环境标识，具体环境以各题证据为准。

**找算子：** [全部 44 题与用途](CATALOG.md) · [按精度、阶段、来源和状态查找](FACETS.md)

**理解算子：** [分类区别与阅读顺序](GUIDE.md) · [验证状态和性能口径](EVIDENCE.md)

## 按功能浏览

| 分类 | 数量 | AKA / 参考 | 主要区分 |
| --- | ---: | ---: | --- |
| [矩阵乘法 GEMM](aka-cuda-operators-20261030/operators/gemm/README.md) | 15 | 6 / 9 | [普通](aka-cuda-operators-20261030/operators/gemm/dense/README.md) 6、[分组](aka-cuda-operators-20261030/operators/gemm/grouped/README.md) 5、[带掩码分组](aka-cuda-operators-20261030/operators/gemm/grouped_masked/README.md) 4 |
| [融合专家计算 MoE](aka-cuda-operators-20261030/operators/moe/README.md) | 5 | 2 / 3 | 路由已给定，融合专家 FC1、激活、FC2 与归约 |
| [多头注意力 FMHA](aka-cuda-operators-20261030/operators/fmha/README.md) | 8 | 5 / 3 | Q/K/V 格式、缩放粒度、分页与连续布局 |
| [多头潜在注意力 MLA](aka-cuda-operators-20261030/operators/mla/README.md) | 4 | 2 / 2 | absorbed、expanded、D512 NVFP4 |
| [稀疏注意力](aka-cuda-operators-20261030/operators/sparse_attention/README.md) | 4 | 1 / 3 | 双缓存、物理索引、滑窗与压缩 KV |
| [线性注意力 GDN / KDA](aka-cuda-operators-20261030/operators/linear_attention/README.md) | 8 | 4 / 4 | prefill / decode / MTP、卷积边界、状态精度 |

## 目录约定

```text
kernel_wiki/
├── README.md                  总入口
├── CATALOG.md                 44 题用途、格式、阶段、来源和验收状态
├── FACETS.md                  多维交叉索引
├── GUIDE.md                   分类区别、选题与源码阅读方法
├── EVIDENCE.md                验证范围、计时口径和已知例外
├── IMPORT.json                导入来源、数量与原清单哈希
├── catalog.json               机器可读索引，路径相对于 kernel_wiki
├── tools/catalog.py           检索、校验与索引补丁生成
└── aka-cuda-operators-20261030/ 原库快照，保留原有层级与文件名
    ├── MANIFEST.json / CHECKSUMS.sha256 / PROBLEM_AUDIT.json
    ├── README.md / CONVENTIONS.md / PROBLEM_NOTES.md
    ├── tools/                 原库的校验与元数据工具
    └── operators/<family>/[<subcategory>/]<operator_id>/
        ├── README.md          逐题入口
        ├── candidate/         当前选定实现，入口 solution.py:Model.forward
        ├── problem/           合同、输入、正确性参考、生产对照与形状域
        └── performance/       统一摘要与原始验收、测量记录
```

一个算子只存一份完整材料；精度、阶段等维度通过索引交叉引用。稳定 ID 和原库内部相对路径均保留，便于追溯引用和复用整个 `candidate/`。各题还可能附带 `PRECISION.md`、`VARIANTS.md`、`USER_ACCEPTANCE.md` 和许可资料。

`catalog.json` 的 `summary` 是 Wiki 的计算概要，`source_description` 摘自导入的 `problem/metadata.json`，其中的 `〔…〕` 表示 Wiki 校注。核对合法输入时，沿条目的 `paths.shape_range` 查看动态参数域和跨字段约束。`paths.shape_cases` 列出固定用例。

若条目有 `paths.annotated_entrypoint`，可直接阅读修正了注释的源码。它与 `paths.entrypoint` 的导入原件逐行对应，`catalog.py check` 核对两者的 Python 语法树和内嵌 CUDA 的非注释内容。由当前 catalog 入口发起的新提取会冻结两份文件；原件仍是实现和 host 的绑定入口，校注只作阅读辅助，声明的校注路径缺失或不合法时明确报输入错误。旧提取是否包含校注，以其 `inputs.lock.json` 的 `source.files` 为准，不能由当前 catalog 推断。[展开式 FP8 MLA 校注版](annotations/mla_expanded_fp8/solution.py)

本批次来自 `/Users/hanyong/aka-cuda-operator-library`，于 2026-11-01 导入。除未列入源清单的 `.DS_Store` 缓存外，**1,266 个文件、51,906,145 字节均与源文件一致**。完整性信息见 [IMPORT.json](IMPORT.json)。

## 命令行检索

从 KernelSchema 仓库根目录执行；仅用 Python 标准库，不需要 CUDA、PyTorch 或 GPU。筛选条件取交集，关键词不区分大小写。

```bash
# 普通 / 分组 / masked 中，所有 MXFP4 相关 GEMM
python3 kernel_wiki/tools/catalog.py list --family gemm --precision mxfp4

# FP32 状态的 GDN prefill
python3 kernel_wiki/tools/catalog.py list gdn --stage prefill --state fp32

# AKA 的注意力 decode 实现
python3 kernel_wiki/tools/catalog.py list --origin aka --stage decode

# 需要补最终回执的三题；或人工接受性能例外的两题
python3 kernel_wiki/tools/catalog.py list --needs-receipt
python3 kernel_wiki/tools/catalog.py list --waiver

# 只看以生产实现为对照的测量
python3 kernel_wiki/tools/catalog.py list --comparator production

# 精确查看入口、接口、合同、性能证据的本地路径
python3 kernel_wiki/tools/catalog.py show fmha_bf16

# 供脚本 / Agent 继续处理
python3 kernel_wiki/tools/catalog.py list --family mla --json
```

也可以直接在编辑器搜索 [CATALOG.md](CATALOG.md)，或使用 `rg 'gdn|nvfp4|masked' kernel_wiki/CATALOG.md`。全部选项见 `python3 kernel_wiki/tools/catalog.py list --help`。

## 验证与维护

```bash
python3 kernel_wiki/tools/catalog.py check
python3 -B kernel_wiki/aka-cuda-operators-20261030/tools/verify.py
python3 -B kernel_wiki/aka-cuda-operators-20261030/tools/test_metadata.py
```

第一条检查索引同步、相对链接、原始清单和全部文件哈希；后两条执行原库的接口、证据绑定与元数据检查。原库检查器严格禁止 `.DS_Store` 等缓存；Finder 再次生成缓存时需清理该快照中的缓存后重试，不应修改证据文件。以上均不执行 GPU 评测。

新增版本时另建批次并保留来源与验收绑定。修正本 Wiki 的分类说明时，修改 `tools/catalog.py`，再运行 `python3 kernel_wiki/tools/catalog.py render`，审阅并应用其输出的补丁以同步三个派生索引。该命令本身不写文件，也不会刷新原库元数据。

需要从实现提取设计时，进入仓库的 [EXTRACT_SCHEMA.md](../EXTRACT_SCHEMA.md)；产物归入 `schema_wiki/`，并引用本批次的实际源码路径。这次整理交付的是实现与证据资料库。
