# 分类区别与阅读方法

[首页](README.md) · [44 题总索引](CATALOG.md) · [多维索引](FACETS.md) · [验证与计时口径](EVIDENCE.md)

先确定算子做什么、输入已经准备到哪一步，再选格式、布局和实现。下面的解释来自本批次各题 `problem/metadata.json`、`reference.py` 与固定用例；完整支持域与跨字段限制仍以题目合同为准。

## GEMM：先看工作组织，再看精度

| 类型 | 输入和工作边界 | 适合比较什么 |
| --- | --- | --- |
| [普通 GEMM](aka-cuda-operators-20261030/operators/gemm/dense/README.md) | 单次矩阵乘法；各题分别规定输出、缩放、可选 C 与别名 | 不同输入格式、块缩放与普通矩阵乘调度 |
| [分组 GEMM](aka-cuda-operators-20261030/operators/gemm/grouped/README.md) | 多组矩阵乘，连续 / PSUM 或 indptr 分组 | 多组任务的工作分配；不能假定每题使用同一种分组 ABI |
| [带掩码分组 GEMM](aka-cuda-operators-20261030/operators/gemm/grouped_masked/README.md) | 每个专家保留物理容量，合同规定只写各组有效前缀 | 容量、有效工作量及候选的实际写回边界 |

- `bf16_grouped_gemm_masked` 的合同要求尾部存储保持原值，参考实现也只写 `d[e, :masked_m[e]]`。当前候选的三条默认路径按整块写回。当一个专家只启用 1 行时，最后一块里的其他行也可能被改写。输入工具的数值比较视图只取有效前缀。[写区合同](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/problem/metadata.json) · [参考实现](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/problem/reference.py) · [候选写回](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/candidate/solution.py)
- `fp8_scaled_gemm` 的导入元数据提到可选 C 别名。公开接口没有 C 参数，参考实现把结果写入并返回同一个 `out`。提供 `logical_m` 时，任务要求 `m = 4 * ceil(logical_m / 4)`。例如固定用例 `logical_m=1, m=4` 会写满 4 行。[输入范围](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/shape_range.json) · [输入生成](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/input.py) · [参考实现](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/reference.py)

格式覆盖如下。表中的空缺表示本批次没有该组合，不能据此推断硬件不支持。

| 格式系列 | 普通 | 分组 | 带掩码分组 |
| --- | ---: | ---: | ---: |
| BF16 | ✓ | ✓ | ✓ |
| BF16 × NVFP4 | ✓ | — | — |
| FP8 scaled | ✓ | ✓ | — |
| FP8 × MXFP4 | ✓ | ✓ | ✓ |
| MXFP4 | ✓ | ✓ | ✓ |
| MXFP8 | ✓ | ✓ | ✓ |

`FP8`、`MXFP8`、`MXFP4`、`NVFP4` 在这里是不同载荷／缩放约定的检索标签，不能仅凭位宽互换。精确编码、量化位置与参考语义见各题 `PRECISION.md`（若有）、`input.py` 和 `reference.py`。

## MoE：融合边界比单个 GEMM 更大

[5 个 MoE 实现](aka-cuda-operators-20261030/operators/moe/README.md)接收已准备的路由、激活与专家权重，覆盖专家 FC1、激活、FC2 和路由加权归约。已有路由不能被误算成候选自己执行了路由生成；单 rank 计算也不等于多 GPU 通信链。

比较低精度版本时还要看中间量语义。例如 [NVFP4 MoE](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/README.md)明确包含 FC2 前的激活量化；[FP8 scaled MoE](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/README.md)有显式 FP8 中间量约定。不能把这些转换移出计时边界后仍沿用原测量结论。

## Attention：先分普通、潜在与稀疏

| 类别 | 本批次的核心区别 | 阅读时首先确认 |
| --- | --- | --- |
| [FMHA](aka-cuda-operators-20261030/operators/fmha/README.md) | 普通 QK → softmax → PV，不同 Q/K/V 格式与缩放粒度 | MHA / GQA / MQA、head dimension、分页或连续布局、mask、LSE |
| [MLA](aka-cuda-operators-20261030/operators/mla/README.md) | 潜在表示对应不同维度和存储关系 | absorbed D576/V512、expanded D192/V128、NVFP4 D512 不能合并为同一接口 |
| [稀疏注意力](aka-cuda-operators-20261030/operators/sparse_attention/README.md) | 只处理选择后的缓存内容，或联合多个缓存 | 物理索引、实际选中 token / page、滑窗与压缩缓存关系 |

需要特别区分的几组实现：

- **FMHA 的 FP8 KV**：[BF16 Q + FP8 KV](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)使用 per-tensor scales；[per-token/head 版本](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)使用内联 token/head scales。它们的 ABI 和工作量不同。
- **两种 NVFP4 FMHA**：[NVFP4 KV](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)的 Q 是 FP8；[NVFP4 QK/PV](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md)的外部 Q/K/V 是 BF16，内部有显式量化，范围是非因果 packed varlen HD128 MHA，不能继承其他 FMHA 的 decode、分页或 GQA 能力。
- **MXFP8 FMHA**：[QKV MXFP8](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)使用连续布局、原生实例与 workspace；当前不含 paged、ragged 或 sink。
- **MLA 的三种几何**。Absorbed FP8 中，V 与 K 的前 512 维共享。Expanded BF16 和 FP8 都对等长 Q/KV 做 ragged prefill，允许因果和非因果模式，也可选 attention sinks。两题各自规定合法的头数，Q 与 KV 的头数须相等。NVFP4 MLA 使用 dense D512 MQA 做 decode / MTP。[Absorbed FP8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md) · [Expanded BF16 参数域](aka-cuda-operators-20261030/operators/mla/mla_expanded_bf16/problem/shape_range.json) · [Expanded FP8 参数域](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/problem/shape_range.json) · [NVFP4 MLA](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md)
- **双缓存与物理索引**：[dual-cache BF16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)与 [FP8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)对单／双缓存做联合 softmax；[paged indexed FP8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)按物理 token 索引访问；[MXFP8 + NVFP4 混合缓存](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)是滑窗与压缩 KV 的 decode / MTP。

多数注意力合同从已准备的 Q/KV、scale 和索引开始，projection、RoPE、cache writer 或 indexer 在边界外。上下文长度也不等于稀疏算子实际处理的 token 数。阶段标签只帮助找题，不能替代 shape_range 中的相关约束，也不能当作整个动态域的验收证明。

展开式 FP8 MLA 的普通路径按 256 缩放 softmax 权重。首个 tile 建立行最大值，此后只有以 log2 为单位的增幅超过 0.75 才更新。主机从本次调用的长度构造调度表，短行重算用独立的共享内存。下面的校注版与导入原件使用同一行号。[修正注释的源码](annotations/mla_expanded_fp8/solution.py) · [导入原件](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py)

## GDN / KDA：融合边界与状态是两条独立轴

| 入口系列 | 计算边界 | 状态版本 |
| --- | --- | --- |
| `gdn_prefill_*` | packed Q/K/V → GDN prefill；不含卷积与输出归一化 | BF16 / FP32 初态；两题都可返回新分配的 FP32 终态 |
| `gdn_conv_mtp_*` | 四抽头因果卷积 → SiLU → 递推 → gated RMSNorm | BF16 / FP32 递推检查点；还维护卷积历史 |
| `gdn_postconv_mtp_*` | 从卷积后输入开始，含归一化、门控、递推与 gated RMSNorm | BF16 / FP32 递推检查点 |
| `gdn_decode_core_bf16_state_fp32` | 普通 decode 的独立 ABI，含卷积、递推与原位 gated RMSNorm | FP32 状态；不是 MTP 检查点接口 |
| `kda_raw_prefill_state_bf16` | 原始门控、QK L2 归一化、逐通道衰减和递推 | 原位 BF16 终态与可选周期检查点 |

全部入口见 [线性注意力目录](aka-cuda-operators-20261030/operators/linear_attention/README.md)。`state_bf16` 不能被理解为所有中间量、初态和终态都用 BF16；例如 GDN prefill 的输出终态另有约定。decode / MTP 的 accepted-token、null request、状态索引和未写区域也是算法合同的一部分。

`gdn_postconv_mtp_bf16_state_fp32` 的任务允许相邻状态槽之间填充 0 到 65536 个 FP32 元素。候选入口要求首地址按 16 字节对齐，槽步长 `HV*128*128 + state_slot_padding` 必须是 4 的倍数。例如填充 1 个元素符合任务域，但会被这个入口拒绝。[任务范围](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/problem/shape_range.json) · [槽步长构造](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/problem/input.py) · [候选入口检查](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/candidate/launch.cu)

## 每个算子的阅读顺序

1. **逐题 README**：先确认稳定 ID、所选实现来源、测量入口和状态。
2. **合同与输入**：读 `problem/metadata.json` 的 `benchmark_contract`，再读 `input.py`、`shape_cases.json`、`shape_range.json`，核对写入、别名、scratch、容差和参数关系。
3. **数学参考**：读 `problem/reference.py`，确认存储精度、量化和状态递推；`problem/solution.py` 是生产对照，两者职责不同。
4. **选定实现**：从 `candidate/solution.py:Model.forward` 跟踪 dispatch、构建入口和实际 CUDA/C++ 文件，复用时保留整个 `candidate/`。接口清单记录 `prepare_model` 钩子和 forward 别名；不能用一个固定调用模板覆盖所有题。
5. **证据**：先读 `performance/SUMMARY.json` 的主测量、环境、对照身份与逐 shape 状态，再追溯原始认证／验收记录。参见 [证据阅读说明](EVIDENCE.md)。

## 用于 schema 提取

从 [EXTRACT_SCHEMA.md](../EXTRACT_SCHEMA.md)进入既有流程，固定本批次算子、candidate 文件和具体 dispatch；同一算子的多个分支不自动等于同一份 plan。用合同材料建立输入输出边界，用源码证明设计决定，用性能证据描述实际测量范围。将结果保存在 `schema_wiki/` 并填写实际来源引用，不靠两个目录同名推定对应关系。
