# 44 个算子总索引

[首页](README.md) · [多维索引](FACETS.md) · [分类与阅读方法](GUIDE.md) · [验证与计时口径](EVIDENCE.md)

由 `tools/catalog.py render` 从原清单和中文分类说明生成。每个名字直达逐题 README；继续进入源码、合同、用例和性能记录。

精度列是主要输入／量化格式，未穷举输出、scale 和累加 dtype；状态精度单列。阶段表示合同用途，不表示阶段下的所有组合都已验收。

## 矩阵乘法 GEMM（15）

### 普通 GEMM（6）

| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |
| --- | --- | --- | --- | --- |
| [bf16_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm/README.md) | BF16 A/B；D = αABᵀ + C，可选 C 与 D 别名 | BF16 | — | REFERENCE；采样验收通过 |
| [bf16_nvfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/README.md) | BF16 × NVFP4 普通 GEMM，原生载荷与缩放布局 | BF16 / NVFP4 | — | AKA；原包认证 |
| [fp8_mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_mxfp4_gemm/README.md) | FP8 × MXFP4 普通 GEMM，含可选 C 累加 | FP8 / MXFP4 | — | REFERENCE；采样验收通过 |
| [fp8_scaled_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/README.md) | 带显式缩放的 FP8 GEMM。接口无 C 输入，写满 m 行并返回调用方的 out，logical_m 只描述输入生成时的行数关系 | FP8 | — | REFERENCE；采样验收通过 |
| [mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp4_gemm/README.md) | MXFP4 普通 GEMM，含可选 C 累加 | MXFP4 | — | AKA；原包认证 |
| [mxfp8_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp8_gemm/README.md) | MXFP8 普通 GEMM，含可选 C 累加 | MXFP8 | — | AKA；原包认证 |

### 分组 GEMM（5）

| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |
| --- | --- | --- | --- | --- |
| [bf16_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/bf16_grouped_gemm/README.md) | BF16 M 分组连续 / PSUM GEMM | BF16 | — | REFERENCE；采样验收通过 |
| [fp8_mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_mxfp4_grouped_gemm/README.md) | FP8 × MXFP4 M 分组连续 / PSUM GEMM | FP8 / MXFP4 | — | REFERENCE；采样验收通过 |
| [fp8_scaled_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/README.md) | 带缩放的 FP8 分组 GEMM，以 indptr 定义分组 | FP8 | — | AKA；原包认证 |
| [mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp4_grouped_gemm/README.md) | MXFP4 M 分组连续 / PSUM GEMM | MXFP4 | — | REFERENCE；采样验收通过 |
| [mxfp8_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp8_grouped_gemm/README.md) | MXFP8 M 分组连续 / PSUM GEMM | MXFP8 | — | REFERENCE；采样验收通过 |

### 带掩码的分组 GEMM（4）

| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |
| --- | --- | --- | --- | --- |
| [bf16_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/README.md) | BF16 专家分组 GEMM。合同只允许写各组有效前缀，当前候选按块写回，尾块可能改写未启用的行 | BF16 | — | AKA；原包认证 |
| [fp8_mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/README.md) | FP8 × MXFP4 专家分组 GEMM，只写有效前缀 | FP8 / MXFP4 | — | AKA；原包认证 |
| [mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp4_grouped_gemm_masked/README.md) | MXFP4 专家分组 GEMM，只写有效前缀 | MXFP4 | — | REFERENCE；采样验收通过 |
| [mxfp8_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp8_grouped_gemm_masked/README.md) | MXFP8 专家分组 GEMM，只写有效前缀 | MXFP8 | — | REFERENCE；采样验收通过 |

## 融合专家计算 MoE（5）

| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |
| --- | --- | --- | --- | --- |
| [bf16_fused_moe](aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/README.md) | 预路由 BF16 专家 FC1 → 激活 → FC2 → 加权归约 | BF16 | — | REFERENCE；采样验收通过 |
| [fp8_mxfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_mxfp4_fused_moe/README.md) | 预路由 FP8 / MXFP4 融合专家计算与加权归约 | FP8 / MXFP4 | — | REFERENCE；采样验收通过 |
| [fp8_scaled_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/README.md) | FP8 块缩放专家 SwiGLU FFN 与加权归约 | FP8 | — | AKA；原包认证 |
| [mxfp8_fused_moe](aka-cuda-operators-20261030/operators/moe/mxfp8_fused_moe/README.md) | 预路由 MXFP8 融合专家计算与加权归约 | MXFP8 | — | REFERENCE；采样验收通过 |
| [nvfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/README.md) | NVFP4 路由 MoE；FC2 前含激活量化 | NVFP4 | — | AKA；原包认证 |

## 多头注意力 FMHA（8）

| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |
| --- | --- | --- | --- | --- |
| [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md) | BF16 Q/K/V → BF16 O；分页 MHA / GQA / MQA | BF16 | prefill / prefix / decode / mtp | AKA；原包认证 |
| [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md) | BF16 Q + FP8 KV；per-tensor 缩放 | BF16 / FP8 | prefill / prefix / decode / mtp | AKA；原包认证 |
| [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md) | FP16 Q/K/V → FP16 O；分页注意力 | FP16 | prefill / prefix / decode / mtp | REFERENCE；采样验收通过 |
| [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md) | FP8 Q/K/V；标量缩放，普通 FMHA 几何 | FP8 | prefill / prefix / decode / mtp | AKA；原包认证 |
| [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md) | BF16 Q + FP8 KV；每 token / head 缩放，O-only | BF16 / FP8 | prefill / prefix / decode / mtp | AKA；原包认证 |
| [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md) | MXFP8 Q/K/V block32；连续布局，无分页 / ragged | MXFP8 | prefill / prefix / decode / mtp | AKA；原包认证 |
| [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md) | FP8 Q + NVFP4 KV；group16 缩放 | FP8 / NVFP4 | prefill / prefix / decode / mtp | REFERENCE；采样验收通过 |
| [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md) | BF16 外部 Q/K/V，内部 NVFP4 QK 与 PV；非因果变长 | BF16 / NVFP4 | prefill | REFERENCE；人工接受性能例外；**待回执** |

## 多头潜在注意力 MLA（4）

| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |
| --- | --- | --- | --- | --- |
| [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md) | 吸收式 FP8 MLA：Q/K D576，V 为 K 前 512 维 | FP8 | prefill / prefix / decode / mtp | REFERENCE；采样验收通过 |
| [mla_expanded_bf16](aka-cuda-operators-20261030/operators/mla/mla_expanded_bf16/README.md) | 展开式 BF16 MLA，Q/K D192、V D128，连续 ragged prefill，Hq=Hkv ∈ {12, 16, 24, 96, 128}，支持因果和非因果模式，可选 attention sinks | BF16 | prefill | AKA；原包认证 |
| [mla_expanded_fp8](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/README.md) | 展开式 FP8 MLA，Q/K D192、V D128，连续 ragged prefill，Hq=Hkv ∈ {12, 24, 32, 96, 128}，支持因果和非因果模式，可选 attention sinks | FP8 | prefill | AKA；原包认证 |
| [mla_nvfp4](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md) | NVFP4 dense MQA：Q/K/V D512，K=V；分页 decode / MTP | NVFP4 | decode / mtp | REFERENCE；采样验收通过 |

## 稀疏注意力（4）

| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |
| --- | --- | --- | --- | --- |
| [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md) | BF16 单 / 双 latent KV 缓存，联合 softmax | BF16 | prefill / decode / mtp | AKA；原包认证 |
| [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md) | FP8 单 / 双 latent KV 缓存，独立 QK / PV descale | FP8 | prefill / decode / mtp | REFERENCE；采样验收通过 |
| [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md) | 物理 token 索引访问分页 FP8 latent KV；D512 / D576 | FP8 | prefill / decode / mtp | REFERENCE；采样验收通过 |
| [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md) | BF16 Q + MXFP8 滑窗 KV + NVFP4 压缩 KV | BF16 / MXFP8 / NVFP4 | decode / mtp | REFERENCE；人工接受性能例外；**待回执** |

## 线性注意力 GDN / KDA（8）

| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |
| --- | --- | --- | --- | --- |
| [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md) | 卷积 → SiLU → GDN 递推 → gated RMSNorm；BF16 检查点 | BF16；state BF16 | decode / mtp | AKA；原包认证 |
| [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md) | 卷积 → SiLU → GDN 递推 → gated RMSNorm；FP32 检查点 | BF16；state FP32 | decode / mtp | REFERENCE；采样验收通过 |
| [gdn_decode_core_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_decode_core_bf16_state_fp32/README.md) | 普通 decode ABI：卷积、GDN 递推与原位 gated RMSNorm | BF16；state FP32 | decode | AKA；原包认证 |
| [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md) | 卷积后 GDN 递推与 gated RMSNorm；BF16 检查点 | BF16；state BF16 | decode / mtp | AKA；原包认证 |
| [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md) | 卷积后 GDN 递推与 gated RMSNorm。合同允许槽间填充 0 到 65536 个 FP32 元素，当前候选要求首地址 16 字节对齐且槽步长为 4 个元素的倍数 | BF16；state FP32 | decode / mtp | REFERENCE；采样验收通过 |
| [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/README.md) | GDN packed prefill；只读 BF16 初态，可选新 FP32 终态 | BF16；state BF16 | prefill | AKA；原包认证 |
| [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md) | GDN packed prefill；只读 FP32 初态，可选新 FP32 终态 | BF16；state FP32 | prefill | REFERENCE；采样验收通过；**待回执** |
| [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md) | KDA 原始门控 prefill；逐通道衰减、BF16 终态与检查点 | BF16；state BF16 | prefill | REFERENCE；采样验收通过 |
