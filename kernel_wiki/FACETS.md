# 多维检索索引

[首页](README.md) · [全部 44 题](CATALOG.md) · [命令行组合筛选](README.md#命令行检索)

同一算子可出现在多个分组；源码只有一份。格式与阶段标签是检索入口，完整 ABI、缩放布局和支持域见每题合同。

## 主要输入与量化格式

### BF16（20）

- [bf16_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm/README.md)
- [bf16_nvfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/README.md)
- [bf16_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/bf16_grouped_gemm/README.md)
- [bf16_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/README.md)
- [bf16_fused_moe](aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/README.md)
- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md)
- [mla_expanded_bf16](aka-cuda-operators-20261030/operators/mla/mla_expanded_bf16/README.md)
- [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)
- [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md)
- [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md)
- [gdn_decode_core_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_decode_core_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md)
- [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md)
- [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/README.md)
- [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md)
- [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)

### FP16（1）

- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)

### FP8（15）

- [fp8_mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_mxfp4_gemm/README.md)
- [fp8_scaled_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/README.md)
- [fp8_mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_mxfp4_grouped_gemm/README.md)
- [fp8_scaled_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/README.md)
- [fp8_mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/README.md)
- [fp8_mxfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_mxfp4_fused_moe/README.md)
- [fp8_scaled_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)
- [mla_expanded_fp8](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/README.md)
- [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)
- [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)

### MXFP8（6）

- [mxfp8_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp8_gemm/README.md)
- [mxfp8_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp8_grouped_gemm/README.md)
- [mxfp8_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp8_grouped_gemm_masked/README.md)
- [mxfp8_fused_moe](aka-cuda-operators-20261030/operators/moe/mxfp8_fused_moe/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)

### MXFP4（7）

- [fp8_mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_mxfp4_gemm/README.md)
- [mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp4_gemm/README.md)
- [fp8_mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_mxfp4_grouped_gemm/README.md)
- [mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp4_grouped_gemm/README.md)
- [fp8_mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/README.md)
- [mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp4_grouped_gemm_masked/README.md)
- [fp8_mxfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_mxfp4_fused_moe/README.md)

### NVFP4（6）

- [bf16_nvfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/README.md)
- [nvfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md)
- [mla_nvfp4](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)

## 递推状态精度

### BF16（4）

- [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md)
- [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md)
- [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/README.md)
- [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)

### FP32（4）

- [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md)
- [gdn_decode_core_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_decode_core_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md)
- [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md)

## 注意力阶段

### prefill（17）

- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)
- [mla_expanded_bf16](aka-cuda-operators-20261030/operators/mla/mla_expanded_bf16/README.md)
- [mla_expanded_fp8](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/README.md)
- [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)
- [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)
- [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)
- [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/README.md)
- [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md)
- [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)

### prefix（8）

- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)

### decode（18）

- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)
- [mla_nvfp4](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md)
- [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)
- [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)
- [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)
- [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md)
- [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md)
- [gdn_decode_core_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_decode_core_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md)
- [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md)

### mtp（17）

- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)
- [mla_nvfp4](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md)
- [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)
- [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)
- [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)
- [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md)
- [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md)
- [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md)

## 实现来源

### AKA（20）

- [bf16_nvfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/README.md)
- [mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp4_gemm/README.md)
- [mxfp8_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp8_gemm/README.md)
- [fp8_scaled_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/README.md)
- [bf16_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/README.md)
- [fp8_mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/README.md)
- [fp8_scaled_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/README.md)
- [nvfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/README.md)
- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [mla_expanded_bf16](aka-cuda-operators-20261030/operators/mla/mla_expanded_bf16/README.md)
- [mla_expanded_fp8](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/README.md)
- [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)
- [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md)
- [gdn_decode_core_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_decode_core_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md)
- [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/README.md)

### REFERENCE（24）

- [bf16_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm/README.md)
- [fp8_mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_mxfp4_gemm/README.md)
- [fp8_scaled_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/README.md)
- [bf16_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/bf16_grouped_gemm/README.md)
- [fp8_mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_mxfp4_grouped_gemm/README.md)
- [mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp4_grouped_gemm/README.md)
- [mxfp8_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp8_grouped_gemm/README.md)
- [mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp4_grouped_gemm_masked/README.md)
- [mxfp8_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp8_grouped_gemm_masked/README.md)
- [bf16_fused_moe](aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/README.md)
- [fp8_mxfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_mxfp4_fused_moe/README.md)
- [mxfp8_fused_moe](aka-cuda-operators-20261030/operators/moe/mxfp8_fused_moe/README.md)
- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)
- [mla_nvfp4](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md)
- [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)
- [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)
- [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md)
- [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md)
- [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)

## 候选验收状态

### 原包认证（20）

- [bf16_nvfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/README.md)
- [mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp4_gemm/README.md)
- [mxfp8_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp8_gemm/README.md)
- [fp8_scaled_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/README.md)
- [bf16_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/README.md)
- [fp8_mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/README.md)
- [fp8_scaled_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/README.md)
- [nvfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/README.md)
- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [mla_expanded_bf16](aka-cuda-operators-20261030/operators/mla/mla_expanded_bf16/README.md)
- [mla_expanded_fp8](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/README.md)
- [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)
- [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md)
- [gdn_decode_core_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_decode_core_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md)
- [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/README.md)

### 采样验收通过（22）

- [bf16_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm/README.md)
- [fp8_mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_mxfp4_gemm/README.md)
- [fp8_scaled_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/README.md)
- [bf16_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/bf16_grouped_gemm/README.md)
- [fp8_mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_mxfp4_grouped_gemm/README.md)
- [mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp4_grouped_gemm/README.md)
- [mxfp8_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp8_grouped_gemm/README.md)
- [mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp4_grouped_gemm_masked/README.md)
- [mxfp8_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp8_grouped_gemm_masked/README.md)
- [bf16_fused_moe](aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/README.md)
- [fp8_mxfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_mxfp4_fused_moe/README.md)
- [mxfp8_fused_moe](aka-cuda-operators-20261030/operators/moe/mxfp8_fused_moe/README.md)
- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)
- [mla_nvfp4](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md)
- [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)
- [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)
- [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md)
- [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md)
- [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)

### 人工接受性能例外（2）

- [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)

## 题目与验收的绑定

### 7 个题目文件逐字节核对（20）

- [bf16_nvfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/README.md)
- [mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp4_gemm/README.md)
- [mxfp8_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp8_gemm/README.md)
- [fp8_scaled_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/README.md)
- [bf16_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/README.md)
- [fp8_mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/README.md)
- [fp8_scaled_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/README.md)
- [nvfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/README.md)
- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [mla_expanded_bf16](aka-cuda-operators-20261030/operators/mla/mla_expanded_bf16/README.md)
- [mla_expanded_fp8](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/README.md)
- [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)
- [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md)
- [gdn_decode_core_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_decode_core_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md)
- [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/README.md)

### 验收核心题目绑定已核对（21）

- [bf16_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm/README.md)
- [fp8_mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_mxfp4_gemm/README.md)
- [fp8_scaled_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/README.md)
- [bf16_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/bf16_grouped_gemm/README.md)
- [fp8_mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_mxfp4_grouped_gemm/README.md)
- [mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp4_grouped_gemm/README.md)
- [mxfp8_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp8_grouped_gemm/README.md)
- [mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp4_grouped_gemm_masked/README.md)
- [mxfp8_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp8_grouped_gemm_masked/README.md)
- [bf16_fused_moe](aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/README.md)
- [fp8_mxfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_mxfp4_fused_moe/README.md)
- [mxfp8_fused_moe](aka-cuda-operators-20261030/operators/moe/mxfp8_fused_moe/README.md)
- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)
- [mla_nvfp4](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md)
- [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)
- [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)
- [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md)
- [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)

### 待最终回执核验（3）

- [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)
- [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md)

## 主要测量口径与对照

### NCU kernel-sum / 生产实现（25）

- [bf16_nvfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/README.md)
- [mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp4_gemm/README.md)
- [mxfp8_gemm](aka-cuda-operators-20261030/operators/gemm/dense/mxfp8_gemm/README.md)
- [fp8_scaled_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/README.md)
- [bf16_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/README.md)
- [fp8_mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/README.md)
- [fp8_scaled_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_scaled_fused_moe/README.md)
- [nvfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/README.md)
- [fmha_bf16](aka-cuda-operators-20261030/operators/fmha/fmha_bf16/README.md)
- [fmha_bf16_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_bf16_fp8/README.md)
- [fmha_fp8](aka-cuda-operators-20261030/operators/fmha/fmha_fp8/README.md)
- [fmha_fp8_per_token_head](aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/README.md)
- [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/README.md)
- [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md)
- [mla_expanded_bf16](aka-cuda-operators-20261030/operators/mla/mla_expanded_bf16/README.md)
- [mla_expanded_fp8](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/README.md)
- [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/README.md)
- [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)
- [gdn_conv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/README.md)
- [gdn_decode_core_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_decode_core_bf16_state_fp32/README.md)
- [gdn_postconv_mtp_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_bf16/README.md)
- [gdn_postconv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md)
- [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/README.md)
- [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md)
- [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)

### CUDA event forward / 历史候选（19）

- [bf16_gemm](aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm/README.md)
- [fp8_mxfp4_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_mxfp4_gemm/README.md)
- [fp8_scaled_gemm](aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/README.md)
- [bf16_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/bf16_grouped_gemm/README.md)
- [fp8_mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/fp8_mxfp4_grouped_gemm/README.md)
- [mxfp4_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp4_grouped_gemm/README.md)
- [mxfp8_grouped_gemm](aka-cuda-operators-20261030/operators/gemm/grouped/mxfp8_grouped_gemm/README.md)
- [mxfp4_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp4_grouped_gemm_masked/README.md)
- [mxfp8_grouped_gemm_masked](aka-cuda-operators-20261030/operators/gemm/grouped_masked/mxfp8_grouped_gemm_masked/README.md)
- [bf16_fused_moe](aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/README.md)
- [fp8_mxfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/fp8_mxfp4_fused_moe/README.md)
- [mxfp8_fused_moe](aka-cuda-operators-20261030/operators/moe/mxfp8_fused_moe/README.md)
- [fmha_fp16](aka-cuda-operators-20261030/operators/fmha/fmha_fp16/README.md)
- [fmha_nvfp4_kv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_kv/README.md)
- [mla_absorbed_fp8](aka-cuda-operators-20261030/operators/mla/mla_absorbed_fp8/README.md)
- [mla_nvfp4](aka-cuda-operators-20261030/operators/mla/mla_nvfp4/README.md)
- [dual_cache_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/README.md)
- [paged_indexed_latent_fp8](aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/README.md)
- [gdn_conv_mtp_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_fp32/README.md)

### 有生产对照缺测项（1）

- [kda_raw_prefill_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)
