# 多头注意力 FMHA：8 题

常规多头注意力及 BF16、FP16、FP8、MXFP8、NVFP4 精度变体。

包含 5 个 AKA 解、3 个参考实现。各题保留完整 candidate、problem 和 performance。

[返回题库首页](../../README.md) · [三题备注](../../PROBLEM_NOTES.md)

| 题目 | 所选解 | problem |
|---|---|---|
| [fmha_bf16](fmha_bf16/README.md) | AKA 的解 | 已核对 |
| [fmha_bf16_fp8](fmha_bf16_fp8/README.md) | AKA 的解 | 已核对 |
| [fmha_fp16](fmha_fp16/README.md) | 参考实现 | 已核对 |
| [fmha_fp8](fmha_fp8/README.md) | AKA 的解 | 已核对 |
| [fmha_fp8_per_token_head](fmha_fp8_per_token_head/README.md) | AKA 的解 | 已核对 |
| [fmha_mxfp8_qkv](fmha_mxfp8_qkv/README.md) | AKA 的解 | 已核对 |
| [fmha_nvfp4_kv](fmha_nvfp4_kv/README.md) | 参考实现 | 已核对 |
| [fmha_nvfp4_qkpv](fmha_nvfp4_qkpv/README.md) | 参考实现 | 待最终回执核验 |
