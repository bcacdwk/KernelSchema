# 融合专家计算 MoE：5 题

融合专家 FC1、激活、FC2 与路由加权归并；独立 grouped GEMM 位于 GEMM 分类。

包含 2 个 AKA 解、3 个参考实现。各题保留完整 candidate、problem 和 performance。

[返回题库首页](../../README.md) · [三题备注](../../PROBLEM_NOTES.md)

| 题目 | 所选解 | problem |
|---|---|---|
| [bf16_fused_moe](bf16_fused_moe/README.md) | 参考实现 | 已核对 |
| [fp8_mxfp4_fused_moe](fp8_mxfp4_fused_moe/README.md) | 参考实现 | 已核对 |
| [fp8_scaled_fused_moe](fp8_scaled_fused_moe/README.md) | AKA 的解 | 已核对 |
| [mxfp8_fused_moe](mxfp8_fused_moe/README.md) | 参考实现 | 已核对 |
| [nvfp4_fused_moe](nvfp4_fused_moe/README.md) | AKA 的解 | 已核对 |
