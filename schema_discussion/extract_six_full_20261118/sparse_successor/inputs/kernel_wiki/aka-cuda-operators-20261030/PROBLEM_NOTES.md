# 三题待核验备注

以下三题已配齐 problem，接口与历史题库一致；现有包缺少最新版原始验证回执，尚不能完成最终题目哈希确认。按用户决定带标记保留。

- [fmha_nvfp4_qkpv](operators/fmha/fmha_nvfp4_qkpv/README.md)
- [gdn_prefill_bf16_state_fp32](operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md)
- [paged_latent_mxfp8_nvfp4](operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md)

FMHA QKPV 与 paged latent 原有的人工性能例外保留在各自 USER_ACCEPTANCE.md、ACCEPTANCE.json 和统一性能摘要中。

其余 21 题核对到验收使用的输入、reference、合同和固定用例；shape_range／roofline 未记录独立的最新版哈希。20 题 AKA 的七个 problem 文件均逐字节核验。详见 [PROBLEM_AUDIT.json](PROBLEM_AUDIT.json)。
