**PASS；FIX / DECISION：无。** 四项修正的最少必要含义已完整，定点范围内未发现新增矛盾：

- **GDN L65**：两门独立计算、148 SM 下不共现的限定，以及完整 `segmented.yaml` 排除 `perm`，均与固定依据一致。
- **MoE L28**：无屏障已限定为 grid 级，并保留 CTA 内会合。
- **MoE L72**：已明确 GEMM2 启动前等待侧流清零完成。
- **MoE L74**：已补齐 `min(E, R) * 3 * I * H <= 96_000_000`，与源码选择门一致。

所给增量与 B1→B2 实际差异一致；相关源码/YAML 依据未变。GEMM 仅核摘要，确认字节未变。以下为实际计算、与指定值一致的最终字节绑定，路径相对 `schema_wiki/aka-cuda-operators-20261030/operators/`：

| 文件 | SHA256 |
|---|---|
| `gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md` | `0924ea2d5cfd82b88a50e4df341395c245b3e147b30c1796f01f9738900e052f` |
| `linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md` | `3e907a68cd81cd5a922ec3a6ba85b65498fde56df3bffd0cdb770a18ebc46255` |
| `moe/nvfp4_fused_moe/dispatch.md` | `f0083272058fb0aa3a01d3aa2edd8a1069c1e97c3202e5db14b7476910871513` |

其他全文覆盖接续初次验收，已接受删减不重开。本轮只读固定候选及指定记录，未读取实时项目、写文件、启动子代理或运行项目测试、结构脚本、平台审计。