**PASS。无 FIX / DECISION。**

- **FMHA（L38、74、80）**：已区分独立 `scan2p_gate` 预判与最终两遍执行门，明确融合优先；`span/gspan` 的 **128-key 组数**单位完整。与固定源码的预判、分支及跨度计算一致，无循环依赖。
- **Sparse（L14）**：已完整区分 Python 的 **CUDA 且 ≥32 MiB 才绑定，否则使用私有 32 MiB 缓冲**，与 C++ 的**计划仍超容时另行分配**。identity 缓存说明与调用点、`root.yaml` 一致。
- **MLA**：仅核对字节与摘要，相对修正前完全未变。

实际增量与指定修复 diff 一致；以 **B0** 直接比较所得差异也与本包 diff 一致。三个实际 SHA256 均匹配指定值，可供后续 editorial 链绑定。

文件均位于 `schema_wiki/aka-cuda-operators-20261030/operators/`：

| 文件 | 实际 SHA256 |
|---|---|
| `fmha/fmha_mxfp8_qkv/dispatch.md` | `7072032d6a1bef569b98323f07cde3f590a509984c10c03327f8a9767a57b6a6` |
| `mla/mla_expanded_fp8/dispatch.md` | `35d6ed680f97a90a1928c5ee21e6577a0e86d10f4c8840fd1dc3f470b12de44d` |
| `sparse_attention/dual_cache_latent_bf16/dispatch.md` | `729d7ff1458653e75dae66f3f503f7b12a90cc7324aa03726535b928ee17ef24` |

本轮仅核实际修正及直接影响，其他覆盖接续[初次验收](/tmp/kernelschema-text-run-20261119.cTT0L4/jobs/WIKI_DISPATCH_A_REVIEW_1/final.md)，不重开已接受删减。未读取实时项目目录、写文件、启动子代理或运行项目测试／平台审计。