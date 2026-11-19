**FIX：2 项；DECISION：无。** 其余删减接受，初审交接的六项事实修正已落实；无需恢复旧实验记录、论证或逐字段修改流水账。

1. **FMHA：补清 scan 预判与执行门的区别。**  
   [dispatch.md:74](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_1/schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md:74) 用未定义的 `scan2p` 选择 160/48 阈值，L80 只保留了含 `!scan_fused` 的最终两遍执行条件。二者不能替用，否则形成循环，并可能误分 `48 < span <= 160` 的融合路径。  
   **最少保留含义与位置：** 在 L74 定义独立预判 `T >= 128 && rows_total <= 512 && !has_window`；融合优先，未融合时才执行两遍。同时补明 `span/gspan` 是最宽 CTA 实际遍历的 **128-key 组数**。依据固定 [fmha_mxfp8_qkv.cu:780](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_1/kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu:780)，以及该文件 L348–363、1640–1653。

2. **Sparse：修正 workspace 替换条件与执行层。**  
   [dispatch.md:14](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_1/schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md:14) 将缺失或不足概括为 C++ 按计划需求替换，遗漏实际的 **32 MiB 绑定门**。小于 32 MiB、即使足够容纳计划的缓冲，也会先被 Python 替换，影响调用方对缓冲复用及私有分配的判断。  
   **最少保留含义与位置：** 在本行区分“CUDA 且至少 32 MiB 才绑定，否则替换私有 32 MiB”与“C++ 在计划仍超容时再分配”。依据固定 [solution.py:152](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_1/kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/solution.py:152)、`dsv4_kernel.cu:6419–6423`，以及同目录 schema `root.yaml:349–353`。

以上交 Carbon 写简短措辞，不默认恢复原段。

全文覆盖本包三份 dispatch，共 **227 行、36,761 B**；相对 B0 差异全部核对，提供的 diff 与直接比较一致。实际 SHA256 与指定值及 manifest 均一致：

| 文件（位于 `schema_wiki/aka-cuda-operators-20261030/operators/`） | SHA256 |
|---|---|
| `fmha/fmha_mxfp8_qkv/dispatch.md` | `7794b8c79120b53a36b9daba08dba22cd4f5df4a8efe1a83339a9893cac8ec2d` |
| `mla/mla_expanded_fp8/dispatch.md` | `35d6ed680f97a90a1928c5ee21e6577a0e86d10f4c8840fd1dc3f470b12de44d` |
| `sparse_attention/dual_cache_latent_bf16/dispatch.md` | `54a89f4c0168b85455c1bb747971066d2e12c2b36608d82b787a5cce045e806f` |

已读 AGENTS.md、Carbon 报告及状态（completed，exit 0）、初审交接与相关事实依据。仅作固定文本及必要源码/YAML核对；未写文件、启动子代理、读取实时改稿或重跑平台测试。metadata 摘要待后续机械维护，不列 FIX；源 README 保全决策未重开。后续仅需复核上述修复及其影响。