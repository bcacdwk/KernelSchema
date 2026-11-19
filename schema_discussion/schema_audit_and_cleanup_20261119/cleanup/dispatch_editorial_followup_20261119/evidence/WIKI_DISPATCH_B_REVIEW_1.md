**FIX（4 项）；DECISION：无。** 主要删减合理，接受删除旧用例、实验成绩、过程记录和 YAML 已承载的重复说明。GEMM 本轮通过；以下仅要求 Carbon 简短修正，不恢复旧段。

1. **[GDN L65](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_B1/schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md:65)：`perm` 与分段不共现的结论失去必要限定。**  
   B0 的推导限定于 148 SM，候选删除该限定，却仍声称 Contract 域内不共现。当前 Contract 未限定 SM 数；按固定源码门公式，`HV=4,N=8,T=16384,sm_count=512` 可同时得到窄臂 `G=2` 和 `perm=true`。这会误排除组合。最少应保留“两门分别计算，完整 segmented YAML 选定无 perm”，或给不共现结论补上有依据的 SM 条件。依据：`root.yaml:37–38,81–85`；固定 `gdn_kernel.cu:3222–3248,3343–3347`。上述是公式反例，不是硬件实测。

2. **[MoE L28](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_B1/schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md:28)：“单核无屏障”扩大了含义。**  
   MEGA 没有 grid 级屏障，但存在 CTA 会合，尤其 ticket 发布前后的同步不可省略。当前说法与 [mega.yaml:121](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_B1/schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mega.yaml:121) 矛盾。最少将“无屏障”限定为 grid 级；无需展开同步流水。

3. **[MoE L72](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_B1/schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md:72)：REDF 丢失侧流清零的完成依赖。**  
   仅保留“侧流并行清零”，不足以保证原子累加不会被清零覆盖。最少补明：**GEMM2 启动前等待侧流清零完成**。依据：固定 `nvfp4_moe.cu:9186–9191,9330–9331`；这是 dispatch 独有局部变化的生命周期条件。

4. **[MoE L74](/tmp/kernelschema-text-run-20261119.cTT0L4/WIKI_CAND_B1/schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md:74)：Staged MEGA 的选择门只剩名称。**  
   “权重乘积门”不能用于判断 `NVFP4_MOE_MEGA=2` 是否命中 rung 91；`R<=64` 和 shared 容量条件并不足够。最少补回 **`min(E,R)*3*I*H <= 96_000_000`**。依据：固定 `solution.py:1423–1425`。保留实验身份即可，无需恢复成绩或推导。

全文覆盖本包三个文件，共 **207 行、39,487 B**；以下 SHA256 均经实际计算，与用户指定值及 manifest 一致：

| 文件（相对 `schema_wiki/aka-cuda-operators-20261030/operators/`） | 覆盖 | SHA256 |
|---|---|---|
| `gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md` | L1–66 | `0924ea2d5cfd82b88a50e4df341395c245b3e147b30c1796f01f9738900e052f` |
| `linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md` | L1–67 | `a1ec49a011f06db918512b8edb50bf78d9df8e30992501991c3391eb37770c2c` |
| `moe/nvfp4_fused_moe/dispatch.md` | L1–74 | `f8563631ea341a1baf7a19bb34d42f3303704eab94907178521589666e047f8e` |

已核对 AGENTS、交接与初审本包意见、Carbon 报告和状态（completed／exit 0）、全部本包增删差异；所给 diff 与固定 B0／候选一致，引用的 YAML／源码与 B0 及 manifest 一致。未写文件、启动子代理、读取实时改稿或重跑源码审计／平台测试。metadata 待维护不计 FIX，已解决的 README 决策未重开。后续仅需复核上述真实修复及其影响。