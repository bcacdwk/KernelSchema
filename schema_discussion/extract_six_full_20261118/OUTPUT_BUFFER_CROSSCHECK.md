# 六算子输出目的地：同类只读核对

结论：除已提交维护的 Sparse 外，新增 **2 个算子、3 个调用者目的地输入缺项**：MoE 的 `output`，以及 FMHA 的 `o_tensor`、`lse_tensor`。Grouped GEMM、MLA、GDN 已分别登记传入目的地和写回结果；本范围内没有新增问题。

本次仅核对已落地 root 的 Contract Input、Output、Compute，以及固定来源中的 metadata、调用入口、输入构造。以读取时各 run 的 live `state.checkpoint` 为准，结束前复读确认以下五个 checkpoint 未变；未沿用 RUNS.json 中较旧的 checkpoint。没有审阅 Design 或其他 ABI 完整性。

| 算子 | 调用者目的地 → 结果登记 | 判定及主要证据 |
| --- | --- | --- |
| fp8_scaled_grouped_gemm | `Input.out_buffer` → `Output.out` | 已登记。root:18、25、45–46 明确同对象返回与全量覆写；metadata:14–20、input.py:67–74、reference.py:26–31 确认调用者提供 `out`。无新增。 |
| nvfp4_fused_moe | 缺 Input → `Output.output` | **新增缺项**。root:14–49 没有该目的地的输入角色，:50、134 已说明调用者对象全量覆写并放入单元素 list 返回。metadata:14–23、input.py:112、116–122、reference.py:96–101 明确传入并写回 `output`。 |
| fmha_mxfp8_qkv | 缺两个 Input → `Output.o_tensor`、`Output.lse_tensor` | **新增缺项**。root:16–39 未登记两个目的地输入；:41–42、97–101、137–139 说明就地结果及可选 LSE。metadata:15–27、input.py:176–190、reference.py:120–127 确认两个对象由当前调用者提供，`lse_tensor` 可为 None。 |
| mla_expanded_fp8 | `Input.out`、`Input.lse` → `Output.attention_output`、`Output.lse_log2` | 已登记。root:34–35、45–46、110–121 说明映射、写回与返回身份；metadata:14–26、input.py:101、113、129、reference.py:78–92 对应。`lse` 的 nullable/default 已登记。无新增。 |
| gdn_prefill_bf16_state_bf16 | `Input.core_attn_out` → `Output.core_attn_out_updated` | 已登记。root:26、30、72–73 明确可选缓冲前缀覆写、余量保留，另两个返回值新分配；metadata:14–18、input.py:242–246、reference.py:35–41 对应。nullable/default 已登记。无新增。 |
| dual_cache_latent_bf16 | 已由独立分流确认并提交 P1 维护 | 本次不重复列项；见 [OUTPUT_BUFFER_TRIAGE.md](OUTPUT_BUFFER_TRIAGE.md) 与 [SPARSE_OUTPUT_BUFFER_MAINTENANCE.json](SPARSE_OUTPUT_BUFFER_MAINTENANCE.json)。 |

表中 `root` 指下表指定 checkpoint 的 `artifacts/root.yaml`；metadata、input.py、reference.py 分别指该算子固定 source 目录的 `problem/metadata.json`、`problem/input.py`、`problem/reference.py`。所有固定来源为 commit `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`。

## 两个新增项的最小范围

冻结 DOC1:43 要求 Input 登记“调用者提供的全部对象，包括可缺省的输入”；:44 要求原地写回结果也登记为 Output。两个新增项都已有完整的写回散文，但没有输入对象条目，不能因旧内容不参与数学计算豁免登记。DOC1:107 要求契约内名称唯一，可用不同的输入名在 meaning/Compute 中绑定源码参数，保留现有输出 ID。

1. **MoE**：补 `Input.output_buffer`（示例名）映射源码 `output`，shape 为 `[token_count, hidden_dim]`、dtype 为 BF16；只同步 `Output.output.meaning` 和现有 `Routing reduction and output` 步骤中的目的地绑定。candidate/solution.py:1023–1033 的当前入口确实接收 `output`；problem/input.py:112 真正提供该对象。reference.py:100 直接 `output.copy_`，并无 None 分配分支，不能仅凭形参 `output=None` 扩充本任务有效域或机械增加 nullable/default。
2. **FMHA**：补 `Input.o_buffer`、`Input.lse_buffer`（示例名），映射源码 `o_tensor`、`lse_tensor`，沿用现有结果的 shape/dtype；LSE 目的地需表达已有的 `nullable: true`、`default: None`，只同步相应输出 meaning 及已有 LSE/写回步骤。candidate/solution.py:213–219、254–259 和 reference.py:120–127 都显示 O 是必传对象、LSE 默认 None，调用返回 None 而直接写入对象；input.py:176–179 提供 O 并按 return_lse 提供 LSE 或 None。这些是当前任务入口，不是下游库参数。

上述是候选的 P1 最小语义范围，未创建维护请求，也不代替独立 reviewer 判断。

## 已登记项的旧值、空值及缺省核对

Grouped GEMM 的 `out_buffer`、MLA 的 `out/lse`、GDN 的 `core_attn_out` 都没有显式 range。按 DOC1:60、200，其非 None 浮点内容默认有限。固定输入构造的目的地初态分别为：

- Grouped GEMM：input.py:21–24 的 guarded 使用 `-17`，:67 使用该 helper。
- MLA：input.py:53–58 的 guarded 使用 `-17`，:101 构造 out 及可选 lse。
- GDN：input.py:182–185 的 guarded 使用传入值，:242 给 core 缓冲传入 `21`。

这些实际初态均在默认有限值域内，未发现本范围所需的 NaN/Inf 反证，因此没有新列 range 问题；也不能反过来把 `-17` 或 `21` 收窄成唯一允许初态。MoE 与 FMHA 的 guarded 同样使用 `-17`（分别 input.py:48–51、153–157），本次新增条目不需要借用 Sparse 的 NaN 事实来判断。

Grouped GEMM 和 MLA 的 `out` 在 candidate 签名中虽默认 None、并有实现分配路径，但各自固定 reference 分别在 reference.py:30、85–89 直接依赖调用者对象，输入构造也确实传入该对象。仅凭较宽的 candidate 签名不足以确认任务契约必须允许缺省；本次不额外列 nullable/default 问题。MLA 的 LSE 与 GDN 的 core 缓冲已有真实 None 分支，现有条目已表达。未把实现内部新分配的输出或 scratch 当作缺漏输入。

## 读取版本定位

根目录：`.schema_extract/v2`。root 路径格式为 `runs/<run_id>/checkpoints/<checkpoint>/artifacts/root.yaml`。source 路径格式为 `snapshots/<snapshot>/source/kernel_wiki/aka-cuda-operators-20261030/operators/<source_suffix>/`。

| 算子 | run_id | checkpoint |
| --- | --- | --- |
| Grouped GEMM | `06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31` | `8136137bff776bd53d021e0c469b7b4361c63257260a7f7feb6573ad2adc3f4f` |
| MoE | `74bcac91-808a-41b2-aab4-85143935ab40` | `838a2961c90b2ef032069fda83b77033550a7296e80d0aa3840cde234fde5139` |
| FMHA | `fb8b3758-92d7-4093-aa18-c2e5028f0a43` | `a9c58e93669010472a46c8d3fbda02f69aa31daee2d450c44b9e2e619014bd09` |
| MLA | `0d78eac5-1e45-4ac0-9725-ab885b5c52dc` | `0c9da892f6fbb8970a66dd662ee6d3b5b6c3188d641b6b862b6e963829d627eb` |
| GDN | `68edadab-4371-402c-8acf-f73eb5ec9f92` | `a714dc12bf8e132093adb0468566a6cf4b9aad824b94ef3e1364395cab0a26b5` |

| 算子 | snapshot | source_suffix |
| --- | --- | --- |
| Grouped GEMM | `39e60da3738fc70943dd784d96dfa8bf4af127c85c971565aa336070a2a41fb6` | `gemm/grouped/fp8_scaled_grouped_gemm` |
| MoE | `adcbd97155004fe0f7ad81f8bc907c0fb57e5c0d4ab26b059d304c727aff1bcc` | `moe/nvfp4_fused_moe` |
| FMHA | `a6e39e6f3a33bb2e40c8cc754837da6fb2a7b4abb62d5aca96a128f4647c839b` | `fmha/fmha_mxfp8_qkv` |
| MLA | `ea6a9a93e0a27a072e7fd06a8f18fe89046186aea865171dced4a3afc3ceb484` | `mla/mla_expanded_fp8` |
| GDN | `0bc296b1b24a767d26095368d83e942ec8c1c2771971ac28e6d7fc9371342786` | `linear_attention/gdn_prefill_bf16_state_bf16` |

DOC1 读取自 MoE snapshot 的 `rules/schema_docs/1.contract.algorithm_requirement.md`；其冻结文件摘要与其他五算子 inputs.lock 中相同。此次只新增本讨论记录，没有修改源码、规范、运行状态、产物或快照，没有派发阶段模型、维护/恢复，也没有运行 GPU 或其他测试。
