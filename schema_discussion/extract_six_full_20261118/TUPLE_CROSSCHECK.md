2026-11-18：六算子 `python_none` 同类问题只读核对

结论：**无新增同类问题，也无须向既有维护增加字段。** 六个当前已落地 checkpoint 的 Contract 共 128 个 Input、10 个 Output；其中 31 个 Input 使用 `dtype: python_none`，Output 无此类型。唯一有明确非 None 对象证据的条目仍为 MLA `Input.sage_attn_sfs`，已纳入 `tuple-abi-input-typing` 维护。

本次从真实 runtime `.schema_extract/v2/runs/<run>/state.json` 的 `checkpoint` 定位产物，不采用 `RUNS.json` 中可能滞后的 checkpoint。下表固定本次读取到的六个 checkpoint；只核对其中 `CONTRACT.Algorithm_requirement.Input/Output`，未把在途 workspace、Design 或旧归档当作当前获批内容。MLA 读取时 P1 已进入维护 reviewing，本次 checkpoint 的 root 内容仍与既有获批 root 相同（SHA-256 `eb60bb4bae117a4c6b72d082d943c7099778827c1abf3e00098bc033ff44d9e0`），本报告不表示维护已完成或已获新批准。

| operator | 字段 | 是否同类 | 具体证据 | 是否需维护 |
| --- | --- | --- | --- | --- |
| `fp8_scaled_grouped_gemm` | Input/Output 全部 | 否 | root Input 12 项、Output 1 项，均无 `python_none`；`scale_granularity_mnk` 使用 `python_int`，`out_dtype` 使用 `torch.dtype`。 | 无新增 |
| `nvfp4_fused_moe` | 10 个 `python_none` Input，详见下列清单 | 未发现 | root 第 24–46 行相关条目的 meaning 均表述缺席或 None；固定 `problem/input.py:112` 对这 10 个字段逐一传入单个 `None`。真正的 tuple `topk_ids=(ids, scores)` 在 root 第 13 行登记为 `tuple(int32, fp32)`，没有借用 `python_none`。 | 无新增 |
| `fmha_mxfp8_qkv` | 12 个 `python_none` Input，详见下列清单 | 未发现 | root 第 26–37 行的 default 均为单个 `None`；固定 `problem/input.py:179–190` 中 payload 未设置这些键，返回接口以 `.get` 取单个 `None`；固定 `candidate/solution.py:220–237`、`problem/reference.py:120` 对应签名默认亦为单个 `None`，无 tuple/list 事实与当前字段冲突。 | 无新增 |
| `mla_expanded_fp8` | `Input.sage_attn_sfs` | **是，既有问题** | root 第 33 行同时写 `dtype: python_none` 和四元 tuple 的 meaning/default；固定 `problem/input.py:129`、`candidate/solution.py:2105`、`problem/reference.py:78` 均明确 `(None, None, None, None)`。DOC1 第 210 行将 `python_none` 定义为仅取单个 Python `None`。 | 已登记；沿用既有维护 |
| `mla_expanded_fp8` | 其余 5 个 `python_none` Input；相邻 `num_elts_per_sage_attn_blk` | 未发现新增同类问题 | 其余 5 项 root 第 30、36–39 行均为单个 None；固定 `problem/input.py:106–129`、`candidate/solution.py:2102,2108–2111`、`problem/reference.py:78` 未提供非 None 容器证据。相邻整数 tuple 是 `python_int`、四个逻辑元素，已在原维护核对范围，不能仅凭其生成值或签名默认值要求固定全零有效域。 | 不增字段、不追加收窄结论 |
| `dual_cache_latent_bf16` | `Input.swa_topk_lens`、`extra_sparse_indices`、`extra_sparse_topk_lens` | 未发现 | root 第 26–28 行均为单个 None；固定 `problem/input.py:119–120` 传入 None；固定 `problem/reference.py:29` 明确断言三项均为 None，`candidate/solution.py:229–230` 同样断言。 | 无新增 |
| `gdn_prefill_bf16_state_bf16` | Input/Output 全部 | 否 | root Input 12 项、Output 3 项，均无 `python_none`。 | 无新增 |

`python_none` Input 完整清单（所有 Output 均无）：

- GEMM：0 项。
- MOE：`routing_bias`、`gemm1_bias`、`gemm2_bias`、`n_group`、`topk_group`、`per_token_scale`、`gemm1_lora_delta`、`valid_hidden_size`、`valid_intermediate_size`、`num_fused_shared_experts`。
- FMHA：`sinks`、`seq_q_lens`、`current_stream`、`amax_o`、`descale_q`、`descale_k`、`descale_v`、`scale_o`、`block_table`、`block_table_v`、`gate`、`sf_o`。
- MLA：`skip_softmax_threshold_scale_factor`、`sage_attn_sfs`、`q_seq_lens_cpu`、`kv_seq_lens_cpu`、`use_fp16_softmax`、`uses_spcompress`。
- Dual-cache：`swa_topk_lens`、`extra_sparse_indices`、`extra_sparse_topk_lens`。
- GDN：0 项。

各 checkpoint 的根文件均位于 `.schema_extract/v2/runs/<run>/checkpoints/<checkpoint>/artifacts/root.yaml`：

| operator | run | 本次已读取 checkpoint |
| --- | --- | --- |
| `fp8_scaled_grouped_gemm` | `06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31` | `3ca6f6eba61233abecdcb6774b7bc2bff4a1b7d4fbac7bb90d2bb1646057406d` |
| `nvfp4_fused_moe` | `74bcac91-808a-41b2-aab4-85143935ab40` | `a09eaa101c3fbd1845763ef490974ce2965bb28522c7ac70a668f26bbe36c0a5` |
| `fmha_mxfp8_qkv` | `fb8b3758-92d7-4093-aa18-c2e5028f0a43` | `03dad136e73c77635cda2130dd51926304e5eea5722d5619920204e63a87ba88` |
| `mla_expanded_fp8` | `0d78eac5-1e45-4ac0-9725-ab885b5c52dc` | `a2f98a28b9f672c052f1d9ca504a4a5d45121df903054c22fbb935d718e3c0c6` |
| `dual_cache_latent_bf16` | `8f0844f9-f279-4781-a19d-0216b7fc4ef5` | `7f8d2a9587de3445338dff53bbe8bfa50764ffcd34cd5047200ed793210f7c7a` |
| `gdn_prefill_bf16_state_bf16` | `68edadab-4371-402c-8acf-f73eb5ec9f92` | `504bc36528cb96acc8ba405bf640c12e7f95819b754f7418fd36485d1504e4f2` |

源码核对均使用各 run `inputs.lock.json` 的 `source.bindings`，以 `git show 24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2:<绑定路径>` 读取固定提交。表中 `problem/...` 与 `candidate/...` 均相对于该 operator 的 `kernel_wiki/aka-cuda-operators-20261030/operators/...` 目录。

“未发现”只回答当前字段是否把已有非 None 对象写为 `python_none`，不以生成样本或默认值证明完整有效域，也不审核其他域约束。按 DOC1 第 58、71 行，dtype 为开放取值，有明确定义的非示例类型不因此非法。本次只新增此讨论报告，未修改源码、规则、状态、产物、快照或维护请求，未触发模型调用。
