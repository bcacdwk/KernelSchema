# Template 四案例来源

本轮人工重提取采用 [四题快照](../../kernel_wiki/aka-four-problems-20261019/README.md)，固定引用为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`（tree `67de35d84d464abce36452a3cc6b85fefa07b937`）。该 tree 与最后一次内容更新 `176a4f8` 一致；2026-11-17 从 Git blobs 原样恢复，470 个文件、52,207,886 字节的路径、内容和执行权限均与固定来源一致。

以下 `R` 指仓库路径 `kernel_wiki/aka-four-problems-20261019`；`source_root` 是各题被选实现的完整目录。host 与 kernel 路径相对于该行 `source_root`，contract/reference 路径相对于该题目录，证据路径相对于 `R`。所有路径均继承上述固定提交引用。

| 案例 / source_root | Host selector / 入口 | 核心 kernel | Contract / reference / 输入 | 验证证据 |
| --- | --- | --- | --- | --- |
| fa4_prefill：[`R/fa4_prefill/aka_solution`](../../kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/) | `solution.py:Model.forward` → `flash_attn/cute/interface.py:_flash_attn_fwd`；分支导航 `_hd256_split_heuristic`、`_hd256_fastpath` | `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py:BlackwellFusedMultiHeadAttentionForward`；`flash_attn/cute/flash_fwd_combine.py:FlashAttentionForwardCombine` | [`metadata.json`](../../kernel_wiki/aka-four-problems-20261019/fa4_prefill/metadata.json)、[`reference.py`](../../kernel_wiki/aka-four-problems-20261019/fa4_prefill/reference.py)、`input.py`、`shape_train.json`、`shape_valid.json` | `fa4_prefill/aka_solution/AGATE_VALIDATION.json`；`evidence/fa4_prefill/`；`validation-20261021/results/fa4_prefill/{validate.json,profile.json,ncu.csv}` |
| fa4_decode：[`R/fa4_decode/aka_solution`](../../kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/) | `solution.py:Model.forward` → `qoder_greenfield_decode_attention/operator.py:run`；`_select_policy`、`_compile_kernel` | `qoder_greenfield_decode_attention/gqa_decode_paged.py:GroupedQueryAttentionDecodePaged`；`qoder_greenfield_decode_attention/gqa_decode.py:GroupedQueryAttentionDecode`（含 combine） | [`metadata.json`](../../kernel_wiki/aka-four-problems-20261019/fa4_decode/metadata.json)、[`reference.py`](../../kernel_wiki/aka-four-problems-20261019/fa4_decode/reference.py)、`input.py`、`shape_train.json`、`shape_valid.json` | `fa4_decode/aka_solution/AGATE_VALIDATION.json`；`evidence/fa4_decode/`；`validation-20261021/results/fa4_decode/{validate.json,profile.json,ncu.csv}` |
| gdn_prefill：[`R/gdn_prefill/aka_solution`](../../kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/) | 已验证单文件 `solution.py:Model.forward` → `chunk_gated_delta_rule_sm100`；可读 selector `gdn_prefill.py:get_gdn_launch_metadata` | 已验证 `solution.py:GatedDeltaNetChunkedKernel`；可读来源 `gated_delta_net_chunked.py`、`gated_delta_net_tile_scheduler.py`，构建关系见 `build_solution.py` | [`metadata.json`](../../kernel_wiki/aka-four-problems-20261019/gdn_prefill/metadata.json)、[`reference.py`](../../kernel_wiki/aka-four-problems-20261019/gdn_prefill/reference.py)、`input.py`、`shape_train.json`、`shape_valid.json`；Agate 适配参考在 `solution/agate_reference/` | `gdn_prefill/aka_solution/AGATE_VALIDATION.json`；`evidence/gdn_prefill/`；`validation-20261021/results/gdn_prefill/{validate.json,profile.json,ncu.csv}`；`validation-20261021/results/gdn_agate_eval/summary.json` |
| qsa_prefill：[`R/qsa_prefill/aka_solution`](../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/) | `solution.py:Model.forward` → `qwen38_core_qsa_cutedsl/operator.py:run`，含预处理、主 kernel 提交和 ticket/tbase 生命周期 | `qwen38_core_qsa_cutedsl/qsa_preprocess.py:QSAPreprocessUnion`；`qwen38_core_qsa_cutedsl/qsa_kernel.py:QSAForward` | [`metadata.json`](../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/metadata.json)、[`reference.py`](../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/reference.py)、`input.py`、`shape_train.json`、`shape_valid.json` | `evidence/qsa_prefill/{correctness.json,environment.json,source_manifest.json,kernel_launches.csv}`；`validation-20261021/results/qsa_prefill/{validate.json,profile.json,ncu.csv}` |

选定版本以根 [MANIFEST.json](../../kernel_wiki/aka-four-problems-20261019/MANIFEST.json) 和各题实际 `aka_solution/` 为准：FA4 prefill 为 v37 延续树（包标签 v43，v53 只附未打包记录）；decode 为 v55；GDN 为保留尾块中性 padding 修复的 BF16/M64；QSA 为 BF16 FIT v3 `fc30f74`。FA4 的 NCU production 对照入口为各题 `production_ncu.py`；GDN / QSA 为各题 `solution/solution.py`，与正确性 `reference.py` 分开引用。

必要源码身份核对（SHA-256；路径相对于 `R`）：

| 文件 | SHA-256 |
| --- | --- |
| `fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` | `64d9715dd199f4c843533f365b5a702305805702900f40bf541035fa64d885f1` |
| `fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py` | `ec930c32758ea4f34e2df9fefa2de90f3ef345dc11fa66948fef74ae016fbe92` |
| `gdn_prefill/aka_solution/solution.py` | `e216c76d8c434233cba9042be79500438a522da13cc42090281cc138ae427c69` |
| `qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py` | `f5232531709cea8de34e90a0aa578e8e3d580feddf31ab20cdab63361de80b8c` |

原 `MANIFEST.json` 所列 222 文件的 SHA-256 / 大小逐项通过；它不覆盖后增的全部验证文件，完整恢复范围以 Git tree 为准。四题 `aka_solution/` 下全部 Python 源文件与 `validation-20261021/bundles/<case>/aka_solution/` 字节一致；历史运行绑定另见 [`runtime_source_check.json`](../../kernel_wiki/aka-four-problems-20261019/validation-20261021/runtime_source_check.json)。本轮只做静态恢复和文件核对，不运行 Extract 自动化或 GPU 实验。

44 题主批次 `aka-cuda-operators-20261030` 与本快照不默认可互换；相似算子名不能替代源码、题目合同和证据绑定核对。重提取前的旧 YAML 清单为 FA4 prefill 11、FA4 decode 7、GDN prefill 2、QSA prefill 1，仅作旧分支导航，不预设本轮具体设计数量或选型。
