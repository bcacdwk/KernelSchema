# Sparse 正常交付验收

**旧 Sparse 已完成 P6并正式交付。** 发布提交时间为 **2026-11-19 07:08:26 UTC / 北京时间15:08:26**。产物目录：[dual_cache_latent_bf16](../../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16)。

本次工具修复后只新增了一次正常 Qwen/max P6 Reviewer 调用；没有前置 P3/P5维护或额外语义审计。四份 YAML 与目录沿用原 `66ad…` 候选字节，没有重写已有设计。

| 验收对象 | 实际结果 |
| --- | --- |
| run | `8f0844f9-f279-4781-a19d-0216b7fc4ef5`，`ready / P6 / published=true` |
| 正式文件 | `root.yaml`、`split_kv.yaml`、`tcgen05.yaml`、`tcgen05_packed.yaml`、`dispatch.md` |
| 最终 checkpoint | `66ad31f314d7ca7e894527d9641bbf3daad59f4ba35278e71e7841dd5e3901da` |
| 新 P6批准 | attempt `a0af816f-b0af-48b9-a525-fc94c90817d7`；真实 `approve`，`changed_files=[]`，原生成功、退出0、终止已确认 |
| P6 receipt摘要 | `f75534e6505751be1918d7787cf0de0305ef68d01793dc065e72082767ab145f` |
| 执行绑定 | `aa2fe99b295fb6d957cfa025cced5923089b025c6e440469fafa34a60131259c` |
| 调用账 | 原47次完整保留，新增1次正常P6，共48次；P6 calls从5到6，无退款或额度追加 |
| 结清 | active/unsettled/unconfirmed均为空，全部48条usage已settled；Runner与CLI退出，writer锁已释放 |

本轮28项交付证据核对全部通过，精简结果和逐文件完整SHA-256见 [DELIVERY_SPARSE.json](DELIVERY_SPARSE.json)。实际发布目录严格只有上述五份正文和 `.extraction/receipt.json`；五份正文摘要与最终 checkpoint、P6批准、Runner validation、P6只读workspace和交付receipt完全一致，发布commit也对应同一checkpoint/目标目录。旧失败 `aa4176a8…` 未回填receipt或采纳为批准。

来源仍是原固定输入 `e5db05dc…`，源commit `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`。原输入锁、源码/语义规则快照及执行器绑定链通过完整性核验；历史调用保留原执行身份，最后P6属于修复执行器 `555d893e…`，原执行器 `b8a1d5e…` 列在交付receipt的 `executor_history`。原P1–P4批准、原47个attempt字节身份、原accepted_receipts和usage账均保留。

没有重复运行大检查，直接核对本次正常Runner已有的完整Template/P6检查证据：`exit_code=0`，四节点全部检查，`excluded_nodes=[]`、`diagnostics=[]`，发布检查策略为 `skip-hardware`。对最终P6单次调用另外执行了只读 `audit --attempt ... --verify`，结果为 **verified**、errors/unavailable均为空；原始结果在 [SPARSE_FINAL_ATTEMPT_AUDIT.json](recovery_reassessment/SPARSE_FINAL_ATTEMPT_AUDIT.json)。

**待用户决定的标记已经伴随正式交付**，位置为 [.extraction/receipt.json](../../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/.extraction/receipt.json) 的 `continuation_audit[0]`，状态 `pending_user_decision`。它准确包含run/算子、P6原失败attempt、工具问题、沿用的checkpoint与阶段/节点批准，以及两项待审计点：

- `p6-retained-scope-inventory` 工具误拒及故障处审批适用性的潜在独立审计需求；旧失败approve保持其原失败事实。
- root 的208寄存器/`without spilling`表述疑点。208有64+16+128个主数组32位槽小计依据，不能据此证明实际总寄存器或无spill；原正文此次未改，也没有声称疑点已修复。

本轮验收只证明交付与记录的一致性。最终P6单次证据核验verified，不代表对全部48次历史原始日志进行了新的全面审计；生命周期等待边界仍有partial历史，纯模型耗时等不可得项保持未知。没有追加模型审计、kernel/GPU正确性或性能测试，也没有扩展固定来源之外的事实核查。额外独立检查或重来按用户决定后再执行，P6通过不等于这些待查点已完成独立审计。

本代理负责的旧 Sparse 续接与正常交付验收已完成；RUNS/issue索引由主代理收口，不把一次交付完成等同广义续接议题全部resolved。
