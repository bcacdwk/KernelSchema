# FMHA：用户授权 xhigh 定点小修后交付

2026-11-19。用户在获知本次 P6 回执问题后明确要求：

> 这种小问题，找个 xhigh 修了就行。不要再打回去继续了。

本次执行范围为原 P6 已定位的小问题：`split_kv.yaml` 引用末行多一行；root 与 split_kv 的 QK scale 归约轴措辞把每32个通道写成每32个keys。原 P6 返回 `reopen`，要求回P3修root再回P6修分支，但回执在证据坐标校验处被拒收。这个历史结果保留，不能改写成原模型已经批准。

按用户此次新指令，由新内部子代理 `fmha_minor_closeout_xhigh`（实际 spawn 配置 `gpt-6-astra / xhigh`）从最新候选定点修正，执行必要的只读结构检查，并以 `completion_basis=user_authorized_minor_fix` 交付。只改这两份设计中属于同一错误的轴词及必要 Version，保留公式、Contract、其他轴、mma_sync 和 dispatch 字节。不再启动 P3/P5/P6 模型调用，不新增广泛语义审计。

交付凭据明确记录原 run/checkpoint/失败 P6、内部子代理身份、具体差异、证据坐标纠正、校验结果和本次用户授权。原自动 run 的状态、回执、批准和47次调用账不伪改；本次交付不冒称正常 Runner 获得了新的独立 P6 批准。

既有 `continuation_audit` 及其他待查点随交付保留，完成后仍由用户决定是否额外检查或重来。其他 issue 的后续修改继续等待用户指挥。
