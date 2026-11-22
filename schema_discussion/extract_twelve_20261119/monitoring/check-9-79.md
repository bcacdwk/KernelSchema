# 巡检 79

采样与完成时间为 2026-11-20 13:15:03 北京时间（UTC+8）；`clock.committed=true`，下次到期为 13:45:03。

已交付 4/12；其余 3 项 P6、5 项 P5。GEMM 的 `P5:split_k_reduce` 回执缺绑定实现入口，节点停止，健康兄弟 `pair_two_pass` 仍在途；采样共 10 个在途调用。

新的 xhigh 子代理初核该 scope 为 reviewer、4/32 次调用，invalid_receipts=3、receipt_repairs=3；此前引用第 276 行越界已被新的入口缺失诊断替代。Supervisor 委派按实际绑定证据核查，符合条件后仅允许一次 `resume --scope P5:split_k_reduce`，保留健康兄弟、原身份、已有稿件、批准与账目，不做整 run 恢复或次数追加。

详细证据与最终处理见 [GEMM scope 恢复目录](../recovery/gemm_p5_split_k_anchor/)。

核验确认原稿的尾部引用已改为 272–275；绑定 `solution.py::Model.forward` 的证据只标 host，implementation 引用均在 kernels.cu，属于真实类别缺项。P5 不适用 P2/P3/P4 的 handoff 1200 字符限制，因此未把该回执较长的 handoff 当作后置故障。

一次正式 scoped 恢复请求 `5c214e5e-794d-4078-a0ec-4edd2c2d764b` 已消费，新 reviewer `2ec6e8ac-f534-4c44-8e19-7a87aa84a390` 在原会话启动并收到入口诊断；该范围为 4→5/32，原补正计数、输入、实例、批准、稿件、checkpoint 和账目保留，健康兄弟继续运行。新结果尚未返回。

13:22:07 的[恢复后复核](check-9-79-after-recovery.json) 确认已交付 4 项，其余 3 项 P6、5 项 P5，共 11 个在途调用。新增 FP8 FMHA `P5:decode_split_kv` 的入口引用自动补正正在由 reviewer 执行，6/32 次、进程存活，无控制屏障；不另发恢复请求。
