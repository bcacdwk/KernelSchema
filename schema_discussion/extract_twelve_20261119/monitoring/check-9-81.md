# 巡检 81

采样与完成时间为 2026-11-20 14:16:04 北京时间（UTC+8）；`clock.committed=true`，下次到期为 14:46:04。

已交付 5/12，本次新增混合精度 Sparse，包含 root 与 tcgen05_tmem。核实 ready/published、全部调用结清、回执等于发布证明、run/checkpoint 匹配，以及两个 YAML 和 dispatch 的实际 SHA，见 [交付核验](../deliveries/paged_latent_mxfp8_nvfp4.json)。正式 dispatch 保留性能用户例外、超限 shape、比较器/逐输出验证回执缺口及采样覆盖限制；未将来源采样当作全域证明。其“本次未运行 GPU”等过程措辞另记为后续文案清理观察，本次未改写已核验字节。

其余 3 项 P6、4 项 P5，共 7 个在途调用。FP8 FMHA 的 `P5:tcgen05_prefill` 缺实现入口，reviewer 在自动补正，4/32 次；FP8 MoE 的 `split_k_decode` 回执行号 575 超过实际 574 行，采样时在补正。其旧 reviewer 在首次定向读取中刚结束，随后只读 inspect 确认已进入新 worker 修订，6/32 次、revisions=1，无当前诊断，不能把这一结算过渡误作停止后重复恢复。

相关状态见 [定向 inspect](check-9-81-inspect.json)。无用户控制或未确认终止，本次保持已有调用与门禁。
