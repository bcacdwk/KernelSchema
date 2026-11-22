# 巡检 65

采样与完成时间为 2026-11-20 06:09:15 北京时间（UTC+8）；`clock.committed=true`，下次到期为 06:39:15。

采样时 3 项 P4、9 项 P3，其中 masked GEMM 和分页索引 Sparse 停于 P3/invalid_receipt；在途调用共 10 个。absorbed MLA 的 P3 证据索引问题仍在自动补正，已用 2/32 次调用；[定向 inspect](check-9-65-inspect.json) 确认其 Runner 和 worker 存活，无控制屏障或未确认终止。

两个停止任务分别由新的 xhigh 子代理核对后按原 run 受控恢复：

| 算子 | 根因 | 原 P3 用量 | 其他已发现、尚未进入本次诊断的问题 |
| --- | --- | --- | --- |
| `fp8_mxfp4_grouped_gemm_masked` | 回执在绑定的 `candidate/solution.py::Model.forward` 处只标 host；implementation 引用都位于 kernel.cu，缺绑定实现入口类别。 | 6/32；receipt_repairs=3，自动补正上限 2。 | handoff 2684 字符、6 非空行，超过 1200 字符限制。 |
| `paged_indexed_latent_fp8` | 同一绑定入口的证据只标 implementation；host 引用位于 launch.cu，缺绑定 host 类别。 | 6/32；receipt_repairs=3，自动补正上限 2。 | handoff 2282 字符、1 非空行，超过 1200 字符限制。 |

两个停止任务均已确认全部调用结清、旧 Runner 退出且锁释放，无 pause/cancel/hold/维护请求。属于回执确实缺项，未发现验证器误判。

Supervisor 授权各执行一次原 run 普通恢复，以既有动态反馈传递真实入口诊断，保留稿件、原会话、累计用量与全部门禁。不额外向状态或 prompt 注入手工材料，不改旧回执、不改工具或调用预算；新结果若再次失败，须先分析变化，不能循环恢复。后置 handoff 问题及前置诊断遮蔽后续错误的事实保留供后续优化。

恢复证据见 [masked GEMM](../recovery/masked_gemm_p3_anchor/) 和 [分页索引 Sparse](../recovery/indexed_sparse_p3_anchor/)。

两项均已各执行一次正式 `resume`，恢复为原 P3 reviewer 运行，调用由 6 增至 7/32；补正/失败/实质退修计数、原会话、既有稿件、批准和旧账目均保留。新调用均通过原生 init 确认真实会话，已携带对应入口诊断。尚未确认 P3 获批。

06:14:10 的[恢复后复核](check-9-65-after-recovery.json) 确认 3 项 P4、9 项 P3，全部 12 项 running，共 12 个在途调用。两项 MLA 分别在补正 P3 的绑定实现入口与证据索引，保持各自现有自动补正。

规则说明的额外事实：Sparse 子代理核对其 P3 首次及最新失败 reviewer 的完整动态 prompt，均未见字符串 `1200`；冻结 `response.md` 第 17 行只要求交接说明“简短”。该观察连同后置 handoff 校验问题留供后续改进参考，本次没有因此变更冻结 prompt、执行器或门禁。
