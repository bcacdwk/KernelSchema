# Schema 修订版本

修订版发布在独立目录，并保留自己的发布回执、固定来源和独立审阅。原始目录用于历史追溯，当前推荐版本以 [Schema Wiki 导航](../README.md)为准。

| 算子 | 提取工具版本 | 修订与核验 |
| --- | --- | --- |
| [fp8_scaled_grouped_gemm](2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | 2.2.4 | root 与 wide 明确 graph 销毁返回码被忽略，仅受检查的构建和 launch 失败触发直接提交；三个节点已完成独立审阅、P6 终审、完整审计及正式 export。 |
| [gdn_prefill_bf16_state_bf16](2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | 2.2.7 | 同源码接续，按缓存/直接 QK、分段及 Perm 存在性独立表达六个完整节点；6/6、P6 终审、最终一次 P6 调用的 `audit --verify` 与正式 export 均已完成，Supervisor 已验收表达修订。 |

分组 GEMM 的源码及输入绑定与[原始版本](../aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md)相同；原始 YAML 和回执保持原字节。修订版回执位于其目录下的 `.extraction/receipt.json`，实验与审计范围见[试用记录](../../schema_extract/TRIAL_REPORT.md)。

GDN 的源码及输入角色绑定保持不变，[原始两节点](../aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md)及回执未覆写；新交付身份见 [2.2.7 回执](2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/receipt.json)。独立 P1 确认实际任务提供 packed 元数据且固定 `norm=false`，不纳入 reference 的额外可选接口。修订保留 core 写回、mini-chunk 读取退休、跨 stream 及外部比较器限制；未修改 kernel，未做 GPU 验证。终审、导出与原目标冲突的区别见[试用记录](../../schema_extract/TRIAL_REPORT.md)。
