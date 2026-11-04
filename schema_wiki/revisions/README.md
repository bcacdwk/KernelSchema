# Schema 修订版本

修订版发布在独立目录，并保留自己的发布回执、固定来源和独立审阅。原始目录用于历史追溯，当前推荐版本以 [Schema Wiki 导航](../README.md)为准。

| 算子 | 提取工具版本 | 修订与核验 |
| --- | --- | --- |
| [fp8_scaled_grouped_gemm](2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | 2.2.4 | root 与 wide 明确 graph 销毁返回码被忽略，仅受检查的构建和 launch 失败触发直接提交；三个节点已完成独立审阅、P6 终审、完整审计及正式 export。 |

分组 GEMM 的源码及输入绑定与[原始版本](../aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md)相同；原始 YAML 和回执保持原字节。修订版回执位于其目录下的 `.extraction/receipt.json`，实验与审计范围见[试用记录](../../schema_extract/TRIAL_REPORT.md)。
