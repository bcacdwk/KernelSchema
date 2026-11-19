# 六算子提取尝试：修订结果

本目录归档六算子提取尝试后续形成的 2.2.4、2.2.7 和 2.5.0 修订结果，保留各自的发布回执、固定来源与独立审阅记录。初始结果、版本关系和暂停清单见[专题总表](../README.md)。文件搬迁前后的路径及 SHA256 见 [LOCATIONS.json](../LOCATIONS.json)；原回执中的发布地址保持当时记录，不因归档改写。

| 算子 | 提取工具版本 | 修订与核验 |
| --- | --- | --- |
| [fp8_scaled_grouped_gemm（七节点）](2.5.0/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | 2.5.0-bridge-receiver-r3 | 固定原33份来源与角色新建提取，参数化实际发射成员，并完整保留 V0、sync、wide 及恢复提交序列。七节点、P6 和正式发布均已完成；交付摘要匹配，最终 P6 审计无证据错误，用量覆盖仍为 partial。 |
| [fp8_scaled_grouped_gemm](2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | 2.2.4 | root 与 wide 明确 graph 销毁返回码被忽略，仅受检查的构建和 launch 失败触发直接提交；三个节点已完成独立审阅、P6 终审、完整审计及正式 export。 |
| [gdn_prefill_bf16_state_bf16](2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | 2.2.7 | 同源码接续，按缓存/直接 QK、分段及 Perm 存在性独立表达六个完整节点；6/6、P6 终审、最终一次 P6 调用的 `audit --verify` 与正式 export 均已完成，Supervisor 已验收表达修订。 |
| [mla_expanded_fp8](2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | 2.5.0 | 固定原30份来源字节与角色新建提取，显式 nullable 与合法分层绑定；持久 root 和单工作 CTA 两节点均独立审阅，P6 与正式发布完成。交付摘要匹配；最终 P6 审计无证据错误，但用量覆盖仍为 partial。 |
| [dual_cache_latent_bf16](2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) | 2.5.0-bridge-receiver-r3-protocol-feedback-r1 | 保留原 source/config，正式连续修订并完成八节点 P6 与 export。任务域、probe 实际计数及 SinkM 保留已修正；交付文件、来源和语义/执行绑定均核对一致。 |

分组 GEMM 的源码及输入绑定与[原始版本](../initial/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md)相同；原始 YAML 和回执保持原字节。修订版回执位于其目录下的 `.extraction/receipt.json`，实验与审计范围见[试用记录](../../../../schema_extract/archive/six-operators/20261103-trial/TRIAL_REPORT.md)。

GEMM 七节点版回执绑定 run `d4e41124-e29c-458c-a1ce-2a9bf87ee0a8`。P6 曾定点重开 root 并要求各本地流水线补齐 TMEM 行组寻址来源限制，相关节点复审后重新完成全树终审；现有完整乘积、行所有权与依赖设计有据，源实现偏差及外部比较器定义缺项继续保留。恢复节点按实际计算提交序列区分，图创建和实例化不计作额外计算。旧 2.2.4 结果保留历史身份；当前交付核验见[执行记录](../../../../schema_extract/archive/six-operators/20261105-recovery/EXECUTION_LOG.md)。

GDN 的源码及输入角色绑定保持不变，[原始两节点](../initial/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md)及回执未覆写；新交付身份见 [2.2.7 回执](2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/receipt.json)。独立 P1 确认实际任务提供 packed 元数据且固定 `norm=false`，不纳入 reference 的额外可选接口。修订保留 core 写回、mini-chunk 读取退休、跨 stream 及外部比较器限制；未修改 kernel，未做 GPU 验证。终审、导出与原目标冲突的区别见[试用记录](../../../../schema_extract/archive/six-operators/20261103-trial/TRIAL_REPORT.md)。

MLA 的新回执绑定 run `531567f1-bcae-45ad-acf6-fd674c055fdc` 和新冻结身份，[原单节点](../initial/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md)及其回执保留。持久调度 root 与单工作 CTA 的 `bootstrap` 分开表达；`bootstrap` 仅共享 root 数学区，调度、流水线和实现交接保留本地定义，原始引用树正常发布。源码 TMEM 寻址、描述符读取退休的落实偏差及外部验收定义缺口保留为来源限制，批准不构成 kernel 正确性认证。阶段重开、交付核验和审计范围见[执行记录](../../../../schema_extract/archive/six-operators/20261105-recovery/EXECUTION_LOG.md)。

稀疏注意力八节点回执绑定原接续 run `8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7`；原三节点目录和回执未覆盖。P6 在独立内容修订及回执诊断兼容修复后，重新检查完整当前树；运行层修复不授予 schema 批准，最终批准来自新的独立 Reviewer。新稿明确两个 scale 与必传 sinks，区分 probe 目标并集和实际并发计数，保留 hidden 前缀、fork/join 及 SinkM 的跨 chunk 生命周期。新目录的九份文件与最终 checkpoint 摘要一致；源共享地址布局和 probe 并发缺陷、外部验收定义缺口仍保留。正式交付与原输出冲突的区别见[执行记录](../../../../schema_extract/archive/six-operators/20261105-recovery/EXECUTION_LOG.md)。
