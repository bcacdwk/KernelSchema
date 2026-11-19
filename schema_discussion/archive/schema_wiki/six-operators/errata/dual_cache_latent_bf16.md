# 路径索引

> 归档说明：原始 2.2.2 交付的阅读校注；本页保留历史内容，版本选择见 Schema Wiki 导航。 [归档索引](../README.md)。

本页是 `dual_cache_latent_bf16` 的[原发布索引](../initial/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md)文案勘误，供当前阅读使用。3 个节点仍指向原 YAML；下列表格与覆盖事实沿用原索引，仅移除过期阶段措辞并调整相对链接。原 Design、发布回执和审阅身份均未改变。

[结果初验](../../../../schema_extract/archive/six-operators/20261103-trial/OPERATOR_REVIEW.md)对现有 YAML 的条件流程保留意见继续有效；本页不构成新的语义复审或批准。

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](../initial/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/root.yaml) | 无 | 默认自动策略下的纯 mma.sync 调用，含 decode、小批量、其余未选 tcgen05 的输入及未编入 tcgen05 的构建回落。 | 基准设计；HB/KC、ring、索引预取、DQ 与 KV split 均为参数变化，多 split 使用 FP32 部分结果及独立 combine。 | 初始 P 的非 dual、非 TC_CFG_QIS2 分支；make_plan → dispatch → run_cfg。 |
| [probe_dual](../initial/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/probe_dual.yaml) | tc_grouped | 初始 P.dual 非零的整次调用；默认 GT=4/8，packed 与 mma.sync 备选均不拆 KV。 | 在父节点上增加内容相关 overlap probe 和双算法调度；可隐藏时 caller stream 先算 packed 前缀，side stream 对剩余 token 选择 packed 或 mma.sync 并回接；否则先 probe，再为整批选择一种算法。 | make_plan 保存非 tc 备选且 GT>tcpack_thresh()；dsv4_run → dual_dispatch_hidden / probe_mean_ratio。仅 0<ratio≤tcpack_thresh() 接受 packed；拒绝后整次调用仍归本节点。 |
| [tc_grouped](../initial/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/tc_grouped.yaml) | root | 无需 probe 的 tcgen05 调用：GT=1 的单 token head block，或元数据已决定采用 packed 的 GT=2 token group。 | 改用 gather4、TMEM 和异步 tcgen05；QK 两个 latent 半段独立累加后相加。GT>1 增加双池索引并集和成员掩码，保留同 token 重复项；不拆 KV。 | 初始 P 的非 dual TC_CFG_QIS2（cfg90）分支；由 tc_eligible 或 pack_gt 与非 tc 备选 splits=1 的门控选入，packed 的 GT≤tcpack_thresh()；dispatch → run_tc → tc_attn_kernel。 |

## 覆盖缺口

P 沿用 root 的默认自动策略：force_cfg=force_s=0、非诊断构建和默认扩展开关，tcpack_thresh()=2.6。W 为 Model.forward 选定 workspace 的字节容量；P 由 dsv4_run 调用 make_plan(H,T,IndexWidth,W,0,0,nsm_cached(),true) 得到，包含未编入 tcgen05 时的回落；此时尚未 probe 或切分 token 范围。tcgen05 的 GT 由 pack_gt 按 head 数、索引宽度、token 数及 SM 数决定。

直接父仅是设计复制来源。各节点的 when 在共享 Contract 全域独立求值，不叠加父条件；精确条件与版本以 YAML 为准。workspace/sink 归一化及非 dual 路径的直接提交、CUDA graph 捕获和重放属于调用方式，不另建节点。

- 在上述默认策略及共享 Contract 内，未发现未分配的 workload；强制配置、诊断、消融及手动修改扩展开关不在本树范围。
- 编译候选耗尽、tensor-map 编码、共享内存配置或 kernel 启动检查失败仍可能报错；不能将这些失败解释为自动改走 root。图调用失败所走的直接提交仍属于原计算路径。
- probe_end 最多轮询 2000000 次，随后只平均已收到的正样本；超时不一定返回 -1，也不证明 probe 已完成。源码指出旧 probe 可能与下一次 sentinel 写发生竞态，未在超时处增加完成等待或缓冲隔离；probe_dual 保留此边界。
- 外部验收 source_gap 沿用 root 的 tolerance：已知 atol=0.01、rtol=0.05、max_mismatch_fraction=0.0；比较器公式、相对误差基准、计算精度、NaN/Inf 处理及返回值/out 写回检查路由仍未给出。该缺口不用于设计或数值保证，不阻塞本次路径定稿。
