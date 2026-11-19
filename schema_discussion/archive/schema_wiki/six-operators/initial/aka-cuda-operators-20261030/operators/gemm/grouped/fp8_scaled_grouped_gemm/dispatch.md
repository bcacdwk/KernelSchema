# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | 初始化选中 UMMA、调度未进入 wide 的契约内工作量。 | 基准：128×128 输出块、成对 128-K 推进、三槽操作数环、两槽 TMEM 对和四个 epilogue warp；dual、B 策略、小 A 盒及尾块为节点内变化。 | solution.py::_Engines 按设备 0 初始化 UMMA；fp8umma_run 经 sched_policy（1539–1655）取得正常构建的 engine=0、sareg=0，进入 umma_gemm<0>。身份 memo 可复用实际 ABI；调度命中复用缓存，未命中按 indptr 快照和几何门控。 |
| [mma_sync](mma_sync.yaml) | root | 初始化未选中 UMMA、成功选用 sync 的契约内分组 GEMM。 | 保留持久分组调度及 CTA 内全 K 归约；改为 64×128 输出块、逐 128-K 推进、八 warp 协作的三级 cp.async 环和 mma.sync 寄存器部分结果。BF16 经 stmatrix 复用操作数 SMEM 后向量写回；网格上限改为 2*nSM。 | solution.py::_Engines.__init__（168–173）仅在 UMMA 未选中时构建 fp8gg.cu；Model.forward 经预选 _fn 调用 fp8gg_run → fp8gg_bm64。这不是 UMMA 运行错误后的 fallback。 |
| [umma_wide](umma_wide.yaml) | root | 初始化选中 UMMA、sched_policy 选中 wide 的多波或有组内行块复用的工作量。 | 保留持久调度，改为 128×256 输出块、逐 128-K 推进、单 MMA issuer、八个 epilogue warp、两槽 256 列 TMEM 和寄存器尺度预取。同一模板统一 w1f（四级环、32 KiB 半块顺序存储）与 w1g（三级环、64 KiB 整块存储）。 | fp8gg_umma.cu::sched_policy（1542–1655）命中复用 engine，否则以有效 indptr 快照的活跃组数、128 行块数、K/128、N/256 和 nSM 执行 wide 门控（1588–1632）；engine=1/2 进入 umma_wide。engine=2 的低 K/复用门控、g_w1g_off 及直接启动即时失败只改变节点内配置。 |

## 覆盖缺口

- 三节点覆盖普通构建在初始化成功后的计算分支，不保证每个运行环境均可执行。引擎、nSM 和流设备索引取自设备 0；没有按输入设备重新选路的支持保证。必要描述符失败、共享内存容量不足或启动错误沿源码返回；w1g 直接启动的即时错误可降为 w1f，UMMA 运行错误不会再选 sync。
- 两个 ABI 均拒绝 N/K 非 128 倍数；UMMA 拒绝 G>116，sync 拒绝 G>4096。这些尺寸在当前 Contract 外，源码没有未对齐或越界尺寸的通用 fallback。
- 身份 memo 命中不重新校验 ABI；调度缓存键不含 indptr 内容和 nSM，可能沿用旧分支，而 kernel 仍读取当前设备边界。覆盖划分保留这些实际缓存状态，不增加元数据不变或自动失效保证。
- 外部验收定义的既有 source_gap 沿用 root 的 tolerance 来源记录，不作数值保证，也不阻塞本阶段分派。

## 选择说明

直接父仅表示设计复制来源；各节点的选择条件在共享 Contract 全域独立求值，不叠加父节点条件。精确条件和版本以各 YAML 为准，计划内派生文件由 P5 生成。

wide 的两种存储配置共用参数化节点；C 绑定与 ctypes、身份缓存命中与重新校验、graph 与直接启动不另建节点。普通构建未启用的诊断强制分支和 SAREG=1 不列为路径。
