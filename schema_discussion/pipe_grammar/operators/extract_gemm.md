**extract_gemm：冻结稿的 pipe 现状审计**

覆盖全部7个 approved 有稿节点、6个精确 pipe 组、14个 phase、40个 role 上下文；root 虽标记 authored=false，已有正文仍计入。A 源声明为19个 role、51块、140个静态动作；B 本地展开与 C 有效展开均为40个 role、107块、289个静态动作，其中67个嵌套块，最深3层。root 单列为3个 role、8块、23个动作。全部 pipeline 本地定义，没有跨节点继承。

| 节点 | phase | A/B 块数 | B role数 |
| --- | --- | ---: | ---: |
| root | main | 8/8 | 3 |
| mma_sync | main | 3/3 | 1 |
| v0_recovery | graph、direct | 8/16 | 6 |
| wide | main | 8/8 | 3 |
| wide_recovery | first、retry | 8/16 | 6 |
| wide_recovery_chain | first、retry、final | 8/24 | 9 |
| wide_recovery_full | first、retry、fallback、final | 8/32 | 12 |

六组分别是 sync Main、共用 Producer、pair Issuer、pair Epilogue、wide Issuer、wide Epilogue，出现次数为1、13、3、3、10、10。四个 recovery 文件复用整个 phase，51→107块主要来自 alias 展开，不是新增56次创作。组指纹不含域、契约和等待边；13处 Producer 虽完全同形，pair 处理两个 K 块，wide 处理一个，输入环复用距离也由3变为 R。

t 是 worker 获配工作的本地序号，group g 由调度解码；没有独立 g 循环。root/v0 的 n=flatten(t,q)，q 遍历 K 块对；wide 的 n=flatten(t,kb)，kb 遍历单块。Producer/Issuer 沿连续 n 推进，使环槽及依赖跨工作接续；Epilogue 外层按 t 初始化 F，再以 n[t] 完成当前工作的归约。sync 则在 t 内直接遍历 kb，每份工作重置 stage 序号。wide recovery 调度中的 q 是提交序号，与 pair 域中的 q 分属不同节点与定义位置。107块全部带 repeat；67个子块全为 repeat+steady，没有嵌套在 prologue/epilogue 的循环。27个列表混排动作和循环，13个列表共有36对相邻循环；这些是静态布局，绑定标志可重叠，不表示运行次数。

Epilogue 的 p[0:2] 先装入两份 scale；n[t] 内捕获当前值后补入后续值。分片正文先 read 第0片，再重复“await 当前片、read 下一片、compute 当前片”；pair 用 d[0:7]，wide 用 d[0:3]。末片等待后先 release 产品槽，再执行最后一次寄存器累加，使全部 TMEM 读完后的复用早于累加收尾。等待边接到 await.done，serial 顺序约束随后的动作。pair 的四个 warp 分行，wide 的两个四-warp组分别拥有列组，再在组内分行；这里每个 phase 只有一个 Epilogue role，成员协作范围由正文限定。

f/z 局部域控制两条写回路径：pair 的 f 上界为 Full(t)，wide 为 Full(t)*2/J，z 为 1-Full(t)。尾块只写有效行，完整块执行 staging 归还、转换发布及异步存储；wide 的 F 实例 J=1、R=4，分两批，G 实例 J=2、R=3，一批覆盖两列组。同一 pipe 因提交参数绑定而有不同容量和批数。条件等待实际位于 f 内的 release:Staging：它等此前所有 bulk store 结束源读取，随后才能覆盖 staging；tail 保留该历史。此处没有局部 q 条件等待或 Hoist。首次完整块无旧 store 可等，但动作仍存在；空 f/z 域则不产生主体动作。

初始化与退役同样依赖位置和正文配合。Producer 的 prologue 初始化 Schedule、资源世代并完成 CTA 汇合；Epilogue 的 prologue 显式等它。每个 t 的 prepare 初始化 F；最外层 epilogue 的 drain 等所有 bulk 目的写完，全部 role 完成后的 CTA 汇合与 ProductRing 退役留在契约中。空 worker 仍有首尾动作。sync 的 drain 则位于每个 t 内，先结束输入复制，再把同一 InputRing 槽解释为输出；store 的完成汇合保护下一工作。预取末尾的空提交仍有事件，与空重复域不同。

恢复阶段的 alias 共用写法，而各次提交由 kernel_done 顺序连接，重新建立 worker、累加器和环状态；都重读原输入并覆盖输出。图执行绑定录制参数，直接提交绑定当次参数，因此复用正文不表示共享在途事件。现有问题主要是读者必须跨域、math、schedule、动作契约核对这些边界；pipe 本身不承载全部完成保证。例如载入完成只覆盖实际传输字节，未载入的高行保留原内容；这些行仍参与计算和归还，但不写输出。产品动作完成同时服务于操作数复用和结果就绪，而产品槽归还还要求所有行与列组结束读取，不能只从相邻动作名称推得。另有源实现限制：六个 TMEM 节点已明确记录行组地址贡献遗漏，正文保留完整行所有权；本审计不把这一偏差计为新语法，也不据静态稿宣称硬件执行正确。

逐节点、逐组及40个上下文的坐标与证据见 [机器复核记录](../reviews/extract_gemm.json)；语义依据为冻结的 [tile_pipeline 文档](../../../schema_extract/archive/20261109-ten-operator-text-audit/rules/schema_docs/5.design.tile_pipeline.md)。
