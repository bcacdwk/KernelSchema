**可表达，控制层次更清楚；未使用 while。** 有序列表直接呈现 worker 进入、工作遍历与退出，两个 if 明确表达可选排空等待。按原动作粒度，无需新增动作、宏或具名段。源码 scan 的二分查找虽然使用 while，仍属于既有 `produce:TilePrefix` 的局部实现；等待原语的轮询也不提升为控制节点。

本稿从[归档原 root](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml)独立重建，按 [Pipe 规则](../README.md#pipe-规则)只投影 `experiment` 与 `tile_pipeline.phase`。数学、调度、资源、成员、模式、动作契约和里程碑继续继承。完整覆盖 7 个 phase、19 个角色、96 个静态动作；有 19 个 for、2 个 if，控制最多嵌套两层。

| phase | 角色数 | 保留的结构 |
| --- | ---: | --- |
| count、scan、scatter、finalize | 4 | 进入协议后直接 produce |
| gather | 1 | 进入后遍历 t |
| fc1 | 7 | A、B、MMA 与四个独立 epilogue |
| fc2 | 7 | 同上，四个 epilogue 各有最终 finish |

- **源码路径。** [Model.forward](/home/shaohanyong.shy/KernelSchema/schema_discussion/pipe_grammar/trial_roots/sources/extract_moe/solution.py:1054)（1054–1108）和 [fast_call](/home/shaohanyong.shy/KernelSchema/schema_discussion/pipe_grammar/trial_roots/sources/extract_moe/nvfp4_fast.c:153)（153–278）复用 ABI 后调用 `moe_run`。固定 [nvfp4_moe.cu](/home/shaohanyong.shy/KernelSchema/schema_discussion/pipe_grammar/trial_roots/sources/extract_moe/nvfp4_moe.cu:9230) 的 `run_pipeline`（9230–9356）对应七 phase；入口分别为 `k_count:582`、`k_scan_tiles:750`、`k_scatter_agg:700`、`k_gather_f:1189`、两 GEMM 和 `k_finalize:1464`／`k_finalize16:1604`。7775、8354 的实例分别落实 cfg97、cfg2=81，均为 224 线程、四 epilogue 的 `f4_body_ws`。
- **作用域。** 所有 phase 提供完整 domain，FC 原有 t/i/g/c 保持；q 只存在于旧 MMA 的局部门控，现已消去，未提升为共享域。A 在 `for(t)` 中 prime，再遍历 `g[t]`；B 直接遍历连续 g。源码 4460–4540、4543–4841 的 g 均跨工作递增，而 A 的两组寄存器每份工作重新预置。三槽 payload、两槽 scale 的复用边继续跨工作生效。c 对应真实的两片 Fragment，显列 0、1 后仍保留域。源码每 stage 内四次更新也使用变量名 q，那属于 MMA 动作内部的真实数据遍历，与此次删除的控制索引不同。
- **条件和依赖。** 两 MMA 均在当前位置执行 `if(t > 0 and Hoist == 0)`，内容仅 `wait:Drain(t,0)`；Hoist 来自固定形状比较 `R >= 16*E`。源码 4855–4966 允许 Hoist=1 时先准备首 stage 的 scale，再等待前一 accumulator。两角色各提供完整 waits_for，每份含三个目标、十一条前驱引用；唯一改名为左侧 `wait:Drain(t,0)@done`。prepare 的三条数据前驱、wait 自己及首 MMA 各自的四条 release 前驱原样保留，其他十七角色不覆盖依赖。等待发生域继承 if；窗口内缺前驱必须报错，不能求事件交集删边。
- **首尾与发布。** 4379 的全局空路由返回先于分配，4439–4440 是一次分配和会合；5374–5375 是全 worker 的最终会合与释放。八个 epilogue 角色均保留 `drain(0) → emit(0) → drain(1) → release → emit(1)`，对应 5050–5064、5143–5157 的早 release。FC2 的 Sink 读取完成等待属于 emit 契约（5170）；全局完成等待 `finish:BulkStores` 对应 5366–5372，只执行于整个工作序列之后。
- **边界。** J=0 时进入仍有事件，退出及 FC2 finish 按原契约为空操作；J>0 但本 worker 无工作时，分配、会合、退出和 FC2 finish 仍存在。单工作无前项 drain；多工作、Hoist=0 从 t=1 起等待，Hoist=1 不产生该 wait，但后续工作的首 MMA 仍受四个 release 约束。尾部无有效输出行只减少 emit 的 payload，不删除 drain/release/emit。空 gather 仍进入，空路由 finalize 仍写零。
- **事件粒度。** 有序列表规定发起次序，不给异步 load、stage、prepare 附加完成等待。for 的工作边界也不清空槽位或重置协议。独立的 scale 发布、MMA 累积完成、寄存器读出、accumulator 归还及最终 bulk 完成继续由原契约和依赖联系；release 后无需等待其他 epilogue，当前 warp 可立即继续转换和写出。
- **来源与检查。** 原引用 `ce1b7d5…` 私有提交本地不可读；固定副本的 11 个文件均已匹配该输入锁 `source.files` 和 manifest 的 SHA-256，原 root 哈希也匹配。未发现本次动作顺序与源码冲突；源码 4422 的旧线程范围注释、8330 段的“override-only”注释由实际 launch 参数及 `solution.py:1308` 的默认选择纠正，未据此改动设计。PyYAML、结构与静态签名检查通过；四组合法归约长度、Hoist 两值及本地工作数 0/1/2/4 共 32 组，448 条 FC 角色轨迹、9078 个动作实例、7310 条条件／边界依赖均与原 root 一致，其余五角色也逐一比对。名称、签名、角色拼错和条件造成的窗口内缺前驱四个负例均被拒绝。条件枚举用于检验两支结构，并不宣称每个组合都能通过完整默认调度选择；全局空工作与局部空 worker 的分配差异另由源码确认。未运行 GPU；有限检查不证明数值正确性或真实重叠。
