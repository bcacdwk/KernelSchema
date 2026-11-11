# template_fa4_prefill：pipe 写法审计

现行 11 个节点的 pipe 同时表达工作生命周期、KV 主迭代、通道半片、条件分支和预取窗口。嵌套深度或 `prologue/epilogue` 出现次数不能直接代表流水复杂度：双 loader 的首尾 QK/PV 明列在主体列表中；joint loader 则在 `e` 循环和 token 契约中表达同一推进次序。全部 YAML 已与冻结输入逐字节核对，相同；`tile_pipeline` 均为本地定义，没有 `from_parent`，父子关系不是本轮 pipe 的继承来源。

统一 census 为 **54 个 role 使用位置、23 个精确 pipe 组、166 个块、236 个静态动作出现**。深度 1/2/3 分别为 54/85/27；112 个嵌套块全为 `repeat+steady`，无 repeat 且仅有 steady 的块为 16 个，显式 prologue/epilogue 为 9/2 个。本例无 YAML alias 或管线继承，A 声明、B 本地、C 有效使用三口径相同。静态动作只计书写位置，不按循环、成员或运行工作数展开；112 也不是无语义容器数量。

覆盖包含 root、rect_joint_load、packed_persistent、single_persistent_dual_load、single_persistent_joint_load、small_persistent_lpt、small_persistent_shared_rescale、small_static_lpt、small_static_shared_rescale、splitkv_dual_load、splitkv_joint_load；逐 role、精确组及使用上下文见 [review](../reviews/template_fa4_prefill.json)，结构计数以[统一 census](../data/operators/template_fa4_prefill.json)为准，声明量不等于运行展开动作数。

root 每个 worker 处理一份工作，LoadQK 先遍历 `d` 装 Q，再遍历 `i→d` 装 K；LoadV 独立遍历 `i→d`。`i` 是 KV 主迭代，`d=[0,2)` 是 QK 通道半片或 PV 输出半片。MMA 的外层无 repeat，`steady` 内依次放首 QK、`i[1:]` 中的两段半片循环、末 PV：

```yaml
- {repeat: d, steady: ['qk:S(0,d)']}
- repeat: i[1:]
  steady:
  - {repeat: d, steady: ['qk:S(i,d)']}
  - {repeat: d, steady: ['pv:Y(i-1,d)']}
- {repeat: d, steady: ['pv:Y(last(i),d)']}
```

原片段在 [root.yaml:233](../../../schema_template/examples/fa4_prefill/root.yaml#L233)。它使 QK 领先 PV 一块，末 PV 消化滞后一块；N=1 时中段为空，首 QK 和末 PV 仍存在。相邻两段 `d` 循环保持两个 QK 半片先于两个 PV 半片的发射顺序。数据边约束 Q/K、P/V、已修正 Y 就绪，复用边约束四槽 K/V 和两槽 score；异步发射有序不等于前一动作已完成，P 被后续 QK 覆盖前已读的保证另在动作契约中。

Softmax 将 capture→Stats→P→ell 组成每轮主体，之后发布 FinalSum；Correction 单列首次 Stats 确认，仅在 `i[1:]` 做 rescale，末尾交替 stage/store 两个输出半片。首次确认不读载荷仍有事件；单槽 OutputStage 的完成契约约束下一次覆盖。一个 rescale 已覆盖两个 Y 半片，Softmax 内部含逐行最大值、掩码和指数，粒度由数学及动作契约说明，pipe 没有逐条列出。small_static 两节点保留这些循环，只在 Softmax 的显式 prologue 发布一次 WorkCoord，两个 loader 的入口等待它（[small_static_lpt.yaml:137](../../../schema_template/examples/fa4_prefill/small_static_lpt.yaml#L137)）。

五个 persistent 节点使用 `t: worktile`、`n=flatten(t,i)`。双 loader 的 LoadV 顶层直接遍历连续 `n`，其余角色按 `t` 使用 `n[t]` 或 `n[t,1:]`；槽序号及 `n+2` 可跨工作。RequestMap 在整个 worker 的 prologue 发布一次，FinalSum 每份工作一次。Q 的下一次装载等待本工作最后 QK；Correction 在第二半片 stage 后、store 前 release Y，下一工作首次 PV 等待此 release，因而末尾 store 与下一工作的 Y 初始化具有重叠空间（[packed_persistent.yaml:134](../../../schema_template/examples/fa4_prefill/packed_persistent.yaml#L134)）。packed、single 和 small 的映射分别为原请求序查找、单请求查找、LPT，已分别核实；两个 shared_rescale 分支虽与各自 LPT 的 pipe 相同，Stats 载荷却改为 Softmax 已计算的一列 alpha。

三个 joint 节点把 K/V 半片按 `e=flatten(p,d)` 编号，persistent 再包含外层 `t`。MMA 的主体只有 `contract:KVStage(e)`，但 `p` 与 K/V 的映射确定 K0,K1,V0,…,Vlast，N=1 为 K0,V0；`e+4` 约束共享四 token 复用，K/V 仍在不同数组。首尾和 `d` 没有消失，其表达位置在 domain、token 契约与依赖索引中（[rect_joint_load.yaml:100](../../../schema_template/examples/fa4_prefill/rect_joint_load.yaml#L100)）。

两个 split 节点以 `active/empty` 的 0/1 域保护非空 attention 与仅写 EmptyLSE 的分支，空域不产生对应动作；有效工作 N=0 与网格越界工作不同。combine 在 attention kernel 完成后另启动。其 Main 显式 `repeat:s`，prologue 装 LSE、以局部 `p` 预装最多三份 PartialO、发布 Weights；每轮先用另一个局部 `p` 窗口装 `s+3`，再 accumulate。两个 `p` 各有独立局部域；前者没有当前 `s`，后者使用外层已绑定的 `s`，也都不是 joint attention 的共享 token 域。尾部窗口变空仍继续消费已装数据，epilogue store 最终输出；四槽复用依赖前四项的 accumulate 完成（[splitkv_dual_load.yaml:296](../../../schema_template/examples/fa4_prefill/splitkv_dual_load.yaml#L296)）。

本例呈现三个解释问题：①“首尾”同时存在于显式区域和 steady 内的有序子块，且 worker 一次与 worktile 一次不是同一范围；②精确相同 pipe 不保证载荷、数学和调度相同，浅 token 循环的含义也依赖管线外正文；③声明域不等于实际有效访问，combine 按线程所持 row group 的末个非空 split 裁剪读取、跳过非正权乘加，最初预读仍可接触未写 payload。这些是当前表达与计数的边界，未据此判定语法缺陷或运行错误；未做 GPU 验证。
