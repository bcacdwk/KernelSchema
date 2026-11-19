# 第 2、5 项的当前写法与修改示意

以下供讨论稿选用。当前片段来自冻结 YAML，只省略上层路径、统一缩进；“先采用”仍用现有字段，“待讨论”涉及规则或字段变化，均未修改生产 YAML。所有例子只核对文本与源码，没有重跑提取或 GPU。

## 第 2 项：内部三步已经登记，调用前后的等待仍靠文字

选用 Sparse 的 `probe_hidden_packed`。`probe` 是探测，`prefix` 写前一段输出，`remainder` 写剩余输出。[当前 YAML:299](../inputs/extract_sparse/probe_hidden_packed.yaml:299) 已登记三次提交，却要在说明中另写输入先后和输出汇合：

```yaml
thesis: |-
  Fork after the caller's preceding work, run the full-batch probe on a side stream
  and a packed prefix on the caller stream, then submit the accepted packed remainder
  behind the probe. The remainder need not wait for the prefix. Queue a side-to-caller
  join after the remainder and behind the prefix, so subsequent caller work observes
  all output rows. One probe and two attention launches
  plus the event pair permit overlap without promising simultaneous residency.
flow:
  launch:
  - phase: [probe]
  - phase: [prefix]
  - phase: [remainder]
    start_after: probe.kernel_done
```

**怎么改：先比较统一约定是否够用。** 在公共约定和工作调度中定义一次：本请求相关输入的首次使用排在调用者前序工作之后；调用者后续 GPU 工作消费输出前，要等全部相关输出生产者完成。这个约定只安排设备上的等待，不要求 CPU 等输出完成，也不给两路输出生产者增加先后顺序。

**预期写法 A：采用上述新约定后，继续使用现有字段。** 下列 `thesis` 是建议的短说明，所依赖的边界约定目前尚未成为正式规则。

```yaml
thesis: |-
  按统一调用边界约定接入输入、交出完整 out。
  probe 与 prefix 可重叠；remainder 只等 probe 完成。
flow:
  launch:
  - phase: [probe]
  - phase: [prefix]
  - phase: [remainder]
    start_after: probe.kernel_done
```

**预期写法 B：若需要每个实例明确接线，可讨论增加边界字段。** 下列 `boundary` 及其字段、`caller` 端点全部是语义草图，**不是现行合法字段或已批准的新语法**。

```yaml
flow:
  boundary:
    input_after: caller.prior_done
    applies_to: [probe, prefix]
    output_before: caller.next_start
    waits_for: [prefix.kernel_done, remainder.kernel_done]
    block_host_for_outputs: false
  launch:
  - phase: [probe]
  - phase: [prefix]
  - phase: [remainder]
    start_after: probe.kernel_done
```

这里 `input_after/applies_to` 合起来表示“调用者前序完成 → probe、prefix 启动”；`waits_for/output_before` 合起来表示“两路输出完成 → 调用者后续 GPU 工作启动”。`caller` 指本次调用所在的设备执行队列边界，不是 CPU 返回位置。并行关系不变：没有 `prefix.kernel_done → remainder`，也没有增加第四次计算提交。

**源码核对与边界。** [源码:6155](../work/extract_sparse/dsv4_kernel.cu:6155) 在调用者队列记录事件并让旁路等待，随后在旁路提交 probe；[源码:6162](../work/extract_sparse/dsv4_kernel.cu:6162) 在调用者队列提交 prefix；[源码:6188](../work/extract_sparse/dsv4_kernel.cu:6188) 在旁路提交 remainder，记录完成事件，再给调用者后续工作排入等待并返回。prefix 和 remainder 覆盖不相交的输出区间，见 [YAML:375](../inputs/extract_sparse/probe_hidden_packed.yaml:375) 和 [YAML:438](../inputs/extract_sparse/probe_hidden_packed.yaml:438)。这些关系允许重叠，不保证两者实际同时驻留 GPU。

CPU 仍执行 [源码:5593](../work/extract_sparse/dsv4_kernel.cu:5593) 的有限探测轮询；这里“不阻塞 CPU”仅指输出汇合不会让 CPU 等两路输出完成。轮询可能提前结束、后续调用重置探测槽可能与旧写入竞争，见 [源码:5600](../work/extract_sparse/dsv4_kernel.cu:5600)；本例的边界写法不会自动修复这一源问题。

现行 [规则:309](../rules/schema_docs/4.design.work_schedule.md:309) 只允许 `<phase_id>.<milestone>` 事件，不能直接把 `caller.prior_done` 塞入 `start_after`。因此 A 需要先定义统一边界语义，B 需要新增外部端点规则；当前材料尚未证明必须选 B。本例真正要讨论的是“这五条先后关系能否短而明确地写全”，不是把三次 launch 串成一条流水线。

## 第 5 项：每次重新算值，但多次调用可能用同一块内存

选用 GDN `root` 的共享工作空间。`prep` 写中间结果，`chain` 消费它们。[当前 YAML:379](../inputs/extract_gdn/root.yaml:379) 把跨调用复用条件放在 `prep.schedule.strategy`：

```yaml
schedule:
  strategy: >-
    The current CUDA stream orders prep, chain and reuse of the grow-only
    workspace across calls. Every consumed handoff slot is rewritten.
    The source provides no cross-stream event, lock or reentrancy protocol.
```

**怎么改：先统一短说明，决定要不要再加结构。** 最少说清“共享哪块内存、本次读取谁写的值、下次何时能改写”。本例不需要把分配器、缓存管理过程全部写成执行图。

**预期写法 A：现在就能放在原有 `strategy` 中的短说明。** 第一、二句补出源码已有的存储身份与沿用条件，后面保留现有正确性边界；这里是可审阅的文字提案，未实际改动 YAML。

```yaml
schedule:
  strategy: |-
    各次调用共用进程内 ws_cache；同设备且容量够时沿用。
    未分配、容量不足或设备改变时，替换为本次所需大小的存储。
    本次 chain 消费的交接槽均由本次 prep 重写。
    跨调用复用依靠同一 CUDA stream 的顺序，前次用完后下次改写。
    源码未提供跨 stream 同步、锁或可重入保证。
```

**预期写法 B：只有决定 schema 要检查跨调用存储复用后，再考虑少量字段。** 下列整个 `shared_storage` 都是讨论草图，`ws_cache` 对应源码中的静态对象；其中条件记录源码所依赖的顺序，不会生成同步机制。

```yaml
shared_storage:
  ws_cache:
    scope: process
    reuse_if: same_device && capacity_bytes >= need
    replace_if: absent || !same_device || capacity_bytes < need
    current_values: chain 消费的槽由本次 prep 重写
    reuse_order: 同一 CUDA stream 上前次用完，再由下次改写
    cross_stream_order: 源码未提供
```

`scope: process` 表示缓存对象的共享范围，不表示地址永远不变；同设备且容量足够时才沿用现有分配。下一次的 shape 也不必相同，容量够用即可。若用新字段，还须定义如何把已有 handoff 的数学值连到这块存储的实际区域；这里不凭这八行宣布完整语法已经设计好。

**源码核对与边界。** [源码:3258](../work/extract_gdn/gdn_kernel.cu:3258) 定义唯一静态 `ws_cache`，[源码:3260](../work/extract_gdn/gdn_kernel.cu:3260) 仅在未分配、容量不足或设备改变时替换分配；同设备时容量只增不减，换设备时可重新分配更小存储。[源码:3254](../work/extract_gdn/gdn_kernel.cu:3254) 说明每次 prep 重写 chain 会消费的槽。[源码:3274](../work/extract_gdn/gdn_kernel.cu:3274) 取得当前 CUDA stream，[源码:3381](../work/extract_gdn/gdn_kernel.cu:3381) 和 [源码:3431](../work/extract_gdn/gdn_kernel.cu:3431) 分别把 prep 和 chain 提交给它。调用内 chain 提前启动、等到真正消费前再等待 prep 的现有安排仍由 [YAML:326](../inputs/extract_gdn/root.yaml:326) 的 launch/handoff 表达；不要为了说明跨调用复用而抹掉这个区别。

这也说明为什么 `retention` 和 `storage: global` 不能独自替代上面的短说明：[retention 规则:100](../rules/schema_docs/3.design.math_reshape.md:100) 可以使用已定义的逻辑实例，**并非只支持 invocation**，但它不指定真实地址；[global 规则:341](../rules/schema_docs/4.design.work_schedule.md:341) 也不规定物理地址或分配器所有权。不能把“值保留一段时间”当成“多次调用用了同一块地址”。

本例能够支持“同 stream 顺序下的共享复用”这一窄结论，不能据此声称任意多 stream 并发都安全，也不据缺少锁就断言当前允许的调用方式必然有错。新增字段不能替源码补出代际、锁或同步。这里的预期效果是把现有前提集中说清，是否值得将其变成专门结构仍需决定。
