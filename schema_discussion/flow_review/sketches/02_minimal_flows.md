# 六算子最小 flow 片段

这些是讨论片段，不是完整 plan。第一段为删除 protocol 字段的提案，第二段沿用当前 flow，第三段只演示 Clean 的局部删减，第四段区分设计约定和实现绑定。没有覆盖或改写任何算子输入，不声明已完成整份旧稿的合法性迁移。

## 1. GDN：删除 protocol 字段的提案，三种时点都保留

只展开一个代表值；原稿其他七项仍需相同关系。

~~~yaml
flow:
  launch:
    - {phase: [prep]}
    - {phase: [chain], start_after: prep.launch_ready}
  handoff:
    Ti:
      producer: prep
      consumer: [chain]
      storage: global
      ready: prep.kernel_done
      wait_at: chain.prep_ready
~~~

chain.prep_ready 引用实际 dependency wait 的 done；消费者此前允许读取独立输入。不能用 prep.launch_ready 替代 prep.kernel_done，也不能用默认入口等待替代有据的晚等待。保留原稿 realization 对 PDL 属性、启动提示和 dependency wait 的已有绑定；规则需将 launch_ready 定义指向这份实现保证。本片段不是在未修改 protocol 填写规则时声称合法省略它。

## 2. MoE direct：单 kernel、两套工作分解、同 worker 存续

~~~yaml
flow:
  launch:
    - {phase: [prelude, direct], order: per_worker}
  handoff:
    Census:
      producer: prelude
      consumer: [direct]
      storage: tile_pipeline.phase.prelude.storage.Census
~~~

这里只把旧 resource 引用迁成当前 storage。各 worker 在本地完成一次复制 census 后推进多个 unit；不增加全 grid 屏障或 kernel。RouteAccum/Arrivals 是 direct 的合并状态，其首次为零/退出清回零以及顺序复用另保留一条短约定，不能从 Census 的 handoff 推导。

## 3. MoE double_verify：删去被启动链覆盖的等待位置

~~~yaml
flow:
  launch:
    - {phase: [probe], protocol: pdl}
    - {phase: [verify], start_after: probe.kernel_done, protocol: pdl}
    - {phase: [count], start_after: verify.kernel_done, protocol: pdl}
    - {phase: [scan], start_after: count.launch_ready, protocol: pdl}
    # scatter/gather/fc1/fc2/finalize 继续保留原有启动链。
  handoff:
    Clean:
      producer: verify
      consumer: [count, scan, scatter, gather, fc1, fc2, finalize]
      storage: global
      ready: verify.kernel_done
~~~

省略原 Clean.wait_at 及只供其使用的 guarded 别名。实际 guard 动作与其他 handoff 仍保留。verify 完成后 count 才可能启动，所有后继启动也不可能早于该完成点。此推导不适用于改由 verify.launch_ready 放行 prepare 的其他分支。这里保留其余旧 protocol 标记只为隔离 Clean 的删减，不声称逐项符合现行 protocol 填写条件；若采纳第1段的字段删除提案，应另行去掉这些标记并保留已有机制绑定。

## 4. 外部调用交接：短约定，不画 host 程序

外部关系必须先由设计/共同调用约定确定，realization.require 只绑定机制，不能首次决定顺序。下面保留设计句以避免错层；外部边仍是文字约定，尚未成为可在 flow 中解析的结构化端点。它并非新增 flow.boundary 字段，也不声称 require 消除了这项缺口。

~~~yaml
# Sparse probe_hidden 的调用交接摘录
design:
  work_schedule:
    thesis: >-
      Probe and prefix start after prior caller input-producing work.
      Remainder starts after probe and need not wait for prefix.
      Subsequent caller work consuming out waits for both attention
      submissions to complete; this does not require a host completion wait.
    flow:
      launch:
        - {phase: [probe]}
        - {phase: [prefix]}
        - {phase: [remainder], start_after: probe.kernel_done}
realization:
  require:
    - item: design.work_schedule::flow
      choice: >-
        Bind the declared caller fork and join to CUDA event dependencies
        between the caller stream and the side stream.
      reason: >-
        The declared input and output boundaries span those streams while
        the two attention submissions remain independent of one another.
~~~

MLA 对应短事实是“本次 Geo 按已声明的顺序构造，完整上传后 MAIN 才启动”。这条前向关系有源支持；不能附带声称 nonblocking side 已等待 caller 的 metadata 生产者，或下一次上传已等待前次 MAIN 的 table 读者。那两项若必须成立，应明示为外部前置/来源限制。

GDN 对应短事实是“每个本次消费的交接区域本次重写；共享 backing 在全部消费者及其在途读完成后才能复用；本实现不提供跨 stream 排他”。不登记 ws_cache 的扩容策略、分配 API 或析构规避方式。

若未来要求把这些外部关系做成可机器检查的图，需专门提出外部端点语法；caller.prior_done、Geo.upload_done 等目前不是合法的 phase 事件引用。
