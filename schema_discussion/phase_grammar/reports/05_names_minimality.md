**三类 phase 定义：保留概念，缩短名字，补清两处规则。**

建议采用 `storage / index / event`，仍是三个可选具名映射。本次样本没有给出必须增加第四种 phase 公共定义的反例。三者分别记录分配身份、索引身份和局部事件接口，彼此不能替代；执行和交接仍需要 role、动作、pipe、waiting 及上游 flow。这个结论限于本次覆盖，不证明所有 kernel 完备，也不表示每个 phase 都需要这三张表。

这是独立的命名与最小性判断，不是用户可用性测试。未读取其他本轮代理报告或综合 README。完整证据、源码 SHA256、阅读范围和草图见 [05_names_minimality.json](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/data/05_names_minimality.json)；[生成脚本](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/scripts/05_names_minimality.py)核对引用原文与固定源码 hash。

**实际覆盖。** 六个 root 均完整读取相关 work_schedule、tile_pipeline、realization、ledger 与选择域；不是重新证明完整 Contract 数学。精读四个分支的相关上下文。所选均为暂停时历史 approved，旧 verb/资源/pipe 写法不算本轮语法合格实例。已读指定的 SCHEMA_VERBOSITY.md 与 SCHEMA_DISCUSSION.md，再回查 YAML 和源码。172 份有设计稿是语料背景，不是本人的语义审阅数量；scaffold 和未开始节点不作成功设计。

| 算子 | root 覆盖重点 | 分支挑战 | 核对源码 |
| --- | --- | --- | --- |
| [GDN v11](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:320) | 单事件消费等待；单 role 多槽与最终状态覆盖 | — | [gdn_kernel.cu](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2211) |
| [GEMM v4](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml:194) | 跨 worktile 连续编号、SMEM/TMEM 双环、写源提前归还 | — | [fp8gg_umma.cu](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu:1015) |
| [MLA v4](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:286) | 动态 worktile、共享索引与真实队列分离、host 建表 | — | [固定 solution.py](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1456) |
| [Sparse v13](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:277) | 一 role 的环/overlay、跨 kernel 全局 partials | probe_hidden_packed v2：外部 fork/join、独立资源实例 | [dsv4_kernel.cu](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6140) |
| [FMHA v9](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:383) | 单 role 的 g/z、容量与 view | tcg_d128_flat_tma_xpose_narrow v4：空 split 覆盖边界 | [fmha_tcg.cuh](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:5225) |
| [MoE v7](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:365) | 单 role worktile；条件 drain；跨 kernel 同名资源 | direct v1：同 worker 跨 phase；verify_fused_big_flat v1：七角色事件组 | [nvfp4_moe.cu 固定副本](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1929) |

**1. 删除存储账会把必要关系搬进文字，而非消掉重复。** `existing_structure`

MoE/direct 的一次 launch 含 prelude、direct，`order: per_worker`；Census 留在 prelude 定义的 SMEM，Exchange 先存 WarpCounts，后存 FinalRoutes。[YAML](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:188)与源码的 `s_nvalid`、`s_warp[16]`、后续 [s_warp 复用](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:2217)对应。删 resource 后，数学标量 Census 和事件名字都推不出它的片上归属，也推不出 Exchange 的共址。把位置/容量放回 math 或 describe 只是换地方登记。

反过来，MoE/root 的 FC1、FC2 分属两个 kernel；`PayloadA: *payload_a` 仅共用字段内容，不能解释成同一 SMEM 活到下一 kernel。同 worker 跨 phase 应合账，不同 worker 实例应分账，改名 storage 不改变这条边界。登记存储从来不自动建立依赖。

**2. “共享索引”不应要求存在第二个 role。** `underspecified_rule`

现行 [docs5 的填写条件](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/rules/5.design.tile_pipeline.md:328)只说多个 role 或资源 view 使用时登记 domain，但 [MoE/root gather](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:617)只有一个 role、没有资源表，仍需要 `t: worktile` 表示上游静态步进的工作序列。源码 [k_gather_f](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1189)确实持续处理后续工作。改成普通局部整数 for 必须重抄 worker_assignment 的起点、步长和终止，未减少概念。

建议填写条件增加“需要具名 worktile/flatten 有序序列”，名字用 index。单 role 的 FMHA `g` 还供 Key/Value 槽位视图使用，应保留；`z:[0,2]` 仅在一个局部循环里使用，可以改成 `for(z=0:2)`。MLA 的 w/n 是工作与版本编号，真正的可变队列是源码 `atomicAdd(P.queue,1u)`；不能把域声明解释成共享可变计数器，也不能隐含各 role 锁步。

**3. 事件接口值得保留，但应明确消费侧的分配规则。** `underspecified_rule`

[GDN prep_ready](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:550)只命名一个消费等待的 done。源码先 [sreg_load，再 griddepcontrol.wait](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2211)：省掉 wait_at 会把等待移至消费入口，损失独立输入准备的安排。允许 work_schedule 直接写 role/verb 事件可以少一层名字，但会把局部动作接口嵌进调度；不是消掉事件关系。

[MoE/verify_fused_big_flat 的 guard_done](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/verify_fused_big_flat.yaml:981)聚合七个 role 的 guard.done，并被 Clean.wait_at 引用。源码 [ZGUARD](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:208)保证每个实际读取先完成 PDL 等待。建议规则直接说：生产 ready 要全部相关生产事件满足；消费 wait_at 要对列表内**每个**实际等待/读取端点施加就绪约束。仅令“全部消费事件已发生”的聚合时点晚于 ready，会容许较早成员先读取。这是规则逻辑反例，不是声称该 MoE 分支真的发生过早读取；其启动/PDL 传递关系另有保护。

event 可以命名一个事件或多个角色事件，位置可以是中段发布、释放、等待完成或读取开始；不是产物 alias，也不生成通知。沿用显式 role/member、索引和上游区域匹配即可，无需 all/any 字段。七个 role 都在一个 CTA 内，不能用 `Epilogue[*]` 代替七个名字：星号选择 worker 成员，不是 warp 或 role。

**4. 存储映射和字节预算要保留，真实预留与设计账要分开。** `existing_structure`

FMHA/root 的单槽 128×256 FP32 TMEM 为 **131,072 字节**；其中 ScoreHalf、scale 区、NumState、Prob8 的区域不能仅由各数学 shape 推出。[资源定义](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:495)中的 part/dtype 是设计选择，不是普通 lane 地址。xpose 分支还用 **81,920 字节** Q/K/V arena 容纳 **67,584 字节**含行间距的 staging；源码 [XPITCH 与容量断言](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:5225)支持保留共址和容量，而不要求在 phase 里重建线程地址语言。

按 root 登记项，SMEM 合计 **83,968 字节**；源码初始请求 **102,912 字节**，差 **18,944 字节**来自未用的 P 预留、Q-scale 预留、exchange 预留和对齐余量（16,384+512+1,024+1,024）。[realization](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:688)还说明后续 occupancy 调整，因此 102,912 也不是最终运行时请求的承诺。不能用 phase 总字节数冒充完整物理占用。

本轮不建议改变 total_size 必填规则。它继续作为可生成、可核对的派生值，不增加单槽 bytes 或第二套预算；样本支持“容量应易读”，没有实证证明“必须由作者手抄派生数”更好。GEMM 的“源已读完可覆写、目标仍在途”属于 `concise_semantics`，要保留短完成契约及必要 release；实际额外 reservation 属于 `other_layer`，影响驻留就留在 realization。

**5. 源码未提供的排序，不能靠改名字补出来。** `source_or_draft_issue`

FMHA/xpose 已明确写出 [prepare.done → stage_partials.issue](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml:642)，并保留空 split 缺 pre-staging join 的来源限制。源码的 [Query 写入](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:3351)与 [PartA 覆盖后才执行的 barrier](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:5412)是不同边界。GDN 最终 mini chunk 也有 stage-0 Kc 读者与 FinalState 写者的退休缺口。存储账能暴露冲突，但不会提供同步。

MoE root 的条件 drain 已在 [当前稿](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:781)中成为显式前驱，对应 [hoist 源码](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:4855)。迁移时采用现有 if、await、released 即可。这是 `existing_structure` 的修订方向，不应把旧报告中“只藏在文字”的历史状态重报成新增能力缺口。

**6. 真正可交给 Coder 的，是删除后不改变约束的细节。** `implementation_detail`

[GDN final_store](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:661)甚至说明最后一轮没有 next load、随后赋给 current 的值未使用；对应源码 [cur = nxt](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2855)。这句话可删。仍保留转置目标、下一批读取先于当前批写出的重叠关系、staging 存续与完成边界，用一句动作说明即可。普通线程地址、barrier 编号、不可达 guard 不新增字段；有因果作用的容量、真正发布/复用 join 不属于这一类。

**7. 外部准备与缓存是上层边界问题，不要扩张这三类定义来兜底。** `other_layer`

MLA 的 [Geo 上传](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1895)在 [主 launch 前等待](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:2007)；Sparse 的 [fork](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6153)和 [join](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6188)连接调用者前后工作，prefix 与 remainder 不必相互等待。GDN 的 [ws_cache](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3254)则是跨请求全局分配。这些真实关系不能被虚构为 worker 的片上 storage 或局部 event。

工作表的工作排序归 work_schedule，外部供给/输出可用性归公共调用边界与调度，具体原语和必要缓存约定归 realization/调用约定。指定旧报告已指出这些问题；本报告没有把它们宣称为首次发现的“从未提 issue”。本轮新建议是第 2、3 条的规则澄清。没有发现属于 `missing_capability` 且应新增 phase 公共定义的反例；不是说上层边界已全部解决。

**名称比较。** 各自选择，不追求押韵或三词形式对称；括号为字符数。

| 现概念 | 候选 | 覆盖范围、冲突及误读 | 选择 |
| --- | --- | --- | --- |
| resource | resource (8) | 过宽，容易期待计算单元或 barrier 等资源也登记 | |
| | storage (7) | 同时覆盖 SMEM、TMEM、寄存器标量与持久状态；定义仍限定 worker 片上存储。与 handoff.storage 同属存储语义，层级可区分 | **storage** |
| | buffer (6) | 短且常见，但偏向数组暂存，容易缩窄寄存器状态与混合区域 | |
| domain | domain (6) | 数学准确，初读用途较泛 | |
| | index (5) | 直接提示索引名；能容纳范围和有序序列，须明确不是当前游标 | **index** |
| | range (5) | 适合整数区间，难覆盖 worktile/flatten；上游已有 range 表示规模允许范围 | |
| milestone | milestone (9) | 常见但易让人想到重要进度或阶段末点 | |
| | event (5) | 单事件、复合事件与消费等待都自然；必须明说静态引用命名，不创建 CUDA event | **event** |
| | marker (6) | 任意位置较自然，但易像调试标记，不突出多事件共同满足 | |

**真实片段的最小草图。** 以下为建议，非现行可直接提交稿。为只比较命名，先将旧 space/member_slot_shape/slot_count 归一化为本轮已有字段；动作必要关系和省略的分区正文仍须保留。

MoE/direct 的现名片段：

```yaml
resource:
  Census: {depth: 1, location: SMEM, shape: [], dtype: int32, total_size: 4}
  Exchange:
    depth: 1
    location: SMEM
    shape: [element: 16]
    dtype: int32
    total_size: 64
    view: {WarpCounts: {}, FinalRoutes: {part: {element: [0, K]}}}
```

改名只换第一层为 `storage:`，并把 handoff 的引用改为：

```yaml
Census: {producer: prelude, consumer: [direct], storage: tile_pipeline.phase.prelude.storage.Census}
```

`order: per_worker` 原样保留。Census 与 Exchange 各只有一份定义，不在 direct 再分配。

FMHA/root 原来登记 `domain: {g: [cLo, "cHi+1"], z: [0, 2]}`，建议只留：

```yaml
index:
  g: [cLo, "cHi+1"]
```

内层 `for(z)` 改为 `for(z=0:2)`，原两半 QK/读取的完整动作列表和同步保证原位保留。Key/Value 的 `(g-cLo)%2` 映射不变。MoE 的单 role `t: worktile` 则继续在 index 登记。

GDN/root 的单事件接口，按本轮 verb 转写后，现名是：

```yaml
chain:
  milestone:
    prep_ready: ["{done MAIN[0]::await:Prepared}"]
```

建议仅改为：

```yaml
chain:
  event:
    prep_ready: ["{done MAIN[0]::await:Prepared}"]
```

`wait_at: chain.prep_ready` 不变，仍引用 initial_state/PRE0 独立工作之后的那个已有等待。完整 before/after 草图与逐条原文证据保存在 JSON 中。
