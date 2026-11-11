# extract_fmha：pipe 现状审阅

已逐组读完 51 个精确 pipe 组，核对全部 43 节点、157 个 phase 上下文和 159 个 role 使用处；同组成员分别核过 domain、契约、等待及数学/调度关联。26 approved、4 review_candidate、13 needs_authoring 均纳入；root 标记 authored=false，但有稿。批准状态与结构可计数不构成完整设计证明。语义依据为冻结的 [微流水线规范](../../../schema_extract/archive/20261109-ten-operator-text-audit/rules/schema_docs/5.design.tile_pipeline.md)，完整逐组及成员定位见 [审阅记录](../reviews/extract_fmha.json)。

| 口径 | roles | 块 / 嵌套块 | 静态动作 | 最大深度 |
|---|---:|---:|---:|---:|
| A_decl 声明 | 159 | 286 / 127 | 1254 | 3 |
| B_local 本地展开 | 159 | 286 / 127 | 1276 | 3 |
| C_effective 有效展开 | 159 | 286 / 127 | 1276 | 3 |
| root（三口径） | 1 | 3 / 2 | 15 | 3 |

43 份管线均本地定义。动作差额 22 全来自 fused_d128_group 的 Even/Odd 标量别名展开，不是运行时迭代。159 个顶层块均无 repeat；127 个子块全是 repeat+steady，88 处引用域名、39 处定义局部域。所有子块都在 steady，prologue/epilogue 内无循环。115 个顶层有 prologue，107 个有 epilogue，44 个仅有 steady；4 个列表共有 8 对相邻循环，7 个列表混合动作与循环。

首先影响解释的是已有稿件边界：5 份草稿 when 为空，分支适用域未闭合；tcg_d256_flat_cp_wide 的选择与数学写四组宽归并，调度仍称 narrow、管线仍为 4 warp 和 SharedDen[128]，故其双层循环不能当成宽归并实现证据。4 个 tc_split 稿保留最大值覆盖孔与临时 O/LSE 写；28 个 flat/segmented split 稿记录空分支入口读取可越出逻辑扫描表。12 个 D128 xpose 稿记录空 split 的 Query/PartA 共区写入竞争，其中 approved 的 flat_tma_xpose_narrow 另有 prepare→stage_partials reuse 边，契约同时承认源路径缺少前置 join；声明约束与源保证不可混同。fused_d128_group 的 cold 拷贝也明确缺少跨 role 完成证明。

主路径有清楚的族界：7 个节点只有 main，36 个节点分成多次提交；其中 flat 14 个为四 phase，segmented 14 个为五 phase，fscan 8 个为三 phase。绝大多数 phase 是一个协作 Main；同 phase 多 role 仅 fused_d128_group 的 Even/Odd 与 tcg_d256_segmented_cp_wide 的 Output/Partners。后者是局部归并分工。所有 worker 只领一份 worktile、无 refill；Ns/Ngm 来自 HostPlan，局部组数随 split 区间变化。这里 flat 指平直扫描，未出现 flatten 域或切片 repeat。

同形的搬运主干还包含真实布局差异。D128 direct 的有效行形成前缀，完整空闲 warp 可略过部分标量读出；xpose 的有效行分散在各头的行段，全体行仍参与相关计算，输出要经过共享暂存和转置。相同组号不会记录这些读者范围与存储复用差异。D256 同形主体中的载入则分别依赖每线程复制组或事务屏障，尺度原子和主体载荷并非始终共用完成依据。D128 的长区间路径提前发出下一组矩阵计算，短区间在当前组内发出；fscan_tma_xpose_wide 草稿另写出了组数上界，因此它的首发和提前发射动作确实没有出现在管线中。

循环变量不是统一时间轴。78 处 g 遍历当前 attention/group-max 区间；Even/Odd 各用 j 映射 g=cLo+w+2*j，长度分别向上/向下取半。root 与 fused_d128_cp 的 z 是两次 64-column score 读出；5 个 TC 主体中的 t 是 16-key 概率/PV 分量，分量内仍包含两个 8-key 半片及全部输出通道。combine 的 c 是通道片，k/j 是八个 split 一批；D256 固定 c∈[0,2)。实际 repeat 无 p/i 绑定，数学里的查询坐标 i、归并组 p 不因此成为 pipe 循环。

扫描有三种已用粒度：TC flat 用一个 j 批序列，契约把它映射为 64/16/1；D256 flat 用三个相邻循环分别走这些批次；D128 flat 的 scan:MState 则在一个动作内完成全部批次。segmented 首段放 prologue，后续 c 批先比较全部摘要，再填常数前缀并重放选中后缀；D128 xpose 把这整段逻辑收在 scan 动作中。条件只改变动作内部实际读取范围，零跳转掩码仍写扫描状态，不等于删除动作。

fscan 把有界前缀扫描放进 split 的 prologue 动作，随后 g 同时推进扫描与 QK/PV，写未加权 partial 和 PartM，combine 才加权；没有独立扫描 phase 或全局逐组快照。最后 split 即使为空也有 MTail 责任。7 个 fscan 稿单列 tail，tma_xpose_narrow 将它并入 stage_partials。窄 combine 的全区间顺序加法与宽 combine 的分组、树形分母、共享发布和组零输出不同；宽路径多次批加载常在单个 reduce/partials 动作内，部分窄路径才显式写双层 repeat。

头部承担初始化、首批搬运、首 QK 或扫描入口；尾部承担最终 PV 完成、分母合并、partial/output 写与释放。相同区域名不固定动作粒度：root 的 PV 完成落在后继迭代 retire 和最终 drain，cp 变体可在每轮退休。159 个 role 默认 serial，15 个使用处另有动作级 async 覆盖。单个动作仍可承载四/八次矩阵收缩、32-channel 读改写、批加载后有序求值或 collective；显式 repeat 数不等于数学求值次数。尾部预取无 payload、空 commit、未被后续矩阵使用的旧 atom 写各有不同契约；空 steady 也不取消头尾协议、零 partial 或 cold 的完成缺口。

以下为已存在的原片段，动作名仅用于定位。

[tc_fused.yaml:380](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_fused.yaml#L380)，位于 g 主体内：

```yaml
- repeat: {t: [0, "KeyGroup/16"]}
  steady: ["probability_value:Chunk(g,t)"]
```

[tcg_d256_flat_cp_narrow.yaml:482](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d256_flat_cp_narrow.yaml#L482)，同一 steady 的相邻循环：

```yaml
- repeat: {j: [0, "floor(GCount/64)"]}
  steady: ["batch64:GroupMax(j)"]
- repeat: {j: [0, "floor((GCount%64)/16)"]}
  steady: ["batch16:GroupMax(j)"]
- repeat: {t: ["64*floor(GCount/64)+16*floor((GCount%64)/16)", GCount]}
  steady: ["tail:Mscan(t)"]
```

组级片段只展示共享动作树；机器记录为每个成员保留自己的块位置、角色契约和阶段域引用。源行号不同、布局不同或草稿选择条件缺失的成员，均没有用代表节点的定位替代。区域中的书写先后还要结合动作完成含义和已列等待阅读，不能仅凭相邻位置认定异步载荷已经可读，或认定尚在途的矩阵操作已经释放了全部输入。

本轮为冻结材料的静态现状审阅，未运行 kernel、展开运行时事件或改动原始 YAML；上述来源限制不转写为语法需求。
