# template_fa4_decode — 全量文字审计 v1.1

当前设计依赖文字来确定若干核心事实，尤其是 slice 读完后归还、提前 turn 放行、空 parity 收尾、cluster 清零可见性和两种 combine 的计算分配。但本例没有证明必须增加固定字段或事件：发现 1 类跨成员等待可由现有语法显式化，1 项现有整区复用机会；其余主要是动作/原语的有意开放语义、细粒度表达成本及来源证据缺口。

输入为审计目录冻结的 7 份英文 YAML 与中文 dispatch；全部状态为 `template`，root v9，其余 v7。它们是规范化示例，不是本轮 GPU/语义验收基线。规则依据为本目录冻结 README、DOC0–6、REVIEW 与 examples README，提示为 AUDIT_PROMPT v1.1。全文先读唯一文本，再核对全部本地定义、选择条件、接口、成员/资源/事件和 Realization；没有使用前轮结论代替阅读。

来源主入口为 `bcacdwk/KernelSchema@01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/solution.py::Model.forward`。从该提交精确取出 aka_solution 下的 operator.py、gqa_decode.py、gqa_decode_paged.py 及对应 reference/input；路径、SHA256 和行数见 [source_manifest](../work/template_fa4_decode/source_manifest.json)。没有将同目录另一套 solution 当作依据。

## 范围、节点与统一统计

270/270 精确组、777/777 使用位置已分类，其中 YAML 257 组/764 声明，dispatch 13 组/13 段。所有节点三个 Design 区域均本地定义，无 YAML alias 和 from_parent，因此 raw 声明量等于 raw 实例量，effective Design 与本地 Design 相等。raw 全字符串 123,431 字符；英文 YAML 开放字段 93,716 字符；中文 dispatch 1,250 字符。精确去重文本 42,180 字符。字符串含公式、引用与标识，不是纯自然语言字数。

| 节点 | 状态/版本 | raw 全字符串 | 开放字段声明字符 | verb 字段/字符 | effective Design 字符 | 非 FORMAL 核心字段声明 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| root | template/v9 | 24,614 | 18,862 | 33/6,987 | 18,263 | 57 |
| q1_cluster | template/v7 | 19,552 | 14,932 | 32/7,066 | 17,621 | 41 |
| q1_direct | template/v7 | 9,423 | 6,983 | 20/3,411 | 8,574 | 25 |
| q1_global | template/v7 | 21,193 | 15,874 | 36/7,495 | 19,165 | 46 |
| q4_cluster | template/v7 | 18,628 | 14,407 | 29/6,504 | 16,473 | 39 |
| q4_compact | template/v7 | 20,325 | 15,403 | 33/6,965 | 17,980 | 42 |
| q4_direct | template/v7 | 9,696 | 7,255 | 20/3,429 | 8,583 | 26 |

全节点 verb 共 203 次声明、41,857 字符；全节点 effective Design 为 106,659 字符，不能再与 raw 相加。root 开放字段 18,862 字符与全树 93,716 字符分开报告，避免复制量主导判断。

以下直接复用 summarize.py 的 `scope_statistics`。表中的“核心”严格指 `core∈{some,all}` 且 category 非 FORMAL；它是字段中含核心片段的计数/面积，不是不可压缩字数。数学和 dispatch.when 中的 MIXED 字段含正式公式，不能把对应面积误叫自然语言。

| 范围 | 精确组/字符 | 声明/字符 | 非 FORMAL 核心组/声明 | 核心字段字符面积 | 候选省字符（声明） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 主比较 W/T/R | 108/25,647 | 300/55,990 | 80/226 | 48,395 | 2,877 |
| Design 数学 | 97/10,138 | 404/32,017 | 12/32 | 7,432 | 677 |
| 硬件 | 9/617 | 9/617 | 9/9 | 617 | 0 |
| YAML dispatch.when | 7/1,971 | 7/1,971 | 5/5 | 1,843 | 364 |
| 中文 dispatch.md | 13/1,250 | 13/1,250 | 0/0 | 0 | 0 |
| 全部英文 YAML | 257/40,930 | 764/93,716 | 110/276 | 59,683 | 3,950 |

全部英文 YAML 有 110 组、276 次非 FORMAL 核心字段声明，覆盖全部 7 节点；其中 W/T/R 为 80 组、226 次声明。另有 q1 无 register redistribution 的摘要及 named-barrier API 绑定两组核心重要性列 `uncertain`，未算入该核心计数。是否含核心与是否需要扩语言是两件事。

逐节点覆盖：

- **root**：v9；q4 global：QueryTile=2/R=32，partial+PDL combine，FP32 nominal scratch 全读；单 softmax、64/168/256 预算、active/corr/tail 与 Ostage 复用两方读者核对。
- **q1_global**：v7；q1 global：QueryTile=1/R=16，数学与 root 全等但参数绑定本地；even/odd softmax 分别 2*j/2*k+1，Correction warp12:16，PDL/kernel_done 及 all-nominal 读取核对。
- **q4_compact**：v7；q4 compact：名义 S8/32，设备 WorkSplit，所有 CTA 提交但不同 N_r；combine 每线程重算并跳过 inactive payload，保留 no-op load/store 事件。
- **q1_cluster**：v7；q1 cluster：S2/4/8/16 参数成员，QueryTile1、双 softmax；成员独立 N_r，empty 参加统计，rank0 清零，XOR max/sum、BF16 reduce-add 与每成员独立载 Q 核对。
- **q4_cluster**：v7；q4 cluster：QueryTile2、S2/4/8/16 与本地选择政策，单 softmax+寄存器重配；empty CTA setup、cluster receiver 初始化与 output zero 分时，rank0 依赖需显式化。
- **q1_direct**：v7；q1 单 KV block，七 role/16warp，SoftmaxOdd 只发 EmptySum；仍有双 Y 槽及 empty parity drain，独立 K/V slice、Normalization alias 和 BF16 out 核对。
- **q4_direct**：v7；q4 单 KV block，每请求两个 QueryTile；六 role/12warp，Softmax 自发 EmptySum，register64/168/256；direct 数学和 q1相同但 bind=QueryTile2。

另核对 scalars 中目录外 2,111 个字符串、285 种精确值：包括 retention、bind/ref、shape/slot 表达式、动作/事件引用、执行模式、正式身份和 8 个 wiki pending 入口，未发现漏入目录的重要散文。未把这些全部额外加到开放字段分类分母。

## 全字段分类

每个 id 的具体原因、所有 occurrence 来源、finding 关联与唯一一项上下文 override 见 [annotations](../annotations/template_fa4_decode.json)。override 用于 q4_compact 的“all nominal slots”：此处枚举所有统计槽，payload 仍只读 active，不能按 root 的同句解释。

| 类别 | 精确组 | 去重字符 | 声明 | 声明字符面积 |
| --- | ---: | ---: | ---: | ---: |
| DUPLICATE | 7 | 561 | 16 | 1,408 |
| NONCORE | 0 | 0 | 0 | 0 |
| STRUCTURABLE | 0 | 0 | 0 | 0 |
| ESSENTIAL_TEXT | 35 | 3,841 | 102 | 12,582 |
| FORMAL | 34 | 4,808 | 157 | 15,666 |
| SOURCE_LIMIT | 0 | 0 | 0 | 0 |
| EXPLANATION | 117 | 7,139 | 321 | 17,296 |
| MIXED | 77 | 25,831 | 181 | 48,014 |
| UNRESOLVED | 0 | 0 | 0 | 0 |

整字段 NONCORE/STRUCTURABLE/SOURCE_LIMIT 为 0，不表示没有局部非核心、可结构化内容或来源限制：这些事实与核心定义处在同一字段，故归 MIXED，再在 finding/quote 中分开。UNRESOLVED=0 只表示能够判定文本职责，不表示 PageIndex 或近似误差等未决事实已经解决。

## 全部机制 findings

下表数量是关联组/使用位置，findings 可重叠，不能相加当独立缺口数。F15 只把 q1_global 的本地数学声明算作适用位置，其组 ID 在其他节点出现不构成其他节点的复用批准。

| Finding | 类别/缺口性质 | 组/适用位置 | 节点 | 结论 |
| --- | --- | ---: | --- | --- |
| F01 | redundancy/none | 121/442 | 7 YAML+dispatch | 摘要和必填短释义应保留简洁；两项 root 排列 strategy 与 worktile_order 完全重复。 |
| F02 | style/none | 14/14 | 7 YAML | 选择谓词、分块和 WorkSplit 是正式定义，不是因为写 Let/otherwise 就有语言漏洞。 |
| F03 | structural_gap/open_semantics | 57/174 | 7 YAML | 对象组合、初始化、no-op 与 primitive done 保证由动作契约定义；名称不自带这些保证。 |
| F04 | structural_gap/representation_cost | 23/74 | 7 YAML | 每 slice 读取退休与环复用可展开，但需定义更多 slice 动作及所有边，属表示成本。 |
| F05 | source_limitation/unresolved | 1/5 | q1_cluster, q1_global, q4_cluster, q4_compact, root | PageIndex 四槽环在 Lead=2 的 V 旧值捕获顺序未证明；不能补猜测边。 |
| F06 | existing_structure/none | 4/6 | q1_cluster, q4_cluster | cluster member0 清零→活跃成员 reduce-add 等待未显式化，现有条件 wait/成员事件可以表达。 |
| F07 | structural_gap/representation_cost | 4/8 | q1_cluster, q4_cluster | rank→split 与 XOR round 可在固定成员数枚举；变长成员/对等事件缺紧凑表示，不等于证明缺字段。 |
| F08 | structural_gap/open_semantics | 7/18 | 7 YAML | warp 合作求权重与每线程重算、内区免 mask 的主计算成本差异需短契约。 |
| F09 | source_limitation/none | 23/72 | 7 YAML | FP8/双 parity/BF16 split-rounding 保留；fast exp2/rcp、contraction 和 BF16 内部行为未知也保留。 |
| F10 | redundancy/none | 24/63 | 7 YAML | active 空域和仍有事件的 no-op 有区别；N_r=1 仍 drain 空 parity。 |
| F11 | structural_gap/open_semantics | 20/64 | 7 YAML | PDL/TMEM/setmaxnreg 提供实现保证；同图的 barrier API 名本身未证核心缺口。 |
| F12 | structural_gap/open_semantics | 15/16 | 7 YAML | 硬件 execution/feature/stack 是有意开放定义；库容量和 wiki pending 限制不自动消失。 |
| F13 | structural_gap/open_semantics | 17/25 | 7 YAML | 返回对象同一性和必传 None 属 ABI；已有 math/range/meaning 能明确，shape 相同不证明别名。 |
| F14 | source_limitation/unresolved | 13/22 | q1_global, q4_compact, root | global inactive scratch 必须有限；首次清零不是每次 handoff.initialize。完整冷启动/失败路径仍未覆盖。 |
| F15 | existing_structure/none | 61/68 | q1_global | q1_global 数学与直接父 root 完全一致，可用 from_parent；不能对其他 sibling 套用。 |
| F16 | noncore/none | 1/1 | q4_compact | compact B=12/S=8 的不可达源备注可移出 YAML，保留全部可达参数。 |

F03/F08/F11/F12/F13 的 `open_semantics` 是对当前核心定义载体的观察；F04/F07 为 `representation_cost`。两项 unresolved 是来源/范围问题。**已证 `missing_field_or_event` 为 0**；这不等于“因为允许自由文本所以结构完整”。普通依赖、别名存储、跨 phase 可见性及首末边界已有结构，而图自身仍不能替代动作和集体的语义定义。

## 实际可用的结构表达及边界

F06 的源依据：paged.py:1392 的输出 store 在 Ostage rendezvous 后 acquire zero_done；:1848 起 member0 在 ramp 清零、GPU fence、向每成员通知，且 member0 因 seq_lens>0 不会 early exit。当前两个 cluster YAML 已有 `Correction/zero:output` 身份，却仅在 LoadV 的 verb 中说等待它。下面只展开相应增量；保留其他 load/issue-turn/data 边和现有 zero/store 契约：

```yaml
info:
  verb:
    wait:
      execution_mode: serial
      meaning: Acquire member 0 output-zero visibility via the existing zero_done protocol.
pipe:
  repeat: i
  steady: ['load:Vblk(i)', 'yield_turn:V(i)']
  epilogue:
    - repeat: active
      steady: ['wait:OutputZero(active)']
    - 'store:output_tile'
waits_for:
  data:
    'wait:OutputZero(active)@issue': ['Correction/publish:Ostage@done']
    'wait:OutputZero(active)@done': ['Correction[0]/zero:output@done']
```

这是将源 store 内已有 acquire 暴露为动作的表达建议，不是增加全体启动等待：清零不阻止 K/V ramp；空成员 active 域为空而原 no-op store 事件仍保留。展开会把实际传输边界从原复合 store 中分出，须作为完整局部改写复核；没有计入 prose 删减字符，也没有把“加一个未定义 wait 名称”当作解法。

F15 的现有复用片段仅适用于 q1_global：

```yaml
design:
  math_reshape: {from_parent: root}
```

root 与 q1_global 的 math_reshape 经 YAML 解析完全相等；QueryTile=1、SplitCount 和 WorkSplit 仍由 q1_global 本地调度绑定。该节点数学区 raw 全字符串可由 6,436 变为 `root` 的 4 字符，effective 数学仍为 6,436；这是结构复用潜力，不并入本次 quote→replacement 节省。q4_direct 与 q1_direct、两个 cluster 的数学也有大段相同，但不是合法直接父同区，不建议为了压缩改树。

已有结构成功承担：PDL 启动用 launch_ready，交接 ready 用 kernel_done，consumer milestone 绑定 wait.done；Maximum/Ostage 物理别名由 resource.view 表达；root/global Ostage 等 Score K-read 与 Value V/P-read 退役，TMEM retire 等所有 tail capture；q1 even/odd 有分别明确的索引边。这些事实不可因长 verb 中重述就被误判为缺字段。

尚依赖开放定义：release 的源读完保证、pair 的 hi/lo 成分、empty parity 的值与信号、retire 的实际释放，以及 XOR peer 转移。F04 未把全块 done 偷换成 per-slice read 完成，F07 未伪造 `Role[r xor 2^j]` 成员语法。把内容移进另一个 strategy/choice 不算结构化。

## 保守短写候选

29 个互不重叠 quote→replacement，适用 58 次声明，按声明净省 **3,950 字符**，为英文 YAML 开放字段的约 **4.21%**。按精确候选各计一次为 2,016 字符；中文 dispatch 无删除候选。W/T/R 省 2,877，数学摘要省 677，when 解释尾句省 364，None 单值重复 range 省 32。主分类面积绝不作为删除量。

主要可复核示例（完整29项、适用 occurrence 与依据见 annotations）：

| 原 quote | 候选 replacement | 依据与合法性 |
| --- | --- | --- |
| `Enumerate batch outside query_tail outside split; split changes fastest.` | 空串，省略 optional strategy | root:241 已完整登记排列；不删 worktile_order。 |
| `Python None only` | 空串，省略 optional range | 保留 root compute 的 python_none 单值定义及两个输入 dtype；没有增加 default。 |
| `Apply upstream normalize and cast once to BF16 when overwriting out.` | 空串，省略 generic produce 覆写 | 保留 produce:output 动作、normalize 公式及 target.output；同一 verb map 其他定义保留。 |
| `Join Correction's existing FinalSum rendezvous; completion admits this role's sum read and Correction's final Y capture.` | `Wait for FinalSum.` | 保留 serial wait、FinalSum→wait 与 wait→capture 的显式边，字段仍非空。 |
| `T is in TMEM columns and Csm is in bytes; these are the values returned by the original sm_100 helper calls in CUTLASS DSL 4.6.1.` | `T and Csm are sm_100 helper limits in CUTLASS DSL 4.6.1: TMEM columns and SMEM bytes.` | 7节点使用的6个精确组；具体容量、公式、setup/empty 特例和库版本均保留。 |
| `FinalMaximum releases the final m before the last parity correction.` | `FinalMaximum publishes the final m.` | 原 Correction pipe 已把发布置于 corr_final rescale 前；保留发布含义和全部事件。 |

其余候选仅删 when 的同机制解释尾句、compact 不可达备注、active 空域重复说明，或缩短保留全部详细定义的 thesis。没有同时删除候选依赖的公式/资源/事件依据；没有因“默认同义”取消空分支事件。必填 meaning/thesis/reason 不直接删成空值。Startup 的最长字段为 1,352 字符，仍含容量、初始化、预取与条件寄存器事实，不能按长度截断。

## 边界核对与限制

- **空/首次/末次**：N_r=0 的 active/head/tail 为空，setup/neutral统计与 epilogue事件仍存在；N_r=1 必须清空、信号并 drain parity1；N_r=2 无中段 correction，N_r>=3 的 corr_final 与 FinalMaximum 发布位置分开。所有 tail capture 都在 retire 前；不能把 no-op 当动作不存在。
- **跨工作**：所有 worker preassigned count=1，没有同 worker 连续多 worktile 的 persistent 链；不据此推断资源能跨 kernel 保留。kernel_done/PDL 交接与新 worker 身份分开。
- **全部读者与别名**：KV→Ostage 的 K 与 V/P 读者均需退休；TMEM→TailNumerator 捕获后可先释放 TMEM，normalization/Ostage 仍可后做。PageIndex 的旧值读者保护仍未知，未添加可能加强顺序的猜测 reuse 边。
- **跨调用/捕获回放**：global 读 inactive scratch 而 compact 跳过，首次初始化来自 input.py:83 的 workspace zeros；没有将其改成每调用 initialize。捕获固定 host selector/标量/地址，设备 seq_lens 内容按当次读；纯捕获没有完成事件。source operator.py:474–582 的 native 建立会再次执行 ffi 和校验/恢复，:652–843 有 native失败fallback；当前 captured算子设计没有完整表达冷启动/异常范围，不宣称覆盖整个 Model.forward。
- **数值与来源**：fast exp2/rcp 独立界、FP32 contraction、BF16 reduce-add 内部算术、PageIndex 顺序依据仍未知；硬件 wiki pending。未运行 GPU，也未直接核对外部 CUTLASS helper/ISA 全文。
- **验证范围**：统一 summarize.py 已通过 270/270 id、7/7节点、证据行号、occurrence、quote唯一性及非重叠校验；这是机械覆盖和格式校验，不是短稿语义等价证明。

机器结果：[audit_summary](../data/template_fa4_decode/audit_summary.json)。人工标注生成脚本与编号全文视图留在 [work](../work/template_fa4_decode/)，仅作为复核证据，没有修改原 YAML、规范、Runner 或 AGENTS.md。
