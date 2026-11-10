# template_fa4_prefill：第2轮全量文字审计（v1.1）

完整覆盖11个template节点、11份YAML、dispatch.md及200个精确文本组。每组所有出现位置和节点组合均已审阅，没有未审阅项或occurrence override。主要结论：当前实例仍依赖自由叙述确定部分核心执行事实，但多数依赖是规范有意开放的动作/硬件/ABI语义；已找到现有结构可承载的片段，没有证明必须新增字段或事件。

**身份与来源。** 冻结manifest的git_head为`259cdbb855c1949ce28032acbb76f81e52b4edfe`，11份输入SHA256均已与manifest核对。源均为`01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/solution.py::Model.forward`；通过git show取得。完整亲读solution/reference，针对机制亲读同commit的interface、sm100_hd256_2cta_fmha_forward及flash_fwd_combine相关段。规则以冻结README、DOC0–6、REVIEW及examples README为准。模板只表示规范化示例，不是GPU/语义已验证基线。

机器明细：[annotations](../annotations/template_fa4_prefill.json)；统一统计：[audit_summary](../data/template_fa4_prefill/audit_summary.json)。人工阅读批次、父子完整diff及源证据保存在[work](../work/template_fa4_prefill/)。

## 统一统计

字符按归一化空白后的len计，不是token，也不能把含公式、引用、ID的字符串总量叫纯自然语言。下表为开放文本字段范围；数学标识符step/ledger.by与选择谓词仍在全字段分类内。

| 范围 | 精确组 | 去重字符 | 声明数 | 声明字符 | 非FORMAL且含核心：组/声明 | 保守候选减少声明字符 |
|---|---:|---:|---:|---:|---:|---:|
| 全量含中文索引 | 200 | 29,540 | 744 | 85,095 | 56/213 | 2,473 |
| 英文YAML | 183 | 27,688 | 727 | 83,243 | 56/213 | 2,473 |
| W/T/R合计 | 70 | 12,866 | 259 | 38,574 | 44/182 | 2,407 |
| 数学Design | 56 | 6,759 | 399 | 35,514 | 3/22 | 0 |
| 硬件execution/feature/stack | 9 | 624 | 9 | 624 | 7/7 | 0 |
| YAML选择谓词 | 11 | 5,028 | 11 | 5,028 | 0/0 | 0 |
| 中文dispatch.md | 17 | 1,852 | 17 | 1,852 | 0/0 | 0 |

W/T/R细分（同一字段口径）：

| 区 | 精确组 | 去重字符 | 声明数 | 声明字符 | 非FORMAL含核心组 |
|---|---:|---:|---:|---:|---:|
| W | 27 | 5,723 | 59 | 11,903 | 15 |
| T | 35 | 6,443 | 130 | 20,546 | 25 |
| R | 8 | 700 | 70 | 6,125 | 4 |

全部YAML字符串声明/实例均为 **106,587** 字符；root为 **11,974**，其中开放文本声明/实例 **9,410**。全部有效Design字符串实例 **86,634**，root为 **6,855**（math 3,264；W 977；T 2,614）。effective只统计Design且不与raw相加。

本样本没有anchor/alias；每处复制均为独立声明，所以raw声明量=实例量。三个Design区全为本地定义，无from_parent，因此当前raw/effective Design相同。Contract仅root存在，分支引用共享契约；不存在把分支缺少Contract误算为更简洁的比较。

## 全字段分类

| 类别 | 精确组 | 去重字符 | 声明数 | 声明字符 |
|---|---:|---:|---:|---:|
| DUPLICATE | 6 | 420 | 22 | 1,735 |
| NONCORE | 1 | 48 | 1 | 48 |
| STRUCTURABLE | 0 | 0 | 0 | 0 |
| ESSENTIAL_TEXT | 13 | 1,029 | 52 | 5,303 |
| FORMAL | 34 | 9,778 | 119 | 20,955 |
| SOURCE_LIMIT | 1 | 71 | 2 | 142 |
| EXPLANATION | 100 | 7,342 | 374 | 21,219 |
| MIXED | 45 | 10,852 | 174 | 35,693 |
| UNRESOLVED | 0 | 0 | 0 | 0 |

上述字符是字段面积，不能当可删除字数。0个整字段STRUCTURABLE不代表无结构化机会：FinalSum别名、TMEM统一资源等均在MIXED字段中，4条existing_structure finding分别记录。FORMAL保留公式与选择谓词，不把这些当“未压缩”。

## 每个节点的覆盖

| 节点（均template） | 版本 | 本地精确组 | 开放文字声明字符 | 非FORMAL含核心组 | 关键绑定 |
|---|---:|---:|---:|---:|---|
| packed_persistent | 5 | 57 | 6,604 | 18 | v5；parent root；Cpad原请求序+4步binary；t/n跨工作环、Q与Y退役、FinalSum复用、RequestMap一次发布已核对。 |
| rect_joint_load | 4 | 54 | 6,265 | 16 | v4；parent root；普通math绑定不变；e=flatten(p,d)，N=1 token次序、e+4约束、K/V映射和单Load realization已核对。 |
| root | 6 | 101 | 9,410 | 27 | v6；共享Contract，普通math，rectangular static双loader；i/d环、S/P/Stats别名、首/末MMA、Correction同CTA发布与全部读者退役已核对。 |
| single_persistent_dual_load | 5 | 57 | 6,592 | 18 | v5；parent packed_persistent；B=1选择及identity/linear mapping；双loader和t/n资源绑定适用于子选择域。 |
| single_persistent_joint_load | 5 | 55 | 6,652 | 16 | v5；parent single_persistent_dual_load；e=flatten(t,p,d)与n=flatten(t,i)互相坐标绑定，跨工作Query与Y等待、单Load已核对。 |
| small_persistent_lpt | 5 | 57 | 7,125 | 18 | v5；parent packed_persistent；B=2..4排除共享alpha特例；LPT网络、linear map、固定步长、t/n各资源与原math已核对。 |
| small_persistent_shared_rescale | 5 | 57 | 7,128 | 18 | v5；parent small_persistent_lpt；B=4/T=16384；Stats数学shape及物理view均1列，Softmax单次alpha与Correction直接读取一致。 |
| small_static_lpt | 4 | 56 | 6,990 | 18 | v4；parent root；Cpad/static一工作，WorkCoord两int一次发布与两loader入口依赖；无t域或跨workY释放。 |
| small_static_shared_rescale | 4 | 56 | 6,993 | 18 | v4；parent small_static_lpt；共享alpha的公式/Stats view/Correction读取同步改变，WorkCoord与static边保留。 |
| splitkv_dual_load | 4 | 82 | 9,698 | 24 | v4；parent root；split数学/两launch和global handoff；active/empty=0/1分支、FP32 partial/LSE、combine四槽/三预取/尾域/未写payload已核对。 |
| splitkv_joint_load | 4 | 80 | 9,786 | 22 | v4；parent splitkv_dual_load；split数学及combine相同；attention改e token单Load，N=0各guard与N=1首/末映射均已核对。 |

root单独有101个本地精确组、27个非FORMAL含核心组；全树200组中56组含非FORMAL核心。W/T/R有44组、182处声明、覆盖全部11节点依赖这类核心文字。该计数包含MIXED中的核心片段，不表示30,148个W/T/R声明字符全都必要。

## 全部机制结论

每条完整finding的unit IDs、全部出现节点、核心删除反例、现有语法排查与证据见annotations；下表列出全部去重机制，组/声明/节点数允许重叠，不能相加。F03的12个位置节点包含dispatch.md。

| Finding | 类别/缺口性质 | 组/声明/节点 | 结论 |
|---|---|---:|---|
| F01 | structural_gap/open_semantics | 6/6/1 | ABI、原地输出身份和连续布局有意由Contract开放正文承载，不能从同shape/dtype或out_after名字推断。 |
| F02 | noncore/none | 1/1/1 | workspace初次清零在本既定实现不产生工作或读取。 |
| F03 | style/none | 49/135/12 | 必填摘要/标题/理由及中文导航大多没有独占核心，但不意味着全部应删除。 |
| F04 | redundancy/none | 2/3/1 | seq_lens正值与单值None的range是同一当前契约中重复约束。 |
| F05 | style/none | 11/11/11 | 11个when都是正式谓词；共同派生式跨文件重复，但父when不继承且当前无独立dispatch宏。 |
| F06 | structural_gap/open_semantics | 9/9/1 | 硬件execution/feature是有意开放的组成与机制定义；当前wiki占位不补定义。 |
| F07 | source_limitation/none | 15/85/11 | 长数学段的分支、FP8舍入/FP32分母、归约分组与快速数学未知界限应分别保留。 |
| F08 | existing_structure/none | 56/399/11 | 7个直接父子对的完整math_reshape相同，可用现有整区引用降低raw复制；局部不同区不能用patch。 |
| F09 | structural_gap/open_semantics | 11/22/11 | 非单调query折叠、LPT具体排列和request-prefix解析仍由strategy中的正式关系/开放算法表达；仅worktile_order ascending并不定义它们。 |
| F10 | structural_gap/open_semantics | 5/27/11 | 输出row的member切分未表达K按token、V按channel-half的不同输入供应映射。 |
| F11 | existing_structure/none | 6/17/11 | N=0 split已有active/empty域结构；无效网格工作拒绝仍是schedule允许的空工作规则，不能将其当作所有pipe无条件存在。 |
| F12 | existing_structure/none | 5/11/11 | FinalSum是局部名字，ell最终版本仅在publish契约中绑定；可用既有数学定义+symbol消除这部分开放对象定义。 |
| F13 | structural_gap/open_semantics | 5/30/11 | 自定义QK/PV的计算/初始化绑定和原语内部读写顺序不能从done有序完全推导；主要不是缺一般barrier事件。 |
| F14 | structural_gap/open_semantics | 5/33/11 | capture和rescale包含独有数据驻留/负载决定，外围依赖已结构化。 |
| F15 | structural_gap/open_semantics | 5/33/11 | joint-load四token推进已经有e域、pipe和e+4边；KVStage的载荷种类/计算与每成员事务数量仍由开放契约建立。 |
| F16 | redundancy/none | 3/7/6 | 已有作用域、累积完成模式与release位置支持三处明确冗余压缩。 |
| F17 | structural_gap/open_semantics | 6/26/11 | stage是有意义的数学结果→OutputStage动作；group完成与coalescent访问意图仍需开放契约，但LSE produce说明可省略。 |
| F18 | structural_gap/representation_cost | 7/14/2 | combine的已定义partial与投机读取字节不同；per-row/每线程的实际访问裁剪和局部就绪仍在契约，不能误添全CTA逐stage等待。 |
| F19 | existing_structure/none | 2/22/11 | 两score槽与Y同一512列分配可用一个TMEM资源多view表达；384线程含闲warp不能由活跃role并集自动推出。 |
| F20 | style/none | 8/70/11 | Realization为有意开放的原语/实现条件绑定；重复上游保证可精简，不将API差异自动记为核心表达缺口。 |
| F21 | source_limitation/none | 17/28/11 | 样本是规范化模板；原实现/参考可定位，但不把文字审计当数值、误差或完整硬件规格验收。 |

**为什么这不是新增事件需求的证明。** 去掉PV读P先于后发QK写同槽的保证，`async_ordered_completion`仍允许后发写先发生而done较晚，形成真实别名覆盖反例；但完整原语保证允许内部重叠，添加PV.done→QK.issue会加强顺序，不能称等价结构化。该保证当前在Design与Realization两处出现，因此保守短写只删Realization重述、保留Design契约。反之，单纯更换barrier API而保持同一group.done和图依赖不是核心反例。

**表达成本而非已证缺字段。** combine已有4槽、3次prologue预取、明确data/reuse边；仍要说明每线程私有读取和按所持row-group的末个非空split裁剪。仅把Main拆成8个warp角色无法证明保持每线程独立就绪，可能凭空强化等待。F18因此标representation_cost；其余8条structural_gap标open_semantics，missing_field_or_event为0。

## 已有语法可用的改写

1. **7个完整数学区复用**：root→rect_joint_load/packed_persistent/small_static_lpt，packed_persistent→single_persistent_dual_load/small_persistent_lpt，single_persistent_dual_load→single_persistent_joint_load，splitkv_dual_load→splitkv_joint_load。对应子节点可写`math_reshape: {from_parent: <直接父ID>}`；M/Bk/N/sigma在消费者重新绑定。两个shared_rescale数学区含Stats改变，不能用父区引用覆盖。当前未改输入，也未把这种整区结构改写计入字符候选。

2. **FinalSum显式数学别名**：在online非空工作分支定义`FinalSum = ell(N-1)`，并登记`symbol.FinalSum: {meaning: Final denominator., shape: [row: M], dtype: fp32}`。保留原`publish:FinalSum(t)`/`FinalSum(active)`、data/reuse边与pipe位置。不能为N=0生成FinalSum。

3. **同一TMEM分配用同一资源表达**：下例只展开static普通节点的实际已有resource/view字段；persistent把i换成n，shared alpha把Stats上端34换33，所有原事件/依赖保持。

```yaml
Accumulator:
  space: TMEM
  member_slot_shape: [row: 128, column: 512]
  dtype: fp32
  slot_count: 1
  view:
    S(i,d): {part: {column: ["128*(i%2)", "128*(i%2)+128"]}}
    P(i): {part: {column: ["128*(i%2)", "128*(i%2)+32"]}, dtype: fp8_e4m3fn}
    Stats(i): {part: {column: ["128*(i%2)+32", "128*(i%2)+34"]}}
    Y: {part: {column: [256, 512]}}
```

这使用一个资源表达score两槽与Y的同一分配，不是把长文移到choice。384线程和idle warp让寄存器仍是实现条件；无证据量化其独立主要成本，不能按活跃role并集臆造idle角色。

4. **空split已有结构**：`active:[0,min(1,N)]`、`empty:[0,1-min(1,N)]`和repeat已保留N=0仅EmptyLSE的事件集合。strategy允许的无效work过滤仍须保持；进一步显式guard必须连同t/n、首末动作一起改，不能只删拒绝句。

## 有据短写与边界

17个精确quote候选共保守减少2,473声明字符（YAML的2.97%；W/T/R 2,407，为6.24%）；数学、硬件、dispatch本轮均未计可删量。主要候选：

| 原quote或定位 | replacement | 含义如何保留 |
|---|---|---|
| `Use two-CTA tcgen05 MMA with cumulative completion and earlier-PV-read-before-later-QK-write ordering.` | `Two-CTA tcgen05 MMA.` | 所有11节点Design的completion模式和读写保证保留，减少902声明字符 |
| `All Y reads are complete; the next tile may initialize Y while the final staged half is being stored.` | 删除可选info.verb.release条目 | 通用release、stage→release→末store及下一work首PV边保留，减少505 |
| `Compute and store LSEpart into PartialLSE for valid rows after publishing FinalSum; correction can already stage Opart.` | 删除可选info.verb.produce条目 | 通用produce、math写出位置、现有pipe及无额外Correction边保留，减少238 |
| `Final BF16 attention with a rectangular static schedule.` | `Compute final BF16 attention.` | 必填describe非空，schedule仍决定rectangular/static |

其余候选包括root成员序号/维度次序重述、root累积完成重述、同CTA来源句，以及已定义python_none/正长度的可选range。不能对MIXED字段按长度截断；尤其Split阈值、N=0未写partial、首次Stats确认、head-major combine视图、前3次预读和全读者退役均不是篇幅噪声。

已检查边界：N=1保留首次QK/PV及末输出；N=0 split无未生成动作的等待；persistent环跨work保留n，Query与Y分别在正确读者完成后复用；ScoreProbability的S/P/Stats按同字节区审阅，Stats首项不读payload；FinalSum下一tile发布等待先前capture；combine头尾预取为空时不生成不存在的load；预读未写PartialO不意味着需要全局清零，零权项不得进入乘加。

跨调用/捕获方面，选定Model.forward直接调用私有_flash_attn_fwd，公有fastpath缓存重放入口不在其调用链。未发现该路径特有初始化/回放提交；静态plan工作实例按DOC0绑定实际执行，不能把录图当已完成kernel。这里不声称审阅了所有外部capture组合。

## 限制与校验

所有56个非FORMAL含核心组都有具体机制归属；UNRESOLVED=0表示本轮分类/范围已完成，不表示源码和误差问题全部解决。完整硬件wiki绑定、fast-math独立误差界及CUDA compiler/driver版本仍未完成；未独立核验全部ISA与第三方库协议实现。来源限制应保留，不升级成新设计。

2736个scalar的全部非开放唯一文本已核对，余项均为公式/引用/类型/动作身份/retention/硬件占位，未发现漏正文。逐节点检查用当前本地完整定义和全部父子diff完成；同文不作为适用性证明。只提出候选，未修改原schema/规范/运行状态，未运行GPU。

`python3 -B schema_extract/archive/20261109-ten-operator-text-audit/scripts/summarize.py --operator template_fa4_prefill`：**VALID；200/200 groups、11/11 nodes、0 unresolved**。统计由scope_statistics回填。较大结构提案和短写尚未接受独立完整短稿等价复审，因此2,473仅为保守候选量。
